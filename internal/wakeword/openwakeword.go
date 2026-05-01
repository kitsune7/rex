package wakeword

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Constants derived from openWakeWord's audio pipeline. They are fixed by the
// upstream ONNX models — do not tune without also retraining the classifier.
const (
	// ChunkSamples is the expected input chunk size per predict call
	// (80 ms @ 16 kHz). Shorter/longer chunks are allowed but add latency.
	ChunkSamples = 1280

	// melHop is the hop size used by the melspectrogram model (10 ms @ 16 kHz).
	melHop = 160

	// melBins is the number of mel frequency bins emitted per frame.
	melBins = 32

	// embeddingWindow is the number of mel frames fed to the embedding model.
	embeddingWindow = 76

	// embeddingStep is the stride (in mel frames) between successive embeddings.
	embeddingStep = 8

	// embeddingDim is the fixed output dimensionality of the embedding model.
	embeddingDim = 96

	// melContextOverlap is the number of prior mel hops included when computing
	// the streaming melspectrogram (matches Python's `160*3`).
	melContextOverlap = 3

	// melBufferMaxFrames caps the melspectrogram ring buffer (~10 seconds).
	melBufferMaxFrames = 10 * 97

	// featureBufferMaxLen caps the embedding ring buffer (~10 seconds of
	// embeddings at embeddingStep=8).
	featureBufferMaxLen = 120

	// rawBufferMaxSamples caps the raw PCM ring buffer (10 seconds @ 16 kHz).
	rawBufferMaxSamples = 16000 * 10
)

// OpenWakeWord orchestrates the three ONNX models that make up an
// openWakeWord detector: a melspectrogram preprocessor, a speech embedding
// network, and a small wake-word classifier. Each call to Predict consumes
// a chunk of int16 PCM samples at 16 kHz and returns a score in [0, 1].
//
// The implementation is a direct port of the streaming path in
// `src/wake_word/wake_word_listener.py` (via the upstream
// openwakeword.utils.AudioFeatures streaming API).
//
// OpenWakeWord is NOT safe for concurrent Predict calls on the same
// instance. Create a separate instance per goroutine (e.g. one for the
// foreground listener, one for the background monitor).
type OpenWakeWord struct {
	mu sync.Mutex

	melSession     *ort.DynamicAdvancedSession
	embedSession   *ort.DynamicAdvancedSession
	wakeSession    *ort.DynamicAdvancedSession
	wakeInputName  string
	wakeOutputName string

	// Reusable tensors for the embedding and wake-word models (fixed shapes).
	embedIn  *ort.Tensor[float32]
	embedOut *ort.Tensor[float32]
	wakeIn   *ort.Tensor[float32]
	wakeOut  *ort.Tensor[float32]

	// Streaming state.
	rawBuffer          []int16   // rolling raw int16 samples
	melBuffer          []float32 // rolling melspec, flattened row-major (frames x melBins)
	melFrames          int       // current frame count in melBuffer
	featureBuffer      []float32 // rolling embeddings, flattened (frames x embeddingDim)
	featureFrames      int
	accumulated        int // raw samples accumulated since last melspec run
	rawRemainder       []int16
	predictionHistory  []float32 // last predictions for warm-up suppression
	predictionsEmitted int       // count of predict() calls since last reset
}

// NewOpenWakeWord loads the three ONNX models and returns a ready-to-use
// detector. ONNX runtime must have been initialised via InitONNX.
func NewOpenWakeWord(paths ModelPaths) (*OpenWakeWord, error) {
	if err := requireInitialised(); err != nil {
		return nil, err
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer opts.Destroy()
	_ = opts.SetIntraOpNumThreads(1)
	_ = opts.SetInterOpNumThreads(1)

	// Melspectrogram uses a dynamic-length input, so use a dynamic session.
	melSession, err := ort.NewDynamicAdvancedSession(paths.Melspectrogram,
		[]string{"input"}, []string{"output"}, opts)
	if err != nil {
		return nil, fmt.Errorf("opening melspec session: %w", err)
	}

	embedSession, err := ort.NewDynamicAdvancedSession(paths.Embedding,
		[]string{"input_1"}, []string{"conv2d_19"}, opts)
	if err != nil {
		melSession.Destroy()
		return nil, fmt.Errorf("opening embedding session: %w", err)
	}

	// Wake-word classifier input/output names are model-dependent; inspect.
	wakeInput, wakeOutput, err := inspectWakeWordIO(paths.WakeWord)
	if err != nil {
		embedSession.Destroy()
		melSession.Destroy()
		return nil, err
	}
	wakeSession, err := ort.NewDynamicAdvancedSession(paths.WakeWord,
		[]string{wakeInput}, []string{wakeOutput}, opts)
	if err != nil {
		embedSession.Destroy()
		melSession.Destroy()
		return nil, fmt.Errorf("opening wake-word session: %w", err)
	}

	// Fixed tensor shapes for embedding (1,76,32,1) and classifier (1,16,96).
	embedIn, err := ort.NewEmptyTensor[float32](ort.NewShape(1, embeddingWindow, melBins, 1))
	if err != nil {
		wakeSession.Destroy()
		embedSession.Destroy()
		melSession.Destroy()
		return nil, fmt.Errorf("embed input tensor: %w", err)
	}
	embedOut, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, 1, embeddingDim))
	if err != nil {
		embedIn.Destroy()
		wakeSession.Destroy()
		embedSession.Destroy()
		melSession.Destroy()
		return nil, fmt.Errorf("embed output tensor: %w", err)
	}
	wakeIn, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 16, embeddingDim))
	if err != nil {
		embedOut.Destroy()
		embedIn.Destroy()
		wakeSession.Destroy()
		embedSession.Destroy()
		melSession.Destroy()
		return nil, fmt.Errorf("wake input tensor: %w", err)
	}
	wakeOut, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		wakeIn.Destroy()
		embedOut.Destroy()
		embedIn.Destroy()
		wakeSession.Destroy()
		embedSession.Destroy()
		melSession.Destroy()
		return nil, fmt.Errorf("wake output tensor: %w", err)
	}

	ww := &OpenWakeWord{
		melSession:     melSession,
		embedSession:   embedSession,
		wakeSession:    wakeSession,
		wakeInputName:  wakeInput,
		wakeOutputName: wakeOutput,
		embedIn:        embedIn,
		embedOut:       embedOut,
		wakeIn:         wakeIn,
		wakeOut:        wakeOut,
	}
	ww.Reset()
	return ww, nil
}

// Close releases all ONNX resources. Safe to call once.
func (w *OpenWakeWord) Close() {
	if w == nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, t := range []interface{ Destroy() error }{w.embedIn, w.embedOut, w.wakeIn, w.wakeOut} {
		if t != nil {
			_ = t.Destroy()
		}
	}
	w.embedIn, w.embedOut, w.wakeIn, w.wakeOut = nil, nil, nil, nil
	for _, s := range []*ort.DynamicAdvancedSession{w.melSession, w.embedSession, w.wakeSession} {
		if s != nil {
			_ = s.Destroy()
		}
	}
	w.melSession, w.embedSession, w.wakeSession = nil, nil, nil
}

// Reset clears all streaming state so the next chunk is treated as the
// start of a new detection window. Intended for use when the listener is
// restarted.
func (w *OpenWakeWord) Reset() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.rawBuffer = w.rawBuffer[:0]
	// Match Python: initial melspec buffer is np.ones((76, 32)).
	w.melBuffer = make([]float32, embeddingWindow*melBins)
	for i := range w.melBuffer {
		w.melBuffer[i] = 1
	}
	w.melFrames = embeddingWindow
	w.featureBuffer = w.featureBuffer[:0]
	w.featureFrames = 0
	w.accumulated = 0
	w.rawRemainder = w.rawRemainder[:0]
	w.predictionHistory = w.predictionHistory[:0]
	w.predictionsEmitted = 0

	// Seed the feature buffer by running the pipeline on 4 seconds of
	// random audio, matching AudioFeatures.__init__ in the upstream Python.
	// This puts the wake-word classifier's input close to the distribution
	// the model was trained on, even before any real audio is received.
	if err := w.seedFeatureBuffer(); err != nil {
		// Seeding is best-effort; if it fails, leave the buffer empty and
		// Predict will fall through to the warm-up suppression path.
		w.featureBuffer = w.featureBuffer[:0]
		w.featureFrames = 0
	}
}

// seedFeatureBuffer approximates the Python initialisation
// `self.feature_buffer = self._get_embeddings(np.random.randint(-1000, 1000, 16000*4))`.
// It runs the mel + embedding pipeline over 4 s of random int16 audio and
// appends the resulting embeddings to the feature buffer.
func (w *OpenWakeWord) seedFeatureBuffer() error {
	const seedSamples = 16000 * 4
	noise := make([]int16, seedSamples)
	// Use a deterministic LCG to avoid importing math/rand here (and so that
	// seeding is reproducible if someone wants to compare against Python).
	var state uint32 = 0x12345678
	for i := range noise {
		state = state*1103515245 + 12345
		noise[i] = int16((int32(state) % 2001) - 1000)
	}

	// Run through the streaming pipeline. streamingFeatures will populate
	// the feature buffer as melspectrograms and embeddings accumulate.
	if _, err := w.streamingFeatures(noise); err != nil {
		return err
	}
	return nil
}

// Predict runs one detection pass over a chunk of int16 samples at 16 kHz
// and returns a score in [0, 1]. Ideally chunks are multiples of 1280
// samples (80 ms); shorter chunks accumulate until enough samples are
// available. For the first few calls the returned score is forced to 0 to
// match the Python warm-up behaviour.
func (w *OpenWakeWord) Predict(chunk []int16) (float32, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	nPrepared, err := w.streamingFeatures(chunk)
	if err != nil {
		return 0, err
	}

	var score float32
	switch {
	case nPrepared >= ChunkSamples:
		// Fold accumulated history: for the last N chunks emitted, run the
		// classifier and take the maximum score (mirrors the upstream
		// "group_predictions" path).
		nChunks := nPrepared / ChunkSamples
		for i := nChunks - 1; i >= 0; i-- {
			start := w.featureFrames - 16 - i
			if start < 0 {
				continue
			}
			s, err := w.runClassifier(start)
			if err != nil {
				return 0, err
			}
			if s > score {
				score = s
			}
		}
	default:
		// Not enough samples accumulated yet; reuse the last score, or 0.
		if n := len(w.predictionHistory); n > 0 {
			score = w.predictionHistory[n-1]
		}
	}

	// Warm-up suppression: force the first 5 frames to zero, matching Python.
	w.predictionsEmitted++
	if w.predictionsEmitted <= 5 {
		score = 0
	}

	w.predictionHistory = append(w.predictionHistory, score)
	if len(w.predictionHistory) > 30 {
		w.predictionHistory = w.predictionHistory[len(w.predictionHistory)-30:]
	}
	return score, nil
}

// streamingFeatures is the Go port of AudioFeatures._streaming_features. It
// buffers raw samples, runs melspectrogram / embedding when a full
// ChunkSamples-worth has accumulated, and returns the number of samples
// processed on this call (>=1280 if melspec ran, else the accumulated count
// awaiting the next call).
func (w *OpenWakeWord) streamingFeatures(chunk []int16) (int, error) {
	// Fold pending remainder into this chunk.
	if len(w.rawRemainder) > 0 {
		merged := make([]int16, 0, len(w.rawRemainder)+len(chunk))
		merged = append(merged, w.rawRemainder...)
		merged = append(merged, chunk...)
		chunk = merged
		w.rawRemainder = w.rawRemainder[:0]
	}

	if w.accumulated+len(chunk) >= ChunkSamples {
		total := w.accumulated + len(chunk)
		remainder := total % ChunkSamples
		if remainder != 0 {
			evenEnd := len(chunk) - remainder
			w.bufferRaw(chunk[:evenEnd])
			w.accumulated += evenEnd
			w.rawRemainder = append(w.rawRemainder, chunk[evenEnd:]...)
		} else {
			w.bufferRaw(chunk)
			w.accumulated += len(chunk)
		}
	} else {
		w.accumulated += len(chunk)
		w.bufferRaw(chunk)
	}

	processed := 0
	if w.accumulated >= ChunkSamples && w.accumulated%ChunkSamples == 0 {
		if err := w.streamingMelspec(w.accumulated); err != nil {
			return 0, err
		}
		if err := w.updateEmbeddings(w.accumulated); err != nil {
			return 0, err
		}
		processed = w.accumulated
		w.accumulated = 0
	}

	if processed != 0 {
		return processed, nil
	}
	return w.accumulated, nil
}

// bufferRaw appends samples to the raw ring buffer, enforcing the 10s cap.
func (w *OpenWakeWord) bufferRaw(chunk []int16) {
	w.rawBuffer = append(w.rawBuffer, chunk...)
	if len(w.rawBuffer) > rawBufferMaxSamples {
		w.rawBuffer = w.rawBuffer[len(w.rawBuffer)-rawBufferMaxSamples:]
	}
}

// streamingMelspec appends new melspectrogram frames covering the latest
// n_samples + 3 hops of raw audio (Python: `-n_samples - 160*3`).
func (w *OpenWakeWord) streamingMelspec(nSamples int) error {
	start := max(0, len(w.rawBuffer)-nSamples-melHop*melContextOverlap)
	slice := w.rawBuffer[start:]

	// Melspec model takes a float32 tensor of shape (1, samples).
	floatSamples := make([]float32, len(slice))
	for i, s := range slice {
		floatSamples[i] = float32(s)
	}

	inputTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(floatSamples))), floatSamples)
	if err != nil {
		return fmt.Errorf("melspec input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	var outVal ort.Value
	outputs := []ort.Value{nil} // auto-allocate
	if err := w.melSession.Run([]ort.Value{inputTensor}, outputs); err != nil {
		return fmt.Errorf("running melspec: %w", err)
	}
	outVal = outputs[0]
	defer func() {
		if outVal != nil {
			_ = outVal.Destroy()
		}
	}()

	outTensor, ok := outVal.(*ort.Tensor[float32])
	if !ok {
		return fmt.Errorf("melspec produced unexpected output type %T", outVal)
	}

	// Shape: (1, 1, frames, melBins). Flatten row-major.
	shape := outTensor.GetShape()
	if len(shape) != 4 || shape[3] != melBins {
		return fmt.Errorf("unexpected melspec shape: %v", shape)
	}
	frames := int(shape[2])
	data := outTensor.GetData()

	// Apply the Python transform: spec = spec/10 + 2.
	for i, v := range data {
		data[i] = v/10 + 2
	}

	// Append to the ring buffer.
	w.melBuffer = append(w.melBuffer, data...)
	w.melFrames += frames
	if w.melFrames > melBufferMaxFrames {
		extra := w.melFrames - melBufferMaxFrames
		w.melBuffer = w.melBuffer[extra*melBins:]
		w.melFrames = melBufferMaxFrames
	}
	return nil
}

// updateEmbeddings runs the embedding model for each newly-accumulated
// 1280-sample chunk (one embedding per chunk) and appends the result to the
// feature buffer.
func (w *OpenWakeWord) updateEmbeddings(accumulated int) error {
	nChunks := accumulated / ChunkSamples
	for i := nChunks - 1; i >= 0; i-- {
		var ndx int
		if i == 0 {
			ndx = w.melFrames
		} else {
			ndx = w.melFrames - embeddingStep*i
		}
		start := ndx - embeddingWindow
		if start < 0 || ndx-start != embeddingWindow {
			continue
		}

		window := w.melBuffer[start*melBins : ndx*melBins]
		copy(w.embedIn.GetData(), window)

		if err := w.embedSession.Run(
			[]ort.Value{w.embedIn},
			[]ort.Value{w.embedOut},
		); err != nil {
			return fmt.Errorf("running embedding: %w", err)
		}
		emb := w.embedOut.GetData() // (1,1,1,96) → 96 floats
		w.featureBuffer = append(w.featureBuffer, emb...)
		w.featureFrames++
		if w.featureFrames > featureBufferMaxLen {
			extra := w.featureFrames - featureBufferMaxLen
			w.featureBuffer = w.featureBuffer[extra*embeddingDim:]
			w.featureFrames = featureBufferMaxLen
		}
	}
	return nil
}

// runClassifier feeds the last 16 embedding frames (starting at index start)
// to the wake-word model and returns the score.
func (w *OpenWakeWord) runClassifier(start int) (float32, error) {
	if start < 0 || start+16 > w.featureFrames {
		return 0, fmt.Errorf("wakeword: feature window out of range (start=%d, frames=%d)", start, w.featureFrames)
	}
	window := w.featureBuffer[start*embeddingDim : (start+16)*embeddingDim]
	copy(w.wakeIn.GetData(), window)
	if err := w.wakeSession.Run(
		[]ort.Value{w.wakeIn},
		[]ort.Value{w.wakeOut},
	); err != nil {
		return 0, fmt.Errorf("running wake-word classifier: %w", err)
	}
	return w.wakeOut.GetData()[0], nil
}

// inspectWakeWordIO peeks at the wake-word ONNX model to discover its input
// and output tensor names (they vary across training runs).
func inspectWakeWordIO(path string) (inputName, outputName string, err error) {
	inputs, outputs, err := ort.GetInputOutputInfo(path)
	if err != nil {
		return "", "", fmt.Errorf("inspecting wake-word model: %w", err)
	}
	if len(inputs) != 1 || len(outputs) != 1 {
		return "", "", fmt.Errorf("wake-word model must have exactly one input and output, got %d in / %d out",
			len(inputs), len(outputs))
	}
	return inputs[0].Name, outputs[0].Name, nil
}
