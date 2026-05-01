package wakeword

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// vadFrameSize is the input chunk size the Silero model expects at 16 kHz.
const vadFrameSize = 512

// VADFramePredictor runs one 512-sample frame through Silero VAD and returns
// the speech probability. Production code uses *VADModel; tests can provide
// a fake implementation to avoid touching ONNX.
type VADFramePredictor interface {
	// PredictFrame processes one float32 frame and returns a probability in
	// [0, 1]. Input length must equal VADFrameSize().
	PredictFrame(frame []float32) (float32, error)

	// VADFrameSize returns the expected input length.
	VADFrameSize() int

	// ResetState clears the recurrent hidden state.
	ResetState()
}

// VADModel wraps the Silero VAD ONNX model and carries the recurrent state
// across frames. It is safe for concurrent use; callers are serialised by an
// internal mutex.
//
// The model accepts 512-sample frames at 16 kHz. Inputs are float32 samples
// in the range expected by the Python reference ([-32768, 32767], i.e. int16
// values cast to float32 without normalisation — matching the Python
// VADProcessor which does `chunk.astype(np.float32)`).
type VADModel struct {
	mu      sync.Mutex
	session *ort.DynamicAdvancedSession
	in      *ort.Tensor[float32]
	state   *ort.Tensor[float32]
	sr      *ort.Scalar[int64]
	out     *ort.Tensor[float32]
	newSt   *ort.Tensor[float32]
}

// NewVADModel loads the Silero VAD ONNX model at path. ONNX runtime must
// have been initialised via InitONNX.
func NewVADModel(path string) (*VADModel, error) {
	if err := requireInitialised(); err != nil {
		return nil, err
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("creating session options: %w", err)
	}
	defer opts.Destroy()
	_ = opts.SetIntraOpNumThreads(1)
	_ = opts.SetInterOpNumThreads(1)

	session, err := ort.NewDynamicAdvancedSession(path,
		[]string{"input", "state", "sr"},
		[]string{"output", "stateN"},
		opts)
	if err != nil {
		return nil, fmt.Errorf("opening silero vad session: %w", err)
	}

	in, err := ort.NewEmptyTensor[float32](ort.NewShape(1, vadFrameSize))
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("creating input tensor: %w", err)
	}
	state, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 128))
	if err != nil {
		in.Destroy()
		session.Destroy()
		return nil, fmt.Errorf("creating state tensor: %w", err)
	}
	sr, err := ort.NewScalar[int64](16000)
	if err != nil {
		state.Destroy()
		in.Destroy()
		session.Destroy()
		return nil, fmt.Errorf("creating sr tensor: %w", err)
	}
	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		sr.Destroy()
		state.Destroy()
		in.Destroy()
		session.Destroy()
		return nil, fmt.Errorf("creating output tensor: %w", err)
	}
	newSt, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 128))
	if err != nil {
		out.Destroy()
		sr.Destroy()
		state.Destroy()
		in.Destroy()
		session.Destroy()
		return nil, fmt.Errorf("creating new state tensor: %w", err)
	}

	return &VADModel{
		session: session,
		in:      in,
		state:   state,
		sr:      sr,
		out:     out,
		newSt:   newSt,
	}, nil
}

// Close releases ONNX resources. Safe to call once.
func (m *VADModel) Close() {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, t := range []interface{ Destroy() error }{m.in, m.state, m.sr, m.out, m.newSt} {
		if t != nil {
			_ = t.Destroy()
		}
	}
	m.in, m.state, m.sr, m.out, m.newSt = nil, nil, nil, nil, nil
	if m.session != nil {
		_ = m.session.Destroy()
		m.session = nil
	}
}

// ResetState zeroes the recurrent state so the next frame is treated as the
// start of a new utterance.
func (m *VADModel) ResetState() {
	m.mu.Lock()
	defer m.mu.Unlock()
	data := m.state.GetData()
	for i := range data {
		data[i] = 0
	}
}

// VADFrameSize returns the input frame size Silero expects (512 samples).
func (m *VADModel) VADFrameSize() int { return vadFrameSize }

// PredictFrame processes one 512-sample frame and returns speech probability.
func (m *VADModel) PredictFrame(frame []float32) (float32, error) {
	if len(frame) != vadFrameSize {
		return 0, fmt.Errorf("wakeword: vad frame must be %d samples, got %d", vadFrameSize, len(frame))
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.session == nil {
		return 0, fmt.Errorf("wakeword: vad model is closed")
	}

	copy(m.in.GetData(), frame)
	if err := m.session.Run(
		[]ort.Value{m.in, m.state, m.sr},
		[]ort.Value{m.out, m.newSt},
	); err != nil {
		return 0, fmt.Errorf("running silero vad: %w", err)
	}
	copy(m.state.GetData(), m.newSt.GetData())
	return m.out.GetData()[0], nil
}

// VADProcessor buffers audio chunks and runs the Silero VAD on every
// completed frame. It is the Go port of the Python VADProcessor in
// src/wake_word/wake_word_listener.py.
//
// A VADProcessor is not safe for concurrent use from multiple goroutines.
// The underlying VADFramePredictor may be shared across processors if it is
// itself safe for concurrent use (VADModel is).
type VADProcessor struct {
	model     VADFramePredictor
	chunkSize int
	buffer    []float32 // pending samples awaiting a full frame
}

// NewVADProcessor creates a VADProcessor bound to the given predictor. The
// predictor's recurrent state is reset on construction.
func NewVADProcessor(model VADFramePredictor) *VADProcessor {
	model.ResetState()
	return &VADProcessor{
		model:     model,
		chunkSize: model.VADFrameSize(),
	}
}

// AddAudio appends samples to the internal buffer.
func (p *VADProcessor) AddAudio(chunk []float32) {
	p.buffer = append(p.buffer, chunk...)
}

// AddInt16 appends int16 samples converted to float32 without normalisation.
// This matches the Python reference which does chunk.astype(np.float32).
func (p *VADProcessor) AddInt16(chunk []int16) {
	start := len(p.buffer)
	p.buffer = append(p.buffer, make([]float32, len(chunk))...)
	for i, s := range chunk {
		p.buffer[start+i] = float32(s)
	}
}

// Process drains buffered audio into frames and returns speech probabilities
// for each completed frame. Any remainder stays in the buffer. Returns nil
// if not enough samples are buffered for a single frame.
func (p *VADProcessor) Process() ([]float32, error) {
	if len(p.buffer) < p.chunkSize {
		return nil, nil
	}

	var out []float32
	for len(p.buffer) >= p.chunkSize {
		prob, err := p.model.PredictFrame(p.buffer[:p.chunkSize])
		if err != nil {
			return nil, err
		}
		out = append(out, prob)
		p.buffer = p.buffer[p.chunkSize:]
	}
	return out, nil
}

// Reset clears the pending sample buffer. The underlying model's recurrent
// state is left untouched; use ResetModelState to clear both.
func (p *VADProcessor) Reset() {
	p.buffer = p.buffer[:0]
}

// ResetModelState clears both the pending buffer and the model's recurrent
// state.
func (p *VADProcessor) ResetModelState() {
	p.buffer = p.buffer[:0]
	p.model.ResetState()
}

// BufferedSamples returns the current number of samples awaiting processing.
func (p *VADProcessor) BufferedSamples() int { return len(p.buffer) }
