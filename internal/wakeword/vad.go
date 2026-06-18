package wakeword

import (
	"fmt"
	"math"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	// vadChunkSize is the number of samples per VAD inference (512 at 16 kHz = 32 ms).
	vadChunkSize = 512

	// vadSampleRate is the expected sample rate for the Silero VAD model.
	vadSampleRate = 16000

	// Silero VAD LSTM hidden state size.
	vadHiddenSize = 64
)

// VADProcessor wraps the Silero VAD ONNX model for speech/silence detection.
//
// The Silero model processes 512-sample chunks at 16 kHz and returns a speech
// probability between 0.0 and 1.0. It maintains internal LSTM state (h and c
// tensors) across calls so it can track speech context.
type VADProcessor struct {
	session *ort.AdvancedSession

	// Input tensors (reused across calls).
	inputTensor *ort.Tensor[float32]
	srTensor    *ort.Tensor[int64]
	hTensor     *ort.Tensor[float32]
	cTensor     *ort.Tensor[float32]

	// Output tensors.
	outputTensor *ort.Tensor[float32]
	hnTensor     *ort.Tensor[float32]
	cnTensor     *ort.Tensor[float32]

	// Accumulator for audio that hasn't been processed yet.
	buf []float32
}

// NewVADProcessor loads the Silero VAD ONNX model from the given path.
// Call ort.InitializeEnvironment() before creating a VADProcessor.
func NewVADProcessor(modelPath string) (*VADProcessor, error) {
	// Silero VAD v5 input/output names and shapes.
	inputTensor := ort.NewTensorWithShape[float32]([]int64{1, vadChunkSize})
	srTensor := ort.NewTensorWithShape[int64]([]int64{1})
	hTensor := ort.NewTensorWithShape[float32]([]int64{2, 1, vadHiddenSize})
	cTensor := ort.NewTensorWithShape[float32]([]int64{2, 1, vadHiddenSize})

	outputTensor := ort.NewTensorWithShape[float32]([]int64{1, 1})
	hnTensor := ort.NewTensorWithShape[float32]([]int64{2, 1, vadHiddenSize})
	cnTensor := ort.NewTensorWithShape[float32]([]int64{2, 1, vadHiddenSize})

	// Set sample rate input.
	srTensor.GetData()[0] = vadSampleRate

	sess, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input", "sr", "h", "c"},
		[]string{"output", "hn", "cn"},
		[]ort.ArbitraryTensor{inputTensor, srTensor, hTensor, cTensor},
		[]ort.ArbitraryTensor{outputTensor, hnTensor, cnTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("load silero VAD model: %w", err)
	}

	return &VADProcessor{
		session:      sess,
		inputTensor:  inputTensor,
		srTensor:     srTensor,
		hTensor:      hTensor,
		cTensor:      cTensor,
		outputTensor: outputTensor,
		hnTensor:     hnTensor,
		cnTensor:     cnTensor,
		buf:          make([]float32, 0, vadChunkSize*4),
	}, nil
}

// Process feeds an audio chunk (int16, 16 kHz, mono) and returns speech
// probabilities for each complete 512-sample window that could be formed.
// Audio that doesn't fill a complete window is buffered for the next call.
func (v *VADProcessor) Process(chunk []int16) ([]float32, error) {
	// Convert to float32 normalised to [-1, 1].
	for _, s := range chunk {
		v.buf = append(v.buf, float32(s)/math.MaxInt16)
	}

	var probs []float32
	for len(v.buf) >= vadChunkSize {
		window := v.buf[:vadChunkSize]
		v.buf = v.buf[vadChunkSize:]

		prob, err := v.infer(window)
		if err != nil {
			return probs, err
		}
		probs = append(probs, prob)
	}
	return probs, nil
}

// Reset clears the audio buffer and resets the LSTM hidden state.
func (v *VADProcessor) Reset() {
	v.buf = v.buf[:0]
	// Zero out h and c tensors.
	for i := range v.hTensor.GetData() {
		v.hTensor.GetData()[i] = 0
	}
	for i := range v.cTensor.GetData() {
		v.cTensor.GetData()[i] = 0
	}
}

// Destroy releases ONNX session resources.
func (v *VADProcessor) Destroy() {
	if v.session != nil {
		v.session.Destroy()
	}
}

// infer runs a single VAD inference on a 512-sample window.
func (v *VADProcessor) infer(window []float32) (float32, error) {
	copy(v.inputTensor.GetData(), window)

	if err := v.session.Run(); err != nil {
		return 0, fmt.Errorf("VAD inference: %w", err)
	}

	prob := v.outputTensor.GetData()[0]

	// Copy output hidden states back to input for next call.
	copy(v.hTensor.GetData(), v.hnTensor.GetData())
	copy(v.cTensor.GetData(), v.cnTensor.GetData())

	return prob, nil
}
