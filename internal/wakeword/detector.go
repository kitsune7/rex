// Package wakeword provides wake word detection and voice activity detection
// using ONNX models. It ports the Python openwakeword pipeline to Go.
package wakeword

import (
	"fmt"
	"log/slog"
	"math"

	ort "github.com/yalue/onnxruntime_go"
)

// Detector runs the 3-model ONNX pipeline for wake word detection.
//
// The pipeline processes audio in sequence:
//  1. melspectrogram.onnx — converts raw 16kHz int16 audio to mel spectrogram features
//  2. embedding_model.onnx — converts mel features to 96-dim embeddings
//  3. hey_rex.onnx — classifies a window of embeddings as wake word or not
//
// Each call to Predict feeds one chunk of audio through the pipeline and returns
// a wake word confidence score between 0.0 and 1.0.
type Detector struct {
	melSession       *ort.AdvancedSession
	embeddingSession *ort.AdvancedSession
	wakeWordSession  *ort.AdvancedSession

	// Threshold for wake word detection.
	Threshold float32

	// Audio sample accumulator for mel spectrogram (needs 1280 samples per frame).
	audioBuf []float32

	// Accumulated mel spectrogram features. The mel model outputs 1x1x32 per frame;
	// the embedding model expects 1x76x32 (76 consecutive frames).
	melBuf [][]float32

	// Accumulated embeddings. The wake word model expects 1x16x96
	// (16 consecutive 96-dim embedding vectors).
	embeddingBuf [][]float32
}

const (
	// melFrameSamples is the number of audio samples the mel model expects per invocation.
	melFrameSamples = 1280

	// melFeatures is the number of mel bands output per frame.
	melFeatures = 32

	// embeddingFrames is how many mel frames the embedding model needs.
	embeddingFrames = 76

	// embeddingDim is the dimensionality of each embedding vector.
	embeddingDim = 96

	// wakeWordFrames is how many embedding vectors the classifier needs.
	wakeWordFrames = 16
)

// NewDetector creates a Detector that loads the three ONNX models from the
// given file paths. Call ort.InitializeEnvironment() before creating a Detector.
func NewDetector(melModelPath, embeddingModelPath, wakeWordModelPath string, threshold float32) (*Detector, error) {
	if threshold <= 0 {
		threshold = 0.5
	}

	// --- mel spectrogram session ---
	melIn := ort.NewTensorWithShape[float32]([]int64{1, melFrameSamples})
	melOut := ort.NewTensorWithShape[float32]([]int64{1, 1, melFeatures})

	melSess, err := ort.NewAdvancedSession(
		melModelPath,
		[]string{"input"},
		[]string{"output"},
		[]ort.ArbitraryTensor{melIn},
		[]ort.ArbitraryTensor{melOut},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("load mel model: %w", err)
	}

	// --- embedding session ---
	embIn := ort.NewTensorWithShape[float32]([]int64{1, embeddingFrames, melFeatures})
	embOut := ort.NewTensorWithShape[float32]([]int64{1, 1, embeddingDim})

	embSess, err := ort.NewAdvancedSession(
		embeddingModelPath,
		[]string{"input"},
		[]string{"output"},
		[]ort.ArbitraryTensor{embIn},
		[]ort.ArbitraryTensor{embOut},
		nil,
	)
	if err != nil {
		melSess.Destroy()
		return nil, fmt.Errorf("load embedding model: %w", err)
	}

	// --- wake word classifier session ---
	wwIn := ort.NewTensorWithShape[float32]([]int64{1, wakeWordFrames, embeddingDim})
	wwOut := ort.NewTensorWithShape[float32]([]int64{1, 1})

	wwSess, err := ort.NewAdvancedSession(
		wakeWordModelPath,
		[]string{"input"},
		[]string{"output"},
		[]ort.ArbitraryTensor{wwIn},
		[]ort.ArbitraryTensor{wwOut},
		nil,
	)
	if err != nil {
		melSess.Destroy()
		embSess.Destroy()
		return nil, fmt.Errorf("load wake word model: %w", err)
	}

	return &Detector{
		melSession:       melSess,
		embeddingSession: embSess,
		wakeWordSession:  wwSess,
		Threshold:        threshold,
		audioBuf:         make([]float32, 0, melFrameSamples*2),
		melBuf:           make([][]float32, 0, embeddingFrames),
		embeddingBuf:     make([][]float32, 0, wakeWordFrames),
	}, nil
}

// Predict processes an audio chunk (int16, 16 kHz, mono) through the full
// pipeline and returns the wake word confidence score (0.0 to 1.0).
//
// Audio is accumulated internally; you can call Predict with any chunk size.
// A score is only produced once enough data has passed through all three models.
// If no score was produced yet, the returned value is 0.
func (d *Detector) Predict(audioChunk []int16) (float32, error) {
	// Convert int16 to float32 normalised to [-1, 1].
	for _, s := range audioChunk {
		d.audioBuf = append(d.audioBuf, float32(s)/math.MaxInt16)
	}

	var lastScore float32

	// Process as many mel frames as possible.
	for len(d.audioBuf) >= melFrameSamples {
		frame := d.audioBuf[:melFrameSamples]
		d.audioBuf = d.audioBuf[melFrameSamples:]

		melFeats, err := d.runMel(frame)
		if err != nil {
			return 0, fmt.Errorf("mel spectrogram: %w", err)
		}
		d.melBuf = append(d.melBuf, melFeats)

		// Trim to keep only the most recent embeddingFrames mel frames.
		if len(d.melBuf) > embeddingFrames {
			d.melBuf = d.melBuf[len(d.melBuf)-embeddingFrames:]
		}

		// Run embedding model when we have enough mel frames.
		if len(d.melBuf) == embeddingFrames {
			emb, err := d.runEmbedding()
			if err != nil {
				return 0, fmt.Errorf("embedding: %w", err)
			}
			d.embeddingBuf = append(d.embeddingBuf, emb)

			if len(d.embeddingBuf) > wakeWordFrames {
				d.embeddingBuf = d.embeddingBuf[len(d.embeddingBuf)-wakeWordFrames:]
			}

			// Run wake word classifier when we have enough embeddings.
			if len(d.embeddingBuf) == wakeWordFrames {
				score, err := d.runWakeWord()
				if err != nil {
					return 0, fmt.Errorf("wake word: %w", err)
				}
				lastScore = score
			}
		}
	}

	return lastScore, nil
}

// Reset clears all internal buffers so the detector starts fresh.
func (d *Detector) Reset() {
	d.audioBuf = d.audioBuf[:0]
	d.melBuf = d.melBuf[:0]
	d.embeddingBuf = d.embeddingBuf[:0]
}

// Destroy releases ONNX session resources.
func (d *Detector) Destroy() {
	if d.melSession != nil {
		d.melSession.Destroy()
	}
	if d.embeddingSession != nil {
		d.embeddingSession.Destroy()
	}
	if d.wakeWordSession != nil {
		d.wakeWordSession.Destroy()
	}
}

// runMel executes the mel spectrogram model on a single frame of audio.
func (d *Detector) runMel(frame []float32) ([]float32, error) {
	inTensor := d.melSession.Inputs()[0].(*ort.Tensor[float32])
	copy(inTensor.GetData(), frame)

	if err := d.melSession.Run(); err != nil {
		return nil, err
	}

	outTensor := d.melSession.Outputs()[0].(*ort.Tensor[float32])
	data := outTensor.GetData()
	result := make([]float32, melFeatures)
	copy(result, data)
	return result, nil
}

// runEmbedding executes the embedding model on the accumulated mel features.
func (d *Detector) runEmbedding() ([]float32, error) {
	inTensor := d.embeddingSession.Inputs()[0].(*ort.Tensor[float32])
	buf := inTensor.GetData()
	idx := 0
	for _, frame := range d.melBuf {
		copy(buf[idx:idx+melFeatures], frame)
		idx += melFeatures
	}

	if err := d.embeddingSession.Run(); err != nil {
		return nil, err
	}

	outTensor := d.embeddingSession.Outputs()[0].(*ort.Tensor[float32])
	data := outTensor.GetData()
	result := make([]float32, embeddingDim)
	copy(result, data)
	return result, nil
}

// runWakeWord executes the wake word classifier on accumulated embeddings.
func (d *Detector) runWakeWord() (float32, error) {
	inTensor := d.wakeWordSession.Inputs()[0].(*ort.Tensor[float32])
	buf := inTensor.GetData()
	idx := 0
	for _, emb := range d.embeddingBuf {
		copy(buf[idx:idx+embeddingDim], emb)
		idx += embeddingDim
	}

	if err := d.wakeWordSession.Run(); err != nil {
		return 0, err
	}

	outTensor := d.wakeWordSession.Outputs()[0].(*ort.Tensor[float32])
	score := outTensor.GetData()[0]

	// Apply sigmoid — openwakeword models output logits.
	score = sigmoid(score)

	slog.Debug("wake word score", "score", score)
	return score, nil
}

func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}
