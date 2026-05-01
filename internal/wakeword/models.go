package wakeword

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// ModelPaths describes where the three ONNX models that make up the
// wake-word pipeline live on disk.
type ModelPaths struct {
	// Melspectrogram is the openWakeWord melspectrogram preprocessor.
	Melspectrogram string
	// Embedding is Google's speech_embedding ONNX model, as repackaged by
	// openWakeWord.
	Embedding string
	// WakeWord is the wake-word classifier (e.g. hey_rex.onnx).
	WakeWord string
	// SileroVAD is the Silero voice-activity-detection ONNX model.
	SileroVAD string
}

// DefaultModelPaths resolves ModelPaths rooted at a project directory.
// Shared models (melspec, embedding, silero_vad) live under
// <root>/models/shared; wake-word classifiers live under
// <root>/models/wake_word_models/<label>/<label>.onnx.
//
// label matches the settings.toml wake_word.path_label value (e.g. "hey_rex").
func DefaultModelPaths(root, label string) ModelPaths {
	shared := filepath.Join(root, "models", "shared")
	return ModelPaths{
		Melspectrogram: filepath.Join(shared, "melspectrogram.onnx"),
		Embedding:      filepath.Join(shared, "embedding_model.onnx"),
		SileroVAD:      filepath.Join(shared, "silero_vad.onnx"),
		WakeWord:       filepath.Join(root, "models", "wake_word_models", label, label+".onnx"),
	}
}

// Verify checks that every model file referenced by ModelPaths exists. It
// returns an aggregated error describing all missing files so the user can
// fix them in one pass.
func (m ModelPaths) Verify() error {
	var missing []string
	for label, p := range map[string]string{
		"melspectrogram": m.Melspectrogram,
		"embedding":      m.Embedding,
		"wake_word":      m.WakeWord,
		"silero_vad":     m.SileroVAD,
	} {
		if _, err := os.Stat(p); err != nil {
			missing = append(missing, fmt.Sprintf("%s (%s)", label, p))
		}
	}
	if len(missing) == 0 {
		return nil
	}
	return fmt.Errorf("wakeword: missing model files: %v — see README for download instructions", missing)
}

// ErrModelsMissing is returned when required model files are absent. Callers
// can use errors.Is to test for it.
var ErrModelsMissing = errors.New("wakeword: required model files are missing")
