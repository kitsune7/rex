package wakeword

import (
	"fmt"
	"os"
	"path/filepath"
)

// Required ONNX model files for the wake word pipeline.
var requiredWakeWordModels = []string{
	"melspectrogram.onnx",
	"embedding_model.onnx",
	"hey_rex.onnx",
}

// RequiredVADModel is the filename of the Silero VAD model.
const RequiredVADModel = "silero_vad.onnx"

// EnsureModels verifies that all required ONNX model files exist under
// modelDir. It checks for the three wake word pipeline models and the
// Silero VAD model. Returns an error listing any missing files.
func EnsureModels(modelDir string) error {
	var missing []string

	for _, name := range requiredWakeWordModels {
		p := filepath.Join(modelDir, name)
		if _, err := os.Stat(p); os.IsNotExist(err) {
			missing = append(missing, p)
		}
	}

	vadPath := filepath.Join(modelDir, RequiredVADModel)
	if _, err := os.Stat(vadPath); os.IsNotExist(err) {
		missing = append(missing, vadPath)
	}

	if len(missing) > 0 {
		return fmt.Errorf("missing required ONNX models: %v", missing)
	}
	return nil
}
