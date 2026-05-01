package wakeword

import (
	"math/rand"
	"runtime"
	"testing"
)

// TestOpenWakeWord_SmokeOnSilence verifies that the pipeline runs without
// error and produces very low scores on 2 seconds of silence.
func TestOpenWakeWord_SmokeOnSilence(t *testing.T) {
	ww := openWakeWordForTest(t)
	defer ww.Close()

	// Feed 25 × 1280-sample chunks (~2 s of silence).
	silence := make([]int16, ChunkSamples)
	var maxScore float32
	for i := 0; i < 25; i++ {
		score, err := ww.Predict(silence)
		if err != nil {
			t.Fatalf("Predict(%d): %v", i, err)
		}
		if score > maxScore {
			maxScore = score
		}
	}
	if maxScore > 0.05 {
		t.Errorf("silence produced score %.4f; expected well under 0.05", maxScore)
	}
}

// TestOpenWakeWord_HandlesShortChunks verifies the pipeline accepts chunks
// smaller than 1280 and accumulates internally.
func TestOpenWakeWord_HandlesShortChunks(t *testing.T) {
	ww := openWakeWordForTest(t)
	defer ww.Close()

	// 10 × 128-sample chunks totalling 1280 samples.
	small := make([]int16, 128)
	for i := 0; i < 10; i++ {
		if _, err := ww.Predict(small); err != nil {
			t.Fatalf("Predict: %v", err)
		}
	}
}

// TestOpenWakeWord_ResetClearsState verifies that Reset puts the detector
// back into the warm-up state where the first 5 scores are forced to zero.
func TestOpenWakeWord_ResetClearsState(t *testing.T) {
	ww := openWakeWordForTest(t)
	defer ww.Close()

	// Prime past the warm-up window.
	silence := make([]int16, ChunkSamples)
	for i := 0; i < 10; i++ {
		if _, err := ww.Predict(silence); err != nil {
			t.Fatalf("Predict: %v", err)
		}
	}

	ww.Reset()

	// Next 5 calls should be exactly 0.
	for i := 0; i < 5; i++ {
		score, err := ww.Predict(silence)
		if err != nil {
			t.Fatalf("Predict after reset: %v", err)
		}
		if score != 0 {
			t.Errorf("post-reset frame %d score = %v, want 0 (warm-up)", i, score)
		}
	}
}

// TestOpenWakeWord_NoisyAudioProducesLowScore — random noise shouldn't
// trigger a wake-word detection.
func TestOpenWakeWord_NoisyAudioProducesLowScore(t *testing.T) {
	ww := openWakeWordForTest(t)
	defer ww.Close()

	r := rand.New(rand.NewSource(42))
	noise := make([]int16, ChunkSamples)

	var maxScore float32
	for i := 0; i < 30; i++ {
		for j := range noise {
			noise[j] = int16(r.Intn(2000) - 1000)
		}
		score, err := ww.Predict(noise)
		if err != nil {
			t.Fatalf("Predict: %v", err)
		}
		if score > maxScore {
			maxScore = score
		}
	}
	if maxScore > 0.1 {
		t.Errorf("random noise produced score %.4f; expected under 0.1", maxScore)
	}
}

func openWakeWordForTest(t *testing.T) *OpenWakeWord {
	t.Helper()
	if runtime.GOOS != "darwin" {
		t.Skip("openWakeWord smoke tests assume Homebrew onnxruntime on darwin")
	}
	paths := DefaultModelPaths(repoRoot(t), "hey_rex")
	if err := paths.Verify(); err != nil {
		t.Skipf("models not present: %v", err)
	}
	if err := InitONNX(""); err != nil {
		t.Skipf("onnx init failed: %v", err)
	}
	ww, err := NewOpenWakeWord(paths)
	if err != nil {
		t.Fatalf("NewOpenWakeWord: %v", err)
	}
	return ww
}
