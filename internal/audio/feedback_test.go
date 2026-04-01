package audio

import (
	"math"
	"testing"
)

func TestGenerateListeningTone(t *testing.T) {
	samples := GenerateListeningTone(OutputSampleRate)
	if len(samples) == 0 {
		t.Fatal("GenerateListeningTone returned empty samples")
	}

	// Expected length: two notes of 100ms each + 50ms gap.
	expectedLen := int(float64(OutputSampleRate) * (noteDuration*2 + gapDuration))
	if len(samples) != expectedLen {
		t.Errorf("expected %d samples, got %d", expectedLen, len(samples))
	}

	// All samples should be in [-1, 1].
	for i, s := range samples {
		if s < -1.0 || s > 1.0 {
			t.Fatalf("sample %d out of range: %f", i, s)
		}
	}
}

func TestGenerateDoneTone(t *testing.T) {
	samples := GenerateDoneTone(OutputSampleRate)
	if len(samples) == 0 {
		t.Fatal("GenerateDoneTone returned empty samples")
	}

	expectedLen := int(float64(OutputSampleRate) * (noteDuration*2 + gapDuration))
	if len(samples) != expectedLen {
		t.Errorf("expected %d samples, got %d", expectedLen, len(samples))
	}
}

func TestGenerateThinkingSequence(t *testing.T) {
	samples := GenerateThinkingSequence(OutputSampleRate)
	if len(samples) == 0 {
		t.Fatal("GenerateThinkingSequence returned empty samples")
	}

	// Expected: two notes of 400ms + two gaps of 50ms.
	expectedLen := int(float64(OutputSampleRate) * (thinkingNoteDuration*2 + thinkingGapDuration*2))
	if len(samples) != expectedLen {
		t.Errorf("expected %d samples, got %d", expectedLen, len(samples))
	}

	// All samples should be within volume bounds.
	for i, s := range samples {
		if math.Abs(float64(s)) > float64(thinkingVolume)+0.01 {
			t.Fatalf("sample %d exceeds thinking volume: %f", i, s)
		}
	}
}

func TestResample(t *testing.T) {
	// Generate 1 second of 440Hz tone at 16kHz.
	src := make([]float32, 16000)
	for i := range src {
		src[i] = float32(math.Sin(2 * math.Pi * 440.0 * float64(i) / 16000.0))
	}

	// Resample to 44100.
	dst := Resample(src, 16000, 44100)
	expectedLen := int(float64(len(src)) * 44100.0 / 16000.0)
	if len(dst) != expectedLen {
		t.Errorf("expected resampled length %d, got %d", expectedLen, len(dst))
	}

	// All values should still be in [-1, 1].
	for i, s := range dst {
		if s < -1.0 || s > 1.0 {
			t.Fatalf("resampled sample %d out of range: %f", i, s)
		}
	}
}

func TestResampleSameRate(t *testing.T) {
	src := []float32{0.1, 0.2, 0.3}
	dst := Resample(src, 44100, 44100)
	if len(dst) != len(src) {
		t.Errorf("same-rate resample changed length: %d -> %d", len(src), len(dst))
	}
}

func TestNormalizeFloat32(t *testing.T) {
	// Samples already in range should be unchanged.
	normal := []float32{-0.5, 0.0, 0.5}
	result := normalizeFloat32(normal)
	for i := range normal {
		if result[i] != normal[i] {
			t.Errorf("sample %d changed: %f -> %f", i, normal[i], result[i])
		}
	}

	// Samples in int16 range should be normalized.
	int16Samples := []float32{-32768, 0, 32767}
	result = normalizeFloat32(int16Samples)
	if result[0] > -0.99 || result[0] < -1.01 {
		t.Errorf("expected ~-1.0, got %f", result[0])
	}
	if result[2] < 0.99 || result[2] > 1.01 {
		t.Errorf("expected ~1.0, got %f", result[2])
	}
}
