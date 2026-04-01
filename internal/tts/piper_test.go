package tts

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestPcmToFloat32_Basic(t *testing.T) {
	// Encode a few known int16 values as little-endian bytes.
	values := []int16{0, math.MaxInt16, math.MinInt16, 1000, -1000}
	buf := make([]byte, len(values)*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
	}

	samples, err := pcmToFloat32(buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(samples) != len(values) {
		t.Fatalf("expected %d samples, got %d", len(values), len(samples))
	}

	// Check zero maps to 0.
	if samples[0] != 0 {
		t.Errorf("expected 0, got %f", samples[0])
	}
	// MaxInt16 should map to ~1.0.
	if math.Abs(float64(samples[1])-1.0) > 0.001 {
		t.Errorf("expected ~1.0, got %f", samples[1])
	}
	// MinInt16 should map to ~-1.0.
	if math.Abs(float64(samples[2])+1.0) > 0.001 {
		t.Errorf("expected ~-1.0, got %f", samples[2])
	}
}

func TestPcmToFloat32_OddBytes(t *testing.T) {
	_, err := pcmToFloat32([]byte{0x01, 0x02, 0x03})
	if err == nil {
		t.Fatal("expected error for odd byte count")
	}
}

func TestPcmToFloat32_Empty(t *testing.T) {
	samples, err := pcmToFloat32(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(samples) != 0 {
		t.Fatalf("expected 0 samples, got %d", len(samples))
	}
}

func TestNewVoice_MissingModel(t *testing.T) {
	_, err := NewVoice("/nonexistent/model.onnx")
	if err == nil {
		t.Fatal("expected error for missing model")
	}
}

func TestGenerateTone(t *testing.T) {
	tone := generateTone(440, 0.1, 22050)
	expectedLen := int(0.1 * 22050)
	if len(tone) != expectedLen {
		t.Fatalf("expected %d samples, got %d", expectedLen, len(tone))
	}

	// All samples should be in [-0.3, 0.3] given amplitude 0.3.
	for i, s := range tone {
		if s < -0.31 || s > 0.31 {
			t.Errorf("sample %d out of range: %f", i, s)
			break
		}
	}
}
