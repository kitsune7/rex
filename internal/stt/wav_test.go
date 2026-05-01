package stt

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestEncodeWAV16Header(t *testing.T) {
	samples := []int16{0, 1, -1, 32767, -32768, 100, 200, 300}
	out, err := encodeWAV16(samples, 16000)
	if err != nil {
		t.Fatalf("encodeWAV16: %v", err)
	}
	if len(out) != 44+len(samples)*2 {
		t.Fatalf("expected %d bytes, got %d", 44+len(samples)*2, len(out))
	}
	if !bytes.Equal(out[0:4], []byte("RIFF")) {
		t.Fatalf("missing RIFF marker")
	}
	if !bytes.Equal(out[8:12], []byte("WAVE")) {
		t.Fatalf("missing WAVE marker")
	}
	if !bytes.Equal(out[12:16], []byte("fmt ")) {
		t.Fatalf("missing fmt chunk marker")
	}
	if !bytes.Equal(out[36:40], []byte("data")) {
		t.Fatalf("missing data chunk marker")
	}
	sr := binary.LittleEndian.Uint32(out[24:28])
	if sr != 16000 {
		t.Fatalf("sample rate: got %d want 16000", sr)
	}
	nchan := binary.LittleEndian.Uint16(out[22:24])
	if nchan != 1 {
		t.Fatalf("channels: got %d want 1", nchan)
	}
	bps := binary.LittleEndian.Uint16(out[34:36])
	if bps != 16 {
		t.Fatalf("bits per sample: got %d want 16", bps)
	}
	// Round-trip the first sample.
	got := int16(binary.LittleEndian.Uint16(out[44:46]))
	if got != samples[0] {
		t.Fatalf("first sample: got %d want %d", got, samples[0])
	}
	got = int16(binary.LittleEndian.Uint16(out[46:48]))
	if got != samples[1] {
		t.Fatalf("second sample: got %d want %d", got, samples[1])
	}
}
