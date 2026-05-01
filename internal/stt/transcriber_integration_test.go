package stt_test

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-audio/wav"

	"rex/internal/stt"
)

// TestTranscriber_JFKSample is an end-to-end test that boots
// whisper-server against a canned WAV and asserts we get a plausible
// transcript back. It is gated behind REX_STT_MODEL so CI (and normal
// `go test ./...`) stays fast and hermetic: only developers that have
// a GGML model handy opt in.
//
// To run:
//
//	REX_STT_MODEL=$HOME/.../ggml-small.en.bin go test ./internal/stt
func TestTranscriber_JFKSample(t *testing.T) {
	model := os.Getenv("REX_STT_MODEL")
	if model == "" {
		t.Skip("set REX_STT_MODEL to a ggml whisper model to run this test")
	}
	if _, err := exec.LookPath("whisper-server"); err != nil {
		t.Skipf("whisper-server not on PATH: %v", err)
	}
	if _, err := os.Stat(model); err != nil {
		t.Skipf("model missing: %v", err)
	}

	samplePath, err := filepath.Abs(filepath.Join("testdata", "jfk.wav"))
	if err != nil {
		t.Fatalf("resolving sample: %v", err)
	}
	samples, err := readInt16WAV(samplePath)
	if err != nil {
		t.Fatalf("reading sample wav: %v", err)
	}

	tr, err := stt.NewTranscriber(stt.Options{
		ModelPath:      model,
		StartTimeout:   90 * time.Second,
		RequestTimeout: 60 * time.Second,
	})
	if err != nil {
		t.Fatalf("NewTranscriber: %v", err)
	}
	t.Cleanup(func() { _ = tr.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	text, err := tr.Transcribe(ctx, samples, false)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	lower := strings.ToLower(text)
	// JFK: "And so, my fellow Americans, ask not what your country can
	// do for you — ask what you can do for your country."
	// We tolerate Whisper's punctuation and contractions by matching
	// just the phrase fragments that any reasonable model run produces.
	for _, needle := range []string{"ask not", "country", "you can do"} {
		if !strings.Contains(lower, needle) {
			t.Fatalf("transcript missing %q: %q", needle, text)
		}
	}
}

// readInt16WAV reads a 16 kHz mono PCM WAV into the []int16 form
// Transcriber expects.
func readInt16WAV(path string) ([]int16, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := wav.NewDecoder(f)
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, err
	}
	out := make([]int16, len(buf.Data))
	for i, v := range buf.Data {
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		out[i] = int16(v)
	}
	return out, nil
}
