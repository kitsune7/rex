//go:build whisper

// Build requirements:
//
// The Transcriber uses whisper.cpp via its Go/CGo bindings. To compile this
// file you need:
//   1. A working C/C++ toolchain (gcc or clang).
//   2. The whisper.cpp library built from source
//      (https://github.com/ggerganov/whisper.cpp).
//   3. CGO_ENABLED=1 and appropriate CGO_CFLAGS / CGO_LDFLAGS pointing at the
//      whisper.cpp headers and library.
//
// Build with:
//
//	go build -tags whisper ./internal/stt/...
//
// The strip.go / strip_test.go files are pure Go and compile without CGo.

package stt

import (
	"fmt"
	"log/slog"
	"strings"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

// Transcriber wraps a whisper.cpp model and provides speech-to-text
// transcription for 16 kHz int16 audio.
type Transcriber struct {
	model whisper.Model
}

// NewTranscriber loads a whisper model from modelPath (e.g.
// "models/whisper/ggml-small.bin") and returns a ready-to-use Transcriber.
func NewTranscriber(modelPath string) (*Transcriber, error) {
	slog.Info("loading whisper model", "path", modelPath)

	model, err := whisper.New(modelPath)
	if err != nil {
		return nil, fmt.Errorf("stt: load whisper model %q: %w", modelPath, err)
	}

	return &Transcriber{model: model}, nil
}

// Close releases the underlying whisper model resources.
func (t *Transcriber) Close() {
	if t.model != nil {
		t.model.Close()
	}
}

// Transcribe converts int16 PCM audio samples (16 kHz, mono) to text.
// If stripWakeWordFlag is true the wake-word prefix is removed from the result.
func (t *Transcriber) Transcribe(audio []int16, stripWakeWordFlag bool) (string, error) {
	if len(audio) == 0 {
		return "", nil
	}

	// Convert int16 samples to float32 in [-1, 1].
	samples := make([]float32, len(audio))
	for i, s := range audio {
		samples[i] = float32(s) / 32768.0
	}

	ctx, err := t.model.NewContext()
	if err != nil {
		return "", fmt.Errorf("stt: new context: %w", err)
	}

	// Configure transcription parameters.
	ctx.SetLanguage("en")

	if err := ctx.Process(samples); err != nil {
		return "", fmt.Errorf("stt: process audio: %w", err)
	}

	// Collect text from all segments.
	var parts []string
	for i := 0; i < ctx.NumSegments(); i++ {
		parts = append(parts, ctx.SegmentText(i))
	}
	text := strings.TrimSpace(strings.Join(parts, " "))

	if stripWakeWordFlag {
		text = stripWakeWord(text)
	} else {
		text = capitalizeFirst(text)
	}

	return text, nil
}
