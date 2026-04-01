package states

import (
	"context"
	"log/slog"
	"strings"
	"time"
)

// trimLower trims whitespace and lowercases a string.
func trimLower(s string) string {
	return strings.TrimSpace(strings.ToLower(s))
}

// speakAndCapture speaks text interruptibly, then listens for a response if
// the user did not interrupt. It returns the transcription of whatever the
// user said (empty string if nothing was captured or understood).
func speakAndCapture(ctx context.Context, deps *Deps, text string, listenTimeout time.Duration) string {
	interrupted, capturedAudio, err := deps.Speaker.SpeakInterruptibly(ctx, text)
	if err != nil {
		slog.Error("speak error", "err", err)
		return ""
	}

	var audio []int16
	stripWakeWord := false

	if interrupted && len(capturedAudio) > 0 {
		slog.Info("speech interrupted with response")
		audio = capturedAudio
		stripWakeWord = true
	} else {
		slog.Info("listening for response")
		audio, err = deps.Listener.ListenForSpeech(ctx, listenTimeout, false)
		if err != nil {
			slog.Error("listen error", "err", err)
			return ""
		}
	}

	if audio == nil {
		return ""
	}

	transcription, err := deps.Transcriber.Transcribe(audio, stripWakeWord)
	if err != nil {
		slog.Error("transcription error", "err", err)
		return ""
	}
	return transcription
}
