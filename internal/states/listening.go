package states

import (
	"context"
	"log/slog"
)

// ListeningHandler implements the LISTENING state.
// It captures or receives audio, transcribes it, and routes to the next state.
type ListeningHandler struct {
	deps             *Deps
	audio            []int16
	isWakeWordTrigger bool
}

// NewListeningHandler creates a ListeningHandler.
func NewListeningHandler(deps *Deps) *ListeningHandler {
	return &ListeningHandler{deps: deps}
}

func (h *ListeningHandler) State() int { return StateListening }

// Enter stores audio passed from a previous state (e.g. wake-word capture or interruption).
func (h *ListeningHandler) Enter(data map[string]any) {
	h.audio = nil
	h.isWakeWordTrigger = false
	if data == nil {
		return
	}
	if a, ok := data["audio"]; ok && a != nil {
		if samples, ok := a.([]int16); ok {
			h.audio = samples
		}
	}
	if v, ok := data["is_wake_word_trigger"]; ok {
		if b, ok := v.(bool); ok {
			h.isWakeWordTrigger = b
		}
	}
}

// Process transcribes existing audio or captures new speech, then decides
// the next state based on the transcription content.
func (h *ListeningHandler) Process(ctx context.Context) StateResult {
	// If no audio was provided (follow-up mode), listen for speech.
	if h.audio == nil {
		h.deps.Audio.PlayListeningTone()
		slog.Info("listening for response")

		timeout := h.deps.Settings.ListeningTimeout
		audio, err := h.deps.Listener.ListenForSpeech(ctx, timeout, true)
		if err != nil {
			slog.Error("listen error", "err", err)
			h.deps.Audio.PlayDoneTone()
			return StateResult{NextState: StateWaitingForWakeWord}
		}
		if audio == nil {
			slog.Info("no response received, ending conversation")
			h.deps.Audio.PlayDoneTone()
			return StateResult{NextState: StateWaitingForWakeWord}
		}
		h.audio = audio
	}

	// Transcribe the captured audio.
	transcription, err := h.deps.Transcriber.Transcribe(h.audio, h.isWakeWordTrigger)
	if err != nil {
		slog.Error("transcription error", "err", err)
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	if transcription == "" {
		if h.deps.IsInConversation() {
			return StateResult{
				NextState: StateListening,
				Data:      map[string]any{"audio": nil, "is_wake_word_trigger": false},
			}
		}
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	slog.Info("transcribed", "text", transcription)

	// Handle "stop" / "stop the timer" commands.
	normalized := trimLower(transcription)
	if normalized == "stop" || normalized == "stop the timer" {
		if h.deps.Timers != nil && h.deps.Timers.StopAnyRinging() {
			slog.Info("timer alarm stopped")
		}
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	// Stop phrases end the conversation during follow-up.
	if h.deps.IsInConversation() && IsStopPhrase(normalized) {
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	return StateResult{
		NextState: StateProcessing,
		Data:      map[string]any{"transcription": transcription},
	}
}

// Exit clears transient audio data.
func (h *ListeningHandler) Exit() {
	h.audio = nil
	h.isWakeWordTrigger = false
}

