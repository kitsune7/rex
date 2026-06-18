package states

import (
	"context"
	"log/slog"
	"strings"
)

// SpeakingHandler implements the SPEAKING state.
// It plays the agent's response via TTS and handles user interruptions.
type SpeakingHandler struct {
	deps                 *Deps
	response             string
	forceEndConversation bool
}

// NewSpeakingHandler creates a SpeakingHandler.
func NewSpeakingHandler(deps *Deps) *SpeakingHandler {
	return &SpeakingHandler{deps: deps}
}

func (h *SpeakingHandler) State() int { return StateSpeaking }

// Enter stores the response text to speak.
func (h *SpeakingHandler) Enter(data map[string]any) {
	h.response = ""
	h.forceEndConversation = false
	if data == nil {
		return
	}
	if v, ok := data["response"].(string); ok {
		h.response = v
	}
	if v, ok := data["force_end_conversation"].(bool); ok {
		h.forceEndConversation = v
	}
}

// Process speaks the response and determines the next state based on
// whether the user interrupted or the response ends with a question.
func (h *SpeakingHandler) Process(ctx context.Context) StateResult {
	if h.response == "" {
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	slog.Info("speaking", "text", h.response)

	interrupted, capturedAudio, err := h.deps.Speaker.SpeakInterruptibly(ctx, h.response)
	if err != nil {
		slog.Error("speak error", "err", err)
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	if interrupted {
		slog.Info("speech interrupted by user")
		return StateResult{
			NextState: StateListening,
			Data: map[string]any{
				"audio":               capturedAudio,
				"is_wake_word_trigger": true,
			},
		}
	}

	if h.forceEndConversation {
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	// Continue listening if Rex asked a question.
	if strings.HasSuffix(strings.TrimSpace(h.response), "?") {
		return StateResult{
			NextState: StateListening,
			Data:      map[string]any{"audio": nil, "is_wake_word_trigger": false},
		}
	}

	return StateResult{NextState: StateWaitingForWakeWord}
}

// Exit clears transient state.
func (h *SpeakingHandler) Exit() {
	h.response = ""
	h.forceEndConversation = false
}
