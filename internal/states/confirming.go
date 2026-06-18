package states

import (
	"context"
	"log/slog"
	"time"
)

const confirmationTimeout = 10 * time.Second

// ConfirmingHandler implements the AWAITING_CONFIRMATION state.
// It prompts the user to approve or reject a pending tool call and
// forwards the decision to the agent.
type ConfirmingHandler struct {
	deps    *Deps
	pending *PendingConfirmation
}

// NewConfirmingHandler creates a ConfirmingHandler.
func NewConfirmingHandler(deps *Deps) *ConfirmingHandler {
	return &ConfirmingHandler{deps: deps}
}

func (h *ConfirmingHandler) State() int { return StateAwaitingConfirmation }

// Enter stores the pending confirmation passed from the processing state.
func (h *ConfirmingHandler) Enter(data map[string]any) {
	h.pending = nil
	if data == nil {
		return
	}
	if v, ok := data["pending"].(*PendingConfirmation); ok {
		h.pending = v
	}
}

// Process speaks the confirmation prompt, captures the user's response,
// and relays the decision to the agent.
func (h *ConfirmingHandler) Process(ctx context.Context) StateResult {
	if h.pending == nil {
		return StateResult{
			NextState: StateSpeaking,
			Data: map[string]any{
				"response": "Something went wrong with the confirmation.",
			},
		}
	}

	slog.Info("asking for confirmation", "prompt", h.pending.ConfirmationPrompt)

	transcription := speakAndCapture(ctx, h.deps, h.pending.ConfirmationPrompt, confirmationTimeout)
	if transcription == "" {
		slog.Info("no confirmation received, cancelling")
		return h.reject(ctx)
	}

	slog.Info("confirmation response", "text", transcription)

	confirmed := IsConfirmation(transcription)
	if confirmed {
		slog.Info("tool call confirmed")
	} else {
		slog.Info("tool call rejected")
	}

	result, err := h.deps.Agent.ConfirmToolCall(ctx, h.pending, confirmed, h.deps.History)
	if err != nil {
		slog.Error("agent confirm error", "err", err)
		return StateResult{
			NextState: StateSpeaking,
			Data: map[string]any{
				"response":               "Sorry, I had trouble processing the confirmation.",
				"force_end_conversation": true,
			},
		}
	}

	h.deps.History = result.History
	return StateResult{
		NextState: StateSpeaking,
		Data: map[string]any{
			"response":               result.Response,
			"force_end_conversation": true,
		},
	}
}

// reject sends a rejection to the agent and returns a speaking result.
func (h *ConfirmingHandler) reject(ctx context.Context) StateResult {
	result, err := h.deps.Agent.ConfirmToolCall(ctx, h.pending, false, h.deps.History)
	if err != nil {
		slog.Error("agent reject error", "err", err)
		return StateResult{
			NextState: StateSpeaking,
			Data: map[string]any{
				"response":               "Cancelled.",
				"force_end_conversation": true,
			},
		}
	}
	h.deps.History = result.History
	return StateResult{
		NextState: StateSpeaking,
		Data: map[string]any{
			"response":               result.Response,
			"force_end_conversation": true,
		},
	}
}

// Exit clears the pending confirmation.
func (h *ConfirmingHandler) Exit() {
	h.pending = nil
}
