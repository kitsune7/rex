package states

import (
	"context"
	"log/slog"
)

// WaitingHandler implements the WAITING_FOR_WAKE_WORD state.
// It idles until the wake word is detected or a reminder becomes due.
type WaitingHandler struct {
	deps *Deps
}

// NewWaitingHandler creates a WaitingHandler.
func NewWaitingHandler(deps *Deps) *WaitingHandler {
	return &WaitingHandler{deps: deps}
}

func (h *WaitingHandler) State() int { return StateWaitingForWakeWord }

// Enter resets conversation state and unmutes timer sounds.
func (h *WaitingHandler) Enter(_ map[string]any) {
	h.deps.ResetConversation()
	if h.deps.Timers != nil {
		h.deps.Timers.Unmute()
	}
	slog.Info("listening for wake word")
}

// Process waits for the wake word or a pending reminder.
func (h *WaitingHandler) Process(ctx context.Context) StateResult {
	// Check for pending reminder delivery first.
	if delivery := h.deps.Scheduler.GetPendingDelivery(); delivery != nil {
		return StateResult{
			NextState: StateDeliveringReminder,
			Data:      map[string]any{"delivery": delivery},
		}
	}

	// Wait for wake word, muting timers on detection.
	onWakeWord := func() {
		if h.deps.Timers != nil {
			h.deps.Timers.Mute()
		}
	}

	audio, err := h.deps.Listener.WaitForWakeWordAndSpeech(ctx, onWakeWord)
	if err != nil {
		// Check for interruption — may be a reminder or shutdown.
		if h.deps.Listener.IsInterrupted() {
			if delivery := h.deps.Scheduler.GetPendingDelivery(); delivery != nil {
				return StateResult{
					NextState: StateDeliveringReminder,
					Data:      map[string]any{"delivery": delivery},
				}
			}
			return StateResult{NextState: StateShuttingDown}
		}
		slog.Error("wake word listener error", "err", err)
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	if ctx.Err() != nil {
		return StateResult{NextState: StateShuttingDown}
	}

	if audio == nil {
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	return StateResult{
		NextState: StateListening,
		Data: map[string]any{
			"audio":               audio,
			"is_wake_word_trigger": true,
		},
	}
}

func (h *WaitingHandler) Exit() {}
