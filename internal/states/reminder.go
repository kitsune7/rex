package states

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

const reminderResponseTimeout = 5 * time.Second

// ReminderHandler implements the DELIVERING_REMINDER state.
// It proactively delivers a due reminder and handles the user's response
// (acknowledge, snooze, or ignore).
type ReminderHandler struct {
	deps     *Deps
	delivery *ReminderDelivery
}

// NewReminderHandler creates a ReminderHandler.
func NewReminderHandler(deps *Deps) *ReminderHandler {
	return &ReminderHandler{deps: deps}
}

func (h *ReminderHandler) State() int { return StateDeliveringReminder }

// Enter stores the reminder delivery and mutes timer alarms.
func (h *ReminderHandler) Enter(data map[string]any) {
	h.delivery = nil
	if data != nil {
		if v, ok := data["delivery"].(*ReminderDelivery); ok {
			h.delivery = v
		}
	}
	if h.deps.Timers != nil {
		h.deps.Timers.Mute()
	}
}

// Process delivers the reminder, listens for the user's response, and
// takes the appropriate action (mark delivered, snooze, or retry).
func (h *ReminderHandler) Process(ctx context.Context) StateResult {
	if h.delivery == nil {
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	// Play ding sound.
	slog.Info("delivering reminder", "id", h.delivery.ID, "message", h.delivery.Message)
	if h.deps.Settings.DingSoundPath != "" {
		_ = h.deps.Audio.PlaySoundFile(h.deps.Settings.DingSoundPath, true)
	}

	// Speak the reminder as a question.
	reminderText := fmt.Sprintf(
		"You have a reminder: %s. Would you like to clear this reminder?",
		h.delivery.Message,
	)

	transcription := speakAndCapture(ctx, h.deps, reminderText, reminderResponseTimeout)
	if transcription == "" {
		slog.Info("no response to reminder, will retry later")
		_ = h.deps.Scheduler.ScheduleRetry(h.delivery.ID)
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	slog.Info("reminder response", "text", transcription)

	// Check for snooze request.
	if dur, ok := ParseSnoozeDuration(transcription); ok {
		if err := h.deps.Scheduler.SnoozeReminder(h.delivery.ID, dur); err != nil {
			slog.Error("snooze error", "err", err)
		}
		minutes := int(dur.Minutes())
		h.speakBlocking(ctx, fmt.Sprintf("Okay, I'll remind you again in %d minutes.", minutes))
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	// Check for confirmation (clear).
	if IsConfirmation(transcription) {
		if err := h.deps.Scheduler.MarkDelivered(h.delivery.ID); err != nil {
			slog.Error("mark delivered error", "err", err)
		}
		h.speakBlocking(ctx, "Reminder cleared.")
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	// Check for explicit rejection or "later".
	normalized := trimLower(transcription)
	if IsRejection(transcription) || strings.Contains(normalized, "later") || strings.Contains(normalized, "not now") {
		_ = h.deps.Scheduler.ScheduleRetry(h.delivery.ID)
		retryMins := 10
		if h.deps.Settings != nil && h.deps.Settings.RetryMinutes > 0 {
			retryMins = h.deps.Settings.RetryMinutes
		}
		h.speakBlocking(ctx, fmt.Sprintf("Okay, I'll remind you again in %d minutes.", retryMins))
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	// Unclear response — retry later.
	slog.Info("unclear reminder response, will retry later")
	_ = h.deps.Scheduler.ScheduleRetry(h.delivery.ID)
	return StateResult{NextState: StateWaitingForWakeWord}
}

// speakBlocking speaks text and blocks until done, ignoring errors.
func (h *ReminderHandler) speakBlocking(ctx context.Context, text string) {
	slog.Info("speaking", "text", text)
	_ = h.deps.Speaker.SpeakBlocking(ctx, text)
}

// Exit unmutes timers and clears delivery info.
func (h *ReminderHandler) Exit() {
	if h.deps.Timers != nil {
		h.deps.Timers.Unmute()
	}
	h.delivery = nil
}
