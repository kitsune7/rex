package scheduler

import (
	"context"
	"log/slog"
	"sync/atomic"
	"time"
)

// ReminderStore provides access to persisted reminders.
type ReminderStore interface {
	GetDueReminders() ([]Reminder, error)
	GetNextPendingTime() (*time.Time, error)
	UpdateReminder(id int64, message *string, dueAt *time.Time) error
	SnoozeReminder(id int64, duration time.Duration) error
	ClearReminder(id int64) error
}

// Reminder represents a persisted reminder record.
type Reminder struct {
	ID      int64
	Message string
	DueAt   time.Time
	Status  string
}

// AudioPlayer plays audio files.
type AudioPlayer interface {
	PlaySoundFile(path string, blocking bool) error
}

// EventBus allows subscribing to named events.
type EventBus interface {
	Subscribe(eventName string, callback func(event any))
}

// Scheduler runs a background loop that wakes precisely when the next reminder
// is due. It uses channels and select instead of polling, and communicates the
// current pending delivery to the state machine via an atomic pointer.
type Scheduler struct {
	reminderStore   ReminderStore
	audioPlayer     AudioPlayer
	dingSoundPath   string
	wakeCh          chan struct{}
	pendingDelivery atomic.Pointer[ReminderDelivery]
	retryMinutes    int
}

// NewScheduler creates a Scheduler and subscribes to schedule-change events on
// the provided EventBus.
func NewScheduler(store ReminderStore, player AudioPlayer, eventBus EventBus, retryMinutes int) *Scheduler {
	s := &Scheduler{
		reminderStore: store,
		audioPlayer:   player,
		dingSoundPath: "sounds/ding.mp3",
		wakeCh:        make(chan struct{}, 1),
		retryMinutes:  retryMinutes,
	}

	eventBus.Subscribe("ReminderScheduleChanged", func(_ any) {
		// Non-blocking write: if the channel already has a value the loop
		// will wake up anyway, so dropping a duplicate signal is fine.
		select {
		case s.wakeCh <- struct{}{}:
		default:
		}
	})

	return s
}

// Start launches the scheduler loop in a goroutine. The loop runs until the
// provided context is cancelled.
func (s *Scheduler) Start(ctx context.Context) {
	go s.run(ctx)
}

func (s *Scheduler) run(ctx context.Context) {
	// Check immediately on startup for any already-due reminders.
	s.deliverDueReminders()

	for {
		nextTime, err := s.reminderStore.GetNextPendingTime()
		if err != nil {
			slog.Error("failed to get next pending time", "error", err)
			// Back off briefly before retrying.
			backoff := time.NewTimer(5 * time.Second)
			select {
			case <-backoff.C:
				continue
			case <-ctx.Done():
				backoff.Stop()
				return
			}
		}

		if nextTime == nil {
			// No pending reminders — wait for a wake signal or shutdown.
			select {
			case <-s.wakeCh:
				continue
			case <-ctx.Done():
				return
			}
		}

		timer := time.NewTimer(time.Until(*nextTime))
		select {
		case <-timer.C:
			s.deliverDueReminders()
		case <-s.wakeCh:
			timer.Stop()
			continue
		case <-ctx.Done():
			timer.Stop()
			return
		}
	}
}

// deliverDueReminders checks the store for due reminders and triggers delivery
// of the first one found. If a delivery is already pending it is a no-op.
func (s *Scheduler) deliverDueReminders() {
	if s.pendingDelivery.Load() != nil {
		return
	}

	reminders, err := s.reminderStore.GetDueReminders()
	if err != nil {
		slog.Error("failed to get due reminders", "error", err)
		return
	}
	if len(reminders) == 0 {
		return
	}

	r := reminders[0]
	delivery := &ReminderDelivery{
		ID:      r.ID,
		Message: r.Message,
		DueAt:   r.DueAt,
	}
	s.pendingDelivery.Store(delivery)

	if err := s.audioPlayer.PlaySoundFile(s.dingSoundPath, true); err != nil {
		slog.Warn("could not play ding sound", "error", err)
	}
}

// GetPendingDelivery returns the current pending delivery, or nil if none.
func (s *Scheduler) GetPendingDelivery() *ReminderDelivery {
	return s.pendingDelivery.Load()
}

// ClearPendingDelivery removes the current pending delivery.
func (s *Scheduler) ClearPendingDelivery() {
	s.pendingDelivery.Store(nil)
}

// MarkDelivered clears a reminder from the store and removes the pending
// delivery so the scheduler can move on.
func (s *Scheduler) MarkDelivered(id int64) error {
	if err := s.reminderStore.ClearReminder(id); err != nil {
		return err
	}
	s.ClearPendingDelivery()
	return nil
}

// ScheduleRetry snoozes the reminder by the configured retry interval and
// clears the pending delivery.
func (s *Scheduler) ScheduleRetry(id int64) error {
	if err := s.reminderStore.SnoozeReminder(id, time.Duration(s.retryMinutes)*time.Minute); err != nil {
		return err
	}
	s.ClearPendingDelivery()
	return nil
}

// SnoozeReminder snoozes the reminder for the given duration and clears the
// pending delivery.
func (s *Scheduler) SnoozeReminder(id int64, duration time.Duration) error {
	if err := s.reminderStore.SnoozeReminder(id, duration); err != nil {
		return err
	}
	s.ClearPendingDelivery()
	return nil
}
