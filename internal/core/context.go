package core

import (
	"time"

	"rex/internal/config"
)

// AudioPlayer provides audio playback capabilities.
type AudioPlayer interface {
	QueueAudio(samples []float32, sampleRate int)
	QueueAudioBlocking(samples []float32, sampleRate int, interruptCheck func() bool) bool
	StartLoop(samples []float32, sampleRate int)
	StopLoop()
	PlaySoundFile(path string, blocking bool) error
	GetSoundDuration(path string) (time.Duration, error)
	PlayListeningTone()
	PlayDoneTone()
	StartThinkingTone()
	StopThinkingTone()
	Mute()
	Unmute()
	Cleanup()
}

// TimerController manages timer alarms.
type TimerController interface {
	Mute()
	Unmute()
	StopAnyRinging() bool
	Cleanup()
}

// ReminderStatus represents the lifecycle state of a reminder.
type ReminderStatus int

const (
	// ReminderPending means the reminder has not yet been delivered.
	ReminderPending ReminderStatus = iota
	// ReminderDelivered means the reminder was delivered to the user.
	ReminderDelivered
	// ReminderCleared means the user acknowledged the reminder.
	ReminderCleared
)

// Reminder is a single scheduled reminder.
type Reminder struct {
	ID        int64
	Message   string
	DueAt     time.Time
	CreatedAt time.Time
	Status    ReminderStatus
}

// ReminderStore persists and queries reminders.
type ReminderStore interface {
	CreateReminder(message string, dueAt time.Time) (int64, error)
	ListReminders() ([]Reminder, error)
	GetDueReminders() ([]Reminder, error)
	GetNextPendingTime() (*time.Time, error)
	UpdateReminder(id int64, message *string, dueAt *time.Time) error
	DeleteReminder(id int64) error
	ClearReminder(id int64) error
	SnoozeReminder(id int64, duration time.Duration) error
}

// Message represents a single entry in conversation history.
type Message struct {
	Role    string
	Content string
}

// AppContext holds all shared dependencies for the Rex application.
// It is passed to state handlers and other components that need access
// to managers, settings, and conversation state.
type AppContext struct {
	Audio            AudioPlayer
	Timers           TimerController
	Reminders        ReminderStore
	Settings         *config.Settings
	EventBus         *EventBus
	History          []Message
	ThreadID         string
	TranscribedText  string
	ResponseText     string
	PendingConfirm   any // will be PendingConfirmation from agent package
	ReminderDelivery any // will be ReminderDelivery from scheduler
}

// ResetConversation clears all conversation state so Rex can start fresh.
func (c *AppContext) ResetConversation() {
	c.History = nil
	c.ThreadID = ""
	c.TranscribedText = ""
	c.ResponseText = ""
	c.PendingConfirm = nil
}

// IsInConversation reports whether Rex is in an active conversation
// (i.e. has conversation history).
func (c *AppContext) IsInConversation() bool {
	return c.History != nil
}
