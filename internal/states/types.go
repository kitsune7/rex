// Package states implements conversation state handlers for the Rex voice assistant.
//
// Each handler corresponds to a stage in the conversation lifecycle:
// wake word detection, listening, processing, speaking, confirmation, and reminder delivery.
package states

import (
	"context"
	"time"
)

// Conversation state constants.
const (
	StateWaitingForWakeWord = iota
	StateListening
	StateProcessing
	StateSpeaking
	StateAwaitingConfirmation
	StateDeliveringReminder
	StateShuttingDown
)

// StateResult is returned by a handler's Process method to indicate the next
// state and any data to pass to it.
type StateResult struct {
	NextState int
	Data      map[string]any
}

// StateHandler is the interface that all conversation state handlers implement.
type StateHandler interface {
	// State returns the conversation state this handler manages.
	State() int
	// Enter is called when transitioning into this state. data may be nil.
	Enter(data map[string]any)
	// Process runs the handler logic and returns the next state.
	Process(ctx context.Context) StateResult
	// Exit is called when transitioning out of this state.
	Exit()
}

// Message represents a single message in the conversation history.
type Message struct {
	Role    string
	Content string
}

// AgentResult holds the response from the LLM agent.
type AgentResult struct {
	Response           string
	History            []Message
	ThreadID           string
	PendingConfirmation *PendingConfirmation
}

// PendingConfirmation represents a tool call awaiting user approval.
type PendingConfirmation struct {
	ConfirmationPrompt string
}

// ReminderDelivery holds the information needed to deliver a reminder.
type ReminderDelivery struct {
	ID      int64
	Message string
}

// Settings holds configuration values used by state handlers.
type Settings struct {
	ListeningTimeout time.Duration
	DingSoundPath    string
	RetryMinutes     int
}

// Listener abstracts wake-word detection and speech capture.
type Listener interface {
	// WaitForWakeWordAndSpeech blocks until the wake word is detected,
	// calls onWakeWord, then captures the following speech audio.
	// Returns nil audio if interrupted or no speech is captured.
	WaitForWakeWordAndSpeech(ctx context.Context, onWakeWord func()) ([]int16, error)
	// ListenForSpeech captures speech audio with the given timeout.
	// Returns nil if no speech is detected before the timeout.
	ListenForSpeech(ctx context.Context, timeout time.Duration, playTones bool) ([]int16, error)
	// IsInterrupted reports whether the listener was externally interrupted.
	IsInterrupted() bool
	// Stop halts the listener.
	Stop()
}

// Transcriber converts captured audio to text.
type Transcriber interface {
	Transcribe(audio []int16, stripWakeWord bool) (string, error)
}

// Speaker plays text-to-speech with interruption support.
type Speaker interface {
	// SpeakInterruptibly speaks the text and returns whether the user
	// interrupted, any audio captured during the interruption, and an error.
	SpeakInterruptibly(ctx context.Context, text string) (interrupted bool, capturedAudio []int16, err error)
	// SpeakBlocking speaks the text and blocks until done.
	SpeakBlocking(ctx context.Context, text string) error
}

// Agent runs user queries through the LLM.
type Agent interface {
	Run(ctx context.Context, query string, history []Message, threadID string) (*AgentResult, error)
	ConfirmToolCall(ctx context.Context, pending *PendingConfirmation, confirmed bool, history []Message) (*AgentResult, error)
}

// AudioPlayer controls sound effects and tones.
type AudioPlayer interface {
	StartThinkingTone()
	StopThinkingTone()
	PlayListeningTone()
	PlayDoneTone()
	PlaySoundFile(path string, blocking bool) error
}

// TimerController manages timer alarm muting.
type TimerController interface {
	Mute()
	Unmute()
	StopAnyRinging() bool
}

// ReminderScheduler manages reminder delivery lifecycle.
type ReminderScheduler interface {
	GetPendingDelivery() *ReminderDelivery
	ClearPendingDelivery()
	MarkDelivered(id int64) error
	ScheduleRetry(id int64) error
	SnoozeReminder(id int64, duration time.Duration) error
}
