// Package core provides the foundational types for the Rex voice assistant,
// including the state machine, event bus, and application context.
package core

import "fmt"

// ConversationState represents the possible states in the Rex conversation flow.
type ConversationState int

const (
	// StateWaitingForWakeWord is the idle state, listening for the wake word.
	StateWaitingForWakeWord ConversationState = iota
	// StateListening is actively capturing user speech.
	StateListening
	// StateProcessing is sending the transcript to the AI agent.
	StateProcessing
	// StateSpeaking is playing back the AI response via TTS.
	StateSpeaking
	// StateAwaitingConfirmation waits for the user to confirm or reject an action.
	StateAwaitingConfirmation
	// StateDeliveringReminder plays a reminder that has come due.
	StateDeliveringReminder
	// StateShuttingDown signals the state machine to stop.
	StateShuttingDown
)

// String returns a human-readable name for the state.
func (s ConversationState) String() string {
	switch s {
	case StateWaitingForWakeWord:
		return "WaitingForWakeWord"
	case StateListening:
		return "Listening"
	case StateProcessing:
		return "Processing"
	case StateSpeaking:
		return "Speaking"
	case StateAwaitingConfirmation:
		return "AwaitingConfirmation"
	case StateDeliveringReminder:
		return "DeliveringReminder"
	case StateShuttingDown:
		return "ShuttingDown"
	default:
		return fmt.Sprintf("ConversationState(%d)", int(s))
	}
}

// StateResult is returned by a StateHandler to indicate the next state and
// any data to pass along to it.
type StateResult struct {
	NextState ConversationState
	Data      map[string]any
}

// StateHandler defines the interface that each conversation state must implement.
type StateHandler interface {
	// State returns the ConversationState this handler is responsible for.
	State() ConversationState
	// Enter is called when the state machine transitions into this state.
	// data contains optional information passed from the previous state.
	Enter(ctx *AppContext, data map[string]any)
	// Process performs the main work for this state and returns the next state.
	Process(ctx *AppContext) StateResult
	// Exit is called when the state machine transitions out of this state.
	Exit(ctx *AppContext)
}
