package core

import (
	"context"
	"fmt"
	"log"
)

// StateMachine manages state transitions for the Rex conversation flow.
// It dispatches to registered StateHandlers and runs until it reaches
// StateShuttingDown or the context is cancelled.
type StateMachine struct {
	handlers     map[ConversationState]StateHandler
	currentState ConversationState
	running      bool
}

// NewStateMachine creates a StateMachine with the given handlers and initial state.
// It returns an error if any non-shutdown state is missing a handler.
func NewStateMachine(handlers []StateHandler, initialState ConversationState) (*StateMachine, error) {
	m := &StateMachine{
		handlers:     make(map[ConversationState]StateHandler, len(handlers)),
		currentState: initialState,
	}

	for _, h := range handlers {
		m.handlers[h.State()] = h
	}

	// Validate that every state except ShuttingDown has a handler.
	allStates := []ConversationState{
		StateWaitingForWakeWord,
		StateListening,
		StateProcessing,
		StateSpeaking,
		StateAwaitingConfirmation,
		StateDeliveringReminder,
	}
	for _, s := range allStates {
		if _, ok := m.handlers[s]; !ok {
			return nil, fmt.Errorf("missing handler for state %s", s)
		}
	}

	return m, nil
}

// Run executes the state machine loop. It blocks until the machine reaches
// StateShuttingDown, Stop is called, or ctx is cancelled.
func (m *StateMachine) Run(ctx context.Context, appCtx *AppContext) {
	m.running = true
	var transitionData map[string]any

	for m.running && m.currentState != StateShuttingDown {
		// Check for context cancellation.
		select {
		case <-ctx.Done():
			m.running = false
			return
		default:
		}

		handler, ok := m.handlers[m.currentState]
		if !ok {
			log.Printf("no handler for state %s, shutting down", m.currentState)
			break
		}

		handler.Enter(appCtx, transitionData)
		transitionData = nil

		result := handler.Process(appCtx)

		handler.Exit(appCtx)

		m.currentState = result.NextState
		transitionData = result.Data
	}

	m.running = false
}

// Stop requests the state machine to stop after the current state completes.
func (m *StateMachine) Stop() {
	m.running = false
	m.currentState = StateShuttingDown
}

// CurrentState returns the state machine's current state.
func (m *StateMachine) CurrentState() ConversationState {
	return m.currentState
}
