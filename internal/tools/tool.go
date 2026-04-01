// Package tools provides agent tools for the Rex voice assistant.
//
// Tools include timer management, reminder management (SQLite-backed),
// math evaluation, and current time retrieval.
package tools

// Tool defines the interface that all agent tools must implement.
type Tool interface {
	// Name returns the tool's identifier used by the agent.
	Name() string
	// Description returns a human-readable description of what the tool does.
	Description() string
	// Parameters returns a JSON-Schema-like map describing the tool's parameters.
	Parameters() map[string]any
	// Execute runs the tool with the given arguments and returns a result string.
	Execute(args map[string]any) (string, error)
	// RequiresConfirmation returns true if the tool needs user confirmation before executing.
	RequiresConfirmation() bool
}

// Event represents a system event emitted by tools.
type Event struct {
	Type string
	Data map[string]any
}

// EventEmitter is the interface for publishing events to the system event bus.
type EventEmitter interface {
	Emit(event Event)
}

// AudioPlayer is the interface for playing alarm sounds.
type AudioPlayer interface {
	StartLoop(soundPath string)
	StopLoop()
}
