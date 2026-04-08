package agent

// Tool defines the interface that all agent tools must implement.
type Tool interface {
	// Name returns the tool's unique identifier used in API calls.
	Name() string

	// Description returns a human-readable description of what the tool does.
	Description() string

	// Parameters returns a JSON Schema describing the tool's input parameters.
	Parameters() map[string]any

	// Execute runs the tool with the given arguments and returns a result string.
	Execute(args map[string]any) (string, error)

	// RequiresConfirmation reports whether the tool should pause for user
	// confirmation before executing.
	RequiresConfirmation() bool
}
