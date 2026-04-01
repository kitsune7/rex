package agent

// AgentResult holds the outcome of an agent invocation.
type AgentResult struct {
	// Response is the final text response from the agent. Empty when a
	// confirmation is pending.
	Response string

	// Pending is non-nil when a tool call requires user confirmation before
	// it can be executed.
	Pending *PendingConfirmation

	// History is the updated conversation history after this invocation.
	History []Message

	// ThreadID identifies the conversation thread.
	ThreadID string
}

// PendingConfirmation represents a tool call that is waiting for the user to
// approve or reject it.
type PendingConfirmation struct {
	ToolName           string
	ToolArgs           map[string]any
	ConfirmationPrompt string
	ToolCallID         string
	ThreadID           string
}

// Message represents a single message in the conversation history.
type Message struct {
	Role       string     // "system", "user", "assistant", "tool"
	Content    string     // text content of the message
	ToolCalls  []ToolCall // present on assistant messages that invoke tools
	ToolCallID string     // present on tool response messages
}

// ToolCall represents a single tool invocation requested by the model.
type ToolCall struct {
	ID   string
	Name string
	Args map[string]any
}
