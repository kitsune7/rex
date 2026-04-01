package agent

import (
	"context"
	"fmt"
)

// FormatConfirmationPrompt builds a human-readable confirmation prompt for the
// given tool call, matching the Python implementation's phrasing.
func FormatConfirmationPrompt(toolName string, args map[string]any) string {
	if toolName == "create_reminder" {
		message, _ := args["message"].(string)
		datetimeStr, _ := args["datetime_str"].(string)
		return fmt.Sprintf("I'm about to create a reminder: '%s' for %s. Should I proceed?", message, datetimeStr)
	}
	return fmt.Sprintf("I'm about to run %s. Should I proceed?", toolName)
}

// ConfirmToolCall handles user confirmation or rejection of a pending tool
// call. If confirmed, the tool is executed and the agent loop continues to
// produce a final text response. If rejected, a cancellation message is
// returned without calling the LLM again.
func (a *Agent) ConfirmToolCall(ctx context.Context, pending *PendingConfirmation, confirmed bool, history []Message) (*AgentResult, error) {
	if !confirmed {
		// Append a tool message indicating cancellation.
		history = append(history, Message{
			Role:       "tool",
			Content:    "User cancelled this action.",
			ToolCallID: pending.ToolCallID,
		})
		return &AgentResult{
			Response: "Okay, I've cancelled that action.",
			History:  history,
			ThreadID: pending.ThreadID,
		}, nil
	}

	// Execute the confirmed tool.
	tool := a.findTool(pending.ToolName)
	if tool == nil {
		return nil, fmt.Errorf("tool %q not found", pending.ToolName)
	}

	result, err := tool.Execute(pending.ToolArgs)
	if err != nil {
		result = fmt.Sprintf("error: %v", err)
	}

	// Append the tool result to history and continue the agent loop.
	history = append(history, Message{
		Role:       "tool",
		Content:    result,
		ToolCallID: pending.ToolCallID,
	})

	return a.runLoop(ctx, history, pending.ThreadID)
}
