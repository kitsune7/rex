package agent

import (
	"strings"
	"testing"
	"time"
)

func TestBuildSystemPrompt(t *testing.T) {
	prompt := BuildSystemPrompt()

	today := time.Now().Format("Monday, January 2, 2006")
	if !strings.Contains(prompt, today) {
		t.Errorf("prompt should contain today's date %q, got: %s", today, prompt)
	}
	if !strings.Contains(prompt, "Rex") {
		t.Error("prompt should mention Rex")
	}
	if !strings.Contains(prompt, "VOICE assistant") {
		t.Error("prompt should mention VOICE assistant")
	}
}

func TestFormatConfirmationPrompt_CreateReminder(t *testing.T) {
	args := map[string]any{
		"message":      "take out trash",
		"datetime_str": "tomorrow at 8am",
	}
	prompt := FormatConfirmationPrompt("create_reminder", args)

	if !strings.Contains(prompt, "take out trash") {
		t.Errorf("expected prompt to contain message, got: %s", prompt)
	}
	if !strings.Contains(prompt, "tomorrow at 8am") {
		t.Errorf("expected prompt to contain datetime, got: %s", prompt)
	}
}

func TestFormatConfirmationPrompt_Generic(t *testing.T) {
	prompt := FormatConfirmationPrompt("delete_something", map[string]any{"id": "123"})

	if !strings.Contains(prompt, "delete_something") {
		t.Errorf("expected prompt to contain tool name, got: %s", prompt)
	}
	if !strings.Contains(prompt, "Should I proceed?") {
		t.Errorf("expected prompt to ask for confirmation, got: %s", prompt)
	}
}

func TestTrimHistory(t *testing.T) {
	a := &Agent{}

	// Build a history with system + 25 messages.
	history := []Message{{Role: "system", Content: "system prompt"}}
	for i := 0; i < 25; i++ {
		history = append(history, Message{Role: "user", Content: "msg"})
	}

	trimmed := a.trimHistory(history)

	if len(trimmed) != maxHistoryMessages {
		t.Errorf("expected %d messages, got %d", maxHistoryMessages, len(trimmed))
	}
	if trimmed[0].Role != "system" {
		t.Error("first message should be system prompt")
	}
}

func TestTrimHistory_NoTrimNeeded(t *testing.T) {
	a := &Agent{}

	history := []Message{
		{Role: "system", Content: "system prompt"},
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}

	trimmed := a.trimHistory(history)

	if len(trimmed) != 3 {
		t.Errorf("expected 3 messages (no trim), got %d", len(trimmed))
	}
}

func TestNewAgent_Defaults(t *testing.T) {
	a := NewAgent("", "", nil)

	if a.model == "" {
		t.Error("model should have a default value")
	}
	if a.client == nil {
		t.Error("client should not be nil")
	}
}

// mockTool is a test double implementing the Tool interface.
type mockTool struct {
	name                 string
	requiresConfirmation bool
	executeResult        string
	executeErr           error
}

func (m *mockTool) Name() string                          { return m.name }
func (m *mockTool) Description() string                   { return "mock tool" }
func (m *mockTool) Parameters() map[string]any            { return map[string]any{"type": "object"} }
func (m *mockTool) Execute(args map[string]any) (string, error) { return m.executeResult, m.executeErr }
func (m *mockTool) RequiresConfirmation() bool            { return m.requiresConfirmation }

func TestFindTool(t *testing.T) {
	tools := []Tool{
		&mockTool{name: "alpha"},
		&mockTool{name: "beta"},
	}
	a := NewAgent("", "", tools)

	if found := a.findTool("alpha"); found == nil {
		t.Error("should find alpha")
	}
	if found := a.findTool("gamma"); found != nil {
		t.Error("should not find gamma")
	}
}

func TestConvertMessages(t *testing.T) {
	a := &Agent{}
	msgs := []Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello", ToolCalls: []ToolCall{
			{ID: "tc1", Name: "foo", Args: map[string]any{"x": float64(1)}},
		}},
		{Role: "tool", Content: "result", ToolCallID: "tc1"},
	}

	converted := a.convertMessages(msgs)

	if len(converted) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(converted))
	}
	if converted[0].Role != "system" {
		t.Error("first message role should be system")
	}
	if converted[3].ToolCallID != "tc1" {
		t.Error("tool message should have ToolCallID")
	}
	if len(converted[2].ToolCalls) != 1 {
		t.Error("assistant message should have 1 tool call")
	}
}

func TestOpenAITools(t *testing.T) {
	tools := []Tool{&mockTool{name: "test_tool"}}
	a := NewAgent("", "", tools)

	oaiTools := a.openAITools()

	if len(oaiTools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(oaiTools))
	}
	if oaiTools[0].Function.Name != "test_tool" {
		t.Errorf("expected tool name test_tool, got %s", oaiTools[0].Function.Name)
	}
}

func TestConfirmToolCall_Rejected(t *testing.T) {
	a := NewAgent("", "", nil)

	pending := &PendingConfirmation{
		ToolName:   "create_reminder",
		ToolArgs:   map[string]any{"message": "test"},
		ToolCallID: "tc-123",
		ThreadID:   "thread-1",
	}

	result, err := a.ConfirmToolCall(nil, pending, false, []Message{
		{Role: "system", Content: "sys"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Response != "Okay, I've cancelled that action." {
		t.Errorf("unexpected response: %s", result.Response)
	}
	if result.ThreadID != "thread-1" {
		t.Errorf("unexpected thread ID: %s", result.ThreadID)
	}
	// History should have system + tool cancellation message.
	if len(result.History) != 2 {
		t.Errorf("expected 2 messages in history, got %d", len(result.History))
	}
}
