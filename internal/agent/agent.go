// Package agent implements the LLM agent with a manual tool-calling loop,
// replacing the previous LangChain/LangGraph implementation.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"

	"github.com/google/uuid"
	openai "github.com/sashabaranov/go-openai"
)

const (
	// maxHistoryMessages is the maximum number of messages retained in the
	// conversation history. When exceeded the oldest messages (after the
	// system prompt) are trimmed.
	maxHistoryMessages = 20

	defaultBaseURL = "http://localhost:1234/v1"
	defaultModel   = "lm-studio"
)

// Agent orchestrates LLM chat completions with tool calling.
type Agent struct {
	client *openai.Client
	tools  []Tool
	model  string
}

// NewAgent creates an Agent that talks to an OpenAI-compatible endpoint at
// baseURL. If baseURL is empty the default local LM Studio URL is used.
// apiKey may be "not-needed" for local servers.
func NewAgent(baseURL, apiKey string, tools []Tool) *Agent {
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	if apiKey == "" {
		apiKey = "not-needed"
	}

	model := os.Getenv("LLM_MODEL")
	if model == "" {
		model = defaultModel
	}

	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = baseURL
	client := openai.NewClientWithConfig(cfg)

	return &Agent{
		client: client,
		tools:  tools,
		model:  model,
	}
}

// Run sends a user query through the agent loop and returns the result. The
// caller supplies the current conversation history and an optional threadID;
// if threadID is empty a new UUID is generated.
func (a *Agent) Run(ctx context.Context, query string, history []Message, threadID string) (*AgentResult, error) {
	if threadID == "" {
		threadID = uuid.New().String()
	}

	// Ensure the system prompt is at the beginning of history.
	if len(history) == 0 || history[0].Role != "system" {
		history = append([]Message{{Role: "system", Content: BuildSystemPrompt()}}, history...)
	}

	// Add the user message.
	history = append(history, Message{Role: "user", Content: query})

	return a.runLoop(ctx, history, threadID)
}

// runLoop is the core tool-calling loop. It sends the conversation to the LLM
// repeatedly until a final text response (no tool calls) is produced, or a
// tool requiring confirmation is encountered.
func (a *Agent) runLoop(ctx context.Context, history []Message, threadID string) (*AgentResult, error) {
	tools := a.openAITools()

	for {
		msgs := a.convertMessages(history)

		resp, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
			Model:    a.model,
			Messages: msgs,
			Tools:    tools,
		})
		if err != nil {
			return nil, fmt.Errorf("chat completion: %w", err)
		}

		if len(resp.Choices) == 0 {
			return nil, fmt.Errorf("empty response from LLM")
		}

		choice := resp.Choices[0]

		// No tool calls -- return the text response.
		if len(choice.Message.ToolCalls) == 0 {
			history = append(history, Message{
				Role:    "assistant",
				Content: choice.Message.Content,
			})
			history = a.trimHistory(history)

			return &AgentResult{
				Response: choice.Message.Content,
				History:  history,
				ThreadID: threadID,
			}, nil
		}

		// Record the assistant message (with tool calls) in history.
		assistantMsg := Message{
			Role:    "assistant",
			Content: choice.Message.Content,
		}
		for _, tc := range choice.Message.ToolCalls {
			var args map[string]any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				slog.Warn("failed to parse tool args", "tool", tc.Function.Name, "err", err)
				args = map[string]any{}
			}
			assistantMsg.ToolCalls = append(assistantMsg.ToolCalls, ToolCall{
				ID:   tc.ID,
				Name: tc.Function.Name,
				Args: args,
			})
		}
		history = append(history, assistantMsg)

		// Process each tool call.
		for _, tc := range assistantMsg.ToolCalls {
			tool := a.findTool(tc.Name)
			if tool == nil {
				history = append(history, Message{
					Role:       "tool",
					Content:    fmt.Sprintf("error: unknown tool %q", tc.Name),
					ToolCallID: tc.ID,
				})
				continue
			}

			// If any tool requires confirmation, pause and return.
			if tool.RequiresConfirmation() {
				prompt := FormatConfirmationPrompt(tc.Name, tc.Args)
				history = a.trimHistory(history)

				return &AgentResult{
					Pending: &PendingConfirmation{
						ToolName:           tc.Name,
						ToolArgs:           tc.Args,
						ConfirmationPrompt: prompt,
						ToolCallID:         tc.ID,
						ThreadID:           threadID,
					},
					History:  history,
					ThreadID: threadID,
				}, nil
			}

			result, execErr := tool.Execute(tc.Args)
			if execErr != nil {
				result = fmt.Sprintf("error: %v", execErr)
			}

			history = append(history, Message{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
			})
		}

		// Loop back to send tool results to the LLM.
	}
}

// findTool returns the Tool with the given name, or nil if not found.
func (a *Agent) findTool(name string) Tool {
	for _, t := range a.tools {
		if t.Name() == name {
			return t
		}
	}
	return nil
}

// trimHistory keeps the system message and the most recent messages, dropping
// the oldest non-system messages when the history exceeds maxHistoryMessages.
func (a *Agent) trimHistory(history []Message) []Message {
	if len(history) <= maxHistoryMessages {
		return history
	}
	// Keep the system message (index 0) plus the last (maxHistoryMessages-1)
	// messages. Build a new slice to avoid aliasing the original backing array.
	tail := history[len(history)-(maxHistoryMessages-1):]
	out := make([]Message, 0, maxHistoryMessages)
	out = append(out, history[0])
	out = append(out, tail...)
	return out
}

// convertMessages translates our internal Message slice to the go-openai format.
func (a *Agent) convertMessages(history []Message) []openai.ChatCompletionMessage {
	out := make([]openai.ChatCompletionMessage, 0, len(history))
	for _, m := range history {
		msg := openai.ChatCompletionMessage{
			Role:    m.Role,
			Content: m.Content,
		}
		if m.ToolCallID != "" {
			msg.ToolCallID = m.ToolCallID
		}
		for _, tc := range m.ToolCalls {
			raw, _ := json.Marshal(tc.Args)
			msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{
				ID:   tc.ID,
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      tc.Name,
					Arguments: string(raw),
				},
			})
		}
		out = append(out, msg)
	}
	return out
}

// openAITools converts the agent's Tool slice into openai.Tool definitions.
func (a *Agent) openAITools() []openai.Tool {
	out := make([]openai.Tool, 0, len(a.tools))
	for _, t := range a.tools {
		params := t.Parameters()
		raw, _ := json.Marshal(params)
		out = append(out, openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        t.Name(),
				Description: t.Description(),
				Parameters:  json.RawMessage(raw),
			},
		})
	}
	return out
}
