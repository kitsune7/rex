package states

import (
	"context"
	"log/slog"
)

// ProcessingHandler implements the PROCESSING state.
// It sends the user's transcription to the LLM agent and routes to the
// appropriate next state based on the response.
type ProcessingHandler struct {
	deps          *Deps
	transcription string
}

// NewProcessingHandler creates a ProcessingHandler.
func NewProcessingHandler(deps *Deps) *ProcessingHandler {
	return &ProcessingHandler{deps: deps}
}

func (h *ProcessingHandler) State() int { return StateProcessing }

// Enter stores the transcription provided by the listening state.
func (h *ProcessingHandler) Enter(data map[string]any) {
	h.transcription = ""
	if data != nil {
		if v, ok := data["transcription"].(string); ok {
			h.transcription = v
		}
	}
}

// Process invokes the agent with the user's query and determines the
// next state from the result.
func (h *ProcessingHandler) Process(ctx context.Context) StateResult {
	if h.transcription == "" {
		return StateResult{NextState: StateWaitingForWakeWord}
	}

	slog.Info("processing query", "text", h.transcription)
	h.deps.Audio.StartThinkingTone()

	result, err := h.deps.Agent.Run(ctx, h.transcription, h.deps.History, h.deps.ThreadID)

	h.deps.Audio.StopThinkingTone()

	if err != nil {
		slog.Error("agent error", "err", err)
		return StateResult{
			NextState: StateSpeaking,
			Data: map[string]any{
				"response":               "Sorry, I encountered an error processing your request.",
				"force_end_conversation": true,
			},
		}
	}

	// Update shared conversation state.
	h.deps.History = result.History
	h.deps.ThreadID = result.ThreadID

	if result.PendingConfirmation != nil {
		return StateResult{
			NextState: StateAwaitingConfirmation,
			Data:      map[string]any{"pending": result.PendingConfirmation},
		}
	}

	return StateResult{
		NextState: StateSpeaking,
		Data:      map[string]any{"response": result.Response},
	}
}

// Exit clears the stored transcription.
func (h *ProcessingHandler) Exit() {
	h.transcription = ""
}
