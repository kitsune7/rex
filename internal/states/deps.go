package states

// Deps bundles every dependency needed by the state handlers. A single Deps
// value is shared across all handlers in a conversation loop.
type Deps struct {
	Listener    Listener
	Transcriber Transcriber
	Speaker     Speaker
	Agent       Agent
	Audio       AudioPlayer
	Timers      TimerController
	Scheduler   ReminderScheduler
	Settings    *Settings

	// Shared conversation state — mutated by handlers during a session.
	History  []Message
	ThreadID string
}

// ResetConversation clears the conversation history and thread ID,
// preparing for a fresh interaction.
func (d *Deps) ResetConversation() {
	d.History = nil
	d.ThreadID = ""
}

// IsInConversation reports whether a conversation is currently active
// (i.e. there is at least one message in the history).
func (d *Deps) IsInConversation() bool {
	return len(d.History) > 0
}
