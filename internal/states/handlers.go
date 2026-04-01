package states

// CreateAllHandlers builds and returns all six conversation state handlers
// wired to the given shared dependencies.
func CreateAllHandlers(deps *Deps) []StateHandler {
	return []StateHandler{
		NewWaitingHandler(deps),
		NewListeningHandler(deps),
		NewProcessingHandler(deps),
		NewSpeakingHandler(deps),
		NewConfirmingHandler(deps),
		NewReminderHandler(deps),
	}
}
