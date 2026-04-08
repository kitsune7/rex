package tools

import "time"

// TimeTool implements the Tool interface for getting the current time.
type TimeTool struct{}

func (t *TimeTool) Name() string        { return "get_current_time" }
func (t *TimeTool) Description() string  { return "Get the current time." }
func (t *TimeTool) RequiresConfirmation() bool { return false }

func (t *TimeTool) Parameters() map[string]any {
	return map[string]any{}
}

func (t *TimeTool) Execute(args map[string]any) (string, error) {
	return time.Now().Format("3:04 PM"), nil
}
