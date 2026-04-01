package tools

// TimerSetTool implements the Tool interface for setting timers.
type TimerSetTool struct {
	Manager *TimerManager
}

func (t *TimerSetTool) Name() string        { return "set_timer" }
func (t *TimerSetTool) Description() string  { return "Set a timer with a duration and optional name." }
func (t *TimerSetTool) RequiresConfirmation() bool { return false }

func (t *TimerSetTool) Parameters() map[string]any {
	return map[string]any{
		"duration": map[string]any{
			"type":        "string",
			"description": "Duration, e.g. '5 minutes', '30 seconds', '1 hour 30 minutes'",
			"required":    true,
		},
		"name": map[string]any{
			"type":        "string",
			"description": "Optional timer name (default: 'timer')",
			"required":    false,
		},
	}
}

func (t *TimerSetTool) Execute(args map[string]any) (string, error) {
	durationStr, _ := args["duration"].(string)
	name, _ := args["name"].(string)
	if name == "" {
		name = "timer"
	}

	dur, err := ParseDuration(durationStr)
	if err != nil {
		return "Could not understand duration '" + durationStr + "'. " +
			"Try formats like '5 minutes', '30 seconds', or '1 hour 30 minutes'.", nil
	}

	return t.Manager.SetTimer(dur, name)
}

// TimerCheckTool implements the Tool interface for checking timer status.
type TimerCheckTool struct {
	Manager *TimerManager
}

func (t *TimerCheckTool) Name() string        { return "check_timers" }
func (t *TimerCheckTool) Description() string  { return "Check the status of all active timers." }
func (t *TimerCheckTool) RequiresConfirmation() bool { return false }

func (t *TimerCheckTool) Parameters() map[string]any {
	return map[string]any{}
}

func (t *TimerCheckTool) Execute(args map[string]any) (string, error) {
	return t.Manager.CheckTimers(), nil
}

// TimerStopTool implements the Tool interface for stopping timers.
type TimerStopTool struct {
	Manager *TimerManager
}

func (t *TimerStopTool) Name() string        { return "stop_timer" }
func (t *TimerStopTool) Description() string  { return "Stop a ringing alarm or cancel a timer." }
func (t *TimerStopTool) RequiresConfirmation() bool { return false }

func (t *TimerStopTool) Parameters() map[string]any {
	return map[string]any{
		"name": map[string]any{
			"type":        "string",
			"description": "Timer name to stop. If omitted, stops the currently ringing alarm.",
			"required":    false,
		},
	}
}

func (t *TimerStopTool) Execute(args map[string]any) (string, error) {
	name, _ := args["name"].(string)
	return t.Manager.StopTimer(name)
}
