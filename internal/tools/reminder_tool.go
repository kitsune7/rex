package tools

import (
	"fmt"
	"strconv"
	"time"
)

// ReminderCreateTool implements the Tool interface for creating reminders.
type ReminderCreateTool struct {
	Manager *ReminderManager
}

func (t *ReminderCreateTool) Name() string        { return "create_reminder" }
func (t *ReminderCreateTool) Description() string  { return "Create a reminder. Requires user confirmation." }
func (t *ReminderCreateTool) RequiresConfirmation() bool { return true }

func (t *ReminderCreateTool) Parameters() map[string]any {
	return map[string]any{
		"message": map[string]any{
			"type":        "string",
			"description": "What to remind about",
			"required":    true,
		},
		"datetime_str": map[string]any{
			"type":        "string",
			"description": "When to remind, e.g. 'tomorrow at 3pm', 'next Tuesday at noon'",
			"required":    true,
		},
	}
}

func (t *ReminderCreateTool) Execute(args map[string]any) (string, error) {
	message, _ := args["message"].(string)
	datetimeStr, _ := args["datetime_str"].(string)

	dueAt, err := ParseDatetime(datetimeStr)
	if err != nil {
		return fmt.Sprintf("Could not understand the date/time '%s'. "+
			"Try formats like 'tomorrow at 3pm', 'next Tuesday at noon', or 'January 15th at 10am'.", datetimeStr), nil
	}

	if dueAt.Before(time.Now()) {
		return fmt.Sprintf("The specified time (%s) is in the past. Please specify a future date and time.",
			dueAt.Format("January 2 at 3:04 PM")), nil
	}

	id, err := t.Manager.CreateReminder(message, dueAt)
	if err != nil {
		return "", err
	}

	_ = id
	formatted := dueAt.Format("Monday, January 2 at 3:04 PM")
	return fmt.Sprintf("Reminder created: '%s' for %s.", message, formatted), nil
}

// ReminderListTool implements the Tool interface for listing reminders.
type ReminderListTool struct {
	Manager *ReminderManager
}

func (t *ReminderListTool) Name() string        { return "list_reminders" }
func (t *ReminderListTool) Description() string  { return "List all pending reminders." }
func (t *ReminderListTool) RequiresConfirmation() bool { return false }

func (t *ReminderListTool) Parameters() map[string]any {
	return map[string]any{}
}

func (t *ReminderListTool) Execute(args map[string]any) (string, error) {
	reminders, err := t.Manager.ListReminders()
	if err != nil {
		return "", err
	}

	if len(reminders) == 0 {
		return "You have no pending reminders.", nil
	}

	lines := []string{"Your pending reminders:"}
	for _, r := range reminders {
		formatted := r.DueDatetime.Format("Monday, January 2 at 3:04 PM")
		lines = append(lines, fmt.Sprintf("- ID %d: '%s' - %s", r.ID, r.Message, formatted))
	}
	return joinStrings(lines, "\n"), nil
}

// ReminderUpdateTool implements the Tool interface for updating reminders.
type ReminderUpdateTool struct {
	Manager *ReminderManager
}

func (t *ReminderUpdateTool) Name() string        { return "update_reminder" }
func (t *ReminderUpdateTool) Description() string  { return "Update a reminder's message and/or time." }
func (t *ReminderUpdateTool) RequiresConfirmation() bool { return false }

func (t *ReminderUpdateTool) Parameters() map[string]any {
	return map[string]any{
		"id": map[string]any{
			"type":        "number",
			"description": "Reminder ID from list_reminders",
			"required":    true,
		},
		"new_message": map[string]any{
			"type":        "string",
			"description": "New message (optional)",
			"required":    false,
		},
		"new_datetime_str": map[string]any{
			"type":        "string",
			"description": "New time (optional)",
			"required":    false,
		},
	}
}

func (t *ReminderUpdateTool) Execute(args map[string]any) (string, error) {
	id, err := toInt64(args["id"])
	if err != nil {
		return "Invalid reminder ID.", nil
	}

	// Check reminder exists
	reminder, err := t.Manager.GetReminder(id)
	if err != nil {
		return "", err
	}
	if reminder == nil {
		return fmt.Sprintf("No reminder found with ID %d. Use list_reminders to see your reminders.", id), nil
	}

	var msgPtr *string
	if newMsg, ok := args["new_message"].(string); ok && newMsg != "" {
		msgPtr = &newMsg
	}

	var duePtr *time.Time
	if newDtStr, ok := args["new_datetime_str"].(string); ok && newDtStr != "" {
		dt, err := ParseDatetime(newDtStr)
		if err != nil {
			return fmt.Sprintf("Could not understand the date/time '%s'. "+
				"Try formats like 'tomorrow at 3pm' or 'next Tuesday at noon'.", newDtStr), nil
		}
		if dt.Before(time.Now()) {
			return fmt.Sprintf("The specified time (%s) is in the past. Please specify a future date and time.",
				dt.Format("January 2 at 3:04 PM")), nil
		}
		duePtr = &dt
	}

	if err := t.Manager.UpdateReminder(id, msgPtr, duePtr); err != nil {
		return "", err
	}

	// Fetch updated reminder
	updated, err := t.Manager.GetReminder(id)
	if err != nil || updated == nil {
		return fmt.Sprintf("No reminder found with ID %d.", id), nil
	}

	formatted := updated.DueDatetime.Format("Monday, January 2 at 3:04 PM")
	return fmt.Sprintf("Reminder updated: '%s' for %s.", updated.Message, formatted), nil
}

// ReminderDeleteTool implements the Tool interface for deleting reminders.
type ReminderDeleteTool struct {
	Manager *ReminderManager
}

func (t *ReminderDeleteTool) Name() string        { return "delete_reminder" }
func (t *ReminderDeleteTool) Description() string  { return "Delete a reminder by ID." }
func (t *ReminderDeleteTool) RequiresConfirmation() bool { return false }

func (t *ReminderDeleteTool) Parameters() map[string]any {
	return map[string]any{
		"id": map[string]any{
			"type":        "number",
			"description": "Reminder ID to delete",
			"required":    true,
		},
	}
}

func (t *ReminderDeleteTool) Execute(args map[string]any) (string, error) {
	id, err := toInt64(args["id"])
	if err != nil {
		return "Invalid reminder ID.", nil
	}

	if err := t.Manager.DeleteReminder(id); err != nil {
		return fmt.Sprintf("No reminder found with ID %d. Use list_reminders to see your reminders.", id), nil
	}

	return fmt.Sprintf("Reminder %d has been deleted.", id), nil
}

// toInt64 converts a map value (float64 from JSON or int) to int64.
func toInt64(v any) (int64, error) {
	switch val := v.(type) {
	case float64:
		return int64(val), nil
	case int:
		return int64(val), nil
	case int64:
		return val, nil
	case string:
		return strconv.ParseInt(val, 10, 64)
	default:
		return 0, fmt.Errorf("cannot convert %T to int64", v)
	}
}
