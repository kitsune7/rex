package tools

import (
	"path/filepath"
	"testing"
	"time"
)

func newTestReminderManager(t *testing.T) (*ReminderManager, *mockEventEmitter) {
	t.Helper()
	events := &mockEventEmitter{}
	dbPath := filepath.Join(t.TempDir(), "test_reminders.db")
	rm, err := NewReminderManager(dbPath, events)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { rm.Close() })
	return rm, events
}

func TestReminderManager_Create(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	due := time.Now().Add(1 * time.Hour).Truncate(time.Second)
	id, err := rm.CreateReminder("Test reminder", due)
	if err != nil {
		t.Fatal(err)
	}
	if id <= 0 {
		t.Errorf("expected positive ID, got %d", id)
	}

	r, err := rm.GetReminder(id)
	if err != nil {
		t.Fatal(err)
	}
	if r == nil {
		t.Fatal("expected reminder, got nil")
	}
	if r.Message != "Test reminder" {
		t.Errorf("expected message 'Test reminder', got %q", r.Message)
	}
	if r.Status != ReminderPending {
		t.Errorf("expected status pending, got %q", r.Status)
	}
}

func TestReminderManager_GetNotFound(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	r, err := rm.GetReminder(9999)
	if err != nil {
		t.Fatal(err)
	}
	if r != nil {
		t.Errorf("expected nil, got %+v", r)
	}
}

func TestReminderManager_ListEmpty(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	reminders, err := rm.ListReminders()
	if err != nil {
		t.Fatal(err)
	}
	if len(reminders) != 0 {
		t.Errorf("expected 0 reminders, got %d", len(reminders))
	}
}

func TestReminderManager_List(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	due1 := time.Now().Add(1 * time.Hour).Truncate(time.Second)
	due2 := time.Now().Add(2 * time.Hour).Truncate(time.Second)
	rm.CreateReminder("First", due1)
	rm.CreateReminder("Second", due2)

	reminders, err := rm.ListReminders()
	if err != nil {
		t.Fatal(err)
	}
	if len(reminders) != 2 {
		t.Fatalf("expected 2 reminders, got %d", len(reminders))
	}
	if reminders[0].Message != "First" {
		t.Errorf("expected first reminder 'First', got %q", reminders[0].Message)
	}
	if reminders[1].Message != "Second" {
		t.Errorf("expected second reminder 'Second', got %q", reminders[1].Message)
	}
}

func TestReminderManager_ListByStatus(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	due := time.Now().Add(1 * time.Hour).Truncate(time.Second)
	id1, _ := rm.CreateReminder("Pending", due)
	id2, _ := rm.CreateReminder("Cleared", due)
	rm.ClearReminder(id2)

	pending, err := rm.ListRemindersByStatus(ReminderPending)
	if err != nil {
		t.Fatal(err)
	}
	if len(pending) != 1 || pending[0].ID != id1 {
		t.Errorf("expected 1 pending reminder with ID %d, got %+v", id1, pending)
	}

	cleared, err := rm.ListRemindersByStatus(ReminderCleared)
	if err != nil {
		t.Fatal(err)
	}
	if len(cleared) != 1 || cleared[0].ID != id2 {
		t.Errorf("expected 1 cleared reminder with ID %d, got %+v", id2, cleared)
	}
}

func TestReminderManager_GetDueReminders(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	past := time.Now().Add(-1 * time.Hour).Truncate(time.Second)
	future := time.Now().Add(1 * time.Hour).Truncate(time.Second)

	id1, _ := rm.CreateReminder("Past due", past)
	rm.CreateReminder("Not yet due", future)

	due, err := rm.GetDueReminders()
	if err != nil {
		t.Fatal(err)
	}
	if len(due) != 1 {
		t.Fatalf("expected 1 due reminder, got %d", len(due))
	}
	if due[0].ID != id1 {
		t.Errorf("expected due reminder ID %d, got %d", id1, due[0].ID)
	}
}

func TestReminderManager_UpdateMessage(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	due := time.Now().Add(1 * time.Hour).Truncate(time.Second)
	id, _ := rm.CreateReminder("Original", due)

	msg := "Updated"
	err := rm.UpdateReminder(id, &msg, nil)
	if err != nil {
		t.Fatal(err)
	}

	r, _ := rm.GetReminder(id)
	if r.Message != "Updated" {
		t.Errorf("expected message 'Updated', got %q", r.Message)
	}
}

func TestReminderManager_UpdateDatetime(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	due := time.Now().Add(1 * time.Hour).Truncate(time.Second)
	newDue := time.Now().Add(2 * time.Hour).Truncate(time.Second)
	id, _ := rm.CreateReminder("Test", due)

	err := rm.UpdateReminder(id, nil, &newDue)
	if err != nil {
		t.Fatal(err)
	}

	r, _ := rm.GetReminder(id)
	if !r.DueDatetime.Equal(newDue) {
		t.Errorf("expected due time %v, got %v", newDue, r.DueDatetime)
	}
}

func TestReminderManager_UpdateNotFound(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	msg := "Test"
	err := rm.UpdateReminder(9999, &msg, nil)
	if err == nil {
		t.Error("expected error for non-existent reminder")
	}
}

func TestReminderManager_Delete(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	due := time.Now().Add(1 * time.Hour).Truncate(time.Second)
	id, _ := rm.CreateReminder("Test", due)

	err := rm.DeleteReminder(id)
	if err != nil {
		t.Fatal(err)
	}

	r, _ := rm.GetReminder(id)
	if r != nil {
		t.Error("expected nil after delete")
	}
}

func TestReminderManager_DeleteNotFound(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	err := rm.DeleteReminder(9999)
	if err == nil {
		t.Error("expected error for non-existent reminder")
	}
}

func TestReminderManager_Clear(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	due := time.Now().Add(1 * time.Hour).Truncate(time.Second)
	id, _ := rm.CreateReminder("Test", due)

	err := rm.ClearReminder(id)
	if err != nil {
		t.Fatal(err)
	}

	r, _ := rm.GetReminder(id)
	if r.Status != ReminderCleared {
		t.Errorf("expected status cleared, got %q", r.Status)
	}
}

func TestReminderManager_Snooze(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	due := time.Now().Add(1 * time.Hour).Truncate(time.Second)
	id, _ := rm.CreateReminder("Test", due)

	err := rm.SnoozeReminder(id, 1*time.Hour)
	if err != nil {
		t.Fatal(err)
	}

	r, _ := rm.GetReminder(id)
	if r.Status != ReminderPending {
		t.Errorf("expected status pending after snooze, got %q", r.Status)
	}
	// Due time should be original + 1 hour
	expectedDue := due.Add(1 * time.Hour)
	if !r.DueDatetime.Equal(expectedDue) {
		t.Errorf("expected due time %v, got %v", expectedDue, r.DueDatetime)
	}
}

func TestReminderManager_GetNextPendingTimeNone(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	next, err := rm.GetNextPendingTime()
	if err != nil {
		t.Fatal(err)
	}
	if next != nil {
		t.Errorf("expected nil, got %v", next)
	}
}

func TestReminderManager_GetNextPendingTime(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	due1 := time.Now().Add(2 * time.Hour).Truncate(time.Second)
	due2 := time.Now().Add(1 * time.Hour).Truncate(time.Second) // earlier
	due3 := time.Now().Add(3 * time.Hour).Truncate(time.Second)

	rm.CreateReminder("Later", due1)
	rm.CreateReminder("Earliest", due2)
	rm.CreateReminder("Latest", due3)

	next, err := rm.GetNextPendingTime()
	if err != nil {
		t.Fatal(err)
	}
	if next == nil {
		t.Fatal("expected a time, got nil")
	}
	if !next.Equal(due2) {
		t.Errorf("expected %v, got %v", due2, *next)
	}
}

func TestReminderManager_GetNextPendingTimeIgnoresCleared(t *testing.T) {
	rm, _ := newTestReminderManager(t)

	due1 := time.Now().Add(1 * time.Hour).Truncate(time.Second)
	due2 := time.Now().Add(2 * time.Hour).Truncate(time.Second)

	id1, _ := rm.CreateReminder("Will be cleared", due1)
	rm.CreateReminder("Still pending", due2)
	rm.ClearReminder(id1)

	next, err := rm.GetNextPendingTime()
	if err != nil {
		t.Fatal(err)
	}
	if next == nil {
		t.Fatal("expected a time, got nil")
	}
	if !next.Equal(due2) {
		t.Errorf("expected %v, got %v", due2, *next)
	}
}

// Event emission tests

func TestReminderManager_CreateEmitsEvent(t *testing.T) {
	rm, events := newTestReminderManager(t)

	due := time.Now().Add(1 * time.Hour)
	rm.CreateReminder("Test", due)

	if len(events.events) != 1 {
		t.Errorf("expected 1 event, got %d", len(events.events))
	}
	if events.events[0].Type != "ReminderScheduleChanged" {
		t.Errorf("expected ReminderScheduleChanged event, got %q", events.events[0].Type)
	}
}

func TestReminderManager_UpdateDueDateEmitsEvent(t *testing.T) {
	rm, events := newTestReminderManager(t)

	due := time.Now().Add(1 * time.Hour)
	newDue := time.Now().Add(2 * time.Hour)
	id, _ := rm.CreateReminder("Test", due)
	events.events = nil // clear

	rm.UpdateReminder(id, nil, &newDue)

	if len(events.events) != 1 {
		t.Errorf("expected 1 event, got %d", len(events.events))
	}
}

func TestReminderManager_UpdateMessageOnlyNoEvent(t *testing.T) {
	rm, events := newTestReminderManager(t)

	due := time.Now().Add(1 * time.Hour)
	id, _ := rm.CreateReminder("Test", due)
	events.events = nil

	msg := "Updated"
	rm.UpdateReminder(id, &msg, nil)

	if len(events.events) != 0 {
		t.Errorf("expected 0 events for message-only update, got %d", len(events.events))
	}
}

func TestReminderManager_DeleteEmitsEvent(t *testing.T) {
	rm, events := newTestReminderManager(t)

	due := time.Now().Add(1 * time.Hour)
	id, _ := rm.CreateReminder("Test", due)
	events.events = nil

	rm.DeleteReminder(id)

	if len(events.events) != 1 {
		t.Errorf("expected 1 event, got %d", len(events.events))
	}
}

func TestReminderManager_SnoozeEmitsEvent(t *testing.T) {
	rm, events := newTestReminderManager(t)

	due := time.Now().Add(1 * time.Hour)
	id, _ := rm.CreateReminder("Test", due)
	events.events = nil

	rm.SnoozeReminder(id, 1*time.Hour)

	if len(events.events) != 1 {
		t.Errorf("expected 1 event, got %d", len(events.events))
	}
}

// ParseDatetime tests

func TestParseDatetime_ExplicitDatetime(t *testing.T) {
	result, err := ParseDatetime("2025-12-25 14:30")
	if err != nil {
		t.Fatal(err)
	}
	if result.Month() != 12 || result.Day() != 25 || result.Hour() != 14 || result.Minute() != 30 {
		t.Errorf("unexpected result: %v", result)
	}
}

func TestParseDatetime_TimeOnly(t *testing.T) {
	result, err := ParseDatetime("3pm")
	if err != nil {
		t.Fatal(err)
	}
	if result.Hour() != 15 {
		t.Errorf("expected hour 15, got %d", result.Hour())
	}
}

func TestParseDatetime_TimeWithMinutes(t *testing.T) {
	result, err := ParseDatetime("3:30 pm")
	if err != nil {
		t.Fatal(err)
	}
	if result.Hour() != 15 || result.Minute() != 30 {
		t.Errorf("expected 15:30, got %d:%d", result.Hour(), result.Minute())
	}
}

func TestParseDatetime_Tomorrow(t *testing.T) {
	result, err := ParseDatetime("tomorrow at 9am")
	if err != nil {
		t.Fatal(err)
	}
	tomorrow := time.Now().AddDate(0, 0, 1)
	if result.Day() != tomorrow.Day() {
		t.Errorf("expected day %d, got %d", tomorrow.Day(), result.Day())
	}
	if result.Hour() != 9 {
		t.Errorf("expected hour 9, got %d", result.Hour())
	}
}

func TestParseDatetime_DateWithTime(t *testing.T) {
	result, err := ParseDatetime("December 25th at 10am")
	if err != nil {
		t.Fatal(err)
	}
	if result.Month() != 12 || result.Day() != 25 || result.Hour() != 10 {
		t.Errorf("unexpected result: %v", result)
	}
}

func TestParseDatetime_Noon(t *testing.T) {
	result, err := ParseDatetime("noon")
	if err != nil {
		t.Fatal(err)
	}
	if result.Hour() != 12 {
		t.Errorf("expected hour 12, got %d", result.Hour())
	}
}

func TestParseDatetime_InvalidEmpty(t *testing.T) {
	_, err := ParseDatetime("")
	if err == nil {
		t.Error("expected error for empty string")
	}
}

func TestParseDatetime_NextTuesday(t *testing.T) {
	result, err := ParseDatetime("next Tuesday at noon")
	if err != nil {
		t.Fatal(err)
	}
	if result.Weekday() != time.Tuesday {
		t.Errorf("expected Tuesday, got %v", result.Weekday())
	}
	if result.Hour() != 12 {
		t.Errorf("expected hour 12, got %d", result.Hour())
	}
	if !result.After(time.Now()) {
		t.Error("expected future date")
	}
}

func TestParseDatetime_InTwoHours(t *testing.T) {
	before := time.Now()
	result, err := ParseDatetime("in 2 hours")
	if err != nil {
		t.Fatal(err)
	}
	expected := before.Add(2 * time.Hour)
	diff := result.Sub(expected)
	if diff < -5*time.Second || diff > 5*time.Second {
		t.Errorf("expected ~%v, got %v", expected, result)
	}
}
