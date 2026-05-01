package reminders

import (
	"path/filepath"
	"sync"
	"testing"
	"time"

	"rex/internal/core"
)

// openTestStore creates a fresh Store backed by a temp file. The event bus is
// optional — pass nil when you don't need to observe emissions.
func openTestStore(t *testing.T, bus *core.EventBus) *Store {
	t.Helper()
	path := filepath.Join(t.TempDir(), "reminders.db")
	s, err := Open(path, bus)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestCreateAndList(t *testing.T) {
	s := openTestStore(t, nil)

	due1 := time.Now().Add(2 * time.Hour).Truncate(time.Second)
	due2 := time.Now().Add(1 * time.Hour).Truncate(time.Second)
	if _, err := s.CreateReminder("later", due1); err != nil {
		t.Fatal(err)
	}
	if _, err := s.CreateReminder("sooner", due2); err != nil {
		t.Fatal(err)
	}

	got, err := s.ListReminders()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 reminders, got %d", len(got))
	}
	if got[0].Message != "sooner" || got[1].Message != "later" {
		t.Errorf("ordering wrong: %q then %q", got[0].Message, got[1].Message)
	}
}

func TestGetDueReminders(t *testing.T) {
	s := openTestStore(t, nil)

	past := time.Now().Add(-1 * time.Hour)
	future := time.Now().Add(1 * time.Hour)
	pastID, err := s.CreateReminder("past due", past)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.CreateReminder("not yet", future); err != nil {
		t.Fatal(err)
	}

	due, err := s.GetDueReminders()
	if err != nil {
		t.Fatal(err)
	}
	if len(due) != 1 || due[0].ID != pastID {
		t.Errorf("expected only past reminder due; got %+v", due)
	}
}

func TestUpdateReminderMessageOnly(t *testing.T) {
	s := openTestStore(t, nil)
	id, _ := s.CreateReminder("original", time.Now().Add(time.Hour))

	newMsg := "updated"
	if err := s.UpdateReminder(id, &newMsg, nil); err != nil {
		t.Fatal(err)
	}
	list, _ := s.ListReminders()
	if list[0].Message != "updated" {
		t.Errorf("message not updated: %q", list[0].Message)
	}
}

func TestUpdateReminderDueTime(t *testing.T) {
	s := openTestStore(t, nil)
	id, _ := s.CreateReminder("x", time.Now().Add(time.Hour))

	newDue := time.Now().Add(2 * time.Hour).Truncate(time.Second)
	if err := s.UpdateReminder(id, nil, &newDue); err != nil {
		t.Fatal(err)
	}
	list, _ := s.ListReminders()
	if !list[0].DueAt.Equal(newDue) {
		t.Errorf("due time not updated: %v vs %v", list[0].DueAt, newDue)
	}
}

func TestUpdateReminderNotFound(t *testing.T) {
	s := openTestStore(t, nil)
	msg := "hi"
	if err := s.UpdateReminder(9999, &msg, nil); err == nil {
		t.Fatal("expected error for missing id")
	}
}

func TestDeleteReminder(t *testing.T) {
	s := openTestStore(t, nil)
	id, _ := s.CreateReminder("x", time.Now().Add(time.Hour))
	if err := s.DeleteReminder(id); err != nil {
		t.Fatal(err)
	}
	list, _ := s.ListReminders()
	if len(list) != 0 {
		t.Errorf("expected empty list, got %v", list)
	}
	if err := s.DeleteReminder(id); err == nil {
		t.Error("deleting missing id should error")
	}
}

func TestClearReminder(t *testing.T) {
	s := openTestStore(t, nil)
	id, _ := s.CreateReminder("x", time.Now().Add(time.Hour))
	if err := s.ClearReminder(id); err != nil {
		t.Fatal(err)
	}
	list, _ := s.ListReminders()
	if list[0].Status != core.ReminderCleared {
		t.Errorf("status = %v, want cleared", list[0].Status)
	}
}

func TestSnoozeReminder(t *testing.T) {
	s := openTestStore(t, nil)
	due := time.Now().Add(time.Hour).Truncate(time.Second)
	id, _ := s.CreateReminder("x", due)
	_ = s.MarkDelivered(id)

	if err := s.SnoozeReminder(id, 30*time.Minute); err != nil {
		t.Fatal(err)
	}
	list, _ := s.ListReminders()
	want := due.Add(30 * time.Minute)
	if !list[0].DueAt.Equal(want) {
		t.Errorf("due after snooze = %v, want %v", list[0].DueAt, want)
	}
	if list[0].Status != core.ReminderPending {
		t.Errorf("snooze should reset to pending, got %v", list[0].Status)
	}
}

func TestGetNextPendingTime(t *testing.T) {
	s := openTestStore(t, nil)

	next, err := s.GetNextPendingTime()
	if err != nil {
		t.Fatal(err)
	}
	if next != nil {
		t.Errorf("empty store should return nil, got %v", *next)
	}

	due2 := time.Now().Add(2 * time.Hour).Truncate(time.Second)
	due1 := time.Now().Add(1 * time.Hour).Truncate(time.Second)
	_, _ = s.CreateReminder("later", due2)
	_, _ = s.CreateReminder("sooner", due1)

	next, err = s.GetNextPendingTime()
	if err != nil {
		t.Fatal(err)
	}
	if next == nil || !next.Equal(due1) {
		t.Errorf("next pending = %v, want %v", next, due1)
	}
}

func TestGetNextPendingTimeIgnoresCleared(t *testing.T) {
	s := openTestStore(t, nil)
	due1 := time.Now().Add(time.Hour).Truncate(time.Second)
	due2 := time.Now().Add(2 * time.Hour).Truncate(time.Second)
	id1, _ := s.CreateReminder("will be cleared", due1)
	_, _ = s.CreateReminder("pending", due2)
	_ = s.ClearReminder(id1)

	next, _ := s.GetNextPendingTime()
	if next == nil || !next.Equal(due2) {
		t.Errorf("next = %v, want %v", next, due2)
	}
}

func TestEventsEmitted(t *testing.T) {
	bus := core.NewEventBus()
	var (
		mu    sync.Mutex
		count int
	)
	bus.Subscribe("ReminderScheduleChanged", func(e core.Event) {
		mu.Lock()
		count++
		mu.Unlock()
	})
	s := openTestStore(t, bus)

	id, _ := s.CreateReminder("x", time.Now().Add(time.Hour))
	if count != 1 {
		t.Errorf("create should emit 1 event, got %d", count)
	}

	// Update message only → no emission.
	msg := "y"
	_ = s.UpdateReminder(id, &msg, nil)
	if count != 1 {
		t.Errorf("message-only update should NOT emit, got count=%d", count)
	}

	// Update due time → emission.
	newDue := time.Now().Add(2 * time.Hour)
	_ = s.UpdateReminder(id, nil, &newDue)
	if count != 2 {
		t.Errorf("due-time update should emit, got count=%d", count)
	}

	_ = s.SnoozeReminder(id, time.Minute)
	if count != 3 {
		t.Errorf("snooze should emit, got count=%d", count)
	}

	_ = s.DeleteReminder(id)
	if count != 4 {
		t.Errorf("delete should emit, got count=%d", count)
	}
}

func TestPersistenceAcrossReopens(t *testing.T) {
	path := filepath.Join(t.TempDir(), "persist.db")
	s, err := Open(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	due := time.Now().Add(time.Hour).Truncate(time.Second)
	if _, err := s.CreateReminder("persistent", due); err != nil {
		t.Fatal(err)
	}
	_ = s.Close()

	s2, err := Open(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()
	list, err := s2.ListReminders()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 1 || list[0].Message != "persistent" || !list[0].DueAt.Equal(due) {
		t.Errorf("persistence broken: got %+v", list)
	}
}

func TestConcurrentWrites(t *testing.T) {
	s := openTestStore(t, nil)

	const workers = 8
	const perWorker = 25
	var wg sync.WaitGroup
	errs := make(chan error, workers*perWorker)

	for w := range workers {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := range perWorker {
				_, err := s.CreateReminder("concurrent", time.Now().Add(time.Duration(w*perWorker+i)*time.Second))
				if err != nil {
					errs <- err
					return
				}
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("concurrent write failed: %v", err)
	}

	list, err := s.ListReminders()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != workers*perWorker {
		t.Errorf("wanted %d rows, got %d", workers*perWorker, len(list))
	}
}
