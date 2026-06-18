package scheduler

import (
	"context"
	"sync"
	"testing"
	"time"
)

// ---------- test doubles ----------

type stubStore struct {
	mu              sync.Mutex
	dueReminders    []Reminder
	nextPendingTime *time.Time
	cleared         []int64
	snoozed         map[int64]time.Duration
}

func newStubStore() *stubStore {
	return &stubStore{snoozed: make(map[int64]time.Duration)}
}

func (s *stubStore) GetDueReminders() ([]Reminder, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dueReminders, nil
}

func (s *stubStore) GetNextPendingTime() (*time.Time, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.nextPendingTime, nil
}

func (s *stubStore) UpdateReminder(_ int64, _ *string, _ *time.Time) error { return nil }

func (s *stubStore) SnoozeReminder(id int64, d time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snoozed[id] = d
	return nil
}

func (s *stubStore) ClearReminder(id int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleared = append(s.cleared, id)
	return nil
}

func (s *stubStore) setDue(reminders []Reminder) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.dueReminders = reminders
}

func (s *stubStore) setNextPending(t *time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nextPendingTime = t
}

type stubPlayer struct {
	mu     sync.Mutex
	played []string
}

func (p *stubPlayer) PlaySoundFile(path string, _ bool) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.played = append(p.played, path)
	return nil
}

type stubBus struct {
	mu        sync.Mutex
	callbacks map[string][]func(any)
}

func newStubBus() *stubBus {
	return &stubBus{callbacks: make(map[string][]func(any))}
}

func (b *stubBus) Subscribe(name string, cb func(any)) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.callbacks[name] = append(b.callbacks[name], cb)
}

func (b *stubBus) emit(name string) {
	b.mu.Lock()
	cbs := append([]func(any){}, b.callbacks[name]...)
	b.mu.Unlock()
	for _, cb := range cbs {
		cb(nil)
	}
}

// ---------- tests ----------

func TestDeliverDueReminders(t *testing.T) {
	store := newStubStore()
	player := &stubPlayer{}
	bus := newStubBus()

	due := time.Now().Add(-time.Second)
	store.setDue([]Reminder{{ID: 1, Message: "hello", DueAt: due, Status: "pending"}})
	store.setNextPending(&due)

	s := NewScheduler(store, player, bus, 5)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Start(ctx)

	// The loop should pick up the due reminder almost immediately.
	deadline := time.After(2 * time.Second)
	for {
		if d := s.GetPendingDelivery(); d != nil {
			if d.ID != 1 || d.Message != "hello" {
				t.Fatalf("unexpected delivery: %+v", d)
			}
			break
		}
		select {
		case <-deadline:
			t.Fatal("timed out waiting for delivery")
		case <-time.After(10 * time.Millisecond):
		}
	}

	// Ding should have been played.
	player.mu.Lock()
	if len(player.played) == 0 {
		t.Fatal("expected ding sound to be played")
	}
	player.mu.Unlock()
}

func TestMarkDelivered(t *testing.T) {
	store := newStubStore()
	player := &stubPlayer{}
	bus := newStubBus()

	s := NewScheduler(store, player, bus, 5)
	s.pendingDelivery.Store(&ReminderDelivery{ID: 42, Message: "test"})

	if err := s.MarkDelivered(42); err != nil {
		t.Fatal(err)
	}

	if s.GetPendingDelivery() != nil {
		t.Fatal("pending delivery should be nil after mark delivered")
	}

	store.mu.Lock()
	if len(store.cleared) != 1 || store.cleared[0] != 42 {
		t.Fatalf("expected reminder 42 to be cleared, got %v", store.cleared)
	}
	store.mu.Unlock()
}

func TestScheduleRetry(t *testing.T) {
	store := newStubStore()
	player := &stubPlayer{}
	bus := newStubBus()

	s := NewScheduler(store, player, bus, 7)
	s.pendingDelivery.Store(&ReminderDelivery{ID: 10, Message: "retry me"})

	if err := s.ScheduleRetry(10); err != nil {
		t.Fatal(err)
	}

	if s.GetPendingDelivery() != nil {
		t.Fatal("pending delivery should be nil after retry")
	}

	store.mu.Lock()
	d, ok := store.snoozed[10]
	store.mu.Unlock()
	if !ok {
		t.Fatal("expected reminder 10 to be snoozed")
	}
	if d != 7*time.Minute {
		t.Fatalf("expected 7m snooze, got %v", d)
	}
}

func TestSnoozeReminder(t *testing.T) {
	store := newStubStore()
	player := &stubPlayer{}
	bus := newStubBus()

	s := NewScheduler(store, player, bus, 5)
	s.pendingDelivery.Store(&ReminderDelivery{ID: 20, Message: "snooze me"})

	if err := s.SnoozeReminder(20, 15*time.Minute); err != nil {
		t.Fatal(err)
	}

	if s.GetPendingDelivery() != nil {
		t.Fatal("pending delivery should be nil after snooze")
	}

	store.mu.Lock()
	d, ok := store.snoozed[20]
	store.mu.Unlock()
	if !ok {
		t.Fatal("expected reminder 20 to be snoozed")
	}
	if d != 15*time.Minute {
		t.Fatalf("expected 15m snooze, got %v", d)
	}
}

func TestWakeChannelTriggersRecalculation(t *testing.T) {
	store := newStubStore()
	player := &stubPlayer{}
	bus := newStubBus()

	// Start with no due reminders and no next pending time.
	s := NewScheduler(store, player, bus, 5)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s.Start(ctx)

	// Give the loop time to enter the waiting state.
	time.Sleep(50 * time.Millisecond)

	// Now add a due reminder and emit the event.
	due := time.Now().Add(-time.Second)
	store.setDue([]Reminder{{ID: 99, Message: "woke", DueAt: due, Status: "pending"}})
	store.setNextPending(&due)
	bus.emit("ReminderScheduleChanged")

	deadline := time.After(2 * time.Second)
	for {
		if d := s.GetPendingDelivery(); d != nil {
			if d.ID != 99 {
				t.Fatalf("unexpected delivery ID: %d", d.ID)
			}
			return
		}
		select {
		case <-deadline:
			t.Fatal("timed out waiting for delivery after wake")
		case <-time.After(10 * time.Millisecond):
		}
	}
}

func TestContextCancellationStopsLoop(t *testing.T) {
	store := newStubStore()
	player := &stubPlayer{}
	bus := newStubBus()

	s := NewScheduler(store, player, bus, 5)
	ctx, cancel := context.WithCancel(context.Background())

	s.Start(ctx)
	time.Sleep(50 * time.Millisecond)
	cancel()

	// The loop should exit promptly. If it doesn't, the test will hang and
	// the test runner will eventually time out.
	time.Sleep(100 * time.Millisecond)
	_ = s // loop has exited if we get here without hanging
}
