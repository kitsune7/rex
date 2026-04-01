package tools

import (
	"strings"
	"testing"
	"time"
)

// mockAudioPlayer implements AudioPlayer for testing.
type mockAudioPlayer struct {
	looping bool
}

func (m *mockAudioPlayer) StartLoop(soundPath string) { m.looping = true }
func (m *mockAudioPlayer) StopLoop()                  { m.looping = false }

// mockEventEmitter implements EventEmitter for testing.
type mockEventEmitter struct {
	events []Event
}

func (m *mockEventEmitter) Emit(event Event) {
	m.events = append(m.events, event)
}

func newTestTimerManager() (*TimerManager, *mockAudioPlayer, *mockEventEmitter) {
	audio := &mockAudioPlayer{}
	events := &mockEventEmitter{}
	tm := NewTimerManager(audio, events, "sounds/fun-timer.mp3")
	return tm, audio, events
}

func TestTimerManager_SetTimer(t *testing.T) {
	tm, _, _ := newTestTimerManager()
	defer tm.Cleanup()

	result, err := tm.SetTimer(60*time.Second, "test")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result, "test") {
		t.Errorf("result should contain timer name, got %q", result)
	}
	if !strings.Contains(result, "1 minute") {
		t.Errorf("result should contain formatted duration, got %q", result)
	}
}

func TestTimerManager_SetTimerCreatesEntry(t *testing.T) {
	tm, _, _ := newTestTimerManager()
	defer tm.Cleanup()

	tm.SetTimer(300*time.Second, "pizza")

	tm.mu.Lock()
	entry, ok := tm.timers["pizza"]
	tm.mu.Unlock()

	if !ok {
		t.Fatal("timer 'pizza' not found")
	}
	if entry.Name != "pizza" {
		t.Errorf("expected name 'pizza', got %q", entry.Name)
	}
	if entry.DurationSeconds != 300.0 {
		t.Errorf("expected duration 300, got %f", entry.DurationSeconds)
	}
	if entry.State != TimerPending {
		t.Errorf("expected state Pending, got %d", entry.State)
	}
}

func TestTimerManager_SetTimerReplacesExisting(t *testing.T) {
	tm, _, _ := newTestTimerManager()
	defer tm.Cleanup()

	tm.SetTimer(60*time.Second, "timer")
	tm.SetTimer(120*time.Second, "timer")

	tm.mu.Lock()
	if len(tm.timers) != 1 {
		t.Errorf("expected 1 timer, got %d", len(tm.timers))
	}
	if tm.timers["timer"].DurationSeconds != 120.0 {
		t.Errorf("expected duration 120, got %f", tm.timers["timer"].DurationSeconds)
	}
	tm.mu.Unlock()
}

func TestTimerManager_CheckTimersEmpty(t *testing.T) {
	tm, _, _ := newTestTimerManager()
	defer tm.Cleanup()

	result := tm.CheckTimers()
	if result != "No active timers." {
		t.Errorf("expected 'No active timers.', got %q", result)
	}
}

func TestTimerManager_CheckTimersShowsRemaining(t *testing.T) {
	tm, _, _ := newTestTimerManager()
	defer tm.Cleanup()

	tm.SetTimer(300*time.Second, "test")
	result := tm.CheckTimers()

	if !strings.Contains(result, "test") {
		t.Errorf("expected result to contain 'test', got %q", result)
	}
	if !strings.Contains(result, "remaining") {
		t.Errorf("expected result to contain 'remaining', got %q", result)
	}
}

func TestTimerManager_StopTimerCancelsPending(t *testing.T) {
	tm, _, _ := newTestTimerManager()
	defer tm.Cleanup()

	tm.SetTimer(300*time.Second, "test")
	result, err := tm.StopTimer("test")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result, "Cancelled") {
		t.Errorf("expected 'Cancelled' in result, got %q", result)
	}

	tm.mu.Lock()
	_, ok := tm.timers["test"]
	tm.mu.Unlock()
	if ok {
		t.Error("timer should have been removed")
	}
}

func TestTimerManager_StopTimerNotFound(t *testing.T) {
	tm, _, _ := newTestTimerManager()
	defer tm.Cleanup()

	result, _ := tm.StopTimer("nonexistent")
	if !strings.Contains(strings.ToLower(result), "not found") && !strings.Contains(result, "No timer") {
		t.Errorf("expected 'not found' or 'No timer', got %q", result)
	}
}

func TestTimerManager_StopAnyRingingReturnsFalse(t *testing.T) {
	tm, _, _ := newTestTimerManager()
	defer tm.Cleanup()

	tm.SetTimer(300*time.Second, "test")
	if tm.StopAnyRinging() {
		t.Error("expected StopAnyRinging to return false when no timer is ringing")
	}
}

func TestTimerManager_MuteUnmute(t *testing.T) {
	tm, _, _ := newTestTimerManager()
	defer tm.Cleanup()

	if tm.muted.Load() {
		t.Error("expected muted=false initially")
	}

	tm.Mute()
	if !tm.muted.Load() {
		t.Error("expected muted=true after Mute()")
	}

	tm.Unmute()
	if tm.muted.Load() {
		t.Error("expected muted=false after Unmute()")
	}
}

func TestTimerManager_TimerFires(t *testing.T) {
	tm, audio, events := newTestTimerManager()
	defer tm.Cleanup()

	tm.SetTimer(50*time.Millisecond, "quick")

	// Wait for the timer to fire
	time.Sleep(200 * time.Millisecond)

	tm.mu.Lock()
	entry, ok := tm.timers["quick"]
	ringing := tm.currentRinging
	tm.mu.Unlock()

	if !ok {
		t.Fatal("timer 'quick' not found after firing")
	}
	if entry.State != TimerRinging {
		t.Errorf("expected state Ringing, got %d", entry.State)
	}
	if ringing != "quick" {
		t.Errorf("expected currentRinging='quick', got %q", ringing)
	}
	if !audio.looping {
		t.Error("expected audio to be looping after timer fires")
	}

	// Check event was emitted
	found := false
	for _, e := range events.events {
		if e.Type == "TimerFired" {
			found = true
		}
	}
	if !found {
		t.Error("expected TimerFired event to be emitted")
	}

	// Stop the ringing timer
	result, _ := tm.StopTimer("")
	if !strings.Contains(result, "Stopped alarm") {
		t.Errorf("expected 'Stopped alarm', got %q", result)
	}
}
