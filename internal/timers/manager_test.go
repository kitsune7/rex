package timers

import (
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"rex/internal/core"
)

// fakePlayer records StartLoop / StopLoop calls without doing any audio I/O.
type fakePlayer struct {
	mu        sync.Mutex
	started   int
	stopped   int
	lastRate  int
	lastBytes int
}

func (f *fakePlayer) StartLoop(samples []float32, sampleRate int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.started++
	f.lastRate = sampleRate
	f.lastBytes = len(samples)
}

func (f *fakePlayer) StopLoop() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopped++
}

func (f *fakePlayer) counts() (int, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.started, f.stopped
}

// newTestManager returns a Manager wired to a fake player and an event bus.
// soundData is preloaded with a non-empty buffer so the alarm path actually
// calls StartLoop on the fake.
func newTestManager(t *testing.T) (*Manager, *fakePlayer, *core.EventBus) {
	t.Helper()
	bus := core.NewEventBus()
	p := &fakePlayer{}
	m := New(Options{Bus: bus, Player: p})
	m.soundData = []float32{0.1, 0.2, 0.3}
	m.soundRate = 44100
	t.Cleanup(m.Cleanup)
	return m, p, bus
}

func TestSetTimerReturnsConfirmation(t *testing.T) {
	m, _, _ := newTestManager(t)
	got := m.SetTimer("test", time.Minute)
	if !strings.Contains(got, "test") || !strings.Contains(got, "1 minute") {
		t.Errorf("SetTimer return = %q", got)
	}
}

func TestSetTimerReplacesExisting(t *testing.T) {
	m, _, _ := newTestManager(t)
	m.SetTimer("t", time.Minute)
	m.SetTimer("t", 2*time.Minute)

	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.timers) != 1 {
		t.Errorf("want 1 timer, have %d", len(m.timers))
	}
	if m.timers["t"].duration != 2*time.Minute {
		t.Errorf("duration = %v, want 2m", m.timers["t"].duration)
	}
}

func TestCheckTimersEmpty(t *testing.T) {
	m, _, _ := newTestManager(t)
	if got := m.CheckTimers(); got != "No active timers." {
		t.Errorf("CheckTimers empty = %q", got)
	}
}

func TestCheckTimersShowsRemaining(t *testing.T) {
	m, _, _ := newTestManager(t)
	m.SetTimer("t", 5*time.Minute)
	got := m.CheckTimers()
	if !strings.Contains(got, "'t'") || !strings.Contains(got, "remaining") {
		t.Errorf("CheckTimers = %q", got)
	}
}

func TestStopPendingTimerCancels(t *testing.T) {
	m, _, _ := newTestManager(t)
	m.SetTimer("t", 5*time.Minute)
	got := m.StopTimer("t")
	if !strings.Contains(got, "Cancelled") {
		t.Errorf("StopTimer = %q, want cancellation message", got)
	}
	m.mu.Lock()
	_, exists := m.timers["t"]
	m.mu.Unlock()
	if exists {
		t.Error("timer still present after cancellation")
	}
}

func TestStopTimerNotFound(t *testing.T) {
	m, _, _ := newTestManager(t)
	got := m.StopTimer("nothing")
	if !strings.Contains(got, "No timer named") {
		t.Errorf("StopTimer missing = %q", got)
	}
}

func TestStopAnyRingingFalseWhenNoneRinging(t *testing.T) {
	m, _, _ := newTestManager(t)
	m.SetTimer("t", time.Minute)
	if m.StopAnyRinging() {
		t.Error("StopAnyRinging should return false when nothing is ringing")
	}
}

func TestMuteAndUnmuteToggle(t *testing.T) {
	m, _, _ := newTestManager(t)
	if m.muted {
		t.Fatal("fresh manager should not be muted")
	}
	m.Mute()
	if !m.muted {
		t.Error("Mute did not set muted")
	}
	m.Unmute()
	if m.muted {
		t.Error("Unmute did not clear muted")
	}
}

// Integration-style test: schedule a short timer, verify it fires, starts
// the alarm loop, emits an event, and that StopAnyRinging silences it.
func TestTimerFiresAndCanBeStopped(t *testing.T) {
	m, player, bus := newTestManager(t)

	var fired atomic.Int32
	var stopped atomic.Int32
	bus.Subscribe("TimerFired", func(e core.Event) { fired.Add(1) })
	bus.Subscribe("TimerStopped", func(e core.Event) { stopped.Add(1) })

	m.SetTimer("blink", 30*time.Millisecond)

	waitFor(t, time.Second, func() bool { return fired.Load() == 1 })

	started, stoppedCalls := player.counts()
	if started != 1 {
		t.Errorf("StartLoop called %d times, want 1", started)
	}
	if stoppedCalls != 0 {
		t.Errorf("StopLoop called %d times before stop, want 0", stoppedCalls)
	}

	if !m.StopAnyRinging() {
		t.Fatal("StopAnyRinging returned false after alarm fired")
	}
	waitFor(t, time.Second, func() bool { return stopped.Load() == 1 })

	_, stoppedCalls = player.counts()
	if stoppedCalls < 1 {
		t.Errorf("StopLoop never called after stop; got %d", stoppedCalls)
	}
}

func TestMuteSilencesRingingAlarm(t *testing.T) {
	m, player, _ := newTestManager(t)
	m.SetTimer("blink", 20*time.Millisecond)
	waitFor(t, time.Second, func() bool {
		s, _ := player.counts()
		return s >= 1
	})
	m.Mute()
	_, stoppedCalls := player.counts()
	if stoppedCalls < 1 {
		t.Errorf("Mute did not call StopLoop: stopped=%d", stoppedCalls)
	}

	beforeStart, _ := player.counts()
	m.Unmute()
	afterStart, _ := player.counts()
	if afterStart <= beforeStart {
		t.Error("Unmute should restart alarm when a timer is ringing")
	}
}

func TestCleanupCancelsAllTimers(t *testing.T) {
	m, _, _ := newTestManager(t)
	m.SetTimer("a", 10*time.Minute)
	m.SetTimer("b", 20*time.Minute)
	m.Cleanup()
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.timers) != 0 {
		t.Errorf("timers remain after Cleanup: %d", len(m.timers))
	}
}

// waitFor polls cond every 5ms until it returns true or the deadline fires.
func waitFor(t *testing.T, within time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("condition not met within %v", within)
}
