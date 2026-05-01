// Package timers provides a TimerManager that schedules named timers and
// plays a looping alarm sound when they fire. It emits TimerFired /
// TimerStopped events on the core EventBus.
package timers

import (
	"fmt"
	"log"
	"sync"
	"time"

	"rex/internal/audio"
	"rex/internal/core"
	"rex/internal/tools"
)

// LoopPlayer is the narrow subset of audio.Manager the timer uses for its
// alarm loop. Tests inject a fake so they don't need PortAudio.
type LoopPlayer interface {
	StartLoop(samples []float32, sampleRate int)
	StopLoop()
}

// TimerState describes the lifecycle of a single timer.
type TimerState int

const (
	TimerPending TimerState = iota
	TimerRinging
)

func (s TimerState) String() string {
	switch s {
	case TimerPending:
		return "pending"
	case TimerRinging:
		return "ringing"
	}
	return fmt.Sprintf("TimerState(%d)", int(s))
}

type timer struct {
	name      string
	duration  time.Duration
	startTime time.Time
	state     TimerState
	fireTimer *time.Timer
}

// Manager schedules named timers and plays an alarm loop when they fire.
// It satisfies core.TimerController.
type Manager struct {
	mu             sync.Mutex
	timers         map[string]*timer
	currentRinging string
	muted          bool

	bus         *core.EventBus
	player      LoopPlayer
	soundData   []float32
	soundRate   int
}

// Options configures a Manager.
type Options struct {
	Bus       *core.EventBus
	Player    LoopPlayer
	SoundPath string // if empty, the manager works with no alarm audio.
}

// New constructs a Manager. If opts.SoundPath is set but cannot be loaded,
// the error is logged and the manager falls back to silent operation.
func New(opts Options) *Manager {
	m := &Manager{
		timers: make(map[string]*timer),
		bus:    opts.Bus,
		player: opts.Player,
	}
	if opts.SoundPath != "" {
		samples, rate, err := audio.LoadSoundFile(opts.SoundPath)
		if err != nil {
			log.Printf("timers: loading alarm sound %q: %v (timers will be silent)", opts.SoundPath, err)
		} else {
			m.soundData = samples
			m.soundRate = rate
		}
	}
	return m
}

// SetTimer schedules (or replaces) a named timer and returns a human-readable
// confirmation string.
func (m *Manager) SetTimer(name string, duration time.Duration) string {
	m.mu.Lock()

	if existing, ok := m.timers[name]; ok {
		if existing.fireTimer != nil {
			existing.fireTimer.Stop()
		}
		if existing.state == TimerRinging {
			m.stopAlarmLocked()
			m.currentRinging = ""
		}
		delete(m.timers, name)
	}

	t := &timer{
		name:      name,
		duration:  duration,
		startTime: time.Now(),
		state:     TimerPending,
	}
	t.fireTimer = time.AfterFunc(duration, func() { m.fire(name) })
	m.timers[name] = t
	m.mu.Unlock()

	return fmt.Sprintf("Timer '%s' set for %s", name, tools.FormatDuration(duration))
}

func (m *Manager) fire(name string) {
	m.mu.Lock()
	t, ok := m.timers[name]
	if !ok || t.state != TimerPending {
		m.mu.Unlock()
		return
	}
	t.state = TimerRinging
	m.currentRinging = name
	m.startAlarmLocked()
	m.mu.Unlock()

	if m.bus != nil {
		m.bus.Emit(core.TimerFired{BaseEvent: core.NewBaseEvent(), Name: name})
	}
}

// CheckTimers returns a status summary of all active timers.
func (m *Manager) CheckTimers() string {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.timers) == 0 {
		return "No active timers."
	}

	lines := make([]string, 0, len(m.timers))
	for name, t := range m.timers {
		if t.state == TimerRinging {
			lines = append(lines, fmt.Sprintf("'%s' is ringing!", name))
			continue
		}
		remaining := t.duration - time.Since(t.startTime)
		if remaining < 0 {
			remaining = 0
		}
		lines = append(lines, fmt.Sprintf("'%s': %s remaining", name, tools.FormatDuration(remaining)))
	}
	return joinLines(lines)
}

// StopTimer cancels a pending timer or silences a ringing alarm. Passing an
// empty name stops whichever alarm is currently ringing.
func (m *Manager) StopTimer(name string) string {
	m.mu.Lock()
	defer m.mu.Unlock()

	if name == "" {
		if m.currentRinging == "" {
			return "No timer is currently ringing."
		}
		name = m.currentRinging
	}

	t, ok := m.timers[name]
	if !ok {
		return fmt.Sprintf("No timer named '%s' found.", name)
	}

	if t.state == TimerRinging {
		m.stopAlarmLocked()
		delete(m.timers, name)
		m.currentRinging = ""
		m.emitStoppedLocked(name)
		return fmt.Sprintf("Stopped alarm for timer '%s'.", name)
	}

	if t.fireTimer != nil {
		t.fireTimer.Stop()
	}
	delete(m.timers, name)
	return fmt.Sprintf("Cancelled timer '%s'.", name)
}

// StopAnyRinging silences whichever alarm is currently ringing (if any) and
// returns true when it had work to do. Used by the state machine's "stop"
// phrase handler.
func (m *Manager) StopAnyRinging() bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.currentRinging == "" {
		return false
	}
	name := m.currentRinging
	t, ok := m.timers[name]
	if !ok || t.state != TimerRinging {
		m.currentRinging = ""
		return false
	}
	m.stopAlarmLocked()
	delete(m.timers, name)
	m.currentRinging = ""
	m.emitStoppedLocked(name)
	return true
}

// Mute pauses the alarm sound without cancelling the timer. Unmute resumes it
// if an alarm is still ringing.
func (m *Manager) Mute() {
	m.mu.Lock()
	m.muted = true
	m.stopAlarmLocked()
	m.mu.Unlock()
}

// Unmute re-enables the alarm sound if one is still ringing.
func (m *Manager) Unmute() {
	m.mu.Lock()
	m.muted = false
	if m.currentRinging != "" {
		if t, ok := m.timers[m.currentRinging]; ok && t.state == TimerRinging {
			m.startAlarmLocked()
		}
	}
	m.mu.Unlock()
}

// Cleanup cancels all outstanding timers and silences the alarm.
func (m *Manager) Cleanup() {
	m.mu.Lock()
	for _, t := range m.timers {
		if t.fireTimer != nil {
			t.fireTimer.Stop()
		}
	}
	m.timers = make(map[string]*timer)
	m.stopAlarmLocked()
	m.currentRinging = ""
	m.mu.Unlock()
}

// --- internal helpers (caller must hold m.mu) ---

func (m *Manager) startAlarmLocked() {
	if m.muted || m.player == nil || m.soundData == nil {
		return
	}
	m.player.StartLoop(m.soundData, m.soundRate)
}

func (m *Manager) stopAlarmLocked() {
	if m.player == nil {
		return
	}
	m.player.StopLoop()
}

func (m *Manager) emitStoppedLocked(name string) {
	if m.bus == nil {
		return
	}
	// Emit outside the lock would be safer if callbacks could re-enter us,
	// but EventBus callbacks are synchronous and we control who subscribes.
	m.bus.Emit(core.TimerStopped{BaseEvent: core.NewBaseEvent(), Name: name})
}

func joinLines(lines []string) string {
	out := ""
	for i, l := range lines {
		if i > 0 {
			out += "\n"
		}
		out += l
	}
	return out
}
