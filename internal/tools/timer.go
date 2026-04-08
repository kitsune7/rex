package tools

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// TimerState represents the current state of a timer.
type TimerState int

const (
	// TimerPending means the timer is counting down.
	TimerPending TimerState = iota
	// TimerRinging means the timer has fired and the alarm is playing.
	TimerRinging
)

// TimerEntry holds the state of a single timer.
type TimerEntry struct {
	Name            string
	DurationSeconds float64
	StartTime       time.Time
	State           TimerState
	cancel          func() bool // stops the AfterFunc timer
}

// TimerManager manages multiple named timers with alarm sound playback.
type TimerManager struct {
	mu             sync.Mutex
	timers         map[string]*TimerEntry
	currentRinging string
	muted          atomic.Bool
	audioPlayer    AudioPlayer
	eventBus       EventEmitter
	soundPath      string
}

// NewTimerManager creates a new TimerManager.
func NewTimerManager(audioPlayer AudioPlayer, eventBus EventEmitter, soundPath string) *TimerManager {
	return &TimerManager{
		timers:      make(map[string]*TimerEntry),
		audioPlayer: audioPlayer,
		eventBus:    eventBus,
		soundPath:   soundPath,
	}
}

func (tm *TimerManager) emitEvent(event Event) {
	if tm.eventBus != nil {
		tm.eventBus.Emit(event)
	}
}

func (tm *TimerManager) startAlarm() {
	if tm.audioPlayer == nil || tm.muted.Load() {
		return
	}
	tm.audioPlayer.StartLoop(tm.soundPath)
}

func (tm *TimerManager) stopAlarm() {
	if tm.audioPlayer != nil {
		tm.audioPlayer.StopLoop()
	}
}

func (tm *TimerManager) timerCallback(name string) {
	tm.mu.Lock()
	entry, ok := tm.timers[name]
	if !ok || entry.State != TimerPending {
		tm.mu.Unlock()
		return
	}
	entry.State = TimerRinging
	tm.currentRinging = name
	tm.mu.Unlock()

	tm.emitEvent(Event{Type: "TimerFired", Data: map[string]any{"timer_name": name}})
	tm.startAlarm()
}

// SetTimer creates or replaces a named timer. Returns a confirmation message.
func (tm *TimerManager) SetTimer(duration time.Duration, name string) (string, error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	// Cancel existing timer with the same name
	if old, ok := tm.timers[name]; ok {
		if old.cancel != nil {
			old.cancel()
		}
		if old.State == TimerRinging {
			tm.stopAlarm()
		}
	}

	secs := duration.Seconds()
	timer := time.AfterFunc(duration, func() {
		tm.timerCallback(name)
	})

	tm.timers[name] = &TimerEntry{
		Name:            name,
		DurationSeconds: secs,
		StartTime:       time.Now(),
		State:           TimerPending,
		cancel:          timer.Stop,
	}

	return fmt.Sprintf("Timer '%s' set for %s", name, FormatDuration(duration)), nil
}

// CheckTimers returns a formatted status of all active timers.
func (tm *TimerManager) CheckTimers() string {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if len(tm.timers) == 0 {
		return "No active timers."
	}

	var lines []string
	for name, entry := range tm.timers {
		if entry.State == TimerRinging {
			lines = append(lines, fmt.Sprintf("'%s' is ringing!", name))
		} else {
			elapsed := time.Since(entry.StartTime).Seconds()
			remaining := entry.DurationSeconds - elapsed
			if remaining < 0 {
				remaining = 0
			}
			lines = append(lines, fmt.Sprintf("'%s': %s remaining", name, FormatDuration(time.Duration(remaining*float64(time.Second)))))
		}
	}
	return strings.Join(lines, "\n")
}

// StopTimer stops a specific timer or the currently ringing alarm.
func (tm *TimerManager) StopTimer(name string) (string, error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	// If no name given, stop the ringing alarm
	if name == "" {
		if tm.currentRinging != "" {
			name = tm.currentRinging
		} else {
			return "No timer is currently ringing.", nil
		}
	}

	entry, ok := tm.timers[name]
	if !ok {
		return fmt.Sprintf("No timer named '%s' found.", name), nil
	}

	if entry.State == TimerRinging {
		tm.stopAlarm()
		delete(tm.timers, name)
		tm.currentRinging = ""
		tm.emitEvent(Event{Type: "TimerStopped", Data: map[string]any{"timer_name": name}})
		return fmt.Sprintf("Stopped alarm for timer '%s'.", name), nil
	}

	// Pending
	if entry.cancel != nil {
		entry.cancel()
	}
	delete(tm.timers, name)
	return fmt.Sprintf("Cancelled timer '%s'.", name), nil
}

// StopAnyRinging stops any currently ringing alarm. Returns true if an alarm was stopped.
func (tm *TimerManager) StopAnyRinging() bool {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if tm.currentRinging == "" {
		return false
	}
	entry, ok := tm.timers[tm.currentRinging]
	if !ok || entry.State != TimerRinging {
		return false
	}

	tm.stopAlarm()
	name := tm.currentRinging
	delete(tm.timers, tm.currentRinging)
	tm.currentRinging = ""
	tm.emitEvent(Event{Type: "TimerStopped", Data: map[string]any{"timer_name": name}})
	return true
}

// Mute suppresses alarm sound. Use Unmute to re-enable.
func (tm *TimerManager) Mute() {
	tm.muted.Store(true)
	tm.stopAlarm()
}

// Unmute re-enables alarm sound. If a timer is currently ringing, restarts the alarm.
func (tm *TimerManager) Unmute() {
	tm.muted.Store(false)
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if tm.currentRinging != "" {
		if entry, ok := tm.timers[tm.currentRinging]; ok && entry.State == TimerRinging {
			tm.startAlarm()
		}
	}
}

// Cleanup stops all timers and alarm sounds.
func (tm *TimerManager) Cleanup() {
	tm.mu.Lock()
	for _, entry := range tm.timers {
		if entry.cancel != nil {
			entry.cancel()
		}
	}
	tm.timers = make(map[string]*TimerEntry)
	tm.mu.Unlock()
	tm.stopAlarm()
}
