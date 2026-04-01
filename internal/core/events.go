package core

import (
	"log"
	"sync"
	"time"
)

// Event is the interface that all events must satisfy.
type Event interface {
	// EventName returns a string identifier for the event type.
	EventName() string
	// Timestamp returns when the event was created.
	Timestamp() time.Time
}

// BaseEvent provides a default Timestamp implementation. Embed it in concrete
// event types so they only need to supply EventName.
type BaseEvent struct {
	CreatedAt time.Time
}

// Timestamp returns the time the event was created.
func (e BaseEvent) Timestamp() time.Time {
	return e.CreatedAt
}

// NewBaseEvent returns a BaseEvent stamped with the current time.
func NewBaseEvent() BaseEvent {
	return BaseEvent{CreatedAt: time.Now()}
}

// ReminderScheduleChanged is emitted when reminders are created, updated, or deleted.
type ReminderScheduleChanged struct {
	BaseEvent
}

// EventName returns "ReminderScheduleChanged".
func (ReminderScheduleChanged) EventName() string { return "ReminderScheduleChanged" }

// TimerFired is emitted when a timer alarm goes off.
type TimerFired struct {
	BaseEvent
	Name string
}

// EventName returns "TimerFired".
func (TimerFired) EventName() string { return "TimerFired" }

// TimerStopped is emitted when a timer alarm is stopped.
type TimerStopped struct {
	BaseEvent
	Name string
}

// EventName returns "TimerStopped".
func (TimerStopped) EventName() string { return "TimerStopped" }

// EventCallback is the signature for event subscriber functions.
type EventCallback func(Event)

// EventBus provides a simple synchronous publish/subscribe mechanism.
// It is safe for concurrent use.
type EventBus struct {
	mu          sync.RWMutex
	subscribers map[string][]EventCallback
}

// NewEventBus creates an empty EventBus.
func NewEventBus() *EventBus {
	return &EventBus{
		subscribers: make(map[string][]EventCallback),
	}
}

// Subscribe registers a callback for the given event name.
func (b *EventBus) Subscribe(eventName string, callback EventCallback) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.subscribers[eventName] = append(b.subscribers[eventName], callback)
}

// Emit sends an event to all subscribers registered for that event name.
// Handlers are called synchronously in subscription order. Handler panics
// are recovered and logged.
func (b *EventBus) Emit(event Event) {
	b.mu.RLock()
	callbacks := make([]EventCallback, len(b.subscribers[event.EventName()]))
	copy(callbacks, b.subscribers[event.EventName()])
	b.mu.RUnlock()

	for _, cb := range callbacks {
		func() {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("panic in event handler for %s: %v", event.EventName(), r)
				}
			}()
			cb(event)
		}()
	}
}

// Clear removes all subscribers.
func (b *EventBus) Clear() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.subscribers = make(map[string][]EventCallback)
}
