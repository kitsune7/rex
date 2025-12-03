"""
Event bus for decoupled communication between Rex components.

Events allow components to communicate without direct dependencies.
For example, the reminder scheduler can emit ReminderScheduleChanged events
without knowing about the CLI or conversation state.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable


# Base event class
@dataclass
class Event:
    """Base class for all events."""

    timestamp: datetime = field(default_factory=datetime.now)


# Reminder events
@dataclass
class ReminderScheduleChanged(Event):
    """Fired when reminders are created, updated, or deleted.

    Used to wake the scheduler so it can recalculate when to next check.
    """

    pass


# Timer events
@dataclass
class TimerFired(Event):
    """Fired when a timer alarm goes off."""

    timer_name: str = ""


@dataclass
class TimerStopped(Event):
    """Fired when a timer alarm is stopped."""

    timer_name: str = ""


class EventBus:
    """
    Simple synchronous event bus for component communication.

    Components can subscribe to event types and will be notified
    when events of that type are emitted.

    Thread-safe: can be used from multiple threads.
    """

    def __init__(self):
        self._subscribers: dict[type, list[Callable[[Event], None]]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, event_type: type[Event], handler: Callable[[Event], None]) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: The event class to subscribe to
            handler: Callable that receives the event when emitted
        """
        with self._lock:
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: type[Event], handler: Callable[[Event], None]) -> None:
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type: The event class to unsubscribe from
            handler: The handler to remove
        """
        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)

    def emit(self, event: Event) -> None:
        """
        Emit an event to all subscribers.

        Handlers are called synchronously in subscription order.

        Args:
            event: The event instance to emit
        """
        with self._lock:
            handlers = list(self._subscribers[type(event)])

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Log but don't propagate handler errors
                print(f"Error in event handler for {type(event).__name__}: {e}")

    def clear(self) -> None:
        """Remove all subscribers."""
        with self._lock:
            self._subscribers.clear()
