"""
Event bus for decoupled communication between Rex components.

Events allow components to communicate without direct dependencies.
For example, the reminder scheduler can emit ReminderDue events
without knowing about the CLI or conversation state.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np


# Base event class
@dataclass
class Event:
    """Base class for all events."""

    timestamp: datetime = field(default_factory=datetime.now)


# Wake word events
@dataclass
class WakeWordDetected(Event):
    """Fired when wake word is detected."""

    confidence: float = 0.0


@dataclass
class WakeWordInterrupt(Event):
    """Fired when wake word interrupts ongoing speech."""

    pass


# Audio capture events
@dataclass
class SpeechCaptureStarted(Event):
    """Fired when speech capture begins."""

    pass


@dataclass
class SpeechCaptureComplete(Event):
    """Fired when speech capture ends with audio data."""

    audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    duration_seconds: float = 0.0


@dataclass
class SpeechCaptureTimeout(Event):
    """Fired when speech capture times out without speech."""

    pass


# Transcription events
@dataclass
class TranscriptionComplete(Event):
    """Fired when speech-to-text completes."""

    text: str = ""
    is_wake_word_stripped: bool = False


# Agent events
@dataclass
class AgentProcessingStarted(Event):
    """Fired when agent starts processing a query."""

    query: str = ""


@dataclass
class AgentResponseReady(Event):
    """Fired when agent has a response ready."""

    response: str = ""
    needs_confirmation: bool = False


@dataclass
class ConfirmationRequired(Event):
    """Fired when a tool call requires user confirmation."""

    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    prompt: str = ""


@dataclass
class ConfirmationReceived(Event):
    """Fired when user responds to confirmation request."""

    confirmed: bool = False


# TTS events
@dataclass
class SpeechStarted(Event):
    """Fired when TTS playback begins."""

    text: str = ""


@dataclass
class SpeechComplete(Event):
    """Fired when TTS playback ends normally."""

    pass


@dataclass
class SpeechInterrupted(Event):
    """Fired when TTS playback is interrupted."""

    pass


# Reminder events
@dataclass
class ReminderDue(Event):
    """Fired when a reminder is due for delivery."""

    reminder_id: int = 0
    message: str = ""


@dataclass
class ReminderDelivered(Event):
    """Fired when user acknowledges a reminder."""

    reminder_id: int = 0


@dataclass
class ReminderSnoozed(Event):
    """Fired when user snoozes a reminder."""

    reminder_id: int = 0
    snooze_minutes: int = 0


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


# Special command events
@dataclass
class StopCommand(Event):
    """Fired when user says stop/nevermind."""

    pass


# System events
@dataclass
class ShutdownRequested(Event):
    """Fired when system shutdown is requested."""

    pass


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
