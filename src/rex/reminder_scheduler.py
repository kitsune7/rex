"""
Reminder scheduler for Rex voice assistant.

Runs a background thread that wakes at the exact time reminders are due,
rather than polling. Uses event-based notification to recalculate wake
times when reminders are created, updated, or deleted.
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import sounddevice as sd
import soundfile as sf

if TYPE_CHECKING:
    from agent.tools.reminder import Reminder, ReminderManager
    from core.events import EventBus
    from rex.settings import ReminderSettings


@dataclass
class ReminderDelivery:
    """Information about a reminder being delivered."""

    reminder: "Reminder"


class ReminderScheduler:
    """
    Background scheduler that wakes precisely when reminders are due.

    Instead of polling at fixed intervals, the scheduler calculates how
    long to sleep until the next reminder is due. It subscribes to
    ReminderScheduleChanged events to recalculate when the schedule changes.
    """

    def __init__(
        self,
        on_reminder_due: Callable[[ReminderDelivery], None] | None = None,
        reminder_manager: "ReminderManager | None" = None,
        reminder_settings: "ReminderSettings | None" = None,
        event_bus: "EventBus | None" = None,
    ):
        """
        Initialize the reminder scheduler.

        Args:
            on_reminder_due: Callback when a reminder is due
            reminder_manager: ReminderManager instance (uses default if not provided)
            reminder_settings: ReminderSettings instance (uses default if not provided)
            event_bus: EventBus for schedule change notifications
        """
        # Import here to avoid circular imports and allow optional DI
        if reminder_manager is None:
            from agent.tools.reminder import get_reminder_manager

            reminder_manager = get_reminder_manager()

        self._reminder_manager = reminder_manager
        self._reminder_settings = reminder_settings
        self._event_bus = event_bus
        self._on_reminder_due = on_reminder_due

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._pending_delivery: ReminderDelivery | None = None
        self._delivery_lock = threading.Lock()
        self._delivery_event = threading.Event()

        # Load ding sound
        sound_path = Path("sounds/ding.mp3")
        if sound_path.exists():
            self._sound_data, self._sample_rate = sf.read(sound_path)
        else:
            self._sound_data = None
            self._sample_rate = None

    def _get_retry_minutes(self) -> int:
        """Get retry minutes from settings or default."""
        if self._reminder_settings:
            return self._reminder_settings.retry_minutes
        # Fallback to loading settings if not provided
        from rex.settings import get_settings

        return get_settings().reminders.retry_minutes

    def start(self):
        """Start the background scheduler thread."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._wake_event.clear()

        # Subscribe to schedule change events
        if self._event_bus:
            from core.events import ReminderScheduleChanged

            self._event_bus.subscribe(ReminderScheduleChanged, self._on_schedule_changed)

        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background scheduler thread."""
        self._running = False
        self._stop_event.set()
        self._wake_event.set()  # Wake up any waiting
        self._delivery_event.set()

        # Unsubscribe from events
        if self._event_bus:
            from core.events import ReminderScheduleChanged

            self._event_bus.unsubscribe(ReminderScheduleChanged, self._on_schedule_changed)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _on_schedule_changed(self, event) -> None:
        """Handler for ReminderScheduleChanged events - wakes the scheduler."""
        self._wake_event.set()

    def _calculate_next_wake_time(self) -> datetime | None:
        """
        Calculate when to next wake up.

        Returns:
            The time to wake up (rounded down to :00 seconds), or None if no pending reminders.
        """
        next_time = self._reminder_manager.get_next_pending_time()

        if next_time is None:
            return None

        # Round down to the start of the minute
        return next_time.replace(second=0, microsecond=0)

    def _scheduler_loop(self):
        """Main scheduler loop that wakes at precise reminder times."""
        # Check immediately on startup for any already-due reminders
        self._check_due_reminders()

        while self._running and not self._stop_event.is_set():
            next_wake = self._calculate_next_wake_time()

            if next_wake is not None:
                # Sleep until the start of the due minute
                sleep_seconds = (next_wake - datetime.now()).total_seconds()
                if sleep_seconds <= 0:
                    # Already due - check immediately
                    self._check_due_reminders()
                    continue

            # Wait until timeout or nudged by schedule change
            self._wake_event.clear()

            if self._stop_event.is_set():
                break

            # Check for due reminders (whether woken by timeout or event)
            self._check_due_reminders()

    def _check_due_reminders(self):
        """Check for due reminders and trigger delivery."""
        # Don't trigger new delivery if one is already pending
        if self.has_pending_delivery():
            return

        # Check for new due reminders
        due_reminders = self._reminder_manager.get_due_reminders()

        if due_reminders:
            # Trigger delivery for the first due reminder
            self._trigger_delivery(due_reminders[0])

    def _trigger_delivery(self, reminder: "Reminder"):
        """Trigger delivery of a reminder."""
        with self._delivery_lock:
            self._pending_delivery = ReminderDelivery(reminder=reminder)
            self._delivery_event.set()

        if self._on_reminder_due:
            self._on_reminder_due(self._pending_delivery)

    def get_pending_delivery(self) -> ReminderDelivery | None:
        """Get the current pending delivery, if any."""
        with self._delivery_lock:
            return self._pending_delivery

    def wait_for_delivery(self, timeout: float | None = None) -> ReminderDelivery | None:
        """
        Wait for a reminder delivery event.

        Args:
            timeout: Maximum time to wait (None for indefinite)

        Returns:
            ReminderDelivery if one is pending, None if timeout
        """
        if self._delivery_event.wait(timeout=timeout):
            with self._delivery_lock:
                delivery = self._pending_delivery
                return delivery
        return None

    def clear_pending_delivery(self):
        """Clear the pending delivery after it's been handled."""
        with self._delivery_lock:
            self._pending_delivery = None
            self._delivery_event.clear()

    def mark_delivered(self, reminder_id: int):
        """Mark a reminder as successfully delivered (user acknowledged)."""
        self._reminder_manager.clear_reminder(reminder_id)
        self.clear_pending_delivery()

    def schedule_retry(self, reminder_id: int):
        """
        Schedule a reminder for retry after the configured interval.

        This updates the reminder's due_datetime rather than tracking retries
        separately, so the reminder persists across restarts.
        """
        retry_minutes = self._get_retry_minutes()
        retry_time = datetime.now() + timedelta(minutes=retry_minutes)

        # Update the reminder's due time - this triggers ReminderScheduleChanged
        self._reminder_manager.update_reminder(reminder_id, due_datetime=retry_time)
        self.clear_pending_delivery()

    def snooze_reminder(self, reminder_id: int, snooze_minutes: int):
        """
        Snooze a reminder for a specified number of minutes.

        Args:
            reminder_id: ID of the reminder to snooze
            snooze_minutes: Number of minutes to snooze
        """
        new_time = datetime.now() + timedelta(minutes=snooze_minutes)
        self._reminder_manager.snooze_reminder(reminder_id, new_time)
        self.clear_pending_delivery()

    def play_ding(self):
        """Play the ding sound to alert the user."""
        if self._sound_data is not None:
            try:
                # Stop any ongoing audio first to avoid conflicts
                sd.stop()
                sd.play(self._sound_data, self._sample_rate)
                # Wait for playback using sleep instead of sd.wait() to avoid threading issues
                duration = len(self._sound_data) / self._sample_rate
                time.sleep(duration + 0.1)  # Small buffer for safety
            except Exception as e:
                print(f"Warning: Could not play ding sound: {e}")

    def has_pending_delivery(self) -> bool:
        """Check if there's a pending delivery."""
        with self._delivery_lock:
            return self._pending_delivery is not None
