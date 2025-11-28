"""
Reminder scheduler for Rex voice assistant.

Runs a background thread that checks for due reminders and triggers
proactive conversations with the user.
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import sounddevice as sd
import soundfile as sf

from agent.tools.reminder import (
    ReminderManager,
    ReminderStatus,
    Reminder,
    get_reminder_manager,
)
from .settings import get_settings


@dataclass
class ReminderDelivery:
    """Information about a reminder being delivered."""

    reminder: Reminder
    attempt: int = 1


class ReminderScheduler:
    """
    Background scheduler that checks for due reminders and triggers delivery.

    The scheduler runs a background thread that periodically checks for
    reminders that are due. When a reminder is due, it signals the main
    loop to handle delivery.
    """

    def __init__(
        self,
        on_reminder_due: Callable[[ReminderDelivery], None] | None = None,
        check_interval: float = 30.0,
    ):
        """
        Initialize the reminder scheduler.

        Args:
            on_reminder_due: Callback when a reminder is due
            check_interval: How often to check for due reminders (seconds)
        """
        self._reminder_manager = get_reminder_manager()
        self._on_reminder_due = on_reminder_due
        self._check_interval = check_interval

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pending_delivery: ReminderDelivery | None = None
        self._delivery_lock = threading.Lock()
        self._delivery_event = threading.Event()

        # Track reminders that failed delivery and their retry times
        self._retry_schedule: dict[int, datetime] = {}
        self._retry_lock = threading.Lock()

        # Load ding sound
        sound_path = Path("sounds/ding.mp3")
        if sound_path.exists():
            self._sound_data, self._sample_rate = sf.read(sound_path)
        else:
            self._sound_data = None
            self._sample_rate = None

    def start(self):
        """Start the background scheduler thread."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background scheduler thread."""
        self._running = False
        self._stop_event.set()
        self._delivery_event.set()  # Wake up any waiting
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _scheduler_loop(self):
        """Main scheduler loop that checks for due reminders."""
        while self._running and not self._stop_event.is_set():
            try:
                self._check_due_reminders()
            except Exception as e:
                print(f"Error in reminder scheduler: {e}")

            # Sleep with periodic wake-up checks
            for _ in range(int(self._check_interval)):
                if self._stop_event.is_set():
                    break
                time.sleep(1.0)

    def _check_due_reminders(self):
        """Check for due reminders and trigger delivery."""
        # Don't trigger new delivery if one is already pending
        if self.has_pending_delivery():
            return

        # First check retry schedule
        with self._retry_lock:
            now = datetime.now()
            retry_due = [rid for rid, retry_time in self._retry_schedule.items() if retry_time <= now]

        # Process retries
        for reminder_id in retry_due:
            reminder = self._reminder_manager.get_reminder(reminder_id)
            if reminder and reminder.status == ReminderStatus.PENDING:
                with self._retry_lock:
                    attempt = 2  # Retries are at least attempt 2
                    del self._retry_schedule[reminder_id]
                self._trigger_delivery(reminder, attempt)
                return  # Handle one at a time

        # Check for new due reminders
        due_reminders = self._reminder_manager.get_due_reminders()

        # Filter out reminders that are already scheduled for retry
        with self._retry_lock:
            due_reminders = [r for r in due_reminders if r.id not in self._retry_schedule]

        if due_reminders:
            # Trigger delivery for the first due reminder
            self._trigger_delivery(due_reminders[0], attempt=1)

    def _trigger_delivery(self, reminder: Reminder, attempt: int):
        """Trigger delivery of a reminder."""
        with self._delivery_lock:
            self._pending_delivery = ReminderDelivery(reminder=reminder, attempt=attempt)
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
        with self._retry_lock:
            self._retry_schedule.pop(reminder_id, None)
        self.clear_pending_delivery()

    def schedule_retry(self, reminder_id: int):
        """Schedule a reminder for retry after the configured interval."""
        settings = get_settings()
        retry_minutes = settings.reminders.retry_minutes
        retry_time = datetime.now() + timedelta(minutes=retry_minutes)

        with self._retry_lock:
            self._retry_schedule[reminder_id] = retry_time

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

        with self._retry_lock:
            self._retry_schedule.pop(reminder_id, None)

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
