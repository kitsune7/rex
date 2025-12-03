"""
Reminder tools for Rex voice assistant.

Provides functionality to create, list, update, and delete reminders with SQLite persistence.
"""

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from dateutil import parser as dateutil_parser
from langchain_core.tools import tool

if TYPE_CHECKING:
    from core.events import EventBus


class ReminderStatus(Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    CLEARED = "cleared"


@dataclass
class Reminder:
    id: int
    message: str
    due_datetime: datetime
    created_at: datetime
    status: ReminderStatus


class ReminderManager:
    """
    Manages reminders with SQLite persistence.

    This is no longer a singleton - instances should be created via
    the AppContext or create_reminder_manager() factory.
    """

    def __init__(self, db_path: str | Path = "data/reminders.db", event_bus: "EventBus | None" = None):
        """
        Initialize the reminder manager.

        Args:
            db_path: Path to the SQLite database file
            event_bus: Optional event bus for emitting schedule change events
        """
        self._db_lock = threading.Lock()
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_bus = event_bus
        self._init_db()

    def _emit_schedule_changed(self) -> None:
        """Emit a schedule changed event if event bus is configured."""
        if self._event_bus is not None:
            from core.events import ReminderScheduleChanged

            self._event_bus.emit(ReminderScheduleChanged())

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize the database schema."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reminders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message TEXT NOT NULL,
                        due_datetime TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending'
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    def _row_to_reminder(self, row: sqlite3.Row) -> Reminder:
        """Convert a database row to a Reminder object."""
        return Reminder(
            id=row["id"],
            message=row["message"],
            due_datetime=datetime.fromisoformat(row["due_datetime"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            status=ReminderStatus(row["status"]),
        )

    def create_reminder(self, message: str, due_datetime: datetime) -> Reminder:
        """
        Create a new reminder.

        Args:
            message: The reminder message
            due_datetime: When the reminder should trigger

        Returns:
            The created Reminder object
        """
        with self._db_lock:
            conn = self._get_connection()
            try:
                now = datetime.now()
                cursor = conn.execute(
                    """
                    INSERT INTO reminders (message, due_datetime, created_at, status)
                    VALUES (?, ?, ?, ?)
                    """,
                    (message, due_datetime.isoformat(), now.isoformat(), ReminderStatus.PENDING.value),
                )
                conn.commit()

                reminder = Reminder(
                    id=cursor.lastrowid,
                    message=message,
                    due_datetime=due_datetime,
                    created_at=now,
                    status=ReminderStatus.PENDING,
                )
            finally:
                conn.close()

        self._emit_schedule_changed()
        return reminder

    def get_reminder(self, reminder_id: int) -> Reminder | None:
        """Get a reminder by ID."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
                row = cursor.fetchone()
                return self._row_to_reminder(row) if row else None
            finally:
                conn.close()

    def list_reminders(self, status: ReminderStatus | None = None) -> list[Reminder]:
        """
        List all reminders, optionally filtered by status.

        Args:
            status: Optional status to filter by

        Returns:
            List of Reminder objects
        """
        with self._db_lock:
            conn = self._get_connection()
            try:
                if status:
                    cursor = conn.execute(
                        "SELECT * FROM reminders WHERE status = ? ORDER BY due_datetime",
                        (status.value,),
                    )
                else:
                    cursor = conn.execute("SELECT * FROM reminders ORDER BY due_datetime")
                return [self._row_to_reminder(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def get_due_reminders(self) -> list[Reminder]:
        """
        Get all pending reminders that are due (due_datetime <= now).

        Returns:
            List of due Reminder objects
        """
        with self._db_lock:
            conn = self._get_connection()
            try:
                now = datetime.now().isoformat()
                cursor = conn.execute(
                    """
                    SELECT * FROM reminders
                    WHERE status = ? AND due_datetime <= ?
                    ORDER BY due_datetime
                    """,
                    (ReminderStatus.PENDING.value, now),
                )
                return [self._row_to_reminder(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def get_next_pending_time(self) -> datetime | None:
        """
        Get the due time of the next pending reminder.

        Returns:
            The earliest due_datetime among pending reminders, or None if no pending reminders.
        """
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    """
                    SELECT MIN(due_datetime) as next_time
                    FROM reminders
                    WHERE status = ?
                    """,
                    (ReminderStatus.PENDING.value,),
                )
                row = cursor.fetchone()
                if row and row["next_time"]:
                    return datetime.fromisoformat(row["next_time"])
                return None
            finally:
                conn.close()

    def update_reminder(
        self,
        reminder_id: int,
        message: str | None = None,
        due_datetime: datetime | None = None,
        status: ReminderStatus | None = None,
    ) -> Reminder | None:
        """
        Update a reminder.

        Args:
            reminder_id: ID of the reminder to update
            message: New message (optional)
            due_datetime: New due datetime (optional)
            status: New status (optional)

        Returns:
            Updated Reminder object, or None if not found
        """
        with self._db_lock:
            conn = self._get_connection()
            try:
                # Build update query dynamically
                updates = []
                params = []

                if message is not None:
                    updates.append("message = ?")
                    params.append(message)
                if due_datetime is not None:
                    updates.append("due_datetime = ?")
                    params.append(due_datetime.isoformat())
                if status is not None:
                    updates.append("status = ?")
                    params.append(status.value)

                if not updates:
                    return self.get_reminder(reminder_id)

                params.append(reminder_id)
                query = f"UPDATE reminders SET {', '.join(updates)} WHERE id = ?"
                conn.execute(query, params)
                conn.commit()

                # Fetch and return updated reminder
                cursor = conn.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
                row = cursor.fetchone()
                result = self._row_to_reminder(row) if row else None
            finally:
                conn.close()

        # Emit schedule changed if due_datetime was updated
        if due_datetime is not None:
            self._emit_schedule_changed()

        return result

    def delete_reminder(self, reminder_id: int) -> bool:
        """
        Delete a reminder.

        Args:
            reminder_id: ID of the reminder to delete

        Returns:
            True if deleted, False if not found
        """
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
                conn.commit()
                deleted = cursor.rowcount > 0
            finally:
                conn.close()

        if deleted:
            self._emit_schedule_changed()

        return deleted

    def clear_reminder(self, reminder_id: int) -> Reminder | None:
        """
        Mark a reminder as cleared.

        Args:
            reminder_id: ID of the reminder to clear

        Returns:
            Updated Reminder object, or None if not found
        """
        return self.update_reminder(reminder_id, status=ReminderStatus.CLEARED)

    def snooze_reminder(self, reminder_id: int, new_due_datetime: datetime) -> Reminder | None:
        """
        Snooze a reminder to a new time.

        Args:
            reminder_id: ID of the reminder to snooze
            new_due_datetime: New due datetime

        Returns:
            Updated Reminder object, or None if not found
        """
        return self.update_reminder(reminder_id, due_datetime=new_due_datetime, status=ReminderStatus.PENDING)


def parse_datetime(datetime_str: str) -> datetime | None:
    """
    Parse a datetime string into a datetime object.

    Supports natural language formats like:
    - "tomorrow at 3pm"
    - "next Tuesday at noon"
    - "December 25th at 9am"
    - "2024-01-15 14:30"

    Args:
        datetime_str: A datetime string to parse

    Returns:
        Parsed datetime, or None if parsing fails
    """
    if not datetime_str or not datetime_str.strip():
        return None

    original = datetime_str.strip().lower()
    modified = original

    # Handle common relative terms that dateutil doesn't understand
    now = datetime.now()

    # Handle "tomorrow"
    if "tomorrow" in modified:
        tomorrow = now + timedelta(days=1)
        # Replace tomorrow with the actual date
        modified = modified.replace("tomorrow", tomorrow.strftime("%Y-%m-%d"))

    # Handle "today"
    if "today" in modified:
        modified = modified.replace("today", now.strftime("%Y-%m-%d"))

    # Handle "noon" -> "12:00 pm"
    modified = modified.replace(" noon", " 12:00 pm").replace("at noon", "at 12:00 pm")
    if modified == "noon":
        modified = "12:00 pm"

    # Handle "midnight" -> "00:00"
    modified = modified.replace(" midnight", " 00:00").replace("at midnight", "at 00:00")
    if modified == "midnight":
        modified = "00:00"

    try:
        # Use dateutil parser with fuzzy matching for natural language
        parsed = dateutil_parser.parse(modified, fuzzy=True)

        # If no time was specified and it parsed to midnight, that's probably not intentional
        # But we'll accept it since the user might mean "start of day"
        return parsed
    except (ValueError, TypeError):
        return None


# Module-level instance for tool access
# This gets set by the AppContext during initialization
_reminder_manager: ReminderManager | None = None


def set_reminder_manager(manager: ReminderManager) -> None:
    """Set the module-level reminder manager instance for tool access."""
    global _reminder_manager
    _reminder_manager = manager


def get_reminder_manager() -> ReminderManager:
    """
    Get the reminder manager instance.

    Returns the configured instance, or creates a default one if not set.
    """
    global _reminder_manager
    if _reminder_manager is None:
        _reminder_manager = ReminderManager()
    return _reminder_manager


# Set of tool names that require confirmation
CONFIRMABLE_TOOLS = {"create_reminder"}


def _create_reminder_impl(message: str, datetime_str: str) -> str:
    """
    Implementation of create_reminder.

    Args:
        message: What to remind the user about
        datetime_str: When to remind them

    Returns:
        Confirmation message with reminder details, or error message.
    """
    due_datetime = parse_datetime(datetime_str)
    if due_datetime is None:
        return (
            f"Could not understand the date/time '{datetime_str}'. "
            "Try formats like 'tomorrow at 3pm', 'next Tuesday at noon', or 'January 15th at 10am'."
        )

    # Check if the time is in the past
    if due_datetime < datetime.now():
        return (
            f"The specified time ({due_datetime.strftime('%B %d at %I:%M %p')}) is in the past. "
            "Please specify a future date and time."
        )

    reminder = get_reminder_manager().create_reminder(message, due_datetime)
    formatted_time = reminder.due_datetime.strftime("%A, %B %d at %I:%M %p")
    return f"Reminder created: '{message}' for {formatted_time}."


@tool
def create_reminder(message: str, datetime_str: str) -> str:
    """Create a reminder. Requires user confirmation.

    Args:
        message: What to remind about
        datetime_str: When (e.g. "3pm", "tomorrow at noon"). If only time given, assume today unless already passed.
    """
    return _create_reminder_impl(message, datetime_str)


def tool_requires_confirmation(tool_name: str) -> bool:
    """Check if a tool requires confirmation before execution."""
    return tool_name in CONFIRMABLE_TOOLS


@tool
def list_reminders() -> str:
    """
    List all pending reminders.

    Returns:
        A formatted list of all pending reminders with their IDs, messages, and due times.
    """
    reminders = get_reminder_manager().list_reminders(status=ReminderStatus.PENDING)

    if not reminders:
        return "You have no pending reminders."

    lines = ["Your pending reminders:"]
    for r in reminders:
        formatted_time = r.due_datetime.strftime("%A, %B %d at %I:%M %p")
        lines.append(f"- ID {r.id}: '{r.message}' - {formatted_time}")

    return "\n".join(lines)


@tool
def update_reminder(reminder_id: int, new_message: str | None = None, new_datetime_str: str | None = None) -> str:
    """Update a reminder's message and/or time.

    Args:
        reminder_id: ID from list_reminders
        new_message: New message (optional)
        new_datetime_str: New time (optional)
    """
    reminder = get_reminder_manager().get_reminder(reminder_id)
    if reminder is None:
        return f"No reminder found with ID {reminder_id}. Use list_reminders to see your reminders."

    new_datetime = None
    if new_datetime_str:
        new_datetime = parse_datetime(new_datetime_str)
        if new_datetime is None:
            return (
                f"Could not understand the date/time '{new_datetime_str}'. "
                "Try formats like 'tomorrow at 3pm' or 'next Tuesday at noon'."
            )
        if new_datetime < datetime.now():
            return (
                f"The specified time ({new_datetime.strftime('%B %d at %I:%M %p')}) is in the past. "
                "Please specify a future date and time."
            )

    updated = get_reminder_manager().update_reminder(
        reminder_id,
        message=new_message,
        due_datetime=new_datetime,
    )

    if updated is None:
        return f"No reminder found with ID {reminder_id}."

    formatted_time = updated.due_datetime.strftime("%A, %B %d at %I:%M %p")
    return f"Reminder updated: '{updated.message}' for {formatted_time}."


@tool
def delete_reminder(reminder_id: int) -> str:
    """Delete a reminder by ID."""
    success = get_reminder_manager().delete_reminder(reminder_id)
    if success:
        return f"Reminder {reminder_id} has been deleted."
    else:
        return f"No reminder found with ID {reminder_id}. Use list_reminders to see your reminders."
