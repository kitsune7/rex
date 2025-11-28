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

from dateutil import parser as dateutil_parser
from langchain_core.tools import tool


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
    """Manages reminders with SQLite persistence."""

    _instance: "ReminderManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ReminderManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._db_lock = threading.Lock()
        self._db_path = Path("data/reminders.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

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

                return Reminder(
                    id=cursor.lastrowid,
                    message=message,
                    due_datetime=due_datetime,
                    created_at=now,
                    status=ReminderStatus.PENDING,
                )
            finally:
                conn.close()

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
                return self._row_to_reminder(row) if row else None
            finally:
                conn.close()

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
                return cursor.rowcount > 0
            finally:
                conn.close()

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


# Global reminder manager instance
_reminder_manager = ReminderManager()


def get_reminder_manager() -> ReminderManager:
    """Get the global ReminderManager instance."""
    return _reminder_manager


# Tool metadata for confirmation requirement
REQUIRES_CONFIRMATION = "requires_confirmation"

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
        return f"Could not understand the date/time '{datetime_str}'. Try formats like 'tomorrow at 3pm', 'next Tuesday at noon', or 'January 15th at 10am'."

    # Check if the time is in the past
    if due_datetime < datetime.now():
        return f"The specified time ({due_datetime.strftime('%B %d at %I:%M %p')}) is in the past. Please specify a future date and time."

    reminder = _reminder_manager.create_reminder(message, due_datetime)
    formatted_time = reminder.due_datetime.strftime("%A, %B %d at %I:%M %p")
    return f"Reminder created: '{message}' for {formatted_time}."


@tool
def create_reminder(message: str, datetime_str: str) -> str:
    """
    Create a reminder for a specific date and time.

    IMPORTANT: This tool requires user confirmation before the reminder is actually created.
    The system will ask the user to confirm the reminder details.

    Args:
        message: What to remind the user about (e.g., "take out the trash", "call mom")
        datetime_str: When to remind them (e.g., "tomorrow at 3pm", "next Tuesday at noon", "December 25th at 9am")

    Returns:
        Confirmation message with reminder details, or error message if datetime couldn't be parsed.

    Examples:
        - create_reminder("take out the trash", "tomorrow at 7am")
        - create_reminder("call mom", "next Sunday at 2pm")
        - create_reminder("dentist appointment", "January 15th at 10am")
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
    reminders = _reminder_manager.list_reminders(status=ReminderStatus.PENDING)

    if not reminders:
        return "You have no pending reminders."

    lines = ["Your pending reminders:"]
    for r in reminders:
        formatted_time = r.due_datetime.strftime("%A, %B %d at %I:%M %p")
        lines.append(f"- ID {r.id}: '{r.message}' - {formatted_time}")

    return "\n".join(lines)


@tool
def update_reminder(reminder_id: int, new_message: str | None = None, new_datetime_str: str | None = None) -> str:
    """
    Update an existing reminder's message and/or time.

    Args:
        reminder_id: The ID of the reminder to update (use list_reminders to see IDs)
        new_message: New message for the reminder (optional)
        new_datetime_str: New date/time for the reminder (optional, e.g., "tomorrow at 5pm")

    Returns:
        Confirmation message or error if the reminder wasn't found.

    Examples:
        - update_reminder(1, new_message="pick up groceries")
        - update_reminder(2, new_datetime_str="next Monday at 9am")
        - update_reminder(3, new_message="call dentist", new_datetime_str="Friday at 2pm")
    """
    reminder = _reminder_manager.get_reminder(reminder_id)
    if reminder is None:
        return f"No reminder found with ID {reminder_id}. Use list_reminders to see your reminders."

    new_datetime = None
    if new_datetime_str:
        new_datetime = parse_datetime(new_datetime_str)
        if new_datetime is None:
            return f"Could not understand the date/time '{new_datetime_str}'. Try formats like 'tomorrow at 3pm' or 'next Tuesday at noon'."
        if new_datetime < datetime.now():
            return f"The specified time ({new_datetime.strftime('%B %d at %I:%M %p')}) is in the past. Please specify a future date and time."

    updated = _reminder_manager.update_reminder(
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
    """
    Delete a reminder.

    Args:
        reminder_id: The ID of the reminder to delete (use list_reminders to see IDs)

    Returns:
        Confirmation message or error if the reminder wasn't found.

    Examples:
        - delete_reminder(1)
        - delete_reminder(5)
    """
    success = _reminder_manager.delete_reminder(reminder_id)
    if success:
        return f"Reminder {reminder_id} has been deleted."
    else:
        return f"No reminder found with ID {reminder_id}. Use list_reminders to see your reminders."
