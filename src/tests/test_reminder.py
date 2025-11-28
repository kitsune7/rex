"""Tests for reminder tools: ReminderManager, CRUD operations, and datetime parsing."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from agent.tools.reminder import (
    ReminderManager,
    ReminderStatus,
    Reminder,
    parse_datetime,
    create_reminder,
    list_reminders,
    update_reminder,
    delete_reminder,
    CONFIRMABLE_TOOLS,
    tool_requires_confirmation,
)


class TestParseDatetime:
    """Tests for parse_datetime function."""

    def test_explicit_datetime(self):
        result = parse_datetime("2025-12-25 14:30")
        assert result is not None
        assert result.month == 12
        assert result.day == 25
        assert result.hour == 14
        assert result.minute == 30

    def test_time_only(self):
        result = parse_datetime("3pm")
        assert result is not None
        assert result.hour == 15

    def test_time_with_minutes(self):
        result = parse_datetime("3:30 pm")
        assert result is not None
        assert result.hour == 15
        assert result.minute == 30

    def test_tomorrow(self):
        result = parse_datetime("tomorrow at 9am")
        assert result is not None
        tomorrow = datetime.now() + timedelta(days=1)
        assert result.day == tomorrow.day
        assert result.hour == 9

    def test_date_with_time(self):
        result = parse_datetime("December 25th at 10am")
        assert result is not None
        assert result.month == 12
        assert result.day == 25
        assert result.hour == 10

    def test_noon(self):
        result = parse_datetime("noon")
        assert result is not None
        assert result.hour == 12

    def test_invalid_returns_none(self):
        # Note: dateutil is very permissive, so few strings are truly invalid
        assert parse_datetime("") is None


class TestReminderManager:
    """Tests for ReminderManager class."""

    @pytest.fixture
    def fresh_reminder_manager(self, tmp_path):
        """Create a fresh ReminderManager with a temp database."""
        # Reset singleton
        ReminderManager._instance = None

        # Create manager - it will try to create the default db path
        manager = ReminderManager()

        # Override the db path to use temp directory
        db_path = tmp_path / "test_reminders.db"
        manager._db_path = db_path
        manager._init_db()

        yield manager

        # Reset singleton after test
        ReminderManager._instance = None

    def test_create_reminder(self, fresh_reminder_manager):
        due = datetime.now() + timedelta(hours=1)
        reminder = fresh_reminder_manager.create_reminder("Test reminder", due)

        assert reminder.id is not None
        assert reminder.message == "Test reminder"
        assert reminder.status == ReminderStatus.PENDING
        assert reminder.due_datetime == due

    def test_get_reminder(self, fresh_reminder_manager):
        due = datetime.now() + timedelta(hours=1)
        created = fresh_reminder_manager.create_reminder("Test", due)

        fetched = fresh_reminder_manager.get_reminder(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.message == "Test"

    def test_get_reminder_not_found(self, fresh_reminder_manager):
        assert fresh_reminder_manager.get_reminder(9999) is None

    def test_list_reminders_empty(self, fresh_reminder_manager):
        reminders = fresh_reminder_manager.list_reminders()
        assert reminders == []

    def test_list_reminders(self, fresh_reminder_manager):
        due1 = datetime.now() + timedelta(hours=1)
        due2 = datetime.now() + timedelta(hours=2)
        fresh_reminder_manager.create_reminder("First", due1)
        fresh_reminder_manager.create_reminder("Second", due2)

        reminders = fresh_reminder_manager.list_reminders()
        assert len(reminders) == 2
        assert reminders[0].message == "First"  # Ordered by due time
        assert reminders[1].message == "Second"

    def test_list_reminders_by_status(self, fresh_reminder_manager):
        due = datetime.now() + timedelta(hours=1)
        r1 = fresh_reminder_manager.create_reminder("Pending", due)
        r2 = fresh_reminder_manager.create_reminder("Cleared", due)
        fresh_reminder_manager.clear_reminder(r2.id)

        pending = fresh_reminder_manager.list_reminders(status=ReminderStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].id == r1.id

        cleared = fresh_reminder_manager.list_reminders(status=ReminderStatus.CLEARED)
        assert len(cleared) == 1
        assert cleared[0].id == r2.id

    def test_get_due_reminders(self, fresh_reminder_manager):
        past = datetime.now() - timedelta(hours=1)
        future = datetime.now() + timedelta(hours=1)

        r1 = fresh_reminder_manager.create_reminder("Past due", past)
        fresh_reminder_manager.create_reminder("Not yet due", future)

        due = fresh_reminder_manager.get_due_reminders()
        assert len(due) == 1
        assert due[0].id == r1.id

    def test_update_reminder_message(self, fresh_reminder_manager):
        due = datetime.now() + timedelta(hours=1)
        reminder = fresh_reminder_manager.create_reminder("Original", due)

        updated = fresh_reminder_manager.update_reminder(reminder.id, message="Updated")
        assert updated is not None
        assert updated.message == "Updated"
        assert updated.due_datetime == due

    def test_update_reminder_datetime(self, fresh_reminder_manager):
        due = datetime.now() + timedelta(hours=1)
        new_due = datetime.now() + timedelta(hours=2)
        reminder = fresh_reminder_manager.create_reminder("Test", due)

        updated = fresh_reminder_manager.update_reminder(reminder.id, due_datetime=new_due)
        assert updated is not None
        assert updated.due_datetime == new_due

    def test_update_reminder_not_found(self, fresh_reminder_manager):
        assert fresh_reminder_manager.update_reminder(9999, message="Test") is None

    def test_delete_reminder(self, fresh_reminder_manager):
        due = datetime.now() + timedelta(hours=1)
        reminder = fresh_reminder_manager.create_reminder("Test", due)

        assert fresh_reminder_manager.delete_reminder(reminder.id) is True
        assert fresh_reminder_manager.get_reminder(reminder.id) is None

    def test_delete_reminder_not_found(self, fresh_reminder_manager):
        assert fresh_reminder_manager.delete_reminder(9999) is False

    def test_clear_reminder(self, fresh_reminder_manager):
        due = datetime.now() + timedelta(hours=1)
        reminder = fresh_reminder_manager.create_reminder("Test", due)

        cleared = fresh_reminder_manager.clear_reminder(reminder.id)
        assert cleared is not None
        assert cleared.status == ReminderStatus.CLEARED

    def test_snooze_reminder(self, fresh_reminder_manager):
        due = datetime.now() + timedelta(hours=1)
        new_due = datetime.now() + timedelta(hours=2)
        reminder = fresh_reminder_manager.create_reminder("Test", due)

        # First mark as delivered (simulating it was triggered)
        fresh_reminder_manager.update_reminder(reminder.id, status=ReminderStatus.DELIVERED)

        snoozed = fresh_reminder_manager.snooze_reminder(reminder.id, new_due)
        assert snoozed is not None
        assert snoozed.due_datetime == new_due
        assert snoozed.status == ReminderStatus.PENDING


class TestReminderTools:
    """Tests for the tool functions."""

    @pytest.fixture
    def fresh_reminder_manager(self, tmp_path):
        """Create a fresh ReminderManager with a temp database."""
        ReminderManager._instance = None

        # Create manager and override db path
        manager = ReminderManager()
        db_path = tmp_path / "test_reminders.db"
        manager._db_path = db_path
        manager._init_db()

        # Patch the global instance used by tools
        with patch("agent.tools.reminder._reminder_manager", manager):
            yield manager

        ReminderManager._instance = None

    def test_create_reminder_tool(self, fresh_reminder_manager):
        # Use a future date
        tomorrow = datetime.now() + timedelta(days=1)
        datetime_str = tomorrow.strftime("%Y-%m-%d at 10am")

        result = create_reminder.invoke({"message": "Test reminder", "datetime_str": datetime_str})

        assert "Reminder created" in result
        assert "Test reminder" in result

    def test_create_reminder_tool_invalid_datetime(self, fresh_reminder_manager):
        result = create_reminder.invoke({"message": "Test", "datetime_str": ""})
        assert "Could not understand" in result

    def test_create_reminder_tool_past_time(self, fresh_reminder_manager):
        past = datetime.now() - timedelta(days=1)
        datetime_str = past.strftime("%Y-%m-%d at 10am")

        result = create_reminder.invoke({"message": "Test", "datetime_str": datetime_str})
        assert "in the past" in result

    def test_list_reminders_tool_empty(self, fresh_reminder_manager):
        result = list_reminders.invoke({})
        assert "no pending reminders" in result.lower()

    def test_list_reminders_tool(self, fresh_reminder_manager):
        due = datetime.now() + timedelta(hours=1)
        fresh_reminder_manager.create_reminder("Test reminder", due)

        result = list_reminders.invoke({})
        assert "Test reminder" in result
        assert "ID" in result

    def test_update_reminder_tool(self, fresh_reminder_manager):
        due = datetime.now() + timedelta(hours=1)
        reminder = fresh_reminder_manager.create_reminder("Original", due)

        result = update_reminder.invoke({"reminder_id": reminder.id, "new_message": "Updated"})

        assert "updated" in result.lower()
        assert "Updated" in result

    def test_update_reminder_tool_not_found(self, fresh_reminder_manager):
        result = update_reminder.invoke({"reminder_id": 9999, "new_message": "Test"})
        assert "not found" in result.lower() or "No reminder" in result

    def test_delete_reminder_tool(self, fresh_reminder_manager):
        due = datetime.now() + timedelta(hours=1)
        reminder = fresh_reminder_manager.create_reminder("Test", due)

        result = delete_reminder.invoke({"reminder_id": reminder.id})

        assert "deleted" in result.lower()

    def test_delete_reminder_tool_not_found(self, fresh_reminder_manager):
        result = delete_reminder.invoke({"reminder_id": 9999})
        assert "not found" in result.lower() or "No reminder" in result

    def test_create_reminder_requires_confirmation_flag(self):
        """Verify that create_reminder is marked as requiring confirmation."""
        assert "create_reminder" in CONFIRMABLE_TOOLS
        assert tool_requires_confirmation("create_reminder") is True
        assert tool_requires_confirmation("list_reminders") is False
