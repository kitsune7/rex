"""Tests for reminder tools: ReminderManager, CRUD operations, and datetime parsing."""

from datetime import datetime, timedelta

import pytest

from agent.tools.reminder import (
    CONFIRMABLE_TOOLS,
    ReminderManager,
    ReminderStatus,
    create_reminder_tools,
    parse_datetime,
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
        # Create manager with temp db path
        db_path = tmp_path / "test_reminders.db"
        manager = ReminderManager(db_path=db_path)

        yield manager

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

    def test_get_next_pending_time_none(self, fresh_reminder_manager):
        """Returns None when no pending reminders exist."""
        assert fresh_reminder_manager.get_next_pending_time() is None

    def test_get_next_pending_time_single(self, fresh_reminder_manager):
        """Returns the due time of the only pending reminder."""
        due = datetime.now() + timedelta(hours=1)
        fresh_reminder_manager.create_reminder("Test", due)

        next_time = fresh_reminder_manager.get_next_pending_time()
        assert next_time is not None
        assert next_time == due

    def test_get_next_pending_time_multiple(self, fresh_reminder_manager):
        """Returns the earliest due time among multiple reminders."""
        due1 = datetime.now() + timedelta(hours=2)
        due2 = datetime.now() + timedelta(hours=1)  # Earlier
        due3 = datetime.now() + timedelta(hours=3)

        fresh_reminder_manager.create_reminder("Later", due1)
        fresh_reminder_manager.create_reminder("Earliest", due2)
        fresh_reminder_manager.create_reminder("Latest", due3)

        next_time = fresh_reminder_manager.get_next_pending_time()
        assert next_time is not None
        assert next_time == due2

    def test_get_next_pending_time_ignores_cleared(self, fresh_reminder_manager):
        """Cleared reminders are not considered."""
        due1 = datetime.now() + timedelta(hours=1)
        due2 = datetime.now() + timedelta(hours=2)

        r1 = fresh_reminder_manager.create_reminder("Will be cleared", due1)
        fresh_reminder_manager.create_reminder("Still pending", due2)
        fresh_reminder_manager.clear_reminder(r1.id)

        next_time = fresh_reminder_manager.get_next_pending_time()
        assert next_time is not None
        assert next_time == due2


class TestReminderManagerEvents:
    """Tests for ReminderManager event emission."""

    @pytest.fixture
    def manager_with_events(self, tmp_path):
        """Create a ReminderManager with an event bus."""
        from core.events import EventBus, ReminderScheduleChanged

        event_bus = EventBus()
        events_received = []

        def handler(event):
            events_received.append(event)

        event_bus.subscribe(ReminderScheduleChanged, handler)

        db_path = tmp_path / "test_reminders.db"
        manager = ReminderManager(db_path=db_path, event_bus=event_bus)

        return manager, events_received

    def test_create_reminder_emits_event(self, manager_with_events):
        manager, events = manager_with_events
        due = datetime.now() + timedelta(hours=1)

        manager.create_reminder("Test", due)

        assert len(events) == 1

    def test_update_due_datetime_emits_event(self, manager_with_events):
        manager, events = manager_with_events
        due = datetime.now() + timedelta(hours=1)
        new_due = datetime.now() + timedelta(hours=2)

        reminder = manager.create_reminder("Test", due)
        events.clear()

        manager.update_reminder(reminder.id, due_datetime=new_due)

        assert len(events) == 1

    def test_update_message_only_no_event(self, manager_with_events):
        """Updating only the message doesn't emit schedule change event."""
        manager, events = manager_with_events
        due = datetime.now() + timedelta(hours=1)

        reminder = manager.create_reminder("Test", due)
        events.clear()

        manager.update_reminder(reminder.id, message="Updated")

        assert len(events) == 0

    def test_delete_reminder_emits_event(self, manager_with_events):
        manager, events = manager_with_events
        due = datetime.now() + timedelta(hours=1)

        reminder = manager.create_reminder("Test", due)
        events.clear()

        manager.delete_reminder(reminder.id)

        assert len(events) == 1

    def test_snooze_reminder_emits_event(self, manager_with_events):
        manager, events = manager_with_events
        due = datetime.now() + timedelta(hours=1)
        new_due = datetime.now() + timedelta(hours=2)

        reminder = manager.create_reminder("Test", due)
        events.clear()

        manager.snooze_reminder(reminder.id, new_due)

        assert len(events) == 1


class TestReminderTools:
    """Tests for the tool functions."""

    @pytest.fixture
    def reminder_tools(self, tmp_path):
        """Create reminder tools with a fresh manager."""
        db_path = tmp_path / "test_reminders.db"
        manager = ReminderManager(db_path=db_path)
        create_reminder, list_reminders, update_reminder, delete_reminder = create_reminder_tools(manager)
        return manager, create_reminder, list_reminders, update_reminder, delete_reminder

    def test_create_reminder_tool(self, reminder_tools):
        _, create_reminder, _, _, _ = reminder_tools
        # Use a future date
        tomorrow = datetime.now() + timedelta(days=1)
        datetime_str = tomorrow.strftime("%Y-%m-%d at 10am")

        result = create_reminder.invoke({"message": "Test reminder", "datetime_str": datetime_str})

        assert "Reminder created" in result
        assert "Test reminder" in result

    def test_create_reminder_tool_invalid_datetime(self, reminder_tools):
        _, create_reminder, _, _, _ = reminder_tools
        result = create_reminder.invoke({"message": "Test", "datetime_str": ""})
        assert "Could not understand" in result

    def test_create_reminder_tool_past_time(self, reminder_tools):
        _, create_reminder, _, _, _ = reminder_tools
        past = datetime.now() - timedelta(days=1)
        datetime_str = past.strftime("%Y-%m-%d at 10am")

        result = create_reminder.invoke({"message": "Test", "datetime_str": datetime_str})
        assert "in the past" in result

    def test_list_reminders_tool_empty(self, reminder_tools):
        _, _, list_reminders, _, _ = reminder_tools
        result = list_reminders.invoke({})
        assert "no pending reminders" in result.lower()

    def test_list_reminders_tool(self, reminder_tools):
        manager, _, list_reminders, _, _ = reminder_tools
        due = datetime.now() + timedelta(hours=1)
        manager.create_reminder("Test reminder", due)

        result = list_reminders.invoke({})
        assert "Test reminder" in result
        assert "ID" in result

    def test_update_reminder_tool(self, reminder_tools):
        manager, _, _, update_reminder, _ = reminder_tools
        due = datetime.now() + timedelta(hours=1)
        reminder = manager.create_reminder("Original", due)

        result = update_reminder.invoke({"reminder_id": reminder.id, "new_message": "Updated"})

        assert "updated" in result.lower()
        assert "Updated" in result

    def test_update_reminder_tool_not_found(self, reminder_tools):
        _, _, _, update_reminder, _ = reminder_tools
        result = update_reminder.invoke({"reminder_id": 9999, "new_message": "Test"})
        assert "not found" in result.lower() or "No reminder" in result

    def test_delete_reminder_tool(self, reminder_tools):
        manager, _, _, _, delete_reminder = reminder_tools
        due = datetime.now() + timedelta(hours=1)
        reminder = manager.create_reminder("Test", due)

        result = delete_reminder.invoke({"reminder_id": reminder.id})

        assert "deleted" in result.lower()

    def test_delete_reminder_tool_not_found(self, reminder_tools):
        _, _, _, _, delete_reminder = reminder_tools
        result = delete_reminder.invoke({"reminder_id": 9999})
        assert "not found" in result.lower() or "No reminder" in result

    def test_create_reminder_requires_confirmation_flag(self):
        """Verify that create_reminder is marked as requiring confirmation."""
        assert "create_reminder" in CONFIRMABLE_TOOLS
        assert tool_requires_confirmation("create_reminder") is True
        assert tool_requires_confirmation("list_reminders") is False
