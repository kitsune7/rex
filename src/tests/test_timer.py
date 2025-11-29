"""Tests for timer tools: parse_duration, _format_duration, and TimerManager."""

from unittest.mock import MagicMock, patch

import pytest

from agent.tools.timer import (
    TimerManager,
    TimerState,
    parse_duration,
    set_timer_manager,
)


class TestParseDuration:
    """Tests for parse_duration function."""

    def test_minutes(self):
        assert parse_duration("5 minutes") == 300.0
        assert parse_duration("1 minute") == 60.0
        assert parse_duration("10 mins") == 600.0
        assert parse_duration("2 min") == 120.0
        assert parse_duration("3m") == 180.0

    def test_seconds(self):
        assert parse_duration("30 seconds") == 30.0
        assert parse_duration("1 second") == 1.0
        assert parse_duration("45 secs") == 45.0
        assert parse_duration("15 sec") == 15.0
        assert parse_duration("10s") == 10.0

    def test_hours(self):
        assert parse_duration("1 hour") == 3600.0
        assert parse_duration("2 hours") == 7200.0
        assert parse_duration("1 hr") == 3600.0
        assert parse_duration("3 hrs") == 10800.0
        assert parse_duration("1h") == 3600.0

    def test_combined_durations(self):
        assert parse_duration("1 hour 30 minutes") == 5400.0
        assert parse_duration("1 minute 30 seconds") == 90.0
        assert parse_duration("2 hours 15 minutes 30 seconds") == 8130.0
        assert parse_duration("1h 30m") == 5400.0
        assert parse_duration("1m 30s") == 90.0

    def test_decimal_values(self):
        assert parse_duration("1.5 hours") == 5400.0
        assert parse_duration("2.5 minutes") == 150.0
        assert parse_duration("30.5 seconds") == 30.5

    def test_bare_number_assumes_minutes(self):
        assert parse_duration("5") == 300.0
        assert parse_duration("10") == 600.0

    def test_case_insensitive(self):
        assert parse_duration("5 MINUTES") == 300.0
        assert parse_duration("30 Seconds") == 30.0
        assert parse_duration("1 HOUR") == 3600.0

    def test_whitespace_handling(self):
        assert parse_duration("  5 minutes  ") == 300.0
        assert parse_duration("1  hour") == 3600.0

    def test_invalid_returns_none(self):
        assert parse_duration("") is None
        assert parse_duration("invalid") is None
        assert parse_duration("no numbers here") is None

    def test_zero_duration_returns_none(self):
        assert parse_duration("0 minutes") is None
        assert parse_duration("0") is None


class TestFormatDuration:
    """Tests for TimerManager._format_duration static method."""

    def test_seconds_only(self):
        assert TimerManager._format_duration(1) == "1 second"
        assert TimerManager._format_duration(30) == "30 seconds"
        assert TimerManager._format_duration(59) == "59 seconds"

    def test_minutes_only(self):
        assert TimerManager._format_duration(60) == "1 minute"
        assert TimerManager._format_duration(120) == "2 minutes"
        assert TimerManager._format_duration(300) == "5 minutes"

    def test_minutes_and_seconds(self):
        assert TimerManager._format_duration(90) == "1 minute 30 seconds"
        assert TimerManager._format_duration(125) == "2 minutes 5 seconds"
        assert TimerManager._format_duration(61) == "1 minute 1 second"

    def test_hours_only(self):
        assert TimerManager._format_duration(3600) == "1 hour"
        assert TimerManager._format_duration(7200) == "2 hours"

    def test_hours_and_minutes(self):
        assert TimerManager._format_duration(3660) == "1 hour 1 minute"
        assert TimerManager._format_duration(5400) == "1 hour 30 minutes"
        assert TimerManager._format_duration(7320) == "2 hours 2 minutes"

    def test_truncates_to_int(self):
        # Float seconds should be truncated
        assert TimerManager._format_duration(30.7) == "30 seconds"
        assert TimerManager._format_duration(90.9) == "1 minute 30 seconds"


class TestTimerManager:
    """Tests for TimerManager class with mocking."""

    @pytest.fixture
    def fresh_timer_manager(self):
        """Create a fresh TimerManager instance for testing."""
        # Mock sound file loading to avoid file dependencies
        with patch("agent.tools.timer.sf.read") as mock_read:
            mock_read.return_value = (MagicMock(), 44100)
            manager = TimerManager()
            # Disable sound playback in tests
            manager._sound_data = None

            # Set as the module-level manager for tool access
            set_timer_manager(manager)

            yield manager

            # Clean up timers
            manager.cleanup()

    def test_set_timer_returns_confirmation(self, fresh_timer_manager):
        result = fresh_timer_manager.set_timer("test", 60.0)
        assert "test" in result
        assert "1 minute" in result

    def test_set_timer_creates_timer(self, fresh_timer_manager):
        fresh_timer_manager.set_timer("pizza", 300.0)

        assert "pizza" in fresh_timer_manager._timers
        timer = fresh_timer_manager._timers["pizza"]
        assert timer.name == "pizza"
        assert timer.duration_seconds == 300.0
        assert timer.state == TimerState.PENDING

    def test_set_timer_replaces_existing(self, fresh_timer_manager):
        fresh_timer_manager.set_timer("timer", 60.0)
        fresh_timer_manager.set_timer("timer", 120.0)

        assert len(fresh_timer_manager._timers) == 1
        assert fresh_timer_manager._timers["timer"].duration_seconds == 120.0

    def test_check_timers_empty(self, fresh_timer_manager):
        result = fresh_timer_manager.check_timers()
        assert result == "No active timers."

    def test_check_timers_shows_remaining(self, fresh_timer_manager):
        fresh_timer_manager.set_timer("test", 300.0)
        result = fresh_timer_manager.check_timers()

        assert "test" in result
        assert "remaining" in result

    def test_stop_timer_cancels_pending(self, fresh_timer_manager):
        fresh_timer_manager.set_timer("test", 300.0)
        result = fresh_timer_manager.stop_timer("test")

        assert "Cancelled" in result
        assert "test" not in fresh_timer_manager._timers

    def test_stop_timer_not_found(self, fresh_timer_manager):
        result = fresh_timer_manager.stop_timer("nonexistent")
        assert "not found" in result.lower() or "No timer" in result

    def test_stop_any_ringing_returns_false_when_none_ringing(self, fresh_timer_manager):
        fresh_timer_manager.set_timer("test", 300.0)
        assert fresh_timer_manager.stop_any_ringing() is False

    def test_mute_and_unmute(self, fresh_timer_manager):
        assert fresh_timer_manager._muted is False

        fresh_timer_manager.mute()
        assert fresh_timer_manager._muted is True

        fresh_timer_manager.unmute()
        assert fresh_timer_manager._muted is False
