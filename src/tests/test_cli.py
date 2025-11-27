"""Tests for CLI special command handling."""

from unittest.mock import MagicMock

import pytest

from rex.cli import _handle_special_command


class TestHandleSpecialCommand:
    """Tests for _handle_special_command function."""

    @pytest.fixture
    def mock_timer_manager(self):
        """Create a mock timer manager."""
        manager = MagicMock()
        manager.stop_any_ringing.return_value = False
        return manager

    def test_stop_command_stops_ringing_timer(self, mock_timer_manager):
        """Test 'stop' command attempts to stop ringing timer."""
        mock_timer_manager.stop_any_ringing.return_value = True

        was_handled, history = _handle_special_command("stop", None, mock_timer_manager)

        assert was_handled is True
        assert history is None
        mock_timer_manager.stop_any_ringing.assert_called_once()

    def test_stop_the_timer_command(self, mock_timer_manager):
        """Test 'stop the timer' command."""
        mock_timer_manager.stop_any_ringing.return_value = True

        was_handled, history = _handle_special_command("stop the timer", None, mock_timer_manager)

        assert was_handled is True
        mock_timer_manager.stop_any_ringing.assert_called_once()

    def test_stop_command_no_ringing_timer(self, mock_timer_manager):
        """Test 'stop' command when no timer is ringing."""
        mock_timer_manager.stop_any_ringing.return_value = False

        was_handled, history = _handle_special_command("stop", None, mock_timer_manager)

        assert was_handled is True
        assert history is None

    def test_nevermind_during_conversation(self, mock_timer_manager):
        """Test 'nevermind' ends conversation when in follow-up mode."""
        existing_history = [{"role": "user", "content": "hello"}]

        was_handled, history = _handle_special_command("nevermind", existing_history, mock_timer_manager)

        assert was_handled is True
        assert history is None

    def test_never_mind_during_conversation(self, mock_timer_manager):
        """Test 'never mind' (with space) ends conversation."""
        existing_history = [{"role": "user", "content": "hello"}]

        was_handled, history = _handle_special_command("never mind", existing_history, mock_timer_manager)

        assert was_handled is True
        assert history is None

    def test_cancel_during_conversation(self, mock_timer_manager):
        """Test 'cancel' ends conversation when in follow-up mode."""
        existing_history = [{"role": "user", "content": "hello"}]

        was_handled, history = _handle_special_command("cancel", existing_history, mock_timer_manager)

        assert was_handled is True
        assert history is None

    def test_forget_it_during_conversation(self, mock_timer_manager):
        """Test 'forget it' ends conversation when in follow-up mode."""
        existing_history = [{"role": "user", "content": "hello"}]

        was_handled, history = _handle_special_command("forget it", existing_history, mock_timer_manager)

        assert was_handled is True
        assert history is None

    def test_stop_phrases_ignored_without_history(self, mock_timer_manager):
        """Test stop phrases (except 'stop') are ignored when not in conversation."""
        # 'nevermind' should not be handled when history is None
        was_handled, history = _handle_special_command("nevermind", None, mock_timer_manager)

        assert was_handled is False
        assert history is None

    def test_cancel_ignored_without_history(self, mock_timer_manager):
        """Test 'cancel' is ignored when not in conversation."""
        was_handled, history = _handle_special_command("cancel", None, mock_timer_manager)

        assert was_handled is False
        assert history is None

    def test_regular_command_not_handled(self, mock_timer_manager):
        """Test regular commands are not handled."""
        existing_history = [{"role": "user", "content": "hello"}]

        was_handled, history = _handle_special_command("what time is it", existing_history, mock_timer_manager)

        assert was_handled is False
        assert history == existing_history

    def test_regular_command_without_history(self, mock_timer_manager):
        """Test regular commands without history are not handled."""
        was_handled, history = _handle_special_command("set a timer", None, mock_timer_manager)

        assert was_handled is False
        assert history is None

    def test_history_preserved_for_unhandled_commands(self, mock_timer_manager):
        """Test that history is preserved when command is not handled."""
        existing_history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]

        was_handled, history = _handle_special_command("tell me a joke", existing_history, mock_timer_manager)

        assert was_handled is False
        assert history is existing_history
