"""Plain-data scenario fixtures for Rex voice-assistant evals."""

from __future__ import annotations

from agent.tools.reminder import ReminderStatus

from .fake_model import text_response, tool_call
from .types import ConfirmTurn, Scenario, SetupAction, UserTurn


def _assert_timer_count(ctx, expected: int) -> None:
    with ctx.timer_manager._timers_lock:
        assert len(ctx.timer_manager._timers) == expected


def _assert_reminder_count(ctx, expected: int) -> None:
    pending = ctx.reminder_manager.list_reminders(status=ReminderStatus.PENDING)
    assert len(pending) == expected


def _assert_set_timer_simple(ctx) -> None:
    _assert_timer_count(ctx, 1)
    assert ctx.last_result is not None
    assert "timer" in ctx.last_result.text.lower()


def _assert_check_timers_after_set(ctx) -> None:
    _assert_timer_count(ctx, 1)
    assert ctx.last_result is not None
    assert "5 minute" in ctx.last_result.text.lower() or "timer" in ctx.last_result.text.lower()


def _assert_stop_ringing_timer(ctx) -> None:
    with ctx.timer_manager._timers_lock:
        assert "kitchen" not in ctx.timer_manager._timers
    assert ctx.last_result is not None
    assert "stop" in ctx.last_result.text.lower()


def _assert_reminder_needs_confirmation(ctx) -> None:
    _assert_reminder_count(ctx, 0)
    assert ctx.last_result is not None
    assert ctx.last_result.needs_followup is True
    assert ctx.last_result.pending_confirmation is not None
    assert ctx.last_result.pending_confirmation.tool_name == "create_reminder"


def _assert_reminder_rejected(ctx) -> None:
    _assert_reminder_count(ctx, 0)
    assert ctx.last_result is not None
    assert ctx.last_result.needs_followup is False


def _assert_delete_reminder_with_id(ctx) -> None:
    _assert_reminder_count(ctx, 0)
    assert ctx.last_result is not None
    assert "deleted" in ctx.last_result.text.lower()


def _assert_malformed_output_handled(ctx) -> None:
    assert ctx.last_result is not None
    assert isinstance(ctx.last_result.text, str)


def _assert_long_conversation_retains_context(ctx) -> None:
    _assert_timer_count(ctx, 1)
    assert ctx.last_result is not None
    assert "pasta" in ctx.last_result.text.lower() or "timer" in ctx.last_result.text.lower()


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="set_timer_simple",
        description="Set a timer with a simple duration",
        steps=(
            UserTurn(
                transcript="Set a timer for 5 minutes",
                model_script=(
                    tool_call("set_timer", {"duration": "5 minutes"}),
                    text_response("[emotion:happy] Five minute timer is set."),
                ),
            ),
        ),
        assert_fn=_assert_set_timer_simple,
    ),
    Scenario(
        id="check_timers_after_set",
        description="Ask for current timers after setting one",
        steps=(
            UserTurn(
                transcript="Set a 5 minute timer",
                model_script=(
                    tool_call("set_timer", {"duration": "5 minutes"}, call_id="call_set"),
                    text_response("[emotion:happy] Timer set."),
                ),
            ),
            UserTurn(
                transcript="What timers do I have?",
                model_script=(
                    tool_call("check_timers", {}),
                    text_response("[emotion:neutral] You have one timer with about 5 minutes left."),
                ),
            ),
        ),
        assert_fn=_assert_check_timers_after_set,
    ),
    Scenario(
        id="stop_ringing_timer",
        description="Stop a ringing timer",
        steps=(
            SetupAction("seed_ringing_timer", {"name": "kitchen"}),
            UserTurn(
                transcript="Stop the timer",
                model_script=(
                    tool_call("stop_timer", {}),
                    text_response("[emotion:neutral] Stopped the kitchen timer."),
                ),
            ),
        ),
        assert_fn=_assert_stop_ringing_timer,
    ),
    Scenario(
        id="reminder_requires_confirmation",
        description="Create a reminder and pause for confirmation",
        steps=(
            UserTurn(
                transcript="Remind me to call mom tomorrow at 3pm",
                model_script=(
                    tool_call(
                        "create_reminder",
                        {"message": "call mom", "datetime_str": "tomorrow at 3pm"},
                    ),
                ),
            ),
        ),
        assert_fn=_assert_reminder_needs_confirmation,
    ),
    Scenario(
        id="reminder_confirmation_rejected",
        description="Reject a reminder confirmation and ensure it is not created",
        steps=(
            UserTurn(
                transcript="Remind me to water the plants tomorrow at 9am",
                model_script=(
                    tool_call(
                        "create_reminder",
                        {"message": "water the plants", "datetime_str": "tomorrow at 9am"},
                    ),
                ),
            ),
            ConfirmTurn(
                approved=False,
                model_script=(text_response("[emotion:neutral] Okay, I won't create that reminder."),),
            ),
        ),
        assert_fn=_assert_reminder_rejected,
    ),
    Scenario(
        id="delete_reminder_with_known_id",
        description="Delete a reminder after listing reminders to learn its ID",
        steps=(
            SetupAction(
                "seed_reminder",
                {"message": "buy milk", "hours_from_now": 2},
            ),
            UserTurn(
                transcript="What reminders do I have?",
                model_script=(
                    tool_call("list_reminders", {}),
                    text_response("[emotion:neutral] You have one reminder to buy milk."),
                ),
            ),
            UserTurn(
                transcript="Delete reminder 1",
                model_script=(
                    tool_call("delete_reminder", {"reminder_id": 1}),
                    text_response("[emotion:neutral] Reminder deleted."),
                ),
            ),
        ),
        assert_fn=_assert_delete_reminder_with_id,
    ),
    Scenario(
        id="malformed_empty_llm_output",
        description="Handle empty LLM output without crashing",
        steps=(
            UserTurn(
                transcript="What time is it?",
                model_script=(text_response(""),),
            ),
        ),
        assert_fn=_assert_malformed_output_handled,
    ),
    Scenario(
        id="long_conversation_tool_context",
        description="Continue a long conversation without losing earlier tool results",
        steps=(
            UserTurn(
                transcript="Set a timer named pasta for 10 minutes",
                model_script=(
                    tool_call("set_timer", {"duration": "10 minutes", "name": "pasta"}, call_id="call_set"),
                    text_response("[emotion:happy] Pasta timer is running."),
                ),
            ),
            *(
                UserTurn(
                    transcript=f"Small talk turn {index}",
                    model_script=(text_response(f"[emotion:neutral] Sure, turn {index}."),),
                )
                for index in range(1, 12)
            ),
            UserTurn(
                transcript="What's the pasta timer doing?",
                model_script=(
                    tool_call("check_timers", {}),
                    text_response("[emotion:neutral] The pasta timer still has several minutes left."),
                ),
            ),
        ),
        assert_fn=_assert_long_conversation_retains_context,
    ),
)

# Integration scenarios use real transcripts but skip scripted model responses.
INTEGRATION_TRANSCRIPTS: tuple[tuple[str, str], ...] = (
    ("set_timer_simple", "Set a timer for 5 minutes"),
    ("check_timers_after_set", "What timers are running?"),
    ("reminder_requires_confirmation", "Remind me to stretch tomorrow at 8am"),
)
