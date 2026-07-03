"""Run transcript-driven Rex voice scenarios against a fake or live model."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from agent.tools.reminder import ReminderManager
from agent.tools.timer import TimerManager, TimerState
from server.voice_agent import VoiceAgent

from .fake_model import make_fake_model
from .types import (
    ConfirmTurn,
    Scenario,
    ScenarioContext,
    ScenarioResult,
    SetupAction,
    UserTurn,
)


def _collect_model_script(steps: tuple) -> list[AIMessage]:
    script: list[AIMessage] = []
    for step in steps:
        if isinstance(step, (UserTurn, ConfirmTurn)):
            script.extend(step.model_script)
    return script


def apply_setup(ctx: ScenarioContext, action: SetupAction) -> None:
    """Run a pre-scenario setup action."""
    from datetime import datetime, timedelta

    if action.action == "seed_reminder":
        hours = action.kwargs.get("hours_from_now", 1)
        message = action.kwargs["message"]
        due = datetime.now() + timedelta(hours=hours)
        ctx.reminder_manager.create_reminder(message, due)
        return

    if action.action == "seed_ringing_timer":
        name = action.kwargs.get("name", "timer")
        ctx.timer_manager.set_timer(name, 60)
        with ctx.timer_manager._timers_lock:
            timer = ctx.timer_manager._timers[name]
            timer.state = TimerState.RINGING
            ctx.timer_manager._current_ringing = name
        return

    raise ValueError(f"Unknown setup action: {action.action}")


def run_scenario(
    scenario: Scenario,
    *,
    timer_manager: TimerManager | None = None,
    reminder_manager: ReminderManager | None = None,
    llm=None,
    db_path: Path | None = None,
) -> ScenarioResult:
    """Execute one scenario and run its assertions."""
    timer_manager = timer_manager or TimerManager()
    if reminder_manager is None:
        reminder_manager = ReminderManager(db_path=db_path or Path("data/eval_reminders.db"))

    if llm is None:
        llm = make_fake_model(_collect_model_script(scenario.steps))

    voice_agent = VoiceAgent(timer_manager, reminder_manager, llm=llm)
    ctx = ScenarioContext(
        timer_manager=timer_manager,
        reminder_manager=reminder_manager,
        voice_agent=voice_agent,
    )

    try:
        for step in scenario.steps:
            if isinstance(step, SetupAction):
                apply_setup(ctx, step)
                continue

            if isinstance(step, UserTurn):
                result = ctx.voice_agent.chat(step.transcript, ctx.thread_id)
                ctx.thread_id = result.thread_id
                ctx.last_result = result
                ctx.results.append(result)
                continue

            if isinstance(step, ConfirmTurn):
                if ctx.thread_id is None:
                    raise AssertionError("ConfirmTurn reached before any chat turn")
                result = ctx.voice_agent.confirm(ctx.thread_id, approved=step.approved)
                ctx.last_result = result
                ctx.results.append(result)
                continue

            raise ValueError(f"Unknown step type: {type(step)!r}")

        scenario.assert_fn(ctx)
        return ScenarioResult(scenario_id=scenario.id, passed=True)
    except Exception as exc:
        return ScenarioResult(scenario_id=scenario.id, passed=False, error=str(exc))
    finally:
        timer_manager.cleanup()
