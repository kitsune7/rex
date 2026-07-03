"""Optional live-model scenario evals for comparing LLM backends."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent.rescue_parsing import RescueParsingChatOpenAI
from agent.tools.reminder import ReminderManager, ReminderStatus
from agent.tools.timer import TimerManager
from server.voice_agent import VoiceAgent

from .scenarios import INTEGRATION_TRANSCRIPTS


@dataclass(frozen=True)
class IntegrationResult:
    scenario_id: str
    passed: bool
    error: str | None = None


def _create_live_model(*, api_base: str, model: str) -> ChatOpenAI:
    return RescueParsingChatOpenAI(
        model=model,
        openai_api_base=api_base,
        api_key=SecretStr("not-needed"),
        temperature=0.2,
    )


def _check_side_effects(
    scenario_id: str,
    *,
    timer_manager: TimerManager,
    reminder_manager: ReminderManager,
    needs_confirmation: bool,
) -> None:
    if scenario_id == "set_timer_simple":
        with timer_manager._timers_lock:
            assert len(timer_manager._timers) >= 1, "Expected a timer to be created"
        return

    if scenario_id == "check_timers_after_set":
        with timer_manager._timers_lock:
            assert len(timer_manager._timers) >= 1, "Expected an active timer"
        return

    if scenario_id == "reminder_requires_confirmation":
        pending = reminder_manager.list_reminders(status=ReminderStatus.PENDING)
        assert needs_confirmation, "Expected confirmation pause for create_reminder"
        assert len(pending) == 0, "Reminder must not be created before confirmation"
        return

    raise ValueError(f"No integration checks defined for {scenario_id}")


def run_integration_evals(
    *,
    api_base: str,
    model: str,
    db_path: Path | None = None,
) -> list[IntegrationResult]:
    """Run a small live-model eval suite against the configured backend."""
    results: list[IntegrationResult] = []
    llm = _create_live_model(api_base=api_base, model=model)

    for scenario_id, transcript in INTEGRATION_TRANSCRIPTS:
        timer_manager = TimerManager()
        reminder_manager = ReminderManager(db_path=db_path or Path("data/integration_eval_reminders.db"))
        voice_agent = VoiceAgent(timer_manager, reminder_manager, llm=llm)
        thread_id = str(uuid.uuid4())

        try:
            if scenario_id == "check_timers_after_set":
                voice_agent.chat("Set a 5 minute timer", thread_id)

            turn = voice_agent.chat(transcript, thread_id)
            _check_side_effects(
                scenario_id,
                timer_manager=timer_manager,
                reminder_manager=reminder_manager,
                needs_confirmation=turn.needs_followup and turn.pending_confirmation is not None,
            )
            results.append(IntegrationResult(scenario_id=scenario_id, passed=True))
        except Exception as exc:
            results.append(
                IntegrationResult(scenario_id=scenario_id, passed=False, error=str(exc))
            )
        finally:
            timer_manager.cleanup()

    return results
