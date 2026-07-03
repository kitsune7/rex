"""Deterministic scenario eval tests for Rex voice workflows."""

from __future__ import annotations

import pytest

from agent.evals.runner import run_scenario
from agent.evals.scenarios import SCENARIOS
from agent.tools.reminder import ReminderManager
from agent.tools.timer import TimerManager


@pytest.fixture
def isolated_managers(tmp_path):
    timer_manager = TimerManager()
    reminder_manager = ReminderManager(db_path=tmp_path / "eval_reminders.db")
    yield timer_manager, reminder_manager
    timer_manager.cleanup()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario.id for scenario in SCENARIOS])
def test_scenario(scenario, isolated_managers):
    timer_manager, reminder_manager = isolated_managers
    result = run_scenario(
        scenario,
        timer_manager=timer_manager,
        reminder_manager=reminder_manager,
        db_path=reminder_manager._db_path,
    )
    assert result.passed, result.error


@pytest.mark.integration
@pytest.mark.parametrize(
    "api_base",
    [pytest.param(None, id="settings-default")],
)
def test_integration_evals(api_base):
    import os

    from agent.evals.integration import run_integration_evals
    from rex.settings import load_settings

    settings = load_settings()
    api_base = api_base or os.environ.get("REX_LLM_API_BASE") or settings.llm.api_base
    model = os.environ.get("REX_LLM_MODEL") or settings.llm.model

    results = run_integration_evals(api_base=api_base, model=model)
    failures = [result for result in results if not result.passed]
    assert not failures, "; ".join(f"{item.scenario_id}: {item.error}" for item in failures)
