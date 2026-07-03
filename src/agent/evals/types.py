"""Shared types for Rex scenario evals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from langchain_core.messages import AIMessage

from agent.tools.reminder import ReminderManager
from agent.tools.timer import TimerManager
from server.voice_agent import ChatTurnResult, VoiceAgent


@dataclass(frozen=True)
class UserTurn:
    """One user transcript and the model responses for that turn."""

    transcript: str
    model_script: tuple[AIMessage, ...]


@dataclass(frozen=True)
class ConfirmTurn:
    """Resolve a pending tool confirmation."""

    approved: bool
    model_script: tuple[AIMessage, ...] = ()


@dataclass(frozen=True)
class SetupAction:
    """Pre-seed manager state before the scenario runs."""

    action: Literal["seed_reminder", "seed_ringing_timer"]
    kwargs: dict = field(default_factory=dict)


ScenarioStep = UserTurn | ConfirmTurn | SetupAction


@dataclass(frozen=True)
class Scenario:
    """One eval scenario."""

    id: str
    description: str
    steps: tuple[ScenarioStep, ...]
    assert_fn: Callable[["ScenarioContext"], None]


@dataclass
class ScenarioContext:
    """Mutable state accumulated while running one scenario."""

    timer_manager: TimerManager
    reminder_manager: ReminderManager
    voice_agent: VoiceAgent
    thread_id: str | None = None
    last_result: ChatTurnResult | None = None
    results: list[ChatTurnResult] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioResult:
    """Outcome of one scenario run."""

    scenario_id: str
    passed: bool
    error: str | None = None
