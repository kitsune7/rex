"""Voice agent wrapper for the server.

Builds a LangChain agent identical to the desktop CLI's, with the same tools
and Bedrock-proxy LLM, but with the system prompt extended to emit an emotion
tag the server can parse off the spoken text.

We deliberately keep this separate from ``agent.agent`` so the desktop CLI
keeps working unchanged.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import date

import logging

from langchain.agents import create_agent as create_lc_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import SecretStr
from rex.settings import load_settings

log = logging.getLogger(__name__)

# Langfuse is optional. If the package or its env vars aren't configured,
# we want chats to still work — just without tracing.
try:
    from langfuse.langchain import CallbackHandler as _LangfuseCallbackHandler
except Exception:  # pragma: no cover - tracing is best-effort
    _LangfuseCallbackHandler = None  # type: ignore[assignment]


def _safe_callback_handler():
    """Build a Langfuse callback handler if it's available, else return None.

    Any failure (missing keys, network error, package not installed) is logged
    once and swallowed so the chat turn still completes.
    """
    if _LangfuseCallbackHandler is None:
        return None
    try:
        return _LangfuseCallbackHandler()
    except Exception:
        log.warning(
            "Langfuse CallbackHandler init failed — continuing without tracing",
            exc_info=True,
        )
        return None

from agent.tools import (
    ReminderManager,
    TimerManager,
    calculate,
    create_reminder_tools,
    create_timer_tools,
    get_current_time,
    tool_requires_confirmation,
)
from agent.tools.reminder import CONFIRMABLE_TOOLS

from .emotion import EMOTION_SYSTEM_PROMPT_SUFFIX, parse_emotion

MAX_HISTORY_MESSAGES = 20


def _server_system_prompt() -> str:
    return (
        "IMPORTANT: Your response will be spoken aloud through a robot. "
        "NEVER use markdown, asterisks, bullet points, or any formatting. "
        "Write naturally as if speaking.\n\n"
        f"You are Rex, a voice assistant. Today is {date.today()}. Be concise."
        + EMOTION_SYSTEM_PROMPT_SUFFIX
    )


@dataclass
class PendingConfirmation:
    """A tool call that's paused waiting for the user to confirm."""

    tool_name: str
    tool_args: dict
    confirmation_prompt: str
    thread_id: str


@dataclass
class ChatTurnResult:
    """Result of one /chat invocation."""

    emotion: str
    text: str
    thread_id: str
    needs_followup: bool
    pending_confirmation: PendingConfirmation | None = None


class VoiceAgent:
    """Thread-safe wrapper around the LangChain agent for HTTP serving.

    The underlying ``MemorySaver`` checkpointer is keyed by ``thread_id``, so
    multiple Reachy sessions could in principle share one VoiceAgent. Tool
    state (timers, reminders) is module-level though, so for now we assume one
    user.
    """

    def __init__(
        self,
        timer_manager: TimerManager,
        reminder_manager: ReminderManager,
    ) -> None:
        set_timer, check_timers, stop_timer = create_timer_tools(timer_manager)
        create_reminder, list_reminders, update_reminder, delete_reminder = (
            create_reminder_tools(reminder_manager)
        )
        tools = [
            get_current_time,
            calculate,
            set_timer,
            check_timers,
            stop_timer,
            create_reminder,
            list_reminders,
            update_reminder,
            delete_reminder,
        ]
        self._checkpointer = MemorySaver()

        # Reach into the same factory the desktop uses but swap the system
        # prompt. We rebuild the LLM here to keep it explicit.
        settings = load_settings()
        llm = ChatOpenAI(
            openai_api_base=settings.llm.api_base,
            api_key=SecretStr("not-needed"),
            temperature=0.7,
        )
        use_interrupt = any(t.name in CONFIRMABLE_TOOLS for t in tools)

        self._agent = create_lc_agent(
            llm,
            tools,
            system_prompt=_server_system_prompt(),
            checkpointer=self._checkpointer,
            interrupt_before=["tools"] if use_interrupt else None,
        )
        self._lock = threading.Lock()

    # ----- public API ------------------------------------------------------

    def chat(
        self,
        transcript: str,
        thread_id: str | None,
    ) -> ChatTurnResult:
        """Run one conversational turn.

        Args:
            transcript: User's transcribed speech.
            thread_id: ID returned from a prior call to continue the
                conversation, or ``None`` to start fresh.
        """
        thread_id = thread_id or str(uuid.uuid4())
        config = self._config(thread_id)

        # Build inputs from the checkpointer's saved history.
        with self._lock:
            state = self._agent.get_state(config)
            history = list(state.values.get("messages", []))
            history.append(HumanMessage(content=transcript))

            response = self._agent.invoke({"messages": history}, config=config)
            response = self._drive_until_user_input(response, config)

        return self._build_result(response, thread_id)

    def confirm(
        self,
        thread_id: str,
        approved: bool,
        modification_request: str | None = None,
    ) -> ChatTurnResult:
        """Resolve a pending tool-call confirmation.

        Args:
            thread_id: Thread ID returned from the prior /chat call.
            approved: ``True`` to run the tool, ``False`` to cancel.
            modification_request: Optional natural-language request that
                replaces the cancelled call (e.g. "make it 10 minutes
                instead of 5").
        """
        config = self._config(thread_id)

        with self._lock:
            if approved:
                response = self._agent.invoke(Command(resume=True), config=config)
            else:
                state = self._agent.get_state(config)
                messages = list(state.values.get("messages", []))
                for msg in reversed(messages):
                    if getattr(msg, "tool_calls", None):
                        tool_call = msg.tool_calls[0]
                        if modification_request:
                            content = (
                                "User declined this action and requested a "
                                f"modification: '{modification_request}'. "
                                "Please propose the action again with updated "
                                "parameters."
                            )
                        else:
                            content = "User cancelled this action."
                        messages.append(
                            ToolMessage(content=content, tool_call_id=tool_call["id"])
                        )
                        break
                self._agent.update_state(config, {"messages": messages})
                # Drive forward so the LLM acknowledges the cancellation.
                response = self._agent.invoke(None, config=config)

            response = self._drive_until_user_input(response, config)

        return self._build_result(response, thread_id)

    # ----- internals -------------------------------------------------------

    def _config(self, thread_id: str) -> dict:
        cb = _safe_callback_handler()
        callbacks = [cb] if cb is not None else []
        return {
            "callbacks": callbacks,
            "configurable": {"thread_id": thread_id},
        }

    def _drive_until_user_input(self, response: dict, config: dict) -> dict:
        """Walk through any auto-executable tool calls until either the agent
        finishes, or it pauses on a tool that needs user confirmation."""
        state = self._agent.get_state(config)
        while state.next:
            tool_info = _extract_pending_tool_call(response)
            if tool_info and tool_requires_confirmation(tool_info[0]):
                # Pause here — caller will handle confirmation.
                return response
            if tool_info:
                # Non-confirmable tool — keep going.
                response = self._agent.invoke(None, config=config)
                state = self._agent.get_state(config)
                continue
            break
        return response

    def _build_result(self, response: dict, thread_id: str) -> ChatTurnResult:
        # If we paused on a confirmation, surface the PendingConfirmation.
        state = self._agent.get_state({"configurable": {"thread_id": thread_id}})
        if state.next:
            tool_info = _extract_pending_tool_call(response)
            if tool_info and tool_requires_confirmation(tool_info[0]):
                name, args = tool_info
                prompt = _format_confirmation_prompt(name, args)
                emotion, clean = parse_emotion(prompt)
                return ChatTurnResult(
                    emotion=emotion or "alert",
                    text=clean,
                    thread_id=thread_id,
                    needs_followup=True,
                    pending_confirmation=PendingConfirmation(
                        tool_name=name,
                        tool_args=args,
                        confirmation_prompt=prompt,
                        thread_id=thread_id,
                    ),
                )

        raw_text = _extract_text_response(response)
        emotion, clean = parse_emotion(raw_text)
        return ChatTurnResult(
            emotion=emotion,
            text=clean,
            thread_id=thread_id,
            needs_followup=_looks_like_question(clean),
        )


# ----- helpers --------------------------------------------------------------


def _extract_text_response(response) -> str:
    if not hasattr(response, "get") or "messages" not in response:
        return str(response)
    messages = response["messages"]
    if not messages:
        return ""
    last = messages[-1]
    if not hasattr(last, "content"):
        return str(last)
    content = last.content
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and "text" in first:
            return first["text"]
        return str(first)
    return str(content)


def _extract_pending_tool_call(response) -> tuple[str, dict] | None:
    messages = response.get("messages", []) if hasattr(response, "get") else []
    for msg in reversed(messages):
        if getattr(msg, "tool_calls", None):
            tc = msg.tool_calls[0]
            return tc["name"], tc["args"]
    return None


def _format_confirmation_prompt(tool_name: str, tool_args: dict) -> str:
    if tool_name == "create_reminder":
        message = tool_args.get("message", "")
        when = tool_args.get("datetime_str", "")
        return (
            f"[emotion:thinking] I'm about to create a reminder: '{message}' "
            f"for {when}. Should I go ahead?"
        )
    args = ", ".join(f"{k}={v}" for k, v in tool_args.items())
    return (
        f"[emotion:thinking] I'm about to run {tool_name}({args}). "
        "Should I go ahead?"
    )


def _looks_like_question(text: str) -> bool:
    """Heuristic: treat trailing '?' as a follow-up prompt."""
    if not text:
        return False
    return text.rstrip().endswith("?")
