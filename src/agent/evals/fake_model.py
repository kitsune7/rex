"""Scriptable fake chat model for deterministic scenario evals."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent.rescue_parsing import rescue_parse_ai_message


class ScriptableFakeChatModel(GenericFakeChatModel):
    """Fake chat model that replays scripted AIMessages in order.

    Applies rescue parsing so text-based tool calls behave like production.
    """

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for generation in result.generations:
            message = getattr(generation, "message", None)
            if isinstance(message, AIMessage):
                generation.message = rescue_parse_ai_message(message)
        return result


def make_fake_model(responses: Sequence[AIMessage]) -> ScriptableFakeChatModel:
    """Build a fake model that emits ``responses`` one per LLM invocation."""
    return ScriptableFakeChatModel(messages=iter(responses))


def tool_call(name: str, args: dict, *, call_id: str | None = None) -> AIMessage:
    """Build an AIMessage with a single structured tool call."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id or f"call_{name}",
                "type": "tool_call",
            }
        ],
    )


def text_response(content: str) -> AIMessage:
    """Build a plain-text AIMessage."""
    return AIMessage(content=content)
