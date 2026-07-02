"""Rescue parsing for text-based tool calls from local LLMs.

Some OpenAI-compatible backends (e.g. omlx with small models) emit tool
invocations as JSON in the message content instead of populating the
structured ``tool_calls`` field. This module detects those payloads and
converts them into LangChain-compatible tool call objects so the agent
graph can execute tools normally.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rex.settings import load_settings

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(r"^<\|[^|]+\|>\s*")
_JSON_OBJECT_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _strip_json_fences(text: str) -> str:
    match = _JSON_FENCE_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def _normalize_tool_args(raw_args: Any) -> dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        raw_args = raw_args.strip()
        if not raw_args:
            return {}
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _parse_tool_call_dict(item: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    name = item.get("name")
    args = _normalize_tool_args(item.get("parameters", item.get("args")))

    function_field = item.get("function")
    if isinstance(function_field, str):
        name = name or function_field
    elif isinstance(function_field, dict):
        name = name or function_field.get("name")
        if "arguments" in function_field:
            args = _normalize_tool_args(function_field["arguments"])
        elif "parameters" in function_field:
            args = _normalize_tool_args(function_field["parameters"])

    if item.get("type") == "function" and isinstance(item.get("name"), str):
        name = name or item["name"]

    if not name or not isinstance(name, str):
        return None

    return name, args


def _extract_json_payload(text: str) -> str | None:
    text = _strip_json_fences(text)
    text = _SPECIAL_TOKEN_RE.sub("", text).strip()
    if text.startswith(("{", "[")):
        return text

    match = _JSON_OBJECT_RE.search(text)
    if match:
        return match.group(1).strip()

    return None


def parse_tool_calls_from_content(content: str | list | None) -> list[dict[str, Any]] | None:
    """Parse text-based tool call JSON from model content.

    Supported shapes include:
    - ``{"name": "get_current_time", "parameters": {}}``
    - ``{"type": "function", "function": "get_current_time", "parameters": {}}``
    - ``{"type": "function", "function": {"name": "...", "arguments": "{}"}}``
    - ``<|python_tag|>{"name": "get_current_time", "parameters": {}}``
    - A JSON array of the above objects
    """
    if content is None:
        return None

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        content = "".join(text_parts)

    if not isinstance(content, str):
        return None

    json_payload = _extract_json_payload(content)
    if json_payload is None:
        return None

    try:
        data = json.loads(json_payload)
    except json.JSONDecodeError:
        return None

    items = data if isinstance(data, list) else [data]
    tool_calls: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        parsed = _parse_tool_call_dict(item)
        if parsed is None:
            continue
        name, args = parsed
        tool_calls.append(
            {
                "name": name,
                "args": args,
                "id": f"call_rescue_{uuid.uuid4().hex[:12]}",
                "type": "tool_call",
            }
        )

    return tool_calls or None


def rescue_parse_ai_message(message: AIMessage) -> AIMessage:
    """Convert text-based tool calls on an AIMessage into structured tool_calls."""
    if message.tool_calls:
        return message

    parsed = parse_tool_calls_from_content(message.content)
    if not parsed:
        return message

    return AIMessage(
        content="",
        tool_calls=parsed,
        id=message.id,
        response_metadata=message.response_metadata,
    )


class RescueParsingChatOpenAI(ChatOpenAI):
    """ChatOpenAI that promotes text-based tool calls to structured tool_calls."""

    def _create_chat_result(self, response, generation_info):
        result = super()._create_chat_result(response, generation_info)
        for generation in result.generations:
            message = getattr(generation, "message", None)
            if isinstance(message, AIMessage):
                generation.message = rescue_parse_ai_message(message)
        return result


def create_chat_model(*, temperature: float = 0.7) -> BaseChatModel:
    """Create the configured chat model with rescue parsing enabled."""
    settings = load_settings()
    return RescueParsingChatOpenAI(
        model=settings.llm.model,
        openai_api_base=settings.llm.api_base,
        api_key=SecretStr("not-needed"),
        temperature=temperature,
    )
