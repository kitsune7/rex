"""Tests for agent response extraction and confirmation flow."""

from langchain.agents import create_agent as lc_create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from agent import agent as agent_module
from agent.agent import (
    MAX_HISTORY_MESSAGES,
    PendingConfirmation,
    _trim_message_history,
    confirm_tool_call,
    extract_text_response,
)


class TestExtractTextResponse:
    """Tests for extract_text_response function."""

    def test_string_content(self):
        """Test extracting string content from response."""
        response = {"messages": [type("Message", (), {"content": "Hello, how can I help?"})()]}
        assert extract_text_response(response) == "Hello, how can I help?"

    def test_list_content_with_text_dict(self):
        """Test extracting content from list of dicts with 'text' field."""
        response = {"messages": [type("Message", (), {"content": [{"text": "Response text here"}]})()]}
        assert extract_text_response(response) == "Response text here"

    def test_list_content_without_text_field(self):
        """Test extracting content from list without 'text' field."""
        response = {"messages": [type("Message", (), {"content": ["plain string"]})()]}
        assert extract_text_response(response) == "plain string"

    def test_empty_messages(self):
        """Test handling empty messages list."""
        response = {"messages": []}
        assert extract_text_response(response) == "[No response generated]"

    def test_non_dict_response(self):
        """Test handling non-dict responses."""
        assert extract_text_response("plain string") == "plain string"
        assert extract_text_response(123) == "123"

    def test_missing_messages_key(self):
        """Test handling response without 'messages' key."""
        response = {"other_key": "value"}
        assert extract_text_response(response) == "{'other_key': 'value'}"

    def test_message_without_content_attribute(self):
        """Test handling message without content attribute."""
        response = {"messages": [{"raw": "data"}]}
        result = extract_text_response(response)
        assert "raw" in result or "data" in result

    def test_none_response(self):
        """Test handling None response."""
        assert extract_text_response(None) == "None"

    def test_last_message_is_used(self):
        """Test that the last message in the list is extracted."""
        response = {
            "messages": [
                type("Message", (), {"content": "First message"})(),
                type("Message", (), {"content": "Second message"})(),
                type("Message", (), {"content": "Last message"})(),
            ]
        }
        assert extract_text_response(response) == "Last message"


def _tool_call_ai(call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "check_timers", "args": {}, "id": call_id, "type": "tool_call"}
        ],
    )


class TestTrimMessageHistory:
    """History trimming must never orphan a tool result from its tool call."""

    def test_short_history_is_returned_unchanged(self):
        history = [HumanMessage(content="hi"), AIMessage(content="hello")]
        assert _trim_message_history(history) is history

    def test_never_starts_on_orphaned_tool_message(self):
        # Build a history where a naive [-N:] slice would begin on a ToolMessage
        # whose parent AIMessage(tool_calls=...) fell outside the window.
        history: list = []
        for turn in range(MAX_HISTORY_MESSAGES):
            history.append(HumanMessage(content=f"user {turn}"))
            history.append(_tool_call_ai(f"call_{turn}"))
            history.append(ToolMessage(content="ok", tool_call_id=f"call_{turn}"))
            history.append(AIMessage(content=f"done {turn}"))

        trimmed = _trim_message_history(history)

        assert len(trimmed) <= MAX_HISTORY_MESSAGES
        assert isinstance(trimmed[0], HumanMessage)
        # Every retained tool result has its matching tool call in the window.
        call_ids = {
            tc["id"]
            for msg in trimmed
            if isinstance(msg, AIMessage)
            for tc in (msg.tool_calls or [])
        }
        for msg in trimmed:
            if isinstance(msg, ToolMessage):
                assert msg.tool_call_id in call_ids

    def test_falls_back_to_dropping_leading_tool_messages(self):
        # A window with no HumanMessage still must not begin on a tool result.
        history = [
            ToolMessage(content="orphan", tool_call_id="call_x")
            for _ in range(MAX_HISTORY_MESSAGES + 2)
        ]
        history.append(AIMessage(content="summary"))

        trimmed = _trim_message_history(history)

        assert not isinstance(trimmed[0], ToolMessage)


class TestConfirmToolCall:
    def test_cancel_does_not_raise_when_graph_paused_at_tools(self, monkeypatch):
        """Cancelling a confirmable tool must update state from the tools node."""

        @tool
        def create_reminder(message: str, datetime_str: str) -> str:
            """Create a reminder."""
            return "created"

        class FakeModel(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        llm = FakeModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "create_reminder",
                                "args": {
                                    "message": "call mom",
                                    "datetime_str": "tomorrow at 3pm",
                                },
                                "id": "call_test",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            )
        )

        checkpointer = MemorySaver()
        fake_agent = lc_create_agent(
            llm,
            [create_reminder],
            system_prompt="You are Rex.",
            checkpointer=checkpointer,
            interrupt_before=["tools"],
        )
        monkeypatch.setattr(agent_module, "_agent", fake_agent)
        monkeypatch.setattr(agent_module, "_checkpointer", checkpointer)

        thread_id = "test-cancel-thread"
        config = {"configurable": {"thread_id": thread_id}, "callbacks": []}
        fake_agent.invoke(
            {"messages": [HumanMessage(content="Remind me to call mom tomorrow at 3pm")]},
            config=config,
        )

        pending = PendingConfirmation(
            tool_name="create_reminder",
            tool_args={"message": "call mom", "datetime_str": "tomorrow at 3pm"},
            confirmation_prompt="Should I proceed?",
            thread_id=thread_id,
        )

        response, history = confirm_tool_call(pending, confirmed=False)

        assert response == "Okay, I've cancelled that action."
        assert any(
            isinstance(message, ToolMessage) and "cancelled" in message.content.lower()
            for message in history
        )
