"""Tests for text-based tool call rescue parsing."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agent.rescue_parsing import (
    parse_tool_calls_from_content,
    rescue_parse_ai_message,
)


class TestParseToolCallsFromContent:
    def test_name_and_parameters(self):
        content = '{"name": "get_current_time", "parameters": {}}'
        calls = parse_tool_calls_from_content(content)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0]["name"] == "get_current_time"
        assert calls[0]["args"] == {}
        assert calls[0]["type"] == "tool_call"
        assert calls[0]["id"].startswith("call_rescue_")

    def test_type_function_with_function_name(self):
        content = '{"type": "function", "function": "get_current_time", "parameters": {}}'
        calls = parse_tool_calls_from_content(content)
        assert calls is not None
        assert calls[0]["name"] == "get_current_time"

    def test_openai_function_object_with_string_arguments(self):
        content = (
            '{"type": "function", "function": {"name": "calculate", '
            '"arguments": "{\\"expression\\": \\"2+2\\"}"}}'
        )
        calls = parse_tool_calls_from_content(content)
        assert calls is not None
        assert calls[0]["name"] == "calculate"
        assert calls[0]["args"] == {"expression": "2+2"}

    def test_json_fence(self):
        content = '```json\n{"name": "check_timers", "parameters": {}}\n```'
        calls = parse_tool_calls_from_content(content)
        assert calls is not None
        assert calls[0]["name"] == "check_timers"

    def test_list_of_calls(self):
        content = (
            '[{"name": "get_current_time", "parameters": {}}, '
            '{"name": "check_timers", "parameters": {}}]'
        )
        calls = parse_tool_calls_from_content(content)
        assert calls is not None
        assert [c["name"] for c in calls] == ["get_current_time", "check_timers"]

    def test_plain_text_is_ignored(self):
        assert parse_tool_calls_from_content("It is 3 PM.") is None

    def test_invalid_json_is_ignored(self):
        assert parse_tool_calls_from_content('{"name": "broken"') is None

    def test_special_token_prefix(self):
        content = '<|python_tag|>{"name": "get_current_time", "parameters": {}}'
        calls = parse_tool_calls_from_content(content)
        assert calls is not None
        assert calls[0]["name"] == "get_current_time"

    def test_json_embedded_in_text(self):
        content = 'Here is the call: {"name": "calculate", "parameters": {"expression": "2+2"}}'
        calls = parse_tool_calls_from_content(content)
        assert calls is not None
        assert calls[0]["name"] == "calculate"
        assert calls[0]["args"] == {"expression": "2+2"}


class TestRescueParseAiMessage:
    def test_leaves_existing_tool_calls_untouched(self):
        original = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_current_time",
                    "args": {},
                    "id": "call_existing",
                    "type": "tool_call",
                }
            ],
        )
        assert rescue_parse_ai_message(original) is original

    def test_promotes_text_tool_call(self):
        original = AIMessage(content='{"name": "get_current_time", "parameters": {}}')
        rescued = rescue_parse_ai_message(original)
        assert rescued.content == ""
        assert len(rescued.tool_calls) == 1
        assert rescued.tool_calls[0]["name"] == "get_current_time"

    def test_unknown_shape_is_ignored(self):
        assert parse_tool_calls_from_content('{"foo": "bar"}') is None


class TestRescueParsingAgentIntegration:
    @staticmethod
    @tool
    def get_current_time() -> str:
        """Get the current time."""
        return "3:04 PM"

    def test_agent_executes_rescued_tool_call(self):
        from langchain.agents import create_agent
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        class RescueFakeChatModel(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                for generation in result.generations:
                    message = getattr(generation, "message", None)
                    if isinstance(message, AIMessage):
                        generation.message = rescue_parse_ai_message(message)
                return result

        llm = RescueFakeChatModel(
            messages=iter(
                [
                    AIMessage(content='{"name": "get_current_time", "parameters": {}}'),
                    AIMessage(content="It's 3:04 PM."),
                ]
            )
        )

        agent = create_agent(
            llm,
            [self.get_current_time],
            system_prompt="You are Rex.",
        )

        response = agent.invoke({"messages": [HumanMessage(content="What time is it?")]})

        tool_messages = [m for m in response["messages"] if m.__class__.__name__ == "ToolMessage"]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "3:04 PM"

        ai_messages = [m for m in response["messages"] if isinstance(m, AIMessage)]
        assert any(m.content and "3:04 PM" in m.content for m in ai_messages)
