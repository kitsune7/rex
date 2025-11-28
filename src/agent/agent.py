from dataclasses import dataclass
import uuid

from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from .base_agent import create_agent, create_checkpointer
from .tools import (
    get_current_time,
    calculate,
    set_timer,
    check_timers,
    stop_timer,
    create_reminder,
    list_reminders,
    update_reminder,
    delete_reminder,
    tool_requires_confirmation,
)
from langfuse.langchain import CallbackHandler

# Maximum number of messages to keep in history to prevent unbounded growth
MAX_HISTORY_MESSAGES = 20

# Cached agent instance and checkpointer
_agent = None
_checkpointer = None


def _get_agent():
    """Get or create the cached agent instance with checkpointer."""
    global _agent, _checkpointer
    if _agent is None:
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
        _checkpointer = create_checkpointer()
        _agent = create_agent(tools, checkpointer=_checkpointer)
    return _agent, _checkpointer


@dataclass
class PendingConfirmation:
    """Represents a tool call waiting for user confirmation."""

    tool_name: str
    tool_args: dict
    confirmation_prompt: str
    thread_id: str


def _format_confirmation_prompt(tool_name: str, tool_args: dict) -> str:
    """Format a human-readable confirmation prompt for a tool call."""
    if tool_name == "create_reminder":
        message = tool_args.get("message", "")
        datetime_str = tool_args.get("datetime_str", "")
        return f"I'm about to create a reminder: '{message}' for {datetime_str}. Should I proceed?"
    else:
        # Generic confirmation for other tools
        args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
        return f"I'm about to run {tool_name}({args_str}). Should I proceed?"


def extract_text_response(response):
    """Extract the text response from an agent response."""
    # Handle non-dict responses
    if not hasattr(response, "get") or "messages" not in response:
        return str(response)

    agent_messages = response["messages"]
    if not agent_messages:
        return "[No response generated]"

    last_message = agent_messages[-1]
    if not hasattr(last_message, "content"):
        return str(last_message)

    content = last_message.content

    # Handle string content directly
    if isinstance(content, str):
        return content

    # Handle list content (e.g., list of dicts with 'text' field)
    if isinstance(content, list) and len(content) > 0:
        if isinstance(content[0], dict) and "text" in content[0]:
            return content[0]["text"]
        return str(content[0])

    return str(content)


def _extract_pending_tool_call(response) -> tuple[str, dict] | None:
    """Extract pending tool call info from agent state after interrupt."""
    messages = response.get("messages", [])
    if not messages:
        return None

    # Look for the last AI message with tool calls
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call = msg.tool_calls[0]  # Get the first pending tool call
            return tool_call["name"], tool_call["args"]

    return None


def run_voice_agent(
    query: str,
    history: list | None = None,
    thread_id: str | None = None,
) -> tuple[str | PendingConfirmation, list, str]:
    """
    Run the voice agent with a user query.

    Args:
        query: The user's message/question.
        history: Optional list of previous messages for conversation context.
                 If None, starts a new conversation.
        thread_id: Optional thread ID for interrupt/resume support.

    Returns:
        A tuple of (response, updated_history, thread_id) where:
        - response is either a string (final response) or PendingConfirmation (needs user confirmation)
        - updated_history includes the new user message and agent response
        - thread_id is the thread ID for potential continuation
    """
    agent, checkpointer = _get_agent()
    langfuse_handler = CallbackHandler()

    # Generate thread ID if not provided
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {
        "callbacks": [langfuse_handler],
        "configurable": {"thread_id": thread_id},
    }

    # Initialize or use existing history
    if history is None:
        history = []

    # Add user message to history
    history.append(HumanMessage(content=query))

    # Invoke agent with full conversation history
    response = agent.invoke({"messages": history}, config=config)

    # Check if we hit an interrupt (paused before tools node)
    state = agent.get_state(config)
    while state.next:  # There's a next step, meaning we're paused at an interrupt
        tool_info = _extract_pending_tool_call(response)
        if tool_info:
            tool_name, tool_args = tool_info

            # Check if this specific tool requires confirmation
            if tool_requires_confirmation(tool_name):
                confirmation_prompt = _format_confirmation_prompt(tool_name, tool_args)
                return (
                    PendingConfirmation(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        confirmation_prompt=confirmation_prompt,
                        thread_id=thread_id,
                    ),
                    history,
                    thread_id,
                )
            else:
                # Tool doesn't need confirmation, continue execution
                response = agent.invoke(None, config=config)
                state = agent.get_state(config)
                continue

        # No tool info found, break out of loop
        break

    # Extract text response
    text_response = extract_text_response(response)

    # Add agent response to history
    history.append(AIMessage(content=text_response))

    # Trim history if it gets too long (keep most recent messages)
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    return text_response, history, thread_id


def confirm_tool_call(
    pending: PendingConfirmation,
    confirmed: bool,
    user_response: str | None = None,
) -> tuple[str, list]:
    """
    Confirm or reject a pending tool call.

    Args:
        pending: The PendingConfirmation object from run_voice_agent
        confirmed: True to proceed with the tool call, False to cancel
        user_response: Optional user response text (used for modification requests)

    Returns:
        A tuple of (response_text, updated_history)
    """
    agent, checkpointer = _get_agent()
    langfuse_handler = CallbackHandler()

    config = {
        "callbacks": [langfuse_handler],
        "configurable": {"thread_id": pending.thread_id},
    }

    if confirmed:
        # Resume execution - the tool will run
        response = agent.invoke(Command(resume=True), config=config)
        text_response = extract_text_response(response)
    else:
        # Cancel - inject a message saying the user cancelled
        # We need to get the current state and add a cancellation
        state = agent.get_state(config)
        messages = list(state.values.get("messages", []))

        # Find the tool call that was pending and create a rejection response
        for msg in reversed(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_call = msg.tool_calls[0]
                # Add a tool message indicating cancellation
                # Include user's modification request if provided
                if user_response:
                    cancel_content = (
                        f"User declined this action and requested a modification: '{user_response}'. "
                        "Please propose the action again with the updated parameters."
                    )
                else:
                    cancel_content = "User cancelled this action."
                cancel_msg = ToolMessage(
                    content=cancel_content,
                    tool_call_id=tool_call["id"],
                )
                messages.append(cancel_msg)
                break

        # Update state with the cancellation and continue
        agent.update_state(config, {"messages": messages})

        # Don't invoke the agent here - just return the cancellation
        # The caller will continue the conversation if needed, which will
        # trigger a fresh agent invocation through run_voice_agent
        text_response = None

        if not text_response or text_response == "[No response generated]":
            text_response = "Okay, I've cancelled that action."

    # Get updated history from state
    final_state = agent.get_state(config)
    history = list(final_state.values.get("messages", []))

    # Trim history if needed
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    return text_response, history
