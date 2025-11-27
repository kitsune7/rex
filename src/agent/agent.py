from .base_agent import create_agent
from .tools import get_current_time, calculate, set_timer, check_timers, stop_timer
from langchain_core.messages import HumanMessage, AIMessage
from langfuse.langchain import CallbackHandler

# Maximum number of messages to keep in history to prevent unbounded growth
MAX_HISTORY_MESSAGES = 20

# Cached agent instance (stateless - all state comes from messages)
_agent = None


def _get_agent():
    """Get or create the cached agent instance."""
    global _agent
    if _agent is None:
        tools = [get_current_time, calculate, set_timer, check_timers, stop_timer]
        _agent = create_agent(tools)
    return _agent


def extract_text_response(response):
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


def run_voice_agent(query: str, history: list | None = None) -> tuple[str, list]:
    """
    Run the voice agent with a user query.

    Args:
        query: The user's message/question.
        history: Optional list of previous messages for conversation context.
                 If None, starts a new conversation.

    Returns:
        A tuple of (response_text, updated_history) where updated_history
        includes the new user message and agent response.
    """
    agent = _get_agent()
    langfuse_handler = CallbackHandler()

    # Initialize or use existing history
    if history is None:
        history = []

    # Add user message to history
    history.append(HumanMessage(content=query))

    # Invoke agent with full conversation history
    response = agent.invoke({"messages": history}, config={"callbacks": [langfuse_handler]})

    # Extract text response
    text_response = extract_text_response(response)

    # Add agent response to history
    history.append(AIMessage(content=text_response))

    # Trim history if it gets too long (keep most recent messages)
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    return text_response, history
