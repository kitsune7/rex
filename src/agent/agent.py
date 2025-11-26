from .base_agent import create_agent
from .tools import get_current_time, calculate, set_timer, check_timers, stop_timer
from langchain_core.messages import HumanMessage, AIMessage
from langfuse.langchain import CallbackHandler

# Maximum number of messages to keep in history to prevent unbounded growth
MAX_HISTORY_MESSAGES = 20


def extract_text_response(response):
    if hasattr(response, "get") and "messages" in response:
        agent_messages = response["messages"]
        if agent_messages:
            last_message = agent_messages[-1]
            if hasattr(last_message, "content"):
                content = last_message.content

                # Handle the case where content is a list of dictionaries with 'text' field
                if isinstance(content, list) and len(content) > 0:
                    # Extract just the text content, ignoring extras
                    if isinstance(content[0], dict) and "text" in content[0]:
                        text_response = content[0]["text"]
                    else:
                        text_response = str(content[0])
                elif isinstance(content, str):
                    text_response = content
                else:
                    text_response = str(content)
            else:
                text_response = str(last_message)
        else:
            text_response = "[No response generated]"
    else:
        text_response = str(response)

    return text_response


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
    tools = [get_current_time, calculate, set_timer, check_timers, stop_timer]
    agent = create_agent(tools)
    langfuse_handler = CallbackHandler()

    # Initialize or use existing history
    if history is None:
        history = []

    # Add user message to history
    history.append(HumanMessage(content=query))

    # Invoke agent with full conversation history
    response = agent.invoke(
        {"messages": history}, config={"callbacks": [langfuse_handler]}
    )

    # Extract text response
    text_response = extract_text_response(response)

    # Add agent response to history
    history.append(AIMessage(content=text_response))

    # Trim history if it gets too long (keep most recent messages)
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    return text_response, history
