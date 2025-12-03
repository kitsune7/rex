from datetime import date

from dotenv import load_dotenv
from langchain.agents import create_agent as create_langchain_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import SecretStr

from .tools.reminder import CONFIRMABLE_TOOLS

load_dotenv()


def get_system_prompt() -> str:
    """Get the system prompt for the agent."""
    return f"""IMPORTANT: Your response will be spoken aloud. NEVER use markdown, asterisks, bullet points, or any formatting. Write naturally as if speaking.

You are Rex, a voice assistant. Today is {date.today()}. Be concise.
"""


def has_confirmable_tools(tools) -> bool:
    """Check if any of the provided tools require confirmation."""
    return any(t.name in CONFIRMABLE_TOOLS for t in tools)


def create_agent(tools, checkpointer=None):
    """
    Create a ReAct agent with the given tools.

    Args:
        tools: List of tools to make available to the agent
        checkpointer: Optional checkpointer for state persistence (required for interrupts)

    Returns:
        A compiled LangGraph agent
    """
    llm = ChatOpenAI(
        openai_api_base="http://localhost:1234/v1",
        api_key=SecretStr("not-needed"),
        temperature=0.7,
    )

    # If any tools require confirmation, interrupt before the "tools" node
    # We'll check which specific tool is being called at runtime
    use_interrupt = has_confirmable_tools(tools) and checkpointer is not None

    return create_langchain_agent(
        llm,
        tools,
        system_prompt=get_system_prompt(),
        checkpointer=checkpointer,
        interrupt_before=["tools"] if use_interrupt else None,
    )


def create_checkpointer() -> MemorySaver:
    """Create a memory checkpointer for interrupt support."""
    return MemorySaver()
