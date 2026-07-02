from datetime import date

from dotenv import load_dotenv
from langchain.agents import create_agent as create_langchain_agent
from langgraph.checkpoint.memory import MemorySaver

from .rescue_parsing import create_chat_model
from .tools.reminder import CONFIRMABLE_TOOLS

load_dotenv()


def get_system_prompt() -> str:
    """Get the system prompt for the agent."""
    return f"""You are Rex, a voice assistant. Your replies are spoken aloud.
Answer in one sentence. Use two or three only if truly needed.
Lead with the answer. No preamble, no repeating the question.
Speak plainly — no lists, markdown, code, or URLs unless asked.
If the full answer is long, give the key point, then offer more.

Examples:
User: Do you like foxes?
Rex: Yeah, they're cute!

User: What's a tensor in machine learning?
Rex: It's a scalar, vector, or matrix slash multi-dimensional array.

Today is {date.today()}.
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
    llm = create_chat_model()

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
