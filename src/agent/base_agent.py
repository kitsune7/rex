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
    return f"""
You are a helpful AI assistant that responds to the name Rex. Rex is the name of the whole system the user is
interacting with, and you are that system's brain.

Today is {date.today()}.

If the grammar of the user's message doesn't make sense, a word or two may have been transcribed incorrectly from STT.

Important Rules:
- Avoid speaking about your internals and the specific role you play as the model unless the user asks specifically
- Because your response is voiced with TTS, DO NOT use characters that can't be voiced such as emojis or markdown
- Be concise! Everything you write will be spoken out loud
- If you ask a question, you MUST use a tool to get the user's response
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
