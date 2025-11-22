from typing import TypedDict, cast

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import SecretStr
from tts import load_voice, speak_text


class QueryState(TypedDict):
    """State for the query graph."""

    user_query: str
    system_prompt: str
    response: str


def llm_node(state: QueryState) -> QueryState:
    """Call the LM Studio model and get the response."""
    llm = ChatOpenAI(
        openai_api_base="http://localhost:1234/v1",
        api_key=SecretStr("not-needed"),
        temperature=0.7,
    )

    messages = [("system", state["system_prompt"]), ("user", state["user_query"])]

    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)

    return cast(
        QueryState,
        {
            "user_query": state["user_query"],
            "system_prompt": state["system_prompt"],
            "response": content,
        },
    )


def query(
    text: str,
    system_prompt: str,
) -> None:
    """
    Query the LM Studio model with the given text and print the response.

    Args:
        text: The user's query text
        system_prompt: The system prompt to set context for the model
    """
    voice = load_voice("joe")

    workflow = StateGraph(QueryState)
    workflow.add_node("llm", llm_node)
    workflow.set_entry_point("llm")
    workflow.add_edge("llm", END)

    app = workflow.compile()

    initial_state = {"user_query": text, "system_prompt": system_prompt, "response": ""}
    result = app.invoke(initial_state)

    print(f"\nRex: {result['response']}")
    speak_text(result["response"], voice)
