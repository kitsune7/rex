from .base_agent import create_agent
from .tools import get_human_text_response, ask_human_for_voice_response
from langchain_core.messages import HumanMessage


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


def run_voice_agent(initial_query: str):
    tools = [ask_human_for_voice_response]
    agent = create_agent(tools)
    response = agent.invoke({"messages": [HumanMessage(content=initial_query)]})
    return extract_text_response(response)


def run_text_agent(initial_query: str):
    tools = [get_human_text_response]
    agent = create_agent(tools)
    response = agent.invoke({"messages": [HumanMessage(content=initial_query)]})
    return extract_text_response(response)
