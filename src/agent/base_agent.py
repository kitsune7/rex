from datetime import date

from dotenv import load_dotenv
from langchain.agents import create_agent as create_langchain_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()


def create_agent(tools):
    llm = ChatOpenAI(
        openai_api_base="http://localhost:1234/v1",
        api_key=SecretStr("not-needed"),
        temperature=0.7,
    )

    system_prompt = f"""
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

    return create_langchain_agent(llm, tools, system_prompt=system_prompt)
