"""
Exercise 1 - a first LangChain call to Claude.

Run:  python -m src.exercises.hello_llm   (from the project root)

Compared with llm_file_assistant.py from Milestone 1, LangChain hides the client setup,
the request format and the response parsing behind one .invoke() call that returns a message object.
"""

import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.config.constants import LLM_MODEL

load_dotenv()

llm = ChatAnthropic(model=LLM_MODEL)

start = time.time()
reply = llm.invoke("In one sentence, what is a resume?")
print(reply.text)
print(f"({time.time() - start:.1f}s, type: {type(reply).__name__})")
