import warnings
warnings.filterwarnings("ignore")

from typing import List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

import config


def create_memory() -> InMemoryChatMessageHistory:
    return InMemoryChatMessageHistory()


def add_exchange(memory: InMemoryChatMessageHistory,question: str,answer: str,) -> None:
    memory.add_user_message(question)
    memory.add_ai_message(answer)


def get_recent_messages(memory: InMemoryChatMessageHistory) -> List[BaseMessage]:
    return memory.messages[-(config.MEMORY_WINDOW * 2):]


def format_history(memory: InMemoryChatMessageHistory) -> str:
    lines = []
    for message in get_recent_messages(memory):
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            # Long answers are trimmed: the rewriter only needs the topic,
            # not the full text of a previous answer.
            lines.append(f"Assistant: {message.content[:300]}")
    return "\n".join(lines)


def clear_memory(memory: InMemoryChatMessageHistory) -> None:
    memory.clear()