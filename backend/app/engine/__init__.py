import os
from app.engine.index import get_index
from app.engine.workflow import ChatEngine


def get_chat_engine() -> ChatEngine:
    system_prompt = os.getenv("SYSTEM_PROMPT")
    context_prompt = os.getenv("CONTEXT_PROMPT")
    top_k = os.getenv("TOP_K", 3)
    index = get_index()

    return ChatEngine(
        index=index,
        top_k=int(top_k),
        system_prompt=system_prompt,
        context_prompt=context_prompt,
    )
