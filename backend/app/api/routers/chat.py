import json
import time
from queue import Queue
from typing import List
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, HTTPException, Request, status
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.utilities.token_counting import TokenCounter
from app.engine import get_chat_engine
from app.engine.workflow import ChatEngine, ProgressEvent

chat_router = r = APIRouter()


class _Message(BaseModel):
    role: MessageRole
    content: str


class _ChatData(BaseModel):
    messages: List[_Message]


stream_part_types = {
    "text": "0",
    "function_call": "1",
    "data": "2",
    "error": "3",
    "assistant_message": "4",
    "assistant_data_stream_part": "5",
    "data_stream_part": "6",
    "message_annotations_stream_part": "7",
}

@r.post("")
async def chat(
    request: Request,
    data: _ChatData,
    chat_engine: ChatEngine = Depends(get_chat_engine),
):
    # check preconditions and get last message
    if len(data.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided",
        )
    lastMessage = data.messages.pop()
    if lastMessage.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from user",
        )
    # convert messages coming from the request to type ChatMessage
    messages = [
        ChatMessage(
            role=m.role,
            content=m.content,
        )
        for m in data.messages
    ]

    # query chat engine
    handler = chat_engine.run(query=lastMessage.content, messages=messages or [])

    # stream response
    async def event_generator():
        async for event in handler.stream_events():
            if await request.is_disconnected():
                break

            if not isinstance(event, ProgressEvent):
                continue
                
            if event.type == "text":
                yield f"{stream_part_types[event.type]}:{json.dumps(event.message)}\n"
            else:
                yield f"{stream_part_types[event.type]}:{json.dumps([event.model_dump()])}\n"
        
        # get final result
        final_result = await handler
        messages = final_result["messages"]
        response = final_result["response"]
        
        # count tokens
        token_counter = TokenCounter()
        input_tokens = token_counter.estimate_tokens_in_messages(messages)
        output_tokens = token_counter.get_string_tokens(response)
        message = f"Input tokens: {input_tokens}, output tokens: {output_tokens}"

        # send final result
        stream_part_type = stream_part_types["data"]
        yield f"{stream_part_type}:{message}\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Experimental-Stream-Data": "true"
        }
    )
