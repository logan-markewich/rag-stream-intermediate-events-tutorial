import os
from typing import Literal

from llama_index.core import Settings
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step


class RetrieveEvent(Event):
    query: str

class PostProcessEvent(Event):
    query: str
    nodes: list[NodeWithScore]

class SynthesizeEvent(Event):
    query: str
    nodes: list[NodeWithScore]

class ProgressEvent(Event):
    type: Literal["data", "text"]
    status: Literal["loading", "done"]
    message: str


CONTEXT_PROMPT = (
    "Here is some extra context from a knowledge base that may help you assist the user with their latest message.\n"
    "If you don't know the answer, just say so. Don't try to make up an answer."
    "\n-----\n{context}\n-----\n"
    "Latest message: {message}"
)


class ChatEngine(Workflow):

    def __init__(
        self, 
        index: BaseIndex, 
        top_k: int,
        system_prompt: str | None = None, 
        context_prompt: str | None = None,
    ) -> None:
        self.index = index
        self.system_prompt = system_prompt
        self.top_k = top_k
        self.context_prompt = context_prompt or CONTEXT_PROMPT

        super().__init__(timeout=None, verbose=False)
    
    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> RetrieveEvent:
        query = ev.get("query")
        messages = ev.get("messages")
        if query is None or messages is None:
            raise ValueError("Query and messages are required!")
        
        memory = ChatMemoryBuffer.from_defaults(llm=Settings.llm, chat_history=messages)
        await ctx.set("memory", memory)

        return RetrieveEvent(query=query)
    
    @step
    async def retrieve(self, ctx: Context, ev: RetrieveEvent) -> PostProcessEvent:
        ctx.write_event_to_stream(ProgressEvent(
            status="loading",
            message="Retrieving relevant nodes...",
            type="data",
        ))

        query = ev.query
        retirever = self.index.as_retriever(
            similarity_top_k=self.top_k,
            embed_model=Settings.embed_model,
        )

        nodes = await retirever.aretrieve(query)

        ctx.write_event_to_stream(ProgressEvent(
            status="done",
            message=f"Retrieved {len(nodes)} relevant nodes for context.",
            type="data",
        ))

        return PostProcessEvent(query=query, nodes=nodes)
    
    @step
    async def post_process(self, ctx: Context, ev: PostProcessEvent) -> SynthesizeEvent:
        
        # ctx.write_event_to_stream(ProgressEvent(
        #     status="loading",
        #     message="Post processing nodes...",
        #     type="data",
        # ))

        # TODO: implement your own post processing here!

        # ctx.write_event_to_stream(ProgressEvent(
        #    status="done",
        #    message="Post processed nodes.",
        #    type="data",
        #))

        return SynthesizeEvent(query=ev.query, nodes=ev.nodes)
    
    @step
    async def synthesize(self, ctx: Context, ev: SynthesizeEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(
            status="loading",
            message="Synthesizing response...",
            type="data",
        ))

        llm = Settings.llm
        
        # get chat history
        memory = await ctx.get("memory")
        chat_history = memory.get(ev.query)
        
        # add context nodes + latest user message to chat history
        context_str = "\n\n".join([n.node.get_content(metadata_mode="none") for n in ev.nodes])
        context_query = self.context_prompt.format(
            context=context_str,
            message=ev.query,
        )
        chat_history.append(ChatMessage(role="user", content=context_query))

        # add system prompt if it exists
        if self.system_prompt:
            chat_history = [ChatMessage(role="system", content=self.system_prompt)] + chat_history

        # stream response
        response_gen = await llm.astream_chat(chat_history)

        ctx.write_event_to_stream(ProgressEvent(
            status="done",
            message="Finished creating response stream.",
            type="data",
        ))

        async for chunk in response_gen:
            ctx.write_event_to_stream(ProgressEvent(
                status="loading",
                message=chunk.delta,
                type="text",
            ))

        return StopEvent(result={
            "query": ev.query,
            "messages": chat_history,
            "response": chunk.message.content,
            "source_nodes": ev.nodes,
        })
