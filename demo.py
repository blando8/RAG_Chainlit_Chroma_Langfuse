import os
import json
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from literalai import LiteralClient
from langchain_chroma import Chroma
from typing import List, Dict, Any
import chainlit as cl
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

from dotenv import load_dotenv
load_dotenv()

# Initialize Langfuse client
langfuse = Langfuse()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=os.getenv("PERSIST_DIRECTORY"),
)

cl.instrument_openai()

#literalai_client = LiteralClient(url=os.getenv("LITERAL_API_URL"), api_key=os.getenv("LITERAL_API_KEY"))
oai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#prompt_path = os.path.join(os.getcwd(), "prompt.json")

#with open(prompt_path, "r") as f:
#    rag_prompt = json.load(f)

#    prompt = literalai_client.api.get_or_create_prompt(
#        name=rag_prompt["name"],
#        template_messages=rag_prompt["template_messages"],
#        settings=rag_prompt["settings"]
#    )

# Get current `production` version of a text prompt
prompt = langfuse.get_prompt("RAG prompt")
compiled_prompt = prompt.compile()



#@observe(as_type="generation")
async def call_llm(question: str, chunks: List[Dict[str, Any]]):
    messages = cl.user_session.get("messages", [])
    
    context = "\n".join([f"{doc['page_content']}" for doc in chunks])
    messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"})

    stream = await oai_client.chat.completions.create(model="gpt-4o-mini", messages=messages, stream=True)
    
    answer_message = cl.user_session.get("answer_message")
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await answer_message.stream_token(token)
    
    await answer_message.update()
    
    return answer_message.content


@observe(as_type="generation")
async def call_llmChatBot(question: str, chunks: List[Dict[str, Any]]):
    messages = cl.user_session.get("messages", [])
    
    context = "\n".join([f"{doc['page_content']}" for doc in chunks])
    messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"})

    response = await oai_client.chat.completions.create(model="gpt-4o-mini", messages=messages, stream=False)
    
    langfuse_context.update_current_observation(
        usage_details = {
        "input": response.usage.prompt_tokens,  # maps to "input_tokens"
        "output": response.usage.completion_tokens,  # maps to "output_tokens"
        "cache_read_input_tokens": response.usage.prompt_tokens_details.cached_tokens,
        },
        model="gpt-4o-mini"
    )
    # Pricing per 1M tokens (cf platform OpenAI)
    input_cost_per_1M = 0.15
    output_cost_per_1M = 0.60
    cached_cost_per_1M = 0.075

    cost_details={
          "input": response.usage.prompt_tokens / 1000000 * input_cost_per_1M,
          "cache_read_input_tokens": response.usage.prompt_tokens_details.cached_tokens / 1000000 * cached_cost_per_1M,
          "output": response.usage.completion_tokens / 1000000 * output_cost_per_1M,
      }
    # Add total
    cost_details["total"] = (
    cost_details["input"]
    + cost_details["output"]
    + cost_details["cache_read_input_tokens"]
    )

    # Update Langfuse
    langfuse_context.update_current_observation(
        cost_details=cost_details,
        model="gpt-4o-mini"
    )
    
    return response.choices[0].message.content


async def retrieve(question: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Retrieve top_k closest vectors from the Chroma index using the provided question.
    """
    retriever = vector_store.as_retriever()
        # search_type="similarity_score_threshold",
        # search_kwargs={"score_threshold": 0.5})
    retrieved_docs = retriever.invoke(question)
    return [doc.dict() for doc in retrieved_docs]

@cl.step(type="retrieval", name="Get Relevant Chunks")
async def get_relevant_chunks(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from dataset based on the question.
    """
    return await retrieve(question, top_k)

@cl.step(type="run", name="RAG Agent")


@observe
async def rag_agentChatBot(question: str) -> str:
    """
    Coordinate the RAG agent flow to generate a response based on the user's question.
    """
    chunks = await get_relevant_chunks(question)
    answer = await call_llm(question, chunks)
    ans = await call_llmChatBot(question, chunks)
    
    return answer

@cl.on_chat_start
async def on_chat_start():
    """
    Send a welcome message and set up the initial user session on chat start.
    """
    await cl.Message(
        content="Hello, How can I help you?",
    ).send()
    #cl.user_session.set("settings", prompt.settings)
    #cl.user_session.set("messages", prompt.format_messages())
    cl.user_session.set("settings", prompt.config)
    cl.user_session.set("messages", [{'role': 'system', 'content':compiled_prompt}])

@cl.on_message
async def main(message: cl.Message):
    """
    Main message handler for incoming user messages.
    """
    messages = cl.user_session.get("messages", [])
    answer_message = cl.Message(content="")
    cl.user_session.set("answer_message", answer_message)
    answer = await rag_agentChatBot(message.content)
    cl.user_session.set("messages", messages + [{"role": "assistant", "content": answer}])
    