import os
import json
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from literalai import LiteralClient
from langchain_chroma import Chroma
from typing import List, Dict, Any
import chainlit as cl

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=os.getenv("PERSIST_DIRECTORY"),
)

cl.instrument_openai()

literalai_client = LiteralClient(url=os.getenv("LITERAL_API_URL"), api_key=os.getenv("LITERAL_API_KEY"))
oai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt_path = os.path.join(os.getcwd(), "prompt.json")

with open(prompt_path, "r") as f:
    rag_prompt = json.load(f)

    prompt = literalai_client.api.get_or_create_prompt(
        name=rag_prompt["name"],
        template_messages=rag_prompt["template_messages"],
        settings=rag_prompt["settings"]
    )

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
async def rag_agent(question: str) -> str:
    """
    Coordinate the RAG agent flow to generate a response based on the user's question.
    """
    chunks = await get_relevant_chunks(question)
    answer = await call_llm(question, chunks)
    return answer

@cl.on_chat_start
async def on_chat_start():
    """
    Send a welcome message and set up the initial user session on chat start.
    """
    await cl.Message(
        content="Hello, How can I help you?",
    ).send()
    cl.user_session.set("settings", prompt.settings)
    cl.user_session.set("messages", prompt.format_messages())

@cl.on_message
async def main(message: cl.Message):
    """
    Main message handler for incoming user messages.
    """
    messages = cl.user_session.get("messages", [])
    answer_message = cl.Message(content="")
    cl.user_session.set("answer_message", answer_message)
    answer = await rag_agent(message.content)
    cl.user_session.set("messages", messages + [{"role": "assistant", "content": answer}])
