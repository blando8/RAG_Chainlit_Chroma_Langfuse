import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from literalai import LiteralClient
from langchain_chroma import Chroma
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=os.getenv("PERSIST_DIRECTORY"),
)

literalai_client = LiteralClient(url=os.getenv("LITERAL_API_URL"), api_key=os.getenv("LITERAL_API_KEY"))
oai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt_path = os.path.join(os.getcwd(), "prompt.json")

# Load the RAG prompt
with open(prompt_path, "r") as f:
    rag_prompt = json.load(f)

    prompt = literalai_client.api.get_or_create_prompt(
        name=rag_prompt["name"],
        template_messages=rag_prompt["template_messages"],
        settings=rag_prompt["settings"]
    )

async def call_llm(question: str, chunks: List[Dict[str, Any]], stream: bool = False, messages = []):
    if messages is None:
        messages = prompt.format_messages()
    
    context = "\n".join([f"{doc['page_content']}" for doc in chunks])
    messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"})

    response = await oai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content

async def retrieve(question: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Retrieve top_k closest vectors from the Chroma index using the provided question.
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    retrieved_docs = retriever.invoke(question)
    return [doc.dict() for doc in retrieved_docs]

async def get_relevant_chunks(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from dataset based on the question.
    """
    return await retrieve(question, top_k)

async def rag_agent(question: str, stream: bool = False, messages: List[Dict[str, str]] = None) -> tuple:
    """
    Coordinate the RAG agent flow to generate a response based on the user's question.
    """
    chunks = await get_relevant_chunks(question)
    answer = await call_llm(question, chunks, stream, messages)
    return answer, chunks

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

# API endpoint
@app.post("/rag_agent")
async def api_rag_agent(request: QuestionRequest):
    try:
        answer, chunks = await rag_agent(request.question)
        return {"answer": answer, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Running as FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
