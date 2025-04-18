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
from langfuse.decorators import observe, langfuse_context

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

#literalai_client = LiteralClient(url=os.getenv("LITERAL_API_URL"), api_key=os.getenv("LITERAL_API_KEY"))
oai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#prompt_path = os.path.join(os.getcwd(), "prompt.json")

# Load the RAG prompt
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

@observe(as_type="generation")#as_type="generation"
async def call_llm(question: str, chunks: List[Dict[str, Any]], stream: bool = False, messages = []):
    if messages is None:
        messages = [{'role': 'system', 'content':compiled_prompt}] #prompt.format_messages()
    
    context = "\n".join([f"{doc['page_content']}" for doc in chunks])
    messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"})

    response = await oai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    
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

@observe()
async def retrieve(question: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Retrieve top_k closest vectors from the Chroma index using the provided question.
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})#search_kwargs={"k": 10}
    retrieved_docs = retriever.invoke(question)
    return [doc.dict() for doc in retrieved_docs]

@observe()
async def get_relevant_chunks(question: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from dataset based on the question.
    """
    return await retrieve(question, top_k)

@observe()
async def rag_agent(question: str, stream: bool = False, messages: List[Dict[str, str]] = None) -> tuple:
    """
    Coordinate the RAG agent flow to generate a response based on the user's question.
    """
    chunks = await get_relevant_chunks(question, top_k=3)
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
