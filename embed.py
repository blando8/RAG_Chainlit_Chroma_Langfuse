import os
from enum import Enum
import time
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

from dotenv import load_dotenv
load_dotenv()

class ChunkingMethod(Enum):
    PAGE = "page"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"

def chunk_pdf(input_path: str, method: ChunkingMethod) -> List[Document]:
    def load_docs(path: str) -> List[Document]:
        docs = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".pdf"):
                    docs.extend(PyPDFLoader(os.path.join(root, file)).load())
        return docs

    docs = load_docs(input_path)

    if method == ChunkingMethod.PAGE:
        return docs
    elif method == ChunkingMethod.RECURSIVE:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    elif method == ChunkingMethod.SEMANTIC:
        text_splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small")) # type: ignore

    return text_splitter.split_documents(docs)

def embed(documents, persist_directory):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=persist_directory)
    
    return vectorstore


if __name__ == "__main__":
    docs = chunk_pdf("./docs", ChunkingMethod.PAGE)#ChunkingMethod.RECURSIVE
    start_time = time.time()
    vector_store = embed(documents=docs, persist_directory=os.getenv("PERSIST_DIRECTORY"))

    print(f"Processed {len(docs)} documents in {time.time() - start_time:.2f} seconds.")