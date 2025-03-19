import os, time


from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from embed import ChunkingMethod, chunk_pdf

from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator


start_time = time.time()

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")))
generator_llm = evaluator_llm 

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings)


docs = chunk_pdf("./docs/", ChunkingMethod.RECURSIVE)
print("Done chunking")

testset = generator.generate_with_langchain_docs(docs, testset_size=10)
test_df = testset.to_pandas()

test_df.to_csv('testset.csv', index=False)

print(f"Processed in {time.time() - start_time:.2f} seconds.")
