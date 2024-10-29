from dotenv import load_dotenv
from literalai import LiteralClient
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import ContextUtilization, Faithfulness, ContextPrecision, ContextRecall, SemanticSimilarity

load_dotenv()

literalai_client = LiteralClient()

literalai_client.instrument_openai()

import requests

def call_rag_api(question, api_url="http://localhost:8000/rag_agent"):
    """
    Make a single call to the RAG API endpoint and return the contexts and answer.
    
    :param question: The question to ask the RAG agent
    :param api_url: The URL of the API endpoint (default: "http://localhost:8000/rag_agent")
    :return: A tuple containing (answer, contexts)
    """
    # Prepare the request payload
    payload = {"question": question}

    try:
        # Make the POST request to the API
        response = requests.post(api_url, json=payload)
        
        # Check if the request was successful
        response.raise_for_status()

        # Parse the JSON response
        result = response.json()

        # Extract the answer and contexts (chunks)
        answer = result["answer"]
        contexts = result["chunks"]

        return answer, contexts

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the API: {e}")
        return None, None

@literalai_client.experiment_item_run
def run_experiment_item(experiment, item):
    
    question = item.input["question"]
    expected_output = item.expected_output["ground_truth"]
    
    answer, chunks = call_rag_api(question)
    
    data_samples = {
        'question': [question],
        'answer': [answer],
        'contexts': [[chunk["page_content"] for chunk in chunks]],
        'ground_truth': [expected_output]
    }

    metrics = [
        Faithfulness(name="Faithfulness"),
        ContextPrecision(name="Context Precision"),
        ContextRecall(name="Context Recall"), 
        SemanticSimilarity(name="Semantic Similarity")
    ]

    with literalai_client.step(type="run", name="Ragas"):
        results = evaluate(Dataset.from_dict(data_samples), metrics=metrics).to_pandas()
    
    row = results.head(1)

    experiment_item = {
        "datasetItemId": item.id,
        "input": { "question": question },
        "output": { "answer": answer,
                   "chunks": chunks },
        "scores": [{ 
            "name": m.name,
            "type": "AI",
            "value": row[m.name][0]
        } for m in metrics],
    }

    experiment.log(experiment_item)



ds = literalai_client.api.get_dataset(name="Testset")
print(f"Found {len(ds.items)} rows in dataset {ds.name} !") # type: ignore

experiment = ds.create_experiment(name="Top k = 10") # type: ignore

for item in ds.items: # type: ignore
    run_experiment_item(experiment, item)
