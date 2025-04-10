from dotenv import load_dotenv
import langfuse
from langfuse import Langfuse
import requests
import os
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall, SemanticSimilarity
from ragas import evaluate
import pandas as pd
from datasets import Dataset
from langfuse.decorators import observe, langfuse_context
import getpass
import argparse


load_dotenv()

username = getpass.getuser()


@observe()
def call_rag_api(question, api_url="http://localhost:8000/rag_agent"):
    """
    Make a single call to the RAG API endpoint and return the contexts and answer.
    
    :param question: The question to ask the RAG agent
    :param api_url: The URL of the API endpoint (default: "http://localhost:8000/rag_agent")
    :return: A tuple containing (answer, contexts)
    """
    
    payload = {"question": question}
    #print(question)

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["answer"], result["chunks"]
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the API: {e}")
        return None, None
    

def fetch_testset(testset_name):
    """Fetches test cases from Langfuse."""
    langfuse = Langfuse()
    
    testset = langfuse.get_dataset(testset_name)
    return testset

@observe()
def evaluate_llm_item(test, session_name, tag):
    """Runs LLM evaluation using test cases."""
    
    lf_trace_id = langfuse_context.get_current_trace_id()
    question = test.input
    expected_output = test.expected_output
    expected_chunk = test.metadata
    
    answer, chunks = call_rag_api(question)

    if answer is None:
        #continue
        print("KKOO")
        
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
        
    results = evaluate(Dataset.from_dict(data_samples), metrics=metrics).to_pandas()
        
    row = results.head(1)

    item_metrics = {
    "datasetItemId": test.id,
    "input": { "question": question },
    "output": { "answer": answer,
               "chunks": chunks },
    "scores": [{ 
        "name": m.name,
        "type": "AI",
        "value": row[m.name][0]
    } for m in metrics],
    }
    
    # Update langfuse trace details
    langfuse_context.update_current_observation(
        input= question,
        output= answer,
        metadata=chunks,
        session_id=session_name,
        tags=[tag],
        user_id=username,
    )
    
    return item_metrics, lf_trace_id


def log_results(item_metrics, lf_trace_id):
    """Logs evaluation results back to Langfuse."""
    
    langfuse = Langfuse()
    # Add scores to trace
    for score in item_metrics["scores"]:
        langfuse.score(
            trace_id=lf_trace_id,
            name=score["name"],
            value=score["value"],
        )
        
        
def main(LF_DATASET_NAME, EXPERIMENT_NAME, SESSION_NAME, TAG):
    
    langfuse = Langfuse()
    testset = fetch_testset(LF_DATASET_NAME)
    print(f"Found {len(testset.items)} rows in dataset {testset.name}!")
    
    for test_sample in testset.items:
        
        with test_sample.observe(
        run_name=EXPERIMENT_NAME,
        #run_description="My 3nd run",
        ) as trace_id:
            
            item_metrics, lf_trace_id = evaluate_llm_item(test_sample, SESSION_NAME, TAG)
            log_results(item_metrics, lf_trace_id)
        
        langfuse.flush()
    
    print("Evaluation completed and results logged.")

if __name__ == "__main__":
    # Initialisation du parser
    parser = argparse.ArgumentParser(description="Script avec arguments nommés")

    # Ajouter des arguments avec clé=valeur
    parser.add_argument("--LF_DATASET_NAME", type=str, help="Nom du testset sur Langfuse")
    parser.add_argument("--EXPERIMENT_NAME", type=str, help="Nom de l'experimentation")
    parser.add_argument("--SESSION_NAME", type=str, help="Nom de session")
    parser.add_argument("--TAG", type=str, help="Tag des traces generées")

    # Parser les arguments
    args = parser.parse_args()
    main(args.LF_DATASET_NAME, args.EXPERIMENT_NAME, args.SESSION_NAME, args.TAG)
