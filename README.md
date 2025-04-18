# RAG using Chainlit - Chroma - Langfuse

This repo shows how to evaluate a RAG:

1. Set up virtual env to run code in (and install dependencies)

    - For Mac & Unix
    ```shell
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    ```
    
    - For Window
    ```shell
    python -m venv .venv && .\.venv\Scripts\activate
    pip install -r requirements.txt
    ```
    
2. Downloads arXiv PDFs and store them in the `docs` folder

    ```shell
    python download_pdf.py
    ```
3. Configure OpenAI and Langfuse by setting environment variables. Create a `.env` file and populate it with your OpenAI and Langfuse details (see `.env.example` for an example).

4. Chunk the PDF files and store them in the `chunks` folder

    ```shell
    python embed.py
    ```
    
5. Add your prompt to Langfuse

    ```shell
    python addPromptToLangfuse.py
    ```

6. Start a chat

    ```shell
    chainlit run demo.py
    ```

7. Create a dataset of questions and answers from your interactions
    
    - Launch the API endpoint
    
    ```shell
    python test.py
    ```
    
    - Create a `.csv` file that contains Questions/Answers/Context to test your RAG system
    
    ```shell
    python generator.py
    ```
    
8. Eval: Create an eval dataset and run eval

    - Create the eval dataset in Langfuse

      ```shell
      python create_dataset.py --LF_DATASET_NAME="eval-dataset"
      ```

    - Run eval

      ```shell
      python run_eval.py --LF_DATASET_NAME="eval-dataset" --EXPERIMENT_NAME="exp V1" --SESSION_NAME="session 1" --TAG="tag-1"
      ```
9. Inspect all Traces,Datasets, and Dataset Runs in the Langfuse Dashboard (at `https://cloud.langfuse.com/`)