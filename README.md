# RAG using Chainlit - Chroma - Literal AI (demo @Datacraft)

This repo shows how to evaluate a RAG:

- pip install chainlit literalai
- put your PDF files in the `docs` folder
- `cp .env.example .env` and set your OpenAI and Literal AI API keys
- `python embed.py` to chunk the PDF files and store them in the `chunks` folder
- `chainlit run demo.py` to start a chat
- create a dataset of questions and answers from your interactions
- `python run_experiment.py` to evaluate the performance of the RAG chatbot
- visualize the results from Literal AI

The `docs.tar.gz` file has size > 100 MB, use git lfs to pull. 
Then `tar -zxvf docs.tar.gz` to chunk these docs (public files about sustainability).
