# Corrective RAG Demo

An implementation of Corrective Retrieval-Augmented Generation (RAG) using LangChain, Azure OpenAI, Azure Cognitive Search, and Azure Blob Storage.

## Features

- Document uploading via Azure Blob Storage
- Document indexing using Azure Cognitive Search
- Intelligent retrieval and query optimization
- Web search fallback using DuckDuckGo
- Advanced generation with document grading
- Interactive Streamlit interface

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/Tak0003/agent-test.git
   cd agent-test
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create `.env` file from the template:
   ```
   cp .env.template .env
   ```

4. Fill in your API keys and configuration in the `.env` file.

5. Run the application:
   ```
   streamlit run app.py
   ```

## Environment Variables

Set up the following environment variables in your `.env` file:

- Azure OpenAI: API keys and endpoints for your Azure OpenAI resources
- Azure Blob Storage: Connection string and container name
- Azure AI Search: Endpoint, API key, and index name
- LangSmith: API key and project configuration for tracing

## Architecture

This demo implements the Corrective RAG pattern for improved retrieval and response generation:

1. Initial retrieval from Azure Cognitive Search
2. Document grading to filter relevant content
3. Query transformation for optimized search
4. Web search fallback when documents are insufficient
5. Answer generation from the most relevant sources

## License

[Your License Information]