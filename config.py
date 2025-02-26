# Azure OpenAI Configuration
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

# Azure Blob Storage
AZURE_BLOB_CONNECTION_STRING = os.environ.get("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.environ.get("AZURE_BLOB_CONTAINER", "documents")

# Azure AI Search
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "azureblob-index")

# LangSmith Configuration
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "true").lower() == "true"
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "corrective-rag-demo")