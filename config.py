# config.py
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file (for local development)
load_dotenv()

# Function to get config from either streamlit secrets or environment variables
def get_config(section, key, env_var_name, default=""):
    # Try to get from Streamlit secrets first
    try:
        if section in st.secrets and key in st.secrets[section]:
            return st.secrets[section][key]
    except:
        pass
    
    # Fall back to environment variables
    return os.environ.get(env_var_name, default)

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = get_config("azure_openai", "api_key", "AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = get_config("azure_openai", "endpoint", "AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = get_config("azure_openai", "api_version", "AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_DEPLOYMENT = get_config("azure_openai", "deployment", "AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_EMBEDDING_DEPLOYMENT = get_config("azure_openai", "embedding_deployment", "AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

# Azure Blob Storage
AZURE_BLOB_CONNECTION_STRING = get_config("azure_blob", "connection_string", "AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = get_config("azure_blob", "container", "AZURE_BLOB_CONTAINER", "documents")

# Azure AI Search
AZURE_SEARCH_ENDPOINT = get_config("azure_search", "endpoint", "AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = get_config("azure_search", "key", "AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = get_config("azure_search", "index", "AZURE_SEARCH_INDEX", "azureblob-index")

# LangSmith Configuration
LANGCHAIN_TRACING_V2_STR = get_config("langchain", "tracing_v2", "LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_TRACING_V2 = LANGCHAIN_TRACING_V2_STR.lower() == "true"
LANGCHAIN_API_KEY = get_config("langchain", "api_key", "LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = get_config("langchain", "project", "LANGCHAIN_PROJECT", "corrective-rag-demo")
