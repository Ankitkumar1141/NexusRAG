import os
from pathlib import Path

from python_dotenv import load_dotenv
load_dotenv()

def _get_int(name: str, default: int) -> int:   
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return int(raw_value)
    except ValueError:
        print(f"[config] '{name}={raw_value}' is not a number. Using the value -> {default}")
        return default


PROJECT_ROOT = Path(__file__).parent
DOCUMENT_DIR = PROJECT_ROOT/"data"/"documents"
EVALUATION_DIR=PROJECT_ROOT/"data"/"evaluation"
VECTORSTORE_DIR=PROJECT_ROOT/"vectorstore"
FAISS_DIR=PROJECT_ROOT/"vectorstore"/"faiss"

DOC_STORE_PATH = PROJECT_ROOT/"vectorstore"/"doc_store.json"

# MistralAI 
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY","")
MISTRAL_CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL")

## Gemini
GEMINI_API_KEY= os.getenv("GEMINI_API_KEY","")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL","")

TEMPERATURE = 0.0 ## introduces randamness and variance in LLM responses

## vector_store
VECTOR_DB = os.getenv("VECTOR_DB","")
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY","")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME","rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESAPCE","NexusRAG")

PARENT_CHUNK_SIZE = _get_int("PARENT_CHUNK_SIZE", 1600)
PARENT_CHUNK_OVERLAP = _get_int("PARENT_CHUNK_OVERLAP", 200)
CHILD_CHUNK_SIZE = _get_int("CHILD_CHUNK_SIZE", 400)
CHILD_CHUNK_OVERLAP = _get_int("CHILD_CHUNK_OVERLAP", 60)


# retrieval pipeline
NUM_QUERY_VARIENTS = _get_int("NUM_QUERY_VARIENTS", 3) ## nultiquery

VECTOR_TOP_K = _get_int("VECTOR_TOP_K", 6) ## extract top k using scemantic search

BM25_TOP_K = _get_int("BM25_TOP_K", 6) ## extract top k using keyword  search - Sparse search

HYBRID_TOP_K = _get_int("HYBRID_TOP_K", 10) ## extract top k chunk after RRF fusion

RERANK_TOP_K = _get_int("RERANK_TOP_K", 4) ## top k chunks for cross_encoder

MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-4.0")) 

MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS", 8000)
RERANKER = os.getenv("RERANKER", "cross-encoder").strip().lower()
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MEMORY_WINDOW = _get_int("MEMORY_WINDOW", 4)


def check_mistral_api() ->None:
    if not MISTRAL_API_KEY:
        raise ValueError(
            "MistralAI key not found"
        )

def check_gemini_api()->None:
    if not GEMINI_API_KEY:
        raise ValueError("" \
        "Gemini AI key not found"
        )
def check_pinecone_config()->None:
    if not PINECONE_API_KEY:
        raise ValueError(
            "PineCone key not found"
        )
    if not PINECONE_INDEX_NAME:
        raise ValueError(
            "PinCone Index name is missing"
        )

def check_model()-> dict:
    return{
        "Chat Model":MISTRAL_CHAT_MODEL,
        "Embedding Model":EMBEDDING_MODEL,
        "VectorDb":VECTOR_DB,
        "Parent_chunk_size":PARENT_CHUNK_SIZE,
        "Child chunk size": CHILD_CHUNK_SIZE
    }