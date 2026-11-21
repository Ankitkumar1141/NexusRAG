import os
import sys
from pathlib import Path

# Ensure project root is always on sys.path for pytest
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Mock default env vars for CI environments where .env is not committed
os.environ.setdefault("MISTRAL_API_KEY", "ci_dummy_mistral_key")
os.environ.setdefault("GEMINI_API_KEY", "ci_dummy_gemini_key")
os.environ.setdefault("VECTOR_DB", "faiss")
os.environ.setdefault("MISTRAL_CHAT_MODEL", "mistral-small-latest")
os.environ.setdefault("GEMINI_EMBEDDING_MODEL", "models/embedding-001")

