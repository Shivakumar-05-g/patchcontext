import os
import sys
import site
from unittest.mock import MagicMock
from pathlib import Path
from dotenv import load_dotenv

# Make sure user-site packages installed by pip are visible to this interpreter.
USER_SITE = site.getusersitepackages()
if USER_SITE and USER_SITE not in sys.path:
    sys.path.append(USER_SITE)

VENDOR_SITE = (Path(__file__).resolve().parent.parent / "vendor" / "python_pkgs").resolve()
if VENDOR_SITE.exists() and str(VENDOR_SITE) not in sys.path:
    sys.path.append(str(VENDOR_SITE))

VENDOR_SITE_2 = (Path(__file__).resolve().parent.parent / "vendor2" / "python_pkgs").resolve()
if VENDOR_SITE_2.exists() and str(VENDOR_SITE_2) not in sys.path:
    sys.path.append(str(VENDOR_SITE_2))

# Monkeypatch VertexAI imports to solve dependency clashes between newer langchain-community and Ragas
class DummyVertexAI:
    pass

sys.modules['langchain_community.chat_models.vertexai'] = MagicMock(ChatVertexAI=DummyVertexAI)
sys.modules['langchain_community.embeddings.vertexai'] = MagicMock(VertexAIEmbeddings=DummyVertexAI)

# Load environment variables from .env file
load_dotenv()

# Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
VECTORSTORE_PATH = VECTORSTORE_DIR / "faiss_index"

# Create directories if they do not exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# API Keys and Tokens
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Model configurations
# default model to Llama 3.1 8b on Groq
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ingestion configurations
GITHUB_REPO = "fastapi/fastapi"

# Retrieval configurations
DEFAULT_K = 5
DEFAULT_FETCH_K = 20

# NLI Model configuration (local cross encoder)
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
