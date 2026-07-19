import logging
from langchain_huggingface import HuggingFaceEmbeddings
from src import config

logger = logging.getLogger(__name__)

def get_embeddings():
    """Initialize and return the local HuggingFace embeddings model."""
    model_name = config.EMBEDDING_MODEL_NAME
    logger.info(f"Initializing local HuggingFace embeddings model: {model_name}...")
    try:
        # Instantiate the embeddings. It runs locally via sentence-transformers.
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # default to CPU for compatibility
            encode_kwargs={'normalize_embeddings': True} # normalizes for cosine similarity
        )
        logger.info("Embeddings model loaded successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load HuggingFace embeddings: {e}")
        raise
