import logging
from langchain_community.vectorstores import FAISS
from src import config

logger = logging.getLogger(__name__)

def build_vector_store(chunks, embeddings):
    """Build a FAISS vector store from document chunks and embeddings."""
    if not chunks:
        logger.error("Cannot build vector store: No document chunks provided.")
        raise ValueError("No chunks provided.")
        
    logger.info(f"Computing embeddings and building FAISS vector store for {len(chunks)} chunks...")
    try:
        db = FAISS.from_documents(chunks, embeddings)
        logger.info("FAISS vector store built successfully.")
        return db
    except Exception as e:
        logger.error(f"Error building FAISS vector store: {e}")
        raise

def save_vector_store(db, path=config.VECTORSTORE_PATH):
    """Persist FAISS index locally."""
    logger.info(f"Saving FAISS index to {path}...")
    try:
        db.save_local(str(path))
        logger.info("FAISS index saved successfully.")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        raise

def load_vector_store(path=config.VECTORSTORE_PATH, embeddings=None):
    """Load FAISS index from local path.
    
    Note: We set allow_dangerous_deserialization=True because we are loading our own 
    locally built index generated during our data ingestion phase. This is standard 
    and safe for this application context.
    """
    if embeddings is None:
        from src.embeddings import get_embeddings
        embeddings = get_embeddings()
        
    logger.info(f"Loading FAISS index from {path}...")
    try:
        db = FAISS.load_local(
            str(path), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully.")
        return db
    except Exception as e:
        logger.error(f"Failed to load FAISS index from {path}. Ensure ingestion has been run. Error: {e}")
        raise
