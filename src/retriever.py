import logging
from src import config
from src.vector_store import load_vector_store

logger = logging.getLogger(__name__)

# Global singleton-like cache for the FAISS vector store
_vector_store_instance = None

def get_cached_vector_store(path=config.VECTORSTORE_PATH):
    """Load and cache the FAISS vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        logger.info("Vector store instance not cached. Loading from disk...")
        try:
            _vector_store_instance = load_vector_store(path)
        except Exception as e:
            logger.error(f"Failed to load vector store from {path}: {e}")
            raise
    return _vector_store_instance

def retrieve_context(query, k=config.DEFAULT_K, fetch_k=config.DEFAULT_FETCH_K, path=config.VECTORSTORE_PATH):
    """Retrieve relevant and diverse documents using MMR retrieval."""
    logger.info(f"Retrieving context for query: '{query}' using MMR (k={k}, fetch_k={fetch_k})...")
    
    try:
        db = get_cached_vector_store(path)
        
        # Instantiate retriever using MMR search type
        # MMR balances relevancy (with query) and diversity (among retrieved docs)
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k
            }
        )
        
        # Perform retrieval
        retrieved_docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(retrieved_docs)} document chunks.")
        return retrieved_docs
        
    except Exception as e:
        logger.error(f"Error during context retrieval: {e}")
        return []
