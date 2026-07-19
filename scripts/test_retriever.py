import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.retriever import retrieve_context
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    logger.info("Starting MMR Retriever Test...")
    
    # Test query
    query = "APIRoute handler caching race condition"
    
    # Retrieve context
    docs = retrieve_context(query, k=5, fetch_k=15)
    
    print(f"\n=== RETRIEVED DOCUMENTS FOR QUERY: '{query}' (Found {len(docs)}) ===")
    for idx, doc in enumerate(docs):
        print(f"\n[{idx + 1}] Source: {doc.metadata.get('source_type', 'unknown')}")
        print(f"URL: {doc.metadata.get('source_url', 'n/a')}")
        print(f"Title: {doc.metadata.get('title', 'n/a')}")
        print(f"Content Preview:\n{doc.page_content[:200]}...")
        print("-" * 40)

if __name__ == "__main__":
    main()
