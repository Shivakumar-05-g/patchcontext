import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag_chain import generate_answer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    logger.info("Starting RAG Generation Test...")
    
    # Test query
    query = "Why did FastAPI modify APIRoute handler caching or routing cache?"
    
    # Generate response
    result = generate_answer(query)
    
    print("\n" + "="*50)
    print(f"QUESTION: {result['question']}")
    print("="*50)
    print(f"ANSWER:\n{result['answer']}")
    print("="*50)
    print(f"Retrieved {len(result['retrieved_docs'])} context chunks.")
    print("="*50)

if __name__ == "__main__":
    main()
