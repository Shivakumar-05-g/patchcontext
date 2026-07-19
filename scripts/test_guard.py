import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.hallucination_guard import HallucinationGuard
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    logger.info("Starting Hallucination Guard Test...")
    
    # 1. Initialize guard (we can test both local NLI and fallback LLM NLI by toggling use_local_nli)
    logger.info("Initializing guard with local NLI enabled...")
    guard = HallucinationGuard(use_local_nli=True)
    
    # 2. Set up dummy retrieved context
    retrieved_docs = [
        Document(
            page_content="Pull Request #16021: Optimize APIRoute handler caching\nSebastián Ramírez merged PR #16021 which optimizes caching by replacing id() with WeakKeyDictionary in effective route handler context.",
            metadata={"source_type": "pull_request", "pr_number": 16021, "source_url": "https://github.com/fastapi/fastapi/pull/16021"}
        )
    ]
    
    # 3. Dummy generated answer containing:
    # - Supported claim + Valid citation
    # - Hallucinated claim + Invalid citation
    sample_answer = (
        "FastAPI merged PR #16021 to optimize routing caching by using a WeakKeyDictionary context. [PR: #16021]\n"
        "This change was authored by Joe Developer. [PR: #99999]\n"
        "Additionally, the system now runs twice as fast. [Commit: 1234567abc]"
    )
    
    # 4. Check
    logger.info("Checking answer for hallucinations...")
    report = guard.check_hallucinations(sample_answer, retrieved_docs)
    
    print("\n" + "="*50)
    print("HALLUCINATION REPORT")
    print("="*50)
    print(f"Is Safe? {report['is_safe']}")
    
    print("\n--- CITATIONS ANALYSIS ---")
    print(f"Validated Citations: {report['validated_citations']}")
    print(f"Invalid Citations (Fabricated): {report['invalid_citations']}")
    
    print("\n--- STATEMENTS ANALYSIS ---")
    for r in report['statement_reports']:
        print(f"\nStatement: '{r['statement']}'")
        print(f"NLI Classification: {r['nli_label']} (confidence: {r['confidence']:.2f})")
        print(f"Flagged as Hallucination? {r['flagged']}")
        print(f"Audited via: {r['method']}")
    print("="*50)

if __name__ == "__main__":
    main()
