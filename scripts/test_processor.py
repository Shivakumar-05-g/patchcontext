import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.github_loader import GitHubLoader
from src.document_processor import DocumentProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    logger.info("Starting Document Processor Test...")
    
    # Initialize Loader & Processor
    loader = GitHubLoader()
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50) # smaller chunks for testing split
    
    # Load previously cached test data
    commits_raw = loader.load_cached_raw_data("test_commits.json")
    prs_raw = loader.load_cached_raw_data("test_prs.json")
    issues_raw = loader.load_cached_raw_data("test_issues.json")
    
    if not (commits_raw and prs_raw and issues_raw):
        logger.error("Test cache files missing. Run scripts/test_loader.py first.")
        return
        
    # Process them
    commit_docs = processor.process_commits(commits_raw)
    pr_docs = processor.process_pull_requests(prs_raw)
    issue_docs = processor.process_issues(issues_raw)
    
    all_docs = commit_docs + pr_docs + issue_docs
    logger.info(f"Total processed documents: {len(all_docs)}")
    
    # Split them
    chunks = processor.split_documents(all_docs)
    logger.info(f"Total split chunks: {len(chunks)}")
    
    print("\n--- SAMPLE CHUNKS & METADATA ---")
    for idx in [0, len(chunks)//2, len(chunks)-1]:
        if idx < len(chunks):
            chunk = chunks[idx]
            print(f"Chunk Index: {idx}")
            print(f"Metadata: {chunk.metadata}")
            print(f"Content Preview (first 150 chars):\n{chunk.page_content[:150]}...")
            print("=" * 40)

if __name__ == "__main__":
    main()
