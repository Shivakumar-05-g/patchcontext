import sys
import argparse
import logging
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.github_loader import GitHubLoader
from src.document_processor import DocumentProcessor
from src.embeddings import get_embeddings
from src.vector_store import build_vector_store, save_vector_store

# Set up logging and reconfigure encoding for stdout on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest")

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest historical data from GitHub and build FAISS index.")
    parser.add_argument(
        "--limit-commits", 
        type=int, 
        default=50, 
        help="Number of commits to fetch (default: 50)"
    )
    parser.add_argument(
        "--limit-prs", 
        type=int, 
        default=50, 
        help="Number of PRs to fetch (default: 50)"
    )
    parser.add_argument(
        "--limit-issues", 
        type=int, 
        default=50, 
        help="Number of issues to fetch (default: 50)"
    )
    parser.add_argument(
        "--force-fetch", 
        action="store_true", 
        help="Force fetching new data from GitHub API instead of loading cached raw data"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=800, 
        help="Chunk size for text splitter (default: 800)"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=100, 
        help="Chunk overlap for text splitter (default: 100)"
    )
    return parser.parse_args()

def load_cached_raw_data(filename):
    file_path = config.DATA_RAW_DIR / filename
    if file_path.exists():
        try:
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded cached data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load cached data from {file_path}: {e}")
    return None

def main():
    args = parse_args()
    logger.info("=== Starting Ingestion Process ===")
    
    commits_file = "commits.json"
    prs_file = "pull_requests.json"
    issues_file = "issues.json"
    
    commits_raw = None
    prs_raw = None
    issues_raw = None
    
    # Try to load cached data first if force-fetch is not enabled
    if not args.force_fetch:
        logger.info("Checking for cached raw data in data/raw/...")
        commits_raw = load_cached_raw_data(commits_file)
        prs_raw = load_cached_raw_data(prs_file)
        issues_raw = load_cached_raw_data(issues_file)
        
    # Fetch from API if cached data is missing or force_fetch is requested
    if commits_raw is None or prs_raw is None or issues_raw is None:
        logger.info("Cached raw data incomplete or --force-fetch active. Initializing GitHub client...")
        loader = GitHubLoader()
        
        if commits_raw is None or args.force_fetch:
            commits_raw = loader.fetch_commits(limit=args.limit_commits)
            loader.save_raw_data(commits_raw, commits_file)
            
        if prs_raw is None or args.force_fetch:
            prs_raw = loader.fetch_pull_requests(limit=args.limit_prs)
            loader.save_raw_data(prs_raw, prs_file)
            
        if issues_raw is None or args.force_fetch:
            issues_raw = loader.fetch_issues(limit=args.limit_issues)
            loader.save_raw_data(issues_raw, issues_file)
            
    # Print statistics of raw data loaded
    logger.info(f"Loaded {len(commits_raw)} commits, {len(prs_raw)} pull requests, and {len(issues_raw)} issues.")
    
    # Process into LangChain Documents
    logger.info("Processing raw data into LangChain Documents...")
    processor = DocumentProcessor(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    
    commit_docs = processor.process_commits(commits_raw)
    pr_docs = processor.process_pull_requests(prs_raw)
    issue_docs = processor.process_issues(issues_raw)
    
    all_docs = commit_docs + pr_docs + issue_docs
    logger.info(f"Total structured documents created: {len(all_docs)}")
    
    # Chunk the documents
    chunks = processor.split_documents(all_docs)
    logger.info(f"Total chunks created: {len(chunks)}")
    
    if not chunks:
        logger.error("No chunks generated! Ingestion aborted.")
        return
        
    # Get embeddings model
    embeddings = get_embeddings()
    
    # Build FAISS vector store
    db = build_vector_store(chunks, embeddings)
    
    # Save FAISS index
    save_vector_store(db, config.VECTORSTORE_PATH)
    
    logger.info("=== Ingestion Process Completed Successfully! ===")

if __name__ == "__main__":
    main()
