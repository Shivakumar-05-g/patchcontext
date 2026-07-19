import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.github_loader import GitHubLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Reconfigure stdout to use utf-8 on Windows to avoid charmap encoding errors
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    logger.info("Starting Github Loader Test...")
    # Initialize loader. This will default to fastapi/fastapi and read environment variables.
    try:
        loader = GitHubLoader()
        
        # Fetch 2 commits
        commits = loader.fetch_commits(limit=2)
        print(f"\n--- COMMITS (Fetched {len(commits)}) ---")
        for c in commits:
            print(f"SHA: {c['sha']}")
            print(f"Author: {c['author']}")
            print(f"Message: {c['message'].splitlines()[0] if c['message'] else ''}")
            print(f"Comments count: {len(c['comments'])}")
            print("-" * 20)
            
        # Fetch 2 PRs
        prs = loader.fetch_pull_requests(limit=2)
        print(f"\n--- PULL REQUESTS (Fetched {len(prs)}) ---")
        for pr in prs:
            print(f"Number: #{pr['number']}")
            print(f"Title: {pr['title']}")
            print(f"Author: {pr['author']}")
            print(f"Comments count: {len(pr['comments'])}")
            print("-" * 20)
            
        # Fetch 2 Issues
        issues = loader.fetch_issues(limit=2)
        print(f"\n--- ISSUES (Fetched {len(issues)}) ---")
        for issue in issues:
            print(f"Number: #{issue['number']}")
            print(f"Title: {issue['title']}")
            print(f"Author: {issue['author']}")
            print(f"Comments/Timeline count: {len(issue['comments'])}")
            print("-" * 20)
            
        # Save cache files to test caching
        logger.info("Saving fetched files as test JSONs...")
        loader.save_raw_data(commits, "test_commits.json")
        loader.save_raw_data(prs, "test_prs.json")
        loader.save_raw_data(issues, "test_issues.json")
        
        # Test loading cache
        cached_commits = loader.load_cached_raw_data("test_commits.json")
        if cached_commits:
            print(f"\nSuccessfully loaded cache. Commits read: {len(cached_commits)}")
            
    except Exception as e:
        logger.error(f"Error occurred during test: {e}")

if __name__ == "__main__":
    main()
