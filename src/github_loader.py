import json
import logging
from datetime import datetime
from github import Github, GithubException
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GitHubLoader:
    def __init__(self, repo_name=config.GITHUB_REPO, token=config.GITHUB_TOKEN):
        self.repo_name = repo_name
        self.token = token
        
        if self.token:
            logger.info("Initializing GitHub client with provided GITHUB_TOKEN.")
            self.github_client = Github(self.token)
        else:
            logger.warning(
                "No GITHUB_TOKEN provided. Using unauthenticated client. "
                "GitHub API rate limits will be highly restricted (60 requests/hour)."
            )
            self.github_client = Github()
            
        try:
            self.repo = self.github_client.get_repo(self.repo_name)
            logger.info(f"Successfully connected to repository: {self.repo_name}")
        except Exception as e:
            logger.error(f"Failed to connect to repository {self.repo_name}: {e}")
            raise

    def fetch_commits(self, limit=100):
        """Fetch recent commits along with their basic metadata and comments."""
        logger.info(f"Fetching latest {limit} commits from {self.repo_name}...")
        commits_data = []
        try:
            commits = self.repo.get_commits()
            count = 0
            for commit in commits:
                if count >= limit:
                    break
                
                # Extract basic commit details
                sha = commit.sha
                message = commit.commit.message
                author = commit.commit.author.name if commit.commit.author else "Unknown"
                # Commit dates can be fetched from commit.commit.author.date
                created_date = commit.commit.author.date.isoformat() if commit.commit.author else None
                source_url = commit.html_url
                
                # Fetch comments on commit if any
                commit_comments = []
                try:
                    for comment in commit.get_comments():
                        commit_comments.append({
                            "id": comment.id,
                            "body": comment.body,
                            "author": comment.user.login if comment.user else "Unknown",
                            "created_date": comment.created_at.isoformat() if comment.created_at else None,
                            "updated_date": comment.updated_at.isoformat() if comment.updated_at else None,
                            "source_url": comment.html_url
                        })
                except Exception as ex:
                    logger.warning(f"Failed to fetch comments for commit {sha}: {ex}")

                commits_data.append({
                    "source_type": "commit",
                    "sha": sha,
                    "message": message,
                    "author": author,
                    "created_date": created_date,
                    "source_url": source_url,
                    "comments": commit_comments
                })
                count += 1
                if count % 10 == 0:
                    logger.info(f"Fetched {count}/{limit} commits...")
            
            logger.info(f"Successfully fetched {len(commits_data)} commits.")
        except Exception as e:
            logger.error(f"Error fetching commits: {e}")
        return commits_data

    def fetch_pull_requests(self, limit=50):
        """Fetch recent PRs, review comments, and conversation comments."""
        logger.info(f"Fetching latest {limit} pull requests from {self.repo_name}...")
        pr_data = []
        try:
            pulls = self.repo.get_pulls(state="all", sort="created", direction="desc")
            count = 0
            for pr in pulls:
                if count >= limit:
                    break
                
                # Basic PR metadata
                number = pr.number
                title = pr.title
                body = pr.body or ""
                author = pr.user.login if pr.user else "Unknown"
                created_date = pr.created_at.isoformat() if pr.created_at else None
                updated_date = pr.updated_at.isoformat() if pr.updated_at else None
                source_url = pr.html_url
                
                # Gather comments
                comments = []
                
                # Issue/Discussion comments on the PR
                try:
                    for comment in pr.get_issue_comments():
                        comments.append({
                            "source_type": "comment",
                            "comment_type": "pr_issue_comment",
                            "id": comment.id,
                            "body": comment.body,
                            "author": comment.user.login if comment.user else "Unknown",
                            "created_date": comment.created_at.isoformat() if comment.created_at else None,
                            "updated_date": comment.updated_at.isoformat() if comment.updated_at else None,
                            "source_url": comment.html_url
                        })
                except Exception as ex:
                    logger.warning(f"Error fetching issue comments for PR #{number}: {ex}")

                # Line-level review comments on the PR
                try:
                    for comment in pr.get_comments():
                        comments.append({
                            "source_type": "comment",
                            "comment_type": "pr_review_comment",
                            "id": comment.id,
                            "body": comment.body,
                            "author": comment.user.login if comment.user else "Unknown",
                            "created_date": comment.created_at.isoformat() if comment.created_at else None,
                            "updated_date": comment.updated_at.isoformat() if comment.updated_at else None,
                            "source_url": comment.html_url
                        })
                except Exception as ex:
                    logger.warning(f"Error fetching review comments for PR #{number}: {ex}")

                pr_data.append({
                    "source_type": "pull_request",
                    "number": number,
                    "title": title,
                    "body": body,
                    "author": author,
                    "created_date": created_date,
                    "updated_date": updated_date,
                    "source_url": source_url,
                    "comments": comments
                })
                count += 1
                if count % 10 == 0:
                    logger.info(f"Fetched {count}/{limit} PRs...")
            
            logger.info(f"Successfully fetched {len(pr_data)} pull requests.")
        except Exception as e:
            logger.error(f"Error fetching pull requests: {e}")
        return pr_data

    def fetch_issues(self, limit=50):
        """Fetch recent Issues (excluding PRs) along with their comments."""
        logger.info(f"Fetching latest {limit} issues from {self.repo_name}...")
        issues_data = []
        try:
            issues = self.repo.get_issues(state="all", sort="created", direction="desc")
            count = 0
            for issue in issues:
                if count >= limit:
                    break
                
                # Check if it's a pull request. Every PR is an issue in the github api,
                # but we only want pure issues here.
                if issue.pull_request is not None:
                    continue
                
                number = issue.number
                title = issue.title
                body = issue.body or ""
                author = issue.user.login if issue.user else "Unknown"
                created_date = issue.created_at.isoformat() if issue.created_at else None
                updated_date = issue.updated_at.isoformat() if issue.updated_at else None
                source_url = issue.html_url
                
                # Fetch comments on the issue
                comments = []
                try:
                    for comment in issue.get_comments():
                        comments.append({
                            "source_type": "comment",
                            "comment_type": "issue_comment",
                            "id": comment.id,
                            "body": comment.body,
                            "author": comment.user.login if comment.user else "Unknown",
                            "created_date": comment.created_at.isoformat() if comment.created_at else None,
                            "updated_date": comment.updated_at.isoformat() if comment.updated_at else None,
                            "source_url": comment.html_url
                        })
                except Exception as ex:
                    logger.warning(f"Error fetching comments for issue #{number}: {ex}")

                issues_data.append({
                    "source_type": "issue",
                    "number": number,
                    "title": title,
                    "body": body,
                    "author": author,
                    "created_date": created_date,
                    "updated_date": updated_date,
                    "source_url": source_url,
                    "comments": comments
                })
                count += 1
                if count % 10 == 0:
                    logger.info(f"Fetched {count}/{limit} issues...")

            logger.info(f"Successfully fetched {len(issues_data)} issues.")
        except Exception as e:
            logger.error(f"Error fetching issues: {e}")
        return issues_data

    def save_raw_data(self, data, filename):
        """Save fetched data to raw data folder."""
        file_path = config.DATA_RAW_DIR / filename
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved data to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {e}")

    def load_cached_raw_data(self, filename):
        """Load data from cache if it exists."""
        file_path = config.DATA_RAW_DIR / filename
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded cached data from {file_path}")
                return data
            except Exception as e:
                logger.error(f"Failed to load cached data from {file_path}: {e}")
        return None
