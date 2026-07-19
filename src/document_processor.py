import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size=800, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_commits(self, commits_raw):
        """Convert raw commits to LangChain Documents."""
        documents = []
        for c in commits_raw:
            sha = c.get("sha", "")
            author = c.get("author", "Unknown")
            created_date = c.get("created_date", "")
            message = c.get("message", "")
            source_url = c.get("source_url", "")
            comments = c.get("comments", [])
            
            # Format text representation
            content_parts = [
                f"Commit SHA: {sha}",
                f"Author: {author}",
                f"Date: {created_date}",
                f"Commit Message:\n{message}"
            ]
            
            if comments:
                content_parts.append("Comments:")
                for comment in comments:
                    comment_author = comment.get("author", "Unknown")
                    comment_body = comment.get("body", "")
                    content_parts.append(f"- {comment_author}: {comment_body}")
                    
            page_content = "\n".join(content_parts)
            
            # Preserve metadata
            metadata = {
                "source_type": "commit",
                "commit_sha": sha,
                "author": author,
                "created_date": created_date,
                "source_url": source_url,
                "title": message.split("\n")[0] if message else ""
            }
            
            documents.append(Document(page_content=page_content, metadata=metadata))
            
        logger.info(f"Processed {len(documents)} commit documents.")
        return documents

    def process_pull_requests(self, prs_raw):
        """Convert raw PRs to LangChain Documents."""
        documents = []
        for pr in prs_raw:
            number = pr.get("number")
            title = pr.get("title", "")
            body = pr.get("body", "")
            author = pr.get("author", "Unknown")
            created_date = pr.get("created_date", "")
            updated_date = pr.get("updated_date", "")
            source_url = pr.get("source_url", "")
            comments = pr.get("comments", [])
            
            # Format text representation
            content_parts = [
                f"Pull Request #{number}: {title}",
                f"Author: {author}",
                f"Created Date: {created_date}",
                f"Description:\n{body}"
            ]
            
            if comments:
                content_parts.append("Comments/Discussion:")
                for comment in comments:
                    comment_author = comment.get("author", "Unknown")
                    comment_body = comment.get("body", "")
                    c_type = comment.get("comment_type", "comment")
                    content_parts.append(f"- {comment_author} ({c_type}): {comment_body}")
                    
            page_content = "\n".join(content_parts)
            
            # Preserve metadata
            metadata = {
                "source_type": "pull_request",
                "pr_number": int(number) if number is not None else None,
                "title": title,
                "author": author,
                "created_date": created_date,
                "updated_date": updated_date,
                "source_url": source_url
            }
            
            documents.append(Document(page_content=page_content, metadata=metadata))
            
        logger.info(f"Processed {len(documents)} PR documents.")
        return documents

    def process_issues(self, issues_raw):
        """Convert raw Issues to LangChain Documents."""
        documents = []
        for issue in issues_raw:
            number = issue.get("number")
            title = issue.get("title", "")
            body = issue.get("body", "")
            author = issue.get("author", "Unknown")
            created_date = issue.get("created_date", "")
            updated_date = issue.get("updated_date", "")
            source_url = issue.get("source_url", "")
            comments = issue.get("comments", [])
            
            # Format text representation
            content_parts = [
                f"Issue #{number}: {title}",
                f"Author: {author}",
                f"Created Date: {created_date}",
                f"Description:\n{body}"
            ]
            
            if comments:
                content_parts.append("Comments/Discussion:")
                for comment in comments:
                    comment_author = comment.get("author", "Unknown")
                    comment_body = comment.get("body", "")
                    content_parts.append(f"- {comment_author}: {comment_body}")
                    
            page_content = "\n".join(content_parts)
            
            # Preserve metadata
            metadata = {
                "source_type": "issue",
                "issue_number": int(number) if number is not None else None,
                "title": title,
                "author": author,
                "created_date": created_date,
                "updated_date": updated_date,
                "source_url": source_url
            }
            
            documents.append(Document(page_content=page_content, metadata=metadata))
            
        logger.info(f"Processed {len(documents)} issue documents.")
        return documents

    def split_documents(self, documents):
        """Split documents into chunks while preserving metadata."""
        if not documents:
            return []
        
        logger.info(f"Splitting {len(documents)} documents using RecursiveCharacterTextSplitter (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Generated {len(chunks)} document chunks.")
        
        # Verify metadata is preserved and is flat (faiss likes flat primitive values)
        for chunk in chunks:
            # Check and flatten any nested metadata if present, though we kept it flat.
            # Clean up metadata value types.
            # Make sure we don't have None values for crucial types where it could fail.
            for key, val in list(chunk.metadata.items()):
                if val is None:
                    # Remove None keys or set them to empty string
                    chunk.metadata[key] = ""
                    
        return chunks
