import re
import logging

logger = logging.getLogger(__name__)

# Regular expressions for matching citations
PR_REGEX = re.compile(r"\[PR:\s*#(\d+)\]")
ISSUE_REGEX = re.compile(r"\[Issue:\s*#(\d+)\]")
COMMIT_REGEX = re.compile(r"\[Commit:\s*([a-fA-F0-9]{7,40})\]")

def extract_citations(text):
    """Extract all citations from text, returning a list of dicts with their type and raw identifier."""
    citations = []
    
    # Extract PRs
    for match in PR_REGEX.finditer(text):
        citations.append({
            "type": "pull_request",
            "id": int(match.group(1)),
            "raw": match.group(0)
        })
        
    # Extract Issues
    for match in ISSUE_REGEX.finditer(text):
        citations.append({
            "type": "issue",
            "id": int(match.group(1)),
            "raw": match.group(0)
        })
        
    # Extract Commits
    for match in COMMIT_REGEX.finditer(text):
        citations.append({
            "type": "commit",
            "id": match.group(1).lower(),
            "raw": match.group(0)
        })
        
    return citations

def validate_citations(citations, retrieved_docs):
    """Validate citations against retrieved documents.
    
    Returns a tuple:
    - list of validated citations with their URL and validity status
    - list of invalid/hallucinated citations
    """
    # Build maps of valid identifiers present in the retrieved metadata
    valid_prs = set()
    valid_issues = set()
    valid_commits = set()
    
    for doc in retrieved_docs:
        meta = doc.metadata
        source_type = meta.get("source_type")
        
        if source_type == "pull_request" and meta.get("pr_number") is not None:
            valid_prs.add(int(meta["pr_number"]))
        elif source_type == "issue" and meta.get("issue_number") is not None:
            valid_issues.add(int(meta["issue_number"]))
        elif source_type == "commit" and meta.get("commit_sha"):
            valid_commits.add(meta["commit_sha"].lower())
            
    validated = []
    invalid_citations = []
    
    for cit in citations:
        cit_type = cit["type"]
        cit_id = cit["id"]
        raw = cit["raw"]
        
        is_valid = False
        url = ""
        
        if cit_type == "pull_request":
            if cit_id in valid_prs:
                is_valid = True
                url = f"https://github.com/fastapi/fastapi/pull/{cit_id}"
            else:
                invalid_citations.append(cit)
                
        elif cit_type == "issue":
            if cit_id in valid_issues:
                is_valid = True
                url = f"https://github.com/fastapi/fastapi/issues/{cit_id}"
            else:
                invalid_citations.append(cit)
                
        elif cit_type == "commit":
            # For commits, support short or long SHA matching
            matched_sha = None
            for sha in valid_commits:
                if sha.startswith(cit_id) or cit_id.startswith(sha):
                    matched_sha = sha
                    break
            
            if matched_sha:
                is_valid = True
                url = f"https://github.com/fastapi/fastapi/commit/{matched_sha}"
                cit["id"] = matched_sha[:8] # normalize to short sha for presentation
            else:
                invalid_citations.append(cit)
                
        if is_valid:
            validated.append({
                "type": cit_type,
                "id": cit_id,
                "url": url,
                "raw": raw,
                "valid": True
            })
            
    return validated, invalid_citations

def convert_citations_to_markdown_links(text, validated_citations):
    """Replace plain text citations with markdown hyperlinks based on validated URLs."""
    updated_text = text
    for cit in validated_citations:
        raw = cit["raw"]
        url = cit["url"]
        cit_type_label = "PR" if cit["type"] == "pull_request" else cit["type"].capitalize()
        replacement = f"[{cit_type_label} #{cit['id']}]({url})"
        updated_text = updated_text.replace(raw, replacement)
    return updated_text
