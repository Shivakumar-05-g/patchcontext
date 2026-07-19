import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class SimpleDoc:
    page_content: str
    metadata: dict


def normalize_ragas_scores(result):
    """Convert a Ragas-like result object into a plain metric dictionary."""
    if result is None:
        return {}

    if isinstance(result, dict):
        return result

    scores = getattr(result, "scores", None)
    if isinstance(scores, dict):
        return scores

    if scores is not None:
        try:
            return dict(scores)
        except Exception:
            pass

    if hasattr(result, "to_pandas"):
        try:
            df = result.to_pandas()
            numeric = df.select_dtypes(include="number")
            if not numeric.empty:
                return {col: float(numeric[col].mean()) for col in numeric.columns}
        except Exception as e:
            logger.warning(f"Could not normalize result via to_pandas(): {e}")

    return {}


def load_benchmark(file_path=None):
    """Load the 50-question benchmark."""
    if file_path is None:
        file_path = config.PROJECT_ROOT / "evaluation" / "benchmark_50.json"

    logger.info(f"Loading benchmark dataset from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    logger.info(f"Loaded {len(benchmark)} questions from benchmark.")
    return benchmark


def load_raw_corpus():
    """Build a lightweight corpus from cached GitHub JSON files."""
    corpus = []

    def add_commit(commit):
        parts = [
            f"Commit SHA: {commit.get('sha', '')}",
            f"Author: {commit.get('author', 'Unknown')}",
            f"Date: {commit.get('created_date', '')}",
            f"Commit Message: {commit.get('message', '')}",
        ]
        comments = commit.get("comments", [])
        if comments:
            parts.append("Comments:")
            for comment in comments:
                parts.append(
                    f"- {comment.get('author', 'Unknown')}: {comment.get('body', '')}"
                )
        corpus.append(
            SimpleDoc(
                page_content="\n".join(parts),
                metadata={
                    "source_type": "commit",
                    "commit_sha": commit.get("sha", ""),
                    "author": commit.get("author", "Unknown"),
                    "source_url": commit.get("source_url", ""),
                    "title": (commit.get("message") or "").split("\n")[0],
                },
            )
        )

    def add_pr(pr):
        parts = [
            f"Pull Request #{pr.get('number')}: {pr.get('title', '')}",
            f"Author: {pr.get('author', 'Unknown')}",
            f"Created Date: {pr.get('created_date', '')}",
            f"Description: {pr.get('body', '')}",
        ]
        comments = pr.get("comments", [])
        if comments:
            parts.append("Comments/Discussion:")
            for comment in comments:
                parts.append(
                    f"- {comment.get('author', 'Unknown')} ({comment.get('comment_type', 'comment')}): {comment.get('body', '')}"
                )
        corpus.append(
            SimpleDoc(
                page_content="\n".join(parts),
                metadata={
                    "source_type": "pull_request",
                    "pr_number": int(pr.get("number")) if pr.get("number") is not None else None,
                    "author": pr.get("author", "Unknown"),
                    "source_url": pr.get("source_url", ""),
                    "title": pr.get("title", ""),
                },
            )
        )

    def add_issue(issue):
        parts = [
            f"Issue #{issue.get('number')}: {issue.get('title', '')}",
            f"Author: {issue.get('author', 'Unknown')}",
            f"Created Date: {issue.get('created_date', '')}",
            f"Description: {issue.get('body', '')}",
        ]
        comments = issue.get("comments", [])
        if comments:
            parts.append("Comments/Discussion:")
            for comment in comments:
                parts.append(
                    f"- {comment.get('author', 'Unknown')}: {comment.get('body', '')}"
                )
        corpus.append(
            SimpleDoc(
                page_content="\n".join(parts),
                metadata={
                    "source_type": "issue",
                    "issue_number": int(issue.get("number")) if issue.get("number") is not None else None,
                    "author": issue.get("author", "Unknown"),
                    "source_url": issue.get("source_url", ""),
                    "title": issue.get("title", ""),
                },
            )
        )

    with open(config.DATA_RAW_DIR / "commits.json", "r", encoding="utf-8") as f:
        for commit in json.load(f):
            add_commit(commit)

    with open(config.DATA_RAW_DIR / "pull_requests.json", "r", encoding="utf-8") as f:
        for pr in json.load(f):
            add_pr(pr)

    with open(config.DATA_RAW_DIR / "issues.json", "r", encoding="utf-8") as f:
        for issue in json.load(f):
            add_issue(issue)

    logger.info(f"Built fallback corpus with {len(corpus)} documents.")
    return corpus


def tokenize(text):
    return {t.lower() for t in WORD_RE.findall(text or "") if len(t) > 2}


def best_doc_text(doc):
    return " ".join(
        [
            doc.page_content,
            doc.metadata.get("title", ""),
            str(doc.metadata.get("author", "")),
            str(doc.metadata.get("pr_number", "")),
            str(doc.metadata.get("issue_number", "")),
            str(doc.metadata.get("commit_sha", "")),
        ]
    ).strip()


def score_doc(question, doc):
    q_tokens = tokenize(question)
    d_tokens = tokenize(best_doc_text(doc))
    overlap = len(q_tokens & d_tokens)

    score = float(overlap)
    q_lower = question.lower()

    if doc.metadata.get("source_type") == "commit" and doc.metadata.get("commit_sha"):
        sha = doc.metadata["commit_sha"].lower()
        if sha[:8] in q_lower or sha in q_lower:
            score += 10
        if "commit" in q_lower:
            score += 1
    elif doc.metadata.get("source_type") == "pull_request" and doc.metadata.get("pr_number") is not None:
        pr_num = str(doc.metadata["pr_number"])
        if f"#{pr_num}" in q_lower or pr_num in q_lower:
            score += 10
        if "pr" in q_lower or "pull request" in q_lower:
            score += 1
    elif doc.metadata.get("source_type") == "issue" and doc.metadata.get("issue_number") is not None:
        issue_num = str(doc.metadata["issue_number"])
        if f"#{issue_num}" in q_lower or issue_num in q_lower:
            score += 10
        if "issue" in q_lower:
            score += 1

    return score


def retrieve_fallback_context(question, corpus, k=5, fetch_k=20):
    ranked = sorted(corpus, key=lambda d: score_doc(question, d), reverse=True)
    return ranked[: max(1, k)]


def summarize_doc(doc):
    text = doc.page_content
    first_line = text.splitlines()[0].strip()
    source_type = doc.metadata.get("source_type")

    if source_type == "commit":
        sha = doc.metadata.get("commit_sha", "")
        author = doc.metadata.get("author", "Unknown")
        msg = doc.metadata.get("title", "")
        return f"{first_line}. This commit was authored by {author}. [Commit: {sha}]"
    if source_type == "pull_request":
        num = doc.metadata.get("pr_number")
        title = doc.metadata.get("title", "")
        return f"{first_line}. The PR explains the change and motivation behind {title}. [PR: #{num}]"
    if source_type == "issue":
        num = doc.metadata.get("issue_number")
        title = doc.metadata.get("title", "")
        return f"{first_line}. The issue documents the reported problem around {title}. [Issue: #{num}]"
    return first_line


def generate_fallback_answer(question, retrieved_docs):
    if not retrieved_docs:
        return "I could not retrieve any relevant repository history documents to answer this question."

    top = retrieved_docs[0]
    q = question.lower()
    source_type = top.metadata.get("source_type")

    if source_type == "commit":
        sha = top.metadata.get("commit_sha", "")
        author = top.metadata.get("author", "Unknown")
        message = top.metadata.get("title", "")
        if "who" in q or "author" in q:
            return f"Commit {sha} was authored by {author}. The commit message says: {message}. [Commit: {sha}]"
        return f"{message}. It was authored by {author}. [Commit: {sha}]"

    if source_type == "pull_request":
        num = top.metadata.get("pr_number")
        title = top.metadata.get("title", "")
        if "why" in q or "motivation" in q:
            return f"PR #{num} ({title}) explains the motivation in the retrieved discussion. [PR: #{num}]"
        return f"PR #{num} ({title}) introduces the change described in the retrieved context. [PR: #{num}]"

    if source_type == "issue":
        num = top.metadata.get("issue_number")
        title = top.metadata.get("title", "")
        if "who" in q or "opened" in q:
            return f"Issue #{num} was opened by {top.metadata.get('author', 'Unknown')}. {title}. [Issue: #{num}]"
        return f"Issue #{num} reports: {title}. [Issue: #{num}]"

    return summarize_doc(top)


def citation_coverage(answer, retrieved_docs):
    from src.citations import extract_citations

    citations = extract_citations(answer)
    if not citations:
        return 0.0

    valid = 0
    for cit in citations:
        if cit["type"] == "pull_request":
            if any(doc.metadata.get("source_type") == "pull_request" and int(doc.metadata.get("pr_number", -1)) == cit["id"] for doc in retrieved_docs):
                valid += 1
        elif cit["type"] == "issue":
            if any(doc.metadata.get("source_type") == "issue" and int(doc.metadata.get("issue_number", -1)) == cit["id"] for doc in retrieved_docs):
                valid += 1
        elif cit["type"] == "commit":
            if any(
                doc.metadata.get("source_type") == "commit"
                and str(doc.metadata.get("commit_sha", "")).lower().startswith(str(cit["id"]).lower())
                for doc in retrieved_docs
            ):
                valid += 1

    return valid / len(citations)


def answer_relevancy(question, answer):
    q = tokenize(question)
    a = tokenize(answer)
    if not q or not a:
        return 0.0
    return len(q & a) / len(q)


def context_precision(question, contexts):
    q = tokenize(question)
    if not contexts:
        return 0.0
    hits = 0
    for ctx in contexts:
        if q & tokenize(ctx):
            hits += 1
    return hits / len(contexts)


def context_recall(expected_keywords, contexts):
    if not expected_keywords:
        return 0.0
    ctx_text = " ".join(contexts).lower()
    hits = 0
    for keyword in expected_keywords:
        if str(keyword).lower() in ctx_text:
            hits += 1
    return hits / len(expected_keywords)


def faithfulness(answer, contexts):
    a = tokenize(answer)
    ctx = tokenize(" ".join(contexts))
    if not a:
        return 0.0
    token_support = len(a & ctx) / len(a)
    citation_support = citation_coverage(answer, [SimpleDoc(page_content=c, metadata={}) for c in contexts])
    return (token_support + citation_support) / 2


def run_evaluation(num_questions=3, benchmark_file=None):
    """Run a local, deterministic fallback evaluation over the benchmark dataset."""
    benchmark = load_benchmark(benchmark_file)
    eval_subset = benchmark[:num_questions]
    logger.info(f"Running evaluation on {len(eval_subset)} questions (out of {len(benchmark)})...")

    corpus = load_raw_corpus()
    evaluation_data = []

    for idx, item in enumerate(eval_subset):
        question = item["question"]
        ground_truth = item["ground_truth"]
        expected_keywords = item.get("expected_context_keywords", [])

        logger.info(f"[{idx+1}/{len(eval_subset)}] Evaluating: '{question}'...")
        retrieved_docs = retrieve_fallback_context(question, corpus, k=config.DEFAULT_K, fetch_k=config.DEFAULT_FETCH_K)
        contexts = [doc.page_content for doc in retrieved_docs]
        answer = generate_fallback_answer(question, retrieved_docs)

        evaluation_data.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "expected_context_keywords": expected_keywords,
            }
        )

    if not evaluation_data:
        return {
            "scores": {},
            "raw_data": [],
            "mode": "fallback",
            "error": "No evaluation data collected.",
        }

    scores = {
        "faithfulness": sum(faithfulness(item["answer"], item["contexts"]) for item in evaluation_data) / len(evaluation_data),
        "answer_relevancy": sum(answer_relevancy(item["question"], item["answer"]) for item in evaluation_data) / len(evaluation_data),
        "context_precision": sum(context_precision(item["question"], item["contexts"]) for item in evaluation_data) / len(evaluation_data),
        "context_recall": sum(context_recall(item["expected_context_keywords"], item["contexts"]) for item in evaluation_data) / len(evaluation_data),
    }

    logger.info("Fallback evaluation completed successfully.")
    return {
        "scores": scores,
        "raw_data": evaluation_data,
        "mode": "fallback",
    }
