import json
import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config


def load_raw_json(filename):
    path = config.DATA_RAW_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_sentence(text):
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\n", " ").strip()
    parts = text.split(". ")
    return parts[0].strip()


def clean_title(title):
    title = (title or "").strip()
    return title.lstrip("📝🔥🐛🔖⬆️").strip()


def build_benchmark():
    commits = load_raw_json("commits.json")
    prs = load_raw_json("pull_requests.json")
    issues = load_raw_json("issues.json")

    questions = []
    next_id = 1

    # Two questions per PR
    for pr in prs:
        number = pr["number"]
        title = clean_title(pr.get("title", ""))
        body = first_sentence(pr.get("body", ""))
        author = pr.get("author", "Unknown")
        issue_hint = f"PR #{number}"

        questions.append({
            "id": next_id,
            "question": f"Why was {issue_hint} opened to {title.lower()}?",
            "ground_truth": f"{issue_hint}, titled '{title}', describes the change and its motivation. {body}".strip(),
            "expected_context_keywords": [str(number), title.split()[0] if title else str(number), author],
        })
        next_id += 1

        questions.append({
            "id": next_id,
            "question": f"What change does {issue_hint} introduce?",
            "ground_truth": f"{issue_hint} introduces the change described by its title: '{title}'. {body}".strip(),
            "expected_context_keywords": [str(number), title.split()[0] if title else str(number), "change"],
        })
        next_id += 1

    # Two questions per commit
    for commit in commits:
        sha = commit["sha"]
        short_sha = sha[:8]
        message = first_sentence(commit.get("message", ""))
        author = commit.get("author", "Unknown")

        questions.append({
            "id": next_id,
            "question": f"What does commit {sha} do?",
            "ground_truth": f"Commit {sha} ({short_sha}) by {author} is described as: {message}".strip(),
            "expected_context_keywords": [short_sha, author, message.split()[0] if message else short_sha],
        })
        next_id += 1

        questions.append({
            "id": next_id,
            "question": f"Who authored commit {sha}?",
            "ground_truth": f"Commit {sha} was authored by {author}. The commit message is: {message}".strip(),
            "expected_context_keywords": [short_sha, author],
        })
        next_id += 1

    # Five questions per issue
    for issue in issues:
        number = issue["number"]
        title = clean_title(issue.get("title", ""))
        body = first_sentence(issue.get("body", ""))
        author = issue.get("author", "Unknown")

        questions.append({
            "id": next_id,
            "question": f"What issue does #{number} report?",
            "ground_truth": f"Issue #{number}, opened by {author}, reports: {title}. {body}".strip(),
            "expected_context_keywords": [str(number), title.split()[0] if title else str(number), author],
        })
        next_id += 1

        questions.append({
            "id": next_id,
            "question": f"What user-facing failure is described in issue #{number}?",
            "ground_truth": f"Issue #{number} describes the user-facing failure: {body}".strip(),
            "expected_context_keywords": [str(number), "404", "fallback"],
        })
        next_id += 1

        questions.append({
            "id": next_id,
            "question": f"Which behavior or path pattern is affected by issue #{number}?",
            "ground_truth": f"Issue #{number} is about the behavior described in the title: {title}. {body}".strip(),
            "expected_context_keywords": [str(number), title.split()[0] if title else str(number)],
        })
        next_id += 1

        questions.append({
            "id": next_id,
            "question": f"What symptom is reported in issue #{number}?",
            "ground_truth": f"Issue #{number} reports the symptom described in its body: {body}".strip(),
            "expected_context_keywords": [str(number), "symptom", "404"],
        })
        next_id += 1

        questions.append({
            "id": next_id,
            "question": f"Who opened issue #{number}?",
            "ground_truth": f"Issue #{number} was opened by {author}. {title}".strip(),
            "expected_context_keywords": [str(number), author],
        })
        next_id += 1

    if len(questions) != 50:
        raise ValueError(f"Expected 50 benchmark questions, got {len(questions)}")

    output_dir = config.PROJECT_ROOT / "evaluation"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "benchmark_50.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(questions)} questions benchmark in {output_file}")


if __name__ == "__main__":
    build_benchmark()
