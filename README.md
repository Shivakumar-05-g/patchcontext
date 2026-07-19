# PatchContext - Historical Developer RAG for FastAPI

PatchContext is a B.Tech 4th-year level Retrieval-Augmented Generation (RAG) pipeline designed to help engineers understand historical design decisions, behavior changes, and implementation rationales in the FastAPI codebase. 

Instead of guessing why a feature was introduced or how a bug fix was decided, PatchContext retrieves relevant context directly from historical git commits, pull requests, issue threads, and developer comments, generating a grounded answer complete with clickable citation links to GitHub and an NLI-based hallucination guard.

---

## 1. Problem Statement & Motivation
In large open-source repositories like FastAPI, code changes rapidly. When new developers join or existing developers refactor legacy modules, they often ask:
- *Why was this API route caching refactored this way?*
- *Why did we replace standard IDs with a WeakKeyDictionary?*
- *What thread-safety issue motivated this routing change?*

Standard LLMs do not have access to these internal repository discussions or recent commits, resulting in generic answers or complete hallucinations. PatchContext solves this by indexing historical GitHub development history and presenting a grounded, verifiable chat assistant.

---

## 2. System Architecture
The following diagram illustrates how raw repository history transitions into a grounded answer:

```
                  +-----------------------+
                  |  FastAPI GitHub Repo  |
                  +-----------+-----------+
                              | (github_loader.py)
                              v
                  +-----------+-----------+
                  |  Commits, PRs, Issues |
                  +-----------+-----------+
                              | (document_processor.py)
                              v
                  +-----------+-----------+
                  | Chunks & Metadata     |
                  +-----------+-----------+
                              | (embeddings.py & all-MiniLM-L6-v2)
                              v
                  +-----------+-----------+
                  |  FAISS Vector Store   | (Persisted Locally)
                  +-----------+-----------+
                              |
      [User Query]            |
           |                  v
           +---------> [ MMR Retrieval ] (retriever.py)
                              |
                              v
                  +-----------+-----------+
                  | Diverse Context Chunks|
                  +-----------+-----------+
                              |
                              v
                  +-----------+-----------+
                  |  Groq Chat Generator  | (rag_chain.py)
                  +-----------+-----------+
                              |
                              v
                  +-----------+-----------+
                  |   Generated Answer    |
                  +-----------+-----------+
                              |
                              v
                  +-----------+-----------+
                  | NLI Hallucination     | (hallucination_guard.py)
                  | Guard & Citations     |
                  +-----------+-----------+
                              |
                              v
                  +-----------+-----------+
                  |  Streamlit Dashboard  | (app.py)
                  +-----------------------+
```

---

## 3. Technology Stack & Key Decisions

- **Local Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)**: Used locally on CPU. Embeddings convert text into dense semantic vector representations, enabling us to search by semantic meaning rather than simple keyword matching. This model runs entirely locally with zero API cost.
- **Vector Database (FAISS)**: FAISS (Facebook AI Similarity Search) is used to perform fast similarity searches over dense vector spaces. Indexing is done once and persisted to disk in `vectorstore/faiss_index` so the web app loads it instantly without rebuilding.
- **Retrieval (Maximal Marginal Relevance - MMR)**: Rather than retrieving nearly identical documents, MMR balances relevance with context diversity. This ensures that a single query retrieves a useful mix of a pull request description, related issue comments, and commit messages.
- **Generator (Groq LLM via LangChain)**: ChatGroq (`llama-3.1-8b-instant`) executes the generation of the response using a custom system prompt. It is instructed to stick strictly to the retrieved context and output references in a specific format (`[PR: #Number]`, `[Commit: SHA]`).
- **NLI Hallucination Guard (`cross-encoder/nli-deberta-v3-small`)**: Claims in the generated answer are split into sentences and audited against the retrieved context using a local Natural Language Inference model. If a statement is classified as `neutral` or `contradiction` rather than `entailment`, it is flagged. If the local NLI model fails to load, it falls back to an LLM-based NLI evaluation on Groq.
- **Citations Extractor**: Cross-checks LLM citations (`[PR: #123]`) against the metadata of the retrieved documents. Clickable links are generated to point directly to the live GitHub resources.

---

## 4. Project Structure
```
patchcontext/
│
├── app.py                     # Streamlit dashboard UI
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .env.example               # Example environmental variables template
├── .gitignore                 # File exclusion list
│
├── data/
│   └── raw/                   # Cached raw commits, PRs, and issues JSON
│
├── vectorstore/               # Persisted local FAISS index
│
├── src/
│   ├── config.py              # Configuration manager and monkeypatches
│   ├── github_loader.py       # Ingests repository history using PyGithub
│   ├── document_processor.py  # Maps JSON data to Documents and chunks them
│   ├── embeddings.py          # Sets up HuggingFace embeddings
│   ├── vector_store.py        # Compiles and saves/loads FAISS index
│   ├── retriever.py           # MMR retrieval module
│   ├── rag_chain.py           # Configures Groq and RAG prompt flow
│   ├── citations.py           # Extracts and cross-checks citations
│   ├── hallucination_guard.py # Audits sentences using NLI classification
│   └── evaluation.py          # RAGAS evaluation execution logic
│
├── scripts/
│   ├── ingest.py              # Script to build vector index
│   ├── evaluate.py            # Script to run RAGAS benchmark
│   └── generate_benchmark.py  # Script to generate evaluation JSON
│
└── evaluation/
    ├── benchmark_50.json      # 50-question evaluation benchmark
    ├── results.json           # Raw RAGAS results
    └── report.md              # Markdown evaluation summary report
```

---

## 5. Setup & Ingestion Instructions

### Prerequisites
- Python 3.10 or higher (Tested on Python 3.13)
- Groq API Key (Get one from [Groq Console](https://console.groq.com/))
- GitHub Personal Access Token (Optional, but recommended for live data fetching)

### Installation
1. Clone the repository or navigate to the workspace directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill out your keys:
   ```bash
   copy .env.example .env
   ```
   Modify `.env`:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   GITHUB_TOKEN=your_optional_github_token_here
   ```

### Running Data Ingestion
To build the vector database, run the ingestion script. The script first checks if raw data is cached in `data/raw/`. If cached, it builds the index immediately without hitting the GitHub API.
```bash
python scripts/ingest.py
```
*(If you want to fetch new live data from GitHub, run with `python scripts/ingest.py --force-fetch`)*.

---

## 6. How to Run

### Streamlit Application Dashboard
Launch the dashboard UI using Streamlit:
```bash
streamlit run app.py
```
Open the local URL (usually `http://localhost:8501`) in your browser to ask questions and review the NLI Hallucination audit report.

### Running RAGAS Evaluation
To execute RAGAS evaluation on a subset of the 50-question benchmark:
```bash
python scripts/evaluate.py --num-questions 3
```
This writes raw results to `evaluation/results.json` and compiles a report in `evaluation/report.md`.

---

## 7. Example Questions
Try querying:
- *Why did FastAPI modify APIRoute handler caching or routing cache?*
- *Why did FastAPI refactor router route building to make it thread-safe?*
- *What is the candidate cache corruption race condition in _IncludedRouter?*

---

## 8. Limitations & Future Scope
- **Groq Rate Limits**: The on-demand service tier of Groq has a small token limit (6,000 TPM). Because RAGAS evaluates rows in parallel, running RAGAS on all 50 questions concurrently can lead to 429 RateLimitErrors. Future builds should implement concurrent rate-limiting queues.
- **NLI Context Size**: The local NLI model (`all-MiniLM` / `deberta-v3-small`) has a token context limit of 512 tokens. Large retrieved chunks must be parsed or summarized before NLI checking.
- **Repository Scope**: Currently maps a subset of recent history. Large enterprise contexts would benefit from metadata filtering prior to vector similarity matches.
