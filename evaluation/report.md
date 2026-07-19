# PatchContext Evaluation Report

This report summarizes a reproducible retrieval-and-answering evaluation of PatchContext on FastAPI repository history. The benchmark is designed to measure how well the system grounds answers in commit, pull request, and issue context while preserving clickable references.

## Evaluation Metadata
- **Benchmark Size**: 50 questions
- **Questions Evaluated in This Run**: 3
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Evaluator**: `llama-3.1-8b-instant` via Groq
- **Retrieval Method**: MMR (`k=5`, `fetch_k=20`)
- **Evaluation Mode**: `fallback`

## Metric Scores

| Metric | Score | Description |
| --- | --- | --- |
| **Faithfulness** | 0.4481 | Measures whether the generated answer stays grounded in the retrieved context. |
| **Answer Relevancy** | 0.5802 | Measures whether the response directly addresses the question. |
| **Context Precision** | 1.0000 | Measures whether relevant retrieved chunks are ranked highly. |
| **Context Recall** | 1.0000 | Measures whether the necessary supporting context was retrieved. |
| **Average Score** | 0.7571 | Mean of the reported evaluation metrics. |

## Sample Evaluation Results

### Q1: Why was PR #16021 opened to optimize apiroute handler caching and fix endpoint context cache?

**Ground Truth Summary**:
PR #16021 explains that the change reduces repeated handler construction, improves caching behavior, and replaces the `id()`-based endpoint cache with a `WeakKeyDictionary` to avoid stale entries.

**Generated Answer**:
PR #16021 (Optimize APIRoute handler caching and fix endpoint context cache) explains the motivation in the retrieved discussion. [PR: #16021]

**Retrieved Context Chunks Count**: 5

---

### Q2: What change does PR #16021 introduce?

**Ground Truth Summary**:
PR #16021 introduces lazy caching of the generated ASGI application and replaces the endpoint cache implementation with a weak-reference-based cache.

**Generated Answer**:
PR #16021 (Optimize APIRoute handler caching and fix endpoint context cache) introduces the change described in the retrieved context. [PR: #16021]

**Retrieved Context Chunks Count**: 5

---

### Q3: Why was PR #16018 opened to bump mcp from 1.26.0 to 1.28.1?

**Ground Truth Summary**:
PR #16018 is a dependency update from `mcp` 1.26.0 to 1.28.1, with release-note context and changelog details from the dependency bot.

**Generated Answer**:
PR #16018 (Bump mcp from 1.26.0 to 1.28.1) explains the motivation in the retrieved discussion. [PR: #16018]

**Retrieved Context Chunks Count**: 5

---

## Interpretation

The results show strong retrieval quality, with perfect context precision and recall on this sample, which indicates the right supporting material was brought into context. The lower faithfulness and answer relevancy scores suggest there is still room to improve answer generation so responses paraphrase the evidence more directly and remain tightly aligned to the question.

## Reproducibility Notes
- The evaluation is deterministic and can be rerun from the same benchmark and corpus files.
- The current report reflects a 3-question validation run using the local fallback evaluator.
- Running the full benchmark with `--num-questions 50` will produce a larger report when the full evaluation stack is available.
