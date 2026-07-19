import sys
import argparse
import json
import logging
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.evaluation import run_evaluation

# Set up logging and reconfigure encoding for stdout on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")

def parse_args():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on historical FastAPI benchmark.")
    parser.add_argument(
        "--num-questions", 
        type=int, 
        default=3, 
        help="Number of questions to evaluate from the benchmark (default: 3 to avoid rate limits)"
    )
    parser.add_argument(
        "--output-report", 
        type=str, 
        default="evaluation/report.md", 
        help="Output path for the Markdown report (default: evaluation/report.md)"
    )
    parser.add_argument(
        "--output-results", 
        type=str, 
        default="evaluation/results.json", 
        help="Output path for raw JSON results (default: evaluation/results.json)"
    )
    return parser.parse_args()

def write_markdown_report(report_path, scores, raw_data, num_eval, mode="unknown", error=None):
    report_file = Path(report_path)
    report_file.parent.mkdir(exist_ok=True, parents=True)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# PatchContext RAGAS Evaluation Report\n\n")
        if error:
            f.write("This report documents the current evaluation setup, but the scoring run did not complete successfully in this workspace.\n\n")
        elif mode == "fallback":
            f.write("This report documents a local fallback evaluation over the FastAPI repository's historical data. The evaluation was run on a subset of the 50-question benchmark and scored with deterministic heuristics because the full RAGAS stack was not available in this workspace.\n\n")
        else:
            f.write("This report documents the performance of the PatchContext RAG pipeline over the FastAPI repository's historical data. The evaluation was run on a subset of the 50-question benchmark.\n\n")
        
        f.write("## Evaluation Metadata\n")
        f.write(f"- **Evaluated Questions Count**: {num_eval} (out of 50 in benchmark)\n")
        f.write(f"- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`\n")
        f.write(f"- **LLM Evaluator**: `{config.GROQ_MODEL}` via Groq\n")
        f.write(f"- **Retrieval Method**: MMR (k={config.DEFAULT_K}, fetch_k={config.DEFAULT_FETCH_K})\n\n")
        f.write(f"- **Evaluation Mode**: `{mode}`\n\n")
        if error:
            f.write("## Run Status\n\n")
            f.write("The evaluation could not complete in this environment because the required Python packages were not available to the active interpreter.\n\n")
            f.write("### Suggested Fix\n")
            f.write("- Re-run the evaluation in a workspace where the fallback evaluator can access the raw cached GitHub data.\n")
            f.write("- If you want the full RAGAS stack, install `ragas` and `datasets` into the same Python environment that runs `python scripts/evaluate.py`.\n\n")
        else:
            f.write("## Metric Scores\n\n")
            f.write("| Metric | Score | Description |\n")
            f.write("| --- | --- | --- |\n")
            f.write(f"| **Faithfulness** | {scores.get('faithfulness', 0.0):.4f} | Measures if the generated answer is strictly grounded in the retrieved context (no hallucination). |\n")
            f.write(f"| **Answer Relevancy** | {scores.get('answer_relevancy', 0.0):.4f} | Measures if the generated answer directly addresses the user's question. |\n")
            f.write(f"| **Context Precision** | {scores.get('context_precision', 0.0):.4f} | Measures if the retrieved context ranks relevant chunks higher. |\n")
            f.write(f"| **Context Recall** | {scores.get('context_recall', 0.0):.4f} | Measures if all necessary ground truth information was successfully retrieved. |\n")
            avg_score = sum([
                scores.get("faithfulness", 0.0),
                scores.get("answer_relevancy", 0.0),
                scores.get("context_precision", 0.0),
                scores.get("context_recall", 0.0),
            ]) / 4
            f.write(f"| **Average Score** | {avg_score:.4f} | Combined overall performance metric. |\n\n")
            
            f.write("## Grounded Performance Details\n\n")
            for idx, item in enumerate(raw_data):
                f.write(f"### Q{idx+1}: {item['question']}\n\n")
                f.write(f"**Expected Ground Truth**:\n> {item['ground_truth']}\n\n")
                f.write(f"**Generated Answer**:\n> {item['answer']}\n\n")
                f.write(f"**Retrieved Context Chunks Count**: {len(item['contexts'])}\n\n")
                f.write("---\n\n")
                
            f.write("## Limitations & Reproducibility\n")
            f.write("- **Rate Limits**: Groq's API has strict rate limits. A delay of 2 seconds was added between queries, and evaluation was limited to a subset. To run the full 50-question evaluation, ensure you have an enterprise Groq tier and execute the script with `--num-questions 50`.\n")
            f.write("- **Small Context**: The historical context was built on a subset of recent FastAPI repo items due to API rate limits during Live GitHub fetching.\n")
        
    logger.info(f"Markdown evaluation report saved to {report_file}")

def main():
    args = parse_args()
    logger.info("=== Starting RAGAS Evaluation ===")
    
    # Run evaluation
    results = run_evaluation(
        num_questions=args.num_questions
    )
    
    if not results:
        logger.error("RAGAS Evaluation failed.")
        sys.exit(1)
        
    scores = results.get("scores", {})
    raw_data = results.get("raw_data", [])
    error = results.get("error")
    mode = results.get("mode", "unknown")
    
    # Convert scores to dict if it's a Ragas Result object
    if hasattr(scores, "scores") and isinstance(scores.scores, dict):
        scores_dict = scores.scores
    elif isinstance(scores, dict):
        scores_dict = scores
    else:
        scores_dict = {}
    
    # Save raw results to JSON
    results_file = Path(args.output_results)
    results_file.parent.mkdir(exist_ok=True, parents=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "status": "ok" if not error else "error",
            "mode": mode,
            "error": error,
            "scores": scores_dict,
            "raw_data": raw_data
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Raw JSON evaluation results saved to {results_file}")
    
    # Write Markdown report
    write_markdown_report(
        report_path=args.output_report,
        scores=scores_dict,
        raw_data=raw_data,
        num_eval=args.num_questions,
        mode=mode,
        error=error
    )
    
    print("\n" + "="*50)
    print("RAGAS EVALUATION COMPLETED")
    print("="*50)
    for k, v in scores_dict.items():
        print(f"{k.capitalize()}: {v:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
