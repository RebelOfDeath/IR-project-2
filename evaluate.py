"""
Simple evaluation pipeline for code completion using Ollama + chrF metric.

Usage:
    # 1. Start Ollama:  docker compose up -d
    # 2. Pull the model: docker exec ollama ollama pull mellum
    # 3. Run evaluation:
    #    python evaluate.py --predictions predictions/python-practice-simple_hybrid.jsonl
    #    python evaluate.py  # uses default prediction file if only one exists

Reads prediction JSONL (context, optional prefix/suffix overrides),
original data JSONL (prefix, suffix), and answers JSONL (middle / ground truth).
Sends FIM prompts to Ollama, computes chrF, reports results.
"""
import argparse
import json
import os
import sys
import time
from statistics import mean
from typing import List, Tuple

import jsonlines
import requests
import sacrebleu
from tqdm import tqdm

# ---------------------------------------------------------------------------
# chrF metric (matches server behaviour: sentence_chrf, score / 100)
# ---------------------------------------------------------------------------

def compute_chrf(reference: str, hypothesis: str) -> float:
    """Compute chrF score between reference and hypothesis. Returns 0-1."""
    score = sacrebleu.sentence_chrf(hypothesis, [reference]).score / 100
    return score


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "JetBrains/Mellum-4b-sft-python")

# Mellum FIM tokens
FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"
FILE_SEPARATOR = "<filename>"
DEFAULT_FILE_SEP = "<|file_sep|>"

# Generation parameters (match server defaults)
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 128
MODEL_CONTEXT_WINDOW = 8192
MAX_PROMPT_TOKENS = MODEL_CONTEXT_WINDOW - MAX_NEW_TOKENS - 50  # Buffer for safety
STOP_TOKENS = [
    "<filename>",
    "<fim_suffix>",
    "<|endoftext|>",
    "<fim_middle>",
    "<fim_prefix>",
]


def format_fim_prompt(prefix: str, suffix: str, context: str = "") -> str:
    """Format a FIM prompt in Mellum format, truncating context if needed."""
    if context:
        context = context.replace(DEFAULT_FILE_SEP, FILE_SEPARATOR)

    # Build FIM parts that must be preserved
    # Mellum expects S-P-M order: <fim_suffix>suffix<fim_prefix>prefix<fim_middle>
    fim_part = f"{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}"

    # Estimate tokens: Mellum uses ~2.2 chars per token for code (measured empirically)
    # Use conservative estimate to avoid truncation of FIM tokens
    CHARS_PER_TOKEN = 2.2
    fim_tokens = len(fim_part) / CHARS_PER_TOKEN
    context_tokens = len(context) / CHARS_PER_TOKEN
    total_tokens = fim_tokens + context_tokens

    # If we're over budget, truncate context from the start
    if total_tokens > MAX_PROMPT_TOKENS:
        # Calculate how much context we can keep (in characters)
        available_context_tokens = MAX_PROMPT_TOKENS - fim_tokens
        if available_context_tokens > 0:
            max_context_chars = int(available_context_tokens * CHARS_PER_TOKEN)
            # Take from the END of context (most recent code is most relevant)
            context = context[-max_context_chars:]
        else:
            context = ""

    return f"{context}{fim_part}"


def ollama_generate(prompt: str) -> str:
    """Call Ollama /api/generate and return the completion text."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_NEW_TOKENS,
                "num_ctx": MODEL_CONTEXT_WINDOW,  # Set context window explicitly
                "stop": STOP_TOKENS,
                "num_keep": -1,  # Enable left truncation by keeping only the most recent tokens
            },
        },
        timeout=300,
    )
    resp.raise_for_status()
    resp_json = resp.json()
    text = resp_json.get("response", "")

    # Log if response is empty for debugging
    if not text:
        prompt_len = resp_json.get("prompt_eval_count", 0)
        eval_count = resp_json.get("eval_count", 0)
        print(f"\nWarning: Empty response. Prompt tokens: {prompt_len}, Generated tokens: {eval_count}", file=sys.stderr)

    # Cut on any special tokens that leaked through
    for tok in [FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE, FILE_SEPARATOR, "<|endoftext|>"]:
        if tok in text:
            text = text[: text.index(tok)]
    return text


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def load_data(
    predictions_path: str,
    data_dir: str,
    stage: str,
    lang: str,
) -> List[Tuple[str, str, str, str]]:
    """
    Load and merge prediction file, original data, and answers.

    Returns list of (prompt, ground_truth_middle, sample_id, context_snippet).
    """
    data_file = os.path.join(data_dir, f"{lang}-{stage}.jsonl")
    answers_file = os.path.join(data_dir, f"answers-{lang}-{stage}.jsonl")

    with jsonlines.open(data_file) as r:
        data_points = list(r)
    with jsonlines.open(answers_file) as r:
        answers = list(r)
    with jsonlines.open(predictions_path) as r:
        predictions = list(r)

    assert len(data_points) == len(answers) == len(predictions), (
        f"Mismatched lengths: data={len(data_points)}, "
        f"answers={len(answers)}, predictions={len(predictions)}"
    )

    samples = []
    for dp, ans, pred in zip(data_points, answers, predictions):
        # Use prediction's prefix/suffix override if provided, else original
        prefix = pred.get("prefix", dp.get("prefix", ""))
        suffix = pred.get("suffix", dp.get("suffix", ""))
        context = pred.get("context", "")
        ground_truth = ans["middle"]
        sample_id = dp.get("id", "?")

        prompt = format_fim_prompt(prefix, suffix, context)
        samples.append((prompt, ground_truth, sample_id, context[:80]))

    return samples


def run_evaluation(
    predictions_path: str,
    data_dir: str = "data",
    stage: str = "practice",
    lang: str = "python",
    output_path: str | None = None,
) -> float:
    """Run full evaluation: generate completions via Ollama, compute chrF."""
    print(f"Model: {MODEL_NAME} @ {OLLAMA_URL}")
    print(f"Predictions: {predictions_path}")
    print(f"Data: {data_dir}/{lang}-{stage}.jsonl")
    print()

    samples = load_data(predictions_path, data_dir, stage, lang)
    print(f"Loaded {len(samples)} samples\n")

    scores = []
    results = []
    start = time.time()

    for prompt, ground_truth, sample_id, ctx_snippet in tqdm(samples, desc="Evaluating"):
        generated = ollama_generate(prompt)
        score = compute_chrf(ground_truth, generated)
        scores.append(score)

        results.append({
            "id": sample_id,
            "generated": generated,
            "ground_truth": ground_truth,
            "chrf": round(score, 4),
        })

    elapsed = time.time() - start
    mean_chrf = mean(scores)

    print(f"\n{'='*50}")
    print(f"  Mean chrF: {mean_chrf:.4f}")
    print(f"  Samples:   {len(scores)}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/len(scores):.1f}s/sample)")
    print(f"{'='*50}")

    # Save detailed results
    if output_path is None:
        pred_name = os.path.splitext(os.path.basename(predictions_path))[0]
        output_path = os.path.join("results", f"{pred_name}-results.jsonl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, "w") as w:
        for r in results:
            w.write(r)

    # Also write summary
    summary_path = output_path.replace("-results.jsonl", "-summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "mean_chrf": round(mean_chrf, 4),
            "num_samples": len(scores),
            "model": MODEL_NAME,
            "predictions_file": predictions_path,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")

    return mean_chrf


def find_default_predictions() -> str | None:
    """Find a prediction file if only one exists."""
    pred_dir = "predictions"
    if not os.path.isdir(pred_dir):
        return None
    files = [f for f in os.listdir(pred_dir) if f.endswith(".jsonl")]
    if len(files) == 1:
        return os.path.join(pred_dir, files[0])
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate code completions via Ollama + chrF")
    parser.add_argument("--predictions", "-p", type=str, default=None,
                        help="Path to predictions JSONL file")
    parser.add_argument("--data-dir", "-d", type=str, default="data",
                        help="Data directory (default: data)")
    parser.add_argument("--stage", "-s", type=str, default="practice",
                        help="Stage name (default: practice)")
    parser.add_argument("--lang", "-l", type=str, default="python",
                        help="Language (default: python)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output results file path")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Ollama model name (default: mellum)")
    parser.add_argument("--ollama-url", type=str, default=None,
                        help="Ollama API URL (default: http://localhost:11434)")
    args = parser.parse_args()

    # Override globals from args
    global MODEL_NAME, OLLAMA_URL
    if args.model:
        MODEL_NAME = args.model
    if args.ollama_url:
        OLLAMA_URL = args.ollama_url

    # Find predictions file
    predictions_path = args.predictions or find_default_predictions()
    if not predictions_path:
        print("Error: No predictions file specified and couldn't find a default.")
        print("Usage: python evaluate.py --predictions <path-to-predictions.jsonl>")
        sys.exit(1)

    if not os.path.exists(predictions_path):
        print(f"Error: Predictions file not found: {predictions_path}")
        sys.exit(1)

    run_evaluation(
        predictions_path=predictions_path,
        data_dir=args.data_dir,
        stage=args.stage,
        lang=args.lang,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
