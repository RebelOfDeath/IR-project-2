"""
Evaluation module for code completion using local vLLM inference.

This module has been simplified to use only local vLLM inference with Mellum,
removing dependencies on external APIs (Grazie, Mistral, Nebius).
"""
import json
import logging
import os
from argparse import Namespace
from asyncio import gather
from pathlib import Path
from statistics import mean
from time import time
from typing import Tuple, Optional, List
from dataclasses import asdict

import jsonlines

from server.invalid_submission import InvalidSubmissionError
from server.scripts.evaluate_predictions import main as evaluate_predictions_main
from server.prepare_task_data import prepare_task_data
from server.config_classes.task_config import TaskConfig
from server.config_classes.inference_engine_config import APIEngineConfig
from server.data_classes import CompletionDataPoint

logger = logging.getLogger(__name__)

# Global vLLM engine instance (singleton to avoid reloading model)
_vllm_engine = None


def get_vllm_engine(model_name: str = "JetBrains/Mellum-4b-sft-python"):
    """Get or create the global vLLM engine instance."""
    global _vllm_engine
    if _vllm_engine is None:
        from server.inference.engines.vllm import VLLMEngine
        logger.info(f"Initializing vLLM engine with model: {model_name}")
        _vllm_engine = VLLMEngine(
            model_name=model_name,
            max_tokens=8192,
            max_new_tokens=384,
            temperature=0.0,
        )
    return _vllm_engine


def prepare_directories():
    """Create necessary directories for evaluation."""
    output_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(output_dir, exist_ok=True)
    datapoints_dir = os.path.join(output_dir, "datapoints")
    os.makedirs(datapoints_dir, exist_ok=True)
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    scores_dir = os.path.join(output_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    return datapoints_dir, predictions_dir, scores_dir


def write_task_data(datapoints_dir, test_annotation_file, user_annotation_file):
    """Prepare task data by merging test annotations with user submissions."""
    task_data = prepare_task_data(test_annotation_file, user_annotation_file)
    data_file = os.path.join(datapoints_dir, "datapoints.jsonl")
    with jsonlines.open(data_file, 'w') as writer:
        writer.write_all(task_data)
    return data_file


def read_task_data(file_path: str) -> List[CompletionDataPoint]:
    """Read task data from jsonl file."""
    data = []
    with jsonlines.open(file_path, mode='r') as reader:
        for data_point in reader:
            data.append(CompletionDataPoint(**data_point))
    return data


def write_predictions(predictions: List[str], task_id: str, task_data_filename: str, output_dir: str) -> str:
    """Write predictions to file."""
    output_file = os.path.join(output_dir, f"{task_id}.json")
    predictions_data = {
        "task_id": task_id,
        "task_data_filename": task_data_filename,
        "predictions": predictions,
        "output_dir": output_dir,
    }
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Writing predictions to %s", output_file)
    with open(output_file, "w") as buffer:
        json.dump(predictions_data, buffer, indent=4)
    return output_file


async def generate_predictions_vllm(
    data_file: str,
    output_dir: str,
    model_name: str = "JetBrains/Mellum-4b-sft-python",
) -> str:
    """
    Generate predictions using local vLLM inference.

    Args:
        data_file: Path to the task data file
        output_dir: Directory to save predictions
        model_name: HuggingFace model name for vLLM

    Returns:
        Path to the predictions file
    """
    task_id = str(time())

    # Read task data
    data = read_task_data(data_file)
    logger.info(f"Loaded {len(data)} datapoints for evaluation")

    # Get or create vLLM engine
    engine = get_vllm_engine(model_name)

    # Generate predictions
    outputs = await engine.generate(data, max_new_tokens=384)
    predictions = [output.get_out_text() for output in outputs]

    # Write predictions
    output_file = write_predictions(predictions, task_id, data_file, output_dir)
    return output_file


def evaluate_predictions_from_file(predictions_file: str, scores_dir: str) -> str:
    """Evaluate predictions and return scores file path."""
    filename = os.path.basename(predictions_file).replace(".json", "")
    args = Namespace(
        predictions_file=predictions_file,
        scores_dir=scores_dir,
        metric_name='chrf'
    )
    evaluate_predictions_main(args)
    scores_file = os.path.join(scores_dir, f"scores_{filename}.jsonl")
    return scores_file


def read_scores(scores_file: str) -> float:
    """Read chrF scores from file and return mean."""
    with jsonlines.open(scores_file, mode="r") as reader:
        for score in reader:
            if score["metric"] == "chrf":
                return mean(score["scores"])
    raise ValueError("No 'chrf' metric found in scores file")


def read_all_scores(scores_file: str) -> List[float]:
    """Read all individual chrF scores from file."""
    with jsonlines.open(scores_file, mode="r") as reader:
        for score in reader:
            if score["metric"] == "chrf":
                return score["scores"]
    raise ValueError("No 'chrf' metric found in scores file")


async def evaluate(
    test_annotation_file: Path,
    user_annotation_file: Path,
    stage: str,
    language: str,
    model_name: str = "JetBrains/Mellum-4b-sft-python",
    **kwargs,
):
    """
    Main evaluation function using local vLLM inference.

    Args:
        test_annotation_file: Path to the test annotation file (ground truth)
        user_annotation_file: Path to the user's submission file (contexts)
        stage: Evaluation stage (e.g., 'public', 'practice')
        language: Programming language ('python' or 'kotlin')
        model_name: HuggingFace model name for vLLM inference
        **kwargs: Additional arguments (ignored)

    Returns:
        Dict with evaluation results including chrF scores
    """
    datapoints_dir, predictions_dir, scores_dir = prepare_directories()

    try:
        data_file = write_task_data(datapoints_dir, test_annotation_file, user_annotation_file)
    except InvalidSubmissionError as ex:
        logger.error("Invalid submission file: %s", str(ex))
        raise ex

    # Generate predictions using vLLM
    logger.info(f"Generating predictions with vLLM model: {model_name}")
    predictions_file = await generate_predictions_vllm(
        data_file=data_file,
        output_dir=predictions_dir,
        model_name=model_name,
    )

    # Evaluate predictions
    scores_file = evaluate_predictions_from_file(
        predictions_file=predictions_file,
        scores_dir=scores_dir,
    )

    # Read scores
    mean_score = read_scores(scores_file)
    all_scores = read_all_scores(scores_file)

    # Format results
    results = {
        "result": [
            {
                f"{language}_{stage}": {
                    "Mellum ChrF": mean_score,
                    "Mean ChrF": mean_score,
                },
            },
        ],
        "scores": all_scores,
        "mean_chrf": mean_score,
        "model": model_name,
        "num_samples": len(all_scores),
    }

    logger.info("Evaluation results: %s", json.dumps({
        "mean_chrf": mean_score,
        "num_samples": len(all_scores),
        "model": model_name,
    }, indent=2))

    return results


# Convenience function for direct evaluation without server
async def evaluate_predictions_file(
    predictions_file: str,
    test_annotation_file: str,
    model_name: str = "JetBrains/Mellum-4b-sft-python",
) -> dict:
    """
    Evaluate a predictions file directly (useful for batch evaluation).

    Args:
        predictions_file: Path to the predictions JSONL file (with 'context' field)
        test_annotation_file: Path to test annotations
        model_name: HuggingFace model for vLLM

    Returns:
        Dict with evaluation results
    """
    from anyio import Path as AsyncPath

    return await evaluate(
        test_annotation_file=AsyncPath(test_annotation_file),
        user_annotation_file=AsyncPath(predictions_file),
        stage="eval",
        language="python",
        model_name=model_name,
    )
