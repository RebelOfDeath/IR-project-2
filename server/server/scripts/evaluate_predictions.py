import json
import logging
from os import makedirs, path
from typing import Any, Union, List, Dict
from typing_extensions import TypedDict

from server.scripts.generate_predictions import read_task_data
from server.metrics.metric_registry import parse_metric

logger = logging.getLogger(__name__)

class MetricResults(TypedDict):
    metric_name: str
    scores: List[Union[float, int]]


def read_predictions(file_path: str) -> Dict[str, Any]:
    logger.info("Reading predictions from: %s", file_path)
    with open(file_path, "r") as buffer:
        return json.load(buffer)


def write_scores(task_id: str, scores_data: dict, scores_dir: str) -> None:
    filepath = path.join(scores_dir, f"scores_{task_id}.jsonl")
    makedirs(path.dirname(filepath), exist_ok=True)
    logger.info("Writing metrics to: %s", filepath)
    with open(filepath, "a", encoding="utf-8") as buffer:
        json.dump(scores_data, buffer)
        buffer.write("\n")


def evaluate_predictions(predictions_file: str, scores_dir: str, metric_name: str = 'chrf'):
    metric = parse_metric(metric_name)

    predictions_data = read_predictions(predictions_file)
    task_data_filename = predictions_data['task_data_filename']
    task_data = read_task_data(task_data_filename)

    scores = list()
    predictions = predictions_data['predictions']
    ground_truth = [dp.completion_snippet for dp in task_data]
    for pred, gt in zip(predictions, ground_truth):
        if pred is None:
            pred = ''
        scores.append(metric(gt, pred))

    scores_data = dict(metric=metric_name, scores=scores)
    write_scores(predictions_data['task_id'], scores_data, scores_dir)
    logger.info("Mean %s score is %s", metric_name, sum(scores) / len(scores))



def main(args):
    evaluate_predictions(args.predictions_file, args.scores_dir, args.metric_name)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Evaluate predictions using specified metric')
    parser.add_argument('--predictions-file', required=True, help='Path to the predictions file')
    parser.add_argument('--scores-dir', required=True, help='Path to the scores file')
    parser.add_argument('--metric-name', '-m', default='chrf', help='Metric to use for evaluation')

    main(parser.parse_args())
