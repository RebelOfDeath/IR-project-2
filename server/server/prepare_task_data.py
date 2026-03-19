from typing import Any, Dict

import jsonlines

from server.invalid_submission import InvalidSubmissionError


def prepare_task(
    datapoint: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    # Load defaults from datapoint, fail if missing
    try:
        repo_name = datapoint["repo"]
        file_path = datapoint["path"]
        completion_snippet = datapoint["middle"]
        file_prefix = datapoint["prefix"]
        file_suffix = datapoint["suffix"]
    except KeyError as ex:
        key, *_ = ex.args
        raise AssertionError(
            f"Required key does not exist: {key}"
        ) from ex

    # Use "prefix" if submitted, otherwise fall back to datapoint
    file_prefix = context.get("prefix", file_prefix)
    # Use "suffix" if submitted, otherwise fall back to datapoint
    file_suffix = context.get("suffix", file_suffix)

    # Load context from the submission, fail if missing
    try:
        composed_context = context["context"]
    except KeyError as ex:
        key, *_ = ex.args
        raise InvalidSubmissionError(
            f"Required key does not exist: {key}"
        ) from ex

    return {
        "repo_name": repo_name,
        "file_path": file_path,
        "composed_context": composed_context,
        "file_prefix": file_prefix,
        "completion_snippet": completion_snippet,
        "file_suffix": file_suffix,
    }


def prepare_task_data(test_annotation_file, user_annotation_file):
    with (
        jsonlines.open(test_annotation_file, mode="r") as test_reader,
        jsonlines.open(user_annotation_file, mode="r") as user_reader,
    ):
        test_datapoints = [datapoint for datapoint in test_reader]
        user_datapoints = [datapoint for datapoint in user_reader]

    if len(test_datapoints) != len(user_datapoints):
        raise InvalidSubmissionError(
            f"Number of samples in the dataset ({len(test_datapoints)}) "
            f"and submitted contexts ({len(user_datapoints)}) must be equal!"
        )

    return [
        prepare_task(datapoint, context)
        for datapoint, context
        in zip(
            test_datapoints,
            user_datapoints,
            strict=True,
        )
    ]
