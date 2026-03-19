import json
import logging.config
import yaml
from os import environ as env
from os import path
from tempfile import gettempdir
from time import sleep

import requests
from dotenv import load_dotenv

from evalai import EvalAI
from evaluate import evaluate

load_dotenv()

with open("logging.yaml", "rt") as buffer:
    config = yaml.safe_load(buffer.read())
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)

# Remote Evaluation Meta Data
# See https://evalai.readthedocs.io/en/latest/evaluation_scripts.html#writing-remote-evaluation-script
evalai_server = env.get("EVAL_AI_SERVER", "https://eval.ai")
evalai_auth_token = env.get("EVAL_AI_AUTH_TOKEN")
evalai_queue_name = env.get("EVAL_AI_QUEUE_NAME")
evalai_challenge_pk = env.get("EVAL_AI_CHALLENGE_PK")

if evalai_auth_token is None:
    logger.error("\"EVAL_AI_AUTH_TOKEN\" not provided. Generate one at: https://eval.ai/web/profile/auth-token!")
    exit(1)

if evalai_queue_name is None:
    logger.error("\"EVAL_AI_QUEUE_NAME\" not provided. Get it from the \"Manage\" section of the challenge page!")
    exit(1)

if evalai_challenge_pk is None:
    logger.error("\"EVAL_AI_CHALLENGE_PK\" not provided. Get it from the \"Manage\" section of the challenge page!")
    exit(1)


def download(url, directory=gettempdir()):
    response = requests.get(url)
    *_, filename = url.split("/")
    filepath = path.join(directory, filename)
    with open(filepath, "wb") as buffer:
        buffer.write(response.content)
    return filepath


def update_running(evalai, submission_pk):
    data = {
        "submission": submission_pk,
        "submission_status": "RUNNING",
    }
    logger.info("Updating to \"running\": %s", data)
    response = evalai.update_submission_status(data)
    logger.debug("Update complete: %s", response)


def update_failed(
    evalai, phase_pk, submission_pk, submission_error, stdout="", metadata=""
):
    data = {
        "challenge_phase": phase_pk,
        "submission": submission_pk,
        "stdout": stdout,
        "stderr": submission_error,
        "submission_status": "FAILED",
        "metadata": metadata,
    }
    logger.warning("Updating to \"failed\": %s", data)
    response = evalai.update_submission_data(data)
    logger.debug("Update complete: %s", response)


def update_finished(
    evalai,
    phase_pk,
    submission_pk,
    result,
    submission_error="",
    stdout="",
    metadata="",
):
    data = {
        "challenge_phase": phase_pk,
        "submission": submission_pk,
        "stdout": stdout,
        "stderr": submission_error,
        "submission_status": "FINISHED",
        "result": result,
        "metadata": metadata,
    }
    logger.info("Updating to \"finished\": %s", data)
    response = evalai.update_submission_data(data)
    logger.debug("Update complete: %s", response)


if __name__ == "__main__":
    evalai = EvalAI(
        challenge_pk=evalai_challenge_pk,
        queue_name=evalai_queue_name,
        api_token=evalai_auth_token,
        api_url=evalai_server,
    )

    while True:
        # Get the message from the queue
        message = evalai.get_message_from_sqs_queue()
        message_body = message.get("body")
        if message_body:
            submission_pk = message_body.get("submission_pk")
            challenge_pk = message_body.get("challenge_pk")
            phase_pk = message_body.get("phase_pk")
            # Get submission details -- This will contain the input file URL
            submission = evalai.get_submission_by_pk(submission_pk)
            phase = evalai.get_challenge_phase_by_pk(phase_pk)
            codename = phase.get("codename")
            status = submission.get("status")
            match status:
                case "failed" | "finished" | "cancelled":
                    receipt_handle = message.get("receipt_handle")
                    evalai.delete_message_from_sqs_queue(receipt_handle)
                    continue
                case "submitted":
                    update_running(evalai, submission_pk)
            input_file = submission.get("input_file")
            submission_file_path = download(input_file)
            try:
                evaluations = evaluate(
                    user_submission_file=submission_file_path,
                    phase_codename=codename,
                )
                evaluation, *_ = evaluations["result"]
                accuracies = evaluation[codename]
                results = [
                    {
                        "split": codename,
                        "show_to_participant": True,
                        "accuracies": accuracies,
                    },
                ]
                update_finished(
                    evalai=evalai,
                    phase_pk=phase_pk,
                    submission_pk=submission_pk,
                    result=json.dumps(results),
                )
            except requests.exceptions.HTTPError as ex:
                submission_error = ex.response.text if ex.response is not None else str(ex)
                update_failed(
                    evalai=evalai,
                    phase_pk=phase_pk,
                    submission_pk=submission_pk,
                    submission_error=submission_error,
                )
            except Exception as ex:
                update_failed(
                    evalai=evalai,
                    phase_pk=phase_pk,
                    submission_pk=submission_pk,
                    submission_error=str(ex),
                )
        sleep(60)
