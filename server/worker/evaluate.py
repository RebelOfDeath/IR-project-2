import json
import logging

import requests

logger = logging.getLogger(__name__)

url = "http://api:8000/evaluate"


def empty_evaluation_result(language, stage):
    return {
        "result": [
            {
                f"{language}_{stage}": {
                    "Mellum ChrF": 0.0,
                    "Codestral ChrF": 0.0,
                    "Qwen-Coder ChrF": 0.0,
                    "Average ChrF": 0.0,
                },
            },
        ],
    }


def evaluate(user_submission_file, phase_codename, **metadata):
    language, stage = phase_codename.split("_", maxsplit=1)

    # Handle but do not evaluate submissions to the private phase. Required to comply with EvalAI.
    if stage == "private":
        logger.info(
            "Received a private submission, skipping (stage=%s, language=%s, metadata=%s)",
            stage, language, metadata,
        )
        return empty_evaluation_result(language, stage)

    with open(user_submission_file, "rb") as buffer:
        logger.info(
            "Submitting a query to the server (stage=%s, language=%s, metadata=%s)",
            stage, language, metadata,
        )
        try:
            response = requests.post(
                url=url,
                files={
                    "submission_file": (user_submission_file, buffer),
                },
                data={
                    "stage": stage,
                    "language": language,
                },
            )
            body = response.text
            response.raise_for_status()
            return json.loads(body)
        except Exception as ex:
            logger.exception(ex)
            raise
        finally:
            logger.info(
                "Received a response (stage=%s, language=%s, metadata=%s)",
                stage, language, metadata,
            )
            logger.info(f"%s (%s)", response.status_code, response.reason)
            for name, value in response.headers.items():
                logger.info(f"%s: %s", name, value)
            logger.info("Evaluation response body: %s", body or "<EMPTY>")
