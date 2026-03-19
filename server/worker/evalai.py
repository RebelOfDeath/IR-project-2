import logging
from enum import StrEnum
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

class EvalAIEndpoint(StrEnum):
    def __new__(cls, value):
        obj = str.__new__(cls, f"/api{value}")
        obj._value_ = f"/api{value}"
        return obj

    JOBS_CHALLENGE_QUEUES = "/jobs/challenge/queues/{}/"
    JOBS_CHALLENGE_UPDATE_SUBMISSION = "/jobs/challenge/{}/update_submission/"
    JOBS_QUEUES = "/jobs/queues/{}/"
    JOBS_SUBMISSION = "/jobs/submission/{}"
    CHALLENGES_CHALLENGE_PHASE = "/challenges/challenge/phase/{}"


class EvalAI:
    """
    Used for facilitating communication with an EvalAI backend.
    """

    def __init__(self, challenge_pk, queue_name, api_token, api_url = "https://eval.ai"):
        self._api_token = api_token
        self._api_url = api_url
        self._queue_name = queue_name
        self._challenge_pk = challenge_pk

    def _make_request(self, endpoint, method="GET", data=None):
        headers = {
            "Authorization": f"Bearer {self._api_token}"
        }
        try:
            url = urljoin(self._api_url, endpoint)
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
            )
            response.raise_for_status()
            logger.debug("Response: %s", response.text)
        except requests.exceptions.RequestException as ex:
            logger.exception(ex)
            raise
        return response.json()

    def get_message_from_sqs_queue(self):
        """
        Get a submission message from the queue.

        Docs: https://eval.ai/api/docs/#operation/get_submission_message_from_queue
        """
        return self._make_request(
            endpoint=EvalAIEndpoint.JOBS_CHALLENGE_QUEUES.format(self._queue_name),
        )

    def delete_message_from_sqs_queue(self, receipt_handle):
        """
        Delete a submission message from the queue.

        Docs: https://eval.ai/api/docs/#operation/delete_submission_message_from_queue
        """
        return self._make_request(
            endpoint=EvalAIEndpoint.JOBS_QUEUES.format(self._queue_name),
            method="POST",
            data={
                "receipt_handle": receipt_handle,
            },
        )

    def update_submission_data(self, data):
        """
        Update the submission data.

        Docs: https://eval.ai/api/docs/#operation/update_submission
        """
        return self._make_request(
            endpoint=EvalAIEndpoint.JOBS_CHALLENGE_UPDATE_SUBMISSION.format(self._challenge_pk),
            method="PUT",
            data=data,
        )

    def update_submission_status(self, data):
        """
        Update the submission status.

        Docs: https://eval.ai/api/docs/#operation/update_submission
        """
        return self._make_request(
            endpoint=EvalAIEndpoint.JOBS_CHALLENGE_UPDATE_SUBMISSION.format(self._challenge_pk),
            method="PATCH",
            data=data,
        )

    def get_submission_by_pk(self, submission_pk):
        return self._make_request(
            endpoint=EvalAIEndpoint.JOBS_SUBMISSION.format(submission_pk),
        )

    def get_challenge_phase_by_pk(self, phase_pk):
        return self._make_request(
            endpoint=EvalAIEndpoint.CHALLENGES_CHALLENGE_PHASE.format(phase_pk),
        )
