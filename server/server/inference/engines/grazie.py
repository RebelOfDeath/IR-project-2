import logging
import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from server.inference.engines.backoff import retry_backoff
from server.inference.engines.truncation_utils import TokenizerAwareTruncator, cut_on_special_tokens
from server.config_classes.context_config import DEFAULT_FILE_SEP
from server.data_classes.completion_datapoints import CompletionDataPoint
from server.inference.engines.base import BaseEngineOutput

logger = logging.getLogger(__name__)


class StagingClient:
    def __init__(self, language: str):
        self.url = f"https://ase.{language}.jet.stgn.gke.ai.intellij.net/service/v5/complete/v2"
        self.profile = f"jet-{language}-medium"
        self.token = os.getenv('GRAZIE_JWT_TOKEN_STGN')
        if not self.token:
            raise ValueError("No API key provided and GRAZIE_JWT_TOKEN_STGN environment variable not set")
        self.headers = {
            "Content-Type": "application/json",
            "Grazie-Authenticate-JWT": self.token,
        }
        self.client = httpx.AsyncClient(headers=self.headers)

    async def aclose(self):
        await self.client.aclose()

    async def complete(self, prefix: str, suffix: str, filename: str, context: str, max_length: int) -> str:
        payload = {
            "profile": self.profile,
            "prefix": prefix,
            "suffix": suffix,
            "filepath": filename,
            "context": [{
                "content": context,
                "type": None,
                "filepath": None,
            }],
            "max_length": max_length,
            "stop_token": None,
        }
        resp = await self.client.post(
            url=self.url,
            headers=self.headers,
            json=payload,
            timeout=10.0,
        )
        resp.raise_for_status()
        json = resp.json()
        return json.get("completion") or json.get("raw_completion") or ""


load_dotenv()

class GrazieV2Engine:
    """Mellum model from Staging exposed by the completion team
    """
    FILE_SEPARATOR = "<filename>"
    SPECIAL_TOKENS = ["<filename>", "<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    MAX_TOKENS = 8_192  # TODO: decide on that
    # MAX_NEW_TOKENS = 384
    RETRIES = 10

    def __init__(self, language: str):
        self.language = language
        self.truncator = TokenizerAwareTruncator(engine_name='grazie', model_name=self.language)
        self.kt_client = StagingClient(language='kt')
        self.py_client = StagingClient(language='py')

    async def aclose(self):
        await self.kt_client.aclose()
        await self.py_client.aclose()

    async def generate(self, datapoints: list[CompletionDataPoint], max_tokens: int | None = None, max_new_tokens: int | None = None) -> list[str]:
        if max_tokens is None:
            max_tokens = self.MAX_TOKENS
        if max_new_tokens is None:
            raise ValueError("max_new_tokens is not set")
            # max_new_tokens = self.MAX_NEW_TOKENS

        outputs = []
        for i, datapoint in enumerate(datapoints):
            if (i + 1) % 10 == 0:
                logger.info(f"Generating %d/%d predictions", i + 1, len(datapoints))

            context = datapoint.composed_context
            filename = datapoint.file_path
            prefix = datapoint.file_prefix if datapoint.file_prefix else ''
            suffix = datapoint.file_suffix if datapoint.file_suffix else ''
            context = context.replace(DEFAULT_FILE_SEP, self.FILE_SEPARATOR)
            truncated_context = self.truncator.truncate_context(
                prefix=prefix,
                suffix=suffix,
                context=context,
                max_tokens=max_tokens,
                max_new_tokens=max_new_tokens,
                allowable_error=100,
                reverse=True,
                file_sep=self.FILE_SEPARATOR,
            )

            if self.language == 'python':
                client = self.py_client
            elif self.language == "kotlin":
                client = self.kt_client
            else:
                raise ValueError(f"Unsupported language: {self.language}")

            r = await retry_backoff(
                self.RETRIES, 30_000, 30_000,
                lambda _: client.complete(
                    prefix=prefix,
                    suffix=suffix,
                    filename=filename,
                    context=truncated_context,
                    max_length=max_new_tokens,
                )
            )
            r_cut = cut_on_special_tokens(r, self.SPECIAL_TOKENS)
            if r_cut != r:
                logger.info("Completion contained special tokens: %s", r)

            outputs.append(GrazieEngineOutput(output=r_cut))

        return outputs


@dataclass
class GrazieEngineOutput(BaseEngineOutput):
    output: str | None = None

    def get_out_text(self) -> str:
        return self.output

    def get_out_extras(self) -> dict:
        return {'engine': 'grazie'}
