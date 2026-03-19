import logging
import os
from dataclasses import dataclass
from typing import List

from mistralai import Mistral

from server.inference.engines.backoff import retry_backoff
from server.inference.engines.truncation_utils import TokenizerAwareTruncator, cut_on_special_tokens
from server.data_classes.completion_datapoints import CompletionDataPoint
from server.config_classes.tokenization_config import TokenizationConfig
from server.config_classes.context_config import DEFAULT_FILE_SEP
from server.inference.engines.base import BaseEngine, BaseEngineOutput

logger = logging.getLogger(__name__)


@dataclass
class CodestralEngineOutput(BaseEngineOutput):
    """Represents the output of a Codestral API call."""
    content: str
    extras: dict

    def get_out_text(self) -> str:
        return self.content

    def get_out_extras(self) -> dict:
        return self.extras


class CodestralEngine(BaseEngine):
    """
    Engine that calls the Mistral API for code completions from Codestral.
    """
    FILE_SEPARATOR = "+++++"
    SPECIAL_TOKENS = ["<unk>", "<pad>", "[PREFIX]", "[MIDDLE]", "[SUFFIX]"]
    MAX_TOKENS = 16_384
    # MAX_NEW_TOKENS = 384

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        **engine_args,
    ):
        """
        Initialize the Codestral engine.
        
        :param model: The model name to use (default: 'codestral-2501')
        :param api_key: Codestral API key. If None, will try to get from CODESTRAL_API_KEY env var
        :param engine_args: Additional arguments to pass to the API
        """
        if api_key is None:
            api_key = os.getenv('CODESTRAL_API_KEY')
            if not api_key:
                raise ValueError("No API key provided and CODESTRAL_API_KEY environment variable not set")

        self.client = Mistral(api_key=api_key)
        self.model = model
        self.engine_kwargs = engine_args
        self.truncator = TokenizerAwareTruncator(engine_name='mistral', model_name=self.model)
        # logger.debug("%s", engine_args)

    async def generate(self, datapoints: List[CompletionDataPoint], max_tokens: int | None = None,
                 max_new_tokens: int | None = None, ** generation_kwargs) -> List[CodestralEngineOutput]:
        """
        Generates code completions using the Codestral API.

        :param datapoints: List of CompletionDataPoint objects containing context and code
        :param **kwargs:
        :return: List of CodestralEngineOutput objects
        """
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
            prefix = datapoint.file_prefix if datapoint.file_prefix else ''
            suffix = datapoint.file_suffix if datapoint.file_suffix else ''
            filename = datapoint.file_path

            prompt, suffix = self.prepare_prompt(prefix, suffix, filename, context, max_tokens, max_new_tokens)

            response = await retry_backoff(
                10, 500, 30_000,
                lambda _: self.client.fim.complete_async(
                    model=self.model,
                    prompt=prompt,
                    suffix=suffix,
                    random_seed=42,
                    timeout_ms=10_000,
                    **self.engine_kwargs
                ))

            completion_text = response.choices[0].message.content
            extras = {
                "engine": "codestral",
                "model": self.model,
            }
            completion_text_cut = cut_on_special_tokens(completion_text, self.SPECIAL_TOKENS)
            if completion_text_cut != completion_text:
                logger.info("Completion contained special tokens: %s", completion_text)

            outputs.append(CodestralEngineOutput(
                content=completion_text_cut,
                extras=extras
            ))

        return outputs

    def tokenize(self, data: List[str], tokenization_config: TokenizationConfig) -> List[List[int]]:
        """
        Tokenization is not implemented for Codestral engine.
        """
        raise NotImplementedError('Codestral engine does not support tokenization')

    def prepare_prompt(self, prefix: str, suffix: str, filename: str, context: str, max_tokens: int, max_nex_tokens: int) -> tuple[str, str]:
        context = context.replace(DEFAULT_FILE_SEP, self.FILE_SEPARATOR)
        prefix = f"{self.FILE_SEPARATOR} {filename}\n\n{prefix}"
        truncated_context = self.truncator.truncate_context(
            prefix=prefix,
            suffix=suffix,
            context=context,
            max_tokens=max_tokens,
            max_new_tokens=max_nex_tokens,
            allowable_error=100
        )
        prompt = f"{truncated_context}\n{prefix}"
        return prompt, suffix

