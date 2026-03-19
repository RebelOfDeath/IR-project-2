import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from openai import AsyncOpenAI as OpenAI

from server.inference.engines.backoff import retry_backoff
from server.inference.engines.truncation_utils import TokenizerAwareTruncator, cut_on_special_tokens
from server.data_classes.completion_datapoints import CompletionDataPoint
from server.config_classes.tokenization_config import TokenizationConfig
from server.config_classes.context_config import DEFAULT_FILE_SEP
from server.inference.engines.base import BaseEngine, BaseEngineOutput

logger = logging.getLogger(__name__)


@dataclass
class NebiusEngineOutput(BaseEngineOutput):
    """Represents the output of a Nebius API call."""
    content: str
    extras: dict

    def get_out_text(self) -> str:
        return self.content

    def get_out_extras(self) -> dict:
        return self.extras


class NebiusEngine(BaseEngine):
    """
    Engine that calls the Nebius API for code completions from Qwen-2.5-Coder-7B.
    """
    FILE_SEPARATOR = "<|file_sep|>"
    SPECIAL_TOKENS = ["<|file_sep|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>"]
    MAX_TOKENS = 16_384
    # MAX_NEW_TOKENS = 384

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        **engine_args,
    ):
        """
        Initialize the Nebius engine.
        
        :param model: The model name to use
        :param api_key: Nebius API key. If None, will try to get from NEBIUS_API_KEY env var
        :param engine_args: Additional arguments to pass to the API
        """
        if api_key is None:
            api_key = os.getenv('NEBIUS_API_KEY')
            if not api_key:
                raise ValueError("No API key provided and NEBIUS_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key, base_url="https://api.studio.nebius.com/v1/")
        self.model = model
        self.engine_kwargs = engine_args
        self.truncator = TokenizerAwareTruncator(engine_name='nebius', model_name=self.model)

    async def generate(self, datapoints: List[CompletionDataPoint], max_tokens: int | None = None,
                 max_new_tokens: int | None = None) -> List[NebiusEngineOutput]:
        """
        Generates code completions using the Nebius API.
        
        :param datapoints: List of CompletionDataPoint objects containing context and code
        :return: List of NebiusEngineOutput objects
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

            prompt = self.prepare_prompt(prefix, suffix, filename, context, max_tokens, max_new_tokens)
            response = await retry_backoff(
                10, 500, 30_000,
                lambda _: self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    timeout=10.0,
                    seed=42,
                    stop=["<|file_sep|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>"],
                    **self.engine_kwargs
                ))
            completion_text = response.choices[0].text

            extras = {
                "engine": "nebius",
                "model": self.model,
            }
            completion_text_cut = cut_on_special_tokens(completion_text, self.SPECIAL_TOKENS)
            if completion_text_cut != completion_text:
                logger.info("Completion contained special tokens: %s", completion_text)

            outputs.append(NebiusEngineOutput(
                content=completion_text_cut,
                extras=extras
            ))

        return outputs


    def tokenize(self, data: List[str], tokenization_config: TokenizationConfig) -> List[List[int]]:
        """
        Tokenization is not implemented for Nebius engine.
        """
        raise NotImplementedError('Nebius engine does not support tokenization')

    def prepare_prompt(self, prefix: str, suffix: str, filename: str, context: str, max_tokens: int, max_nex_tokens: int) -> str:
        context = context.replace(DEFAULT_FILE_SEP, self.FILE_SEPARATOR)
        truncated_context = self.truncator.truncate_context(
            prefix=prefix,
            suffix=suffix,
            context=context,
            max_tokens=max_tokens,
            max_new_tokens=max_nex_tokens,
            allowable_error=100
        )
        # TODO: Do we need to add repo name here?
        filename = f"\n{self.FILE_SEPARATOR}{filename}\n"
        # Currently do not add the filename as it crashes the results for some reason.
        prompt = truncated_context + filename + '<|fim_prefix|>' + prefix + '<|fim_suffix|>' + suffix + '<|fim_middle|>'
        return prompt
