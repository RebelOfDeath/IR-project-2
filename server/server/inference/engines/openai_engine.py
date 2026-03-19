import logging
import os
from dataclasses import dataclass
from typing import Optional, List

from openai import AsyncOpenAI as OpenAI

from server.inference.engines.backoff import retry_backoff
from server.config_classes.tokenization_config import TokenizationConfig
from server.inference.engines.base import BaseEngine, BaseEngineOutput

logger = logging.getLogger(__name__)


@dataclass
class OpenAIEngineOutput(BaseEngineOutput):
    """Represents the output of an OpenAI API call."""
    content: str
    extras: dict

    def get_out_text(self) -> str:
        return self.content

    def get_out_extras(self) -> dict:
        return self.extras


class OpenAIEngine(BaseEngine):
    """
    Engine that calls the OpenAI API for text completions using the Chat API.
    """

    def __init__(
        self, 
        model: str, 
        openai_api_key: str | None = None,
        **engine_args,
    ):
        """
        :param openai_api_key: Your OpenAI API key.
        :param model: The model name (e.g., 'gpt-4-turbo-preview').
        :param openai_args: Additional kwargs to pass to client.chat.completions.create().
        """
        if openai_api_key is None:
            openai_api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.engine_kwargs = engine_args
        self.logger = logger

    async def generate(
        self,
        input_texts: Optional[List[str]] = None,
        input_ids: Optional[List[List[int]]] = None,
        **generation_kwargs
    ) -> List[OpenAIEngineOutput]:
        """
        Generates text completions from OpenAI API using the Chat API.
        """
        if input_texts is None and input_ids is None:
            raise ValueError("One of `input_texts` or `input_ids` must be not None")

        outputs = []
        for prompt in input_texts:
            response = await retry_backoff(
                10, 500, 30_000,
                lambda _: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": "You are an expert programmer. Complete the code accurately and maintain consistent style."},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=10.0,
                    **self.engine_kwargs,
                    **generation_kwargs
                ))
            completion_text = response.choices[0].message.content
            extras = {
                "engine": "openai",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            outputs.append(OpenAIEngineOutput(content=completion_text, extras=extras))
        return outputs

    def tokenize(self, data: List[str], tokenization_config: TokenizationConfig) -> List[List[int]]:
        raise NotImplementedError('OpenAI engine does not support tokenization')
