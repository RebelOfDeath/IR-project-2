"""
vLLM-based inference engine for local Mellum model evaluation.
Supports FIM (fill-in-the-middle) code completion.
"""
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, List, Optional

from transformers import PreTrainedTokenizerBase, AutoTokenizer

from server.config_classes.context_config import DEFAULT_FILE_SEP
from server.data_classes.completion_datapoints import CompletionDataPoint
from server.inference.engines.base import BaseEngine, BaseEngineOutput
from server.inference.engines.truncation_utils import cut_on_special_tokens

logger = logging.getLogger(__name__)


@dataclass
class VLLMEngineOutput(BaseEngineOutput):
    output: str
    cumulative_logprob: Optional[float] = None

    def get_out_text(self) -> str:
        return self.output

    def get_out_extras(self) -> dict:
        return {'engine': 'vllm', 'cumulative_logprob': self.cumulative_logprob}


class VLLMEngine(BaseEngine):
    """
    vLLM-based engine for local inference with Mellum or other HuggingFace models.

    Supports FIM (fill-in-the-middle) format for code completion.
    """

    # Mellum-specific tokens
    FILE_SEPARATOR = "<filename>"
    FIM_PREFIX = "<fim_prefix>"
    FIM_SUFFIX = "<fim_suffix>"
    FIM_MIDDLE = "<fim_middle>"
    SPECIAL_TOKENS = [FILE_SEPARATOR, FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE]

    # Default limits
    DEFAULT_MAX_TOKENS = 8192
    DEFAULT_MAX_NEW_TOKENS = 384

    def __init__(
        self,
        model_name: str = "JetBrains/Mellum-4b-sft-python",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Lazy load vLLM to avoid import errors when not using this engine
        try:
            from vllm import LLM, SamplingParams
            self._vllm_available = True
        except ImportError:
            logger.warning("vLLM not installed. Install with: pip install vllm")
            self._vllm_available = False
            return

        logger.info(f"Loading vLLM model: {model_name}")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            **kwargs
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=["\n\n", "<|endoftext|>", "<filename>"],  # Stop tokens for completion
        )

        # Load tokenizer for truncation
        self._tokenizer = self.llm.get_tokenizer()
        self._tokenizer.truncation_side = 'left'

        logger.info(f"vLLM model loaded successfully: {model_name}")

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def _format_fim_prompt(
        self,
        prefix: str,
        suffix: str,
        context: str = "",
    ) -> str:
        """
        Format input for FIM (fill-in-the-middle) completion.

        Mellum format:
        <context><fim_prefix><prefix><fim_suffix><suffix><fim_middle>
        """
        # Replace default file separator with Mellum's
        if context:
            context = context.replace(DEFAULT_FILE_SEP, self.FILE_SEPARATOR)

        prompt = f"{context}{self.FIM_PREFIX}{prefix}{self.FIM_SUFFIX}{suffix}{self.FIM_MIDDLE}"
        return prompt

    def _truncate_context(
        self,
        prefix: str,
        suffix: str,
        context: str,
    ) -> str:
        """
        Truncate context to fit within token budget.
        Preserves prefix and suffix, truncates context from the beginning.
        """
        # Calculate tokens for prefix and suffix
        prefix_tokens = len(self._tokenizer.encode(prefix))
        suffix_tokens = len(self._tokenizer.encode(suffix))

        # Reserve tokens for FIM markers and generation
        overhead = 50  # FIM tokens and safety margin
        available_for_context = self.max_tokens - prefix_tokens - suffix_tokens - self.max_new_tokens - overhead

        if available_for_context <= 0:
            logger.warning("No room for context after prefix/suffix")
            return ""

        # Reverse context blocks to prioritize most relevant (closest to completion point)
        if self.FILE_SEPARATOR in context:
            blocks = [b for b in context.split(self.FILE_SEPARATOR) if b.strip()]
            blocks = blocks[::-1]  # Reverse to prioritize later blocks
            context = self.FILE_SEPARATOR.join(blocks)

        # Truncate line by line from the beginning
        context_lines = context.splitlines(keepends=True)
        truncated_lines = []
        current_tokens = 0

        for line in reversed(context_lines):
            line_tokens = len(self._tokenizer.encode(line))
            if current_tokens + line_tokens > available_for_context:
                break
            truncated_lines.insert(0, line)
            current_tokens += line_tokens

        return "".join(truncated_lines)

    async def generate(
        self,
        datapoints: List[CompletionDataPoint],
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> List[VLLMEngineOutput]:
        """
        Generate completions for a batch of datapoints.
        """
        if not self._vllm_available:
            raise RuntimeError("vLLM is not installed")

        if max_new_tokens is not None:
            from vllm import SamplingParams
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=max_new_tokens,
                stop=["\n\n", "<|endoftext|>", "<filename>"],
            )

        # Prepare prompts
        prompts = []
        for datapoint in datapoints:
            prefix = datapoint.file_prefix or ""
            suffix = datapoint.file_suffix or ""
            context = datapoint.composed_context or ""

            # Truncate context to fit
            truncated_context = self._truncate_context(prefix, suffix, context)

            # Format as FIM prompt
            prompt = self._format_fim_prompt(prefix, suffix, truncated_context)
            prompts.append(prompt)

        # Generate completions
        logger.info(f"Generating {len(prompts)} completions with vLLM")
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Process outputs
        results = []
        for output in outputs:
            completion_text = output.outputs[0].text

            # Cut on special tokens
            completion_text = cut_on_special_tokens(completion_text, self.SPECIAL_TOKENS)

            results.append(VLLMEngineOutput(
                output=completion_text,
                cumulative_logprob=output.outputs[0].cumulative_logprob if hasattr(output.outputs[0], 'cumulative_logprob') else None
            ))

        return results

    def tokenize(self, data: List[str]) -> List[List[int]]:
        """Tokenize a list of strings."""
        return [self._tokenizer.encode(text) for text in data]


# For backward compatibility and testing
if __name__ == '__main__':
    import asyncio

    async def test():
        engine = VLLMEngine(
            model_name="JetBrains/Mellum-4b-sft-python",
            max_new_tokens=150,
        )

        # Create test datapoint
        from server.data_classes.completion_datapoints import CompletionDataPoint

        test_dp = CompletionDataPoint(
            repo_name="test/repo",
            file_path="test.py",
            file_prefix="def hello_world():\n    ",
            file_suffix="\n    return result",
            composed_context="",
            completion_snippet="result = 'Hello, World!'",
            is_fim=True,
        )

        outputs = await engine.generate([test_dp])
        for output in outputs:
            print(f"Generated: {output.get_out_text()}")

    asyncio.run(test())
