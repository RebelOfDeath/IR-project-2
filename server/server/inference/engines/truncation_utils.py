import logging
import re
import string
from pathlib import Path

from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TokenizerAwareTruncator:
    def __init__(self, engine_name: str, model_name: str):
        self.engine_name = engine_name
        self.model_name = model_name
        if engine_name == 'nebius':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encode_call = lambda text: self.tokenizer(text)['input_ids']
        elif engine_name == 'mistral':
            # TODO: use .from_model?
            is_tekken = True
            self.tokenizer = MistralTokenizer.v3(is_tekken=is_tekken).instruct_tokenizer
            self.encode_call = lambda text: self.tokenizer.encode_user_message(UserMessage(content=text), available_tools=None, is_last=False, is_first=False)
        elif engine_name == 'grazie':
            tokenizer_path = Path(__file__).parent / "mellum_tokenizer" / "tokenizer-jetbrains-jet-py-medium-004-dpo-2"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.encode_call = lambda text: self.tokenizer(text, return_token_type_ids=False)['input_ids']
        else:
            raise ValueError(f"Unsupported engine: {engine_name}, model: {model_name}")

    def truncate_context(self,
                         context: str,
                         prefix: str,
                         suffix: str,
                         max_new_tokens: int,
                         max_tokens: int,
                         allowable_error: int | float = 0.1,
                         reverse: bool = False,
                         file_sep: str = None,
                         ) -> str:
        if isinstance(allowable_error, float):
            max_tokens = int(max_tokens * (1 - allowable_error))
        elif isinstance(allowable_error, int):
            max_tokens = max_tokens - allowable_error
        else:
            raise ValueError(f"allowable_error must be float or int, got {type(allowable_error)}")
        num_tokens = max_new_tokens
        num_tokens += len(self.encode_call(prefix))
        num_tokens += len(self.encode_call(suffix))
        if num_tokens > max_tokens:
            logger.warning("`num_tokens` from completion file (%s) > `max_tokens` (%s)", num_tokens, max_tokens)

        if reverse:
            if file_sep is None:
                logger.warning("file_sep is not specified, cannot reverse context")
            else:
                file_blocks = [block for block in context.split(file_sep) if len(block) > 0]
                file_blocks = file_blocks[::-1]
                file_blocks = [file_sep + block for block in file_blocks]
                context = ''.join(file_blocks)
        context_lines = context.splitlines(keepends=True)
        truncated_context = ''
        while len(context_lines) > 0:
            curr_line = context_lines.pop(-1)
            line_tokens = len(self.encode_call(curr_line))
            num_tokens += line_tokens
            if num_tokens > max_tokens:
                break
            else:
                truncated_context = curr_line + truncated_context

        return truncated_context


def cut_on_special_tokens(completion: str, special_tokens: list[str]) -> str:
    if len(special_tokens) == 0:
        return completion

    # Escape special regex characters in tokens and join with OR
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = '|'.join(escaped_tokens)

    # Find the first match
    match = re.search(pattern, completion)

    if match:
        return completion[:match.start()]

    return completion


def main():
    # truncator = TokenizerAwareTruncator('mistral', 'Qwen/Qwen2.5-Coder-7B')
    truncator = TokenizerAwareTruncator('grazie', 'Qwen/Qwen2.5-Coder-7B')
    print(truncator.truncate_context(
        context='\n'.join(list(string.ascii_letters)),
        prefix='prefix for testing\n',
        suffix='suffix for testing\n',
        max_new_tokens=100,
        max_tokens=145,
        allowable_error=0.1,
    ))

if __name__ == '__main__':
    main()
