from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenizationConfig:
    repo_name_identifier: str
    max_length: Optional[int]
    context_join_identifier: str = '\n\n'
    fim_prefix_identifier: str = ''
    fim_suffix_identifier: str = ''
    fim_middle_identifier: str = ''
