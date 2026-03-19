from dataclasses import dataclass
from typing import Optional, List


@dataclass
class CompletionDataPoint:
    repo_name: str
    file_path: str
    composed_context: str
    file_prefix: str
    completion_snippet: str
    file_suffix: Optional[str] = None
    prediction: Optional[str] = None
    is_fim: bool = False

    def __post_init__(self):
        if self.file_suffix is not None:
            self.is_fim = True

    @property
    def completion_lines(self) -> List[str]:
        return self.completion_snippet.split('\n')
