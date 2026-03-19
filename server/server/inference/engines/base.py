from abc import ABC, abstractmethod
from typing import List


class BaseEngineOutput(ABC):
    @abstractmethod
    def get_out_text(self) -> str:
        pass

    @abstractmethod
    def get_out_extras(self) -> dict:
        pass


class BaseEngine(ABC):
    @abstractmethod
    async def generate(self,
        input_texts: list[str] | None = None,
        input_ids: list[list[int]] | None = None,
        **generation_kwargs
    ) -> List[BaseEngineOutput]:
        pass

    @abstractmethod
    def tokenize(self, data: List[str]) -> List[List[int]]:
        pass
