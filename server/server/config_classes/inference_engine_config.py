from dataclasses import dataclass, field
from typing import Optional


@dataclass
class APIEngineConfig:
    model: str
    temperature: float = 0.0
    # seed: Optional[int] = None
    top_p: Optional[float] = 1
    max_tokens: Optional[int] = None
    # max_completion_tokens: Optional[int] = None

@dataclass
class VLLMEngineConfig:
    hf_model_path: str
    prompt_max_len: Optional[int] = None
    sampling_params: dict = field(default_factory=dict)

InferenceEngineConfig = VLLMEngineConfig
