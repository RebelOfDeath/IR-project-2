import time
from dataclasses import dataclass
from typing import Optional

from server.config_classes.inference_engine_config import InferenceEngineConfig
from server.config_classes.tokenization_config import TokenizationConfig


@dataclass
class TaskConfig:
    inference_engine_config: InferenceEngineConfig
    task_data_filename: str
    output_dir: str
    max_context_length: int
    tokenization_config: Optional[TokenizationConfig] = None
    task_id: Optional[str] = None
    prediction_data_filename: str = 'predictions_{task_id}.json'
    language: str = None

    def __post_init__(self):
        if self.task_id is None:
            self.task_id = str(time.time())
        self.prediction_data_filename = self.prediction_data_filename.format(task_id=self.task_id)
