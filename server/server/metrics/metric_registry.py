from typing import Callable, Union, Dict

# from codegen_metrics import chrf
from server.metrics.sacrebleu_compat import sentence_chrf

def chrf(reference_code: str, generated_code: str) -> float:
    # print("reference code:\n", reference_code, "\n")
    # print("generated code:\n", generated_code, "\n")
    score = sentence_chrf(generated_code, [reference_code]).score / 100
    # print("chrf score:", score)
    return score

MetricScore = Union[float, int]

MetricType = Callable[[str, str], MetricScore]



def stripped_chrf(reference_code: str, generated_code: str) -> float:
    return chrf(reference_code=reference_code.strip(), generated_code=generated_code.strip())


METRIC_REGISTRY: Dict[str, MetricType] = {
    'chrf': chrf,
    'stripped_chrf': stripped_chrf
    # Add more metrics here. Names must be in lower-case.
}

def parse_metric(metric_name: str) -> MetricType:
    normalized_metric_name = metric_name.lower()
    if normalized_metric_name in METRIC_REGISTRY:
        return METRIC_REGISTRY[normalized_metric_name]
    raise ValueError(f'Unknown metric "{metric_name}". Choose one from {list(METRIC_REGISTRY.keys())}')
