import json
import jsonlines
import logging
from asyncio import run
from dataclasses import asdict
from os import makedirs, path
from typing import List

from server.config_classes.inference_engine_config import APIEngineConfig
from server.config_classes.task_config import TaskConfig
from server.config_classes.tokenization_config import TokenizationConfig
from server.data_classes import CompletionDataPoint
from server.inference import OpenAIEngine, CodestralEngine, NebiusEngine, GrazieV2Engine

logger = logging.getLogger(__name__)

prompt_template_openai = """You are an expert programmer. Complete the code by filling in the missing middle section.

Repository: {repo_name}

Context files and information:
{composed_context:.{max_context_length}}

File to complete:
<PREFIX>
{file_prefix}
<MISSING_CODE>
{file_suffix}
</PREFIX>

Instructions:
1. Write only the missing code that should go between the prefix and suffix
2. Ensure the code flows naturally from the prefix to the suffix and is syntactically correct
3. Maintain consistent style with the existing code
4. Do not include any explanations, only output the code

Complete the missing section:
"""

def read_task_data(file_path: str) -> List[CompletionDataPoint]:
    data = list()
    with jsonlines.open(file_path, mode='r') as reader:
        for data_point in reader:
            data.append(CompletionDataPoint(**data_point))
    return data

def write_predictions(predictions: List[str], config: TaskConfig) -> None:
    predictions_data = asdict(config)
    predictions_data["predictions"] = predictions
    output_file = path.join(config.output_dir, config.prediction_data_filename)
    makedirs(config.output_dir, exist_ok=True)
    logger.info("Writing predictions to %s", output_file)
    with open(output_file, "w") as buffer:
        json.dump(predictions_data, buffer, indent=4)

def extract_code(output):
    if "```python" in output:
        return output.split("```python")[1].split("```")[0].strip()
    if "```kotlin" in output:
        return output.split("```kotlin")[1].split("```")[0].strip()
    if "```java" in output:
        return output.split("```java")[1].split("```")[0].strip()
    if "```" in output:
        return output.split("```")[1].strip()
    if "<code>" in output:
        return output.split("<code>")[1].split("</code>")[0].strip()
    return output.strip()

async def generate_predictions(task_config: TaskConfig, engine: str, model: str):
    engine_config = task_config.inference_engine_config
    data = read_task_data(task_config.task_data_filename)

    # Create appropriate engine
    if isinstance(engine_config, APIEngineConfig):
        if engine == 'openai':
            engine = OpenAIEngine(**asdict(engine_config))

            # Structured prompt template for FIM task
            input_texts = []
            for datapoint in data:
                prompt_template = prompt_template_openai.format(
                    repo_name=datapoint.repo_name,
                    composed_context=datapoint.composed_context,
                    file_prefix=datapoint.file_prefix,
                    file_suffix=datapoint.file_suffix,
                    max_context_length=task_config.max_context_length
                )
                input_texts.append(prompt_template)

            outputs = await engine.generate(input_texts=input_texts)
            for output in outputs:
                output.content = extract_code(output.content)
        elif engine == 'grazie' and model == 'mellum':
            engine = GrazieV2Engine(language=task_config.language)
            try:
                outputs = await engine.generate(data, max_tokens=None, max_new_tokens=engine_config.max_tokens)
            finally:
                await engine.aclose()
        # Mellum should probably not be under APIEngineConfig
        # TODO: support grazie models though chat API
        # https://code.jetbrains.team/p/grazi/repositories/grazie-ml/files/fdbaa7c850214ad621a3fe83896a944d6f274e96/libs/grazie_api_gateway_client/README.md?tab=preview#chat
        elif engine == 'grazie':
            pass
        elif engine == 'mistral' and model == 'codestral-2501':
            engine = CodestralEngine(**asdict(engine_config))
            outputs = await engine.generate(data, max_tokens=None, max_new_tokens=engine_config.max_tokens)
        elif engine == 'nebius' and model == 'Qwen/Qwen2.5-Coder-7B':
            engine = NebiusEngine(**asdict(engine_config))
            outputs = await engine.generate(data, max_tokens=None, max_new_tokens=engine_config.max_tokens)

    else:
        raise ValueError(f'Unsupported engine type: {type(engine_config)}')

    predictions = [output.get_out_text() for output in outputs]
    write_predictions(predictions, task_config)



def create_engine_cfg(args):
    sampling_params = {
        'temperature': args.temperature,
        'min_tokens': 15,  # That doesn't seem to work
        'max_tokens': 384,  # That is actually a max_new_tokens
    }
    return APIEngineConfig(
        model=args.model,
        temperature=sampling_params['temperature'],
        max_tokens=sampling_params['max_tokens'],
    )

async def main(args):
    _qwen_tok_cfg = TokenizationConfig(
        repo_name_identifier='<|repo_name|>',
        max_length=32,
        context_join_identifier='\n\n',
        fim_prefix_identifier='<|fim_prefix|>',
        fim_suffix_identifier='<|fim_suffix|>',
        fim_middle_identifier='<|fim_middle|>'
    )

    task_config = TaskConfig(
        # tokenization_config=_qwen_tok_cfg,
        inference_engine_config=create_engine_cfg(args),
        task_data_filename=args.data_file,
        task_id=args.task_id,
        prediction_data_filename='{task_id}.json',
        output_dir=args.output_dir,
        max_context_length=args.max_ctx_len,  # This is not used for the competition
        language=args.language,
    )

    await generate_predictions(task_config, args.engine, args.model)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Generate predictions using various engines')
    parser.add_argument('--engine', choices=['vllm', 'openai', 'grazie'], default='vllm',
                        help='Engine to use for generation')
    parser.add_argument('--model', required=True,
                        help='Model path (for vllm) or model id (for openai or grazie)')
    parser.add_argument('--data-file', required=True, help='Input data file in jsonl format')
    parser.add_argument('--task-id', default=None,
                       help='Task identifier')
    parser.add_argument('--output-dir', default='data/predictions',
                       help='Directory to save prediction outputs')
    # TODO: this should probably be computed based on model-specific max seq len instead
    parser.add_argument('--max-ctx-len', type=int, default=1000,
                       help='Maximum length for context in tokens')
    parser.add_argument('--temp', dest='temperature', type=float, default=0.0,
                       help='Temperature for generation')
    parser.add_argument('--language', type=str, help='Language split of the dataset')

    run(main(parser.parse_args()))
