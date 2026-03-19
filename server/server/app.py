import traceback

from anyio import NamedTemporaryFile, Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from os import getcwd

from server.context import correlation_id
from server.headers import X_REQUEST_ID
from server.evaluation import evaluate
from server.middleware import CorrelationIdMiddleware

# Load environment variables from .env file
load_dotenv()

cwd = getcwd()

app = FastAPI(title="ASE2025 Evaluation Server")
app.add_middleware(CorrelationIdMiddleware)

@app.get("/")
async def root():
    return {"message": "ASE2025 Evaluation Server is running (vLLM mode)"}

@app.get("/health")
async def health():
    """Health check endpoint with vLLM status."""
    from server.evaluation import _vllm_engine
    return {
        "status": "healthy",
        "vllm_loaded": _vllm_engine is not None,
        "default_model": "JetBrains/Mellum-4b-sft-python",
    }

@app.post("/evaluate")
async def evaluate_submission(
    submission_file: UploadFile = File(...),
    stage: str = Form(...),
    language: str = Form(...),
    model: str = Form(default="JetBrains/Mellum-4b-sft-python"),
):
    """
    Evaluate a submission file against the reference data using local vLLM inference.

    Args:
        submission_file: A jsonlines file containing the user's submission (with 'context' field)
        stage: The stage/phase of the evaluation (e.g., 'public', 'practice')
        language: The language of the submission ('python' or 'kotlin')
        model: HuggingFace model name for vLLM inference (default: JetBrains/Mellum-4b-sft-python)

    Returns:
        JSON with evaluation results including chrF scores
    """
    test_annotations = Path(__file__).parent / "annotations" / f"annotations-{language}-{stage}.jsonl"
    if not await test_annotations.exists():
        raise FileNotFoundError(f"Test annotation file not found: {test_annotations}")
    async with NamedTemporaryFile(prefix="ase2025-submission-", suffix=".jsonl") as temporary:
        user_annotations = Path(temporary.name)
        while chunk := await submission_file.read(size=8196):
            await temporary.write(chunk)
        await temporary.flush()
        return await evaluate(
            test_annotation_file=test_annotations,
            user_annotation_file=user_annotations,
            stage=stage,
            language=language,
            model_name=model,
        )

@app.exception_handler(Exception)
async def exception(_request: Request, ex: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": [
                {
                    ex.__class__.__name__: str(ex),
                },
            ],
        },
        headers={
            X_REQUEST_ID: correlation_id.get(),
        },
    )

if __name__ == "__main__":
    from uvicorn import run as uvicorn
    uvicorn(
        app="app:app",
        host="0.0.0.0",
        port=8000,
        log_config="logging.yaml",
        log_level="info",
        reload=True,
    )
