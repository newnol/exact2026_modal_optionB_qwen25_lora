from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.llm_client import VLLMClient
from app.pipelines.type1 import solve_type1
from app.pipelines.type2 import solve_type2
from app.schemas import PredictRequest, PredictResponseItem, fallback_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.type1_llm = VLLMClient(
        settings,
        base_url=settings.resolved_type1_base_url(),
        model_name=settings.resolved_type1_model_name(),
    )
    app.state.type2_llm = VLLMClient(
        settings,
        base_url=settings.resolved_type2_base_url(),
        model_name=settings.resolved_type2_model_name(),
    )
    yield


app = FastAPI(
    title="EXACT 2026 Submission Server",
    version="0.2.0-lora-routing",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "mock_mode": settings.mock_mode,
        "type1_vllm_base_url": settings.resolved_type1_base_url(),
        "type1_model_name": settings.resolved_type1_model_name() or "auto-from-/v1/models",
        "type2_vllm_base_url": settings.resolved_type2_base_url(),
        "type2_model_name": settings.resolved_type2_model_name() or "auto-from-/v1/models",
        "type2_fallback_to_llm": settings.type2_fallback_to_llm,
    }


def _ensure_runtime_clients() -> None:
    """Initialize vLLM clients if lifespan has not run yet.

    This keeps tests and some ASGI runners safe; normal uvicorn startup uses lifespan.
    """
    settings = get_settings()
    if not hasattr(app.state, "type1_llm"):
        app.state.type1_llm = VLLMClient(
            settings,
            base_url=settings.resolved_type1_base_url(),
            model_name=settings.resolved_type1_model_name(),
        )
    if not hasattr(app.state, "type2_llm"):
        app.state.type2_llm = VLLMClient(
            settings,
            base_url=settings.resolved_type2_base_url(),
            model_name=settings.resolved_type2_model_name(),
        )


@app.post("/predict", response_model=list[PredictResponseItem])
async def predict(payload: PredictRequest) -> list[PredictResponseItem]:
    settings = get_settings()
    _ensure_runtime_clients()

    if payload.type == "type1":
        item = await solve_type1(payload, app.state.type1_llm, mock_mode=settings.mock_mode)
    elif payload.type == "type2":
        item = await solve_type2(
            payload,
            app.state.type2_llm,
            mock_mode=settings.mock_mode,
            use_llm_fallback=settings.type2_fallback_to_llm,
        )
    else:
        item = fallback_response(payload, f"Unsupported type: {payload.type}")

    # Competition requires a list even for one input query.
    return [item]


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    # Do not leak stack traces to the evaluator. Validation errors still use FastAPI defaults.
    return JSONResponse(status_code=500, content={"detail": f"Unhandled server error: {exc}"})
