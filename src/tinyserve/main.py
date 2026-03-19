from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from tinyserve.model_runner import ModelRunner
from tinyserve.schemas import GenerateRequest, GenerateResponse, HealthResponse

runner = ModelRunner.from_env()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await runner.start()
    try:
        yield
    finally:
        await runner.shutdown()


app = FastAPI(
    title="TinyServe",
    description="Phase 2 queue-based scheduler with dynamic batching",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return runner.health()


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        return await runner.generate(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc
