from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse

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
    description="Phase 4 with queue-based batching and SSE streaming",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def demo_page() -> FileResponse:
    return FileResponse("src/tinyserve/static/index.html")


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


@app.post("/v1/generate/stream")
async def generate_stream(request: GenerateRequest) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in runner.stream_generate_sse(request):
                yield event
        except RuntimeError as exc:
            payload = {"type": "error", "error": str(exc)}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            payload = {"type": "error", "error": f"inference failed: {exc}"}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
