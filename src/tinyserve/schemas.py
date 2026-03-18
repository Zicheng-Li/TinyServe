from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    do_sample: bool = Field(default=True)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    enable_thinking: bool = Field(default=False)


class GenerateResponse(BaseModel):
    text: str
    model: str
    device: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str | None
    loaded: bool
