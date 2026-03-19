from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_id: str
    max_input_chars: int
    max_batch_size: int
    max_batch_wait_ms: int
    queue_max_size: int

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            model_id=os.getenv("TINYSERVE_MODEL_ID", "Qwen/Qwen3-1.7B"),
            max_input_chars=int(os.getenv("TINYSERVE_MAX_INPUT_CHARS", "12000")),
            max_batch_size=int(os.getenv("TINYSERVE_MAX_BATCH_SIZE", "1")),
            max_batch_wait_ms=int(os.getenv("TINYSERVE_MAX_BATCH_WAIT_MS", "0")),
            queue_max_size=int(os.getenv("TINYSERVE_QUEUE_MAX_SIZE", "256")),
        )
