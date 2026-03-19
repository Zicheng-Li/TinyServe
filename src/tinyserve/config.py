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
    cache_implementation: str

    @staticmethod
    def from_env() -> "Settings":
        cache_implementation = (
            os.getenv("TINYSERVE_CACHE_IMPLEMENTATION", "dynamic").strip().lower()
        )
        supported_cache_impls = {"dynamic", "static"}
        if cache_implementation not in supported_cache_impls:
            raise ValueError(
                "TINYSERVE_CACHE_IMPLEMENTATION must be one of: "
                "dynamic, static"
            )

        return Settings(
            model_id=os.getenv("TINYSERVE_MODEL_ID", "Qwen/Qwen3-1.7B"),
            max_input_chars=int(os.getenv("TINYSERVE_MAX_INPUT_CHARS", "12000")),
            max_batch_size=int(os.getenv("TINYSERVE_MAX_BATCH_SIZE", "4")),
            max_batch_wait_ms=int(os.getenv("TINYSERVE_MAX_BATCH_WAIT_MS", "50")),
            queue_max_size=int(os.getenv("TINYSERVE_QUEUE_MAX_SIZE", "256")),
            cache_implementation=cache_implementation,
        )
