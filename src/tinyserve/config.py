from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_id: str
    max_input_chars: int

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            model_id=os.getenv("TINYSERVE_MODEL_ID", "Qwen/Qwen3-1.7B"),
            max_input_chars=int(os.getenv("TINYSERVE_MAX_INPUT_CHARS", "12000")),
        )
