from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinyserve.config import Settings
from tinyserve.schemas import GenerateRequest, GenerateResponse, HealthResponse


@dataclass
class _ModelState:
    model: Any | None = None
    tokenizer: Any | None = None
    device: str | None = None
    loaded: bool = False


class ModelRunner:
    """
    Phase 1 runner:
    - model is loaded once at startup
    - every request calls model.generate directly
    - requests are serialized with a lock to avoid memory spikes
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = _ModelState()
        self._load_lock = asyncio.Lock()
        self._generate_lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "ModelRunner":
        return cls(settings=Settings.from_env())

    async def load(self) -> None:
        async with self._load_lock:
            if self.state.loaded:
                return
            await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self) -> None:
        device, dtype = self._pick_device()
        tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_id,
            torch_dtype=dtype,
            device_map=None,
        )
        model.to(device)
        model.eval()

        self.state.model = model
        self.state.tokenizer = tokenizer
        self.state.device = str(device)
        self.state.loaded = True

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        if len(request.prompt) > self.settings.max_input_chars:
            raise RuntimeError(
                f"prompt too long: {len(request.prompt)} chars > "
                f"{self.settings.max_input_chars} chars"
            )

        if not self.state.loaded:
            await self.load()

        async with self._generate_lock:
            start = time.perf_counter()
            result = await asyncio.to_thread(self._generate_blocking, request)
            latency_ms = (time.perf_counter() - start) * 1000.0

        return GenerateResponse(
            text=result["text"],
            model=self.settings.model_id,
            device=self.state.device or "unknown",
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
            latency_ms=round(latency_ms, 2),
        )

    def _generate_blocking(self, request: GenerateRequest) -> dict[str, Any]:
        assert self.state.model is not None
        assert self.state.tokenizer is not None
        assert self.state.device is not None

        tokenizer = self.state.tokenizer
        model = self.state.model
        device = self.state.device

        messages = [{"role": "user", "content": request.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=request.enable_thinking,
        )
        model_inputs = tokenizer([prompt_text], return_tensors="pt")
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if request.do_sample:
            generation_kwargs["temperature"] = request.temperature
            generation_kwargs["top_p"] = request.top_p

        with torch.inference_mode():
            generated = model.generate(**model_inputs, **generation_kwargs)

        input_len = model_inputs["input_ids"].shape[1]
        output_ids = generated[0][input_len:]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return {
            "text": output_text,
            "prompt_tokens": int(input_len),
            "completion_tokens": int(output_ids.shape[0]),
        }

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=self.settings.model_id,
            device=self.state.device,
            loaded=self.state.loaded,
        )

    @staticmethod
    def _pick_device() -> tuple[str, torch.dtype]:
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        if torch.cuda.is_available():
            return "cuda", torch.float16
        return "cpu", torch.float32
