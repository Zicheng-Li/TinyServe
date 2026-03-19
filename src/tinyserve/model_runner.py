from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
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


@dataclass
class _QueuedRequest:
    request_id: str
    payload: GenerateRequest
    arrival_time: float
    future: asyncio.Future[GenerateResponse]


class ModelRunner:
    """
    Phase 2 runner:
    - API handler enqueues requests
    - background scheduler pulls requests and builds batches
    - batched inference fills each request's Future
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = _ModelState()
        self._load_lock = asyncio.Lock()

        self.request_queue: asyncio.Queue[_QueuedRequest] = asyncio.Queue(
            maxsize=settings.queue_max_size
        )
        self._pending: list[_QueuedRequest] = []
        self._scheduler_task: asyncio.Task[None] | None = None
        self._accepting = True

    @classmethod
    def from_env(cls) -> "ModelRunner":
        return cls(settings=Settings.from_env())

    async def start(self) -> None:
        await self.load()
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(
                self._scheduler_loop(), name="tinyserve-scheduler"
            )

    async def shutdown(self) -> None:
        self._accepting = False
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None

        while not self.request_queue.empty():
            item = self.request_queue.get_nowait()
            if not item.future.done():
                item.future.set_exception(RuntimeError("server is shutting down"))

        for item in self._pending:
            if not item.future.done():
                item.future.set_exception(RuntimeError("server is shutting down"))
        self._pending.clear()

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
        if not self._accepting:
            raise RuntimeError("server is shutting down")

        await self.start()

        loop = asyncio.get_running_loop()
        response_future: asyncio.Future[GenerateResponse] = loop.create_future()
        queued_request = _QueuedRequest(
            request_id=uuid.uuid4().hex,
            payload=request,
            arrival_time=time.perf_counter(),
            future=response_future,
        )
        await self.request_queue.put(queued_request)
        return await response_future

    async def _scheduler_loop(self) -> None:
        while True:
            batch = await self._collect_batch()
            if not batch:
                continue

            try:
                responses = await asyncio.to_thread(self._infer_batch_blocking, batch)
            except Exception as exc:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(RuntimeError(f"inference failed: {exc}"))
                continue

            for req, response in zip(batch, responses):
                if not req.future.done():
                    req.future.set_result(response)

    async def _collect_batch(self) -> list[_QueuedRequest]:
        first = await self._pull_next_request()
        batch = [first]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (self.settings.max_batch_wait_ms / 1000.0)

        while len(batch) < self.settings.max_batch_size:
            pending_idx = self._find_compatible_pending_index(first)
            if pending_idx is not None:
                batch.append(self._pending.pop(pending_idx))
                continue

            timeout = deadline - loop.time()
            if timeout <= 0:
                break

            try:
                candidate = await asyncio.wait_for(self.request_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break

            if self._is_compatible(first, candidate):
                batch.append(candidate)
            else:
                self._pending.append(candidate)

        return batch

    async def _pull_next_request(self) -> _QueuedRequest:
        if self._pending:
            return self._pending.pop(0)
        return await self.request_queue.get()

    def _find_compatible_pending_index(self, target: _QueuedRequest) -> int | None:
        for idx, candidate in enumerate(self._pending):
            if self._is_compatible(target, candidate):
                return idx
        return None

    @staticmethod
    def _is_compatible(a: _QueuedRequest, b: _QueuedRequest) -> bool:
        if a.payload.do_sample != b.payload.do_sample:
            return False
        if a.payload.enable_thinking != b.payload.enable_thinking:
            return False
        if not a.payload.do_sample:
            return True

        temperature_close = abs(a.payload.temperature - b.payload.temperature) < 1e-6
        top_p_close = abs(a.payload.top_p - b.payload.top_p) < 1e-6
        return temperature_close and top_p_close

    def _infer_batch_blocking(
        self, requests: list[_QueuedRequest]
    ) -> list[GenerateResponse]:
        assert self.state.model is not None
        assert self.state.tokenizer is not None
        assert self.state.device is not None

        tokenizer = self.state.tokenizer
        model = self.state.model
        device = self.state.device

        prompt_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": req.payload.prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=req.payload.enable_thinking,
            )
            for req in requests
        ]

        model_inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        max_new_tokens = max(req.payload.max_new_tokens for req in requests)
        first = requests[0].payload
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": first.do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if first.do_sample:
            generation_kwargs["temperature"] = first.temperature
            generation_kwargs["top_p"] = first.top_p

        with torch.inference_mode():
            outputs = model.generate(**model_inputs, **generation_kwargs)

        attention_mask = model_inputs["attention_mask"]
        responses: list[GenerateResponse] = []
        finished_at = time.perf_counter()

        for idx, req in enumerate(requests):
            input_len = int(attention_mask[idx].sum().item())
            max_tokens_for_request = req.payload.max_new_tokens
            output_ids = outputs[idx][input_len : input_len + max_tokens_for_request]
            output_text = tokenizer.decode(
                output_ids.detach().cpu(), skip_special_tokens=True
            ).strip()
            completion_tokens = int(output_ids.shape[0])

            responses.append(
                GenerateResponse(
                    text=output_text,
                    model=self.settings.model_id,
                    device=device,
                    prompt_tokens=input_len,
                    completion_tokens=completion_tokens,
                    total_tokens=input_len + completion_tokens,
                    latency_ms=round((finished_at - req.arrival_time) * 1000.0, 2),
                )
            )

        return responses

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=self.settings.model_id,
            device=self.state.device,
            loaded=self.state.loaded,
            queue_size=self.request_queue.qsize(),
            pending_size=len(self._pending),
            scheduler_running=self._scheduler_task is not None
            and not self._scheduler_task.done(),
        )

    @staticmethod
    def _pick_device() -> tuple[str, torch.dtype]:
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        if torch.cuda.is_available():
            return "cuda", torch.float16
        return "cpu", torch.float32
