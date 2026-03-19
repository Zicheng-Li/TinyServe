# TinyServe: Phase 2

This is the Phase 2 implementation of TinyServe:

- FastAPI endpoint for text generation
- Requests are enqueued into an `asyncio.Queue`
- A background scheduler builds dynamic batches and runs one batched inference
- Each request gets its own response via `asyncio.Future`

## 1) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Choose model

Default model:

```bash
export TINYSERVE_MODEL_ID="Qwen/Qwen3-1.7B"
```

Optional input guardrail:

```bash
export TINYSERVE_MAX_INPUT_CHARS=12000
```

Batch scheduler controls:

```bash
export TINYSERVE_MAX_BATCH_SIZE=4
export TINYSERVE_MAX_BATCH_WAIT_MS=50
export TINYSERVE_QUEUE_MAX_SIZE=256
```

## 3) Run server

```bash
uvicorn tinyserve.main:app --host 0.0.0.0 --port 8000 --app-dir src
```

## 4) Test

Health check:

```bash
curl http://localhost:8000/health
```

Generation:

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain what dynamic batching is in simple terms.",
    "max_new_tokens": 128,
    "do_sample": true,
    "temperature": 0.7,
    "top_p": 0.8,
    "enable_thinking": false
  }'
```

## Notes

- First startup will download model weights from Hugging Face and can take time.
- For consistent latency/throughput baseline, keep `enable_thinking=false`.
- On Apple Silicon, the service uses `mps` if available.
- `/health` now includes queue and scheduler status fields.

## 5) Load testing

```bash
python testing/load_test.py --total 60 --concurrency 10 --label phase2_batching
```

More examples:

- `testing/README.md`
