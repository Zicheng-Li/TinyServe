# Testing

Load testing script for TinyServe Phase 2.

## Quick start

Run with default settings (60 requests, concurrency 10):

```bash
python testing/load_test.py
```

## Common examples

Baseline-like run (no effective batching from server side):

```bash
python testing/load_test.py --label phase1_like --total 60 --concurrency 10 --max-new-tokens 96
```

Heavier run:

```bash
python testing/load_test.py --label phase2_120x10 --total 120 --concurrency 10 --max-new-tokens 128
```

Write full report to JSON:

```bash
python testing/load_test.py --output-json testing/report_phase2.json
```

## How to compare Phase 1 vs Phase 2 behavior

1. Restart server with:
   - `TINYSERVE_MAX_BATCH_SIZE=1`
   - `TINYSERVE_MAX_BATCH_WAIT_MS=0`
2. Run `python testing/load_test.py --label phase1_like`.
3. Restart server with:
   - `TINYSERVE_MAX_BATCH_SIZE=4`
   - `TINYSERVE_MAX_BATCH_WAIT_MS=50`
4. Run `python testing/load_test.py --label phase2_batching`.
5. Compare:
   - `tokens_per_s`
   - `req_per_s`
   - `server_latency_ms p95`

## Result:
`python testing/load_test.py --total 60 --concurrency 10 --label phase2_batching`

label=phase2_batching
ok=60 error=0 total=60 concurrency=10
elapsed_s=266.839
req_per_s=0.225
tokens_per_s=19.787
server_latency_ms p50=37936.31 p95=55087.38
end_to_end_ms p50=37947.41 p95=55096.64

`python testing/load_test.py --total 60 --concurrency 10 --label phase1_like`

label=phase1_like
ok=60 error=0 total=60 concurrency=10
elapsed_s=308.434
req_per_s=0.195
tokens_per_s=17.119
server_latency_ms p50=50815.63 p95=52029.83
end_to_end_ms p50=50818.68 p95=52032.57