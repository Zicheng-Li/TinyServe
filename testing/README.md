# Testing

Load testing script for TinyServe.

## Quick start

Run with default settings (60 requests, concurrency 10):

```bash
python testing/load_test.py
```

## Output metrics

The script prints:

- `req_per_s`
- `tokens_per_s`
- `server_latency_ms p95`
- `memory_mb rss_peak`
- `mps_memory_mb peak`
- `cache_implementation` and whether it was applied

## Plotting

Generate comparison charts:

```bash
python testing/plot_results.py
```

Output files:

- `testing/charts/phase1_vs_phase2.png`
- `testing/charts/cache_dynamic_vs_static.png`

## Common examples

Quick sanity run:

```bash
python testing/load_test.py --total 20 --concurrency 5 --max-new-tokens 64 --label quick
```

Save full report:

```bash
python testing/load_test.py --label dynamic --output-json testing/report_dynamic.json
```

Disable memory sampling:

```bash
python testing/load_test.py --no-sample-memory
```

## Compare KV-cache implementations

Use the same prompt/load settings for fair comparison.

1. `dynamic`:
   - Restart server with `export TINYSERVE_CACHE_IMPLEMENTATION=dynamic`
   - Run `python testing/load_test.py --label cache_dynamic --output-json testing/report_cache_dynamic.json`
2. `static`:
   - Restart server with `export TINYSERVE_CACHE_IMPLEMENTATION=static`
   - Run `python testing/load_test.py --label cache_static --output-json testing/report_cache_static.json`

Focus on:

- throughput: `tokens_per_s`
- tail latency: `server_latency_ms_p95`
- memory: `process_rss_mb_peak` and `mps_allocated_mb_peak`

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
