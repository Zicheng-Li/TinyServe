from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass


@dataclass
class CallResult:
    ok: bool
    wall_ms: float
    status_code: int | None = None
    latency_ms: float | None = None
    total_tokens: int | None = None
    error: str | None = None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int((len(sorted_values) - 1) * p)
    return sorted_values[idx]


def call_generate(url: str, payload: dict, timeout_sec: float) -> CallResult:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body_raw = resp.read().decode("utf-8")
            body = json.loads(body_raw)
            wall_ms = (time.perf_counter() - start) * 1000.0
            return CallResult(
                ok=True,
                wall_ms=wall_ms,
                status_code=resp.status,
                latency_ms=float(body.get("latency_ms", 0.0)),
                total_tokens=int(body.get("total_tokens", 0)),
            )
    except urllib.error.HTTPError as exc:
        wall_ms = (time.perf_counter() - start) * 1000.0
        try:
            err_body = exc.read().decode("utf-8")
        except Exception:
            err_body = str(exc)
        return CallResult(
            ok=False,
            wall_ms=wall_ms,
            status_code=exc.code,
            error=f"HTTPError: {err_body}",
        )
    except Exception as exc:
        wall_ms = (time.perf_counter() - start) * 1000.0
        return CallResult(ok=False, wall_ms=wall_ms, error=str(exc))


def run_load_test(args: argparse.Namespace) -> dict:
    payload = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "enable_thinking": args.enable_thinking,
    }

    started_at = time.perf_counter()
    results: list[CallResult] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(call_generate, args.url, payload, args.timeout_sec)
            for _ in range(args.total)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed_s = time.perf_counter() - started_at

    ok_results = [r for r in results if r.ok]
    fail_results = [r for r in results if not r.ok]
    server_latencies = [r.latency_ms for r in ok_results if r.latency_ms is not None]
    wall_latencies = [r.wall_ms for r in ok_results]
    total_tokens = sum(r.total_tokens or 0 for r in ok_results)

    summary = {
        "label": args.label,
        "url": args.url,
        "total": args.total,
        "concurrency": args.concurrency,
        "ok": len(ok_results),
        "error": len(fail_results),
        "elapsed_s": round(elapsed_s, 3),
        "req_per_s": round((len(ok_results) / elapsed_s) if elapsed_s > 0 else 0.0, 3),
        "tokens_per_s": round((total_tokens / elapsed_s) if elapsed_s > 0 else 0.0, 3),
        "server_latency_ms_p50": round(percentile(server_latencies, 0.50), 2),
        "server_latency_ms_p95": round(percentile(server_latencies, 0.95), 2),
        "end_to_end_ms_p50": round(percentile(wall_latencies, 0.50), 2),
        "end_to_end_ms_p95": round(percentile(wall_latencies, 0.95), 2),
        "payload": payload,
    }

    if fail_results:
        summary["first_error"] = fail_results[0].error

    return {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TinyServe phase-2 load test (concurrency + throughput)."
    )
    parser.add_argument("--url", default="http://localhost:8000/v1/generate")
    parser.add_argument("--total", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--prompt",
        default="Please explain what dynamic batch processing is in three sentences.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument("--label", default="run")
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_load_test(args)
    summary = report["summary"]

    print(f"label={summary['label']}")
    print(
        f"ok={summary['ok']} error={summary['error']} total={summary['total']} "
        f"concurrency={summary['concurrency']}"
    )
    print(f"elapsed_s={summary['elapsed_s']}")
    print(f"req_per_s={summary['req_per_s']}")
    print(f"tokens_per_s={summary['tokens_per_s']}")
    print(
        "server_latency_ms "
        f"p50={summary['server_latency_ms_p50']} "
        f"p95={summary['server_latency_ms_p95']}"
    )
    print(
        "end_to_end_ms "
        f"p50={summary['end_to_end_ms_p50']} "
        f"p95={summary['end_to_end_ms_p95']}"
    )
    if "first_error" in summary:
        print(f"first_error={summary['first_error']}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"saved_report={args.output_json}")


if __name__ == "__main__":
    main()
