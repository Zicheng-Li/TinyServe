from __future__ import annotations

import argparse
import json
import threading
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


@dataclass
class MemoryPeaks:
    process_rss_mb: float | None = None
    mps_allocated_mb: float | None = None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int((len(sorted_values) - 1) * p)
    return sorted_values[idx]


def fetch_json(url: str, timeout_sec: float) -> dict | None:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status >= 400:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


class HealthSampler:
    def __init__(self, health_url: str, health_timeout_sec: float, interval_sec: float):
        self.health_url = health_url
        self.health_timeout_sec = health_timeout_sec
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self._peaks = MemoryPeaks()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def peaks(self) -> MemoryPeaks:
        with self._lock:
            return MemoryPeaks(
                process_rss_mb=self._peaks.process_rss_mb,
                mps_allocated_mb=self._peaks.mps_allocated_mb,
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            health = fetch_json(self.health_url, self.health_timeout_sec)
            if health:
                rss = health.get("process_rss_mb")
                mps = health.get("mps_allocated_mb")
                with self._lock:
                    self._peaks.process_rss_mb = _max_optional(
                        self._peaks.process_rss_mb, _to_float_or_none(rss)
                    )
                    self._peaks.mps_allocated_mb = _max_optional(
                        self._peaks.mps_allocated_mb, _to_float_or_none(mps)
                    )
            self._stop_event.wait(self.interval_sec)


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _max_optional(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


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

    health_start = fetch_json(args.health_url, args.health_timeout_sec)
    sampler: HealthSampler | None = None
    if args.sample_memory:
        sampler = HealthSampler(
            health_url=args.health_url,
            health_timeout_sec=args.health_timeout_sec,
            interval_sec=args.memory_sample_interval_sec,
        )
        sampler.start()

    started_at = time.perf_counter()
    results: list[CallResult] = []

    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(call_generate, args.url, payload, args.timeout_sec)
            for _ in range(args.total)
        ]
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if args.progress_every > 0 and completed % args.progress_every == 0:
                print(f"progress={completed}/{args.total}")

    elapsed_s = time.perf_counter() - started_at

    if sampler is not None:
        sampler.stop()
    health_end = fetch_json(args.health_url, args.health_timeout_sec)

    ok_results = [r for r in results if r.ok]
    fail_results = [r for r in results if not r.ok]
    server_latencies = [r.latency_ms for r in ok_results if r.latency_ms is not None]
    wall_latencies = [r.wall_ms for r in ok_results]
    total_tokens = sum(r.total_tokens or 0 for r in ok_results)

    peak_from_sampler = sampler.peaks() if sampler is not None else MemoryPeaks()

    start_rss = _to_float_or_none((health_start or {}).get("process_rss_mb"))
    end_rss = _to_float_or_none((health_end or {}).get("process_rss_mb"))
    peak_rss = _max_optional(peak_from_sampler.process_rss_mb, _max_optional(start_rss, end_rss))

    start_mps = _to_float_or_none((health_start or {}).get("mps_allocated_mb"))
    end_mps = _to_float_or_none((health_end or {}).get("mps_allocated_mb"))
    peak_mps = _max_optional(peak_from_sampler.mps_allocated_mb, _max_optional(start_mps, end_mps))

    summary = {
        "label": args.label,
        "url": args.url,
        "health_url": args.health_url,
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
        "cache_implementation": (health_end or health_start or {}).get("cache_implementation"),
        "cache_implementation_applied": (health_end or health_start or {}).get(
            "cache_implementation_applied"
        ),
        "process_rss_mb_start": start_rss,
        "process_rss_mb_end": end_rss,
        "process_rss_mb_peak": peak_rss,
        "process_peak_rss_mb_end": _to_float_or_none(
            (health_end or {}).get("process_peak_rss_mb")
        ),
        "mps_allocated_mb_start": start_mps,
        "mps_allocated_mb_end": end_mps,
        "mps_allocated_mb_peak": peak_mps,
        "payload": payload,
    }

    if health_start is None or health_end is None:
        summary["health_note"] = "health endpoint unavailable for one or both snapshots"

    if fail_results:
        summary["first_error"] = fail_results[0].error

    return {
        "summary": summary,
        "results": [asdict(r) for r in results],
        "health_start": health_start,
        "health_end": health_end,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TinyServe load test (throughput, p95 latency, memory snapshots)."
    )
    parser.add_argument("--url", default="http://localhost:8000/v1/generate")
    parser.add_argument("--health-url", default="http://localhost:8000/health")
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
    parser.add_argument("--health-timeout-sec", type=float, default=5.0)
    parser.add_argument(
        "--sample-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Poll /health during test to estimate memory peaks.",
    )
    parser.add_argument("--memory-sample-interval-sec", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=10)
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
    print(
        "memory_mb "
        f"rss_start={summary['process_rss_mb_start']} "
        f"rss_end={summary['process_rss_mb_end']} "
        f"rss_peak={summary['process_rss_mb_peak']}"
    )
    print(
        "mps_memory_mb "
        f"start={summary['mps_allocated_mb_start']} "
        f"end={summary['mps_allocated_mb_end']} "
        f"peak={summary['mps_allocated_mb_peak']}"
    )
    print(
        f"cache_implementation={summary['cache_implementation']} "
        f"applied={summary['cache_implementation_applied']}"
    )

    if "health_note" in summary:
        print(f"health_note={summary['health_note']}")
    if "first_error" in summary:
        print(f"first_error={summary['first_error']}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"saved_report={args.output_json}")


if __name__ == "__main__":
    main()
