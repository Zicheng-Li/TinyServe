from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["summary"]


def ensure_matplotlib_cache(workdir: Path) -> None:
    mpl_dir = workdir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def plot_phase_comparison(out_path: Path, phase1: dict, phase2: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    throughput_labels = ["req/s", "tokens/s"]
    throughput_phase1 = [phase1["req_per_s"], phase1["tokens_per_s"]]
    throughput_phase2 = [phase2["req_per_s"], phase2["tokens_per_s"]]

    latency_labels = ["server p50", "server p95", "e2e p50", "e2e p95"]
    latency_phase1 = [
        phase1["server_p50_ms"],
        phase1["server_p95_ms"],
        phase1["e2e_p50_ms"],
        phase1["e2e_p95_ms"],
    ]
    latency_phase2 = [
        phase2["server_p50_ms"],
        phase2["server_p95_ms"],
        phase2["e2e_p50_ms"],
        phase2["e2e_p95_ms"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = 0.35

    x0 = np.arange(len(throughput_labels))
    axes[0].bar(x0 - width / 2, throughput_phase1, width=width, label="phase1_like")
    axes[0].bar(x0 + width / 2, throughput_phase2, width=width, label="phase2_batching")
    axes[0].set_xticks(x0, throughput_labels)
    axes[0].set_title("Phase1 vs Phase2 Throughput")
    axes[0].set_ylabel("value")
    axes[0].legend()

    x1 = np.arange(len(latency_labels))
    axes[1].bar(x1 - width / 2, latency_phase1, width=width, label="phase1_like")
    axes[1].bar(x1 + width / 2, latency_phase2, width=width, label="phase2_batching")
    axes[1].set_xticks(x1, latency_labels, rotation=20)
    axes[1].set_title("Phase1 vs Phase2 Latency (ms)")
    axes[1].set_ylabel("ms")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cache_comparison(out_path: Path, dynamic: dict, static: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["req/s", "tokens/s", "server p95(ms)", "rss peak(MB)", "mps peak(MB)"]
    d_values = [
        dynamic["req_per_s"],
        dynamic["tokens_per_s"],
        dynamic["server_latency_ms_p95"],
        dynamic.get("process_rss_mb_peak") or 0.0,
        dynamic.get("mps_allocated_mb_peak") or 0.0,
    ]
    s_values = [
        static["req_per_s"],
        static["tokens_per_s"],
        static["server_latency_ms_p95"],
        static.get("process_rss_mb_peak") or 0.0,
        static.get("mps_allocated_mb_peak") or 0.0,
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, d_values, width=width, label="dynamic")
    ax.bar(x + width / 2, s_values, width=width, label="static")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_title("KV Cache Dynamic vs Static")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_phase_summary(phase1: dict, phase2: dict) -> None:
    print("\n[Phase1 vs Phase2]")
    print(
        f"req/s: {phase1['req_per_s']} -> {phase2['req_per_s']} "
        f"({pct_change(phase2['req_per_s'], phase1['req_per_s']):+.2f}%)"
    )
    print(
        f"tokens/s: {phase1['tokens_per_s']} -> {phase2['tokens_per_s']} "
        f"({pct_change(phase2['tokens_per_s'], phase1['tokens_per_s']):+.2f}%)"
    )
    print(
        f"elapsed_s: {phase1['elapsed_s']} -> {phase2['elapsed_s']} "
        f"({pct_change(phase2['elapsed_s'], phase1['elapsed_s']):+.2f}%)"
    )
    print(
        f"server p50(ms): {phase1['server_p50_ms']} -> {phase2['server_p50_ms']} "
        f"({pct_change(phase2['server_p50_ms'], phase1['server_p50_ms']):+.2f}%)"
    )
    print(
        f"server p95(ms): {phase1['server_p95_ms']} -> {phase2['server_p95_ms']} "
        f"({pct_change(phase2['server_p95_ms'], phase1['server_p95_ms']):+.2f}%)"
    )


def print_cache_summary(dynamic: dict, static: dict) -> None:
    print("\n[Dynamic vs Static]")
    print(
        f"req/s: {dynamic['req_per_s']} -> {static['req_per_s']} "
        f"({pct_change(static['req_per_s'], dynamic['req_per_s']):+.2f}%)"
    )
    print(
        f"tokens/s: {dynamic['tokens_per_s']} -> {static['tokens_per_s']} "
        f"({pct_change(static['tokens_per_s'], dynamic['tokens_per_s']):+.2f}%)"
    )
    print(
        f"server p95(ms): {dynamic['server_latency_ms_p95']} -> {static['server_latency_ms_p95']} "
        f"({pct_change(static['server_latency_ms_p95'], dynamic['server_latency_ms_p95']):+.2f}%)"
    )
    print(
        f"rss peak(MB): {dynamic.get('process_rss_mb_peak')} -> {static.get('process_rss_mb_peak')} "
        f"({pct_change((static.get('process_rss_mb_peak') or 0.0), (dynamic.get('process_rss_mb_peak') or 1.0)):+.2f}%)"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot TinyServe benchmark comparisons.")
    p.add_argument(
        "--dynamic-json", default="testing/report_cache_dynamic.json", type=Path
    )
    p.add_argument("--static-json", default="testing/report_cache_static.json", type=Path)
    p.add_argument("--out-dir", default="testing/charts", type=Path)

    p.add_argument("--phase1-label", default="phase1_like")
    p.add_argument("--phase1-elapsed-s", type=float, default=308.434)
    p.add_argument("--phase1-req-per-s", type=float, default=0.195)
    p.add_argument("--phase1-tokens-per-s", type=float, default=17.119)
    p.add_argument("--phase1-server-p50-ms", type=float, default=50815.63)
    p.add_argument("--phase1-server-p95-ms", type=float, default=52029.83)
    p.add_argument("--phase1-e2e-p50-ms", type=float, default=50818.68)
    p.add_argument("--phase1-e2e-p95-ms", type=float, default=52032.57)

    p.add_argument("--phase2-label", default="phase2_batching")
    p.add_argument("--phase2-elapsed-s", type=float, default=266.839)
    p.add_argument("--phase2-req-per-s", type=float, default=0.225)
    p.add_argument("--phase2-tokens-per-s", type=float, default=19.787)
    p.add_argument("--phase2-server-p50-ms", type=float, default=37936.31)
    p.add_argument("--phase2-server-p95-ms", type=float, default=55087.38)
    p.add_argument("--phase2-e2e-p50-ms", type=float, default=37947.41)
    p.add_argument("--phase2-e2e-p95-ms", type=float, default=55096.64)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1 = {
        "label": args.phase1_label,
        "elapsed_s": args.phase1_elapsed_s,
        "req_per_s": args.phase1_req_per_s,
        "tokens_per_s": args.phase1_tokens_per_s,
        "server_p50_ms": args.phase1_server_p50_ms,
        "server_p95_ms": args.phase1_server_p95_ms,
        "e2e_p50_ms": args.phase1_e2e_p50_ms,
        "e2e_p95_ms": args.phase1_e2e_p95_ms,
    }
    phase2 = {
        "label": args.phase2_label,
        "elapsed_s": args.phase2_elapsed_s,
        "req_per_s": args.phase2_req_per_s,
        "tokens_per_s": args.phase2_tokens_per_s,
        "server_p50_ms": args.phase2_server_p50_ms,
        "server_p95_ms": args.phase2_server_p95_ms,
        "e2e_p50_ms": args.phase2_e2e_p50_ms,
        "e2e_p95_ms": args.phase2_e2e_p95_ms,
    }

    dynamic_summary = load_summary(args.dynamic_json)
    static_summary = load_summary(args.static_json)

    print_phase_summary(phase1, phase2)
    print_cache_summary(dynamic_summary, static_summary)

    ensure_matplotlib_cache(out_dir)

    phase_plot = out_dir / "phase1_vs_phase2.png"
    cache_plot = out_dir / "cache_dynamic_vs_static.png"
    plot_phase_comparison(phase_plot, phase1, phase2)
    plot_cache_comparison(cache_plot, dynamic_summary, static_summary)

    print(f"\nSaved: {phase_plot}")
    print(f"Saved: {cache_plot}")


if __name__ == "__main__":
    main()
