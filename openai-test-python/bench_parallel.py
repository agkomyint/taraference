#!/usr/bin/env python3
"""
Parallel / aggregate throughput bench for tarafer --serve.

Core metrics (GOAL.md):
  - single_stream_accepted_tps : one client decode rate
  - aggregate_accepted_tps     : total completion tokens / wall across all clients

Architecture note (current taraference):
  Server holds one InferenceEngine behind a Mutex + one KV arena.
  Concurrent HTTP clients **queue** — they do not decode in parallel on GPU.
  This script measures that reality: concurrency curve should be flat for
  aggregate tok/s, while latency rises ~linearly with concurrency.

Usage:
  # server (one GPU process)
  $env:TARAFER_IGNORE_EOS = "1"
  .\\target\\release\\tarafer.exe models\\Qwen3.5-4B-Q4_K_M.gguf --serve 8787 -n 256 --ctx 1024

  # concurrency sweep (default): 1,2,4,8 clients × fixed wall budget
  python bench_parallel.py --sweep 1,2,4,8 --duration 30 --max-tokens 128

  # pure max aggregate: short gens, one client, long enough sample
  python bench_parallel.py --concurrency 1 --duration 60 --max-tokens 64

  # project time-to-1M from measured aggregate tps
  python bench_parallel.py --concurrency 1 --duration 45 --max-tokens 128 --project-1m
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field


DEFAULT_BASE = "http://127.0.0.1:8787/v1"
DEFAULT_PROMPT = "Continue: 1 2 3 4 5 6 7 8 9 10"


@dataclass
class ReqResult:
    completion_tokens: int
    prompt_tokens: int
    wall_s: float
    finish_reason: str | None
    wait_error: str | None = None


@dataclass
class RunResult:
    concurrency: int
    max_tokens: int
    duration_s: float
    wall_s: float
    requests_ok: int
    requests_err: int
    total_completion: int
    total_prompt: int
    aggregate_tps: float
    mean_req_tps: float
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    projected_1m_s: float | None
    notes: str = ""
    per_req: list[ReqResult] = field(default_factory=list, repr=False)


def http_json(url: str, payload: dict | None = None, timeout: float = 600.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_model(base: str, preferred: str) -> str:
    if preferred and preferred != "auto":
        return preferred
    models = http_json(f"{base}/models")
    ids = [m["id"] for m in models.get("data", [])]
    if not ids:
        raise SystemExit("GET /v1/models returned no models")
    return ids[0]


def wait_healthy(root: str, timeout_s: float = 60.0) -> None:
    health = f"{root.rstrip('/')}/health"
    deadline = time.perf_counter() + timeout_s
    last: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.3)
    raise SystemExit(f"server not healthy at {health}: {last}")


def one_completion(
    base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> ReqResult:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a high-throughput text generator."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "seed": seed,
    }
    t0 = time.perf_counter()
    try:
        body = http_json(
            f"{base}/chat/completions",
            payload,
            timeout=max(120.0, max_tokens * 2.0),
        )
    except Exception as e:  # noqa: BLE001
        return ReqResult(0, 0, time.perf_counter() - t0, None, str(e))
    wall = time.perf_counter() - t0
    usage = body.get("usage") or {}
    finish = None
    choices = body.get("choices") or []
    if choices:
        finish = choices[0].get("finish_reason")
    return ReqResult(
        completion_tokens=int(usage.get("completion_tokens") or 0),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        wall_s=wall,
        finish_reason=finish,
    )


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def run_load(
    base: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    duration_s: float,
    temperature: float,
    prompt: str,
    seed0: int,
) -> RunResult:
    """
    Keep `concurrency` workers busy for ~duration_s wall time.
    Each worker issues sequential requests until the stop event fires
    (in-flight request always finishes — counted).
    """
    stop = threading.Event()
    results: list[ReqResult] = []
    lock = threading.Lock()
    counter = {"n": 0}

    def worker(worker_id: int) -> None:
        while not stop.is_set():
            with lock:
                n = counter["n"]
                counter["n"] += 1
            r = one_completion(
                base,
                model,
                prompt,
                max_tokens,
                temperature,
                seed0 + n * 1009 + worker_id,
            )
            with lock:
                results.append(r)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(worker, i) for i in range(concurrency)]
        # Hold the gate open for duration; then signal stop and drain.
        time.sleep(duration_s)
        stop.set()
        for f in as_completed(futs):
            f.result()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r.wait_error is None and r.completion_tokens > 0]
    err = [r for r in results if r.wait_error is not None or r.completion_tokens <= 0]
    total_c = sum(r.completion_tokens for r in ok)
    total_p = sum(r.prompt_tokens for r in ok)
    lat = [r.wall_s for r in ok]
    req_tps = [r.completion_tokens / r.wall_s for r in ok if r.wall_s > 0]
    agg = total_c / wall if wall > 0 else 0.0

    return RunResult(
        concurrency=concurrency,
        max_tokens=max_tokens,
        duration_s=duration_s,
        wall_s=wall,
        requests_ok=len(ok),
        requests_err=len(err),
        total_completion=total_c,
        total_prompt=total_p,
        aggregate_tps=agg,
        mean_req_tps=statistics.fmean(req_tps) if req_tps else 0.0,
        mean_latency_s=statistics.fmean(lat) if lat else 0.0,
        p50_latency_s=percentile(lat, 50),
        p95_latency_s=percentile(lat, 95),
        projected_1m_s=(1_000_000 / agg) if agg > 0 else None,
        notes=(
            "clients share one Mutex GPU engine — aggregate should not scale with concurrency"
            if concurrency > 1
            else "single stream"
        ),
        per_req=results,
    )


def print_run(r: RunResult) -> None:
    print(
        f"  c={r.concurrency:<3d} max_tok={r.max_tokens:<4d}  "
        f"agg={r.aggregate_tps:6.1f} t/s  "
        f"mean_req={r.mean_req_tps:6.1f} t/s  "
        f"lat p50={r.p50_latency_s:5.2f}s p95={r.p95_latency_s:5.2f}s  "
        f"ok={r.requests_ok} err={r.requests_err}  "
        f"tok={r.total_completion:,}  "
        f"1M≈{fmt_duration(r.projected_1m_s)}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Parallel aggregate throughput bench for tarafer")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE))
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "auto"))
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument(
        "--sweep",
        default="",
        help="comma list of concurrency levels, e.g. 1,2,4,8 (overrides --concurrency)",
    )
    p.add_argument(
        "--max-tokens-sweep",
        default="",
        help="comma list of max_tokens, e.g. 32,64,128,256,512 (single-client shape sweep)",
    )
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--duration", type=float, default=30.0, help="seconds per load point")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project-1m", action="store_true")
    p.add_argument("--json-out", default="")
    p.add_argument("--no-wait", action="store_true")
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    root = base[: -len("/v1")]

    if not args.no_wait:
        print(f"waiting for {root}/health …", flush=True)
        wait_healthy(root)

    model = resolve_model(base, args.model)
    print("=== tarafer PARALLEL / AGGREGATE throughput ===", flush=True)
    print(f"model={model}", flush=True)
    print(f"base={base}", flush=True)
    print(f"temperature={args.temperature}  duration/point={args.duration}s", flush=True)
    print(
        "engine=single Mutex + one KV (concurrent clients queue; aggregate ≈ single-stream)",
        flush=True,
    )
    print(flush=True)

    runs: list[RunResult] = []

    # Shape sweep: find best max_tokens for single-stream aggregate tps
    if args.max_tokens_sweep:
        shapes = [int(x) for x in args.max_tokens_sweep.split(",") if x.strip()]
        print(f"-- max_tokens shape sweep (c=1): {shapes}", flush=True)
        for mt in shapes:
            r = run_load(
                base,
                model,
                concurrency=1,
                max_tokens=mt,
                duration_s=args.duration,
                temperature=args.temperature,
                prompt=args.prompt,
                seed0=args.seed + mt,
            )
            runs.append(r)
            print_run(r)
        best = max(runs, key=lambda x: x.aggregate_tps)
        print(
            f"\nBEST shape: max_tokens={best.max_tokens} → "
            f"{best.aggregate_tps:.1f} aggregate t/s  "
            f"(1M ≈ {fmt_duration(best.projected_1m_s)})",
            flush=True,
        )
        print(flush=True)

    # Concurrency sweep
    if args.sweep:
        levels = [int(x) for x in args.sweep.split(",") if x.strip()]
    else:
        levels = [args.concurrency]

    print(f"-- concurrency sweep (max_tokens={args.max_tokens}): {levels}", flush=True)
    conc_runs: list[RunResult] = []
    for c in levels:
        r = run_load(
            base,
            model,
            concurrency=c,
            max_tokens=args.max_tokens,
            duration_s=args.duration,
            temperature=args.temperature,
            prompt=args.prompt,
            seed0=args.seed + 10_000 * c,
        )
        conc_runs.append(r)
        runs.append(r)
        print_run(r)

    if len(conc_runs) >= 2:
        base_agg = conc_runs[0].aggregate_tps
        print(flush=True)
        print("concurrency scale-up vs c=1 aggregate:", flush=True)
        for r in conc_runs:
            ratio = (r.aggregate_tps / base_agg) if base_agg > 0 else 0.0
            print(
                f"  c={r.concurrency}: agg={r.aggregate_tps:.1f} t/s  "
                f"({ratio:.2f}×)  p50_lat={r.p50_latency_s:.2f}s",
                flush=True,
            )
        print(
            "\nIf ratios stay ~1.0× while latency rises, GPU is saturated by one stream "
            "(expected today — no continuous batching).",
            flush=True,
        )

    best = max(runs, key=lambda x: x.aggregate_tps) if runs else None
    if best and args.project_1m:
        print(flush=True)
        print("=== time-to-1M projection (from best measured aggregate) ===", flush=True)
        print(
            f"  best: c={best.concurrency} max_tok={best.max_tokens} "
            f"→ {best.aggregate_tps:.1f} tok/s",
            flush=True,
        )
        print(f"  time_to_1_000_000 ≈ {fmt_duration(best.projected_1m_s)}", flush=True)

    summary = {
        "model": model,
        "base_url": base,
        "runs": [
            {k: v for k, v in asdict(r).items() if k != "per_req"} for r in runs
        ],
        "best_aggregate_tps": best.aggregate_tps if best else None,
        "best_config": (
            {"concurrency": best.concurrency, "max_tokens": best.max_tokens}
            if best
            else None
        ),
        "projected_1m_s": best.projected_1m_s if best else None,
        "projected_1m_human": fmt_duration(best.projected_1m_s) if best else None,
        "architecture": "single_mutex_engine_no_continuous_batching",
    }
    print(flush=True)
    print("=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.json_out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
