#!/usr/bin/env python3
"""
Time-to-N-tokens bench against a tarafer OpenAI-compatible server.

Loops short-prompt chat completions until `target` completion tokens are
generated, measuring wall time and decode throughput.

Designed for pure generation volume (not quality):
  - temperature=0 (greedy / fastest path)
  - non-streaming by default (less HTTP overhead)
  - server should use TARAFER_IGNORE_EOS=1 so each request fills max_tokens

Usage:
  # terminal 1
  $env:TARAFER_IGNORE_EOS=1
  .\\target\\release\\tarafer.exe models\\Qwen3.5-4B-Q4_K_M.gguf --serve 8787 -n 900 --ctx 1024

  # terminal 2
  python bench_1m.py --target 1000000 --max-tokens 900
  python bench_1m.py --target 20000          # quick calibration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field


DEFAULT_BASE = "http://127.0.0.1:8787/v1"
# Long continuous generation prompt — still benefits from IGNORE_EOS on the server.
DEFAULT_PROMPT = (
    "Write continuously. Output only plain text. Never stop. "
    "Count upward forever: 1 2 3 4 5 …"
)


@dataclass
class RequestStats:
    completion_tokens: int
    prompt_tokens: int
    wall_s: float
    finish_reason: str | None
    decode_tps: float  # completion_tokens / wall (includes prefill + HTTP)


@dataclass
class BenchState:
    target: int
    total_completion: int = 0
    total_prompt: int = 0
    requests: int = 0
    walls: list[float] = field(default_factory=list)
    per_req: list[RequestStats] = field(default_factory=list)
    t0: float = 0.0

    def overall_tps(self, now: float | None = None) -> float:
        elapsed = (now if now is not None else time.perf_counter()) - self.t0
        if elapsed <= 0:
            return 0.0
        return self.total_completion / elapsed


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


def wait_healthy(base_root: str, timeout_s: float = 120.0) -> None:
    """base_root is like http://127.0.0.1:8787 (no /v1)."""
    health = f"{base_root.rstrip('/')}/health"
    deadline = time.perf_counter() + timeout_s
    last_err: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception as e:  # noqa: BLE001 — connection retry loop
            last_err = e
            time.sleep(0.5)
    raise SystemExit(f"server not healthy at {health}: {last_err}")


def one_completion(
    base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
    seed: int,
) -> RequestStats:
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a high-throughput text generator."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "seed": seed,
    }
    t0 = time.perf_counter()
    if not stream:
        body = http_json(url, payload, timeout=max(600.0, max_tokens * 0.5))
        wall = time.perf_counter() - t0
        usage = body.get("usage") or {}
        completion = int(usage.get("completion_tokens") or 0)
        prompt_tok = int(usage.get("prompt_tokens") or 0)
        finish = None
        choices = body.get("choices") or []
        if choices:
            finish = choices[0].get("finish_reason")
        return RequestStats(
            completion_tokens=completion,
            prompt_tokens=prompt_tok,
            wall_s=wall,
            finish_reason=finish,
            decode_tps=completion / wall if wall > 0 else 0.0,
        )

    # Streaming path: count chunks; usage may be absent — approximate via pieces.
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    chars = 0
    # Rough token estimate if server omits usage on stream end
    pieces = 0
    with urllib.request.urlopen(req, timeout=max(600.0, max_tokens * 0.5)) as resp:
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_s = line[5:].strip()
            if data_s == "[DONE]":
                break
            try:
                chunk = json.loads(data_s)
            except json.JSONDecodeError:
                continue
            for ch in chunk.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content") or ""
                if delta:
                    chars += len(delta)
                    pieces += 1
    wall = time.perf_counter() - t0
    # Stream path lacks usage on this server — estimate ~4 chars/token as fallback.
    # Prefer piece count when content is short tokens; use max of both heuristics.
    est = max(pieces, chars // 4) if (pieces or chars) else 0
    return RequestStats(
        completion_tokens=est,
        prompt_tokens=0,
        wall_s=wall,
        finish_reason="stream",
        decode_tps=est / wall if wall > 0 else 0.0,
    )


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def print_progress(state: BenchState, last: RequestStats) -> None:
    now = time.perf_counter()
    elapsed = now - state.t0
    tps = state.overall_tps(now)
    remaining = max(0, state.target - state.total_completion)
    eta = remaining / tps if tps > 0 else float("inf")
    pct = 100.0 * state.total_completion / state.target if state.target else 0.0
    print(
        f"[{pct:5.1f}%] req={state.requests:4d}  "
        f"tok={state.total_completion:,}/{state.target:,}  "
        f"last={last.completion_tokens} tok @ {last.decode_tps:.1f} t/s  "
        f"overall={tps:.1f} t/s  "
        f"elapsed={fmt_duration(elapsed)}  "
        f"ETA={fmt_duration(eta) if eta != float('inf') else '?'}  "
        f"finish={last.finish_reason}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Time-to-N-tokens bench for tarafer --serve")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE))
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "auto"))
    p.add_argument("--target", type=int, default=1_000_000, help="completion tokens goal")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="tokens per request (keep under server max_seq - prompt)",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--stream", action="store_true", help="use SSE stream (slower / less accurate)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--report-every",
        type=int,
        default=1,
        help="print progress every N requests",
    )
    p.add_argument(
        "--json-out",
        default="",
        help="write final summary JSON to this path",
    )
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="do not wait for /health before starting",
    )
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    root = base[: -len("/v1")]

    if not args.no_wait:
        print(f"waiting for {root}/health …", flush=True)
        wait_healthy(root)

    try:
        model = resolve_model(base, args.model)
    except urllib.error.URLError as e:
        print(f"cannot reach {base}: {e}", file=sys.stderr)
        return 1

    print("=== tarafer time-to-N bench ===", flush=True)
    print(f"base_url     = {base}", flush=True)
    print(f"model        = {model}", flush=True)
    print(f"target       = {args.target:,} completion tokens", flush=True)
    print(f"max_tokens   = {args.max_tokens} per request", flush=True)
    print(f"temperature  = {args.temperature}", flush=True)
    print(f"stream       = {args.stream}", flush=True)
    print(
        "tip: start server with TARAFER_IGNORE_EOS=1 so each request fills max_tokens",
        flush=True,
    )
    print(flush=True)

    state = BenchState(target=args.target)
    state.t0 = time.perf_counter()

    try:
        while state.total_completion < args.target:
            need = args.target - state.total_completion
            max_tok = min(args.max_tokens, need)
            try:
                stats = one_completion(
                    base,
                    model,
                    args.prompt,
                    max_tok,
                    args.temperature,
                    args.stream,
                    args.seed + state.requests,
                )
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                print(f"HTTP {e.code}: {body}", file=sys.stderr)
                return 1
            except urllib.error.URLError as e:
                print(f"connection error: {e}", file=sys.stderr)
                return 1

            if stats.completion_tokens <= 0:
                print(
                    "warning: 0 completion tokens — check EOS early-stop "
                    "(set TARAFER_IGNORE_EOS=1 on the server) or max_tokens/ctx",
                    file=sys.stderr,
                )
                if state.requests == 0:
                    return 2
                break

            state.total_completion += stats.completion_tokens
            state.total_prompt += stats.prompt_tokens
            state.requests += 1
            state.walls.append(stats.wall_s)
            state.per_req.append(stats)

            if state.requests % max(1, args.report_every) == 0:
                print_progress(state, stats)

            # Safety: if server keeps returning tiny batches, abort after noise
            if stats.completion_tokens < max(1, max_tok // 20) and stats.finish_reason == "stop":
                print(
                    "note: early EOS (finish=stop). Re-run server with "
                    "TARAFER_IGNORE_EOS=1 for pure volume.",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\ninterrupted by user", flush=True)

    elapsed = time.perf_counter() - state.t0
    tps = state.total_completion / elapsed if elapsed > 0 else 0.0
    eta_1m = 1_000_000 / tps if tps > 0 else None
    remaining_to_1m = max(0, 1_000_000 - state.total_completion)
    eta_remaining = remaining_to_1m / tps if tps > 0 else None

    summary = {
        "model": model,
        "base_url": base,
        "target": args.target,
        "total_completion_tokens": state.total_completion,
        "total_prompt_tokens": state.total_prompt,
        "requests": state.requests,
        "wall_s": elapsed,
        "wall_human": fmt_duration(elapsed),
        "overall_completion_tps": tps,
        "max_tokens_per_req": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
        "projected_time_to_1m_s": eta_1m,
        "projected_time_to_1m_human": fmt_duration(eta_1m) if eta_1m else None,
        "eta_remaining_to_1m_s": eta_remaining,
        "eta_remaining_to_1m_human": fmt_duration(eta_remaining) if eta_remaining else None,
        "avg_req_wall_s": (sum(state.walls) / len(state.walls)) if state.walls else 0.0,
        "avg_req_tps": (
            sum(r.decode_tps for r in state.per_req) / len(state.per_req) if state.per_req else 0.0
        ),
        "hit_target": state.total_completion >= args.target,
    }

    print(flush=True)
    print("=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    if args.json_out:
        out = args.json_out
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "requests": [asdict(r) for r in state.per_req]}, f, indent=2)
        print(f"wrote {out}", flush=True)

    return 0 if summary["hit_target"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
