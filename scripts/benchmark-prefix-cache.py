#!/usr/bin/env python3
"""Benchmark repeated-prefix OpenAI chat completions against Taraference."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request


SYSTEM_PROMPT = (
    "Classify Wikimedia editing. Return only compact JSON with keys category, "
    "topics, and plain_summary. category must be exactly one of: Current affairs, "
    "Politics, Sports, Entertainment, Science and technology, Business and economy, "
    "Health, Biography, History and culture, Geography, Society and education, Other. "
    "Choose the most specific fit. topics must have one to three short labels. "
    "plain_summary must be one factual sentence. Never invent events. No markdown."
)

EVENTS = [
    {
        "title": "Ada Lovelace",
        "project": "enwiki",
        "type": "edit",
        "activity": {"edits": 4, "editors": 3, "comments": ["Expanded biography"]},
    },
    {
        "title": "2026 Thailand general election",
        "project": "enwiki",
        "type": "edit",
        "activity": {"edits": 7, "editors": 5, "comments": ["Updated results"]},
    },
    {
        "title": "UEFA Champions League",
        "project": "dewiki",
        "type": "edit",
        "activity": {"edits": 5, "editors": 4, "comments": ["Match statistics"]},
    },
    {
        "title": "Bangkok",
        "project": "thwiki",
        "type": "edit",
        "activity": {"edits": 3, "editors": 2, "comments": ["Population data"]},
    },
    {
        "title": "Quantum computing",
        "project": "enwiki",
        "type": "edit",
        "activity": {"edits": 6, "editors": 4, "comments": ["Technical correction"]},
    },
    {
        "title": "Spirited Away",
        "project": "jawiki",
        "type": "edit",
        "activity": {"edits": 4, "editors": 3, "comments": ["Production details"]},
    },
    {
        "title": "World Health Organization",
        "project": "frwiki",
        "type": "edit",
        "activity": {"edits": 8, "editors": 5, "comments": ["Updated programme"]},
    },
    {
        "title": "Creative Commons",
        "project": "commonswiki",
        "type": "edit",
        "activity": {"edits": 3, "editors": 2, "comments": ["License metadata"]},
    },
]


def post_completion(endpoint: str, event: dict[str, object], timeout: float) -> dict[str, object]:
    body = json.dumps(
        {
            "model": "Qwen2.5-3B-Instruct-Q4_K_M",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(event, separators=(",", ":"))},
            ],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    wall_ms = (time.perf_counter() - started) * 1000
    usage = payload.get("usage", {})
    details = usage.get("prompt_tokens_details") or {}
    return {
        "title": event["title"],
        "wall_ms": round(wall_ms, 2),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "cached_tokens": int(details.get("cached_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "finish_reason": payload.get("choices", [{}])[0].get("finish_reason"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8790")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--output")
    args = parser.parse_args()

    endpoint = f"{args.url.rstrip('/')}/v1/chat/completions"
    results: list[dict[str, object]] = []
    try:
        for index in range(args.requests):
            result = post_completion(endpoint, EVENTS[index % len(EVENTS)], args.timeout)
            results.append(result)
            print(
                f"{index + 1:02d}  wall={result['wall_ms']:8.2f} ms  "
                f"prompt={result['prompt_tokens']:3d}  cached={result['cached_tokens']:3d}  "
                f"generated={result['completion_tokens']:2d}  {result['title']}",
                flush=True,
            )
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"benchmark request failed: {error}")
        return 1

    walls = [float(item["wall_ms"]) for item in results]
    warm_walls = walls[1:] or walls
    elapsed_seconds = sum(walls) / 1000
    summary = {
        "label": args.label,
        "requests": len(results),
        "cold_wall_ms": round(walls[0], 2),
        "warm_average_wall_ms": round(statistics.mean(warm_walls), 2),
        "warm_median_wall_ms": round(statistics.median(warm_walls), 2),
        "overall_requests_per_minute": round(len(results) / elapsed_seconds * 60, 2),
        "warm_requests_per_minute": round(60_000 / statistics.mean(warm_walls), 2),
        "average_cached_tokens_warm": round(
            statistics.mean(float(item["cached_tokens"]) for item in results[1:] or results), 2
        ),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(summary, output, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
