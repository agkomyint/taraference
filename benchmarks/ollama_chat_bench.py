"""Deterministic single-stream multi-turn Ollama benchmark."""

import argparse
import json
import time
import urllib.request


PROMPTS = [
    "hi, who are you?",
    "what can you help me with in one sentence?",
    "ok give me 3 bullet ideas for a weekend project",
    "expand on the second idea a bit more",
    "summarize everything we talked about so far",
]


def post(payload):
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    turns = []
    started = time.perf_counter()
    for index, prompt in enumerate(PROMPTS, 1):
        messages.append({"role": "user", "content": prompt})
        result = post(
            {
                "model": args.model,
                "messages": messages,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "num_ctx": 5000,
                    "num_predict": 128,
                    "temperature": 0,
                    "seed": 0,
                },
            }
        )
        answer = result["message"]["content"]
        messages.append({"role": "assistant", "content": answer})
        count = int(result.get("eval_count", 0))
        duration = int(result.get("eval_duration", 0))
        turn = {
            "turn": index,
            "eval_count": count,
            "eval_duration_ns": duration,
            "eval_tps": count * 1e9 / duration if duration else 0.0,
            "prompt_eval_count": int(result.get("prompt_eval_count", 0)),
            "done_reason": result.get("done_reason"),
            "answer": answer,
        }
        turns.append(turn)
        print(json.dumps(turn, ensure_ascii=False), flush=True)

    count = sum(turn["eval_count"] for turn in turns)
    duration = sum(turn["eval_duration_ns"] for turn in turns)
    summary = {
        "model": args.label or args.model,
        "ollama_model": args.model,
        "runtime": "ollama",
        "mode": "single-stream multi-turn",
        "total_eval_count": count,
        "total_eval_duration_ns": duration,
        "overall_decode_tps": count * 1e9 / duration if duration else 0.0,
        "decode_tps_first": turns[0]["eval_tps"],
        "decode_tps_last": turns[-1]["eval_tps"],
        "decode_drop_pct": 100.0
        * (1.0 - turns[-1]["eval_tps"] / turns[0]["eval_tps"]),
        "wall_s": time.perf_counter() - started,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
