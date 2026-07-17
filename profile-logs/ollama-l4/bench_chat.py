import json
import sys
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
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


messages = [{"role": "system", "content": "You are a helpful assistant."}]
turns = []
started = time.perf_counter()
for index, prompt in enumerate(PROMPTS, 1):
    messages.append({"role": "user", "content": prompt})
    result = post(
        {
            "model": "tara-qwen25-3b-q4km",
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
    duration_ns = int(result.get("eval_duration", 0))
    tps = count * 1e9 / duration_ns if duration_ns else 0.0
    turn = {
        "turn": index,
        "eval_count": count,
        "eval_duration_ns": duration_ns,
        "eval_tps": tps,
        "prompt_eval_count": int(result.get("prompt_eval_count", 0)),
        "prompt_eval_duration_ns": int(result.get("prompt_eval_duration", 0)),
        "load_duration_ns": int(result.get("load_duration", 0)),
        "done_reason": result.get("done_reason"),
        "answer": answer,
    }
    turns.append(turn)
    print(json.dumps(turn, ensure_ascii=False), flush=True)

total_count = sum(turn["eval_count"] for turn in turns)
total_duration_ns = sum(turn["eval_duration_ns"] for turn in turns)
summary = {
    "model": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "runtime": "ollama",
    "mode": "single-stream multi-turn",
    "total_eval_count": total_count,
    "total_eval_duration_ns": total_duration_ns,
    "overall_decode_tps": total_count * 1e9 / total_duration_ns,
    "decode_tps_first": turns[0]["eval_tps"],
    "decode_tps_last": turns[-1]["eval_tps"],
    "decode_drop_pct": 100.0 * (1.0 - turns[-1]["eval_tps"] / turns[0]["eval_tps"]),
    "wall_s": time.perf_counter() - started,
}
print("SUMMARY " + json.dumps(summary), flush=True)
