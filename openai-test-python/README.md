# openai-test-python

Standalone **OpenAI Python SDK** chat client. Talks only to an HTTP API that implements the OpenAI-compatible surface (`/v1/models`, `/v1/chat/completions`). It does **not** start, stop, or configure any backend process.

## Setup

```powershell
cd openai-test-python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Point `--base-url` at whatever compatible server you already have running:

```powershell
python assistant.py --base-url http://127.0.0.1:3000/v1

python assistant.py --base-url http://127.0.0.1:3000/v1 --once "hi"

python assistant.py --base-url http://127.0.0.1:3000/v1 --once "hi" --no-stream
```

| Flag / env | Default |
|------------|---------|
| `--base-url` / `OPENAI_BASE_URL` | `http://127.0.0.1:3000/v1` |
| `--api-key` / `OPENAI_API_KEY` | `local` |
| `--model` / `OPENAI_MODEL` | `auto` (first id from `/v1/models`) |
| `--max-tokens` | `256` |
| `--no-stream` | streaming **on** by default |
| `--system` | `You are a helpful assistant.` |

Requires OpenAI SDK **1.x** (`openai>=1.40,<2`).

## Throughput / parallel bench (no OpenAI SDK)

`bench_parallel.py` and `bench_1m.py` use only the stdlib (`urllib`) against
`tarafer --serve`.

**Architecture today:** one process = one GGUF = one `Mutex<InferenceEngine>` +
one KV arena. Concurrent HTTP clients **queue**; they do not decode in parallel.
Aggregate tok/s ≈ single-stream tok/s. Latency rises with concurrency.

```powershell
# terminal 1 — pure volume path
$env:TARAFER_IGNORE_EOS = "1"
.\target\release\tarafer.exe models\Qwen3.5-4B-Q4_K_M.gguf --serve 8787 -n 900 --ctx 1024

# terminal 2 — concurrency curve + max_tokens shape + 1M projection
python bench_parallel.py --sweep 1,2,4,8 --max-tokens 128 --duration 30 --project-1m
python bench_parallel.py --max-tokens-sweep 64,128,256,512 --duration 30 --project-1m
python bench_1m.py --target 10000 --max-tokens 512   # short calibration
# python bench_1m.py --target 1000000 --max-tokens 512  # full 1M (hours)
```

Core metrics (see [v0.7.0 roadmap](../roadmap/0.7.0/ROADMAP.md)):
- `aggregate_accepted_tps` — total completion tokens / wall (what this bench optimizes)
- `single_stream_accepted_tps` — one client decode rate (server log `decode_tps`)
