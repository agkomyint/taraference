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
