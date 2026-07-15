# taraference

CUDA multi-turn inference for **Qwen2.5** GGUF (defaults for **RTX 3050 Ti 4GB**).

## Layout

| Piece | Where | Role |
|-------|--------|------|
| **Inference** | `crates/core` | GGUF load, CUDA forward, `InferenceEngine`, `Session`, chat template |
| **Server** | `crates/cli/src/serve` | OpenAI-compatible HTTP (`/v1/models`, `/v1/chat/completions`) |
| **CLI** | `crates/cli` | interactive chat, `--profile`, `--serve` |

```powershell
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
```

No flags needed. Defaults:

| Setting | Value | Why |
|---------|------:|-----|
| Context | **5000** | Multi-turn room; KV fits with ~1.8 GiB Q4 weights on 4 GB |
| Max new tokens | **512** | Full answers without mid-sentence cutoffs |

Optional one-shot: `--prompt "Hello"`. Chat: type messages, `/quit`, `/reset`.

### OpenAI-compatible server

```powershell
cargo run --release -- models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --serve
# port (default 8787)
cargo run --release -- models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --serve 3000
```

One process loads one GGUF; API model id = file stem (e.g. `Qwen2.5-0.5B-Instruct-Q4_K_M`). No multi-model switching.

| Endpoint | Method | Notes |
|----------|--------|--------|
| `/health` | GET | liveness (`{"status":"ok"}`) |
| `/v1/models` | GET | that one model id |
| `/v1/chat/completions` | POST | always runs the loaded weights |

```powershell
curl http://127.0.0.1:8787/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"Qwen2.5-0.5B-Instruct-Q4_K_M","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Each request is **stateless** (full `messages` history → fresh KV prefill). Supports **`stream: true`** (SSE, OpenAI chunk format + `[DONE]`). Requests are serialized on one GPU engine.

Any OpenAI-compatible client can call the server (curl, official SDKs, etc.).  
`openai-test-python/` is a **standalone** SDK example and does not manage this process.

### Profile / benchmark (multi-turn + CPU/GPU)

```powershell
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile
```

Each `--profile` run prints the report **and** saves it under **`profile-logs/`**:

| File | Purpose |
|------|---------|
| `profile_<date>_<model>_<decode>.txt` | Full report + `SUMMARY_KV` (model stem in name) |
| `latest_<model>.txt` | Last run for that model (fair A/B compare) |
| `latest.txt` | Most recent run overall |
| `index.csv` | One row per run: stamp, **model**, decode, t/s, drop %, ctx |

Re-run after changes; **vs PREVIOUS** compares only the same model (`latest_<model>.txt`), so 0.5B is never mixed with 3B.

### KV + attention (long multi-turn)

| Feature | What |
|---------|------|
| **f16 KV** | Keys/values stored as half precision (~½ VRAM & attention BW vs f32) |
| **`fast` (default)** | Tiled online attention — fixed smem (`Q` + tile), no `scores[ctx]` |
| Incremental multi-turn | Append-only cache; only new tokens are prefilled |

### A/B decode backends (`--decode`)

Backends are a **registry** — add/remove without touching `layer.rs` launch code.

| Name | Meaning |
|------|---------|
| `fast` / `fastv1` | v1: parallel softmax (`scores[ctx]` smem) |
| **`fastv2`** (default) | v2: tiled online attn (fixed smem) |
| `basic` | serial softmax baseline |
| `online` | online decode (1 tok); prefill → `fastv2` |

```powershell
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode fastv2
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode fast
```

Logs: `profile-logs/profile_<date>_<model>_<decode>.txt`.

#### Add `fastv3` (or delete a loser)

| Step | Add | Delete (no improve) |
|------|-----|---------------------|
| 1 | `kernels/attn/fast_v3.cu` with `attn_fast_v3` | delete that `.cu` |
| 2 | `include_str!("attn/fast_v3.cu")` in `kernels/mod.rs` | remove that include |
| 3 | one row in `decode.rs` **`REGISTRY`** | remove that row |
| 4 | `--profile --decode fastv3` | done |

Do **not** edit `layer.rs` for a normal causal kernel — launch is data-driven from `AttnLaunch`.


## Crates

| Path | Role |
|------|------|
| `crates/cli` | Binary: chat, profile, OpenAI server |
| `crates/cli/src/serve` | OpenAI HTTP API |
| `crates/core` | Inference engine + session + CUDA |
| `crates/core/src/cuda/` | load, matmul, forward, KV |
| `crates/core/src/cuda/kernels/*.cu` | NVRTC device code fragments |
| `crates/gguf` | GGUF mmap reader |
