# taraference

CUDA multi-turn inference for **Qwen2.5-3B-Instruct Q4_K_M** (defaults for **RTX 3050 Ti 4GB**).

```powershell
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
```

No flags needed. Defaults:

| Setting | Value | Why |
|---------|------:|-----|
| Context | **5000** | Multi-turn room; KV fits with ~1.8 GiB Q4 weights on 4 GB |
| Max new tokens | **512** | Full answers without mid-sentence cutoffs |

Optional one-shot: `--prompt "Hello"`. Chat: type messages, `/quit`, `/reset`.

### Profile / benchmark (multi-turn + CPU/GPU)

```powershell
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile
```

Each `--profile` run prints the report **and** saves it under **`profile-logs/`**:

| File | Purpose |
|------|---------|
| `profile_YYYY-MM-DD_HH-mm-ss_<decode>.txt` | Full report + `SUMMARY_KV` block |
| `latest.txt` | Copy of the most recent run |
| `index.csv` | One row per run (decode t/s, drop %, ctx) for quick compare |

Re-run after changes; the CLI prints **vs PREVIOUS** deltas from `latest.txt`.

### KV + attention (long multi-turn)

| Feature | What |
|---------|------|
| **f16 KV** | Keys/values stored as half precision (~½ VRAM & attention BW vs f32) |
| **`fast` (default)** | Tiled online attention — fixed smem (`Q` + tile), no `scores[ctx]` |
| Incremental multi-turn | Append-only cache; only new tokens are prefilled |

### A/B decode backends (`--decode`)

| Name | Meaning |
|------|---------|
| `fast` | f16 KV + tiled online attn (default) |
| `basic` | f16 KV + serial softmax baseline |
| `online` | f16 KV + online softmax on decode (prefill uses fast) |

```powershell
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode basic
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode fast
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode online
```

Add more backends in `crates/core/src/cuda/decode.rs` + a CUDA kernel + branch in `layer.rs` `launch_attn`.


## Layout

| Path | Role |
|------|------|
| `crates/cli` | CLI entry |
| `crates/gguf` | GGUF mmap reader |
| `crates/core/src/cuda/` | load, matmul, forward, KV |
| `crates/core/src/cuda/kernels/*.cu` | NVRTC device code fragments |
| `crates/core/src/tokenizer/` | BPE + specials |
| `crates/core/src/session.rs` | multi-turn chat |
