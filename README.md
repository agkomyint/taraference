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

### A/B decode backends (`--decode`)

Keep all attention paths; switch without deleting code:

| Name | Meaning |
|------|---------|
| `fast` | Parallel softmax (default) |
| `basic` | Serial softmax baseline |
| `online` | Online softmax on decode tokens (prefill uses fast) |

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
