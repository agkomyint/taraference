# taraference

CUDA multi-turn GGUF inference, optimized for single-stream decode on NVIDIA GPUs.

**CLI name:** **`tarafer`** (short command). The repo/crate is still `taraference`.

**Performance goal:** one user, maximum decode tokens/sec — not multi-user concurrency. See [GOAL.md](GOAL.md).

**Performance scoreboard:** all profiling and speed claims use
`Qwen2.5-3B-Instruct-Q4_K_M.gguf`. See [GOAL.md](GOAL.md).

**v0.5 decode path:** packed Q4_K/Q6_K × Q8 DP4A kernels, fused equal-width
gate+up projections, cooperative in-block Q6 split-K reduction, eight-way flash
decode, CUDA graphs, and dynamic NVRTC `sm_XX` targeting.

**Day-to-day loop** (edit locally, sync/build/profile directly on SSH T4): [WORKFLOW.md](WORKFLOW.md).

---

## Fast install (recommended) — prebuilt binary

**Use this if you want first inference ASAP** (Lightning Studio, cloud SSH, Ubuntu GPU box, Windows NVIDIA PC, etc.).

You **do not** need Rust, Cargo, or a local compile. CI ships ready **Linux** and **Windows** x86_64 binaries on every version tag.

### What you need (runtime only)

| Requirement | Notes |
|-------------|--------|
| **OS** | Linux **x86_64** (Ubuntu 22.04-class) or Windows **x86_64** |
| **GPU** | NVIDIA (T4, 3050 Ti, A10, …). Arch is detected at load (`sm_75`, `sm_86`, …) |
| **Driver** | `nvidia-smi` works |
| **CUDA toolkit 13.x** | Must include **NVRTC** (runtime kernel compile). Many cloud images already have this |
| **Disk** | ~10 MB binary + ~380 MB for the 0.5B model (or ~2 GB for 3B) |

macOS: use [Install from source](#install-from-source) for now (no prebuilt yet).

### One-liner install + chat (Linux, on PATH)

```bash
# 1) download binary (~seconds)
curl -fsSL -o tarafer-linux-x86_64.tar.gz \
  https://github.com/agkomyint/taraference/releases/latest/download/tarafer-linux-x86_64.tar.gz
tar -xzf tarafer-linux-x86_64.tar.gz
chmod +x tarafer

# 2) put on PATH → ~/.local/bin/tarafer
./tarafer install
# if needed:  export PATH="$HOME/.local/bin:$PATH"

# 3) model + run
tarafer --download 0.5b
tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

### One-liner install + chat (Windows)

```powershell
Invoke-WebRequest -Uri "https://github.com/agkomyint/taraference/releases/latest/download/tarafer-windows-x86_64.zip" -OutFile tarafer-windows-x86_64.zip
Expand-Archive .\tarafer-windows-x86_64.zip -DestinationPath .
.\tarafer.exe install
# if needed:  $env:Path += ";$env:USERPROFILE\.local\bin"
.\tarafer.exe --download 0.5b
.\tarafer.exe models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

Interactive chat: type messages, `/reset`, `/quit`.  

### Tara 1.4.1 Base (custom top-2 MoE)

The official Tara 1.4.1 safetensors release is supported through a dedicated
taraference-native sparse-MoE Q4 pack. It preserves the model's trained four
experts, top-2 routing, full 32K vocabulary, and tied embedding/output head.

```powershell
# Interactive base-model testing
.\scripts\chat-tara-1.4.1.ps1

# Alpaca SFT checkpoint (Q8 reference)
.\scripts\chat-tara-1.4.1.ps1 -Sft

# One-shot continuation
.\scripts\chat-tara-1.4.1.ps1 -Prompt "The future of clean energy is"

# Fixed-length speed test
.\scripts\chat-tara-1.4.1.ps1 -Benchmark -N 128
```

Quality testing defaults to Q8. Pass `-Q4Fast` only for the faster base-model
Q4 path. The Tara 1.4.1 launcher enables temperature `0.7`, top-p `0.9`, and
repetition penalty `1.1`; pass no sampling flags to `tarafer` when measuring the
fully GPU greedy speed path.

Rebuild the pack from `D:\Tara_HQ\artifacts\release\tara1.4.1` with:

```powershell
.\scripts\export-tara-1.4.1.ps1
```

This custom architecture uses a native pack directory rather than a standard
llama.cpp GGUF. Do not enable `TARAFER_SPEED` or force top-1 when evaluating
quality; Tara 1.4.1 was trained with top-2 routing.
One-shot prompt:

```bash
tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --prompt "Say hi in one short sentence." -n 64
```

### After cloning the repo

```bash
git clone https://github.com/agkomyint/taraference.git
cd taraference
chmod +x scripts/get-binary.sh
./scripts/get-binary.sh              # → ~/.local/bin/tarafer
tarafer --download 0.5b
tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

Pin a version: `./scripts/get-binary.sh v0.2.0`

### Update on the same machine (no re-clone)

Once `tarafer` is installed, pull the newest GitHub release binary in place:

```bash
tarafer update                 # replace this binary with latest release
tarafer update v0.2.0          # pin a tag
tarafer update --install       # download latest into ~/.local/bin/tarafer
```

You do **not** need to re-run `curl` by hand after the first install.

### OpenAI-compatible server (same binary)

```bash
tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --serve
# default http://127.0.0.1:8787  — use --serve 3000 for another port
```

### Larger model (3B)

```bash
tarafer --download 3b
tarafer models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
```

Fits comfortably on **16 GB** GPUs (e.g. Tesla T4). On **4 GB** laptops prefer **0.5B** or lower context (`--ctx`).

### Why this is “fast”

| Path | Typical time to first chat |
|------|----------------------------|
| **Prebuilt release** (above) | Download binary + model only (no compile) |
| **Build from source** | Rust install + `cargo build --release` (minutes) |

Release assets (see [Releases](https://github.com/agkomyint/taraference/releases)):

| Asset | Purpose |
|-------|---------|
| `tarafer-linux-x86_64.tar.gz` | Linux packed binary (use this) |
| `tarafer` | Linux binary, unpacked |
| `tarafer-windows-x86_64.zip` | Windows packed binary (use this) |
| `tarafer.exe` | Windows binary, unpacked |
| `*.sha256` | Checksums |

| Command | What it does |
|---------|----------------|
| `tarafer install` | Copy binary → `~/.local/bin/tarafer` (PATH) |
| `tarafer update` | Self-update from latest GitHub Release |
| `tarafer --download 0.5b` | Fetch GGUF weights |

### Troubleshooting (prebuilt)

| Symptom | Fix |
|---------|-----|
| `tarafer: command not found` | Run `./tarafer install` and add `~/.local/bin` to `PATH` |
| `nvidia-smi` missing | Install NVIDIA driver / use a GPU machine |
| NVRTC / CUDA load errors | Install **CUDA 13.x toolkit** (not only the driver) |
| `CUDA_ERROR_INVALID_PTX` on old binaries | `tarafer update` to **v0.1.2+** (runtime GPU arch detection) |
| HF download slow / rate-limited | Set `HF_TOKEN` and re-run `--download` |

---

## Install from source

Use this to **develop** kernels, change code, or when no prebuilt matches your OS.

```powershell
# Windows
git clone https://github.com/agkomyint/taraference.git
cd taraference
.\scripts\install.ps1
```

```bash
# Linux
git clone https://github.com/agkomyint/taraference.git
cd taraference
./scripts/install.sh
```

No flags required. That installs Rust if needed, builds release, downloads models into `models/`, and runs `tarafer install` onto `~/.local/bin` (Linux).

Then:

```text
tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
# or: ./target/release/tarafer models/...
```

**Extra needs vs prebuilt:** Rust stable + Cargo, C++ linker (MSVC / `build-essential`). Same GPU + CUDA 13.x NVRTC at **run** time.

Optional flags for the install scripts: `--skip-models`, `--force` (see [`scripts/README.md`](scripts/README.md)).

Try “clone from zero” in a plain Linux container (git only, no GPU): [`test/`](test/).

## Layout

| Piece | Where | Role |
|-------|--------|------|
| **Inference** | `crates/core` | GGUF load, CUDA forward, `InferenceEngine`, `Session`, chat template |
| **Server** | `crates/cli/src/serve` | OpenAI-compatible HTTP (`/v1/models`, `/v1/chat/completions`) |
| **CLI** | `crates/cli` | interactive chat, `--profile`, `--serve` |

### Download models (Hugging Face)

Supported **Qwen2.5 Instruct Q4_K_M** (bartowski) land in **`models/`** (gitignored):

| Tag | ~Size | Notes |
|-----|------:|-------|
| `0.5b` | 0.4 GiB | fastest profile |
| `1.5b` | ~1 GiB | small step up |
| `3b` | ~1.9 GiB | default mid-small |
| **`7b`** | ~4.7 GiB | good on **T4 16 GB** |
| **`14b`** | ~9 GiB | T4 OK for Q4; lower `--ctx` if OOM |

```bash
tarafer --download list          # catalog
tarafer --download 7b            # larger model for profile
tarafer --download 14b
tarafer --download large         # 7b + 14b
tarafer --download profile       # 0.5b + 3b + 7b ladder
tarafer --download all           # 0.5b + 3b only (install default; not huge)
tarafer --download everything    # all sizes
```

```powershell
# Windows (from source)
cargo run --release -- --download 7b
cargo run --release -- --download large --models-dir D:\taraference\models
```

Optional: set `HF_TOKEN` if Hugging Face rate-limits you.

```powershell
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
```

No flags needed after models are present. Defaults:

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
| **`flash` (default)** | Eight-way split-KV flash decode; tiled `fastv2` prefill fallback |
| Incremental multi-turn | Append-only cache; only new tokens are prefilled |

### A/B decode backends (`--decode`)

Backends are a **registry** — add/remove without touching `layer.rs` launch code.

| Name | Meaning |
|------|---------|
| **`flash`** (default) | Eight-way split-KV flash decode + tiled prefill |
| `fastv2` | Compatibility/A-B fallback: tiled online attention |

```powershell
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile
cargo run --release -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode fastv2
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
