# taraference

CUDA multi-turn inference for **Qwen2.5** GGUF (defaults for **RTX 3050 Ti 4GB**).

**CLI name:** **`tarafer`** (short command). The repo/crate is still `taraference`.

**Performance goal:** one user, maximum decode tokens/sec — not multi-user concurrency. See [GOAL.md](GOAL.md).

---

## Fast install (recommended) — prebuilt Linux binary

**Use this if you want first inference ASAP** (Lightning Studio, cloud SSH, Ubuntu GPU box, etc.).

You **do not** need Rust, Cargo, or a local compile. CI ships a ready Linux binary on every version tag.

### What you need (runtime only)

| Requirement | Notes |
|-------------|--------|
| **OS** | Linux **x86_64** (Ubuntu 22.04-class is what we build on) |
| **GPU** | NVIDIA (T4, 3050 Ti, A10, …). Arch is detected at load (`sm_75`, `sm_86`, …) |
| **Driver** | `nvidia-smi` works |
| **CUDA toolkit 13.x** | Must include **NVRTC** (runtime kernel compile). Many cloud images already have this |
| **Disk** | ~10 MB binary + ~380 MB for the 0.5B model (or ~2 GB for 3B) |

Windows / macOS: use [Install from source](#install-from-source) for now (no prebuilt yet).

### One-liner install + chat (on PATH)

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

Interactive chat: type messages, `/reset`, `/quit`.  
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
| `tarafer-linux-x86_64.tar.gz` | Packed binary (use this) |
| `tarafer` | Same binary, unpacked |
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

Use this to **develop** kernels, change code, or run on **Windows**.

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

Supported Q4_K_M weights land in **`models/`** (gitignored):

```powershell
# both 0.5B + 3B (skip files that already exist)
cargo run --release -- --download

# only one
cargo run --release -- --download 0.5b
cargo run --release -- --download 3b

# force re-download
cargo run --release -- --download all --force

# custom directory
cargo run --release -- --download --models-dir D:\taraference\models
```

Sources (bartowski GGUF): `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, `Qwen2.5-3B-Instruct-Q4_K_M.gguf`.  
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
