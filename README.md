# ⚡ Taraference (`tarafer`)

> **Hyper-optimized single-user LLM & embedding inference engine built in Rust & CUDA — custom-engineered for maximum single-stream performance on specific NVIDIA GPUs & model families.**

[![Release](https://img.shields.io/github/v/release/agkomyint/taraference?style=flat-square&color=blue)](https://github.com/agkomyint/taraference/releases)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey?style=flat-square)](#system-requirements)
[![CUDA](https://img.shields.io/badge/CUDA-13.x-green?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-downloads)

---

## 🎯 Design Philosophy & Scope

See the [v0.7.0 roadmap](roadmap/0.7.0/ROADMAP.md) for planned priorities and
release checks.

> [!IMPORTANT]
> **Taraference is NOT a general-purpose, multi-tenant LLM server.**  
> It is a specialized, single-user performance engine built with hand-crafted CUDA kernels for specific model architectures.

- **👤 Single-User Decode Speed**: Engineered exclusively for maximum tokens per second for **1 user** — not multi-user batching density or concurrent server throughput.
- **🔬 Specialized Architecture Support**: Rather than supporting hundreds of arbitrary model types poorly, Taraference focuses on specialized, hand-tuned CUDA kernels for specific model families:
  - **Qwen2.5** (Official baseline & 3B scoreboard model)
  - **Qwen3.5** (High-speed kernel profiling & 4B evaluation)
  - **Qwen3 Embedding** (Dedicated 4B vector embedding & retrieval model)

If you need multi-tenant batch serving across dozens of concurrent users or support for niche legacy architectures, use frameworks like vLLM or llama.cpp. If you want **maximum single-stream generation speed** and zero-overhead execution on supported models, use **Taraference**.

---

## 🌟 What is Taraference?

**Taraference** (CLI command: **`tarafer`**) is a native local inference engine for Large Language Models (LLMs) and **Text Embedding Models** in GGUF format.

By combining a lightweight **Rust control plane** with **just-in-time compiled CUDA kernels** (NVRTC) tuned directly to your NVIDIA GPU compute architecture, `tarafer` bypasses heavy Python overhead and delivers instant startup with lightning-fast streaming generation.

---

## 🚀 Key Features

- **⚡ Specialized Split-K CUDA Kernels**: Custom packed GEMV kernels (Q4_K, Q6_K, Q8_0) with 8-way split-KV flash decoding for maximum single-stream tokens per second.
- **🔤 Native Text Embeddings**: Specialized support for `Qwen3-Embedding-4B` via `POST /v1/embeddings`, featuring Matryoshka dimension truncation and L2 vector normalization.
- **📦 Compact Native CLI**: A small Rust executable with no Python or PyTorch runtime. Optional image inference uses ONNX Runtime provider libraries and a separately downloaded model.
- **🔌 OpenAI-Compatible API Server**: Run a local server supporting `/v1/chat/completions` (streaming) and `/v1/embeddings` to plug into any app or web UI.
- **📥 Built-in Model Downloader**: Download supported quantized chat and embedding models directly from Hugging Face (`tarafer --download 3b` or `tarafer --download embed-4b`).
- **💬 Interactive Terminal Chat**: Rich terminal interface supporting multi-turn conversations, context resetting, and customized prompts.
- **📊 Real-Time Benchmarking**: Measure exact tokens/second, prefill performance, and VRAM utilization with the built-in `--profile` command.
- **🎯 Dynamic Hardware Compilation**: Auto-detects your GPU hardware architecture (`sm_75`, `sm_86`, `sm_89`, `sm_90`, etc.) and compiles tuned kernels on launch.
- **🖼️ Local Background Removal**: Run BiRefNet Lite locally and export transparent PNGs with `tarafer image remove-bg`.

### Background removal

Install the optional 115 MB FP16 model once, then create a transparent PNG:

```bash
tarafer image model download
tarafer image remove-bg photo.jpg
```

This writes `photo-no-bg.png`. `--device auto` tries CUDA, DirectML on Windows,
then CPU. See [the setup and model notes](docs/BACKGROUND_REMOVAL.md).

### Burmese text-to-speech (non-commercial)

```powershell
python tools/tts/export_f5_myanmar.py
tarafer tts say "မင်္ဂလာပါ" -o hello.wav --steps 16
```

Runs F5-Myanmar-TTS v2 (CC BY-NC 4.0) on ONNX Runtime. See [docs/TTS.md](docs/TTS.md).

---

## ⏱️ Quick Start (30 Seconds)

Get up and running with your first AI model in seconds without needing to build from source.

### 🐧 Linux (Ubuntu / Debian / Cloud SSH)

```bash
# 1. Download and unpack ready-to-run binary (~seconds)
curl -fsSL -o tarafer-linux-x86_64.tar.gz \
  https://github.com/agkomyint/taraference/releases/latest/download/tarafer-linux-x86_64.tar.gz
tar -xzf tarafer-linux-x86_64.tar.gz
chmod +x tarafer

# 2. Add to your PATH
./tarafer install

# 3. Download a lightweight 0.5B model & start chatting!
tarafer --download 0.5b
tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

---

### 🪟 Windows (PowerShell)

```powershell
# 1. Download and extract binary package
Invoke-WebRequest -Uri "https://github.com/agkomyint/taraference/releases/latest/download/tarafer-windows-x86_64.zip" -OutFile tarafer-windows-x86_64.zip
Expand-Archive .\tarafer-windows-x86_64.zip -DestinationPath .

# 2. Install CLI executable
.\tarafer.exe install

# 3. Download model & run chat
.\tarafer.exe --download 0.5b
.\tarafer.exe models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

---

## 💡 How to Use `tarafer`

### 1️⃣ Interactive Chat Mode
Start a natural conversation directly in your terminal:
```bash
tarafer models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
```
> **Chat commands:**  
> - Type your message and hit `Enter`  
> - `/reset` — Clear conversation history  
> - `/quit` — Exit chat session  

---

### 2️⃣ One-Shot Single Prompt
Send a quick prompt and receive streaming output straight to stdout:
```bash
tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --prompt "Explain quantum computing in two simple sentences."
```

---

### 3️⃣ Run an OpenAI-Compatible API Server (Chat & Completion)
Turn your GPU machine into a local OpenAI-compatible inference API:
```bash
tarafer models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --serve
```
Default endpoint: `http://127.0.0.1:8787` *(or specify a port with `--serve 3000`)*.

> [!WARNING]
> The server listens on all network interfaces and does **not** implement
> authentication. OpenAI client API-key or `Authorization` values are accepted
> only for client compatibility and are not validated. Keep it behind a
> firewall or an authenticating reverse proxy; do not expose it directly to an
> untrusted network or the public internet.

For non-recurrent transformer models, the server automatically reuses the
exact token prefix shared with the preceding chat request. This accelerates
repeated system prompts and any other identical leading prompt content without
changing OpenAI request semantics. Non-streaming responses report the hit as
`usage.prompt_tokens_details.cached_tokens`, and server logs include both
`cached_prompt_tok` and `prefilled_prompt_tok`. Hybrid/recurrent models retain
full-prefill behavior until recurrent prefix snapshots are supported. Set
`TARAFER_PREFIX_CACHE_OFF=1` to disable reuse for correctness or performance
A/B comparisons.

#### Example Chat Request (via `curl`):
```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-3B-Instruct-Q4_K_M",
    "messages": [{"role": "user", "content": "What are 3 tips for learning Rust?"}],
    "stream": true
  }'
```

---

### 4️⃣ Text Embeddings API (`/v1/embeddings`)
Taraference supports high-performance vector embeddings for semantic search, vector databases, and RAG workflows.

#### Step 1: Download the official 4B Embedding Model
```bash
tarafer --download embed-4b
```

#### Step 2: Serve the Embedding Model
```bash
tarafer models/Qwen3-Embedding-4B-Q4_K_M.gguf --serve
```

#### Step 3: Query the OpenAI-Compatible Embeddings Endpoint
```bash
curl http://127.0.0.1:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Embedding-4B-Q4_K_M",
    "input": ["Retrieval-Augmented Generation (RAG) with local vector embeddings."],
    "dimensions": 1024
  }'
```
> **Features:**  
> - Accepts single strings or arrays of text strings (`"input": ["text1", "text2"]`).  
> - **Matryoshka Truncation**: Truncate output vector dimensions dynamically (e.g. `256`, `512`, `1024`, or full `2560` dims).  
> - Automatically applies L2 vector normalization.  

---

### 5️⃣ Benchmark & Speed Test
Measure your system's exact inference speed (tokens/sec) and performance metrics:
```bash
tarafer models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile
```

### Experimental out-of-core MoE paging

`--experimental` enables a Kimi-K3-inspired storage path for Tara MoE pack
directories. Routed expert files remain packed on disk and only one complete
MoE layer is resident on the GPU at a time:

```bash
tarafer path/to/tara-moe-pack --experimental --prompt "Hello" -n 8
```

This first implementation is a correctness and memory-capacity path, not a
speed path. It synchronously reads all experts for each routed layer because
the current CUDA router indexes a layer-wide expert slab. CUDA graph capture is
disabled. Runtime logs report layer reads and a final summary containing cache
hits, misses, disk traffic, and load time. GGUF files are rejected explicitly;
normal inference is unchanged when the flag is absent.

---

## 📥 Supported Models & Catalog

`tarafer` focuses on optimized support for the following model architectures:

| Tag | Target Model | Recommended VRAM | Model Size | Type | Focus Area |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`0.5b`** | Qwen2.5 0.5B Instruct | 4 GB | ~0.4 GB | Chat | Ultra-fast response, laptop GPUs |
| **`1.5b`** | Qwen2.5 1.5B Instruct | 4 GB - 6 GB | ~1.0 GB | Chat | Light text generation |
| **`3b`** *(Default)* | Qwen2.5 3B Instruct | 8 GB - 16 GB | ~1.9 GB | Chat | Official Scoreboard Baseline |
| **`4b`** | Qwen3.5 4B Instruct | 8 GB - 16 GB | ~2.5 GB | Chat | Modern fast kernel evaluation |
| **`7b`** | Qwen2.5 7B Instruct | 12 GB - 16 GB | ~4.7 GB | Chat | High-accuracy general tasks |
| **`14b`** | Qwen2.5 14B Instruct | 16 GB+ | ~9.0 GB | Chat | Large model inference on T4/A10 |
| **`embed-4b`** | Qwen3 Embedding 4B | 6 GB - 8 GB | ~2.5 GB | **Embedding** | Multilingual & Code retrieval |
| *(Custom)* | Tara 1.4.1 | 8 GB - 16 GB | ~1.5 GB | **MoE Chat** | Custom sparse top-2 MoE architecture |

---

## 💻 System Requirements

| Component | Minimum / Recommended |
| :--- | :--- |
| **Operating System** | Linux (x86_64, Ubuntu 22.04+) or Windows 10/11 (x86_64) |
| **GPU** | NVIDIA GPU (GTX 16-series, RTX 20/30/40 series, T4, A10, A100, H100) |
| **Driver & Runtime** | NVIDIA Display Driver + **CUDA 13.x Toolkit** (with NVRTC support) |
| **Storage** | ~10 MB binary + model file space (~0.4 GB to 9 GB depending on model) |

---

## 🔄 Self-Updating

Updating `tarafer` to the latest release takes just one command without re-downloading or rebuilding manually:

```bash
tarafer update
```

---

## 🔬 Technical Architecture & Deep Dive

Interested in low-level GPU kernel designs, Split-K GEMV matrix math, embedding pipeline specifics, NVRTC CUDA compilation, or building from source?

👉 **Check out the full [TECHNICAL.md](TECHNICAL.md) guide** for detailed information on:
- Internal crate architecture (`crates/core`, `crates/cli`, `crates/gguf`)
- Custom CUDA kernel implementations (`gemv.cu`, `embed.cu`, Flash Decode)
- Text Embedding pipeline execution (EOS hidden state extraction & Matryoshka L2 normalization)
- Memory management & FP16 Key-Value cache design
- Step-by-step instructions for building & hacking on the codebase
