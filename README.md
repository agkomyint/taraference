# ⚡ Taraference (`tarafer`)

> **Blazingly fast, lightweight LLM inference engine built in Rust & CUDA — tuned for maximum single-user generation speed on NVIDIA GPUs.**

[![Release](https://img.shields.io/github/v/release/agkomyint/taraference?style=flat-square&color=blue)](https://github.com/agkomyint/taraference/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey?style=flat-square)](#system-requirements)
[![CUDA](https://img.shields.io/badge/CUDA-13.x-green?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-downloads)

---

## 🌟 What is Taraference?

**Taraference** (CLI command: **`tarafer`**) is an ultra-optimized, native local inference engine for Large Language Models (LLMs) in GGUF format.

While traditional inference frameworks focus on multi-user server throughput or heavy Python runtimes, **Taraference is built for speed, simplicity, and low latency**. It bypasses heavy Python overhead entirely by combining a high-performance **Rust control plane** with **just-in-time compiled CUDA kernels** (NVRTC) directly tuned to your specific NVIDIA GPU architecture.

Whether you're running a small 0.5B model on a laptop GPU or a 14B parameter model on a server GPU, `tarafer` delivers instant launch times and smooth, lightning-fast streaming text generation.

---

## 🚀 Key Features

- **⚡ Blazing Token Generation**: Custom packed CUDA kernels (Q4_K, Q6_K, Q8_0) with 8-way split-KV flash decoding for maximum single-stream tokens per second.
- **📦 Zero-Dependency Single Binary**: A single ~10MB executable. No PyTorch, no Python runtime, no complex environment setups.
- **🔌 Drop-In OpenAI API Server**: Run an OpenAI-compatible server (`/v1/chat/completions`) with full streaming support (`stream: true`) to plug into any app, web UI, or SDK.
- **📥 Built-in Model Downloader**: Download popular quantized models directly from Hugging Face with one simple command (`tarafer --download 3b`).
- **💬 Interactive Terminal Chat**: Rich terminal interface supporting multi-turn conversations, context resetting, and customized prompts.
- **📊 Real-Time Benchmarking**: Measure exact tokens/second, prefill performance, and VRAM utilization with the built-in `--profile` command.
- **🎯 Automatic GPU Tuning**: Automatically detects your GPU hardware compute capability (`sm_75`, `sm_86`, `sm_89`, `sm_90`, etc.) and compiles tuned kernels on launch.

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

### 3️⃣ Run an OpenAI-Compatible API Server
Turn your GPU machine into a local OpenAI-compatible inference API:
```bash
tarafer models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --serve
```
Default endpoint: `http://127.0.0.1:8787` *(or specify a port with `--serve 3000`)*.

#### Example API Request (via `curl`):
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

### 4️⃣ Benchmark & Speed Test
Measure your system's exact inference speed (tokens/sec) and performance metrics:
```bash
tarafer models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile
```

---

## 📥 Model Hub Catalog

`tarafer` includes a built-in catalog to instantly fetch popular GGUF model weights from Hugging Face:

```bash
tarafer --download <tag>
```

| Tag | Recommended VRAM | Model Size | Best For |
| :--- | :---: | :---: | :--- |
| **`0.5b`** | 4 GB | ~0.4 GB | Ultra-fast response, laptop GPUs, quick tests |
| **`1.5b`** | 4 GB - 6 GB | ~1.0 GB | Light coding & text generation tasks |
| **`3b`** *(Default)* | 8 GB - 16 GB | ~1.9 GB | High quality balance, ideal for daily use |
| **`7b`** | 12 GB - 16 GB | ~4.7 GB | Advanced reasoning & comprehensive answers |
| **`14b`** | 16 GB+ | ~9.0 GB | Large-scale model performance on enterprise GPUs |

> **Useful Download Shortcuts:**  
> - `tarafer --download list` — See full model catalog  
> - `tarafer --download profile` — Download 0.5B, 3B, and 7B models for speed comparisons  
> - `tarafer --download all` — Download standard default models  

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

Interested in low-level GPU kernel designs, Split-K GEMV matrix math, NVRTC CUDA compilation, or building from source?

👉 **Check out the full [TECHNICAL.md](TECHNICAL.md) guide** for detailed information on:
- Internal crate architecture (`crates/core`, `crates/cli`, `crates/gguf`)
- Custom CUDA kernel implementations (`gemv.cu`, Flash Decode)
- Memory management & FP16 Key-Value cache design
- Step-by-step instructions for building & hacking on the codebase

---

## 📄 License

Distributed under the [MIT License](LICENSE).
