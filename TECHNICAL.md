# Taraference — Technical Architecture & Engine Specs

This document contains low-level technical specifications, internal kernel architecture, developer workflows, and benchmarking details for **Taraference** (`tarafer`).

---

## 🏗 System Architecture & Design Principles

Taraference is built from the ground up in **Rust** and **CUDA** with a single target in mind: **Maximum single-stream token generation throughput on NVIDIA consumer and enterprise GPUs**.

Unlike multi-tenant engines (vLLM, TGI) that optimize for batch throughput and concurrent request density, Taraference optimizes for **zero-latency single-user interactive decode token generation**.

```
                           +---------------------------+
                           |  tarafer CLI / REST API   |
                           +-------------+-------------+
                                         |
                                         v
                           +---------------------------+
                           |      taraference-core     |
                           |   (Session & Engine Core) |
                           +-------------+-------------+
                                         |
                                         v
       +---------------------------------+---------------------------------+
       |                                                                   |
       v                                                                   v
+--------------+                                                   +---------------+
|  tarafer-gguf|                                                   |   CUDA Engine |
| (mmap parser)|                                                   | (cudarc/NVRTC)|
+--------------+                                                   +---------------+
                                                                           |
                                                    +----------------------+----------------------+
                                                    |                      |                      |
                                                    v                      v                      v
                                             +--------------+       +--------------+       +--------------+
                                             | Split-K GEMV |       | Flash Decode |       | Dynamic NVRTC|
                                             |  (Q4/Q5/Q8)  |       | (8-way KV)   |       | CUDA Graphs  |
                                             +--------------+       +--------------+       +--------------+
```

---

## 🛠 Low-Level Kernel Specifications

### 1. Mixed-Precision Quantized Matrix Vector Multiplication (GEMV)
- **Quantization Support:** GGUF `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `f16`, `f32`.
- **Fused Projections:** Dual fused gate + up projections, reducing kernel launch overhead by 50% during SwiGLU / MoE expert execution.
- **Warp DP4A Acceleration:** Vectorized 8-bit dot products utilizing hardware int8 tensor/simd operations for scale unpacking.
- **Split-K In-Block Reduction:** Cooperative thread block reduction for high-width hidden dimensions.

### 2. Attention & KV Cache Memory Management
- **Half-Precision (f16) KV Cache:** Native FP16 Key-Value cache storage, halving VRAM bandwidth consumption compared to FP32.
- **Flash Decode Architecture:** 8-way split-KV flash decode kernel for long multi-turn contexts.
- **Append-Only Context State:** Stateful incremental KV allocation ensuring prompt prefill is only computed on new incoming tokens.

### 3. Execution Pipeline & Dynamic NVRTC
- **Runtime NVRTC Compilation:** CUDA kernel templates compiled dynamically at load time, automatically targeting the detected device compute capability (`sm_75`, `sm_86`, `sm_89`, `sm_90`).
- **CUDA Graph Capture:** High-frequency decode loops captured into static CUDA Graphs to eliminate driver host-to-device kernel launch latencies.

---

## 📦 Workspace Crate Layout

| Path | Crate | Role & Functionality |
|------|-------|----------------------|
| `crates/cli` | `taraference` | Main executable binary (`tarafer`). Provides interactive chat, HuggingFace model downloader, OpenAI REST server, and benchmarking profiler. |
| `crates/cli/src/serve` | N/A | Axum-based OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/models`, `/health`). |
| `crates/core` | `taraference-core` | Complete inference engine runtime. Handles model weights loading, CUDA context initialization, KV cache management, and session state. |
| `crates/core/src/cuda/` | N/A | Low-level GPU memory allocators, tensor loaders, matrix multiplication wrappers, and NVRTC CUDA kernel definitions (`kernels/*.cu`). |
| `crates/gguf` | `taraference-gguf` | High-performance zero-copy mmap reader for GGUF model files. |

---

## 🧪 Profiling & Benchmarking Architecture

Taraference includes a built-in profiling engine accessible via `--profile`.

### Executing a Profile Run
```bash
tarafer models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile
```

### Profile Artifact Structure
Every profile run generates structured logs in `profile-logs/`:
- `profile_<date>_<model>_<decode>.txt`: Comprehensive execution breakdown, breakdown per layer, and KV cache stats.
- `latest_<model>.txt`: Benchmark baseline for the specific model for A/B comparison.
- `latest.txt`: Most recent overall run benchmark.
- `index.csv`: CSV log recording timestamp, model name, backend, tokens/sec, KV memory footprint, and context window size.

---

## ⚡ Custom Decoding Backends

Decoding kernels use a modular registry interface defined in `crates/core/src/cuda/decode.rs`.

| Backend Name | Description | Recommended Usage |
|--------------|-------------|-------------------|
| `flash` *(default)* | 8-way split-KV flash decode with tiled prefill | General production and long-context inference |
| `fastv2` | Tiled online attention fallback | Performance verification & A/B comparison |

To benchmark an alternative backend:
```bash
tarafer models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode fastv2
```

---

## 🔩 Building from Source

### Prerequisites
- **Rust Toolchain:** Stable edition (`rustc`, `cargo`)
- **C++ Compiler:** `gcc` / `clang` on Linux or `MSVC` (`cl.exe`) on Windows
- **NVIDIA CUDA Toolkit:** Version 13.x (including NVRTC library headers)
- **NVIDIA GPU Driver:** Modern driver supporting CUDA 13.x

### Build Commands
```bash
# Clone repository
git clone https://github.com/agkomyint/taraference.git
cd taraference

# Build release binary
cargo build --release -p taraference

# Run tests
cargo test --workspace
```

---

## 📜 Developer Guidelines & Scoreboard Policy

For official contribution rules, model testing baselines, and performance scoreboard tracking:
- Refer to [`AGENTS.md`](../AGENTS.md) for profiling rules.
- Refer to [`GOAL.md`](../GOAL.md) for target throughput benchmarks.
- Refer to [`WORKFLOW.md`](../WORKFLOW.md) for SSH development workflows.
