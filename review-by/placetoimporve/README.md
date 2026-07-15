# Future Roadmap & Actionable Improvements for `taraference` Core Engine

This document synthesizes state-of-the-art LLM inference optimization techniques across the industry (`llama.cpp`, `ExLlamaV2`, `vLLM`, `SGLang`, `FlashAttention-3`, `Speculative Decoding`, and `Marlin`) and translates them into concrete, actionable improvement tasks for the **`taraference`** core inference engine ([`crates/core`](file:///d:/taraference/crates/core)).

---

## 🚀 1. Immediate / High-Impact CUDA & Mechanical Sympathy Tasks

### [ ] 1.1 Implement Split-K GEMV Reduction for Decoding (`GEMV_REVSPLITK`)
- **Industry Context**: In `llama.cpp` and `GemLite`/`Triton`, single-token auto-regressive decoding (`n_tok == 1`) is purely memory-bandwidth bound. When computing $y = W \cdot x$ for tall/skinny quantized weight matrices ($M = 2048\text{--}3072, N = 1$), launching standard warps across output columns ($N$) leaves many Streaming Multiprocessors (SMs) underutilized on modern GPUs.
- **Current `taraference` State**: [`gemv_q4_k`](file:///d:/taraference/crates/core/src/cuda/kernels/gemv.cu#L83), [`gemv_q6_k`](file:///d:/taraference/crates/core/src/cuda/kernels/gemv.cu#L109), and [`gemv_q8_0`](file:///d:/taraference/crates/core/src/cuda/kernels/gemv.cu#L135) use `GEMV_WARPS = 8` (8 warps per block computing 8 output columns) with input vector staging in shared memory (`xs[]`).
- **Action Item**:
  1. Write a new CUDA kernel `gemv_q4_k_splitk` in [`gemv.cu`](file:///d:/taraference/crates/core/src/cuda/kernels/gemv.cu) that splits the $K$-dimension (inner reduction dimension `n_rows`) across $S$ thread blocks (e.g., $S=4$ or $S=8$).
  2. Each block computes partial dot-product accumulators into a small temporary device buffer `partial_out[S * n_cols]`.
  3. Launch a fast secondary reduction kernel `gemv_splitk_reduce` (or use atomic adds if deterministic ordering is not strictly mandated) to sum the partials into `out[j]`.
  4. **Expected Gain**: Up to **15%–30% higher decode `tok/s`** on RTX 3050 Ti by fully saturating all SMs during single-token generation.

### [ ] 1.2 Capture Decoding Loop into CUDA Graphs (`CudaGraph`)
- **Industry Context**: High-performance inference engines (`vLLM`, `TensorRT-LLM`, `ExLlamaV2`, `llama.cpp`) record the exact sequence of GPU kernel launches for single-token generation into a **CUDA Graph** (`cuGraph`).
- **Current `taraference` State**: During every token step in [`Session::generate_with`](file:///d:/taraference/crates/core/src/session.rs#L195-L220), `forward_greedy` launches ~3–4 individual CUDA kernels per layer across $L=36$ layers (~110–145 API calls per token). On Windows (`pwsh`), CPU-side driver call overhead and synchronization latency can bottleneck generation.
- **Action Item**:
  1. Leverage `cudarc::driver::CudaGraph` and `CudaStream::begin_capture()` / `end_capture()`.
  2. After the initial prompt prefill is complete, capture the execution of `forward_chunk(tokens=[next], pos0=cache.len, compute_logits=true, cache)` once for each typical sequence bucket or fixed pointer address.
  3. Replay the captured graph (`graph.launch()`) during the token generation loop.
  4. **Expected Gain**: **Eliminates 100% of CPU-side kernel launch and driver overhead**, saving 1–3 ms per token on Windows driver stacks.

### [ ] 1.3 Operator Fusion for Chunked Prefill (`RMS_NORM` + `QKV GEMM`)
- **Industry Context**: `ExLlamaV2` and `llama.cpp` aggressively fuse pre-attention normalization (`RMS_NORM`) and multi-projection linear transformations directly into single kernels (`qkv_gemm_fused`) to minimize global HBM roundtrips during prompt prefilling.
- **Current `taraference` State**: In [`layer.rs`](file:///d:/taraference/crates/core/src/cuda/layer.rs#L44-L90), `rms_norm` writes `xb` to HBM, after which three separate `gemv` / `gemm` calls (`wq`, `wk`, `wv`) read `xb` back from HBM.
- **Action Item**:
  1. Combine `wq`, `wk`, and `wv` weight allocations into a single contiguous concatenated GPU matrix (`wqkv`) during model loading (`load.rs`).
  2. Launch a single `gemm` or `gemv` call over `wqkv`, writing directly into a unified `qkv_buf`.
  3. **Expected Gain**: **Fewer HBM read/write passes and 2 fewer kernel launches per layer** during both prefill and decode.

---

## 🧠 2. Quantization, Memory Bandwidth & Tensor Core Tasks (Medium Term)

### [ ] 2.1 Integrate Marlin-Style Tensor Core INT4 Kernels (`FP16` Activations $\times$ `INT4` Weights)
- **Industry Context**: Standard CUDA SIMT scalar quantization kernels struggle to achieve theoretical 4x speedups over FP16 due to sub-optimal memory coalescing and lack of Tensor Core utilization. **Marlin** (`vLLM` / `SGLang`) uses asynchronous global memory loads (`cp.async`), double buffering in shared memory, and `mma.sync` Tensor Core instructions.
- **Current `taraference` State**: Uses NVRTC SIMT dot-product loops (`dot_q4_k_col_xs` in `gemv.cu`) which run on standard CUDA cores.
- **Action Item**:
  1. Add an optional NVRTC kernel module (`kernels/matmul/marlin_q4.cu`) implementing `mma`-based matrix multiply for `Q4_K` / `Q4_0` weights.
  2. Expose a new command-line flag `--kernel marlin` or auto-select when `batch_size > 1` (chunked prefill).
  3. **Expected Gain**: **2x–3x prefill throughput speedup** when processing multi-token prompt chunks on Ampere (`sm_86`) Tensor Cores.

### [ ] 2.2 KV Cache Quantization (`FP8` / `INT8` / `Q4_0` Cache)
- **Industry Context**: Long-context inference (>8,000 tokens) on consumer GPUs (`4GB` / `8GB` VRAM) quickly runs out of memory when storing `f16` KV caches. Engines like `llama.cpp` (`--cache-type-k q8_0`/`q4_0`) and `Aphrodite`/`vLLM` (`FP8_E4M3` cache) compress cached keys and values.
- **Current `taraference` State**: [`CudaKv`](file:///d:/taraference/crates/core/src/cuda/kv.rs#L5) stores keys and values in `f16` (`u16` slices, 18.43 KiB/token for Qwen2.5-3B).
- **Action Item**:
  1. Add `WType::Q8_0` or `FP8` variants to `CudaKv::alloc_kv()`.
  2. Update `copy_kv_f16` to `copy_kv_q8` (`u8` scales + quantized values) and adjust `kv_load` inside [`fast_v2.cu`](file:///d:/taraference/crates/core/src/cuda/kernels/attn/fast_v2.cu#L53) to dequantize keys/values on-the-fly in shared memory before computing attention scores.
  3. **Expected Gain**: **Halves KV cache VRAM from ~18 KiB/tok down to ~9 KiB/tok**, enabling up to **10,000+ context tokens** on a 4 GB RTX 3050 Ti alongside Q4 weights.

---

## ⚡ 3. Speculative & Algorithmic Acceleration Tasks

### [ ] 3.1 Training-Free Speculative Decoding (`Prompt Lookup Decoding`)
- **Industry Context**: **Prompt Lookup Decoding (PLD)** is a zero-cost, model-agnostic speculative decoding strategy. For tasks like coding, editing, summarization, and RAG/chat, future tokens frequently echo phrases from the input prompt or earlier turns. PLD uses fast CPU/GPU n-gram matching against existing context (`cache.len`) to propose candidate token sequences (2–5 tokens), which the main target model verifies in a single parallel chunked forward pass (`n_tok > 1`).
- **Current `taraference` State**: Strict 1-token greedy generation loop (`for step in 0..max_new`).
- **Action Item**:
  1. In [`Session::generate_with`](file:///d:/taraference/crates/core/src/session.rs#L195), after decoding token $t$, check if the last 2 or 3 tokens ($n$-gram) match any previously generated or prompt sequence.
  2. If a match is found, propose the next $K=3$ tokens from that historical match as draft candidates $D = [t_1, t_2, t_3]$.
  3. Run `model.forward_chunk(D, pos0, compute_logits=true, cache)` (`n_tok = K`) in a single pass.
  4. Verify which draft tokens match the model's argmax output at each position and accept all matching tokens simultaneously.
  5. **Expected Gain**: **1.4x–2.2x effective generation speedup** on code and grounded chat tasks with **zero additional VRAM or secondary model weights**.

### [ ] 3.2 Feature-Level Speculative Extrapolation (`EAGLE-3` / `Medusa` Hooks)
- **Industry Context**: When maximum token-per-second output is required without a secondary draft model, attaching lightweight multi-token prediction heads (`Medusa` / `EAGLE`) directly to the target model's final hidden states allows predicting $3\text{--}5$ future tokens concurrently using tree attention.
- **Action Item**: Add architectural support in [`GpuLayer`](file:///d:/taraference/crates/core/src/cuda/types.rs#L25) and `engine.rs` to load optional `medusa.*.weight` or `eagle.*.weight` tensors from GGUF if present.

---

## 🌐 4. Server, Concurrency & Architecture Adaptability Tasks

### [ ] 4.1 PagedAttention & Continuous Batching for OpenAI API (`/v1/chat/completions`)
- **Industry Context**: `vLLM`'s **PagedAttention** partitions KV cache into fixed virtual memory blocks (e.g., 16 tokens/block) mapped via a `block_table`. Coupled with **Continuous Batching** (iteration-level scheduling), this allows new requests to join the inference pipeline immediately as older requests finish tokens, eliminating VRAM fragmentation and request queue serialization.
- **Current `taraference` State**: [`InferenceEngine::chat_completion`](file:///d:/taraference/crates/core/src/engine.rs#L118-L150) serializes requests across one static `CudaKv` arena (`self.kv.clear()` per request).
- **Action Item**:
  1. Transition `CudaKv` from contiguous `[max_seq, stride]` slices into a block pool `[num_blocks, block_size=16, stride]`.
  2. Pass a `block_table` (`CudaSlice<i32>`) to [`attn_fast_v2`](file:///d:/taraference/crates/core/src/cuda/kernels/attn/fast_v2.cu#L8) so `kv_load` resolves virtual sequence indices $t$ to physical block addresses: `physical_block = block_table[t / 16]; offset = t % 16;`.
  3. Update `serve/mod.rs` (`axum` server) to use an asynchronous channel queue (`mpsc::channel`) that feeds a continuous batching step loop inside `InferenceEngine`.
  4. **Expected Gain**: **3x–10x higher server throughput under multi-user concurrent HTTP traffic** without memory exhaustion.

### [ ] 4.2 Dynamic GPU Architecture Detection (`sm_XX` Auto-Config)
- **Current `taraference` State**: Hardcoded `arch: Some("sm_86")` inside [`CudaModel::load_with`](file:///d:/taraference/crates/core/src/cuda/load.rs#L62).
- **Action Item**:
  1. Query `cuDeviceGetAttribute` via `ctx.get_device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR)` before NVRTC compilation.
  2. Dynamically format the target architecture string (`format!("sm_{}{}", major, minor)`).
  3. **Expected Gain**: Seamless out-of-the-box compatibility across Turing (`sm_75`), Ampere (`sm_86`), Ada Lovelace (`sm_89`), and Hopper (`sm_90`) without manual recompilation errors or suboptimal SIMT code generation.

### [ ] 4.3 GPU-Side Non-Deterministic Sampling (`Sampler` Module)
- **Current `taraference` State**: Only performs deterministic greedy decoding (`argmax_f32` in [`layer.rs`](file:///d:/taraference/crates/core/src/cuda/layer.rs#L463)).
- **Action Item**:
  1. Create a new `cuda/sampler.rs` module and define a `SamplerConfig` (`temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`).
  2. Implement an NVRTC kernel `sample_top_k_top_p` that performs parallel softmax and top-$K$ filtering on GPU logits before sampling via a random seed (`curand`).
  3. Expose these parameters in `SessionOptions` and the `/v1/chat/completions` request struct.
  4. **Expected Gain**: Full OpenAI specification compliance and high quality creative/conversational outputs.

---

## 📊 Summary Checklist Matrix

| Priority | Task ID | Feature / Improvement | Target Component | Complexity | Expected Impact |
| :---: | :---: | :--- | :--- | :---: | :--- |
| 🔥 **P0** | **1.1** | **Split-K GEMV Reduction** | `cuda/kernels/gemv.cu` | Medium | +15%–30% Single-Token Decode Speed |
| 🔥 **P0** | **1.2** | **CUDA Graph Capture** | `cuda/forward.rs`, `session.rs` | Medium | 0 CPU Launch Overhead (-1~3ms/tok) |
| 🔥 **P0** | **4.2** | **Dynamic `sm_XX` Detection** | `cuda/load.rs` | Low | Universal GPU Compatibility |
| ⚡ **P1** | **3.1** | **Prompt Lookup Decoding** | `session.rs`, `forward.rs` | Low | +40%–120% Speed on Code/Editing |
| ⚡ **P1** | **4.3** | **GPU-Side Sampling Engine** | `cuda/sampler.rs`, `serve/` | Medium | Temperature, Top-K/P API Support |
| ⚡ **P1** | **1.3** | **QKV Operator Fusion** | `cuda/layer.rs`, `load.rs` | Medium | Fewer HBM Reads & Kernel Launches |
| 🛠️ **P2** | **2.2** | **`Q8_0` / `FP8` KV Cache** | `cuda/kv.rs`, `fast_v2.cu` | High | 2x Longer Context Window in VRAM |
| 🛠️ **P2** | **2.1** | **Marlin INT4 Tensor Core `mma`**| `cuda/kernels/matmul/` | High | 2x–3x Faster Prompt Chunk Prefill |
| 🛠️ **P2** | **4.1** | **PagedAttention & Continuous Batching** | `cuda/kv.rs`, `engine.rs` | Very High | Enterprise Multi-User Server Scaling |
