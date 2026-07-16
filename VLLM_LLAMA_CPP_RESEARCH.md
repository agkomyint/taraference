# vLLM and llama.cpp GPU inference research for Taraference

## Scope

This is a read-only source study of two local repositories. Neither repository was modified, built, or benchmarked during this review.

- vLLM: `D:\vllm`, commit `1b30ae4ca403588f1df37901d3b530d1fe5c48c4`
- llama.cpp: `D:\llama.cpp`, commit `07d937828636e305bc0cfe738b288f9ab05ff748`
- Review date: 2026-07-16

The goal is not to copy either engine. It is to identify methods that fit Taraference's GGUF Q4_K_M workload, especially single-request decode on T4, RTX 3050 Ti, L4, and later H100.

## Executive summary

The two projects optimize different layers of the problem:

| Project | Strongest relevant methods | Best fit for Taraference |
|---|---|---|
| llama.cpp | GGUF-native quantized vector kernels, per-architecture launch tables, fused quantized operations, separate small-batch and multi-token paths | Immediate decode work on T4, 3050 Ti, and L4 |
| vLLM | Prepacked tensor-core weight kernels, shape heuristics, persistent graph buffers, paged KV cache, speculative decoding | Experimental Ada backend, H100 work, batching and serving |

For the current Qwen2.5-3B Q4_K_M single-token scoreboard, llama.cpp's MMVQ design is the closer reference. Marlin is important, but it is not a drop-in replacement: it expects a different prepacked weight layout and is primarily a W4A16 tensor-core GEMM design. A Taraference implementation would need a load-time GGUF Q4_K conversion, preservation of Q4_K scale/min semantics, and proof that tensor cores are beneficial at `M=1`.

The highest-value implementation order is:

1. Build an explicit kernel-selection table keyed by GPU architecture, quant type, matrix shape, and token count.
2. Bring Taraference's Q4_K and Q6_K packed dot-product path closer to llama.cpp's register-level DP4A organization.
3. Fuse gate projection, up projection, bias, SiLU, and multiply so intermediate FFN vectors are not written and read again.
4. Add a dedicated multi-token quantized matrix path for prefill and speculative verification.
5. Prototype a Marlin-inspired prepacked Q4_K backend for SM80/86/89 only after the GGUF-native path is fully measured.
6. Use a CUTLASS/Machete-style backend for SM90 H100 rather than stretching an Ada kernel onto Hopper.

No speed gain should be assumed from these ideas. Each item requires a correctness test and two repeatable profiles with the mandatory 3B scoreboard model.

## llama.cpp methods

### 1. MMVQ is specialized for very small token counts

The main quantized matrix-vector implementation is in:

- `D:\llama.cpp\ggml\src\ggml-cuda\mmvq.cu`
- `D:\llama.cpp\ggml\src\ggml-cuda\vecdotq.cuh`
- `D:\llama.cpp\ggml\src\ggml-cuda\quantize.cu`

MMVQ is selected only for small numbers of input columns/tokens. llama.cpp then switches to MMQ or another matrix path as the token count grows. This avoids forcing one kernel organization to serve both decode (`M=1`) and prefill (`M>1`).

Its launch configuration is a table, not a single rule. `calc_nwarps()` considers:

- GPU architecture family;
- quantization type;
- number of destination columns/tokens;
- register pressure and complexity of the quantized dot product.

For the generic NVIDIA path, one to four destination columns use four warps. The Turing table reduces Q2_K through Q6_K single-column work to two warps. This directly supports Taraference's direction of using a four-warp L4/Ada kernel while retaining a distinct T4 path.

Recommended Taraference adaptation:

- Replace broad rules such as `cc_major >= 8` with a data table keyed by `sm_75`, `sm_86`, `sm_89`, and later `sm_90`.
- Store separate settings for Q4_K, Q6_K, projection shape, and token count.
- Include warps per output, rows per block, split-K threshold, and fused/non-fused kernel choice.
- Select the table once during model loading; do not branch repeatedly in the hot loop.

### 2. Packed GGUF values are consumed directly with DP4A

`vecdotq.cuh` defines separate register-level dot products for Q4_K x Q8_1 and Q6_K x Q8_1. Important properties are:

- packed nibbles and high bits are decoded in registers;
- four signed 8-bit products are accumulated with DP4A;
- scale and minimum corrections are accumulated separately for Q4_K;
- loops are compile-time unrolled;
- the activation block stores data needed by the quant format, avoiding repeated work in the weight loop.

For Q4_K, llama.cpp computes both the quantized dot term and the activation-sum term needed by the block minimum. For Q6_K, low and high bits are combined in registers and shifted to the signed range before DP4A.

Recommended Taraference adaptation:

- Audit the instruction and memory-load count of `dot_q4_k_*` and `dot_q6_k_*` against these implementations.
- Prefer aligned 32-bit or 128-bit loads followed by register unpacking over byte-by-byte global loads.
- Keep the original GGUF layout as the correctness reference. Any repacked layout must have an exhaustive CPU-vs-GPU block test before model profiling.
- Add deterministic tests for every position in a 256-value Q4_K and Q6_K super-block. The rejected L4 Q6 experiment showed that plausible throughput is meaningless when bit layout is wrong.

This is more compatible with Taraference than Marlin because it operates directly on GGUF K-quants.

### 3. Fusion extends beyond two projections

llama.cpp's MMVQ kernel can fuse:

- the normal projection and gate projection;
- bias loads;
- optional scales;
- SiLU/GEGLU-style activation;
- multiplication of the gate and up results.

Bias and gate metadata are prefetched before the main dot-product loop. Both projection accumulators remain in registers, and the final activation is applied before the result is stored.

Taraference already shares activation reads for gate and up projection, but still materializes both output vectors before the separate SiLU-multiply kernel. A fused epilogue could remove:

- two global output stores from the dual projection;
- two global reads by the activation kernel;
- one kernel launch per transformer layer.

Recommended design:

- Add a dedicated FFN decode kernel, not a flag-heavy universal kernel.
- Accumulate gate and up together.
- Perform the cross-warp reduction.
- Apply `silu(gate) * up` in the final writer warp.
- Write only the final FFN activation.

This is the most concrete next decode experiment because it reduces memory traffic and launch work without changing the model's quantization format.

### 4. MMQ is separate and uses a different activation layout

The larger-token path is in:

- `D:\llama.cpp\ggml\src\ggml-cuda\mmq.cu`
- `D:\llama.cpp\ggml\src\ggml-cuda\mmq.cuh`

MMQ quantizes activations into a layout designed for tiled quantized matrix multiplication. It uses architecture-dependent tile configurations and can use stream-K decomposition on recent NVIDIA hardware. Template instantiations are generated per quant type instead of routing all types through runtime branches.

Recommended Taraference adaptation:

- Keep GEMV/MMVQ for ordinary one-token decode.
- Add a true tiled quantized GEMM for prompt prefill and speculative verification.
- Select the crossover using token count, K/N shape, quant type, and architecture.
- Avoid optimizing prefill by increasing the complexity or register use of the decode kernel.

This will matter more for H100 and batching than for the current single-token L4 number.

### 5. Flash attention has multiple architecture and shape paths

Relevant files include:

- `D:\llama.cpp\ggml\src\ggml-cuda\fattn-vec.cuh`
- `D:\llama.cpp\ggml\src\ggml-cuda\fattn-tile.cuh`
- `D:\llama.cpp\ggml\src\ggml-cuda\fattn-mma-f16.cuh`
- `D:\llama.cpp\ggml\src\ggml-cuda\cp-async.cuh`

llama.cpp does not use one attention kernel for every shape. It has vector, tiled, WMMA, and MMA implementations with compile-time specializations. On Ampere and newer, the MMA path can stage K/V tiles through shared memory with `cp.async`, including multiple pipeline stages and explicit wait/commit operations.

Recommended Taraference adaptation:

- Keep the current vector/flash decode for short context until profiling shows attention is material.
- Add context-length buckets for attention selection.
- Consider `cp.async` only for SM80+ and only where enough K/V work exists to overlap global-memory latency.
- Generate specialized head-dimension variants instead of a large runtime-generic kernel.

At the current 3B L4 profile, model weights dominate bandwidth, so this is behind FFN and quantized projection work.

### 6. CUDA graphs are cached and updated, not treated as one immutable capture

The CUDA graph implementation is mainly in:

- `D:\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu`
- `D:\llama.cpp\ggml\src\ggml-cuda\common.cuh`

llama.cpp keeps graph instances per graph key, checks whether tensor/node properties changed, captures when required, attempts `cudaGraphExecUpdate`, and falls back to reinstantiation if an update is incompatible. Unused graph entries are evicted.

Recommended Taraference adaptation:

- Retain stable device addresses for decode buffers.
- Key graphs by execution shape and backend, not by an assumption that all decode steps are identical.
- Update kernel parameters or graph executables when context-dependent values change.
- Measure graph replay coverage and fallbacks in the profiler.

Taraference already has CUDA graph support, so the useful lesson is lifecycle robustness rather than simply enabling capture.

## vLLM methods

### 1. Marlin prepacking and tensor-core pipeline

Relevant files are:

- `D:\vllm\csrc\libtorch_stable\quantization\marlin\marlin.cuh`
- `D:\vllm\csrc\libtorch_stable\quantization\marlin\marlin.cu`
- `D:\vllm\csrc\libtorch_stable\quantization\marlin\marlin_template.h`
- `D:\vllm\csrc\libtorch_stable\quantization\marlin\marlin_mma.h`
- `D:\vllm\csrc\libtorch_stable\quantization\marlin\gptq_marlin_repack.cu`
- `D:\vllm\csrc\libtorch_stable\quantization\marlin\awq_marlin_repack.cu`

Marlin's central methods are:

- repack weights once into a tensor-core-friendly layout;
- use 256 threads/eight warps to hide latency while retaining registers per warp;
- use a four-stage shared-memory pipeline;
- load global data asynchronously with `cp.async` on SM80+;
- dequantize close to the MMA operation;
- use hardware MMA instructions with FP16/BF16/FP8/INT8 variants;
- choose thread K/N tile shapes with problem-shape heuristics;
- split work across SMs and coordinate partial results through workspace locks or atomic accumulation.

The important general lesson is not the exact tile size. It is that weight layout, load pipeline, compute primitive, and scheduling heuristic are designed together.

Constraints for Taraference:

- GGUF Q4_K has super-block scales and minimums that do not match standard GPTQ/AWQ packing.
- A new packed representation consumes extra VRAM unless the original weights are released.
- Model load time will increase.
- `M=1` can underutilize tensor-core GEMM layouts.
- Repacking must be deterministic and validated at the block and full-layer levels.

Recommended experiment:

1. Convert only one Q4_K layer at load time into an experimental packed layout.
2. Keep a CPU reference for that layer.
3. Benchmark `M=1`, `M=2..8`, and prefill shapes separately on SM86 and SM89.
4. Continue only if the end-to-end 3B profile wins, not merely the isolated GEMM.

### 2. Shape-specific scheduling is a first-class component

Marlin tries multiple valid thread configurations and accounts for:

- problem M, N, and K;
- quantization bit width and group size;
- shared-memory consumption;
- blocks per SM;
- number of SMs;
- activation ordering and zero points;
- bias and output accumulation mode.

Machete takes the same idea further by generating multiple CUTLASS schedules and selecting them with a heuristic. There is explicitly no single schedule expected to perform best for all shapes.

Recommended Taraference adaptation:

- Introduce a small `HardwarePlan` selected at model load.
- Its key should contain compute capability, SM count, memory bandwidth class, quant type, N/K shape, and operation class.
- Start with a checked-in table derived from mandatory profiles.
- Later add an opt-in first-run tuner whose results are cached by GPU UUID, driver, Taraference version, model architecture, and exact tensor shape.
- Never tune using generated text as the only correctness signal.

### 3. Machete is the better conceptual reference for H100

The overview is in:

- `D:\vllm\csrc\libtorch_stable\quantization\machete\Readme.md`
- `D:\vllm\csrc\libtorch_stable\quantization\machete\machete_mainloop.cuh`
- `D:\vllm\csrc\libtorch_stable\quantization\machete\machete_prepack_kernel.cuh`

Machete is described as a CUTLASS-based successor to Marlin for Hopper. It prepackages weights for wider shared-memory loads and generates type, tile, scheduler, scale, and zero-point specializations.

Recommended Taraference adaptation:

- Treat H100/SM90 as a separate backend family.
- Use CUTLASS/CuTe or equivalent Hopper primitives rather than an NVRTC kernel designed around Ada DP4A.
- Prepack weights into a layout suitable for Hopper tensor cores/TMA.
- Generate only the exact Q4_K-derived specializations Taraference supports to control compile time and binary size.

This work should follow L4 correctness and dispatch infrastructure. H100 should not inherit L4 constants merely because both have compute capability major version 8 or higher; H100 is SM90 and architecturally different.

### 4. Persistent buffers enable reliable full-model CUDA graphs

Relevant vLLM areas include:

- `D:\vllm\vllm\v1\worker\gpu\model_runner.py`
- `D:\vllm\vllm\v1\worker\gpu\cudagraph_utils.py`
- `D:\vllm\vllm\v1\worker\gpu\input_batch.py`

vLLM copies changing inputs into preallocated buffers whose addresses were recorded during graph capture. It pads metadata to captured sizes and maintains multiple graph sizes/modes. This makes full-model replay possible despite changing request contents.

Recommended Taraference adaptation:

- Preallocate all decode inputs, scalar metadata, logits, and scratch buffers.
- Update buffer contents, not pointers.
- Capture graph buckets only where the launch topology changes.
- Include sampling in the graph only if it removes a measurable synchronization and preserves deterministic testing.

### 5. Paged and quantized KV cache primarily improve serving and long context

Relevant files include:

- `D:\vllm\vllm\v1\attention\backend.py`
- `D:\vllm\vllm\v1\attention\ops\chunked_prefill_paged_decode.py`
- `D:\vllm\vllm\v1\worker\gpu\block_table.py`

vLLM stores KV in blocks addressed through block tables, supports multiple physical layouts, and has FP8 KV modes. This enables efficient request growth, reuse, batching, and memory allocation without large contiguous per-request buffers.

Recommended Taraference adaptation:

- Do not prioritize paging for the current single-session 3B speed target; indirection can add overhead.
- Add paged KV when concurrent serving or prefix reuse becomes a goal.
- Test FP8 KV as a separate long-context backend on L4/H100, where reduced KV bandwidth may outweigh conversion cost.
- Preserve contiguous F16 KV as the short-context baseline.

### 6. Speculative decoding changes accepted tokens per target pass

vLLM contains several speculative systems under `D:\vllm\vllm\v1\spec_decode`, including draft-model, Medusa, and other proposer designs. It tracks draft throughput, acceptance rate, and mean accepted length.

This is different from making one target-model token faster. It can raise user-visible tokens per second by verifying several candidates in one target pass, but requires:

- a compatible draft/proposer;
- an efficient multi-token verification kernel;
- high acceptance rate;
- correct rollback and KV-cache handling;
- reporting accepted-token throughput separately from raw target decode speed.

Recommended Taraference adaptation:

- First build a strong `M=2..8` quantized verification path.
- Then evaluate PLD/ngram or a small draft model.
- Record acceptance length, target passes per accepted token, and end-to-end tokens/s.
- Do not compare speculative accepted tokens/s to ordinary one-token decode without labeling the method.

## Proposed Taraference hardware plan

| GPU | Architecture | First-choice decode direction | Avoid assuming |
|---|---:|---|---|
| Tesla T4 | SM75 | GGUF-native DP4A MMVQ, two-warp K-quant variants, conservative register use | `cp.async`, Ada launch constants |
| RTX 3050 Ti | SM86 | Four-warp GGUF-native path, optional staged loads, thermal-aware profiling | L4 bandwidth and sustained clocks |
| NVIDIA L4 | SM89 | Four-warp fused FFN, FP8 experiments, optional Q4_K prepack prototype | H100/TMA scheduling |
| NVIDIA H100 | SM90 | CUTLASS/Machete-style generated tensor-core backend, larger batched tiles | Ada DP4A being optimal |

GPU RAM size should choose memory features such as packed-weight retention, KV capacity, graph buckets, and workspace size. It should not directly choose warp count. Warp count and tile shape should come from architecture, SM resources, operation shape, and measured occupancy.

## Implementation roadmap

### Phase 1: measurement and dispatch foundation

- Add per-operation CUDA event timing for quantization, QKV, output projection, FFN gate/up, SiLU multiply, down projection, attention, and sampling.
- Record kernel choice, registers if available, block dimensions, shared memory, and graph replay status.
- Create `HardwarePlan` with exact SM-family entries.
- Keep the existing kernel as fallback for unknown GPUs.

### Phase 2: GGUF-native decode improvements

- Reorganize Q4_K/Q6_K packed loads and DP4A loops using llama.cpp as a conceptual reference.
- Implement the fused gate/up/SiLU/multiply epilogue.
- Tune Q4_K and Q6_K independently.
- Specialize common Qwen2.5 matrix shapes at compile time.

### Phase 3: multi-token path

- Add a tiled Q8-activation quantized MMQ backend.
- Tune crossover points for decode, short verification, and prefill.
- Add stream-K only where output geometry leaves SMs idle.

### Phase 4: prepacked tensor-core experiment

- Define a documented Q4_K-derived packed layout.
- Write exhaustive pack/unpack and layer-output tests.
- Prototype SM86/SM89 MMA kernels with staged asynchronous loads.
- Measure VRAM, load time, isolated kernel time, and end-to-end tokens/s.

### Phase 5: H100 backend

- Add SM90-specific build and dispatch.
- Use generated CUTLASS/CuTe schedules and Hopper data movement.
- Tune single-request and batched serving independently.

## Benchmark and correctness gates

Every performance experiment must follow Taraference's scoreboard policy:

- Model: `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf`
- Use that model for `--profile`, speed A/B, regression claims, and iterative loops.
- Never use the 0.5B model as a speed signal.
- Run a deterministic kernel-level correctness test before generation.
- Run a coherent smoke generation before profiling.
- Profile the baseline and candidate at least twice on the same GPU and thermal state.
- Reject a candidate if output is corrupt, even if its displayed tokens/s is higher.
- Separate ordinary decode tokens/s, speculative accepted tokens/s, prefill tokens/s, and batch throughput.

Recommended result key:

```text
gpu_uuid + compute_capability + driver + taraference_commit + model_sha256
+ kernel_plan + decode_backend + graph_mode + context_bucket
```

## Cache research: latency versus decode speed

The word "cache" covers several independent mechanisms. They must be measured
against the metric they can actually improve:

| Cache | Taraference status | Expected benefit | Single-stream decode tok/s |
|---|---|---|---|
| Per-layer KV cache | Implemented, f16 K/V | Avoid recomputing previous tokens | Essential, but already in the baseline |
| CUDA graph executable | Implemented for one-token decode | Reduce CPU launch overhead | Already in the baseline |
| Globally quantized Q8 activation | Implemented and reused by fused projections | Avoid redundant input quantization | Already in the baseline |
| Prefix/prompt KV reuse | Not yet a cross-request server cache | Skip repeated system/prompt prefill | Improves TTFT and request throughput, not steady decode rate |
| Paged KV allocation | Not implemented | Sharing, eviction, batching, prefix blocks | Mainly serving scalability at short contexts |
| FP8/quantized KV | Experimental future option | Capacity and long-context attention bandwidth | Negligible at short context; requires quality validation |
| GPU L2 weight residency | Hardware-managed | Reuse small hot data | The 1.79 GiB model cannot fit in L4 on-chip cache |

Measured on the L4 with the mandatory Qwen2.5 3B model, the Q6 output head ran
at 1.14 ms, 96.3% DRAM throughput, and 288.64 GB/s. At the multi-turn profile's
final context (about 459 tokens), f16 KV reads are only about 16.5 MB per token,
versus roughly 1.8 GB of model weights. Halving KV traffic therefore saves less
than 0.5% of total streamed bytes in this test. Prefix caching should still be
implemented for production serving, but it must not be presented as a route
from roughly 100 to 120 ordinary single-stream decode tok/s.

vLLM's relevant design is block-hashed prefix reuse plus paged KV block tables;
see `vllm/v1/worker/block_table.py`, `vllm/v1/worker/gpu/block_table.py`, and the
common-prefix handling in `vllm/v1/worker/gpu_model_runner.py`. The transferable
Taraference design is an immutable block hash over model identity, adapter and
token prefix, with reference-counted GPU KV blocks and LRU eviction. llama.cpp's
session/KV reuse is more suitable as a simpler single-session reference than as
a high-concurrency scheduler.

## Ideas not recommended as the first step

- Copying Marlin source or layout directly. Its input format and integration assumptions differ from GGUF Q4_K.
- Selecting kernels only from total VRAM. Capacity is not a proxy for scheduler, register file, tensor-core generation, or bandwidth.
- One universal kernel for T4 through H100.
- Paged KV cache solely to improve the current single-session short-context score.
- FP8 KV without a long-context A/B and output-quality check.
- Speculative decoding before multi-token verification is efficient.
- Reporting a microkernel win as an end-to-end Taraference win.

## Source map

### llama.cpp

- Small-batch quantized dispatch and fusion: `ggml/src/ggml-cuda/mmvq.cu`
- Q4_K/Q6_K DP4A implementation: `ggml/src/ggml-cuda/vecdotq.cuh`
- Activation Q8_1 conversion: `ggml/src/ggml-cuda/quantize.cu`
- Multi-token quantized matrix path: `ggml/src/ggml-cuda/mmq.cu`, `mmq.cuh`
- Async copy helpers: `ggml/src/ggml-cuda/cp-async.cuh`
- Flash-attention variants: `ggml/src/ggml-cuda/fattn-*.cuh`
- CUDA graph lifecycle: `ggml/src/ggml-cuda/ggml-cuda.cu`, `common.cuh`

### vLLM

- Marlin pipeline and scheduler: `csrc/libtorch_stable/quantization/marlin/`
- Machete Hopper design: `csrc/libtorch_stable/quantization/machete/`
- Hardware/backend selection: `vllm/model_executor/layers/quantization/`, `vllm/platforms/`
- CUDA graph persistent-buffer machinery: `vllm/v1/worker/gpu/`
- Paged attention and KV layouts: `vllm/v1/attention/`
- Speculative decoding: `vllm/v1/spec_decode/`

These paths are references for concepts and experiments. Their licenses and attribution requirements must be reviewed before reusing any implementation code.
