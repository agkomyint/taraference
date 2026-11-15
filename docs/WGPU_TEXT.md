# Native wgpu text inference

The experimental `taraference-wgpu` engine implements dense **Qwen2/Qwen2.5 GGUF**
in Rust and WGSL. It runs on a native wgpu adapter, alongside the existing CUDA
engine. It does not call CUDA, ONNX Runtime, Python, or another inference engine.
Qwen3.5's hybrid attention/DeltaNet, MoE, embeddings, chat sessions and server
integration are not implemented in this backend yet.

## Run

The independent executable needs no CUDA toolkit or CUDA Rust dependency:

```powershell
cargo run -p taraference-wgpu --release -- text gen "Explain ownership in Rust." -m models/Qwen2.5-3B-Instruct-Q4_K_M.gguf -n 64
```

The main CLI keeps CUDA as its default. Enable the optional backend and select it:

```powershell
cargo run -p taraference --release --features wgpu -- text gen "Explain ownership in Rust." -m 3b-qwen25 --backend wgpu -n 64
```

The main CLI also supports the existing sampling flags and JSON output. Its wgpu
context is capped at 2048 tokens with a diagnostic when a larger limit was requested.
The standalone executable defaults to 512; prompt plus generation must fit.

## Implementation and limits

- Supported source formats are F32, F16, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0; norm and bias
  vectors must be F32. The original GGUF file is never modified. Q4 payloads stay packed;
  Q6 payloads stay packed with a lossless device layout described below.
- Embedding lookup, RMSNorm, Q/K/V projections and biases, NeoX RoPE, causal grouped-query
  attention, SwiGLU, residual updates, full-vocabulary output and greedy argmax run in WGSL.
- Activations/reductions are fp32. KV entries are packed IEEE fp16. No optional subgroup,
  shader-f16, CUDA warp or DP4A capability is required by these shaders.
- Q4_K/Q6_K decode kernels unpack complete quantization blocks and reuse scales/bytes.
  Attention-output and feed-forward-down projections fuse the residual update.
- Compatible Q4_K feed-forward gate/up projections share one decode kernel, reuse input
  loads, accumulate independently and write SwiGLU directly. Other formats retain the
  unfused path. `TARAFER_WGPU_FUSE_GATE_UP=0` disables this fusion for diagnosis.
- Q6_K blocks can use a 256-byte device layout: the 192 original quant payload bytes
  plus sixteen fp32 `d * scale` coefficients, replacing the original 210-byte layout.
  These products are exact in fp32 for finite fp16 `d` and signed 8-bit scales; this
  adds no quantization. It trades weight storage for less per-token unpacking. A tensor
  falls back to the original layout if the expanded binding exceeds the adapter limit.
  `TARAFER_WGPU_Q6_SCALES=0` disables the expansion. Kernel settings in benchmark reports
  include the expanded tensor count and resident weight-buffer bytes, excluding KV/scratch.
- Prefill uses weight-format-specialized tiled fp32 GEMM for up to 32 tokens at a time,
  with causal attention and persistent KV across chunks. Both shared tiles are padded
  to avoid bank conflicts; tile depth defaults to 32 and can be set to 16/32/64 with
  `TARAFER_WGPU_GEMM_TILE_K` for diagnosis. Shared locations are initialized explicitly
  before reads, allowing redundant workgroup-memory clearing to be omitted.
  `TARAFER_WGPU_PREFILL_BATCH=1` selects sequential prefill for diagnosis; valid values
  are 1..32, default 32. Decode has no CUDA-graph equivalent.
- Token readback reuses a staging buffer and is copied in the forward command submission.
- Supported geometry: head dimension even and <=128, full-head RoPE, embedding width equal
  to query heads times head dimension, and a context of at most 2048 tokens.
- A large output/embedding tensor can exceed the default WebGPU 128 MiB binding limit.
  The engine requests the adapter's available native buffer limits and returns an error
  when a tensor exceeds them. Weight sharding for smaller adapter limits is not implemented.
- A hardware adapter is required; software CPU adapters are rejected. Vulkan on the
  RTX 3050 Ti Laptop GPU is tested. AMD, Intel and Apple hardware still need validation.

## Compare with existing Taraference CUDA

Build the comparison executable with CUDA enabled only for the reference backend:

```powershell
cargo build -p taraference-wgpu --release --features cuda-compare
target/release/tarafer-wgpu.exe bench --backend cuda -n 64 --ctx 512 --runs 3 --warmup 1 --report target/text-cuda.json
target/release/tarafer-wgpu.exe bench --backend wgpu -n 64 --ctx 512 --runs 3 --warmup 1 --report target/text-wgpu.json
python tools/compare_text_backends.py target/text-cuda.json target/text-wgpu.json --out target/text-comparison.json
```

Run the backends sequentially with other GPU work and compilation stopped. Both use
the same model, ChatML prompt/tokenizer, context and full vocabulary. The benchmark
resets the KV cache between requests, warms the complete request once, and measures
three requests. CUDA uses its existing default Flash decode and CUDA graphs; wgpu
uses its native kernels. Report precision differences rather than claiming identical math:
CUDA uses its existing activation quantization/DP4A paths where selected, while wgpu
multiplies the original quantized weights by fp32 activations.

Throughput measures **N-1 decode steps**, excluding the first output token, which
belongs to prefill/TTFT. Timing includes the token readback on each backend. Prefill
throughput includes the first full-vocabulary output projection. Fixed-length runs
intentionally continue through EOS to keep workloads comparable; their entire output
is not a quality evaluation. Normal `text gen` stops on EOS.

Each report includes binary full-vocabulary logits at four teacher-forced positions.
The same authored continuation is fed to both engines, so diverging free generation
cannot invalidate the comparison. The Python checker rejects different model sizes,
contexts, tokenized prompts, output counts or build profiles, then checks finite logits,
cosine similarity >=0.99 and relative RMSE <=0.05. It also reports top-1 matches, top-5
overlap and the matching generated prefix. This is a regression check against the
existing CUDA engine, not a perplexity or broad quality evaluation.

### Independent fp32 reference

The CUDA and wgpu paths do **not** use identical activation math. When the strict
CUDA-equivalence gate fails, keep that result visible and check the new implementation
against an independent fp32 reference rather than loosening the gate:

```powershell
python -m pip install --target target/gguf-reference --no-deps gguf==0.19.0
python tools/verify_wgpu_text_reference.py target/text-wgpu.json --gguf-lib target/gguf-reference --out target/text-wgpu-reference.json
```

This optional developer check also needs PyTorch and NumPy. It uses gguf-py's independent
dequantizer, evaluates the causal Qwen2 forward pass in PyTorch fp32 with TF32 disabled,
and rounds K/V to fp16 to match the cache. Layers and output-head chunks are loaded
individually to fit the reference GPU. The tool uses PyTorch CUDA for the reference
only; the product's wgpu path never invokes PyTorch or CUDA. It requires relative RMSE
<=0.005, cosine >=0.9999 and matching top-1 at every checked position.

## Second optimization pass (2026-11-12)

Same **Qwen2.5-3B-Instruct-Q4_K_M.gguf**, full vocabulary and fp32 activations,
**RTX 3050 Ti Laptop GPU / Vulkan**, release builds. The prior optimized executable
was preserved and rerun in this session; these paired baselines control for changes
in hardware conditions rather than mixing numbers from different runs.

31-token sky prompt, 64 output tokens, context 512, one warmup and three measured runs:

| Metric (median) | Previous optimized wgpu | New wgpu | Existing CUDA |
|---|---:|---:|---:|
| Decode | 24.70 tok/s | **29.40 tok/s** | 69.58 tok/s |
| Prefill | 43.20 tok/s | **46.00 tok/s** | 102.00 tok/s |
| TTFT | 717.51 ms | **673.94 ms** | 303.94 ms |

This pass adds **19.0% decode throughput** and **6.5% prefill throughput** over the
previous binary in the paired test. CUDA remains **2.37x faster in decode**. The
182-token/16-output single-run check improved from 21.74 to 26.51 decode tok/s and
38.50 to 42.04 prefill tok/s; TTFT fell from 4,727.53 to 4,329.72 ms. Treat that case
as a longer-context check rather than a robust performance estimate.

Changes: paired Q4_K gate/up projections with direct SwiGLU output, format-specialized
prefill pipelines, deeper reduction tiles with padding on both shared arrays, redundant
shared-memory initialization removed, and lossless aligned Q6_K scale expansion.
Wider logical reduction groups were tested and rejected after profiling made them slower.

The Q6 layout increases weight storage: for this model there are 37 expanded tensors,
36 fused feed-forward layers and **2,054,471,680 bytes (about 1.91 GiB)** of resident
weight buffers. This excludes KV cache, scratch and driver memory; it is not a peak-VRAM
measurement. Original model files and quantized values are unchanged. The scale products
are exact, verified against fp64 arithmetic across signed-byte scales and representative
finite fp16 values, including subnormals. The loader falls back per tensor if its expanded
binding would exceed the adapter limit.

Scale preparation happens once at model load. Release initialization in these two
new-backend runs took 3.55 s and 2.85 s; startup varies with file/shader caches and is
separate from the warm throughput figures. Use release builds for performance work:
the unoptimized debug main-CLI smoke run took about 25 s to load, while producing the
correct output. No cold-start speedup is claimed.

All checked logit buffers at the eight short/long teacher-forced positions are
**byte-identical to the previous optimized binary**. Generated prefixes match 64/64 and
16/16 tokens. The same cached independent fp32 checks pass: worst relative RMSE
0.0008716 (short) and 0.0016127 (long), with matching top-1 at every position.
CPU coefficient tests, all source-format and device-layout GPU fixtures, tail-tile tests,
paired-projection tests, model batch-boundary tests and main/optional-CUDA build checks pass.

Full paired reports: `target/wgpu-v2final-{before,after,cuda}-{sky,long}.json`;
summaries: `target/wgpu-v2final-{sky,long}-comparison.json`. Expanded layout and
fusion settings are recorded in each new report's `wgpu_kernel_settings` field.

## First optimization pass (2026-11-12)

Paired release benchmarks on the **RTX 3050 Ti Laptop GPU, 4 GiB, driver 596.36**, using
**Qwen2.5-3B-Instruct-Q4_K_M.gguf**, full 151,936-token vocabulary, fp32 wgpu activations
and packed-fp16 KV. The weights, prompt, context and output length are unchanged.
The original executable was preserved and rerun; all three backends ran sequentially
after compilation and GPU tests finished, with timestamps disabled.

Sky prompt below: 31 input tokens, 64 output tokens, context 512, one full warmup and
three measured runs per implementation. Values are medians:

| Metric | wgpu before | wgpu optimized | Existing CUDA |
|---|---:|---:|---:|
| Decode | 7.30 tok/s | **21.47 tok/s** | 66.39 tok/s |
| Prefill | 8.34 tok/s | **36.97 tok/s** | 95.48 tok/s |
| TTFT | 3,718.23 ms | **838.51 ms** | 324.67 ms |

This is **2.94× faster wgpu decode** and **4.43× faster prefill**. CUDA is still
**3.09× faster in decode** in this paired test. Earlier short optimization probes
reached approximately 25 decode tok/s; the repeated 64-token measurements above are
the reported comparison, not the best individual probe.

For the 182-token prompt, one warmup and one measured 16-token run, wgpu decode rose
from 6.97 to 20.90 tok/s, prefill from 7.88 to 36.06 tok/s, and TTFT fell from
23,101.78 to 5,046.93 ms. CUDA measured 64.95 decode tok/s and 1,964.83 ms TTFT.
Treat this single-run case primarily as a multi-chunk/cross-128-token validation.

Validation: both explicit GPU tests pass, covering all seven packing formats, GEMV,
tiled GEMM with bias/residuals and tail tiles, and model prefill lengths 1/31/32/33/65/182
with cache reset, invalid input rejection and decode continuation. Optimized full-model
logits also pass the unchanged independent fp32 criteria against the cached PyTorch
fixtures: worst relative RMSE **0.0008716** (short) and **0.0016127** (long), with all
eight top-1 choices matching. Exact tokenized inputs and verification positions are
checked against the original reference-source reports before using cached fixtures.
Generated tokens match the original wgpu run for **64/64** and **16/16** tokens.
The main CLI builds and uses the same optimized backend. No new accuracy reduction
or vocabulary truncation was introduced; non-NVIDIA hardware is still unvalidated.

Paired reports are `target/wgpu-opt-{before,after,cuda}-{sky,long}.json`, with summaries
in `target/wgpu-opt-{sky,long}-comparison.json`. Recheck a saved pair with:

```powershell
python tools/compare_wgpu_optimization.py target/wgpu-opt-before-sky.json target/wgpu-opt-after-sky.json target/wgpu-opt-cuda-sky.json --reference target/text-wgpu-sky-reference.json --reference-source target/text-wgpu-sky-final.json --out target/wgpu-opt-sky-comparison.json
```

## Initial implementation baseline (2026-11-12, before optimization)

**Model:** `Qwen2.5-3B-Instruct-Q4_K_M.gguf`, 1,929,903,264 bytes, full 151,936-token
vocabulary. **Hardware:** NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GiB, driver 596.36,
Windows. Both executables are release builds. CUDA uses the existing Flash decoder,
Q4_K×Q8 DP4A paths and active CUDA graphs; wgpu uses Vulkan, original packed weights,
fp32 activations and packed-fp16 KV.

For `Explain why the sky is blue in a short paragraph.`, 31 input tokens, 64 output
tokens, context 512, one warmup and three measured runs per backend:

| Metric | Taraference CUDA | Native wgpu |
|---|---:|---:|
| Median decode | 70.81 tok/s | 7.33 tok/s |
| Median prefill | 103.47 tok/s | 8.18 tok/s |
| Median TTFT | 299.59 ms | 3,788.42 ms |
| Load, separate from throughput | 1,656.11 ms | 2,215.21 ms |

CUDA is **9.66× faster in decode** in this test. The native port is functional but
not performance competitive yet. Sequential prefill and unfused scalar quantized GEMV
are major optimization gaps; cross-platform portability does not establish speed parity.

A separate 182-token prompt exercises attention beyond the 128-thread workgroup boundary.
With 16 output tokens, one warmup and one measured run, decode was 65.38 tok/s CUDA
versus 7.05 tok/s wgpu; TTFT was 1,969.30 ms versus 23,078.47 ms. This single-run case
is primarily a longer-context correctness check, not a robust performance estimate.

Validation results:

- All seven GPU unpack/GEMV fixtures pass. Shared core tests pass with CUDA (27 tests)
  and without CUDA (17 tests). The main CLI builds and returns `4` for the arithmetic
  smoke prompt; unsupported Qwen3.5 is rejected explicitly.
- Independent fp32 reference checks pass at all eight positions: worst relative RMSE
  **0.0006721** for the short prompt and **0.0018541** for the longer prompt. Top-1
  matches at every checked position. Reference: PyTorch 2.11.0+cu128, gguf-py 0.19.0.
- CUDA and wgpu also agree on all eight checked top-1 choices. Free generation matches
  for the first **53/64** tokens on the sky prompt and **16/16** on the longer prompt.
- The strict 5% CUDA-logit-equivalence check **fails** on some positions: maximum
  relative RMSE is **0.09828** (short) and **0.06834** (long). Keep this distinction
  from the passing independent fp32 checks. CUDA activation quantization and precision
  differ; no bitwise equivalence or broad quality parity is claimed.

Raw reports, binary logits and comparisons from this session are in
`target/text-{cuda,wgpu}-{sky,long}-final.*`, `target/text-{sky,long}-comparison.json`
and `target/text-wgpu-{sky,long}-reference.json`. The comparison executable was also
preserved as `target/release/tarafer-wgpu-compare.exe`; `tarafer-wgpu.exe` was then built
without the `cuda-compare` feature to validate the independent native build.

## Tests

```powershell
cargo test -p taraference-core --no-default-features
cargo test -p taraference-wgpu --lib -- --ignored --nocapture
cargo check -p taraference --features wgpu
cargo tree -p taraference-wgpu
```

The explicitly requested GPU tests check unpacking, GEMV and tiled GEMM for all seven
supported weight formats against independent CPU fixtures, including partial tiles,
bias and residual fusion. A model-level test compares batched and sequential prefill
at lengths 1, 31, 32, 33, 65 and 182, including decode continuation, cache reset and invalid
input rejection. Run GPU tests with `--test-threads=1` to serialize hardware work.
The default standalone dependency tree contains neither `cudarc` nor CUDA/ORT backends.

For per-dispatch GPU timestamps (diagnosis only, excluded from speed measurements):

```powershell
$env:TARAFER_WGPU_PROFILE = "1"
cargo run -p taraference-wgpu --release --example profile -- target/wgpu-profile.json
Remove-Item Env:TARAFER_WGPU_PROFILE
```

This diagnostic requires the adapter's optional timestamp-query support. Normal
inference requires no timestamp feature. The first profile found 119.69 ms of 123.60 ms
decode GPU time in GEMV (about 97%); block GEMV plus residual fusion reduced total
profiled GPU time to 38.65 ms and dispatch count from 580 to 508. Profiling adds
instrumentation overhead, so use the uninstrumented benchmark for throughput claims.
