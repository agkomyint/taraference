# Goals: fast single-stream inference and a 750 tok/s frontier

## North stars

1. Maximize **accepted decode tokens per second for one user** on one GPU.
2. Reach **750 accepted decode tok/s** on at least one useful model without hiding
   concurrency inside the number.
3. Keep a separate aggregate-throughput metric for future batching work.

Decode tok/s is generated/accepted tokens divided by decode wall time. Prefill and
TTFT are secondary. Correct text, stable KV state, and repeatable results are gates.

## Scoreboard

All profiling, A/B work, performance claims, and iterative speed loops use only
`Qwen2.5-3B-Instruct-Q4_K_M.gguf`. The 750 tok/s north star remains attached to
this usable 3B scoreboard unless this policy is explicitly revised.

Never compare tok/s across models, quantizations, GPUs, prompt scripts, or metric
definitions. Always state model, quant, GPU, compute capability, backend, and
single-stream versus aggregate.

Modern architectures may be implemented and correctness-tested, but they are not
performance scoreboards under the current policy.

## What we optimize

Priority order:

1. Packed quantized GEMV/MMVQ or Marlin-class Tensor Core kernels.
2. Multi-token accepted decode: EAGLE/MTP/draft verification.
3. CUDA graphs and elimination of CPU/GPU synchronization.
4. Fused normalization, projections, activation, and residual operations.
5. Flash attention and quantized KV for long contexts.
6. Correct architecture-specific dispatch (`sm_75`, `sm_86`, Ada and newer).

Paged attention, continuous batching, and prefix sharing are in the aggregate track.
They do not count as a single-user win unless per-user accepted tok/s also improves.

## Required reporting

Every accepted performance change reports:

- exact model and quantization;
- GPU name, compute capability, NVRTC architecture, clocks/thermal caveats;
- `overall_decode_tps`, first/last decode tok/s, and context drop;
- single-stream or aggregate definition;
- two candidate runs plus a same-environment baseline when the delta is small;
- correctness/quality observations and any numerical approximation introduced.

## v0.4 regression target

The frozen v0.3 T4 baseline is Qwen2.5-3B Q4_K_M, default fastv2, approximately
31.28 tok/s in a fresh same-host run. v0.4 must beat it on T4 and remain faster on
the RTX 3050 Ti laptop under comparable thermal conditions.

Measured v0.4 candidate (2026-07-16, coherent 128-token single-stream run):

- Tesla T4 (`sm_75`): **59.572 tok/s**, up 90.5% from v0.3's 31.279 tok/s.
- RTX 3050 Ti Laptop (`sm_86`): **37.849 tok/s**, up 40.1% from the preceding
  v0.4 candidate's 27.02 tok/s; first-to-last drop was 0%.
- Path: flash decode, CUDA graph, 32-warp Q4×Q8 DP4A, packed Q6 decode weights,
  fused QKV/gate-up, and reusable global-Q8 activation quantization.

## 750 tok/s interpretation

Published serving numbers often aggregate many concurrent requests. Taraference
records both:

- `single_stream_accepted_tps`: the latency north star;
- `aggregate_accepted_tps`: all accepted output tokens across active requests.

The 750 sprint is not complete unless the reported field explicitly reaches 750.
If only aggregate reaches it, label that as the aggregate milestone, not the
single-stream milestone.

## v0.5 measured candidate

Qwen2.5-3B-Instruct-Q4_K_M on Tesla T4 (`sm_75`), coherent single-stream decode:

- fixed 128-token prompt: **64.0–64.1 tok/s**;
- mandatory multi-turn profile: **58.926–59.045 overall decode tok/s**;
- first/last profile: **53.603–53.893 / 59.226–59.381 tok/s**;
- context drop: **-10.49% to -10.18%** (later turns were faster);
- peak VRAM: **2737 MiB**.

The v0.5 additions are an activation-reusing equal-width Q4_K gate+up kernel,
aligned compact Q6_K decode weights, and a cooperative four-warp Q6_K down
projection that performs split-K reduction inside each output block.
