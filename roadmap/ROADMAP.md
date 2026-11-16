# Taraference roadmap

> The noun/verb CLI (`text`, `speech`, `image`, `serve`, `models`, `bench`, `doctor`) is the only
> interface; the old flag-style spellings are gone. The model registry (pinned revision + SHA-256, licence prompt,
> `list|info|pull|verify|rm`) covers text, embeddings, speech and image; see
> [docs/MODELS.md](../docs/MODELS.md). `speech transcribe` (Burmese) and `bench speech` exist on ONNX Runtime, unreleased; see
> [docs/STT.md](../docs/STT.md).

## Direction

Taraference is a **lean local runtime for small, specialised AI models**, each optimised to run
as fast as possible on modest hardware (a 4 GB laptop GPU is the reference machine).

It is **not** an agent framework, not a Claude/ChatGPT competitor, and does not serve large
language models. A model belongs here if it is small, useful on its own, and we can make it fast.

One CLI, one model registry, one OpenAI-compatible server, many small tasks:

| Task | Status | Engine |
|---|---|---|
| Text generation (small GGUF LLMs: Qwen2.5-3B, Qwen3.5 0.8B–4B) | shipped | custom CUDA GGUF engine |
| Text embeddings | shipped | custom CUDA GGUF engine |
| Image background removal (BiRefNet Lite) | shipped in 0.7 | ONNX Runtime |
| Text-to-speech, Burmese (F5-Myanmar-TTS, MMS-TTS) | shipped, unreleased | ONNX Runtime |
| Speech-to-text, Burmese (`freococo/myanmar_asr`) | working, unreleased | ONNX Runtime |
| Video (e.g. small segmentation/captioning) | research only | undecided |
| More specialised models (OCR, translation, classification) | backlog | by demand |

## CLI: one noun per task, verbs underneath

```
tarafer text   gen <prompt|->  [-m MODEL] [-n N] [--temperature …] [--json]
tarafer text   chat            [-m MODEL]            # thin REPL; may be dropped (see below)
tarafer text   embed <text|->  [-m MODEL]
tarafer speech say <text|->    -o out.wav [--engine f5|mms] [--lexicon FILE]
tarafer speech transcribe <audio>  [-m MODEL] [--lang my]        # en needs a second model
tarafer image  remove-bg <file>
tarafer serve  [--port 8787]                          # one server, all tasks
tarafer models list | pull <id> | rm <id> | info <id> # one registry, all task types
tarafer bench  <task> [-m MODEL]                      # same report for every task (text today)
tarafer doctor                                        # GPU, ORT providers, installed models
tarafer update | install
```

Rules:

- **Verbs are subcommands, never flags.**
- **Every task is pipe-friendly.** `-` reads stdin; `--json` prints machine-readable results;
  progress goes to stderr.
- **Advanced tuning is hidden.** `--decode`, `--no-cuda-graph` and `--experimental` move under
  `--help-advanced`. Sampling flags apply to `text` only.
- **Chat is optional.** `text gen` is the primary text interface and `serve` covers clients. The
  REPL is a thin wrapper over the same session; if it costs maintenance, drop it. No tools, no agents,
  no memory.

## Platform principles

taraference is a runtime and installer, not a model distributor.

- **Weights never live in the repo, releases or our servers.** `models pull` fetches from the
  upstream publisher; the user accepts the upstream licence.
- **Licence is shown before download.** Non-commercial or gated models need `--accept-license`,
  and `models list --commercial` filters to commercially usable ones. (Not built yet: the catalog
  has no licence field today, so `models list` only points at the upstream repo.)
- **Pinned revision and SHA-256** per registry entry, verified after download.
- **Conversion runs on the user's machine** (e.g. the F5 ONNX export). Hosting converted files is
  the repo owner's decision and is not assumed.
- **Lean by default:** a new task is added only if the model is small, useful alone, and fast here.
- This is engineering guidance, not legal advice; check each upstream licence yourself.

## Server

`tarafer serve` exposes one OpenAI-compatible surface, loaded lazily per task:

| Endpoint | Task |
|---|---|
| `/v1/chat/completions`, `/v1/completions` | text (exists) |
| `/v1/embeddings` | embed (exists) |
| `/v1/audio/speech` | speech say (new) |
| `/v1/audio/transcriptions` | speech transcribe (new) |

## Model registry

One `models` registry (implemented, `docs/MODELS.md`) covers every task. Each entry
records: id, task, source, pinned revision and SHA-256, size, **licence and whether it is
non-commercial**, and the runtime. `models list` shows the licence so non-commercial weights are
never a surprise. Weights stay out of releases. Hosting converted ONNX files (F5, MMS) so users
need no Python export is a **decision for the repo owner**; it is not assumed here.

## Engines

- **Backend direction:** keep CUDA for NVIDIA and add native wgpu alongside it for
  broader GPU support across AMD, Intel and Apple hardware through the available
  Vulkan, Direct3D 12 and Metal backends. Support depends on adapter capabilities;
  compiling for a platform alone does not establish model compatibility or performance.
  CPU execution remains a separate path. Browser WebGPU is outside the initial scope.
- **GGUF engine (custom CUDA):** text and embeddings. Unchanged; this is where the speed work
  and the Qwen2.5-3B scoreboard live.
- **ONNX Runtime:** speech and image today, with CUDA → DirectML → CPU fallback. One shared
  session/provider loader (currently duplicated across image and tts). From 0.9.0 it is an
  opt-in fallback only; native engines (shared `taraference-ops`) replace it per model.
- **Benchmarks:** every task reports load time, run time and a real-time factor or throughput, plus
  the exact model name. Task numbers are never compared across tasks, and the Qwen2.5 scoreboard
  stays separate.

### Additional GPU backend: native wgpu (speech first)

Goal: ship F5-Myanmar-TTS v2 on native wgpu first, then extend to text inference.
Keep the existing CUDA text engine alongside it. The 0.8.0 F5 milestone below is
also the first wgpu milestone; a CUDA speech implementation is not a prerequisite.
Browser WebGPU is outside this milestone.

Implementation status (2026-11-12): an experimental fp32 Rust/Burn-wgpu F5 + Vocos
path now generates speech on the RTX 3050 Ti Laptop GPU through Vulkan. Fixed-input
DiT/final-mel/Vocos comparisons against ORT pass, and a same-seed 16-step WAV passes
the reconstructed log-mel correlation check. See `docs/TTS.md` for reproduction.
Release performance, per-layer PyTorch checks, listening evaluation and non-NVIDIA
hardware validation remain open. Vocos weight preparation still uses a one-time
Python repacking step; inference itself uses neither Python nor ORT.

- [ ] Build a CUDA-independent wgpu compute layer: adapter selection and diagnostics,
      capability checks, buffers, uploads, scratch allocation and dispatch. Respect adapter
      binding/buffer limits; begin with portable fp32 correctness and add fp16 where supported.
- [x] Implement the F5 DiT and Vocos operations and weight loading described below. Reuse
      the existing Rust text normalization, reference-audio processing, seeded noise,
      Euler/CFG sampler and inverse FFT where their interfaces permit.
- [ ] Validate individual operations, DiT steps, final mel and waveform against fixed
      PyTorch/ORT fixtures before claiming end-to-end support. Retain ORT during validation.
- [ ] Add explicit speech backend selection, diagnostics and benchmarking. Selecting wgpu
      must actually execute DiT and Vocos through wgpu, report the adapter, and return an
      actionable error if unsupported; do not silently substitute ORT or CPU networks.
- [x] Provide a speech-only build/run path without CUDA tooling or the CUDA text engine.
      CPU audio preprocessing and inverse FFT are acceptable and must be documented.
- [ ] Validate on representative AMD, Intel and Apple GPUs before claiming support for them.
- [ ] After speech acceptance, add GGUF text: separate compute selection from model-family
      dispatch, implement quantized matrix operations and KV caches, and validate
      `Qwen2.5-3B-Instruct-Q4_K_M.gguf` before `Qwen3.5-4B-Q4_K_M.gguf` and DeltaNet.

Acceptance for the first wgpu milestone: a CUDA-independent speech build loads the original
F5/Vocos weights and generates validated audio through wgpu on documented hardware.
Report exact model/checkpoint, backend, adapter, precision, sampler steps, text, audio
length, load/synthesis time, real-time factor and peak GPU memory. Compare against ORT on
the same hardware and inputs where possible. There is no fixed speed target or assumed
performance parity with CUDA.

### Native wgpu text: initial Qwen2/Qwen2.5 backend

An experimental native packed-GGUF implementation now exists in `crates/wgpu`.
The main CLI selects it with `text gen --backend wgpu` in a `--features wgpu` build;
the standalone `tarafer-wgpu` build has no CUDA dependency. CUDA remains the default.
See `docs/WGPU_TEXT.md` for supported shapes, testing and reproducible CUDA comparisons.
The path uses fp32 activations and packed-fp16 KV. It now has block-specialized
Q4_K/Q6_K GEMV, fused residual and gate/up/SwiGLU operations, lossless Q6 scale
expansion and tiled prefill in batches of up to 32 tokens.
On the RTX 3050 Ti Laptop GPU, the latest Qwen2.5-3B-Instruct-Q4_K_M release comparison
measured 69.58 tok/s CUDA versus 29.40 tok/s optimized wgpu (median of three warm
64-token runs). The latest pass added 19% decode throughput over a paired rerun of
the preceding binary (24.70 tok/s), with unchanged checked logits and vocabulary.
Prefill measured 46.00 tok/s. Q6 expansion trades extra weight storage for speed;
resident weight buffers for the baseline are about 1.91 GiB, excluding KV/scratch.
Independent fp32 reference checks pass; strict numerical equivalence to the CUDA
activation-quantized path does not pass at every checked position. Full details and
reproduction commands are in `docs/WGPU_TEXT.md`; no speed or quality parity is assumed.
Qwen3.5/DeltaNet, additional kernel tuning, GPU validation and service integration
remain open; this is not completion of the broader cross-platform text milestone.

## 0.8.0 focus: native F5 engine (no ONNX, no Python)

**Goal:** `tarafer models pull f5-myanmar` fetches the original upstream safetensors and
`tarafer speech say` runs the DiT and vocoder on our own Rust + wgpu engine first. No export step, no Python,
no ONNX Runtime for this model. Until it lands, MMS is the default speech engine so a fresh
install works on the first run.

Why: we control the code and the speed path (resident weights, fused kernels, optional fp16),
and the model installs like every other one. Baseline to beat: F5-Myanmar-TTS v2, 16 steps,
RTF 0.76 on an RTX 3050 Ti 4 GB via ONNX Runtime (22 s text, 24 steps: 0.91).

### Tasks

- [ ] **T0 Default engine:** make `speech say` default to `mms`; keep `--engine f5` opt-in.
- [ ] **T1 Scope:** list every tensor name, shape and dtype in the upstream checkpoint and map it
      to the DiT layers (text embed + ConvNeXt, input/time embed, ~22 AdaLN blocks, rotary
      attention, MLP, final norm) and the Vocos backbone. Write the table to `docs/F5_ENGINE.md`.
- [ ] **T2 Weights:** safetensors loader (fp32/fp16 to our tensor type), registry entry for the
      upstream repo with pinned revision and SHA-256, `Install::Download`.
- [ ] **T3 Reference harness:** dump per-layer activations from PyTorch for a fixed text, noise
      and step, plus the final mel; store as test fixtures outside the repo (not committed).
- [ ] **T4 DiT forward (correct first):** native wgpu implementation; each layer within
      tolerance of PyTorch on the fixtures; end-to-end log-mel correlation >= 0.985 (the current
      ONNX figure) with identical noise.
- [ ] **T5 Vocos network:** ConvNeXt backbone to complex spectrogram, reuse the existing inverse
      FFT; waveform correlation against the ONNX/PyTorch output.
- [ ] **T6 Speed:** tune wgpu GEMM/attention, keep weights and sampler state resident,
      reduce dispatch/readback overhead and add fp16 where supported; measure with `bench speech`.
- [ ] **T7 Cutover:** `--engine f5` uses the native engine; delete the ONNX export script and the
      F5 ONNX path once parity holds; update `docs/TTS.md`, `docs/MODELS.md`.

### Acceptance

- Fixed noise and seed: log-mel correlation vs PyTorch >= 0.985 and listening check on the
  donation and office test texts (author judges quality; not claimed otherwise).
- Report RTF at 16 steps against the ONNX baseline on the same text and machine,
  with the exact model/checkpoint, adapter, backend and precision. Investigate regressions;
  cross-platform availability and validated output are the first milestone.
- Fits 4 GB VRAM at the default settings; fresh machine with no Python can `pull` and `say`.
- Licence stays CC BY-NC 4.0 and is shown at pull time.

### Risks

- Numerical drift in fp16 attention or AdaLN; keep an fp32 fallback until parity is shown.
- The kernel work benefits F5 only; other TTS models need their own engine.

## 0.9.0 focus: own every engine (no ONNX Runtime in the default path)

**Goal:** every shipped task runs on code we own: one Rust op library with wgpu and CUDA backends, one safetensors/GGUF
weight loader, no ONNX Runtime and no Python on the default path. ORT stays only as an opt-in
fallback (`--runtime ort`) for models that have no native port yet, so new models can still ship fast.

Why: control over speed (fusion, fp16/int8, CUDA graphs), setup cost (image setup is 2-4 s per
call today), memory on 4 GB cards, and install experience (`models pull` and run, nothing else).

Depends on 0.8.0: the F5 engine is where the shared ops get written and proven. 0.9.0 generalises
them and ports the other two ORT models.

### Tasks

- [ ] **N0 Shared op library (`taraference-ops`):** extract from the text and F5 engines the
      kernels every model reuses: GEMM (fp16/fp32, optional int8), LayerNorm/RMSNorm/AdaLN,
      softmax + attention (rotary and plain), conv1d/conv2d (direct + im2col), transposed conv,
      depthwise conv, GELU/SiLU/Snake activations, STFT/iSTFT, resize/interpolate. One test
      harness that compares each op to a PyTorch fixture within a stated tolerance. Keep
      wgpu independent of CUDA; share model semantics while allowing backend-specific kernels.
- [ ] **N1 Graph-free model definitions:** each model is a plain Rust struct that loads named
      tensors and runs a hand-written forward pass; no generic graph interpreter.
- [ ] **N2 MMS-TTS Burmese native (VITS):** text encoder (relative-position attention), stochastic
      duration predictor, normalising flow, HiFi-GAN decoder. Load the upstream safetensors
      directly. Parity: waveform/log-mel correlation vs PyTorch with fixed noise; RTF no worse
      than the ORT baseline (0.12-0.14 on RTX 3050 Ti 4 GB).
- [ ] **N3 BiRefNet Lite native:** Swin backbone, decoder with ASPP/deformable conv
      (write the deformable conv kernel), bilateral reference refinement. Parity: mask IoU
      >= 0.99 vs ORT on a fixed image set; setup well under the current 2 s, inference no worse
      than 2.3 s at 1024x1024 on the same machine; fp16 end to end, fits 4 GB.
- [ ] **N4 Warm sessions in `serve`:** lazy-load per task, keep weights resident, add
      `/v1/audio/speech` and an image endpoint so setup cost is paid once.
- [ ] **N5 Memory and speed pass:** fp16 or int8 weights where parity holds, backend-specific dispatch optimization
      (CUDA graphs on CUDA), one shared allocator/arena so several tasks fit in 4 GB.
- [ ] **N6 Remove ORT from the default path:** `ort` crate becomes an optional feature, default
      builds and release archives link no ONNX Runtime DLLs; delete the DirectML/CPU provider
      glue once native CPU fallbacks are acceptable (or document GPU-only for those tasks).
- [ ] **N7 Cross-platform:** native CPU path (SIMD GEMM) for models that must run without a GPU;
      Linux CI build with the CUDA kernels; document supported GPUs (sm_ targets).

### Acceptance

- `cargo tree` for the default build shows no `ort`; release archive has no ORT DLLs.
- Each ported model has a fixture-based parity test in CI (no weights committed; tests skip when
  the model is not pulled).
- `tarafer bench speech|image` reports load, run and RTF/throughput for native vs the ORT
  baseline, same machine, exact model names.
- A fresh Windows machine with no Python: `models pull` then `say`, `remove-bg`, `text gen` all work.

### Risks

- Deformable conv and the VITS flow are the hardest ops; budget the most time for N3.
- fp16 drift per model; keep fp32 reference paths until parity is shown.
- Every port is model-specific; do not claim a general ONNX replacement.
- Scope creep: a model joins this list only if it already passed the "lean by default" rule.

## Milestones

1. **CLI restructure (done):** noun/verb parser, `doctor`, one `--help`, no legacy spellings,
   slimmed `bench`. No change to inference.
2. **Registry (done):** `models list|info|pull|verify|rm`, pinned hashes, licence prompt, image and
   speech migrated. Open: update check. The F5 export step is replaced by the native engine below.
   - **Native F5 engine (0.8.0):** see "0.8.0 focus" above.
   - **Own every engine (0.9.0):** see "0.9.0 focus" above (MMS, BiRefNet, shared ops, no ORT).
3. **Speech-to-text (0.10.0):** `speech transcribe` runs `freococo/myanmar_asr` (Burmese
   Conformer CTC, Apache-2.0 per the upstream model card, 74 MiB, RTF 0.004 on the 4 GB GPU and
   0.006 on CPU; [docs/STT.md](../docs/STT.md)). Still open: `/v1/audio/transcriptions`, an
   independent Burmese accuracy measurement, and an English/multilingual model (Whisper
   tiny/base/small, ONNX exports exist; licence and 4 GB fit to be verified).
4. **TTS completion (0.10.0):** MyanmarTTS (CC0) native port as the commercially usable engine
   (built on the 0.9.0 op library, not ONNX); VoxCPM2 (Apache-2.0) feasibility on 4 GB.
   `bench speech` times the ONNX Runtime engines and, in a `--features wgpu` build, native F5
   (fp16, 10.0-11.2 s against 9.5-10.2 s on ONNX Runtime CUDA for 7.4 s of speech, RTX 3050 Ti
   Laptop GPU, Vulkan; 5-10% slower, not yet faster).
5. **Specialised models (0.11.0+):** add by demand, one at a time, each with a licence check and a
   benchmark. Video is research only until a model is small enough and clearly useful.

## Acceptance checks

- `tarafer --help` lists the nouns above and nothing else; each noun's help fits one screen.
- `models list --commercial` hides non-commercial entries.
- A fresh `models pull` verifies the pinned SHA-256 and shows the licence before download.
- `bench` produces a comparable report for text, speech and image with the exact model named.
- Release archives exclude all weights.

## Out of scope

- Agents, tool calling, retrieval pipelines, long-term memory, plugin systems.
- Serving large language models (target ceiling is the 4B class; the 9B model was dropped from the
  registry as too slow for the reference machine).
- Training or fine-tuning.
- Claiming quality results that have not been checked by listening or by a labelled test set.
