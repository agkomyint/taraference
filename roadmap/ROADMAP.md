# Taraference roadmap

> The noun/verb CLI (`text`, `speech`, `image`, `serve`, `models`, `bench`, `doctor`) is the only
> interface; the old flag-style spellings are gone. The model registry (pinned revision + SHA-256, licence prompt,
> `list|info|pull|verify|rm`) covers text, embeddings, speech and image; see
> [docs/MODELS.md](../docs/MODELS.md). `speech transcribe` is open.

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
| Speech-to-text | not started | ONNX Runtime (candidates below) |
| Video (e.g. small segmentation/captioning) | research only | undecided |
| More specialised models (OCR, translation, classification) | backlog | by demand |

## CLI: one noun per task, verbs underneath

```
tarafer text   gen <prompt|->  [-m MODEL] [-n N] [--temperature …] [--json]
tarafer text   chat            [-m MODEL]            # thin REPL; may be dropped (see below)
tarafer text   embed <text|->  [-m MODEL]
tarafer speech say <text|->    -o out.wav [--engine f5|mms] [--lexicon FILE]
tarafer speech transcribe <audio>  [-m MODEL] [--lang my|en]     # new
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

- **GGUF engine (custom CUDA):** text and embeddings. Unchanged; this is where the speed work
  and the Qwen2.5-3B scoreboard live.
- **ONNX Runtime:** speech, image, and future specialised models, with CUDA → DirectML → CPU
  fallback. One shared session/provider loader (currently duplicated across image and tts).
- **Benchmarks:** every task reports load time, run time and a real-time factor or throughput, plus
  the exact model name. Task numbers are never compared across tasks, and the Qwen2.5 scoreboard
  stays separate.

## 0.8.0 focus: native F5 engine (no ONNX, no Python)

**Goal:** `tarafer models pull f5-myanmar` fetches the original upstream safetensors and
`tarafer speech say` runs the DiT and vocoder on our own CUDA code. No export step, no Python,
no ONNX Runtime for this model. Until it lands, MMS is the default speech engine so a fresh
install works on the first run.

Why: we control the code and the speed path (fp16, fused kernels, CUDA graph over the step loop),
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
- [ ] **T4 DiT forward (correct first):** straightforward implementation; each layer within
      tolerance of PyTorch on the fixtures; end-to-end log-mel correlation >= 0.985 (the current
      ONNX figure) with identical noise.
- [ ] **T5 Vocos network:** ConvNeXt backbone to complex spectrogram, reuse the existing inverse
      FFT; waveform correlation against the ONNX/PyTorch output.
- [ ] **T6 Speed:** reuse the GEMM/attention kernels, fp16 weights, CUDA graph over the sampler
      step loop, then measure with `bench speech`.
- [ ] **T7 Cutover:** `--engine f5` uses the native engine; delete the ONNX export script and the
      F5 ONNX path once parity holds; update `docs/TTS.md`, `docs/MODELS.md`.

### Acceptance

- Fixed noise and seed: log-mel correlation vs PyTorch >= 0.985 and listening check on the
  donation and office test texts (author judges quality; not claimed otherwise).
- RTF at 16 steps no worse than the ONNX baseline (0.76), measured on the same text and machine,
  with the exact model name reported.
- Fits 4 GB VRAM at the default settings; fresh machine with no Python can `pull` and `say`.
- Licence stays CC BY-NC 4.0 and is shown at pull time.

### Risks

- Numerical drift in fp16 attention or AdaLN; keep an fp32 fallback until parity is shown.
- The kernel work benefits F5 only; other TTS models need their own engine.

## Milestones

1. **CLI restructure (done):** noun/verb parser, `doctor`, one `--help`, no legacy spellings,
   slimmed `bench`. No change to inference.
2. **Registry (done):** `models list|info|pull|verify|rm`, pinned hashes, licence prompt, image and
   speech migrated. Open: update check. The F5 export step is replaced by the native engine below.
   - **Native F5 engine (0.8.0):** see "0.8.0 focus" above.
3. **Speech-to-text (0.9.0):** pick a model by benchmark, add `speech transcribe`
   and `/v1/audio/transcriptions`. Candidates, all to be verified for licence, Burmese accuracy
   and 4 GB fit: Whisper tiny/base/small (multilingual, ONNX exports exist), and
   `freococo/myanmar_asr` (Burmese-specific, reported Apache-2.0).
4. **TTS completion (0.9.0):** MyanmarTTS (CC0) ONNX port as the commercially usable engine;
   VoxCPM2 (Apache-2.0) feasibility on 4 GB; `/v1/audio/speech`; `bench speech`.
5. **Specialised models (0.10.0+):** add by demand, one at a time, each with a licence check and a
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
