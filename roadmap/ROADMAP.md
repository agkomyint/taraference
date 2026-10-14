# Taraference roadmap

> The noun/verb CLI (`text`, `speech`, `image`, `serve`, `models`, `bench`, `doctor`) is the only
> interface; the old flag-style spellings are gone. Registry pinning, licence prompts and
> `speech transcribe` are open.

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

One `models` registry replaces `--download` tags and per-feature download commands. Each entry
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

## Milestones

1. **CLI restructure (done):** noun/verb parser, `doctor`, one `--help`, no legacy spellings,
   slimmed `bench`. No change to inference.
2. **Registry (0.8.0):** `models list|pull|rm|info`, pinned hashes, licence column. Migrate image
   and tts model setup into it.
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
- Serving large language models (target ceiling is roughly the 9B class that already runs on the
  reference machine).
- Training or fine-tuning.
- Claiming quality results that have not been checked by listening or by a labelled test set.
