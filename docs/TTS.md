# Burmese text-to-speech in taraference

`tarafer speech` synthesises Burmese speech locally on ONNX Runtime (CUDA, DirectML or CPU),
with an opt-in native wgpu backend for F5-Myanmar-TTS v2.
No Python is needed at run time. This document records which open models exist, which are
integrated, what was measured, and what is still open.

> **Licences matter.** F5-Myanmar-TTS and MMS-TTS are **CC BY-NC 4.0 (non-commercial only)**.
> MyanmarTTS is **CC0** and VoxCPM2 is **Apache-2.0**, so they are the options for commercial use.
> No model weights are shipped in this repository or in releases.

## Status at a glance

| Engine | Licence | Params / rate | In `tarafer`? | Measured RTF* | Needs GPU? |
|---|---|---|---|---|---|
| **F5-Myanmar-TTS v2** (`freococo/F5-Myanmar-TTS`) | CC BY-NC 4.0 | 337M DiT, 24 kHz | Yes (`--engine f5`, default) | 0.75–0.91 (16–24 steps) | Practically yes (CPU ≈ 147 s for 6 s) |
| **MMS-TTS Burmese** (`facebook/mms-tts-mya`) | CC BY-NC 4.0 | VITS, 16 kHz, 114 MB ONNX | Yes (`--engine mms`) | 0.13 (GPU), 0.14 (CPU) | No |
| **MyanmarTTS** (`freococo/MyanmarTTS`) | **CC0** | 31M StableTTS + Vocos, 44.1 kHz, 63 MB fp16 | Not yet. Verified in PyTorch only | 0.11–0.13 (PyTorch, CUDA) | No expected |
| **VoxCPM2** (`OpenBMB/VoxCPM`) | **Apache-2.0** | 2B, 48 kHz | Not yet. Not tested | Not measured | ~8 GB VRAM per vendor |

\* RTF = synthesis time / audio time (lower is faster; < 1 is faster than real time). Measured on an
RTX 3050 Ti Laptop (4 GB), Windows, ONNX Runtime CUDA, same ~22 s Burmese donation text, model load
excluded. Quality has **not** been scored by the author of this document; judge by listening.

## Usage

```powershell
tarafer speech status
tarafer speech say "မင်္ဂလာပါ" -o hello.wav                          # F5, best quality so far
tarafer speech say --file story.txt -o story.wav --steps 16          # F5, faster
tarafer speech say --file story.txt --engine mms -o story.wav        # MMS, fastest, runs on CPU
tarafer speech say "…" -o cloned.wav --ref-audio v.wav --ref-text "exact transcript of v.wav"   # F5 only
```

### Text cleanup (all engines)

The models only read Burmese script, so `say` first normalises the text:

- Digits (`7`, `၅`, `1,250`, `3.5`) and `%` are read as Burmese number words (`7နာရီ` → `ခုနစ်နာရီ`).
- Emoji and symbols are removed; the count is reported.
- Latin words are not speakable. Give them Burmese spellings with `--lexicon FILE`
  (`Client=ကလိုင်းယင့်`, see [tts-lexicon.example.txt](tts-lexicon.example.txt)). Unlisted words are
  skipped and named in a warning, never dropped silently.
- `--no-normalize` sends the text through unchanged.

Number reading and the sample spellings follow common usage but have not been checked by a native speaker.

| Option | Default | Notes |
|---|---|---|
| `--lexicon` | none | `English=Burmese` word list |
| `--no-normalize` | off | skip cleanup |
| `--engine` | `f5` | `f5` or `mms` |
| `--model-dir` | per engine | `models/tts/f5-myanmar` or `models/tts/mms-mya` |
| `--device` | `auto` | `auto` (ORT CUDA → DirectML → CPU), `cuda`, `directml`, `cpu`, or native `wgpu` (F5 only; build feature required) |
| `--steps` | 32 | F5 only. 12–16 is ~2× faster with a small quality loss |
| `--cfg` | 2.0 | F5 only. Guidance strength |
| `--speed` | 1.0 | Speaking-rate multiplier |
| `--seed` | 0 | F5 only. MMS is stochastic and ignores it |

Text is split at `။` and newlines; chunks are synthesised separately and joined with short pauses.

## Setup

### Native F5 on wgpu (experimental)

F5's DiT, attention, sampler updates and the Vocos network run through Burn 0.18's
wgpu/WGSL backend. Text cleanup, reference-mel extraction and inverse FFT run in Rust
on the CPU. Selecting `wgpu` does not create ONNX sessions or silently fall back to ORT.
CUDA text inference remains available in the main executable.

The native backend accepts the original `freococo/F5-Myanmar-TTS` v2 checkpoint
(1024 hidden width, 22 DiT blocks) and `charactr/vocos-mel-24khz` weights. Prepare
the files locally after obtaining the checkpoints under their upstream licences:

```powershell
pip install torch safetensors
python tools/tts/prepare_f5_wgpu.py --f5-dir PATH/TO/F5-CHECKPOINT --vocos PATH/TO/pytorch_model.bin
```

The F5 directory must contain `model.safetensors`, `vocab.txt` and `ref_sample.wav`.
The preparation script copies these unchanged and repacks Vocos' PyTorch state dict
as `vocos.safetensors`. Python is needed for this one-time repacking only; inference
needs neither Python nor ONNX files. `models pull f5-myanmar` does not yet install
this native layout automatically.

Build and run the standalone speech executable without the CUDA text engine or ORT:

```powershell
cargo run -p taraference-tts --release --no-default-features --features wgpu --bin tarafer-speech -- "မင်္ဂလာပါ။" --steps 16 -o speech-wgpu.wav
```

Or enable it alongside the existing engines in the main CLI:

```powershell
cargo run -p taraference --release --features wgpu -- speech say "မင်္ဂလာပါ။" --engine f5 --device wgpu --steps 16 -o speech-wgpu.wav
```

The backend prints the adapter and graphics API. The DiT runs in half precision on Vulkan
adapters and in fp32 elsewhere (`TARAFER_WGPU_PRECISION=fp16|fp32` overrides this; the text
encoder and Vocos stay in fp32), and `TARAFER_WGPU_TRACE=1` prints per-step times. Compiled
kernels and Burn's autotune results are kept in the user cache directory
(`taraference/wgpu-kernels`, `taraference/wgpu-autotune`); the first run for a new class of
sentence length is tuned once and can take tens of seconds longer. Measured speed is in
[STT.md](STT.md). The backend caps each chunk at 2048 total mel frames
including reference audio. Shorten the reference or split the input if this limit is
exceeded. Only the exact F5/Vocos architecture above is supported; MMS remains on ORT.
AMD, Intel and Apple hardware require their own validation before support is claimed.
Use release builds for performance measurements; first-use shader compilation affects
latency. No CUDA performance parity is assumed.

Reproduce fixed-input parity checks against the existing ONNX exports:

```powershell
python tools/tts/f5_wgpu_fixtures.py --frames 160 --steps 4
cargo run -p taraference-tts --release --no-default-features --features wgpu --example wgpu_parity -- models/tts/f5-myanmar target/f5-wgpu-fixtures
cargo test -p taraference-tts --no-default-features --features wgpu
```

The parity executable checks DiT velocity, sampled final mel, and real/imaginary Vocos
spectrograms. It fails on nonfinite outputs, shape mismatches, cosine similarity below
0.985, relative RMSE above 5% for DiT/final mel, or above 1% for Vocos. The 160-frame
case also exercises the attention query-chunk boundary. These numeric checks do not
replace a Burmese listening evaluation.

Initial validation (2026-11-12): Windows, RTX 3050 Ti Laptop GPU (4 GB), Vulkan,
wgpu fp32, F5-Myanmar-TTS v2 checkpoint revision
`f6365609e33412c833382c3481653e0b0ca0c43b`, Vocos revision
`0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`:

| Check | Result |
|---|---|
| 160-frame DiT velocity vs ORT | cosine 0.9999986; relative RMSE 0.0016665 |
| Four-step sampled final mel vs ORT | cosine 0.9999966; relative RMSE 0.0025961 |
| Vocos real / imaginary vs ORT | relative RMSE 0.0000069 / 0.0000068 |
| `မင်္ဂလာပါ။`, 16 steps, CFG 2, speed 1, seed 42, bundled reference | 37,632 samples at 24 kHz (1.568 s), non-silent |
| Reconstructed log-mel correlation vs same-seed ORT WAV | 0.9972936 (threshold 0.985) |
| Final debug run after vectorized weight loading | 5.814 s load; 23.241 s synthesis; RTF 14.822 |
| Sampled total GPU memory during that run (1 s polling) | 1,544 MiB maximum; includes other GPU users |

Debug timings include shader compilation and are correctness results, not release
performance benchmarks. The first run before the loading improvement took 37.345 s
synthesis (RTF 23.817). Waveform correlation was 0.7782794; numeric mel agreement does not establish
perceptual equivalence. No listening-quality or non-NVIDIA hardware result is claimed.
The reference WAV was generated with the existing ORT CPU engine using identical inputs.
All eight tests in the wgpu-only build and seven tests in the default ORT build pass;
the main CLI checks with `--features wgpu`. The standalone wgpu dependency tree contains
neither ORT nor CUDA backends.
To compare regenerated WAVs:

```powershell
python tools/tts/compare_tts_wav.py target/f5-ort-reference.wav target/f5-wgpu.wav
```

### F5-Myanmar-TTS

Exported once from the original PyTorch weights (Python, `torch`, a CUDA GPU for the default fp16
export, ~3 GB temporary disk):

```powershell
pip install f5-myanmar-tts onnx onnxruntime
python tools/tts/export_f5_myanmar.py            # writes models/tts/f5-myanmar/ ; --fp32 for CPU export
```

| File | Size | Purpose |
|---|---|---|
| `dit.onnx` | ~684 MB | DiT denoiser with classifier-free guidance folded in (fp16 weights) |
| `vocos.onnx` | ~54 MB | mel → complex spectrogram (ISTFT is done in Rust) |
| `vocab.txt` | 11 KB | character vocabulary |
| `ref_sample.wav` | 215 KB | default reference voice, 24 kHz |

The export checks the ONNX graph against PyTorch at a different sequence length and fails on mismatch.

### MMS-TTS Burmese

A ready-made ONNX build is published in `willwade/mms-tts-multilingual-models-onnx` under `mya/`.
Install with `tarafer models pull mms-mya` (files go to `models/tts/mms-mya/`, SHA-256 verified).
Inputs: blank-interleaved character ids, `x_length`, `noise_scale` 0.667,
`length_scale` 1/speed, `noise_scale_w` 0.8. Output: 16 kHz waveform.

## Findings

### Candidates researched

| Model | Source | Licence | Verdict |
|---|---|---|---|
| F5-Myanmar-TTS v2 | [Hugging Face](https://huggingface.co/freococo/F5-Myanmar-TTS) | CC BY-NC 4.0 | Integrated. 794 h of Burmese training data (author's figure). Highest fidelity of the integrated engines on paper; slowest. |
| MMS-TTS Burmese | [Hugging Face](https://huggingface.co/facebook/mms-tts-mya) | CC BY-NC 4.0 | Integrated. Very fast, single pass, runs on CPU. Expected to sound more robotic. |
| MyanmarTTS | [Hugging Face](https://huggingface.co/freococo/MyanmarTTS) | **CC0** | Works in PyTorch (RTF 0.11–0.13). Needs an ONNX export of text encoder, duration predictor, flow decoder and Vocos before it can run in `tarafer`. |
| VoxCPM2 | [GitHub](https://github.com/OpenBMB/VoxCPM) | **Apache-2.0** | Burmese is one of its 30 listed languages (vendor-reported 1.42% CER). 2B parameters, ~8 GB VRAM, so it likely does not fit a 4 GB GPU without quantisation. Community ONNX/GGUF builds exist ([zinooko ONNX](https://huggingface.co/zinooko/VoxCPM2-Burmese-ONNX-Final), no licence stated). Untested here. |
| hpbyte/myanmar-tts | [GitHub](https://github.com/hpbyte/myanmar-tts) | CC BY-NC-SA 4.0 | Tacotron thesis project. Not pursued. |
| burmeseitman/myanmar-tts-pipeline | [GitHub](https://github.com/burmeseitman/myanmar-tts-pipeline) | Unclear | Uses Microsoft cloud voices, so it is not a local model. Out of scope. |

No Burmese voices were found for Piper, Kokoro, Coqui, XTTS or Orpheus. Qwen3-TTS, Chatterbox
Multilingual and Fish Speech were not confirmed to support Burmese.

Training corpora, if fine-tuning is ever wanted:
[Open-Burmese-Speech](https://github.com/thantzinphyo-gg/Open-Burmese-Speech) and OpenSLR SLR80.

### F5-Myanmar-TTS measurements (6 s utterance unless noted)

| Steps | Synth time | RTF |
|---|---|---|
| 32 | 9.8 s (includes warm-up) | ~1.5 |
| 16 | 4.6 s | 0.76 |
| 12 | 3.6 s | 0.59 |
| 24, 22.9 s text in 2 chunks | 20.7 s | 0.91 |

Model load is ~3.7 s. With identical noise, the Rust pipeline and the PyTorch reference agree at
0.985 log-mel correlation. `TARAFER_TTS_NOISE=<file.f32>` (frames × 100 little-endian f32) feeds fixed
noise for parity checks.

### Same-text comparison (22 s Burmese donation request, CUDA)

| Engine | Load | Synth | Audio | RTF |
|---|---|---|---|---|
| F5-Myanmar-TTS, 24 steps | 3.7 s | 20.7 s | 22.9 s | 0.91 |
| MMS-TTS, GPU | 1.1 s | 2.7 s | 21.7 s | 0.13 |
| MMS-TTS, CPU | 0.8 s | 3.1 s | 21.2 s | 0.14 |
| MyanmarTTS (PyTorch), 12 steps | 2.1 s | 2.3 s | 18.4 s | 0.13 |
| MyanmarTTS (PyTorch), 24 steps | n/a | 2.1 s | 18.4 s | 0.11 |

MyanmarTTS timings are from the original Python package, not from `tarafer`, and include a patched
audio loader. They should not be treated as `tarafer` numbers until an ONNX port exists. These are
TTS real-time factors, not LLM token rates, and are not comparable to the Qwen2.5 scoreboard.

## Known limits

- Burmese only. F5's text frontend (cleanup, syllable-count duration estimate) is Burmese-specific.
- F5 has no duration predictor; unusual text can come out slightly long or short (use `--speed`).
- F5 is capped at 8000 mel frames (~5.3 min) per chunk; long text is chunked.
- Reference-voice cloning needs an exact transcript of the reference audio and is untested.
- DirectML is untested; CPU works but F5 is impractically slow on CPU.
- The F5 export needs Python today. Hosting the exported ONNX for a `tarafer models pull`
  command would remove that, but publishing the files is a decision for the repo owner.

## Next steps

1. Listen to the four engines' output on the same text and pick a default.
2. Port MyanmarTTS (CC0) to ONNX and add `--engine myanmartts`, giving a commercially usable engine.
3. Test VoxCPM2 feasibility on a 4 GB GPU (community GGUF Q8_0 or ONNX) for an Apache-2.0 option.
4. Done: `tarafer bench speech` prints load, first run, warm run and RTF per engine ([STT.md](STT.md)).
5. Host or script the F5 export so users do not need Python (`models pull f5-myanmar` is manual today).

## Verification

`cargo test -p taraference-tts` covers the text frontend, chunking, mel and ISTFT helpers.
