# Burmese text-to-speech in taraference

`tarafer tts` synthesises Burmese speech locally on ONNX Runtime (CUDA, DirectML or CPU).
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
tarafer tts status
tarafer tts say "မင်္ဂလာပါ" -o hello.wav                          # F5, best quality so far
tarafer tts say --file story.txt -o story.wav --steps 16          # F5, faster
tarafer tts say --file story.txt --engine mms -o story.wav        # MMS, fastest, runs on CPU
tarafer tts say "…" -o cloned.wav --ref-audio v.wav --ref-text "exact transcript of v.wav"   # F5 only
```

| Option | Default | Notes |
|---|---|---|
| `--engine` | `f5` | `f5` or `mms` |
| `--model-dir` | per engine | `models/tts/f5-myanmar` or `models/tts/mms-mya` |
| `--device` | `auto` | `auto` (CUDA → DirectML → CPU), `cuda`, `directml`, `cpu` |
| `--steps` | 32 | F5 only. 12–16 is ~2× faster with a small quality loss |
| `--cfg` | 2.0 | F5 only. Guidance strength |
| `--speed` | 1.0 | Speaking-rate multiplier |
| `--seed` | 0 | F5 only. MMS is stochastic and ignores it |

Text is split at `။` and newlines; chunks are synthesised separately and joined with short pauses.

## Setup

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
Put `model.onnx` (114 MB) and `tokens.txt` in `models/tts/mms-mya/`. (There is no download command
yet.) Inputs: blank-interleaved character ids, `x_length`, `noise_scale` 0.667,
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
- The F5 export needs Python today. Hosting the exported ONNX for a `tarafer tts model download`
  command would remove that, but publishing the files is a decision for the repo owner.

## Next steps

1. Listen to the four engines' output on the same text and pick a default.
2. Port MyanmarTTS (CC0) to ONNX and add `--engine myanmartts`, giving a commercially usable engine.
3. Test VoxCPM2 feasibility on a 4 GB GPU (community GGUF Q8_0 or ONNX) for an Apache-2.0 option.
4. Add `tarafer tts bench` to print the comparison table above automatically.
5. Add a model download command so users do not need Python.

## Verification

`cargo test -p taraference-tts` covers the text frontend, chunking, mel and ISTFT helpers.
