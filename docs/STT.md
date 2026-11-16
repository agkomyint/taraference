# Speech-to-text (Burmese)

`tarafer speech transcribe` turns Burmese speech into text, fully offline.

| | |
|---|---|
| Model | [`freococo/myanmar_asr`](https://huggingface.co/freococo/myanmar_asr) at `e40d48e3`, Apache-2.0 |
| Architecture | 12-layer Conformer encoder + CTC head, 23.8M parameters, 2,566 Burmese syllables |
| Runtime | ONNX Runtime (CUDA, DirectML or CPU), fp32, 74 MiB |
| Input | WAV at any sample rate, mono or stereo; other formats through `ffmpeg` on PATH |
| Decoding | greedy CTC; audio over 20 s is cut at quiet points |

## Setup

Upstream ships PyTorch weights only, so the ONNX file is exported on your machine (weights are
never hosted here):

```
pip install torch safetensors onnx onnxruntime huggingface_hub
python tools/stt/export_myanmar_asr.py        # writes models/stt/myanmar-asr/{model.onnx,tokens.txt}
```

The script pins the upstream revision and the SHA-256 of every input, folds the feature
normalisation into the graph, and checks the ONNX output against PyTorch before finishing.

## Use

```
tarafer speech transcribe talk.wav                 # text on stdout, timings on stderr
tarafer speech transcribe talk.mp3 -o talk.txt     # needs ffmpeg
tarafer speech transcribe talk.wav --syllables     # space-separated syllables
tarafer speech transcribe talk.wav --device cpu
tarafer speech status
```

## Benchmark

`tarafer bench speech [--device auto|cuda|directml|cpu] [--runs N] [--steps N] [--audio FILE]`
times every installed speech engine: load, first run, best warm run and RTF (run time divided by
audio time). Without `--audio`, speech-to-text is timed on the benchmark sentence as spoken by an
installed TTS engine, and the transcript is printed so the round trip can be read.

Measured 2026-11-16 on an RTX 3050 Ti Laptop GPU (4 GB), Windows 11, ONNX Runtime CUDA, one 7-8 s
sentence, best of 3 warm runs:

| task | model | load s | first s | warm s | RTF |
|---|---|---|---|---|---|
| tts | MMS-TTS Burmese (`facebook/mms-tts-mya`, VITS ONNX) | 1.10 | 1.26 | 0.904 | 0.111 |
| tts | F5-Myanmar-TTS v2 (ONNX, 32 steps) | 4.59 | 9.49 | 9.280 | 1.246 |
| stt | `freococo/myanmar_asr` (Conformer CTC, fp32 ONNX) | 0.74 | 0.09 | 0.030 | 0.004 |

A build with `--features wgpu` adds the native engine as a fourth row. Same machine and sentence,
Vulkan, half precision, kernel and autotune caches warm, two consecutive runs (2026-11-16; the
laptop GPU was at 77-78 C after each, so the second run is slower for both engines):

| task | model | load s | first s | warm s | RTF |
|---|---|---|---|---|---|
| tts | F5-Myanmar-TTS v2 (ONNX Runtime CUDA, 32 steps) | 6.38 | 9.68 | 9.493 | 1.275 |
| tts | F5-Myanmar-TTS v2 (native wgpu fp16, 32 steps) | 6.40 | 29.49 | 9.961 | 1.338 |
| tts | F5-Myanmar-TTS v2 (ONNX Runtime CUDA, 32 steps) | 6.05 | 10.07 | 10.193 | 1.369 |
| tts | F5-Myanmar-TTS v2 (native wgpu fp16, 32 steps) | 5.08 | 11.15 | 11.214 | 1.506 |

The native engine is 5-10% slower than ONNX Runtime here (it was 1.3-1.4x slower before the
weights were stored pre-transposed, attention ran unchunked and Burn's autotune was enabled), so
ONNX Runtime stays the default. The first native run on a machine compiles its GPU kernels, and
the first run for each new class of sentence length measures the matmul and reduction variants
once (tens of seconds up to about 150 s; the 29.49 s first run above includes part of that).
Both results are kept under the user cache directory (`taraference/wgpu-kernels` and
`taraference/wgpu-autotune`). Native fp32 took 38 s for the same sentence before these changes
and has not been re-measured.

On the CPU alone the same speech-to-text run took 0.043 s (RTF 0.006), so no GPU is needed.

## Accuracy and limits

- Upstream reports a 9.42% syllable error rate on its own held-out set; this repo has not
  measured accuracy on an independent test set.
- The Rust front end reproduces the PyTorch reference transcript exactly on the four clips
  tried (16 kHz, 44.1 kHz stereo, 24 kHz, and MMS-TTS output).
- Burmese only. English and mixed speech need a second model (Whisper is the candidate).
- Output is plain syllables as recognised: no word spacing and little punctuation.
- No timestamps, no streaming, and no `/v1/audio/transcriptions` endpoint yet.
