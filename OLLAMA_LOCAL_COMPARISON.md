# Local Taraference vs Ollama comparison

Date: 2026-07-16

## Hardware and runtimes

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GiB, `sm_86`
- Taraference: v0.5.2 release build
- Ollama: 0.30.7
- Mode: single-stream, deterministic greedy, five-turn chat
- Context: 5000 tokens
- Maximum reply: 128 tokens

Ollama imported the exact same GGUF files used by Taraference. It did not use a
different Ollama registry model:

- `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
- `Qwen2.5-3B-Instruct-Q4_K_M.gguf`

## Results

| Model and condition | Taraference | Ollama | Relative result |
|---|---:|---:|---:|
| 3B, cool/cold representative run | 70.117 tok/s | 68.960 tok/s | Taraference +1.7% |
| 3B, controlled hot run | 49.040 tok/s | 43.813 tok/s | Taraference +11.9% |
| 0.5B, clean-residency exploratory run | 161.640 tok/s | 225.761 tok/s | Ollama +39.7% |

The 3B cool/cold row uses each runtime's first run with no competing model left
resident. The hot row was deliberately repeated after the laptop reached its
87 C target, with the other runtime unloaded before every run. The corrected
0.5B row started Ollama at 69 C and Taraference at 73 C. Before both runs,
`ollama ps` was empty and NVIDIA reported zero resident compute memory.

During the Ollama 0.5B run, `ollama ps` showed only
`tara-qwen25-05b-q4km`, with a 536 MB model allocation and 603 MiB total GPU
memory. Taraference peaked at 591 MiB. The Ollama 3B model was not resident.

## Quality and stopping

Both 3B runtimes produced coherent answers. Taraference generated 355 accepted
tokens and Ollama generated 335 in the representative runs. Both stopped four
turns normally and hit the 128-token limit on the final summary.

The 0.5B model was less reliable in both runtimes. Taraference hit the token cap
twice and repeated project ideas. Ollama was faster, but identified itself as
Claude and also repeated content. Its prompt templating/output was therefore not
text-identical to Taraference even though the weights and quantization matched.

## Invalid runs excluded

Two Taraference 3B runs at 55.740 and 46.060 tok/s were excluded because Ollama
still held its 1.9 GB model in VRAM. Total GPU use reached 3920/4096 MiB and the
laptop reached 88 C. These are residency-contention measurements, not a fair
runtime comparison.

## Interpretation

Taraference's 3B quantized decode path is competitive with Ollama on this GPU.
The 0.5B result exposes a different bottleneck: with only 0.36 GiB of weights,
fixed kernel-launch, synchronization, and attention overhead dominate. Improving
small-model speed requires fewer launches or a more persistent/fused decode path;
the result must not be used as a 3B scoreboard claim.

## Reproduction

```powershell
ollama create tara-qwen25-05b-q4km -f benchmarks\ollama-qwen25-05b.Modelfile
ollama create tara-qwen25-3b-q4km -f benchmarks\ollama-qwen25-3b.Modelfile

.\target\release\tarafer.exe models\Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile
python benchmarks\ollama_chat_bench.py tara-qwen25-3b-q4km `
  --label Qwen2.5-3B-Instruct-Q4_K_M.gguf
```
