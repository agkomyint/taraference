# Qwen3.5-4B `q35fast` A/B variant

**Model:** `Qwen3.5-4B-Q4_K_M.gguf`  
**GPU:** RTX 3050 Ti Laptop 4GB (`sm_86`)  
**Goal:** ~55 decode tok/s single-stream on this laptop  

## How to run A/B

```powershell
# Baseline (default) — Q5_K → Q8_0 hot path, full quality path
.\target\release\tarafer.exe models\Qwen3.5-4B-Q4_K_M.gguf --profile

# q35fast variant — Q5_K → Q4_K hot path (less weight traffic + DP4A)
$env:TARAFER_Q35_FAST = "1"
.\target\release\tarafer.exe models\Qwen3.5-4B-Q4_K_M.gguf --profile
Remove-Item Env:TARAFER_Q35_FAST
```

Aliases: `TARAFER_VARIANT=q35fast` (also accepts `fast` / `q4`).

Default binary is **unchanged** without the env flag.

## What changed

| Piece | Default | `q35fast` |
|-------|---------|-----------|
| Hot tensors (`attn_qkv`, `ssm_out` ×24) | Q5_K → **Q8_0** | Q5_K → **Q4_K** |
| Bytes for those tensors | ~0.80 GiB | ~0.42 GiB |
| GDN fused GEMV | `gemv_hybrid_gdn_4way` (Q8+Q4) | `gemv_q4_gdn_4way` (Q4+Q4+Q8) |
| FFN / flash / CUDA graph | same | same |

Tradeoff: requant Q5→Q4 is slightly lossier than Q5→Q8. Multi-turn chat remained coherent in measured runs (self-ID as Qwen3.5, bullets, summaries).

## Measured (2026-07-24, this laptop)

| Run | Overall decode | First | Last | Drop | Peak VRAM |
|-----|----------------|-------|------|------|-----------|
| Baseline multi-turn | **46.5 t/s** | 48.6 | 44.9 | 8% | ~96% |
| q35fast multi-turn (run1) | **52.5 t/s** | 57.0 | 50.3 | 12% | ~89% |
| q35fast multi-turn (run2) | **55.4 t/s** | 58.4 | 52.6 | 10% | ~89% |
| q35fast multi-turn (run3) | **53.0 t/s** | 57.5 | 50.8 | 12% | ~89% |
| q35fast short cold `-n 64` | **59.9 t/s** | — | — | — | — |

**~55 overall is reachable** on a good thermal window; sustained multi-turn lands ~52–55 depending on clocks/temp. First-turn and short-prompt decode sit comfortably above 55.

vs baseline: about **+13% to +19%** overall decode, **~7% less peak VRAM**.

## Why this works

Layer timing showed FFN ~0.35 ms and GDN mixer ~0.37 ms per layer. GDN `attn_qkv` (8192×2560) and `ssm_out` (4096×2560) were Q5_K in the GGUF. Expanding them to Q8 for DP4A **doubled** that traffic. Requantizing to Q4 keeps DP4A kernels and cuts that bandwidth roughly in half, which was the remaining wall on a 192 GB/s laptop GPU.
