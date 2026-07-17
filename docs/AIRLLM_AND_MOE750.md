# AirLLM inspiration → Tara MoE-400 @ 750 tok/s

Reference: [lyogavin/airllm](https://github.com/lyogavin/airllm)

## What AirLLM optimizes

| Technique | Purpose | Fit for 750 t/s? |
|-----------|---------|------------------|
| **Layer-wise streaming** (load 1 layer from disk → compute → free) | Fit 70B–671B on 4–12 GB VRAM | **No** — disk-bound; destroys single-stream t/s. Our MoE-400 pack is ~0.3 GiB and already fits. |
| **Prefetch** (overlap load + compute) | Hide disk latency | Only if we ever stream experts from host/NVMe |
| **Block-wise weight quant (4/8-bit)** | Smaller load + ~3× when I/O bound | **Yes** — same idea as Q4 pack for **HBM bandwidth** |
| **Sparse MoE touch** (only active experts) | Quality at fixed active budget | **Yes** — our top-k path |

AirLLM’s headline is **memory**, not **accepted tok/s**. Our north star is **single_stream_accepted_tps ≥ 750** on a named GPU with **real top-k routing** (no fixed-expert cheat, no IGNORE_EOS for product claims).

## What we take from AirLLM

1. **Sparse weight access** — only top-k experts per token (packed Q8 columns, device-side index).
2. **Block quant path** — Q8 now; Q4 next so ~98M active fits the 750 BW budget.
3. **No quality shortcuts in product metrics** — real router, natural EOS; IGNORE_EOS is debug-only.

## What we do **not** copy

- Layer streaming from disk on the decode hot path.
- Claiming 750 with `TARAFER_MOE_FIXED` or forced max_new-only benches as the product number.

## Engine path (MoE-400) — status

| Step | Status | Result (3050 Ti) |
|------|--------|------------------:|
| Device router top-k + packed experts + CUDA graphs | **done** | **~455 t/s real routing Q8** (was ~184 with host sync) |
| Q4_0 pack export + load | **done** | ~357 t/s — naive nibble kernel; needs DP4A/opt |
| Optimized Q4 GEMV (or Q4_K) | **next** | Target ~750 if BW-limited at ~0.55 B/param |
| Arch trim (`expert_ff`, top_k=1) | backup | If Q4 opt still short |

**No FIXED cheat for product numbers.** `TARAFER_IGNORE_EOS` is only for under-trained speed benches, not quality claims.

## Run (honest real routing)

```powershell
cd D:\taraference
$env:TARAFER_TOKENIZER_GGUF = "models\tara-sprint-80m-Q8_0.gguf"
# Do NOT set TARAFER_MOE_FIXED
.\target\release\tarafer.exe "D:\Tara_HQ\departments\taraference_750_department\exports\tara-moe-400-a120-q8pack" --prompt "hi" -n 128
# Q4 pack (after DP4A opt should beat Q8):
# .\target\release\tarafer.exe "...\exports\tara-moe-400-a120-q4pack" --prompt "hi" -n 128
```
