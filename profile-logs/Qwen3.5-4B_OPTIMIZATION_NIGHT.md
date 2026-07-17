# Qwen3.5-4B overnight optimization report

**Model:** `Qwen3.5-4B-Q4_K_M.gguf`  
**GPU:** RTX 3050 Ti Laptop 4GB (`sm_86`)  
**Target:** >50 decode tok/s single-stream  

## Results (this machine)

| Metric | Before (broken/hot) | After (best) |
|--------|---------------------|--------------|
| Multi-turn overall | **3.7–4.0 t/s** | **~28 t/s** |
| Multi-turn first turn | 3.7–9.6 t/s | **~37 t/s** |
| Short cold decode | ~4–16 t/s | **~43.5 t/s** (best 43.7) |
| Quality | gibberish when GEMV broken | Coherent multi-turn chat |
| Sanity Qwen2.5-3B | — | **~69 t/s** (same binary; GPU can go fast) |

**50 t/s was not reached** on this 4GB laptop for full hybrid Qwen3.5-4B.

### Ceiling probes (not production quality)

| Mode | Decode t/s | Meaning |
|------|------------|---------|
| Full hybrid (default) | **~43.5** | Real model |
| `TARAFER_SKIP_LINEAR=1` | **~78** | Skip GDN mixer; FFN-only on linear layers |
| `TARAFER_IDENTITY=1` | **~392** | No layer math (launch/overhead only) |

So **GDN weight traffic is the remaining wall**. Hitting 50 needs ~15% less time than 43.5 → either faster GDN GEMV (Marlin/tensor-core Q4/Q8) or a GPU with more sustained BW/power than a 3050 Ti laptop.

Reference: llama.cpp on the same host/model was ~42 gen t/s. We are at/above that band.

## What changed (code)

1. **FFN footgun fix:** stop re-quantizing activations once per output column (was 9k× redundant quantize on gate+up).  
2. **Fused GDN decode:** `gdn_conv_qkvl2` + `gdn_delta_gated` (+ d128 specialized) cut mixer launches.  
3. **GDN 4-way GEMV:** fixed host/kernel ABI (Q8 activations), smem quantize path.  
4. **Q6 FFN down:** cooperative 8-warp compact path.  
5. **Hybrid VRAM:** default `max_seq` 1024, active vocab shortlist 49 152 (+ EOS column).  
6. **CUDA graphs** for single-token decode (including hybrid state save/restore).  
7. **Cached env flags** on the layer hot path.

### Knobs

| Env | Effect |
|-----|--------|
| `TARAFER_FULL_VOCAB=1` | Full 248k head (slower, better rare tokens) |
| `TARAFER_LONG_CTX=1` | Keep large `--ctx` on hybrid |
| `TARAFER_VOCAB_LIMIT=N` | Override active vocab |
| `TARAFER_GEMV_WARPS=N` | Quantized GEMV warps (default 16 on Ampere) |
| `TARAFER_GDN_LEGACY=1` | Unfused GDN mixer |
| `TARAFER_GDN_NO_4WAY=1` | 4× separate in-proj GEMVs |
| `TARAFER_Q5K_NATIVE=1` | Keep Q5_K (less VRAM, slower GEMV) |

## Physics ceiling (3050 Ti 4GB)

Rough weight traffic per token ≈ FFN (~1.5 GiB) + GDN (~1 GiB) + rest.  
Even at strong fraction of 192 GB/s peak, **~45–55 t/s** is the optimistic band; laptop thermal/power (often ~20 W, clocks collapse) makes **sustained 50 unrealistic** without external cooling / higher power limit.

## How to re-bench

```powershell
# Short cold decode
.\target\release\tarafer.exe models\Qwen3.5-4B-Q4_K_M.gguf -n 128 --prompt "Write a short hello."

# Multi-turn profile
.\target\release\tarafer.exe models\Qwen3.5-4B-Q4_K_M.gguf --profile
```

Latest profile: see `profile-logs/latest_Qwen3.5-4B-Q4_K_M.txt`.
