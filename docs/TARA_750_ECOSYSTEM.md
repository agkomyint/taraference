# Tara ↔ taraference 750 tok/s ecosystem

Training and co-design live in Tara HQ:

```text
D:\Tara_HQ\departments\taraference_750_department\
```

## Flagship SKU (locked)

| | **Tara MoE 1B-A100** |
|--|--|
| **Total** | **~1.0B** (E=18 experts) |
| **Active** | **~100M** (top_k=1, same shape as moe-500-a100) |
| **Target** | **≥750** single-stream decode tok/s |
| **Config** | `taraference_750_department/configs/tara_moe_1b_a100.json` |
| **Spec** | `taraference_750_department/TARA_MOE_1B_A100.md` |

**Idea:** market **1B total**; pay BW only for **~100M active**. Grow total via **more experts**; grow tok/s via **engine** at fixed active.

### Today vs target (same active band)

| Pack / shape | Total | Active | Warm t/s (3050 Ti, Q4) |
|--|--:|--:|--:|
| speed750 | smaller | ~34M | **~750–790** (proven) |
| moe-500-a100 | ~0.47B | ~100M | **~500** (engine baseline) |
| **1b-a100** | **~1.0B** | **~100M** | **~500 now → 750 goal** |

1B-A100 does **not** cost more per token than moe-500 — only more VRAM for cold experts. The 750 gap is **~1.5× engine efficiency** at fixed active.

## Why not 3B/4B dense

3B/4B models on this stack top out well below 750 single-stream tok/s.  
The 750 vehicle is **sparse MoE with ~70–120M active** (flagship: **~100M active / ~1B total**).

## Co-design split

| Side | Owns |
|------|------|
| **Train (Tara HQ)** | Configs, data, long train, router health, export Q4 pack |
| **Engine (taraference)** | Fused MoE decode, Q4 BW, CUDA graphs, vocab shortlist, profile metric |
| **Shared contract** | d/L/ff/E/top_k, pack layout, `single_stream_accepted_tps ≥ 750` |

Engine priorities for 100M@750: mega MoE FFN fuse → aligned Q4 → split-K (see `TARA_MOE_1B_A100.md`).

## Ladder (context)

| Codename | Total | Active | Role |
|--|--:|--:|--|
| speed750 | low | ~34M | Proven 750; quality limited |
| moe-500-a100 | ~0.47B | ~100M | Engine baseline @ 100M active |
| **1b-a100** | **~1.0B** | **~100M** | **Flagship marketing + 750 target** |
| Future | 1.3B+ | ~100M | More experts only |

## Metric

Always `single_stream_accepted_tps` / decode tok/s from taraference on a **named GPU + quant**.  
Never confuse with multi-user aggregate throughput.  
Never claim 750 with `TARAFER_MOE_FIXED` or top_k cheats as the product number.
