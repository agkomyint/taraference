# Goal: single-user maximum tokens per second

## North star

**Optimize for one user, maximum decode tokens per second (`tok/s`).**

Not multi-user throughput. Not concurrent HTTP scale. Not “how many chats can share one GPU.”

If a change raises total server QPS but **lowers** single-stream decode `tok/s` for one chat, it is **out of scope** for this goal.

---

## What we optimize

| Metric | Meaning | Priority |
|--------|---------|----------|
| **Decode tok/s** | Generated tokens / decode wall time (single stream) | **Primary** |
| **TTFT** | Time to first token (prompt prefill + first decode) | Secondary — keep reasonable, do not sacrifice decode for tiny TTFT wins |
| **Prefill tok/s** | Prompt processing speed | Important for long prompts / multi-turn append; still secondary to decode |
| **Decode drop vs context** | First-turn vs late-turn decode `tok/s` as `ctx` grows | Track and reduce (attention/KV path) |
| **Est. weight bandwidth** | From profile: how close decode is to GPU memory peak | Diagnostic for GEMV/kernel quality |

**Success looks like:** higher **single-request** decode `tok/s` on the same model + same GPU, measured with `--profile` (same model stem, same decode backend family, fair A/B).

Use:

- `profile-logs/latest_Qwen2.5-3B-Instruct-Q4_K_M.txt` (and matching `index.csv` rows)
- `SUMMARY_KV` fields: `overall_decode_tps`, `decode_tps_first`, `decode_tps_last`, `decode_drop_pct`

---

## Model policy (mandatory)

| Rule | Detail |
|------|--------|
| **Only 3B for profile / improve / A/B** | **Qwen2.5-3B-Instruct-Q4_K_M** is the **sole** model for performance work |
| **Do not use 0.5B** | Not for profiling, not for “quick” speed A/B, not for declaring wins, not for agent iteration |
| **Download** | Prefer `tarafer --download 3b` for the scoreboard model |
| **Path** | `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf` (local) or `~/models/…` / `/tmp/tara-models/…` (SSH) |

**0.5B** may still exist in the download catalog for end users who want a tiny demo, but **this project’s optimize loop never uses it.**

Canonical measure:

```text
tarafer models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode fastv2
# or --decode flash / --cuda-graph as the experiment requires
```

---

## What we explicitly do **not** optimize (for now)

These matter for production multi-tenant serving. They are **not** the current goal:

| Out of scope | Why |
|--------------|-----|
| Continuous batching of many requests | Raises aggregate tok/s; often hurts **per-user** latency/tok/s |
| PagedAttention for packing many sequences | Memory efficiency under concurrency, not single-stream peak |
| Prefix / radix cache across users | Shared-prefix multi-session throughput |
| Request queues, fairness, max concurrent jobs | Server product features |
| Multi-replica / load balancing | Scale-out, not one-stream speed |
| DeepSpeed-style multi-user inference serving | Different problem |

**Exception:** work that *also* helps single-stream (e.g. better attention kernels, CUDA graphs, fused matmul) is still in scope even if servers use the same idea for batching later.

---

## Target workload

- **One** interactive session (CLI chat or one API client at a time).
- **Scoreboard model only:** **Qwen2.5-3B-Instruct-Q4_K_M** (see Model policy above). Larger models (7B/14B) are optional later; **not** 0.5B.
- CUDA path in `crates/core`.
- Hardware focus: consumer NVIDIA (e.g. RTX 3050 Ti 4GB) for smoke; **authoritative** numbers on SSH GPU (e.g. T4) with the **same 3B** weights.
- Multi-turn chat with growing KV is normal — optimize so **late-context decode** stays as close as possible to early-context decode.

Serialized server use (`--serve` with one request at a time) is fine as a delivery mode. **Do not** design the engine around many concurrent `/v1/chat/completions` callers.

---

## In-scope levers (single-user tok/s)

Ordered by fit to this goal (not a commitment to implement all at once):

1. **Decode path efficiency** — bandwidth-friendly GEMV (e.g. split-K), fewer wasted warps, higher effective GB/s on weights.
2. **CUDA graphs (or equivalent)** — cut CPU/driver launch overhead per token on the fixed single-token loop.
3. **Operator fusion** — e.g. RMSNorm + QKV, fewer HBM round-trips and launches per layer.
4. **Attention / KV for long context** — better online/tiled attention so decode does not collapse as `ctx` grows; optional KV quant if it preserves or improves effective tok/s and fits VRAM.
5. **Speculative decode that helps one stream** — e.g. prompt lookup decoding, draft/verify, Medusa/EAGLE-style multi-token — when it raises **accepted tokens per second** for one user.
6. **Quant / Tensor Core paths** when they improve **single-stream** prefill or decode on this hardware (not only large-batch matmul).
7. **Correct GPU arch targeting** (`sm_XX`) so kernels are not miscompiled for the device.

Prefer changes that show up as higher `overall_decode_tps` in profile logs.

---

## Decision rule

Before accepting a design or PR aimed at “performance”:

1. Does it improve **one user’s** decode `tok/s` (or hold decode flat while clearly improving TTFT/prefill for the same session)?
2. Can we measure it with **`--profile`** on a fixed model without multi-client load generators?
3. If it only helps when `batch_size` or concurrent requests ≫ 1, **defer** it.

When in doubt: **single-stream decode tok/s wins.**

---

## How we measure

```text
# ONLY 3B — never 0.5B for this goal
cargo run --release -p taraference -- models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode fastv2
```

Compare only `latest_Qwen2.5-3B-Instruct-Q4_K_M.txt` / matching `index.csv` rows.  
**Never** use 0.5B (or other sizes) as a stand-in when judging a win.

Report at least:

- **GPU identity** — `gpu_name`, `gpu_compute_cap`, `gpu_nvrtc_arch` (logs always include these; do not A/B tok/s across different GPUs without labeling)
- `overall_decode_tps`
- `decode_tps_first` / `decode_tps_last` / `decode_drop_pct`
- optional: prefill tok/s, TTFT, est. weight BW, GPU clocks/thermal notes

A change is a **win** when single-user decode tok/s goes up (or decode holds and prefill/TTFT improve materially) without unacceptable quality regressions (wrong tokens, broken multi-turn KV, etc.).

---

## Relationship to other docs

| Doc | Role |
|-----|------|
| [README.md](README.md) | How to install, run, serve, profile |
| [WORKFLOW.md](WORKFLOW.md) | **How** we iterate: SSH GPU baseline → laptop code → release CI → `tarafer update` → re-profile |
| **GOAL.md** (this file) | **What** we optimize for and what we ignore for now |
| `review-by/placetoimporve/` | Technique backlog; filter that backlog through this goal |

Server and multi-user features may exist for convenience. They are not the performance north star until this document is explicitly revised.

---

## One-sentence summary

**Make one chat as many tokens per second as possible on one GPU; ignore multi-user packing until that single-stream path is strong.**
