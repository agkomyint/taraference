# Workflow: ship, measure, improve (toward GOAL.md)

This is the **operating loop** for reaching [GOAL.md](GOAL.md):

> **One user, maximum decode tokens per second** on a real GPU — not multi-user scale.

We use two machines on purpose:

| Machine | Role |
|---------|------|
| **Laptop** (e.g. Windows + RTX 3050 Ti) | Edit code, compile, smoke-test “does it run?”, catch Rust/API breaks |
| **SSH GPU box** (e.g. Lightning T4) | **Authoritative** single-stream `--profile` numbers; optimize for **that** GPU’s architecture |

**Rule:** A change is only a **win** if decode `tok/s` goes up (or holds while TTFT/prefill improve) on the **SSH GPU**, **same model**, measured with `--profile`. Laptop “it runs” is necessary but **not** sufficient.

### Model policy (mandatory — read this)

| Do | Don’t |
|----|--------|
| **Only** profile / A/B / improve with **Qwen2.5-3B-Instruct-Q4_K_M** | Use **0.5B** for profiling, speed A/B, or “quick” wins |
| `tarafer --download 3b` for the scoreboard model | Treat 0.5B tok/s as the project scoreboard |
| Compare `latest_Qwen2.5-3B-Instruct-Q4_K_M.txt` only | Mix model sizes in one “win” narrative |

Canonical file:

```text
Qwen2.5-3B-Instruct-Q4_K_M.gguf
```

(See [GOAL.md](GOAL.md) — Model policy.)

---

## Big picture loop

```text
  ┌──────────────────────────────────────────────────────────────────┐
  │  1. SETUP SSH GPU box (once)                                      │
  │  2. DOWNLOAD release → install tarafer on PATH                    │
  │  3. BASELINE profile → **3B only** (save numbers)                 │
  │  4. CHANGE code on laptop (new method / kernel / decode)          │
  │  5. SMOKE-TEST on laptop (build + short 3B run if VRAM allows)    │
  │  6. PUSH tag → Release CI builds Linux binary                     │
  │  7. WAIT for green CI + GitHub Release assets                     │
  │  8. SSH: tarafer update → re-profile **3B** vs baseline           │
  │  9. DECIDE: keep / tweak / try another approach                   │
  └───────────────────────────┬──────────────────────────────────────┘
                              │
                              └── repeat from 4 until GOAL metrics move
```

Never skip **measure on SSH**. Never “optimize for a GPU you only imagined.”  
Never use **0.5B** in this loop.

---

## Phase 0 — Know the goal (before any code)

Read [GOAL.md](GOAL.md). Primary scoreboard:

| Field | Why |
|-------|-----|
| `overall_decode_tps` | **Primary** — single-stream decode tok/s |
| `decode_tps_first` / `decode_tps_last` / `decode_drop_pct` | Long multi-turn health |
| Prefill / TTFT | Secondary |
| Est. weight BW, GPU util | Diagnostics |

**Out of scope for now:** multi-user batching, packing many chats, server QPS. See GOAL.md.

Filter every idea from `review-by/placetoimporve/` through: *does this raise one-user decode tok/s?*

---

## Phase 1 — Setup SSH (once per studio / box)

### 1.1 Connect

```bash
ssh <user>@ssh.lightning.ai
# or your cloud SSH host
```
ssh s_01kxk8x9fbqrpejxsczrq3wscg@ssh.lightning.ai

### 1.2 Record **this** GPU (optimize for it)

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
nvcc --version   # CUDA toolkit for NVRTC; want 13.x on current releases
```

Write down:

| Field | Example (Lightning) | Your box |
|-------|---------------------|----------|
| Name | Tesla T4 | |
| Compute capability | **7.5 → sm_75** | |
| VRAM | 15360 MiB | |
| Driver / CUDA | 580.x / 13.0 | |

**Arch note:** Runtime NVRTC should print `NVRTC arch=sm_XX` matching `compute_cap` (e.g. T4 → `sm_75`, 3050 Ti → `sm_86`).  
If you ever hardcode `sm_86` again, T4 will break or run wrong — always **detect live GPU** (see `crates/core` load path).

### 1.3 Shell PATH (if needed)

```bash
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
# persist in ~/.bashrc or ~/.zshrc if you want
```

---

## Phase 2 — Download release (fast path, no Rust on SSH)

```bash
curl -fsSL -o tarafer-linux-x86_64.tar.gz \
  https://github.com/agkomyint/taraference/releases/latest/download/tarafer-linux-x86_64.tar.gz
tar -xzf tarafer-linux-x86_64.tar.gz
chmod +x tarafer
./tarafer install          # → ~/.local/bin/tarafer
which tarafer
tarafer --version
```

Or after clone: `./scripts/get-binary.sh`

**Models (scoreboard = 3B only):**

```bash
mkdir -p ~/models && cd ~/models   # or any fixed dir you always reuse
tarafer --download 3b
# DO NOT download/profile 0.5b for speed work
```

Keep the 3B path stable so profiles are comparable run-to-run.

---

## Phase 3 — Baseline: see the result (before changing code)

Always establish a **baseline** on the SSH GPU with the **current release** and **3B only**.

```bash
cd ~/work   # any dir; profile-logs/ is written relative to cwd
export PATH="$HOME/.local/bin:$PATH"

# ONLY scoreboard model
tarafer ~/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode fastv2
# optional A/B same 3B: --decode flash  /  --cuda-graph  /  --no-cuda-graph
```

### What to capture

From the printed report / `profile-logs/latest_<model>.txt` (`SUMMARY_KV`):

```text
gpu_name=Tesla T4
gpu_compute_cap=7.5
gpu_nvrtc_arch=sm_75
gpu_driver=...
gpu_vram_total_mib=15360
overall_decode_tps=...
decode_tps_first=...
decode_tps_last=...
decode_drop_pct=...
overall_prefill_tps=...
wall_s=...
```

`index.csv` also has `gpu_name`, `gpu_compute_cap`, `gpu_nvrtc_arch` so you can filter laptop vs SSH runs.  
If “vs PREVIOUS” shows a **different GPU/arch**, the tok/s delta is **not** a fair code A/B.

Also keep:

- `profile-logs/index.csv` (one row per run; filter to 3B model stem)
- GPU name + compute_cap in a note (“baseline T4 sm_75, 3B, release vX.Y.Z”)

**Only** compare `latest_Qwen2.5-3B-Instruct-Q4_K_M.txt` (and same-stem rows).  
**Never** use 0.5B (or other sizes) when declaring a win.

Example baseline you already measured on Lightning T4 (illustrative **3B**):

| Model | overall decode | drop first→last | VRAM peak |
|-------|---------------:|----------------:|----------:|
| **3B Q4** | ~31 tok/s | ~8% | ~2.2 GiB |

Your next baselines replace these after each accepted release.

---

## Phase 4 — Develop on the laptop (implement new methodology)

### 4.1 Work locally

```text
# Windows example
cd D:\taraference
# edit crates/core (kernels, decode registry, matmul, load, …)
```

In-scope levers (from GOAL.md): better GEMV, fusion, attention/KV long-ctx, CUDA graphs, speculative decode that helps **one** stream, correct `sm_XX`, etc.

### 4.2 Smoke-test on laptop (correctness first)

```powershell
# build
cargo build --release -p taraference

# if laptop has NVIDIA + CUDA 13.x — smoke / profile on **3B only**
.\target\release\tarafer.exe models\Qwen2.5-3B-Instruct-Q4_K_M.gguf --prompt "hi" -n 32
# local profile (numbers are for *this* GPU, not T4 — still 3B only)
.\target\release\tarafer.exe models\Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode fastv2
```

**Laptop goals:**

- Compiles
- Loads GGUF
- NVRTC uses **laptop** arch (e.g. `sm_86`)
- Chat / prompt / multi-turn not obviously broken
- New `--decode` name appears if you added a backend

**Laptop goals that do *not* replace SSH:**

- Absolute tok/s ranking vs production SSH GPU  
- Declaring “this kernel is faster on T4” without measuring T4  

If the laptop has **no** GPU / no CUDA, still:

- `cargo check` / `cargo build --release`  
- CI will compile Linux binary; you **must** still validate on SSH before trusting numbers  

### 4.3 GPU architecture discipline

| Do | Don’t |
|----|--------|
| Query live compute capability for NVRTC | Hardcode `sm_86` only |
| Profile on the **target** SSH GPU | Assume T4 ≈ 3050 Ti |
| Add decode backends via registry + `.cu` | Special-case only laptop in kernels without measuring SSH |
| Note T4 = Turing `sm_75` (no newer Tensor Core features) | Use Hopper/Ada-only tricks without fallback |

When designing for SSH T4 specifically: prefer bandwidth-friendly SIMT paths that help **Turing**, not only Ada.

---

## Phase 5 — Push release CI and wait

### 5.1 Commit on `main` (or your branch → merge)

```bash
git status
git add -A   # review first
git commit -m "feat: <short why this should raise single-stream decode tok/s>"
git push origin main
```

### 5.2 Cut a version tag (triggers Release workflow)

```bash
git tag v0.3.0 -m "v0.3.0: <method name> — expect higher decode tok/s on T4"
git push origin v0.3.0
```

### 5.3 Wait for green CI

```bash
gh run list --workflow=release.yml --limit 3
gh run watch   # or open Actions on GitHub
```

Success criteria:

- Job **Build Linux x86_64** green  
- Release page has `tarafer-linux-x86_64.tar.gz` + `tarafer`  
- No need to install CUDA on the builder (dynamic load); runtime still needs CUDA 13 + GPU on SSH  

Typical build time with cache: ~1 minute.

### 5.4 If CI fails

Fix on laptop → push → new tag (or re-run after fix on same tag only if you force-moved tags carefully; prefer `v0.3.1`).

---

## Phase 6 — SSH: update binary and re-measure

On the **same** SSH box, same models, same `--decode` when comparing:

```bash
export PATH="$HOME/.local/bin:$PATH"

# Pull latest release into PATH install
tarafer update --install
tarafer --version

cd ~/work   # same place as baseline profile-logs if you want easy compare
# ONLY 3B
tarafer ~/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode fastv2
```

Confirm log line: `NVRTC arch=sm_75` (or your box’s arch).

### Compare fairly

| Compare | How |
|---------|-----|
| Same model stem | **only** `latest_Qwen2.5-3B-Instruct-Q4_K_M.txt` vs previous |
| Same decode family | e.g. both `fastv2`, or A/B `fastv2` vs `flash` |
| Same profile script | default multi-turn in `--profile` |
| Win? | Higher `overall_decode_tps` (or lower `decode_drop_pct` without losing overall) |

```bash
# quick look (3B only)
grep -E 'overall_decode|decode_drop|decode_tps_' profile-logs/latest_Qwen2.5-3B*.txt
grep 3B profile-logs/index.csv
```

---

## Phase 7 — Decide (keep / change / abandon)

| Outcome | Action |
|---------|--------|
| **Win** (decode tok/s up, quality OK) | Keep release; update baseline notes; next idea |
| **Tie / noise** | Re-run profile once; check clocks/thermal; don’t ship narrative without signal |
| **Regress** | Revert or fix; do **not** leave broken latest if others use it — ship a revert tag |
| **Idea wrong for T4** | Park it; try next method from backlog filtered by GOAL.md |
| **Works on laptop, worse on T4** | Arch/bandwidth mismatch — redesign for SSH GPU, don’t average the two |

**If not good:** change approach on laptop → new release → SSH update again.  
**If good:** you may use that release daily (`tarafer` chat / `--serve`) and treat it as the new production binary on SSH.

---

## Daily / weekly cheat sheet

### First time on a new SSH studio

```text
ssh → nvidia-smi → curl release → tarafer install → --download 3b → --profile 3B → save baseline
```

### Every performance experiment

```text
laptop: implement → build/smoke on 3B
git tag vX.Y.Z → push → wait CI green
ssh: tarafer update --install → --profile 3B → compare SUMMARY_KV → keep or iterate
```

### Commands you will use most

| Where | Command |
|-------|---------|
| SSH | `tarafer install` / `tarafer update` / `tarafer update --install` |
| SSH | `tarafer --download 3b` |
| SSH | `tarafer …/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile --decode fastv2` |
| SSH | same GGUF for chat or `--serve` |
| Laptop | `cargo build --release -p taraference` |
| Laptop | `tarafer.exe models\Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile …` |
| Either | `git tag` + GitHub Actions Release |

---

## Roles of each environment (summary)

```text
                    GOAL: max single-user decode tok/s
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
   ┌─────────────┐                                   ┌─────────────────┐
   │   Laptop    │  correctness, iteration speed     │   SSH GPU box   │
   │  3050 Ti    │  cargo, kernels, decode registry  │   T4 sm_75      │
   │  sm_86      │  "does it work?"                  │  "is it faster?"│
   └──────┬──────┘                                   └────────▲────────┘
          │  git push + tag v*                                │
          └────────────► GitHub Release CI ──tarafer update ──┘
                         (Linux x86_64 binary)
```

---

## Anti-patterns (avoid)

1. **Only** profiling on laptop and shipping “faster” without SSH numbers.  
2. Optimizing multi-user / continuous batching while GOAL.md still says single-stream.  
3. Hardcoding GPU arch.  
4. **Using 0.5B (or any non-3B) for profile / improve / win claims** — forbidden.  
5. Changing decode backend *and* code in one A/B without isolating variables.  
6. Skipping baseline before a big methodology change.  
7. Leaving SSH on an old binary while developing for weeks (always `update` before final measure).

---

## Checklist: “ready to claim a win”

- [ ] Change motivated by GOAL.md (single-stream decode)  
- [ ] Laptop: builds + short inference OK  
- [ ] Tag pushed; Release CI green; assets `tarafer-linux-x86_64.tar.gz`  
- [ ] SSH: `tarafer update --install`; `NVRTC arch=` matches box  
- [ ] SSH: `--profile` on **3B only** (same path as baseline)  
- [ ] `overall_decode_tps` improved (or documented secondary win)  
- [ ] `decode_drop_pct` not badly worse without explanation  
- [ ] Notes updated (baseline version + numbers)  

---

## Related docs

| Doc | Role |
|-----|------|
| [GOAL.md](GOAL.md) | **What** we optimize and ignore |
| [WORKFLOW.md](WORKFLOW.md) (this file) | **How** we iterate: SSH ↔ laptop ↔ release |
| [README.md](README.md) | Install, commands, serve, profile flags |
| [scripts/README.md](scripts/README.md) | `get-binary.sh`, install scripts |
| `review-by/placetoimporve/` | Idea backlog (filter through GOAL) |
| GitHub Releases / Actions | Binary factory |

---

## One-sentence summary

**Baseline on the SSH GPU, invent on the laptop, ship a Linux release, `tarafer update` on SSH, re-profile for single-user decode tok/s — keep only what that GPU actually rewards.**
