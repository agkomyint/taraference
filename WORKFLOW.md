# Workflow: develop and profile directly on the SSH GPU

## Roles

| Environment | Role |
|---|---|
| Windows laptop / RTX 3050 Ti | Edit, unit tests, build, correctness, second-GPU validation |
| SSH T4 (`sm_75`) | Authoritative kernel iteration and v0.3 regression numbers |
| Release CI | Packaging only after a measured candidate wins |

## Iteration loop

1. Use the mandatory `Qwen2.5-3B-Instruct-Q4_K_M.gguf` scoreboard.
2. Record the installed baseline on the SSH GPU using the exact model and flags.
3. Make one isolated source change locally.
4. Sync `Cargo.toml`, `Cargo.lock`, and `crates/` to a dedicated SSH work directory.
5. Build there with `cargo build --release -p taraference`.
6. Smoke-test text generation and CUDA architecture detection.
7. Run `--profile` on the track's exact model.
8. Repeat the candidate; re-run baseline when the delta is close to noise.
9. Reject regressions immediately. Keep only the winning kernel/backend.
10. Run workspace tests and profile the laptop GPU before packaging.

Do not wait for release CI during kernel iteration and do not install candidate
binaries over the known-good `~/.local/bin/tarafer`.

## SSH setup

```bash
ssh <user>@ssh.lightning.ai
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
. "$HOME/.cargo/env"
```

Use an isolated directory such as `/tmp/tara-v04-dev`. Keep the installed release
binary as the baseline and the source-built binary as the candidate.

## v0.3 regression commands

```bash
MODEL=/home/zeus/content/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf

# baseline release
mkdir -p /tmp/tara-ab/base && cd /tmp/tara-ab/base
~/.local/bin/tarafer "$MODEL" --profile --decode fastv2 --no-cuda-graph

# candidate source build; v0.4 default is flash + CUDA graph
mkdir -p /tmp/tara-ab/candidate && cd /tmp/tara-ab/candidate
/tmp/tara-v04-dev/target/release/tarafer "$MODEL" --profile
```

Capture `gpu_name`, `gpu_compute_cap`, `gpu_nvrtc_arch`, `overall_decode_tps`,
`decode_tps_first`, `decode_tps_last`, and `decode_drop_pct`.

## Modern model integration gate

Modern models may be added for correctness work, but do not profile them or use
them in speed A/B loops under the current scoreboard policy. Before proposing a
future policy change:

- implement its GGUF architecture/config and tokenizer behavior;
- add a deterministic short correctness fixture;
- verify KV growth and multi-turn chat;
- run it on both T4 and RTX 3050 Ti;
- freeze one exact Q4_K_M filename and a prospective profile script for review.

Do not substitute a different small model and carry its tok/s into the 750 claim.

## Candidate decision

Keep a candidate only when:

- accepted single-stream decode improves on the selected scoreboard;
- generated text remains coherent and numerical approximation is documented;
- first/last context behavior does not regress unacceptably;
- the code works on both `sm_75` and `sm_86`;
- losing flags, registry rows, kernels, and temporary workarounds are removed.

After those gates, bump the version, build the Linux release, and optionally install
it. Tagging/pushing is not part of the performance experiment itself.
