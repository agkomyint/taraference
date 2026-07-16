# RTX 3050 Ti local 3B milestone

Date: 2026-07-16

## Required model

All speed measurements in this note use the scoreboard model:

```text
models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
```

## Result

- Exact/default best observed official profile: **70.117 decode tok/s**.
- Experimental fixed 128-token test: **100.3 decode tok/s** with
  `TARAFER_FFN_SKIP=3` and `TARAFER_VOCAB_LIMIT=100000`.
- Experimental five-turn profile: **90.286 decode tok/s** overall, falling
  from **99.088** on turn 1 to **82.305** on turn 5 as GPU temperature reached
  86 C.

The fixed-test crossing is a useful experimental milestone, but it is not a
production win: skipping every third FFN damaged generation quality. The
five-turn output became repetitive and all five replies reached `max_new`.

## Safety and behavior

Both experimental controls are opt-in environment variables. With neither set,
taraference uses the normal full model path.

The active-vocabulary experiment retains Qwen's high-ID EOS token (`151645`) as
an explicit extra output column, so enabling a low-ID shortlist no longer makes
normal ChatML termination unreachable.

## Evidence

- Exact baseline: `profile-logs/profile_2026-07-16_22-13-57_Qwen2.5-3B-Instruct-Q4_K_M_flash.txt`
- Experimental sustained run: `profile-logs/profile_2026-07-16_22-50-03_Qwen2.5-3B-Instruct-Q4_K_M_flash.txt`

## Next technical direction

The exact path is primarily bandwidth-bound. FFN accounts for about 71% of
single-token layer time, with gate/up and down projections already moving about
159-161 GB/s. A genuine quality-preserving 100 tok/s result therefore needs
accepted multi-token work per weight read (efficient speculative verification)
or a better compressed representation, rather than arbitrary layer removal.
