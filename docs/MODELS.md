# Model registry

One registry covers every task: text, embeddings, speech and image. Entries live in
`crates/cli/src/registry.rs`. Each records the upstream repo and **pinned revision**, per-file
**SHA-256**, size, **licence**, whether commercial use is allowed, and how it is installed.
taraference never hosts or ships weights.

## Lifecycle

| Step | Command | What happens |
|---|---|---|
| Discover | `tarafer models list [--task text\|embed\|speech\|image] [--commercial]` | Table of ids, install state, size, licence |
| Inspect | `tarafer models info <id>` | Source URL and revision, licence, files, hashes, local state |
| Install | `tarafer models pull <id>...` | Shows licence; non-commercial needs a `y` or `--accept-license`; streams to `.partial`, verifies SHA-256, then renames |
| Check | `tarafer models verify [id]` | Re-hashes installed files (detects corruption or a swapped file) |
| Use | `text gen -m <id>`, `speech say`, `image remove-bg` | Tasks find files at the registry paths |
| Remove | `tarafer models rm <id>` | Deletes the files and an empty directory |
| Health | `tarafer doctor` | GPU and installed/partial models |

`--models-dir` (default `models`) applies to every `models` command. `HF_TOKEN` is sent for gated repos.

## Installed models

| id | task | licence | commercial | install |
|---|---|---|---|---|
| `0.8b`, `2b`, `4b` | text (Qwen3.5 Q4_K_M) | Apache-2.0 | yes | `models pull` |
| `3b-qwen25` | text (Qwen2.5-3B, scoreboard baseline) | Qwen Research License | **no** | `models pull` |
| `embed-4b` | embeddings (Qwen3-Embedding-4B) | Apache-2.0 | yes | `models pull` |
| `mms-mya` | speech, Burmese (MMS-TTS) | CC BY-NC 4.0 | **no** | `models pull` |
| `f5-myanmar` | speech, Burmese (F5-TTS v2) | CC BY-NC 4.0 | **no** | manual export (`tools/tts/export_f5_myanmar.py`) |
| `myanmar-asr` | speech-to-text, Burmese (Conformer CTC) | Apache-2.0 | yes | manual export (`tools/stt/export_myanmar_asr.py`) |
| `birefnet-lite` | image background removal | MIT | yes | `models pull` |

Licences come from the upstream Hugging Face model metadata (checked 2026-10-14). This is
engineering guidance, not legal advice: read the upstream licence before commercial use.

## Adding a model

1. Add an `Entry` to `REGISTRY` (task, repo, pinned commit, files, licence, commercial flag).
2. Fill `sha256` from the repo's file listing at that commit (`api/models/<repo>/tree/<commit>`
   gives `lfs.oid`) or hash the file yourself. The unit test requires 64-hex hashes for every
   downloadable entry.
3. If the task needs a runtime, add its front-end (`text`, `speech`, `image`) and point it at
   `Entry::main_path`. Use `Install::Manual` when weights must be built locally.
4. `cargo test -p taraference` and `tarafer models pull <id> && tarafer models verify <id>`.

## Gaps

- No `models pull` for `f5-myanmar` (needs a Python export); `verify` reports it as unpinned.
- No update check: a pin moves only when the registry entry changes with a release.
- Speech and image front-ends read fixed default paths; custom locations use their own flags.
