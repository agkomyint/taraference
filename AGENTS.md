# Agent notes (taraference)

## Scoreboard & Profiling Models

**Official Scoreboard & Baseline (`Qwen2.5`):**
- `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf` (download: `tarafer --download 3b-qwen25`)
Use this for historical v0.4/v0.5 regression comparisons and the 750 tok/s north star.

**Modern Profiling & Evaluation (`Qwen3.5`):**
- `models/Qwen3.5-4B-Q4_K_M.gguf` (download: `tarafer --download 4b`)
You are permitted to run `--profile`, speed A/B testing, and kernel iteration on `Qwen3.5-4B` and other supported models. Always explicitly note the exact model name when reporting metrics.

**Do not use 0.8B/0.5B** for claiming top-line speed wins or official scoreboard numbers.

Policy details live in [GOAL.md](GOAL.md) and [WORKFLOW.md](WORKFLOW.md).
