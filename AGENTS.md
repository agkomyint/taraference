# Agent notes (taraference)

## Scoreboard model — mandatory

**Only use Qwen2.5-3B-Instruct-Q4_K_M for:**

- `--profile`
- speed A/B (decode backends, CUDA graph, kernels, PLD, …)
- claiming a win or regression
- iterative improve loops

**Do not use 0.5B** for any of the above (not even a quick tok/s signal).

```text
models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
# download: tarafer --download 3b
```

Policy details live in [GOAL.md](GOAL.md) and [WORKFLOW.md](WORKFLOW.md).
