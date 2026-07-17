# Details

Date : 2026-07-17 01:41:14

Directory d:\\taraference

Total : 71 files,  13182 codes, 692 comments, 1278 blanks, all 15152 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [taraference/.github/workflows/release.yml](/taraference/.github/workflows/release.yml) | YAML | 117 | 13 | 22 | 152 |
| [taraference/AGENTS.md](/taraference/AGENTS.md) | Markdown | 10 | 0 | 6 | 16 |
| [taraference/GOAL.md](/taraference/GOAL.md) | Markdown | 74 | 0 | 31 | 105 |
| [taraference/LOCAL\_3050TI\_MILESTONE.md](/taraference/LOCAL_3050TI_MILESTONE.md) | Markdown | 32 | 0 | 15 | 47 |
| [taraference/OLLAMA\_LOCAL\_COMPARISON.md](/taraference/OLLAMA_LOCAL_COMPARISON.md) | Markdown | 54 | 0 | 20 | 74 |
| [taraference/README.md](/taraference/README.md) | Markdown | 225 | 0 | 87 | 312 |
| [taraference/VLLM\_LLAMA\_CPP\_RESEARCH.md](/taraference/VLLM_LLAMA_CPP_RESEARCH.md) | Markdown | 299 | 0 | 133 | 432 |
| [taraference/WORKFLOW.md](/taraference/WORKFLOW.md) | Markdown | 59 | 0 | 22 | 81 |
| [taraference/benchmarks/ollama\_chat\_bench.py](/taraference/benchmarks/ollama_chat_bench.py) | Python | 77 | 1 | 12 | 90 |
| [taraference/crates/cli/src/download.rs](/taraference/crates/cli/src/download.rs) | Rust | 285 | 19 | 20 | 324 |
| [taraference/crates/cli/src/main.rs](/taraference/crates/cli/src/main.rs) | Rust | 249 | 38 | 27 | 314 |
| [taraference/crates/cli/src/profile.rs](/taraference/crates/cli/src/profile.rs) | Rust | 970 | 35 | 45 | 1,050 |
| [taraference/crates/cli/src/self\_update.rs](/taraference/crates/cli/src/self_update.rs) | Rust | 186 | 9 | 24 | 219 |
| [taraference/crates/cli/src/serve/mod.rs](/taraference/crates/cli/src/serve/mod.rs) | Rust | 384 | 7 | 36 | 427 |
| [taraference/crates/cli/src/serve/openai.rs](/taraference/crates/cli/src/serve/openai.rs) | Rust | 91 | 5 | 14 | 110 |
| [taraference/crates/core/src/chat.rs](/taraference/crates/core/src/chat.rs) | Rust | 98 | 15 | 17 | 130 |
| [taraference/crates/core/src/config.rs](/taraference/crates/core/src/config.rs) | Rust | 207 | 13 | 18 | 238 |
| [taraference/crates/core/src/cuda/decode.rs](/taraference/crates/core/src/cuda/decode.rs) | Rust | 202 | 42 | 29 | 273 |
| [taraference/crates/core/src/cuda/forward.rs](/taraference/crates/core/src/cuda/forward.rs) | Rust | 340 | 17 | 24 | 381 |
| [taraference/crates/core/src/cuda/kernels/attn/fast\_v2.cu](/taraference/crates/core/src/cuda/kernels/attn/fast_v2.cu) | CUDA C++ | 160 | 9 | 24 | 193 |
| [taraference/crates/core/src/cuda/kernels/attn/flash.cu](/taraference/crates/core/src/cuda/kernels/attn/flash.cu) | CUDA C++ | 219 | 16 | 27 | 262 |
| [taraference/crates/core/src/cuda/kernels/common.cu](/taraference/crates/core/src/cuda/kernels/common.cu) | CUDA C++ | 165 | 12 | 10 | 187 |
| [taraference/crates/core/src/cuda/kernels/deltanet.cu](/taraference/crates/core/src/cuda/kernels/deltanet.cu) | CUDA C++ | 515 | 79 | 30 | 624 |
| [taraference/crates/core/src/cuda/kernels/embed.cu](/taraference/crates/core/src/cuda/kernels/embed.cu) | CUDA C++ | 300 | 3 | 16 | 319 |
| [taraference/crates/core/src/cuda/kernels/gemm.cu](/taraference/crates/core/src/cuda/kernels/gemm.cu) | CUDA C++ | 279 | 6 | 6 | 291 |
| [taraference/crates/core/src/cuda/kernels/gemv.cu](/taraference/crates/core/src/cuda/kernels/gemv.cu) | CUDA C++ | 1,374 | 38 | 82 | 1,494 |
| [taraference/crates/core/src/cuda/kernels/mod.rs](/taraference/crates/core/src/cuda/kernels/mod.rs) | Rust | 10 | 7 | 2 | 19 |
| [taraference/crates/core/src/cuda/kernels/ops.cu](/taraference/crates/core/src/cuda/kernels/ops.cu) | CUDA C++ | 296 | 8 | 18 | 322 |
| [taraference/crates/core/src/cuda/kv.rs](/taraference/crates/core/src/cuda/kv.rs) | Rust | 23 | 15 | 5 | 43 |
| [taraference/crates/core/src/cuda/layer.rs](/taraference/crates/core/src/cuda/layer.rs) | Rust | 1,414 | 21 | 27 | 1,462 |
| [taraference/crates/core/src/cuda/load.rs](/taraference/crates/core/src/cuda/load.rs) | Rust | 728 | 28 | 20 | 776 |
| [taraference/crates/core/src/cuda/matmul.rs](/taraference/crates/core/src/cuda/matmul.rs) | Rust | 855 | 28 | 30 | 913 |
| [taraference/crates/core/src/cuda/mod.rs](/taraference/crates/core/src/cuda/mod.rs) | Rust | 12 | 1 | 3 | 16 |
| [taraference/crates/core/src/cuda/model.rs](/taraference/crates/core/src/cuda/model.rs) | Rust | 158 | 27 | 10 | 195 |
| [taraference/crates/core/src/cuda/types.rs](/taraference/crates/core/src/cuda/types.rs) | Rust | 162 | 17 | 11 | 190 |
| [taraference/crates/core/src/engine.rs](/taraference/crates/core/src/engine.rs) | Rust | 162 | 16 | 19 | 197 |
| [taraference/crates/core/src/lib.rs](/taraference/crates/core/src/lib.rs) | Rust | 13 | 5 | 3 | 21 |
| [taraference/crates/core/src/quant.rs](/taraference/crates/core/src/quant.rs) | Rust | 6 | 2 | 2 | 10 |
| [taraference/crates/core/src/session.rs](/taraference/crates/core/src/session.rs) | Rust | 365 | 26 | 32 | 423 |
| [taraference/crates/core/src/tokenizer/bytes.rs](/taraference/crates/core/src/tokenizer/bytes.rs) | Rust | 49 | 4 | 6 | 59 |
| [taraference/crates/core/src/tokenizer/mod.rs](/taraference/crates/core/src/tokenizer/mod.rs) | Rust | 224 | 5 | 18 | 247 |
| [taraference/crates/core/src/tokenizer/special.rs](/taraference/crates/core/src/tokenizer/special.rs) | Rust | 42 | 2 | 4 | 48 |
| [taraference/crates/cuda\_probe/src/main.rs](/taraference/crates/cuda_probe/src/main.rs) | Rust | 10 | 0 | 1 | 11 |
| [taraference/crates/gguf/examples/list\_types.rs](/taraference/crates/gguf/examples/list_types.rs) | Rust | 22 | 0 | 2 | 24 |
| [taraference/crates/gguf/examples/probe.rs](/taraference/crates/gguf/examples/probe.rs) | Rust | 38 | 0 | 1 | 39 |
| [taraference/crates/gguf/examples/probe2.rs](/taraference/crates/gguf/examples/probe2.rs) | Rust | 23 | 3 | 1 | 27 |
| [taraference/crates/gguf/examples/probe3.rs](/taraference/crates/gguf/examples/probe3.rs) | Rust | 19 | 1 | 1 | 21 |
| [taraference/crates/gguf/examples/probe4.rs](/taraference/crates/gguf/examples/probe4.rs) | Rust | 10 | 1 | 1 | 12 |
| [taraference/crates/gguf/examples/probe5.rs](/taraference/crates/gguf/examples/probe5.rs) | Rust | 5 | 0 | 1 | 6 |
| [taraference/crates/gguf/examples/probe\_norm.rs](/taraference/crates/gguf/examples/probe_norm.rs) | Rust | 24 | 0 | 1 | 25 |
| [taraference/crates/gguf/examples/probe\_ssm.rs](/taraference/crates/gguf/examples/probe_ssm.rs) | Rust | 16 | 1 | 1 | 18 |
| [taraference/crates/gguf/src/error.rs](/taraference/crates/gguf/src/error.rs) | Rust | 29 | 0 | 12 | 41 |
| [taraference/crates/gguf/src/lib.rs](/taraference/crates/gguf/src/lib.rs) | Rust | 8 | 6 | 3 | 17 |
| [taraference/crates/gguf/src/reader.rs](/taraference/crates/gguf/src/reader.rs) | Rust | 187 | 8 | 28 | 223 |
| [taraference/crates/gguf/src/types.rs](/taraference/crates/gguf/src/types.rs) | Rust | 184 | 8 | 10 | 202 |
| [taraference/crates/gguf/src/value.rs](/taraference/crates/gguf/src/value.rs) | Rust | 271 | 3 | 14 | 288 |
| [taraference/openai-test-python/README.md](/taraference/openai-test-python/README.md) | Markdown | 25 | 0 | 11 | 36 |
| [taraference/openai-test-python/assistant.py](/taraference/openai-test-python/assistant.py) | Python | 146 | 17 | 24 | 187 |
| [taraference/openai-test-python/requirements.txt](/taraference/openai-test-python/requirements.txt) | pip requirements | 1 | 0 | 1 | 2 |
| [taraference/profile-logs/ollama-l4/bench\_chat.py](/taraference/profile-logs/ollama-l4/bench_chat.py) | Python | 72 | 0 | 8 | 80 |
| [taraference/profile-logs/ollama-l4/ollama-run1.jsonl](/taraference/profile-logs/ollama-l4/ollama-run1.jsonl) | JSON Lines | 6 | 0 | 1 | 7 |
| [taraference/profile-logs/ollama-l4/ollama-run2.jsonl](/taraference/profile-logs/ollama-l4/ollama-run2.jsonl) | JSON Lines | 6 | 0 | 1 | 7 |
| [taraference/profile-logs/ollama-l4/ollama-warm-run2.jsonl](/taraference/profile-logs/ollama-l4/ollama-warm-run2.jsonl) | JSON Lines | 6 | 0 | 1 | 7 |
| [taraference/profile-logs/ollama-l4/ollama-warm-run3.jsonl](/taraference/profile-logs/ollama-l4/ollama-warm-run3.jsonl) | JSON Lines | 6 | 0 | 1 | 7 |
| [taraference/review-by/core\_inference\_engine\_review.md](/taraference/review-by/core_inference_engine_review.md) | Markdown | 87 | 0 | 33 | 120 |
| [taraference/review-by/placetoimporve.md](/taraference/review-by/placetoimporve.md) | Markdown | 93 | 0 | 23 | 116 |
| [taraference/review-by/placetoimporve/README.md](/taraference/review-by/placetoimporve/README.md) | Markdown | 93 | 0 | 23 | 116 |
| [taraference/scripts/README.md](/taraference/scripts/README.md) | Markdown | 43 | 0 | 20 | 63 |
| [taraference/scripts/get-binary.sh](/taraference/scripts/get-binary.sh) | Shell Script | 71 | 19 | 14 | 104 |
| [taraference/scripts/install.ps1](/taraference/scripts/install.ps1) | PowerShell | 96 | 20 | 15 | 131 |
| [taraference/scripts/install.sh](/taraference/scripts/install.sh) | Shell Script | 95 | 16 | 18 | 129 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)