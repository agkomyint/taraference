# Embedding model support proposal

Date: 2026-07-29

## Decision

Add first-class support for the official
[`Qwen/Qwen3-Embedding-4B-GGUF`](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF),
using `Qwen3-Embedding-4B-Q4_K_M.gguf` as the frozen first-party checkpoint.

This model should be Taraference's first embedding target. It gives the best
combination of retrieval quality, multilingual/code coverage, local deployment
size, official GGUF availability, and reuse of Taraference's existing dense Qwen
CUDA implementation.

This does not replace or modify the generation scoreboard. Generation performance
claims remain attached to `Qwen2.5-3B-Instruct-Q4_K_M.gguf` under
[`roadmap/0.7.0/ROADMAP.md`](../roadmap/0.7.0/ROADMAP.md).

## Why this model

- Official Apache-2.0 Qwen release and official GGUF repository.
- 4B parameters, 36 layers, 32K advertised context, and 2560-dimensional output.
- Supports more than 100 languages, code retrieval, task instructions, and
  Matryoshka-style dimensions from 32 through 1024.
- Qwen reports a 69.45 multilingual MTEB task mean for the 4B model, versus
  64.33 for its 0.6B model and 59.56 for BGE-M3 in the same model-card
  comparison. Treat these as
  vendor-reported reference results until Taraference reproduces its own subset.
- The official GGUF supplies Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0, and F16.
  Q4_K_M is the right initial target because it is an official conversion and is
  the practical option for Taraference's RTX 3050 Ti validation GPU. Q8_0 alone
  is too large for that 4 GB target once KV and workspaces are included.
- The GGUF reports `general.architecture = qwen3`. Taraference already parses
  generic dense GGUF metadata and loads the relevant Qwen-style tensors,
  including optional Q/K normalization weights. This makes the integration much
  smaller than adding an XLM-R/BERT encoder architecture.

## Alternatives considered

| Model | Strength | Why not first |
|---|---|---|
| `Qwen3-Embedding-0.6B` | Small and efficient; 1024 dimensions | Lower multilingual and retrieval quality than the selected 4B model. Consider later as an opt-in low-memory tier. |
| `BAAI/bge-m3` | Mature, multilingual, 8192 tokens; dense, sparse, and multi-vector retrieval | XLM-R encoder and its pooling/output heads are a new architecture. Sparse and ColBERT outputs also expand the API and kernel scope substantially. |
| `nomic-embed-text-v1.5` | Small, proven long-context English model with adjustable dimensions | BERT encoder requires a new bidirectional architecture; English-focused and weaker strategic fit than Qwen3 multilingual/code retrieval. |

## Required Taraference work

### 1. Model identity and download

- Add `BackendKind::Qwen3Embedding`.
- Detect embedding purpose from explicit supported-model registry plus GGUF
  metadata; do not infer that every `qwen3` checkpoint is an embedder.
- Add download tag `embed-4b`:
  - repo: `Qwen/Qwen3-Embedding-4B-GGUF`
  - file: `Qwen3-Embedding-4B-Q4_K_M.gguf`
- Keep embeddings out of the generation download groups and scoreboard profiles.

### 2. Correct embedding forward path

- Tokenize each input with room for, and append, `<|endoftext|>`.
- Run the causal Qwen3 transformer without the vocabulary projection.
- Apply the final `output_norm` to the last non-padding token hidden state.
- If a shorter Matryoshka dimension is requested, truncate to that dimension.
- L2-normalize after truncation and return `f32`.
- Reset/reuse KV state independently for every input. Initial implementation can
  process inputs serially; true multi-sequence batching can follow.

The existing CUDA functions named `embed_*` perform token embedding lookup. They
do not currently produce sentence embeddings.

### 3. Public API

Implement the OpenAI-compatible endpoint:

`POST /v1/embeddings`

Minimum request fields:

```json
{
  "model": "Qwen3-Embedding-4B-Q4_K_M",
  "input": ["first text", "second text"],
  "encoding_format": "float",
  "dimensions": 1024
}
```

Return the standard ordered `data[].embedding`, `index`, `model`, and token
`usage` fields. Accept either one string or a list of strings. Reject empty input,
unsupported dimensions, over-context input, and `encoding_format` values other
than `float` in v1.

Also expose a core API similar to:

```rust
pub fn embeddings(
    &mut self,
    inputs: &[String],
    dimensions: Option<usize>,
) -> anyhow::Result<Vec<Vec<f32>>>
```

Do not automatically add a retrieval instruction to every string. Documents
normally have no instruction. Offer an explicit query-task helper or leave query
formatting to the caller:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: <query text>
```

### 4. Context and memory policy

Although the model advertises 32K context, Taraference currently uses a persistent
KV arena and a maximum prefill chunk of 256. Start with an explicitly documented
8192-token API cap, matching Qwen's reference example, and verify actual memory on
both T4 and RTX 3050 Ti. Do not advertise 32K until it passes memory and
correctness tests on the supported backend.

## Acceptance gates

1. Load the exact official Q8_0 GGUF on both `sm_75` and `sm_86`.
2. Compare full 1024-d vectors with Transformers or llama.cpp on at least 20 fixed
   English, Thai, Chinese, and code inputs:
   - cosine similarity against the reference vector at least 0.999;
   - vector L2 norm within `1e-4` of 1.0.
3. Verify reduced dimensions 256, 512, and 768 against truncate-then-normalize
   reference output.
4. Retrieval fixture: the correct document ranks first for at least 9 of 10
   deterministic query/corpus cases, including cross-lingual and code cases.
5. `/v1/embeddings` schema tests for string/list input, stable ordering, dimensions,
   usage accounting, empty input, and context overflow.
6. Report embedding throughput separately as inputs/s and tokens/s with exact
   model, quantization, GPU, batch size, sequence-length distribution, and vector
   dimension. Never mix it with generation decode tok/s or the 750 tok/s target.

## Delivery sequence

1. Loader smoke test and frozen reference fixtures.
2. Single-input core embedding path.
3. OpenAI-compatible endpoint and download registry.
4. Serial multi-input support and correctness validation.
5. Multi-sequence batching optimization.
6. Optional `Qwen3-Embedding-0.6B` low-memory tier using the same interface after
   the 4B acceptance gates pass.

## Sources

- Qwen official model and evaluation card:
  <https://huggingface.co/Qwen/Qwen3-Embedding-4B>
- Qwen official GGUF:
  <https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF>
- BGE-M3 model card:
  <https://huggingface.co/BAAI/bge-m3>
- Nomic Embed v1.5 model card:
  <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>
