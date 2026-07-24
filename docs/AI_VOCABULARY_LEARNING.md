# AI Vocabulary & Research Glossary

A learning-first dictionary of terms you will see when studying **large language models (LLMs)**, **transformers**, **attention**, and **local inference engines** (like taraference / `tarafer`).

Use this as a **vocab sheet**, not a full textbook. Read a section, then look terms up again when you hit them in papers, blogs, or code.

**Suggested learning path**

1. [Foundations](#1-foundations) → what AI / ML / deep learning are  
2. [Neural network building blocks](#2-neural-network-building-blocks)  
3. [Language modeling basics](#3-language-modeling-basics)  
4. [Transformers](#4-transformers)  
5. [Attention (deep dive)](#5-attention-deep-dive)  
6. [Inference vs training](#6-inference-vs-training)  
7. [Sampling & generation](#7-sampling--generation)  
8. [Quantization & formats](#8-quantization--formats)  
9. [Hardware & CUDA](#9-hardware--cuda)  
10. [Mixture of Experts & hybrids](#10-mixture-of-experts--hybrids)  
11. [Evaluation & metrics](#11-evaluation--metrics)  
12. [Common model names & families](#12-common-model-names--families)  
13. [Quick A–Z index](#13-quick-az-index)

Related project docs: [GOAL.md](../GOAL.md), [WORKFLOW.md](../WORKFLOW.md), [docs/](./).

---

## 1. Foundations

| Term | Plain meaning |
|------|----------------|
| **AI (Artificial Intelligence)** | Broad field: machines that perform tasks that look “smart” (language, vision, planning). |
| **ML (Machine Learning)** | Subset of AI: systems that learn patterns from data instead of only hand-written rules. |
| **Deep Learning** | ML with **neural networks** that have many layers (often millions–billions of parameters). |
| **Model** | The learned (or randomly initialized) function: weights + architecture that map inputs → outputs. |
| **Parameters / Weights** | Numbers inside the model that were set by training. More parameters often means more capacity (and cost). |
| **Architecture** | The blueprint: how layers connect (e.g. transformer with GQA + SwiGLU). Weights fill in that blueprint. |
| **Training** | Adjusting weights so the model gets better at a task (usually next-token prediction for LLMs). |
| **Inference** | Running a trained model to produce outputs. No weight updates. What `tarafer` does. |
| **Dataset** | Collection of examples used to train or evaluate a model. |
| **Benchmark** | Standard test suite used to compare models (MMLU, HumanEval, etc.) or engines (tok/s). |
| **Generalization** | Doing well on new inputs not seen exactly during training. |
| **Overfitting** | Memorizing training data too closely; poor on new data. |
| **Underfitting** | Model too weak or under-trained; fails even on training data. |
| **Supervised learning** | Learn from labeled input→output pairs. |
| **Unsupervised learning** | Learn structure without explicit labels (clustering, some pretraining). |
| **Self-supervised learning** | Labels come from the data itself (e.g. predict the next word). How LLMs are pre-trained. |
| **Fine-tuning (SFT)** | Further training a pretrained model on a narrower task or instruction data. |
| **RLHF** | Reinforcement Learning from Human Feedback — align model outputs with human preferences. |
| **Alignment** | Making model behavior safer / more helpful / closer to intended use. |
| **Foundation model** | Large pretrained model usable as a base for many downstream tasks. |
| **LLM (Large Language Model)** | Transformer (usually) trained on text to predict / generate language at large scale. |
| **Multimodal** | Handles more than one modality (text + image + audio, etc.). |
| **Agent** | System that uses a model to take multi-step actions (tools, browse, code) toward a goal. |
| **Prompt** | The input text (and sometimes system instructions) given to the model. |
| **Context / Context window** | How much text (in tokens) the model can “see” at once. |
| **Token** | A chunk of text the model reads/writes (word piece, subword, or byte). Not always one word. |
| **Tokenizer** | Algorithm + vocabulary that converts text ↔ token IDs. |
| **Vocabulary (vocab)** | The set of all tokens the tokenizer/model knows (often 32k–200k+). |
| **Embedding** | Dense vector representation of a token (or other object) the network can compute with. |
| **Logits** | Raw unnormalized scores the model outputs for each next-token candidate (before softmax). |
| **Probability distribution** | Softmax of logits: each token gets a probability that sums to 1. |
| **Hallucination** | Confident-looking output that is factually wrong or invented. |
| **Temperature** | Sampling knob: higher → more random; lower → more peaked / greedy-like. |
| **Open weights** | Model weights publicly downloadable (may still have license restrictions). |
| **Closed / proprietary model** | Weights not released; only API access. |

---

## 2. Neural network building blocks

| Term | Plain meaning |
|------|----------------|
| **Neuron / Unit** | A simple compute node: weighted sum + nonlinearity (loosely analogous to a brain cell). |
| **Layer** | A group of units applied together (linear, attention, norm, etc.). |
| **MLP / FFN / Feed-forward network** | Dense layers applied independently per position after attention in a transformer block. |
| **Linear layer / Dense / Fully connected / Projection** | Matrix multiply \(y = xW + b\) (bias optional). The workhorse of transformers. |
| **Activation function** | Nonlinearity (ReLU, GELU, SiLU/Swish, etc.) that lets networks model complex functions. |
| **ReLU** | \(\max(0, x)\) — classic cheap activation. |
| **GELU** | Smooth activation used in many transformers (GPT-2 era and after). |
| **SiLU / Swish** | \(x \cdot \sigma(x)\); used in modern LLMs, often inside SwiGLU. |
| **SwiGLU** | Gated FFN: combine gate and up projections with SiLU; common in LLaMA/Qwen-style models. |
| **Gate / Gating** | Multiply one path by a learned “how much to pass” signal (elementwise). |
| **Residual connection / Skip connection** | Add the block input back to its output: \(x + f(x)\). Helps deep nets train and run stably. |
| **Normalization** | Rescale activations so layers stay numerically well-behaved. |
| **LayerNorm** | Normalize across features for each token; older transformer default. |
| **RMSNorm** | Root-mean-square norm (no mean subtraction). Common in LLaMA, Qwen, etc. |
| **BatchNorm** | Normalize across batch dimension — rare in modern LLM transformers. |
| **Dropout** | Randomly zero activations during training to regularize; usually off at inference. |
| **Bias** | Additive term in a linear layer. Some modern models are bias-free in many projections. |
| **Gradient** | How much loss changes if you nudge a weight — used by optimizers during training. |
| **Backpropagation** | Algorithm that computes gradients through the network efficiently. |
| **Optimizer** | Method that updates weights (SGD, Adam, AdamW, …). |
| **Loss / Objective** | Scalar the training process minimizes (e.g. cross-entropy of next token). |
| **Cross-entropy** | Standard classification loss; for LLMs, next-token prediction. |
| **Epoch** | One full pass over the training dataset. |
| **Batch / Mini-batch** | Group of examples processed together. |
| **Learning rate** | Step size for weight updates. |
| **Regularization** | Techniques to reduce overfitting (weight decay, dropout, …). |
| **Initialization** | How weights are set before training (important for stability). |
| **Tensor** | Multi-dimensional array (scalar, vector, matrix, higher rank). Core data type in DL. |
| **Shape / Rank / Dimension** | Tensor size metadata, e.g. `[batch, seq, hidden]`. |
| **Broadcasting** | Rules for combining tensors of compatible shapes without explicit replication. |
| **dtype** | Numeric type: `f32`, `f16`, `bf16`, `int8`, … |

---

## 3. Language modeling basics

| Term | Plain meaning |
|------|----------------|
| **Language model (LM)** | Model that assigns probabilities to sequences of tokens / predicts the next token. |
| **Autoregressive (AR)** | Generate one token at a time; each new token depends on previous ones. Standard chat LLMs. |
| **Causal LM** | AR model that can only attend leftward (past tokens), not future tokens. |
| **Masked LM (MLM)** | Predict masked tokens using both sides (BERT-style). Not the usual chat generator. |
| **Pretraining** | Large-scale self-supervised training (next token / span prediction) on huge corpora. |
| **Continued pretraining** | More pretraining on domain data after the original pretrain. |
| **Instruction tuning** | Fine-tune so the model follows user instructions in chat form. |
| **Chat template / ChatML** | Formatting rules that wrap system/user/assistant turns for the model. |
| **System prompt** | High-priority instructions that set role, style, safety, tools, etc. |
| **Few-shot / Zero-shot / Many-shot** | How many examples you put in the prompt before asking the real question. |
| **In-context learning (ICL)** | Model “learns” a pattern from examples inside the prompt without weight updates. |
| **BPE (Byte Pair Encoding)** | Popular subword tokenization: merges frequent byte/char pairs into tokens. |
| **SentencePiece** | Tokenization library/format used by many models (often Unigram or BPE). |
| **Special tokens** | Non-text control tokens: BOS, EOS, PAD, IM_START, tool markers, etc. |
| **BOS / EOS** | Beginning / end of sequence tokens. |
| **UNK** | Unknown token for characters outside vocab (modern BPE often avoids this). |
| **Byte-level BPE** | Tokenizes from raw bytes so almost any text can be encoded. |
| **Context length / Max sequence length** | Hard (or soft) limit on tokens in one forward pass / KV cache. |
| **Long context** | Training or techniques that extend usable context (128k, 1M, …). |
| **Position encoding** | How the model knows token order (absolute, relative, RoPE, ALiBi, …). |
| **Perplexity (PPL)** | How “surprised” the model is by text; lower is better for modeling quality. |
| **N-gram** | Fixed window of N tokens; classic pre-neural LM idea. |
| **Next-token prediction** | Core LM training task: given past tokens, predict the next. |

---

## 4. Transformers

The **Transformer** (Vaswani et al., 2017, *Attention Is All You Need*) replaced recurrence with attention as the main sequence mixer.

| Term | Plain meaning |
|------|----------------|
| **Transformer** | Architecture of stacked blocks: usually attention + FFN + norms + residuals. |
| **Encoder** | Bidirectional stack (sees full sequence). Used in BERT, translation encoders. |
| **Decoder** | Causal stack for generation. Chat LLMs are mostly **decoder-only**. |
| **Encoder–decoder** | Seq2seq (T5, classic translation): encoder reads input, decoder generates output. |
| **Decoder-only** | Single causal stack (GPT, LLaMA, Qwen, most chat models). |
| **Block / Layer (transformer)** | One repeated unit: (optional) pre-norm → attention → residual → pre-norm → FFN → residual. |
| **Pre-norm vs Post-norm** | Whether LayerNorm/RMSNorm is before or after the sublayer. Modern LLMs usually pre-norm. |
| **Hidden size / Model dim (\(d_{model}\))** | Width of the residual stream (e.g. 2048, 4096). |
| **Intermediate size / FFN dim** | Expanded width inside the FFN (often ~2.5–4× hidden). |
| **Number of layers (\(n_{layer}\))** | Depth of the stack. |
| **Head count (\(n_{head}\))** | Number of attention heads for queries. |
| **Head dim (\(d_{head}\))** | Usually \(d_{model} / n_{head}\). Size of each head’s Q/K/V slice. |
| **Tied embeddings** | Input embedding matrix shared (transposed) with output LM head to save params. |
| **LM head / Output projection** | Final linear map from hidden state → vocab logits. |
| **Weight tying** | Sharing parameters across two places (often embed ↔ LM head). |
| **Parallel residual / Parallel attention+FFN** | Some designs run attn and FFN in parallel from the same input (e.g. some GPT-J style). |
| **Scaling laws** | Empirical rules relating loss to model size, data size, and compute. |
| **Chinchilla scaling** | Finding that many models were under-trained on data relative to parameter count. |

### Projections inside one attention layer

| Term | Plain meaning |
|------|----------------|
| **Q / Query** | “What am I looking for?” vectors for each token. |
| **K / Key** | “What do I contain?” vectors used for matching against queries. |
| **V / Value** | “What information do I pass?” vectors mixed according to attention weights. |
| **O / Output projection** | Linear map after concatenating heads back to \(d_{model}\). |
| **QKV projection** | The three (or fused) linear layers that produce Q, K, V from hidden states. |
| **Fused QKV** | One big matmul that produces Q, K, V together for speed. |

---

## 5. Attention (deep dive)

Attention lets each token build a **weighted combination** of other tokens’ values, with weights from how well queries match keys.

### Core math (conceptual)

For one head, with queries \(Q\), keys \(K\), values \(V\):

\[
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

| Term | Plain meaning |
|------|----------------|
| **Attention** | Mechanism that mixes information across sequence positions using similarity of Q and K. |
| **Self-attention** | Q, K, V all come from the **same** sequence (the model’s own tokens). |
| **Cross-attention** | Q from one sequence, K/V from another (decoder attending to encoder). |
| **Scaled dot-product attention** | The classic formula above: scores = Q·K / √d, then softmax, then weighted V. |
| **Attention scores / logits (attn)** | Pre-softmax similarities between each query position and every key position. |
| **Attention weights / map** | Post-softmax probabilities over keys for each query (rows sum to 1). |
| **Softmax** | Turns a vector of scores into a probability distribution (non-negative, sum to 1). |
| **Scaling factor \(\sqrt{d_k}\)** | Prevents dot products from growing too large with head dimension (stabilizes softmax). |
| **Multi-Head Attention (MHA)** | Several attention heads in parallel, each with its own Q/K/V, then concatenated. |
| **Head** | One independent attention channel with its own projection slice. |
| **Multi-Query Attention (MQA)** | Many Q heads share **one** K and **one** V head → much smaller KV cache. |
| **Grouped-Query Attention (GQA)** | Compromise: groups of Q heads share a K/V head. Common in LLaMA-2/3, Qwen, etc. |
| **KV heads / `n_kv_head`** | Number of key/value heads (≤ query heads in GQA/MQA). |
| **Causal mask / Causal attention** | Block looking at future positions so token \(t\) only sees ≤ \(t\). Required for AR decode. |
| **Attention mask** | General 0/1 (or −inf) pattern: padding, causal, sliding window, document boundaries, etc. |
| **Padding mask** | Ignore pad tokens so they don’t get attention mass. |
| **Full attention / Dense attention** | Every query can attend to all allowed keys (quadratic cost in sequence length). |
| **Quadratic complexity** | Standard attention cost/memory ~ \(O(n^2)\) in sequence length \(n\). |
| **Sliding window attention** | Each token only attends within a local window (e.g. Mistral-style). |
| **Sparse attention** | Only some position pairs are computed (patterns, blocks, learned sparsity). |
| **Global tokens** | Special positions that can attend broadly in sparse schemes. |
| **Flash Attention** | IO-aware exact (or near-exact) attention algorithm that reduces HBM traffic → much faster/less memory. |
| **Flash decode** | Decode-oriented flash-style kernels optimized for **one (or few) new queries** vs long KV. |
| **Paged Attention** | KV cache stored in non-contiguous pages (vLLM-style) for flexible batching / less fragmentation. |
| **Memory-efficient attention** | Family of algorithms that avoid materializing full \(n \times n\) matrices. |
| **Online softmax** | Numerically stable streaming softmax used inside flash-style kernels. |
| **Attention sink** | Phenomenon/technique: early tokens keep high attention mass; sometimes kept explicitly for long context. |
| **RoPE (Rotary Position Embedding)** | Rotate Q/K by position-dependent angles so attention becomes relative-position aware. |
| **Partial RoPE** | Only some dimensions of each head are rotated. |
| **NTK-aware / YaRN / Position interpolation** | Methods to extend RoPE models beyond original training context. |
| **ALiBi** | Attention with Linear Biases: add distance-based bias instead of RoPE. |
| **Absolute positional embedding** | Learned or sinusoidal vectors added to token embeddings by index (original transformer). |
| **Relative positional embedding** | Encode “how far apart” tokens are rather than absolute index. |
| **QK-Norm** | Normalize Q and/or K before attention scores (stability in some modern models). |
| **Attention dropout** | Dropout on attention weights during training. |
| **Softmax temperature (attn)** | Extra temperature on attention scores (distinct from sampling temperature). |
| **Gated attention / Attention output gate** | Extra gate on attention outputs (seen in some architectures). |
| **Linear attention** | Reformulations that avoid full \(n \times n\) with kernel feature maps / recurrent forms. |
| **State-space models (SSM)** | Sequence models with fixed-size recurrent state (Mamba, etc.) — alternative/complement to softmax attention. |
| **DeltaNet / Gated DeltaNet** | Linear-attention / recurrent mixing families; taraference has hybrid support ideas here. |
| **Hybrid attention** | Mix of full softmax layers and linear/SSM layers in one stack. |

### Prefill, decode, and the KV cache

| Term | Plain meaning |
|------|----------------|
| **Prefill** | Process the whole prompt at once: compute K/V for all prompt tokens, get first new logits. |
| **Decode** | Generate tokens one-by-one (or in small speculative chunks), reusing cached K/V. |
| **KV cache** | Stored keys and values from past tokens so decode doesn’t recompute full attention history. |
| **KV cache growth** | Cache size grows with **sequence length × layers × KV heads × head dim × dtype**. |
| **Context drop / long-context slowdown** | Decode tok/s falls as KV grows (more memory traffic for attention). |
| **Incremental decoding** | Synonym-ish for autoregressive decode with cache. |
| **Static cache vs dynamic cache** | Preallocated max-length buffer vs grow-as-you-go. |
| **Cache quantization** | Store K/V in lower precision (int8/fp8) to save memory/bandwidth. |
| **Prefix caching / prompt cache** | Reuse KV for a shared system prompt or repeated prefix across requests. |
| **TTFT (Time To First Token)** | Latency until the first generated token (dominated by prefill for long prompts). |
| **TPOT / Inter-token latency** | Time between successive generated tokens (decode). |
| **Throughput (tok/s)** | Tokens generated per second. Define carefully: single-stream vs aggregate. |
| **Single-stream decode** | One user/request path — taraference’s primary optimization target. |
| **Batching** | Multiple sequences processed together for GPU utilization. |
| **Continuous batching** | Server inserts/removes sequences as they finish (vLLM-style serving). |
| **Speculative decoding** | Draft model proposes several tokens; target model verifies in parallel → higher accepted tok/s. |
| **EAGLE / MTP** | Multi-token prediction / draft-head families for speculative speedups. |
| **Accepted tokens** | Tokens that pass verification in speculative setups (or simply emitted tokens). |
| **Tree attention / tree decoding** | Verify multiple draft branches in one pass. |

---

## 6. Inference vs training

| Term | Plain meaning |
|------|----------------|
| **Forward pass** | Compute outputs from inputs given current weights. |
| **Backward pass** | Compute gradients (training only). |
| **Inference engine** | Software that loads weights and runs forward passes efficiently (llama.cpp, vLLM, taraference, …). |
| **Serving** | Running models as a network service for many clients. |
| **Offline inference** | Batch jobs without interactive latency constraints. |
| **Online / interactive inference** | Chat-like latency matters (TTFT + smooth decode). |
| **Graph compilation / CUDA Graphs** | Record a fixed GPU launch sequence and replay it with less CPU overhead. |
| **Kernel** | A GPU function (CUDA) that runs many threads in parallel. |
| **Fused kernel** | One kernel that does multiple math steps (e.g. RMSNorm + residual + quantize). |
| **Operator fusion** | Compiler/runtime combining ops to cut memory round-trips. |
| **Memory-bound vs compute-bound** | Whether runtime is limited by bandwidth or by FLOPs. Decode is often memory-bound. |
| **Arithmetic intensity** | FLOPs per byte moved — high means more compute-bound friendly. |
| **Roofline model** | Visual model of whether a kernel is bandwidth- or compute-limited. |
| **GEMM** | General matrix multiply (big batches / prefill). |
| **GEMV** | Matrix–vector multiply (classic single-token decode for dense layers). |
| **MMVQ / quantized GEMV** | Fast matmul paths where weights are quantized and activations may be too. |
| **Tensor Cores** | NVIDIA hardware units for mixed-precision matrix math (e.g. FP16/BF16/INT8/FP8). |
| **DP4A** | Integer instruction: 4× int8 dot-accumulate — used in some fast quant decode kernels. |
| **Marlin / AWQ / GPTQ kernels** | Specialized high-performance quantized inference kernels/methods. |
| **Weight-only quantization** | Weights low-precision; activations higher precision at runtime. |
| **Activation quantization** | Also quantize activations (harder; dynamic scales often needed). |
| **KV-aware / attention-optimized path** | Kernels specialized for long KV during decode. |
| **Deterministic vs nondeterministic** | Same inputs always same outputs vs floating-point/race variability. |
| **Numerical stability** | Avoiding overflow/underflow/NaNs (softmax, norms, low-precision). |
| **bf16 / fp16 / tf32** | Reduced-precision float formats used for speed on modern GPUs. |
| **Mixed precision** | Different dtypes in different parts of the network. |

---

## 7. Sampling & generation

After the model produces **logits**, sampling decides the next token ID.

| Term | Plain meaning |
|------|----------------|
| **Greedy decoding** | Always pick \(\arg\max\) token. Deterministic, can be repetitive. |
| **Sampling** | Draw a token from a distribution derived from logits. |
| **Temperature** | Divide logits by \(T\) before softmax. \(T\to 0\) → greedy-like; \(T>1\) → flatter. |
| **Top-k** | Keep only the \(k\) highest-prob tokens, renorm, sample. |
| **Top-p / Nucleus sampling** | Keep the smallest set of tokens whose cumulative prob ≥ \(p\), then sample. |
| **Min-p** | Filter tokens below a fraction of the top token’s probability. |
| **Typical sampling** | Prefer tokens with information content near the expected entropy. |
| **Repetition penalty** | Down-weight tokens already used to reduce loops. |
| **Frequency / presence penalty** | OpenAI-style penalties for repeated or already-present tokens. |
| **Stop sequences** | Strings/token patterns that end generation early. |
| **Max new tokens** | Cap on how many tokens to generate. |
| **Beam search** | Keep multiple partial hypotheses (more common in translation than open chat). |
| **JSON / grammar / constrained decoding** | Force outputs to match a schema or CFG. |
| **Logprobs** | Log probabilities of chosen (or top) tokens — useful for scoring. |
| **Seed** | RNG seed for reproducible sampling. |
| **Streaming** | Emit tokens to the user as soon as each is produced. |
| **Chat completion** | API-style: messages in, assistant message out. |
| **Completion / continuation** | Raw prompt continuation without chat roles. |
| **Teacher forcing** | Training technique: feed gold previous tokens (not the model’s own samples). |

---

## 8. Quantization & formats

| Term | Plain meaning |
|------|----------------|
| **Quantization** | Represent numbers with fewer bits to save memory and bandwidth (sometimes speed). |
| **Precision** | Bit width / numeric format of weights or activations. |
| **fp32 / float32** | 32-bit float — accurate, heavy. |
| **fp16 / float16** | 16-bit float — faster/lighter, narrower range. |
| **bf16 / bfloat16** | 16-bit float with fp32-like range, less mantissa precision. |
| **int8 / int4** | Integer formats for aggressive compression. |
| **fp8** | 8-bit float formats (E4M3/E5M2) on newer GPUs. |
| **Block-wise / group-wise quant** | Scale factors per block of weights (e.g. 32 or 256 elements). |
| **GGUF** | Widely used file format for local llama.cpp-style models (metadata + tensors). |
| **GGML type / quant type** | e.g. `Q4_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0` — different quality/speed tradeoffs. |
| **Q4_K_M** | Popular 4-bit K-quant “medium” variant; common local default (also taraference scoreboard). |
| **K-quants** | GGML quantization family with super-blocks and mixed precision tricks. |
| **Dequantization** | Expand quantized weights back toward higher precision for math (or fuse into kernels). |
| **Calibration** | Data pass used by some methods (GPTQ/AWQ) to choose scales/zeros well. |
| **GPTQ** | Post-training weight quantization method (layer-wise Hessian-ish approximation). |
| **AWQ** | Activation-aware weight quantization — protect salient weights. |
| **SmoothQuant** | Migrate activation difficulty into weights for easier INT8. |
| **Safetensors** | Safe, fast tensor serialization format (common for HF weights). |
| **PyTorch `.pt` / `.bin`** | Older weight dumps (pickle risks on untrusted `.pt`). |
| **ONNX** | Portable graph format for cross-runtime inference. |
| **Packing** | Bit-pack low-precision weights tightly in memory for bandwidth. |
| **Scale / zero-point** | Affine dequant parameters: \(x \approx scale \cdot (q - zero)\). |
| **Per-channel scales** | Separate scale per output channel of a weight matrix. |
| **Dynamic quant** | Scales computed at runtime from activation statistics. |
| **Static quant** | Scales fixed after calibration. |

---

## 9. Hardware & CUDA

| Term | Plain meaning |
|------|----------------|
| **GPU** | Graphics Processing Unit — massively parallel processor ideal for matmuls. |
| **CPU offload** | Keep some layers/weights on CPU RAM when VRAM is tight (slower). |
| **VRAM / HBM** | GPU memory. KV cache + weights must fit (or stream). |
| **Bandwidth** | How fast memory can be read/written — often the decode bottleneck. |
| **FLOP / FLOPs** | Floating-point operations (count of math work). |
| **TFLOPS** | Trillion FLOPs per second — peak compute rating. |
| **CUDA** | NVIDIA’s parallel computing platform/API. |
| **CUDA core** | General-purpose GPU ALUs. |
| **SM (Streaming Multiprocessor)** | GPU building block that runs warps. |
| **sm_75, sm_86, …** | Compute capability codes (T4=75, RTX 3050 Ti laptop=86, …). |
| **Warp** | Group of 32 CUDA threads that execute in lockstep. |
| **Block / Grid** | Hierarchy of thread groups for a kernel launch. |
| **Shared memory** | Fast on-SM memory shared by threads in a block. |
| **Registers** | Fastest per-thread storage; limited. |
| **HBM / global memory** | Large off-SM GPU memory. |
| **Occupancy** | How many warps an SM can host relative to hardware limits. |
| **Kernel launch overhead** | CPU/GPU cost to start a kernel — CUDA graphs reduce this. |
| **NVRTC** | NVIDIA Runtime Compilation — compile CUDA C++ to PTX/SASS at runtime. |
| **PTX** | NVIDIA virtual ISA intermediate. |
| **SASS** | Actual GPU machine code for a given architecture. |
| **Stream / Event** | CUDA concurrency and timing primitives. |
| **Synchronization (`cudaDeviceSynchronize`)** | Wait for GPU work to finish — can kill tok/s if overused. |
| **PCIe** | Bus between CPU and GPU; host↔device copies are expensive. |
| **Unified memory** | Automatically migrated CPU/GPU memory (convenient, not always fastest). |
| **Multi-GPU / Tensor Parallel (TP)** | Split one model across GPUs (columns/rows of matrices). |
| **Pipeline Parallel (PP)** | Split layers across GPUs. |
| **Data Parallel (DP)** | Same model replica per GPU, different batch shards. |
| **NCCL** | NVIDIA library for multi-GPU collectives. |
| **nvidia-smi** | CLI to inspect GPU util, memory, clocks, processes. |
| **Thermal throttling** | GPU slows clocks when hot — confuses benchmarks on laptops. |
| **Clock locking** | Fix clocks for more repeatable benchmarks. |

---

## 10. Mixture of Experts & hybrids

| Term | Plain meaning |
|------|----------------|
| **MoE (Mixture of Experts)** | FFN replaced by many expert FFNs; a router picks a few per token. |
| **Expert** | One FFN specialist inside an MoE layer. |
| **Router / Gating network** | Small network that scores experts for each token. |
| **Top-k routing** | Activate the \(k\) highest-scoring experts (e.g. top-2). |
| **Sparse MoE** | Only a few experts run per token → more params without full dense FLOPs. |
| **Dense model** | Every parameter participates every token (no expert sparsity). |
| **Load balancing loss** | Training term so experts get used evenly. |
| **Expert parallelism** | Place different experts on different devices. |
| **Shared expert** | Expert always-on in some MoE designs (e.g. DeepSeek-style variants). |
| **Router weight mode** | How selected expert probs are normalized (project-specific: selected vs full softmax). |
| **Prefill routing** | Which experts run during the prompt pass — must match training/serving semantics. |
| **Hybrid model** | Mix of layer types (full attention + linear attention/SSM, or dense + MoE). |

---

## 11. Evaluation & metrics

| Term | Plain meaning |
|------|----------------|
| **Accuracy** | Fraction correct on a labeled task. |
| **Exact match (EM)** | Output string matches the gold answer exactly. |
| **F1** | Harmonic mean of precision and recall (token/span overlap tasks). |
| **BLEU / ROUGE** | N-gram overlap metrics for translation/summarization (imperfect for open chat). |
| **MMLU** | Multitask knowledge / exam-style benchmark. |
| **HumanEval / MBPP** | Code generation benchmarks. |
| **Arena / ELO / LMSYS** | Human preference battles between models. |
| **Win rate** | How often judges prefer model A over B. |
| **Latency** | Wall time for a request (or until first/next token). |
| **Throughput** | Work per unit time (tokens/s, requests/s). |
| **`overall_decode_tps`** | End-to-end decode tokens per second (project metric style). |
| **Goodput** | Useful throughput after accounting for rejects/retries/failures. |
| **Correctness gate** | “Fast but wrong” doesn’t count — outputs and KV state must be valid. |
| **Regression test** | Ensure a change didn’t break speed or quality vs a known baseline. |
| **A/B test** | Compare two implementations under matched conditions. |
| **Baseline** | Reference measurement you try to beat. |
| **Scoreboard model** | Fixed model used for fair speed comparisons (here: Qwen2.5-3B Q4_K_M). |
| **Ablation** | Remove/change one component to measure its contribution. |
| **Wall clock** | Real elapsed time (includes all overheads). |
| **Profiler** | Tool that attributes time to kernels/ops (`ncu`, Nsight, custom timers). |
| **Trace** | Timeline of GPU/CPU events. |

---

## 12. Common model names & families

| Name / family | What people mean |
|---------------|------------------|
| **GPT** | Generative Pre-trained Transformer; also OpenAI product line. |
| **BERT** | Bidirectional encoder LM (MLM); not a casual chat generator. |
| **T5 / UL2** | Encoder–decoder text-to-text models. |
| **LLaMA / Llama** | Meta’s influential open-ish decoder-only family (RMSNorm, SwiGLU, RoPE, GQA in later versions). |
| **Mistral / Mixtral** | Efficient dense / MoE models; sliding window in early Mistral. |
| **Qwen / Qwen2.5 / Qwen3** | Alibaba model family widely used for local inference and research. |
| **DeepSeek** | Strong open models; notable MoE and reasoning variants. |
| **Gemma** | Google open model family. |
| **Phi** | Microsoft small capable models. |
| **SmolLM** | Small efficient models (some skip RoPE on certain layers). |
| **Mamba** | Selective SSM architecture (linear-time sequence modeling). |
| **Transformer-XL / XLNet / Reformer / Performer / Longformer** | Historical long-context / efficient-attention lines — good paper vocab. |
| **Diffusion LM** | Non-AR generation paradigms (less common for chat today). |
| **Vision Transformer (ViT)** | Transformer on image patches. |
| **CLIP** | Joint vision–language embeddings. |
| **Whisper** | Speech-to-text transformer family. |

### Papers & ideas worth knowing by name

| Name | Why it matters |
|------|----------------|
| **Attention Is All You Need (2017)** | Introduced the Transformer. |
| **GPT / GPT-2 / GPT-3 papers** | Scaled decoder-only LMs + in-context learning. |
| **BERT** | Bidirectional pretraining era. |
| **RoPE paper** | Rotary embeddings used almost everywhere now. |
| **FlashAttention / FlashAttention-2/3** | Practical exact attention at long context. |
| **GQA paper** | Grouped-query attention for cheaper inference. |
| **LoRA** | Low-rank adapters for cheap fine-tuning. |
| **QLoRA** | LoRA on quantized base weights. |
| **RLHF / InstructGPT** | Preference alignment stack popularized for assistants. |
| **Constitutional AI / DPO / GRPO / PPO** | Alignment / preference optimization methods. |
| **Speculative decoding papers** | Lossless (or controlled) speedups via draft+verify. |
| **vLLM / PagedAttention** | High-throughput serving ideas. |
| **llama.cpp / GGML / GGUF** | Ubiquitous local inference stack & formats. |

---

## 13. Quick A–Z index

Jump terms when you only remember the acronym.

| | | |
|--|--|--|
| **ALiBi** | **AR** autoregressive | **AWQ** |
| **BOS/EOS** | **BPE** | **bf16** |
| **Causal mask** | **ChatML** | **Context window** |
| **Cross-attention** | **CUDA / CUDA Graph** | **Decode** |
| **d_model / d_head** | **DP4A** | **dtype** |
| **Embedding** | **EAGLE** | **Expert / MoE** |
| **FFN / MLP** | **Flash Attention** | **FLOPs** |
| **fp16/fp32/fp8** | **Fused kernel** | **Gating / SwiGLU** |
| **GEMM / GEMV** | **GGUF** | **GQA / MQA / MHA** |
| **Greedy** | **Hallucination** | **HBM / VRAM** |
| **Head / head_dim** | **Hidden size** | **Hybrid model** |
| **ICL** | **Inference** | **int4/int8** |
| **KV cache** | **K-quant / Q4_K_M** | **Kernel** |
| **LayerNorm / RMSNorm** | **Linear attention** | **LLM** |
| **LoRA / QLoRA** | **Logits / logprobs** | **LM head** |
| **Mamba / SSM** | **Marlin** | **Masked LM** |
| **MHA** | **Mixed precision** | **MLP** |
| **MoE** | **MQA** | **MTP** |
| **n_head / n_kv_head** | **Nucleus (top-p)** | **NVRTC** |
| **Online softmax** | **Paged Attention** | **Parameter** |
| **Perplexity** | **Position encoding** | **Prefill** |
| **Pre-norm** | **Prompt** | **PTX** |
| **Q/K/V/O** | **QLoRA** | **Quantization** |
| **Residual** | **RLHF** | **RoPE** |
| **Router** | **Safetensors** | **Sampling** |
| **Self-attention** | **SFT** | **SiLU / GELU / ReLU** |
| **sm_XX** | **Softmax** | **Speculative decoding** |
| **Streaming** | **SwiGLU** | **System prompt** |
| **Temperature** | **Tensor** | **Tensor Core** |
| **Tokenizer / token** | **Top-k / top-p** | **Transformer** |
| **TTFT** | **Throughput (tok/s)** | **ViT** |
| **Warp** | **Weight** | **Window attention** |

---

## 14. “Explain it like I’m new” — attention in one page

1. Text is cut into **tokens** and each becomes a vector (**embedding**).  
2. Vectors flow through many **layers**. In each layer, **self-attention** lets every token look at other tokens.  
3. For each token, the model builds a **query** (“what do I need?”), and every token offers a **key** (“what do I have?”) and a **value** (“what content can I give?”).  
4. Similarity of query·key → **scores** → **softmax** → **weights**.  
5. The token’s new representation is a **weighted sum of values**.  
6. Multiple **heads** do this in parallel with different projections (different “viewpoints”).  
7. An **FFN** then processes each position independently.  
8. **Residuals + RMSNorm** keep the deep stack stable.  
9. At the end, the **LM head** turns the last hidden state into **logits** over the vocabulary.  
10. **Sampling** picks the next token; append it; repeat (**decode**), reusing the **KV cache**.

That loop is modern chat AI in miniature.

---

## 15. Terms you’ll see specifically in this repo

These appear in taraference code and docs — good “local dialect” on top of general AI vocab.

| Term | In this project |
|------|-----------------|
| **taraference** | The repo / core library for CUDA GGUF inference. |
| **tarafer** | CLI binary name. |
| **single-stream accepted tok/s** | North-star speed metric for one user. |
| **scoreboard model** | `Qwen2.5-3B-Instruct-Q4_K_M.gguf` for official speed claims. |
| **fast_v2 / flash decode** | Optimized attention decode kernels under `crates/core/src/cuda/kernels/attn/`. |
| **GQA** | Read from GGUF metadata (`attention.head_count` vs `attention.head_count_kv`). |
| **RoPE / partial RoPE** | Config fields for rotary dims and layer skipping (e.g. SmolLM3-style). |
| **LayerKind** | Full softmax attention vs linear (Gated DeltaNet) layers. |
| **RouterWeightMode** | MoE router probability handling (`selected_softmax` vs `full_softmax`). |
| **GGUF reader** | `crates/gguf` parses model files into tensors + metadata. |
| **Session / TurnStats** | Multi-turn chat state, timings, stop reasons. |
| **SamplingOptions** | Temperature, top-p, penalties, etc. in `sampler.rs`. |
| **CUDA graph** | Replay fixed decode path with less launch overhead. |
| **NVRTC arch** | Runtime compile targeting detected `sm_XX`. |
| **Prefill correctness** | Especially important for MoE: wrong expert routing → fluent garbage. |
| **Profile / A/B** | Measure decode tok/s under matched conditions (see GOAL.md / WORKFLOW.md). |

---

## 16. Study checklist (use this to self-test)

Can you explain each in one or two sentences?

- [ ] Token vs word vs embedding  
- [ ] Logits vs probabilities vs sampling  
- [ ] Self-attention vs cross-attention  
- [ ] Why divide by \(\sqrt{d_k}\)?  
- [ ] Why causal mask for chat models?  
- [ ] MHA vs GQA vs MQA (and effect on KV cache)  
- [ ] What RoPE is doing to Q and K  
- [ ] Prefill vs decode  
- [ ] What the KV cache stores and why it grows  
- [ ] Why decode is often memory-bound  
- [ ] What quantization trades away  
- [ ] What MoE saves and what it risks  
- [ ] TTFT vs inter-token latency vs tok/s  
- [ ] Single-stream vs aggregate throughput  
- [ ] What Flash Attention optimizes (hint: memory traffic)  

If you can teach those out loud, you have the core vocabulary to read inference code and modern LLM papers.

---

## 17. Further learning (starting points)

- Original transformer: *Attention Is All You Need*  
- Illustrated guides: “The Illustrated Transformer”, “The Illustrated GPT-2” (Jay Alammar)  
- Attention kernels: FlashAttention papers & blogs  
- Local inference formats: GGUF / llama.cpp docs  
- Serving systems: vLLM PagedAttention notes  
- This repo: [GOAL.md](../GOAL.md) for what “fast” means here; `crates/core/src/` for a real engine layout  

---

*This file is a living study sheet. Add terms as you meet them in papers, Discord, or code reviews — vocabulary is a superpower in AI research and systems work.*
