# Important: Fix MoE Prefill Correctness Before Tara 1.5

## Problem

The Tara 1.4.1 Hugging Face model applies expert-capacity accounting across all
tokens in a prompt. Routes that exceed an expert's capacity can be dropped.

Taraference currently performs MoE prefill token by token. Each token is routed
independently, so capacity-overflow routes are never dropped. This means the HF
and Taraference hidden states can differ before the first decoded token, even
when the checkpoint, tokenizer, prompt, expert weights, and top-k setting match.

Q8 weight and activation approximations introduce additional numerical drift,
but they are separate from this routing-semantics mismatch.

## Why It Matters

- Greedy output can diverge within the first few generated tokens.
- A fast decode tok/s result does not prove correct prefill behavior.
- SFT quality can appear worse after export even when tensor conversion is valid.
- Tara 1.5 training and inference must use compatible routing semantics.

## Required Before Tara 1.5 Quality Claims

1. Decide and document the canonical inference behavior:
   - reproduce HF prompt-wide capacity and route dropping in Taraference; or
   - train Tara 1.5 with deterministic no-drop routing that matches Taraference.
2. Add an identical-prompt HF-versus-Taraference correctness test.
3. Compare prompt token IDs, selected experts, routing weights, layer outputs,
   final logits, and initial greedy tokens.
4. Test FP32/reference weights before Q8 so routing bugs and quantization drift
   can be measured separately.
5. Require coherent output before accepting performance measurements.

## Performance Expectation

Fixing prompt-wide routing may reduce prefill speed or increase TTFT. It should
not materially reduce steady-state single-token decode tok/s, because decode
already processes one token at a time. Report prefill and decode separately.

## Current Evidence (Tara 1.4.1 Idea SFT)

- Architecture: `tara_moe_141`, 12 layers, hidden size 448, four experts, top-2.
- Q8 exported-weight relative RMSE was about 0.5% with cosine similarity near
  0.99999 for inspected embedding, attention, dense FFN, and expert tensors.
- HF FP32 was already repetitive, so the base/SFT remains the primary quality
  limitation; the prefill mismatch is nevertheless a real backend correctness
  issue that must not be carried silently into Tara 1.5.
