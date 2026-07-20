# MoE router-weight compatibility

Tara MoE deployment packs store `router_weight_mode` in `meta.json`.

- `full_softmax`: selected expert weights retain their probabilities from the
  softmax over all experts. Tara 1.5 top-1 training uses this mode so the
  language-model task loss can train the router.
- `selected_softmax`: softmax is recomputed over only the selected top-k
  experts. This is the legacy Taraference behavior. A missing metadata field is
  interpreted as this mode for backward compatibility.

The Q8 and Q4 MoE exporters read this field from the Hugging Face
`config.json`, validate it, and preserve it in the pack metadata. Taraference
rejects unknown values rather than silently changing model math.

The deployable sparse format is currently the Tara MoE pack directory
(`meta.json` plus Q8/Q4 tensor files). `export_to_gguf.ps1` is the dense Sprint
conversion path and is not the sparse Tara 1.5 MoE exporter.

Current CUDA limits are 1--64 experts and top-k up to 8. Both the fused
RMSNorm/router/quantize path and the standalone f16-expert router implement the
same weight mode. The mode adds no kernel launch and only the full-softmax
denominator over the small expert count.

## Required checks before release

1. Exporter output reports the expected `router_weight_mode`.
2. Taraference load output reports the same `router_weights=...` value.
3. Q8 and Q4 packs both load and generate on GPU.
4. CUDA graph capture reports `OK` and subsequent tokens replay successfully.
5. An older pack without the field loads as `selected_softmax`.

