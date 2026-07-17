//! Architecture + hyperparams from GGUF metadata (Qwen2 / Qwen3 / Qwen3.5).

use anyhow::{anyhow, Result};
use taraference_gguf::GgufFile;

/// Per-layer mixer kind for hybrid models (Qwen3.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Softmax attention + KV cache.
    FullAttention,
    /// Gated DeltaNet linear attention (fixed-size recurrent state).
    LinearAttention,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub attention_head_dim: usize,
    /// SmolLM3 omits RoPE on every fourth layer; zero means all layers use RoPE.
    pub no_rope_layer_interval: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// Rotary dims applied inside each head (partial RoPE). 0 → full head_dim.
    pub rope_dim: usize,
    /// True when full-attention Q projection is fused with a per-dim output gate
    /// (Qwen3.5: wq cols = 2 * n_head * head_dim).
    pub fused_q_gate: bool,
    /// Hybrid Gated DeltaNet params (Qwen3.5). Zero/empty when unused.
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_n_k_heads: usize,
    pub ssm_n_v_heads: usize,
    pub ssm_inner_size: usize,
    /// Per-layer kind; length == n_layer. All FullAttention for non-hybrid.
    pub layer_kinds: Vec<LayerKind>,
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = gguf
            .architecture()
            .ok_or_else(|| anyhow!("missing general.architecture"))?
            .to_string();
        let p = |s: &str| format!("{architecture}.{s}");
        let u = |key: String| -> Result<u32> {
            gguf.meta_u32(&key)
                .or_else(|| gguf.meta_u64(&key).map(|v| v as u32))
                .ok_or_else(|| anyhow!("missing {key}"))
        };
        let n_embd = u(p("embedding_length"))? as usize;
        let n_layer = u(p("block_count"))? as usize;
        let n_head = u(p("attention.head_count"))? as usize;
        let n_head_kv = gguf
            .meta_u32(&p("attention.head_count_kv"))
            .or_else(|| {
                gguf.meta_u64(&p("attention.head_count_kv"))
                    .map(|v| v as u32)
            })
            .unwrap_or(n_head as u32) as usize;
        let attention_head_dim = gguf
            .meta_u32(&p("attention.key_length"))
            .or_else(|| gguf.meta_u64(&p("attention.key_length")).map(|v| v as u32))
            .unwrap_or((n_embd / n_head) as u32) as usize;
        let value_head_dim = gguf
            .meta_u32(&p("attention.value_length"))
            .or_else(|| {
                gguf.meta_u64(&p("attention.value_length"))
                    .map(|v| v as u32)
            })
            .unwrap_or(attention_head_dim as u32) as usize;
        if value_head_dim != attention_head_dim {
            anyhow::bail!(
                "different attention key/value lengths are unsupported: key={attention_head_dim}, value={value_head_dim}"
            );
        }
        let no_rope_layer_interval = if architecture == "smollm3" { 4 } else { 0 };
        let n_ff = u(p("feed_forward_length"))? as usize;
        let n_ctx = gguf
            .meta_u32(&p("context_length"))
            .or_else(|| gguf.meta_u64(&p("context_length")).map(|v| v as u32))
            .unwrap_or(4096) as usize;
        let n_vocab = gguf
            .meta_u32(&p("vocab_size"))
            .or_else(|| gguf.meta_u64(&p("vocab_size")).map(|v| v as u32))
            .map(|v| v as usize)
            .or_else(|| {
                gguf.tensor("token_embd.weight")
                    .and_then(|t| t.dims.get(1).map(|d| *d as usize))
            })
            .ok_or_else(|| anyhow!("vocab size"))?;
        let rope_theta = gguf.meta_f32(&p("rope.freq_base")).unwrap_or(1_000_000.0);
        let rms_eps = gguf
            .meta_f32(&p("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        let is_qwen35 = architecture == "qwen35" || architecture == "qwen3_5";
        let rope_dim = if is_qwen35 {
            gguf.meta_u32(&p("rope.dimension_count"))
                .or_else(|| gguf.meta_u64(&p("rope.dimension_count")).map(|v| v as u32))
                .map(|v| v as usize)
                .unwrap_or(attention_head_dim)
        } else {
            attention_head_dim
        };

        let (ssm_conv_kernel, ssm_state_size, ssm_n_k_heads, ssm_n_v_heads, ssm_inner_size) =
            if is_qwen35 {
                let conv = gguf
                    .meta_u32(&p("ssm.conv_kernel"))
                    .or_else(|| gguf.meta_u64(&p("ssm.conv_kernel")).map(|v| v as u32))
                    .unwrap_or(4) as usize;
                let state = gguf
                    .meta_u32(&p("ssm.state_size"))
                    .or_else(|| gguf.meta_u64(&p("ssm.state_size")).map(|v| v as u32))
                    .ok_or_else(|| anyhow!("missing {}.ssm.state_size", architecture))?
                    as usize;
                let n_k = gguf
                    .meta_u32(&p("ssm.group_count"))
                    .or_else(|| gguf.meta_u64(&p("ssm.group_count")).map(|v| v as u32))
                    .ok_or_else(|| anyhow!("missing {}.ssm.group_count", architecture))?
                    as usize;
                let n_v = gguf
                    .meta_u32(&p("ssm.time_step_rank"))
                    .or_else(|| gguf.meta_u64(&p("ssm.time_step_rank")).map(|v| v as u32))
                    .ok_or_else(|| anyhow!("missing {}.ssm.time_step_rank", architecture))?
                    as usize;
                let inner = gguf
                    .meta_u32(&p("ssm.inner_size"))
                    .or_else(|| gguf.meta_u64(&p("ssm.inner_size")).map(|v| v as u32))
                    .unwrap_or((state * n_v) as u32) as usize;
                (conv, state, n_k, n_v, inner)
            } else {
                (0, 0, 0, 0, 0)
            };

        let layer_kinds = if is_qwen35 {
            let interval = gguf
                .meta_u32(&p("full_attention_interval"))
                .or_else(|| {
                    gguf.meta_u64(&p("full_attention_interval"))
                        .map(|v| v as u32)
                })
                .unwrap_or(4) as usize;
            let interval = interval.max(1);
            (0..n_layer)
                .map(|i| {
                    // Match llama.cpp: layer (i+1) % interval == 0 → full attention.
                    if (i + 1) % interval == 0 {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::LinearAttention
                    }
                })
                .collect()
        } else {
            vec![LayerKind::FullAttention; n_layer]
        };

        Ok(Self {
            architecture,
            n_vocab,
            n_embd,
            n_layer,
            n_head,
            n_head_kv,
            attention_head_dim,
            no_rope_layer_interval,
            n_ff,
            n_ctx,
            rope_theta,
            rms_eps,
            rope_dim,
            fused_q_gate: is_qwen35,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_n_k_heads,
            ssm_n_v_heads,
            ssm_inner_size,
            layer_kinds,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.attention_head_dim
    }

    pub fn is_hybrid(&self) -> bool {
        self.layer_kinds
            .iter()
            .any(|k| matches!(k, LayerKind::LinearAttention))
    }

    pub fn is_linear_layer(&self, li: usize) -> bool {
        self.layer_kinds
            .get(li)
            .map(|k| matches!(k, LayerKind::LinearAttention))
            .unwrap_or(false)
    }

    /// Q projection output width (includes fused gate when present).
    pub fn q_proj_dim(&self) -> usize {
        let base = self.n_head * self.attention_head_dim;
        if self.fused_q_gate {
            base * 2
        } else {
            base
        }
    }

    pub fn ssm_key_dim(&self) -> usize {
        self.ssm_state_size * self.ssm_n_k_heads
    }

    pub fn ssm_value_dim(&self) -> usize {
        self.ssm_state_size * self.ssm_n_v_heads
    }

    pub fn ssm_conv_channels(&self) -> usize {
        self.ssm_key_dim() * 2 + self.ssm_value_dim()
    }

    pub fn ssm_state_elems(&self) -> usize {
        // [n_v_heads, d_k, d_v] with d_k == d_v == state_size for Qwen3.5
        self.ssm_n_v_heads * self.ssm_state_size * self.ssm_state_size
    }

    pub fn ssm_conv_state_elems(&self) -> usize {
        self.ssm_conv_kernel.saturating_sub(1) * self.ssm_conv_channels()
    }
}
