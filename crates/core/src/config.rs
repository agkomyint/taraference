//! Architecture + hyperparams from GGUF metadata (Qwen2 / Llama-style).

use anyhow::{anyhow, Result};
use taraference_gguf::GgufFile;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
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
            .or_else(|| gguf.meta_u64(&p("attention.head_count_kv")).map(|v| v as u32))
            .unwrap_or(n_head as u32) as usize;
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
        Ok(Self {
            architecture,
            n_vocab,
            n_embd,
            n_layer,
            n_head,
            n_head_kv,
            n_ff,
            n_ctx,
            rope_theta,
            rms_eps,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}
