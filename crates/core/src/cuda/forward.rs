//! Chunked prefill / decode entry points.

use super::kv::CudaKv;
use super::layer::ChunkDims;
use super::model::CudaModel;
use super::types::MAX_BATCH;
use anyhow::{bail, Context, Result};

impl CudaModel {
    /// Run tokens through the model; return greedy next-token id.
    pub fn forward_greedy(&mut self, tokens: &[u32], cache: &mut CudaKv) -> Result<u32> {
        if tokens.is_empty() {
            bail!("empty tokens");
        }
        if cache.len + tokens.len() > cache.max_seq {
            bail!("context full");
        }

        let mut offset = 0usize;
        while offset < tokens.len() {
            let n = (tokens.len() - offset).min(MAX_BATCH);
            let chunk = &tokens[offset..offset + n];
            let pos0 = cache.len + offset;
            let is_last = offset + n == tokens.len();
            self.forward_chunk(chunk, pos0, is_last, cache)?;
            offset += n;
        }
        cache.len += tokens.len();

        self.stream.synchronize().context("cuda sync")?;
        let idx = self.stream.clone_dtoh(&self.argmax_buf)?;
        Ok(idx[0] as u32)
    }

    fn forward_chunk(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        compute_logits: bool,
        cache: &mut CudaKv,
    ) -> Result<()> {
        let n_tok_u = tokens.len();
        let head_dim = self.cfg.head_dim();
        let n_kv_heads = self.cfg.n_head_kv;
        let d = ChunkDims {
            n_tok: n_tok_u as i32,
            n_tok_u,
            n_embd: self.cfg.n_embd as i32,
            n_embd_u: self.cfg.n_embd,
            n_ff_u: self.cfg.n_ff,
            n_head: self.cfg.n_head as i32,
            n_head_u: self.cfg.n_head,
            n_kv: n_kv_heads as i32,
            n_kv_heads,
            head_dim,
            hd: head_dim as i32,
            stride: (n_kv_heads * head_dim) as i32,
            stride_u: n_kv_heads * head_dim,
            pos0,
            pos0_i: pos0 as i32,
            eps: self.cfg.rms_eps,
            theta: self.cfg.rope_theta,
            scale: (head_dim as f32).sqrt().recip(),
        };

        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        self.embed_batch(&ids)?;

        for li in 0..self.layers.len() {
            self.run_layer(li, &d, cache)?;
        }
        if compute_logits {
            self.logits_from_last(&d)?;
        }
        Ok(())
    }
}
