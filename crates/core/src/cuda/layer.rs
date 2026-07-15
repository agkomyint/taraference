//! One transformer block + final logits head.

use super::decode::DecodeBackend;
use super::kv::CudaKv;
use super::matmul::{gemm, gemv};
use super::model::CudaModel;
use anyhow::Result;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

/// Dimensions for one forward chunk (prefill or decode).
pub(crate) struct ChunkDims {
    pub n_tok: i32,
    pub n_tok_u: usize,
    pub n_embd: i32,
    pub n_embd_u: usize,
    pub n_ff_u: usize,
    pub n_head: i32,
    pub n_head_u: usize,
    pub n_kv: i32,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hd: i32,
    pub stride: i32,
    pub stride_u: usize,
    pub pos0: usize,
    pub pos0_i: i32,
    pub eps: f32,
    pub theta: f32,
    pub scale: f32,
}

impl CudaModel {
    pub(crate) fn run_layer(
        &mut self,
        li: usize,
        d: &ChunkDims,
        cache: &mut CudaKv,
    ) -> Result<()> {
        self.attn_block(li, d, cache)?;
        self.ffn_block(li, d)?;
        Ok(())
    }

    fn attn_block(&mut self, li: usize, d: &ChunkDims, cache: &mut CudaKv) -> Result<()> {
        unsafe {
            self.stream
                .launch_builder(&self.k.rms_norm)
                .arg(&self.x)
                .arg(&self.layers[li].attn_norm)
                .arg(&mut self.xb)
                .arg(&d.n_embd)
                .arg(&d.n_tok)
                .arg(&d.eps)
                .launch(LaunchConfig {
                    grid_dim: (d.n_tok_u as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        // Decode: fuse bias into GEMV (saves 3 launches/layer). Prefill: GEMM + bias.
        if d.n_tok == 1 {
            gemv(
                &self.stream,
                &self.k,
                &self.layers[li].wq,
                &self.xb,
                &mut self.q,
                self.layers[li].bq.as_ref(),
                None,
            )?;
            gemv(
                &self.stream,
                &self.k,
                &self.layers[li].wk,
                &self.xb,
                &mut self.k_buf,
                self.layers[li].bk.as_ref(),
                None,
            )?;
            gemv(
                &self.stream,
                &self.k,
                &self.layers[li].wv,
                &self.xb,
                &mut self.v_buf,
                self.layers[li].bv.as_ref(),
                None,
            )?;
        } else {
            gemm(
                &self.stream,
                &self.k,
                &self.layers[li].wq,
                &self.xb,
                &mut self.q,
                d.n_tok,
            )?;
            if let Some(ref b) = self.layers[li].bq {
                let feat = d.n_embd;
                unsafe {
                    self.stream
                        .launch_builder(&self.k.add_bias)
                        .arg(&mut self.q)
                        .arg(b)
                        .arg(&feat)
                        .arg(&d.n_tok)
                        .launch(LaunchConfig::for_num_elems((d.n_embd_u * d.n_tok_u) as u32))?;
                }
            }
            gemm(
                &self.stream,
                &self.k,
                &self.layers[li].wk,
                &self.xb,
                &mut self.k_buf,
                d.n_tok,
            )?;
            if let Some(ref b) = self.layers[li].bk {
                let feat = (d.n_kv_heads * d.head_dim) as i32;
                unsafe {
                    self.stream
                        .launch_builder(&self.k.add_bias)
                        .arg(&mut self.k_buf)
                        .arg(b)
                        .arg(&feat)
                        .arg(&d.n_tok)
                        .launch(LaunchConfig::for_num_elems(
                            (d.n_kv_heads * d.head_dim * d.n_tok_u) as u32,
                        ))?;
                }
            }
            gemm(
                &self.stream,
                &self.k,
                &self.layers[li].wv,
                &self.xb,
                &mut self.v_buf,
                d.n_tok,
            )?;
            if let Some(ref b) = self.layers[li].bv {
                let feat = (d.n_kv_heads * d.head_dim) as i32;
                unsafe {
                    self.stream
                        .launch_builder(&self.k.add_bias)
                        .arg(&mut self.v_buf)
                        .arg(b)
                        .arg(&feat)
                        .arg(&d.n_tok)
                        .launch(LaunchConfig::for_num_elems(
                            (d.n_kv_heads * d.head_dim * d.n_tok_u) as u32,
                        ))?;
                }
            }
        }

        unsafe {
            self.stream
                .launch_builder(&self.k.rope)
                .arg(&mut self.q)
                .arg(&d.n_head)
                .arg(&d.hd)
                .arg(&d.pos0_i)
                .arg(&d.n_tok)
                .arg(&d.theta)
                .launch(LaunchConfig {
                    grid_dim: (d.n_head_u as u32, d.n_tok_u as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            self.stream
                .launch_builder(&self.k.rope)
                .arg(&mut self.k_buf)
                .arg(&d.n_kv)
                .arg(&d.hd)
                .arg(&d.pos0_i)
                .arg(&d.n_tok)
                .arg(&d.theta)
                .launch(LaunchConfig {
                    grid_dim: (d.n_kv_heads as u32, d.n_tok_u as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            let kv_elems = (d.n_tok_u * d.stride_u) as u32;
            self.stream
                .launch_builder(&self.k.copy_kv)
                .arg(&self.k_buf)
                .arg(&mut cache.k[li])
                .arg(&d.pos0_i)
                .arg(&d.n_tok)
                .arg(&d.stride)
                .launch(LaunchConfig::for_num_elems(kv_elems))?;
            self.stream
                .launch_builder(&self.k.copy_kv)
                .arg(&self.v_buf)
                .arg(&mut cache.v[li])
                .arg(&d.pos0_i)
                .arg(&d.n_tok)
                .arg(&d.stride)
                .launch(LaunchConfig::for_num_elems(kv_elems))?;
        }

        self.launch_attn(li, d, cache)?;

        if d.n_tok == 1 {
            gemv(
                &self.stream,
                &self.k,
                &self.layers[li].wo,
                &self.xb,
                &mut self.xb2,
                None,
                None,
            )?;
        } else {
            gemm(
                &self.stream,
                &self.k,
                &self.layers[li].wo,
                &self.xb,
                &mut self.xb2,
                d.n_tok,
            )?;
        }
        let residual_n = (d.n_embd_u * d.n_tok_u) as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k.add)
                .arg(&mut self.x)
                .arg(&self.xb2)
                .arg(&residual_n)
                .launch(LaunchConfig::for_num_elems((d.n_embd_u * d.n_tok_u) as u32))?;
        }
        Ok(())
    }

    fn ffn_block(&mut self, li: usize, d: &ChunkDims) -> Result<()> {
        unsafe {
            self.stream
                .launch_builder(&self.k.rms_norm)
                .arg(&self.x)
                .arg(&self.layers[li].ffn_norm)
                .arg(&mut self.xb)
                .arg(&d.n_embd)
                .arg(&d.n_tok)
                .arg(&d.eps)
                .launch(LaunchConfig {
                    grid_dim: (d.n_tok_u as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        if d.n_tok == 1 {
            gemv(
                &self.stream,
                &self.k,
                &self.layers[li].gate,
                &self.xb,
                &mut self.hb,
                None,
                None,
            )?;
            gemv(
                &self.stream,
                &self.k,
                &self.layers[li].up,
                &self.xb,
                &mut self.hb2,
                None,
                None,
            )?;
        } else {
            gemm(
                &self.stream,
                &self.k,
                &self.layers[li].gate,
                &self.xb,
                &mut self.hb,
                d.n_tok,
            )?;
            gemm(
                &self.stream,
                &self.k,
                &self.layers[li].up,
                &self.xb,
                &mut self.hb2,
                d.n_tok,
            )?;
        }
        let ff_n = (d.n_ff_u * d.n_tok_u) as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k.silu_mul)
                .arg(&mut self.hb)
                .arg(&self.hb2)
                .arg(&ff_n)
                .launch(LaunchConfig::for_num_elems((d.n_ff_u * d.n_tok_u) as u32))?;
        }
        if d.n_tok == 1 {
            gemv(
                &self.stream,
                &self.k,
                &self.layers[li].down,
                &self.hb,
                &mut self.xb2,
                None,
                None,
            )?;
        } else {
            gemm(
                &self.stream,
                &self.k,
                &self.layers[li].down,
                &self.hb,
                &mut self.xb2,
                d.n_tok,
            )?;
        }
        let residual_n = (d.n_embd_u * d.n_tok_u) as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k.add)
                .arg(&mut self.x)
                .arg(&self.xb2)
                .arg(&residual_n)
                .launch(LaunchConfig::for_num_elems((d.n_embd_u * d.n_tok_u) as u32))?;
        }
        Ok(())
    }

    /// Dispatch attention kernel based on [`DecodeBackend`].
    fn launch_attn(&mut self, li: usize, d: &ChunkDims, cache: &mut CudaKv) -> Result<()> {
        let seq_len = d.pos0 + d.n_tok_u;

        // Online path only for single-token decode; multi-token uses fast.
        let use_online =
            self.decode == DecodeBackend::Online && d.n_tok == 1 && d.head_dim <= 256;

        if use_online {
            let seq_len_i = seq_len as i32;
            let smem = (d.head_dim * 2 * 4) as u32; // qh + red
            unsafe {
                self.stream
                    .launch_builder(&self.k.attn_online)
                    .arg(&self.q)
                    .arg(&cache.k[li])
                    .arg(&cache.v[li])
                    .arg(&mut self.xb)
                    .arg(&d.n_head)
                    .arg(&d.n_kv)
                    .arg(&d.hd)
                    .arg(&seq_len_i)
                    .arg(&d.scale)
                    .launch(LaunchConfig {
                        grid_dim: (d.n_head_u as u32, 1, 1),
                        block_dim: (d.head_dim as u32, 1, 1),
                        shared_mem_bytes: smem,
                    })?;
            }
            return Ok(());
        }

        let f: &CudaFunction = match self.decode {
            DecodeBackend::Basic => &self.k.attn_basic,
            DecodeBackend::Fast | DecodeBackend::Online => &self.k.attn_fast,
        };

        // fast: Q[head_dim] + scores[ATTN_TILE=64] (fixed; does not grow with ctx)
        // basic: scores[seq_len] baseline (A/B only)
        const ATTN_TILE: usize = 64;
        let smem = match self.decode {
            DecodeBackend::Basic => (seq_len.max(1) * 4) as u32,
            DecodeBackend::Fast | DecodeBackend::Online => {
                ((d.head_dim + ATTN_TILE) * 4) as u32
            }
        };
        let attn_threads = 128u32;

        unsafe {
            self.stream
                .launch_builder(f)
                .arg(&self.q)
                .arg(&cache.k[li])
                .arg(&cache.v[li])
                .arg(&mut self.xb)
                .arg(&d.n_head)
                .arg(&d.n_kv)
                .arg(&d.hd)
                .arg(&d.pos0_i)
                .arg(&d.n_tok)
                .arg(&d.scale)
                .launch(LaunchConfig {
                    grid_dim: (d.n_head_u as u32, d.n_tok_u as u32, 1),
                    block_dim: (attn_threads, 1, 1),
                    shared_mem_bytes: smem,
                })?;
        }
        Ok(())
    }

    pub(crate) fn logits_from_last(&mut self, d: &ChunkDims) -> Result<()> {
        unsafe {
            self.stream
                .launch_builder(&self.k.copy_last)
                .arg(&self.x)
                .arg(&mut self.x1)
                .arg(&d.n_tok)
                .arg(&d.n_embd)
                .launch(LaunchConfig::for_num_elems(d.n_embd_u as u32))?;
            let one = 1i32;
            self.stream
                .launch_builder(&self.k.rms_norm)
                .arg(&self.x1)
                .arg(&self.output_norm)
                .arg(&mut self.xb1)
                .arg(&d.n_embd)
                .arg(&one)
                .arg(&d.eps)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        if let Some(ref ow) = self.output {
            gemv(&self.stream, &self.k, ow, &self.xb1, &mut self.logits, None, None)?;
        } else {
            gemv(
                &self.stream,
                &self.k,
                &self.token_embd,
                &self.xb1,
                &mut self.logits,
                None,
                None,
            )?;
        }
        let n_vocab = self.cfg.n_vocab as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k.argmax)
                .arg(&self.logits)
                .arg(&n_vocab)
                .arg(&mut self.argmax_buf)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}
