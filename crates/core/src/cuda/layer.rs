//! One transformer block + final logits head.

use super::decode::{find_by_name, AttnLaunch};
use super::kv::CudaKv;
use super::matmul::{
    gemm, gemv, try_gemv_global_q8, try_gemv_pair, try_gemv_q4_dual, try_gemv_q4_ffn, try_gemv_qkv,
    GemvResidual,
};
use super::model::CudaModel;
use anyhow::Result;
use cudarc::driver::{LaunchConfig, PushKernelArg};

fn layer_timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("TARAFER_LAYER_TIMING").is_some())
}

fn decode_should_skip_ffn(layer: usize, n_layer: usize) -> bool {
    static SPEC: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(spec) = SPEC.get_or_init(|| std::env::var("TARAFER_FFN_SKIP").ok()) else {
        return false;
    };
    if let Ok(stride) = spec.parse::<usize>() {
        return stride >= 2 && (layer + 1) % stride == 0;
    }
    if let Some(count) = spec
        .strip_prefix("middle:")
        .and_then(|s| s.parse::<usize>().ok())
    {
        let count = count.min(n_layer);
        let start = (n_layer - count) / 2;
        return layer >= start && layer < start + count;
    }
    if let Some(mask) = spec.strip_prefix("mask:") {
        return mask
            .split(',')
            .filter_map(|s| s.parse::<usize>().ok())
            .any(|one_based| one_based == layer + 1);
    }
    false
}

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
    /// Read pos0 from `CudaModel::d_pos0` (CUDA-graph safe single-token path).
    pub use_device_pos: bool,
}

impl CudaModel {
    pub(crate) fn run_layer(&mut self, li: usize, d: &ChunkDims, cache: &mut CudaKv) -> Result<()> {
        let timing = layer_timing_enabled() && d.n_tok == 1;
        let start = if timing {
            Some(
                self.stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?,
            )
        } else {
            None
        };
        self.attn_block(li, d, cache)?;
        let middle = if timing {
            Some(
                self.stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?,
            )
        } else {
            None
        };
        let skip_ffn = d.n_tok == 1 && decode_should_skip_ffn(li, self.layers.len());
        if !skip_ffn {
            self.ffn_block(li, d)?;
        }
        if let (Some(start), Some(middle)) = (start, middle) {
            let end = self
                .stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            eprintln!(
                "layer_timing | layer={li} attn_ms={:.4} ffn_ms={:.4}",
                start.elapsed_ms(&middle)?,
                middle.elapsed_ms(&end)?
            );
        }
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

        // Decode: fuse bias into GEMV. Prefill: GEMM + bias.
        // Prefer Q5_0 Q+K+V fuse; else Q+K pair + separate V; else three GEMVs.
        if d.n_tok == 1 {
            if try_gemv_qkv(
                &self.stream,
                &self.k,
                &self.layers[li].wq,
                &self.layers[li].wk,
                &self.layers[li].wv,
                &self.xb,
                &mut self.q,
                &mut self.k_buf,
                &mut self.v_buf,
                self.layers[li].bq.as_ref(),
                self.layers[li].bk.as_ref(),
                self.layers[li].bv.as_ref(),
                &mut self.q8_x,
                &mut self.q8_d,
            )? {
                // fused Q+K+V
            } else if try_gemv_pair(
                &self.stream,
                &self.k,
                &self.layers[li].wq,
                &self.layers[li].wk,
                &self.xb,
                &mut self.q,
                &mut self.k_buf,
                self.layers[li].bq.as_ref(),
                self.layers[li].bk.as_ref(),
                &mut self.q8_x,
                &mut self.q8_d,
            )? {
                gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].wv,
                    &self.xb,
                    &mut self.v_buf,
                    self.layers[li].bv.as_ref(),
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
            } else {
                gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].wq,
                    &self.xb,
                    &mut self.q,
                    self.layers[li].bq.as_ref(),
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
                gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].wk,
                    &self.xb,
                    &mut self.k_buf,
                    self.layers[li].bk.as_ref(),
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
                gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].wv,
                    &self.xb,
                    &mut self.v_buf,
                    self.layers[li].bv.as_ref(),
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
            }
        } else {
            gemm(
                &self.stream,
                &self.k,
                &self.layers[li].wq,
                &self.xb,
                &mut self.q,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
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
                &mut self.gemv_partial,
                self.gemv_partial_stride,
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
                &mut self.gemv_partial,
                self.gemv_partial_stride,
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
            let q_norm = self.layers[li].attn_q_norm.as_ref();
            let k_norm = self.layers[li].attn_k_norm.as_ref();
            let use_rope = self.cfg.no_rope_layer_interval == 0
                || (li + 1) % self.cfg.no_rope_layer_interval != 0;
            if let (Some(qw), Some(kw)) = (q_norm, k_norm) {
                let launch = LaunchConfig {
                    grid_dim: (d.n_head_u as u32, d.n_tok_u as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                if d.use_device_pos {
                    self.stream
                        .launch_builder(&self.k.qk_norm_rope_d)
                        .arg(&mut self.q)
                        .arg(qw)
                        .arg(&d.n_head)
                        .arg(&d.hd)
                        .arg(&self.d_pos0)
                        .arg(&d.n_tok)
                        .arg(&d.theta)
                        .arg(&d.eps)
                        .launch(launch)?;
                    self.stream
                        .launch_builder(&self.k.qk_norm_rope_d)
                        .arg(&mut self.k_buf)
                        .arg(kw)
                        .arg(&d.n_kv)
                        .arg(&d.hd)
                        .arg(&self.d_pos0)
                        .arg(&d.n_tok)
                        .arg(&d.theta)
                        .arg(&d.eps)
                        .launch(LaunchConfig {
                            grid_dim: (d.n_kv_heads as u32, d.n_tok_u as u32, 1),
                            ..launch
                        })?;
                } else {
                    self.stream
                        .launch_builder(&self.k.qk_norm_rope)
                        .arg(&mut self.q)
                        .arg(qw)
                        .arg(&d.n_head)
                        .arg(&d.hd)
                        .arg(&d.pos0_i)
                        .arg(&d.n_tok)
                        .arg(&d.theta)
                        .arg(&d.eps)
                        .launch(launch)?;
                    self.stream
                        .launch_builder(&self.k.qk_norm_rope)
                        .arg(&mut self.k_buf)
                        .arg(kw)
                        .arg(&d.n_kv)
                        .arg(&d.hd)
                        .arg(&d.pos0_i)
                        .arg(&d.n_tok)
                        .arg(&d.theta)
                        .arg(&d.eps)
                        .launch(LaunchConfig {
                            grid_dim: (d.n_kv_heads as u32, d.n_tok_u as u32, 1),
                            ..launch
                        })?;
                }
            } else if d.use_device_pos {
                if use_rope {
                    self.stream
                        .launch_builder(&self.k.rope_d)
                        .arg(&mut self.q)
                        .arg(&d.n_head)
                        .arg(&d.hd)
                        .arg(&self.d_pos0)
                        .arg(&d.n_tok)
                        .arg(&d.theta)
                        .launch(LaunchConfig {
                            grid_dim: (d.n_head_u as u32, d.n_tok_u as u32, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    self.stream
                        .launch_builder(&self.k.rope_d)
                        .arg(&mut self.k_buf)
                        .arg(&d.n_kv)
                        .arg(&d.hd)
                        .arg(&self.d_pos0)
                        .arg(&d.n_tok)
                        .arg(&d.theta)
                        .launch(LaunchConfig {
                            grid_dim: (d.n_kv_heads as u32, d.n_tok_u as u32, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
            } else {
                if use_rope {
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
                }
            }

            if d.use_device_pos {
                let kv_elems = (d.n_tok_u * d.stride_u) as u32;
                self.stream
                    .launch_builder(&self.k.copy_kv_d)
                    .arg(&self.k_buf)
                    .arg(&mut cache.k[li])
                    .arg(&self.d_pos0)
                    .arg(&d.n_tok)
                    .arg(&d.stride)
                    .launch(LaunchConfig::for_num_elems(kv_elems))?;
                self.stream
                    .launch_builder(&self.k.copy_kv_d)
                    .arg(&self.v_buf)
                    .arg(&mut cache.v[li])
                    .arg(&self.d_pos0)
                    .arg(&d.n_tok)
                    .arg(&d.stride)
                    .launch(LaunchConfig::for_num_elems(kv_elems))?;
            } else {
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
        }

        self.launch_attn(li, d, cache)?;

        if d.n_tok == 1 {
            // Decode: x = x + Wo·attn  (fuse residual add into GEMV).
            if !try_gemv_global_q8(
                &self.stream,
                &self.k,
                &self.layers[li].wo,
                &self.xb,
                &mut self.x,
                None,
                GemvResidual::InPlace,
                &mut self.q8_x,
                &mut self.q8_d,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )? {
                gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].wo,
                    &self.xb,
                    &mut self.x,
                    None,
                    GemvResidual::InPlace,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
            }
        } else {
            gemm(
                &self.stream,
                &self.k,
                &self.layers[li].wo,
                &self.xb,
                &mut self.xb2,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
            let residual_n = (d.n_embd_u * d.n_tok_u) as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k.add)
                    .arg(&mut self.x)
                    .arg(&self.xb2)
                    .arg(&residual_n)
                    .launch(LaunchConfig::for_num_elems((d.n_embd_u * d.n_tok_u) as u32))?;
            }
        }
        Ok(())
    }

    fn ffn_block(&mut self, li: usize, d: &ChunkDims) -> Result<()> {
        let timing = layer_timing_enabled() && d.n_tok == 1;
        let t0 = if timing {
            Some(
                self.stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?,
            )
        } else {
            None
        };
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
        let t1 = if timing {
            Some(
                self.stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?,
            )
        } else {
            None
        };
        let mut fused_ffn = false;
        if d.n_tok == 1 {
            fused_ffn = try_gemv_q4_ffn(
                &self.stream,
                &self.k,
                &self.layers[li].gate,
                &self.layers[li].up,
                &self.xb,
                &mut self.hb,
                &mut self.q8_x,
                &mut self.q8_d,
            )?;
            // gate+up often same quant (Q5_0 or Q4_K) → stage xb once.
            if !fused_ffn
                && !try_gemv_q4_dual(
                    &self.stream,
                    &self.k,
                    &self.layers[li].gate,
                    &self.layers[li].up,
                    &self.xb,
                    &mut self.hb,
                    &mut self.hb2,
                    &mut self.q8_x,
                    &mut self.q8_d,
                )?
                && !try_gemv_pair(
                    &self.stream,
                    &self.k,
                    &self.layers[li].gate,
                    &self.layers[li].up,
                    &self.xb,
                    &mut self.hb,
                    &mut self.hb2,
                    None,
                    None,
                    &mut self.q8_x,
                    &mut self.q8_d,
                )?
            {
                gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].gate,
                    &self.xb,
                    &mut self.hb,
                    None,
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
                gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].up,
                    &self.xb,
                    &mut self.hb2,
                    None,
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
            }
        } else {
            gemm(
                &self.stream,
                &self.k,
                &self.layers[li].gate,
                &self.xb,
                &mut self.hb,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
            gemm(
                &self.stream,
                &self.k,
                &self.layers[li].up,
                &self.xb,
                &mut self.hb2,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
        }
        let t2 = if timing {
            Some(
                self.stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?,
            )
        } else {
            None
        };
        if !fused_ffn {
            let ff_n = (d.n_ff_u * d.n_tok_u) as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k.silu_mul)
                    .arg(&mut self.hb)
                    .arg(&self.hb2)
                    .arg(&ff_n)
                    .launch(LaunchConfig::for_num_elems((d.n_ff_u * d.n_tok_u) as u32))?;
            }
        }
        let t3 = if timing {
            Some(
                self.stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?,
            )
        } else {
            None
        };
        if d.n_tok == 1 {
            // Decode: x = x + Wdown·(silu(gate)⊙up)  (fuse residual add into GEMV).
            if !try_gemv_global_q8(
                &self.stream,
                &self.k,
                &self.layers[li].down,
                &self.hb,
                &mut self.x,
                None,
                GemvResidual::InPlace,
                &mut self.q8_x,
                &mut self.q8_d,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )? {
                gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].down,
                    &self.hb,
                    &mut self.x,
                    None,
                    GemvResidual::InPlace,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
            }
        } else {
            gemm(
                &self.stream,
                &self.k,
                &self.layers[li].down,
                &self.hb,
                &mut self.xb2,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
            let residual_n = (d.n_embd_u * d.n_tok_u) as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k.add)
                    .arg(&mut self.x)
                    .arg(&self.xb2)
                    .arg(&residual_n)
                    .launch(LaunchConfig::for_num_elems((d.n_embd_u * d.n_tok_u) as u32))?;
            }
        }
        if let (Some(t0), Some(t1), Some(t2), Some(t3)) = (t0, t1, t2, t3) {
            let t4 = self
                .stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            eprintln!(
                "ffn_timing | layer={li} norm_ms={:.4} gate_up_ms={:.4} act_ms={:.4} down_ms={:.4}",
                t0.elapsed_ms(&t1)?,
                t1.elapsed_ms(&t2)?,
                t2.elapsed_ms(&t3)?,
                t3.elapsed_ms(&t4)?
            );
        }
        Ok(())
    }

    /// Dispatch attention from [`DecodeBackend`] registry (no per-backend match arms).
    fn launch_attn(&mut self, li: usize, d: &ChunkDims, cache: &mut CudaKv) -> Result<()> {
        self.launch_attn_spec(self.decode.spec(), li, d, cache)
    }

    fn launch_attn_spec(
        &mut self,
        spec: &super::decode::DecodeSpec,
        li: usize,
        d: &ChunkDims,
        cache: &mut CudaKv,
    ) -> Result<()> {
        let seq_len = d.pos0 + d.n_tok_u;

        match spec.launch {
            AttnLaunch::Causal {
                kernel,
                kernel_d,
                smem,
                block_threads,
            } => {
                let smem_bytes = smem.bytes(d.head_dim, seq_len);
                if d.use_device_pos {
                    let kd = kernel_d.ok_or_else(|| {
                        anyhow::anyhow!("decode backend has no device-pos kernel for graphs")
                    })?;
                    let f = self.k.attn(kd)?;
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
                            .arg(&self.d_pos0)
                            .arg(&d.n_tok)
                            .arg(&d.scale)
                            .launch(LaunchConfig {
                                grid_dim: (d.n_head_u as u32, d.n_tok_u as u32, 1),
                                block_dim: (block_threads, 1, 1),
                                shared_mem_bytes: smem_bytes,
                            })?;
                    }
                } else {
                    let f = self.k.attn(kernel)?;
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
                                block_dim: (block_threads, 1, 1),
                                shared_mem_bytes: smem_bytes,
                            })?;
                    }
                }
                Ok(())
            }
            AttnLaunch::Flash {
                partial,
                partial_d,
                reduce,
                smem,
                block_threads,
                prefill_as,
                n_split,
            } => {
                // Prefill multi-token: use fastv2 causal.
                if d.n_tok != 1 {
                    let fb = find_by_name(prefill_as).ok_or_else(|| {
                        anyhow::anyhow!("flash prefill_as={prefill_as:?} missing from REGISTRY")
                    })?;
                    return self.launch_attn_spec(fb, li, d, cache);
                }
                let smem_bytes = smem.bytes(d.head_dim, seq_len);
                let n_split_i = n_split as i32;
                let one = 1i32;
                if d.use_device_pos {
                    let f = self.k.attn(partial_d)?;
                    unsafe {
                        self.stream
                            .launch_builder(f)
                            .arg(&self.q)
                            .arg(&cache.k[li])
                            .arg(&cache.v[li])
                            .arg(&mut self.flash_partial)
                            .arg(&d.n_head)
                            .arg(&d.n_kv)
                            .arg(&d.hd)
                            .arg(&self.d_pos0)
                            .arg(&one)
                            .arg(&n_split_i)
                            .arg(&d.scale)
                            .launch(LaunchConfig {
                                grid_dim: (d.n_head_u as u32, n_split, 1),
                                block_dim: (block_threads, 1, 1),
                                shared_mem_bytes: smem_bytes,
                            })?;
                    }
                } else {
                    let f = self.k.attn(partial)?;
                    unsafe {
                        self.stream
                            .launch_builder(f)
                            .arg(&self.q)
                            .arg(&cache.k[li])
                            .arg(&cache.v[li])
                            .arg(&mut self.flash_partial)
                            .arg(&d.n_head)
                            .arg(&d.n_kv)
                            .arg(&d.hd)
                            .arg(&d.pos0_i)
                            .arg(&one)
                            .arg(&n_split_i)
                            .arg(&d.scale)
                            .launch(LaunchConfig {
                                grid_dim: (d.n_head_u as u32, n_split, 1),
                                block_dim: (block_threads, 1, 1),
                                shared_mem_bytes: smem_bytes,
                            })?;
                    }
                }
                let fr = self.k.attn(reduce)?;
                unsafe {
                    self.stream
                        .launch_builder(fr)
                        .arg(&self.flash_partial)
                        .arg(&mut self.xb)
                        .arg(&d.n_head)
                        .arg(&d.hd)
                        .arg(&n_split_i)
                        .arg(&one)
                        .launch(LaunchConfig {
                            grid_dim: (d.n_head_u as u32, 1, 1),
                            block_dim: (d.head_dim as u32, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
                Ok(())
            }
        }
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
        use super::matmul::try_gemv_global_q8;
        if let Some(ref ow) = self.output {
            if !try_gemv_global_q8(
                &self.stream,
                &self.k,
                ow,
                &self.xb1,
                &mut self.logits,
                None,
                GemvResidual::None,
                &mut self.q8_x,
                &mut self.q8_d,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )? {
                gemv(
                    &self.stream,
                    &self.k,
                    ow,
                    &self.xb1,
                    &mut self.logits,
                    None,
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
            }
        } else {
            if !try_gemv_global_q8(
                &self.stream,
                &self.k,
                &self.token_embd,
                &self.xb1,
                &mut self.logits,
                None,
                GemvResidual::None,
                &mut self.q8_x,
                &mut self.q8_d,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )? {
                gemv(
                    &self.stream,
                    &self.k,
                    &self.token_embd,
                    &self.xb1,
                    &mut self.logits,
                    None,
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
            }
        }
        if let Some(ref special) = self.output_special {
            let active = self
                .output
                .as_ref()
                .map_or(self.token_embd.n_cols, |m| m.n_cols);
            if !try_gemv_global_q8(
                &self.stream,
                &self.k,
                special,
                &self.xb1,
                &mut self.special_logit,
                None,
                GemvResidual::None,
                &mut self.q8_x,
                &mut self.q8_d,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )? {
                gemv(
                    &self.stream,
                    &self.k,
                    special,
                    &self.xb1,
                    &mut self.special_logit,
                    None,
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
            }
            let mut destination = self.logits.slice_mut(active..active + 1);
            self.stream
                .memcpy_dtod(&self.special_logit, &mut destination)?;
        }
        let n_vocab = (self
            .output
            .as_ref()
            .map_or(self.token_embd.n_cols, |m| m.n_cols)
            + usize::from(self.output_special.is_some())) as i32;
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
