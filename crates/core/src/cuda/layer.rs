//! One transformer block + final logits head.

use super::decode::{find_by_name, AttnLaunch};
use super::kv::CudaKv;
use super::matmul::{
    gemm, gemv, try_gemv_gdn_4way, try_gemv_global_q8, try_gemv_pair, try_gemv_q4_dual, try_gemv_q4_ffn, try_gemv_qkv,
    GemvResidual,
};
use super::model::CudaModel;
use super::types::{FullAttnWeights, LayerAttn, LinearAttnWeights};
use anyhow::Result;
use cudarc::driver::{LaunchConfig, PushKernelArg};

fn layer_timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("TARAFER_LAYER_TIMING").is_some())
}

fn flag_identity() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("TARAFER_IDENTITY").is_some())
}
fn flag_skip_linear() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("TARAFER_SKIP_LINEAR").is_some())
}
fn flag_no_out_gate() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("TARAFER_NO_OUT_GATE").is_some())
}
fn flag_q_gate_half() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("TARAFER_Q_GATE_HALF").is_some())
}
fn flag_no_rope() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("TARAFER_NO_ROPE").is_some())
}
fn flag_full_rope() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("TARAFER_FULL_ROPE").is_some())
}
fn flag_gdn_legacy() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("TARAFER_GDN_LEGACY").is_some())
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
        // Debug: skip entire transformer body (identity residual).
        if !flag_identity() {
            self.attn_block(li, d, cache)?;
            let middle = if timing {
                Some(
                    self.stream.record_event(Some(
                        cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                    ))?,
                )
            } else {
                None
            };
            let skip_ffn = d.n_tok == 1 && decode_should_skip_ffn(li, self.layers.len());
            if !skip_ffn {
                self.ffn_block(li, d)?;
            }
            if let (Some(start), Some(middle)) = (start, middle) {
                let end = self.stream.record_event(Some(
                    cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                ))?;
                eprintln!(
                    "layer_timing | layer={li} attn_ms={:.4} ffn_ms={:.4}",
                    start.elapsed_ms(&middle)?,
                    middle.elapsed_ms(&end)?
                );
            }
            return Ok(());
        }
        // identity path: leave residual stream unchanged
        let _ = (li, d, cache, start);
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
        // Layer storage is stable for this call; raw pointer avoids borrow conflicts
        // with gemv/gemm mutably borrowing `self`.
        let layer_ptr = &self.layers[li] as *const super::types::GpuLayer;
        match unsafe { &(*layer_ptr).attn } {
            LayerAttn::Full(full) => self.attn_block_full(li, d, cache, full),
            LayerAttn::Linear(lin) => self.attn_block_linear(li, d, cache, lin),
        }
    }

    fn attn_block_full(
        &mut self,
        li: usize,
        d: &ChunkDims,
        cache: &mut CudaKv,
        full: &FullAttnWeights,
    ) -> Result<()> {
        let fused = full.fused_q_gate;
        let apply_out_gate = fused && !flag_no_out_gate();

        if d.n_tok == 1 {
            if try_gemv_qkv(
                    &self.stream,
                    &self.k,
                    &full.wq,
                    &full.wk,
                    &full.wv,
                    &self.xb,
                    &mut self.q,
                    &mut self.k_buf,
                    &mut self.v_buf,
                    full.bq.as_ref(),
                    full.bk.as_ref(),
                    full.bv.as_ref(),
                    &mut self.q8_x,
                    &mut self.q8_d,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?
            {
                // fused Q+K+V
            } else if try_gemv_pair(
                    &self.stream,
                    &self.k,
                    &full.wq,
                    &full.wk,
                    &self.xb,
                    &mut self.q,
                    &mut self.k_buf,
                    full.bq.as_ref(),
                    full.bk.as_ref(),
                    &mut self.q8_x,
                    &mut self.q8_d,
                )?
            {
                gemv(
                    &self.stream,
                    &self.k,
                    &full.wv,
                    &self.xb,
                    &mut self.v_buf,
                    full.bv.as_ref(),
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
            } else {
                gemv(
                    &self.stream,
                    &self.k,
                    &full.wq,
                    &self.xb,
                    &mut self.q,
                    full.bq.as_ref(),
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
                gemv(
                    &self.stream,
                    &self.k,
                    &full.wk,
                    &self.xb,
                    &mut self.k_buf,
                    full.bk.as_ref(),
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
                gemv(
                    &self.stream,
                    &self.k,
                    &full.wv,
                    &self.xb,
                    &mut self.v_buf,
                    full.bv.as_ref(),
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
            }
        } else {
            gemm(
                &self.stream,
                &self.k,
                &full.wq,
                &self.xb,
                &mut self.q,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
            if let Some(ref b) = full.bq {
                let feat = full.wq.n_cols as i32;
                unsafe {
                    self.stream
                        .launch_builder(&self.k.add_bias)
                        .arg(&mut self.q)
                        .arg(b)
                        .arg(&feat)
                        .arg(&d.n_tok)
                        .launch(LaunchConfig::for_num_elems(
                            (full.wq.n_cols * d.n_tok_u) as u32,
                        ))?;
                }
            }
            gemm(
                &self.stream,
                &self.k,
                &full.wk,
                &self.xb,
                &mut self.k_buf,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
            if let Some(ref b) = full.bk {
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
                &full.wv,
                &self.xb,
                &mut self.v_buf,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
            if let Some(ref b) = full.bv {
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

        // Fused Q|gate → compact Q in `q`, gate in `gate_buf`.
        if fused {
            let n_q = (d.n_head_u * d.head_dim * d.n_tok_u) as i32;
            let mode = if flag_q_gate_half() { 1i32 } else { 0i32 };
            unsafe {
                self.stream
                    .launch_builder(&self.k.split_q_gate)
                    .arg(&self.q)
                    .arg(&mut self.gdn_out)
                    .arg(&mut self.gate_buf)
                    .arg(&d.n_head)
                    .arg(&d.hd)
                    .arg(&d.n_tok)
                    .arg(&mode)
                    .launch(LaunchConfig {
                        grid_dim: (d.n_head_u as u32, d.n_tok_u as u32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
                self.stream
                    .launch_builder(&self.k.copy_f32)
                    .arg(&self.gdn_out)
                    .arg(&mut self.q)
                    .arg(&n_q)
                    .launch(LaunchConfig::for_num_elems(n_q as u32))?;
            }
        }

        unsafe {
            let q_norm = full.attn_q_norm.as_ref();
            let k_norm = full.attn_k_norm.as_ref();
            let use_rope = (self.cfg.no_rope_layer_interval == 0
                || (li + 1) % self.cfg.no_rope_layer_interval != 0)
                && !flag_no_rope();
            let force_full_rope = flag_full_rope();
            let partial =
                !force_full_rope && self.cfg.rope_dim > 0 && self.cfg.rope_dim < d.head_dim;
            let n_rot = if !use_rope {
                0i32 // rms only (debug / no-rope)
            } else if partial {
                self.cfg.rope_dim as i32
            } else {
                d.hd
            };

            if let (Some(qw), Some(kw)) = (q_norm, k_norm) {
                let launch = LaunchConfig {
                    grid_dim: (d.n_head_u as u32, d.n_tok_u as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                // Always use partial-rope kernel when we have an explicit n_rot
                // (including 0 = norm only, or rope_dim < head_dim).
                if partial || !use_rope || n_rot != d.hd {
                    if d.use_device_pos {
                        self.stream
                            .launch_builder(&self.k.qk_norm_partial_rope_d)
                            .arg(&mut self.q)
                            .arg(qw)
                            .arg(&d.n_head)
                            .arg(&d.hd)
                            .arg(&n_rot)
                            .arg(&self.d_pos0)
                            .arg(&d.n_tok)
                            .arg(&d.theta)
                            .arg(&d.eps)
                            .launch(launch)?;
                        self.stream
                            .launch_builder(&self.k.qk_norm_partial_rope_d)
                            .arg(&mut self.k_buf)
                            .arg(kw)
                            .arg(&d.n_kv)
                            .arg(&d.hd)
                            .arg(&n_rot)
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
                            .launch_builder(&self.k.qk_norm_partial_rope)
                            .arg(&mut self.q)
                            .arg(qw)
                            .arg(&d.n_head)
                            .arg(&d.hd)
                            .arg(&n_rot)
                            .arg(&d.pos0_i)
                            .arg(&d.n_tok)
                            .arg(&d.theta)
                            .arg(&d.eps)
                            .launch(launch)?;
                        self.stream
                            .launch_builder(&self.k.qk_norm_partial_rope)
                            .arg(&mut self.k_buf)
                            .arg(kw)
                            .arg(&d.n_kv)
                            .arg(&d.hd)
                            .arg(&n_rot)
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
            } else if use_rope {
                if d.use_device_pos {
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
                } else {
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

        if apply_out_gate {
            let n = (d.n_head_u * d.head_dim * d.n_tok_u) as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k.mul_sigmoid)
                    .arg(&mut self.xb)
                    .arg(&self.gate_buf)
                    .arg(&n)
                    .launch(LaunchConfig::for_num_elems(n as u32))?;
            }
        }

        if d.n_tok == 1 {
            if !try_gemv_global_q8(
                &self.stream,
                &self.k,
                &full.wo,
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
                    &full.wo,
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
                &full.wo,
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

    fn attn_block_linear(
        &mut self,
        li: usize,
        d: &ChunkDims,
        cache: &mut CudaKv,
        lin: &LinearAttnWeights,
    ) -> Result<()> {
        // Debug: skip GDN and leave residual stream unchanged (identity mixer).
        if flag_skip_linear() {
            let _ = (li, d, cache, lin);
            return Ok(());
        }
        let n_k = lin.n_k_heads as i32;
        let n_v = lin.n_v_heads as i32;
        let d_k = lin.state_size as i32;
        let d_v = lin.state_size as i32;
        let conv_ch = lin.conv_channels as i32;
        let kernel = lin.conv_kernel as i32;
        let n_v_elems = (lin.n_v_heads * d.n_tok_u) as i32;
        // Qwen3.5: d_k == d_v == state_size (typically 128). Enables fused decode/prefill.
        let square_state = d_k == d_v
            && lin.state_size > 0
            && lin.state_size <= 256
            && !flag_gdn_legacy();

        if d.n_tok == 1 {
            if !try_gemv_gdn_4way(
                &self.stream,
                &self.k,
                &lin.wqkv,
                &lin.w_gate,
                &lin.ssm_beta,
                &lin.ssm_alpha,
                &self.xb,
                &mut self.q,
                &mut self.gdn_z,
                &mut self.gdn_beta,
                &mut self.gdn_alpha,
                &mut self.q8_x,
                &mut self.q8_d,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )? {
                gemv(
                    &self.stream,
                    &self.k,
                    &lin.wqkv,
                    &self.xb,
                    &mut self.q,
                    None,
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
                gemv(
                    &self.stream,
                    &self.k,
                    &lin.w_gate,
                    &self.xb,
                    &mut self.gdn_z,
                    None,
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
                gemv(
                    &self.stream,
                    &self.k,
                    &lin.ssm_beta,
                    &self.xb,
                    &mut self.gdn_beta,
                    None,
                    GemvResidual::None,
                    &mut self.gemv_partial,
                    self.gemv_partial_stride,
                )?;
                gemv(
                    &self.stream,
                    &self.k,
                    &lin.ssm_alpha,
                    &self.xb,
                    &mut self.gdn_alpha,
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
                &lin.wqkv,
                &self.xb,
                &mut self.q,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
            gemm(
                &self.stream,
                &self.k,
                &lin.w_gate,
                &self.xb,
                &mut self.gdn_z,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
            gemm(
                &self.stream,
                &self.k,
                &lin.ssm_beta,
                &self.xb,
                &mut self.gdn_beta,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
            gemm(
                &self.stream,
                &self.k,
                &lin.ssm_alpha,
                &self.xb,
                &mut self.gdn_alpha,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
        }

        unsafe {
            if d.n_tok == 1 && square_state {
                // Fast path: 2 launches after projections
                //   1) conv + split + L2(Q/K)
                //   2) prep(α,β) + delta rule + gated RMS → xb
                let blk = 128u32;
                self.stream
                    .launch_builder(&self.k.gdn_conv_qkvl2_one)
                    .arg(&self.q)
                    .arg(&lin.conv1d)
                    .arg(&mut cache.conv[li])
                    .arg(&mut self.gdn_q)
                    .arg(&mut self.gdn_k)
                    .arg(&mut self.gdn_v)
                    .arg(&n_k)
                    .arg(&n_v)
                    .arg(&d_k)
                    .arg(&kernel)
                    .arg(&d.eps)
                    .launch(LaunchConfig {
                        grid_dim: (lin.n_k_heads as u32, 1, 1),
                        block_dim: (blk, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
                if lin.state_size == 128 {
                    self.stream
                        .launch_builder(&self.k.gdn_delta_gated_one_d128)
                        .arg(&self.gdn_q)
                        .arg(&self.gdn_k)
                        .arg(&self.gdn_v)
                        .arg(&self.gdn_alpha)
                        .arg(&self.gdn_beta)
                        .arg(&lin.ssm_dt)
                        .arg(&lin.ssm_a)
                        .arg(&mut cache.ssm[li])
                        .arg(&lin.ssm_norm)
                        .arg(&self.gdn_z)
                        .arg(&mut self.xb)
                        .arg(&n_k)
                        .arg(&n_v)
                        .arg(&d.eps)
                        .launch(LaunchConfig {
                            grid_dim: (lin.n_v_heads as u32, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                } else {
                    self.stream
                        .launch_builder(&self.k.gdn_delta_gated_one)
                        .arg(&self.gdn_q)
                        .arg(&self.gdn_k)
                        .arg(&self.gdn_v)
                        .arg(&mut self.gdn_alpha)
                        .arg(&mut self.gdn_beta)
                        .arg(&lin.ssm_dt)
                        .arg(&lin.ssm_a)
                        .arg(&mut cache.ssm[li])
                        .arg(&lin.ssm_norm)
                        .arg(&self.gdn_z)
                        .arg(&mut self.xb)
                        .arg(&n_k)
                        .arg(&n_v)
                        .arg(&d_k)
                        .arg(&d.eps)
                        .launch(LaunchConfig {
                            grid_dim: (lin.n_v_heads as u32, 1, 1),
                            block_dim: (blk, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
            } else if d.n_tok > 1 && square_state {
                // Prefill: prep → conv → fused split+L2 → fused delta+gated_rms
                self.stream
                    .launch_builder(&self.k.gdn_prep_decay_beta)
                    .arg(&mut self.gdn_alpha)
                    .arg(&mut self.gdn_beta)
                    .arg(&lin.ssm_dt)
                    .arg(&lin.ssm_a)
                    .arg(&n_v)
                    .arg(&d.n_tok)
                    .launch(LaunchConfig::for_num_elems(n_v_elems as u32))?;
                self.stream
                    .launch_builder(&self.k.causal_conv1d)
                    .arg(&self.q)
                    .arg(&lin.conv1d)
                    .arg(&mut cache.conv[li])
                    .arg(&mut self.gdn_conv)
                    .arg(&conv_ch)
                    .arg(&kernel)
                    .arg(&d.n_tok)
                    .launch(LaunchConfig {
                        grid_dim: ((lin.conv_channels as u32 + 255) / 256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
                self.stream
                    .launch_builder(&self.k.gdn_split_l2_seq)
                    .arg(&self.gdn_conv)
                    .arg(&mut self.gdn_q)
                    .arg(&mut self.gdn_k)
                    .arg(&mut self.gdn_v)
                    .arg(&n_k)
                    .arg(&n_v)
                    .arg(&d_k)
                    .arg(&d_v)
                    .arg(&d.n_tok)
                    .arg(&d.eps)
                    .launch(LaunchConfig {
                        grid_dim: (lin.n_k_heads as u32, d.n_tok_u as u32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
                self.stream
                    .launch_builder(&self.k.gdn_delta_gated_seq)
                    .arg(&self.gdn_q)
                    .arg(&self.gdn_k)
                    .arg(&self.gdn_v)
                    .arg(&self.gdn_alpha)
                    .arg(&self.gdn_beta)
                    .arg(&mut cache.ssm[li])
                    .arg(&lin.ssm_norm)
                    .arg(&self.gdn_z)
                    .arg(&mut self.xb)
                    .arg(&n_k)
                    .arg(&n_v)
                    .arg(&d_k)
                    .arg(&d.n_tok)
                    .arg(&d.eps)
                    .launch(LaunchConfig {
                        grid_dim: (lin.n_v_heads as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            } else {
                // Generic fallback (non-square state or debug).
                self.stream
                    .launch_builder(&self.k.gdn_prep_decay_beta)
                    .arg(&mut self.gdn_alpha)
                    .arg(&mut self.gdn_beta)
                    .arg(&lin.ssm_dt)
                    .arg(&lin.ssm_a)
                    .arg(&n_v)
                    .arg(&d.n_tok)
                    .launch(LaunchConfig::for_num_elems(n_v_elems as u32))?;

                if d.n_tok == 1 {
                    self.stream
                        .launch_builder(&self.k.causal_conv1d_one)
                        .arg(&self.q)
                        .arg(&lin.conv1d)
                        .arg(&mut cache.conv[li])
                        .arg(&mut self.gdn_conv)
                        .arg(&conv_ch)
                        .arg(&kernel)
                        .launch(LaunchConfig {
                            grid_dim: ((lin.conv_channels as u32 + 255) / 256, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    self.stream
                        .launch_builder(&self.k.split_qkv_l2_one)
                        .arg(&self.gdn_conv)
                        .arg(&mut self.gdn_q)
                        .arg(&mut self.gdn_k)
                        .arg(&mut self.gdn_v)
                        .arg(&n_k)
                        .arg(&n_v)
                        .arg(&d_k)
                        .arg(&d_v)
                        .arg(&d.eps)
                        .launch(LaunchConfig {
                            grid_dim: (lin.n_k_heads as u32, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    if d_k == d_v {
                        self.stream
                            .launch_builder(&self.k.gated_delta_one)
                            .arg(&self.gdn_q)
                            .arg(&self.gdn_k)
                            .arg(&self.gdn_v)
                            .arg(&self.gdn_alpha)
                            .arg(&self.gdn_beta)
                            .arg(&mut cache.ssm[li])
                            .arg(&mut self.gdn_out)
                            .arg(&n_k)
                            .arg(&n_v)
                            .arg(&d_k)
                            .launch(LaunchConfig {
                                grid_dim: (lin.n_v_heads as u32, 1, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: 0,
                            })?;
                    } else {
                        self.stream
                            .launch_builder(&self.k.gated_delta_seq)
                            .arg(&self.gdn_q)
                            .arg(&self.gdn_k)
                            .arg(&self.gdn_v)
                            .arg(&self.gdn_alpha)
                            .arg(&self.gdn_beta)
                            .arg(&mut cache.ssm[li])
                            .arg(&mut self.gdn_out)
                            .arg(&n_k)
                            .arg(&n_v)
                            .arg(&d_k)
                            .arg(&d_v)
                            .arg(&d.n_tok)
                            .launch(LaunchConfig {
                                grid_dim: (lin.n_v_heads as u32, 1, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: 0,
                            })?;
                    }
                } else {
                    self.stream
                        .launch_builder(&self.k.causal_conv1d)
                        .arg(&self.q)
                        .arg(&lin.conv1d)
                        .arg(&mut cache.conv[li])
                        .arg(&mut self.gdn_conv)
                        .arg(&conv_ch)
                        .arg(&kernel)
                        .arg(&d.n_tok)
                        .launch(LaunchConfig {
                            grid_dim: ((lin.conv_channels as u32 + 255) / 256, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    self.stream
                        .launch_builder(&self.k.split_qkv_conv)
                        .arg(&self.gdn_conv)
                        .arg(&mut self.gdn_q)
                        .arg(&mut self.gdn_k)
                        .arg(&mut self.gdn_v)
                        .arg(&n_k)
                        .arg(&n_v)
                        .arg(&d_k)
                        .arg(&d_v)
                        .arg(&d.n_tok)
                        .launch(LaunchConfig {
                            grid_dim: (d.n_tok_u as u32, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    self.stream
                        .launch_builder(&self.k.l2_norm_heads)
                        .arg(&mut self.gdn_q)
                        .arg(&n_k)
                        .arg(&d_k)
                        .arg(&d.n_tok)
                        .arg(&d.eps)
                        .launch(LaunchConfig {
                            grid_dim: (lin.n_k_heads as u32, d.n_tok_u as u32, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    self.stream
                        .launch_builder(&self.k.l2_norm_heads)
                        .arg(&mut self.gdn_k)
                        .arg(&n_k)
                        .arg(&d_k)
                        .arg(&d.n_tok)
                        .arg(&d.eps)
                        .launch(LaunchConfig {
                            grid_dim: (lin.n_k_heads as u32, d.n_tok_u as u32, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    self.stream
                        .launch_builder(&self.k.gated_delta_seq)
                        .arg(&self.gdn_q)
                        .arg(&self.gdn_k)
                        .arg(&self.gdn_v)
                        .arg(&self.gdn_alpha)
                        .arg(&self.gdn_beta)
                        .arg(&mut cache.ssm[li])
                        .arg(&mut self.gdn_out)
                        .arg(&n_k)
                        .arg(&n_v)
                        .arg(&d_k)
                        .arg(&d_v)
                        .arg(&d.n_tok)
                        .launch(LaunchConfig {
                            grid_dim: (lin.n_v_heads as u32, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }

                self.stream
                    .launch_builder(&self.k.gated_rms_norm)
                    .arg(&self.gdn_out)
                    .arg(&lin.ssm_norm)
                    .arg(&self.gdn_z)
                    .arg(&mut self.xb)
                    .arg(&n_v)
                    .arg(&d_v)
                    .arg(&d.n_tok)
                    .arg(&d.eps)
                    .launch(LaunchConfig {
                        grid_dim: (lin.n_v_heads as u32, d.n_tok_u as u32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }

        if d.n_tok == 1 {
            // Project gated-RMS result in xb onto residual stream.
            if !try_gemv_global_q8(
                &self.stream,
                &self.k,
                &lin.ssm_out,
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
                    &lin.ssm_out,
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
                &lin.ssm_out,
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
        // Snapshot FFN kind without holding a borrow across GEMV calls.
        let is_moe = matches!(self.layers[li].ffn, super::types::LayerFfn::Moe(_));
        if is_moe {
            return self.ffn_block_moe(li, d);
        }
        self.ffn_block_dense(li, d)
    }

    fn ffn_block_dense(&mut self, li: usize, d: &ChunkDims) -> Result<()> {
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
        let (gate, up, down) = match &self.layers[li].ffn {
            super::types::LayerFfn::Dense { gate, up, down } => {
                // SAFETY: pointers only used for the duration of this function; no layer mut.
                (gate as *const _, up as *const _, down as *const _)
            }
            super::types::LayerFfn::Moe(_) => unreachable!("dense path"),
        };
        // Reconstruct refs for GEMV (layer weights immutable during forward).
        let gate = unsafe { &*gate };
        let up = unsafe { &*up };
        let down = unsafe { &*down };

        let mut fused_ffn = false;
        if d.n_tok == 1 {
            fused_ffn = try_gemv_q4_ffn(
                &self.stream,
                &self.k,
                gate,
                up,
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
                    gate,
                    up,
                    &self.xb,
                    &mut self.hb,
                    &mut self.hb2,
                    &mut self.q8_x,
                    &mut self.q8_d,
                )?
                && !try_gemv_pair(
                    &self.stream,
                    &self.k,
                    gate,
                    up,
                    &self.xb,
                    &mut self.hb,
                    &mut self.hb2,
                    None,
                    None,
                    &mut self.q8_x,
                    &mut self.q8_d,
                )?
            {
                // Q4_0 / Q8 global quant path (required for MoE Q4 packs).
                if !try_gemv_global_q8(
                    &self.stream,
                    &self.k,
                    gate,
                    &self.xb,
                    &mut self.hb,
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
                        gate,
                        &self.xb,
                        &mut self.hb,
                        None,
                        GemvResidual::None,
                        &mut self.gemv_partial,
                        self.gemv_partial_stride,
                    )?;
                }
                if !try_gemv_global_q8(
                    &self.stream,
                    &self.k,
                    up,
                    &self.xb,
                    &mut self.hb2,
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
                        up,
                        &self.xb,
                        &mut self.hb2,
                        None,
                        GemvResidual::None,
                        &mut self.gemv_partial,
                        self.gemv_partial_stride,
                    )?;
                }
            }
        } else {
            gemm(
                &self.stream,
                &self.k,
                gate,
                &self.xb,
                &mut self.hb,
                d.n_tok,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
            gemm(
                &self.stream,
                &self.k,
                up,
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
                down,
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
                    down,
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
                down,
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

    /// Sparse MoE FFN: top-k experts only (decode + token-serial prefill).
    fn ffn_block_moe(&mut self, li: usize, d: &ChunkDims) -> Result<()> {
        if d.n_tok_u == 1 {
            return self.ffn_block_moe_one(li, d);
        }
        // Expert choice differs per token — run serial single-token MoE for prefill.
        let n = d.n_embd_u;
        let n_tok = d.n_tok_u;
        let batch = self.stream.clone_dtoh(&self.x)?;
        let mut out_batch = batch.clone();
        for t in 0..n_tok {
            self.stream
                .memcpy_htod(&batch[t * n..(t + 1) * n], &mut self.x)?;
            let d1 = ChunkDims {
                n_tok: 1,
                n_tok_u: 1,
                n_embd: d.n_embd,
                n_embd_u: d.n_embd_u,
                n_ff_u: d.n_ff_u,
                n_head: d.n_head,
                n_head_u: d.n_head_u,
                n_kv: d.n_kv,
                n_kv_heads: d.n_kv_heads,
                head_dim: d.head_dim,
                hd: d.hd,
                stride: d.stride,
                stride_u: d.stride_u,
                pos0: d.pos0 + t,
                pos0_i: d.pos0_i + t as i32,
                eps: d.eps,
                theta: d.theta,
                scale: d.scale,
                use_device_pos: false,
            };
            self.ffn_block_moe_one(li, &d1)?;
            let out = self.stream.clone_dtoh(&self.x)?;
            out_batch[t * n..(t + 1) * n].copy_from_slice(&out[..n]);
        }
        self.stream.memcpy_htod(&out_batch, &mut self.x)?;
        Ok(())
    }

    fn ffn_block_moe_one(&mut self, li: usize, d: &ChunkDims) -> Result<()> {
        debug_assert_eq!(d.n_tok, 1);
        let one = 1i32;
        unsafe {
            self.stream
                .launch_builder(&self.k.rms_norm)
                .arg(&self.x)
                .arg(&self.layers[li].ffn_norm)
                .arg(&mut self.xb)
                .arg(&d.n_embd)
                .arg(&one)
                .arg(&d.eps)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let (n_experts, top_k, expert_ff, gate_col_bytes, up_col_bytes, down_col_bytes) =
            match &self.layers[li].ffn {
                super::types::LayerFfn::Moe(m) => (
                    m.n_experts,
                    m.top_k,
                    m.expert_ff,
                    m.gate_all.col_bytes,
                    m.up_all.col_bytes,
                    m.down_all.col_bytes,
                ),
                _ => unreachable!(),
            };
        let n_exp_i = n_experts as i32;
        let n_embd_i = d.n_embd;
        let top_k_i = top_k as i32;
        let expert_ff_i = expert_ff as i32;
        let gate_cb = gate_col_bytes as i32;
        let up_cb = up_col_bytes as i32;
        let down_cb = down_col_bytes as i32;
        let zero_res = 0i32;

        // Device router top-k → moe_idx / moe_w (no host sync; CUDA-graph safe).
        {
            let router = match &self.layers[li].ffn {
                super::types::LayerFfn::Moe(m) => &m.router as *const _,
                _ => unreachable!(),
            };
            let router = unsafe { &*router };
            unsafe {
                self.stream
                    .launch_builder(&self.k.moe_router_topk)
                    .arg(router)
                    .arg(&self.xb)
                    .arg(&mut self.moe_idx)
                    .arg(&mut self.moe_w)
                    .arg(&n_exp_i)
                    .arg(&n_embd_i)
                    .arg(&top_k_i)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }

        // Quantize xb once for gate/up (n_rows = n_embd).
        {
            use super::matmul::quantize_q8;
            quantize_q8(
                &self.stream,
                &self.k,
                &self.xb,
                &mut self.q8_x,
                &mut self.q8_d,
                n_embd_i,
            )?;
        }

        let warps = self.k.gemv_quantized_warps.max(1);
        let threads = warps * 32;
        let grid_gate = ((expert_ff as u32) + warps - 1) / warps;
        let grid_down = ((d.n_embd_u as u32) + warps - 1) / warps;

        for slot in 0..top_k {
            let slot_i = slot as i32;
            let (gate_all, up_all, down_all): (
                *const super::types::GpuMat,
                *const super::types::GpuMat,
                *const super::types::GpuMat,
            ) = match &self.layers[li].ffn {
                super::types::LayerFfn::Moe(m) => (&m.gate_all, &m.up_all, &m.down_all),
                _ => unreachable!(),
            };
            let gate_all = unsafe { &*gate_all };
            let up_all = unsafe { &*up_all };
            let down_all = unsafe { &*down_all };

            let expert_slot_k = match gate_all.wtype {
                super::types::WType::Q4_0 => &self.k.gemv_q4_0_expert_slot,
                _ => &self.k.gemv_q8_expert_slot,
            };
            // gate
            unsafe {
                self.stream
                    .launch_builder(expert_slot_k)
                    .arg(&gate_all.data)
                    .arg(&self.q8_x)
                    .arg(&self.q8_d)
                    .arg(&mut self.hb)
                    .arg(&self.moe_idx)
                    .arg(&slot_i)
                    .arg(&n_embd_i)
                    .arg(&expert_ff_i)
                    .arg(&gate_cb)
                    .arg(&zero_res)
                    .arg(&self.xb) // unused residual
                    .launch(LaunchConfig {
                        grid_dim: (grid_gate, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            // up
            unsafe {
                self.stream
                    .launch_builder(expert_slot_k)
                    .arg(&up_all.data)
                    .arg(&self.q8_x)
                    .arg(&self.q8_d)
                    .arg(&mut self.hb2)
                    .arg(&self.moe_idx)
                    .arg(&slot_i)
                    .arg(&n_embd_i)
                    .arg(&expert_ff_i)
                    .arg(&up_cb)
                    .arg(&zero_res)
                    .arg(&self.xb)
                    .launch(LaunchConfig {
                        grid_dim: (grid_gate, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            unsafe {
                self.stream
                    .launch_builder(&self.k.silu_mul)
                    .arg(&mut self.hb)
                    .arg(&self.hb2)
                    .arg(&expert_ff_i)
                    .launch(LaunchConfig::for_num_elems(expert_ff as u32))?;
            }
            // Quantize hb for down (n_rows = expert_ff).
            {
                use super::matmul::quantize_q8;
                quantize_q8(
                    &self.stream,
                    &self.k,
                    &self.hb,
                    &mut self.q8_x,
                    &mut self.q8_d,
                    expert_ff_i,
                )?;
            }
            let down_slot_k = match down_all.wtype {
                super::types::WType::Q4_0 => &self.k.gemv_q4_0_expert_slot,
                _ => &self.k.gemv_q8_expert_slot,
            };
            unsafe {
                self.stream
                    .launch_builder(down_slot_k)
                    .arg(&down_all.data)
                    .arg(&self.q8_x)
                    .arg(&self.q8_d)
                    .arg(&mut self.xb2)
                    .arg(&self.moe_idx)
                    .arg(&slot_i)
                    .arg(&expert_ff_i)
                    .arg(&n_embd_i)
                    .arg(&down_cb)
                    .arg(&zero_res)
                    .arg(&self.xb)
                    .launch(LaunchConfig {
                        grid_dim: (grid_down, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            let n = d.n_embd;
            unsafe {
                self.stream
                    .launch_builder(&self.k.scale_add_slot)
                    .arg(&mut self.x)
                    .arg(&self.xb2)
                    .arg(&self.moe_w)
                    .arg(&slot_i)
                    .arg(&n)
                    .launch(LaunchConfig::for_num_elems(d.n_embd_u as u32))?;
            }
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
