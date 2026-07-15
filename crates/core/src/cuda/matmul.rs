//! Quantized GEMV / GEMM / embed launches.

use super::model::CudaModel;
use super::types::{GpuMat, Kernels, WType};
use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Must match `GEMV_WARPS` in gemv.cu.
const GEMV_WARPS: u32 = 8;
const GEMV_THREADS: u32 = GEMV_WARPS * 32;

pub(crate) fn lc_gemv(n_cols: u32, n_rows: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n_cols + GEMV_WARPS - 1) / GEMV_WARPS, 1, 1),
        block_dim: (GEMV_THREADS, 1, 1),
        shared_mem_bytes: n_rows * 4,
    }
}

pub(crate) fn lc_gemm_cols(n_cols: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n_cols, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Residual stream handling for single-token GEMV.
#[derive(Clone, Copy)]
pub(crate) enum GemvResidual<'a> {
    None,
    /// `out = W x + residual` (separate buffer).
    #[allow(dead_code)] // ready for other fused paths
    Add(&'a CudaSlice<f32>),
    /// `y = W x + y` — fuse residual add into the matvec (decode residual stream).
    /// Kernel only touches index `j` as read-then-write; safe alias of out/residual.
    InPlace,
}

/// `y = W x [+ bias] [+ residual]` (single-token GEMV).
pub(crate) fn gemv(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    w: &GpuMat,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    residual: GemvResidual<'_>,
) -> Result<()> {
    let n_rows = w.n_rows as i32;
    let n_cols = w.n_cols as i32;
    let col_bytes = w.col_bytes as i32;
    let use_bias: i32 = if bias.is_some() { 1 } else { 0 };
    // 0 = none, 1 = separate residual buf, 2 = in-place out[j] += (see gemv.cu).
    let use_res: i32 = match residual {
        GemvResidual::None => 0,
        GemvResidual::Add(_) => 1,
        GemvResidual::InPlace => 2,
    };
    // Dummy pointer when unused (flags gate loads).
    let bias_ptr: &CudaSlice<f32> = bias.unwrap_or(x);
    let f = match w.wtype {
        WType::Q4K => &k.gemv_q4,
        WType::Q5_0 => &k.gemv_q5,
        WType::Q6K => &k.gemv_q6,
        WType::Q8_0 => &k.gemv_q8,
    };
    unsafe {
        let mut lb = stream.launch_builder(f);
        lb.arg(&w.data)
            .arg(x)
            .arg(y)
            .arg(&n_rows)
            .arg(&n_cols)
            .arg(&col_bytes)
            .arg(&use_bias)
            .arg(bias_ptr)
            .arg(&use_res);
        match residual {
            GemvResidual::None | GemvResidual::InPlace => {
                // Mode 2 reads residual from `out`; pointer unused but required.
                lb.arg(x);
            }
            GemvResidual::Add(r) => {
                lb.arg(r);
            }
        }
        lb.launch(lc_gemv(w.n_cols as u32, w.n_rows as u32))?;
    }
    Ok(())
}

pub(crate) fn gemm(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    w: &GpuMat,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    n_tok: i32,
) -> Result<()> {
    if n_tok == 1 {
        return gemv(stream, k, w, x, y, None, GemvResidual::None);
    }
    let n_rows = w.n_rows as i32;
    let n_cols = w.n_cols as i32;
    let col_bytes = w.col_bytes as i32;
    let f = match w.wtype {
        WType::Q4K => &k.gemm_q4,
        WType::Q5_0 => &k.gemm_q5,
        WType::Q6K => &k.gemm_q6,
        WType::Q8_0 => &k.gemm_q8,
    };
    unsafe {
        stream
            .launch_builder(f)
            .arg(&w.data)
            .arg(x)
            .arg(y)
            .arg(&n_rows)
            .arg(&n_cols)
            .arg(&col_bytes)
            .arg(&n_tok)
            .launch(lc_gemm_cols(w.n_cols as u32))?;
    }
    Ok(())
}

impl CudaModel {
    pub(crate) fn embed_batch(&mut self, tokens: &[i32]) -> Result<()> {
        if tokens.len() == 1 {
            return self.embed_one(tokens[0]);
        }
        let n_tok = tokens.len() as i32;
        self.stream.memcpy_htod(tokens, &mut self.tok_buf)?;
        let n_rows = self.token_embd.n_rows as i32;
        let col_bytes = self.token_embd.col_bytes as i32;
        let f = match self.token_embd.wtype {
            WType::Q4K => &self.k.embed_q4,
            WType::Q5_0 => &self.k.embed_q5,
            WType::Q6K => &self.k.embed_q6,
            WType::Q8_0 => &self.k.embed_q8,
        };
        unsafe {
            self.stream
                .launch_builder(f)
                .arg(&self.token_embd.data)
                .arg(&mut self.x)
                .arg(&self.tok_buf)
                .arg(&n_tok)
                .arg(&n_rows)
                .arg(&col_bytes)
                .launch(LaunchConfig {
                    grid_dim: (n_tok as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn embed_one(&mut self, token: i32) -> Result<()> {
        let n_rows = self.token_embd.n_rows as i32;
        let col_bytes = self.token_embd.col_bytes as i32;
        let f = match self.token_embd.wtype {
            WType::Q4K => &self.k.embed_q4_one,
            WType::Q5_0 => &self.k.embed_q5_one,
            WType::Q6K => &self.k.embed_q6_one,
            WType::Q8_0 => &self.k.embed_q8_one,
        };
        unsafe {
            self.stream
                .launch_builder(f)
                .arg(&self.token_embd.data)
                .arg(&mut self.x)
                .arg(&token)
                .arg(&n_rows)
                .arg(&col_bytes)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}
