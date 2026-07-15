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

/// `y = W x [+ bias] [+ residual]` (single-token GEMV).
pub(crate) fn gemv(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    w: &GpuMat,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    residual: Option<&CudaSlice<f32>>,
) -> Result<()> {
    let n_rows = w.n_rows as i32;
    let n_cols = w.n_cols as i32;
    let col_bytes = w.col_bytes as i32;
    let use_bias: i32 = if bias.is_some() { 1 } else { 0 };
    let use_res: i32 = if residual.is_some() { 1 } else { 0 };
    // Dummy pointer when unused (flags gate loads).
    let bias_ptr: &CudaSlice<f32> = bias.unwrap_or(x);
    let res_ptr: &CudaSlice<f32> = residual.unwrap_or(x);
    let f = match w.wtype {
        WType::Q4K => &k.gemv_q4,
        WType::Q6K => &k.gemv_q6,
        WType::Q8_0 => &k.gemv_q8,
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
            .arg(&use_bias)
            .arg(bias_ptr)
            .arg(&use_res)
            .arg(res_ptr)
            .launch(lc_gemv(w.n_cols as u32, w.n_rows as u32))?;
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
        return gemv(stream, k, w, x, y, None, None);
    }
    let n_rows = w.n_rows as i32;
    let n_cols = w.n_cols as i32;
    let col_bytes = w.col_bytes as i32;
    let f = match w.wtype {
        WType::Q4K => &k.gemm_q4,
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
