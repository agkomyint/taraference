//! Quantized GEMV / GEMM / embed launches.

use super::model::CudaModel;
use super::types::{GpuMat, Kernels, WType};
use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Must match `GEMV_WARPS` in gemv.cu.
const GEMV_WARPS: u32 = 8;
const GEMV_THREADS: u32 = GEMV_WARPS * 32;
/// Must match `GEMV_SPLIT_MAX` in gemv.cu / load.rs.
pub(crate) const GEMV_SPLIT_MAX: u32 = 8;
/// Only split-K when staging `x` would use large shared memory (FFN-tall mats).
/// Below this, classic GEMV already has enough column-parallelism; a reduce would hurt.
const GEMV_SPLITK_MIN_ROWS: usize = 4096;
/// Cap smem for baseline path; if larger, prefer split-K even if rows slightly lower.
const GEMV_BASELINE_SMEM_CAP: usize = 24 * 1024;

pub(crate) fn lc_gemv(n_cols: u32, n_rows: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n_cols + GEMV_WARPS - 1) / GEMV_WARPS, 1, 1),
        block_dim: (GEMV_THREADS, 1, 1),
        shared_mem_bytes: n_rows * 4,
    }
}

fn lc_gemv_splitk(n_cols: u32, n_split: u32, rows_per_split: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n_cols + GEMV_WARPS - 1) / GEMV_WARPS, n_split, 1),
        block_dim: (GEMV_THREADS, 1, 1),
        shared_mem_bytes: rows_per_split * 4,
    }
}

pub(crate) fn lc_gemm_cols(n_cols: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n_cols, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// How many K-splits to use (1 = classic GEMV).
pub(crate) fn gemv_n_split(n_rows: usize) -> u32 {
    let smem = n_rows * 4;
    if n_rows < GEMV_SPLITK_MIN_ROWS && smem <= GEMV_BASELINE_SMEM_CAP {
        return 1;
    }
    // Tall FFN down (e.g. n_rows=11008): 4-way split cuts smem ~4× and multiplies blocks.
    // (Avoid 8-way: extra reduce traffic can erase the win on laptop GPUs.)
    4u32.min(GEMV_SPLIT_MAX)
}

/// Residual stream handling for single-token GEMV.
#[derive(Clone, Copy)]
pub(crate) enum GemvResidual<'a> {
    None,
    #[allow(dead_code)]
    Add(&'a CudaSlice<f32>),
    /// `y = W x + y`
    InPlace,
}

/// `y = W x [+ bias] [+ residual]` (single-token GEMV).
/// Uses split-K when `n_rows` is large (needs `partial` buffer from the model).
pub(crate) fn gemv(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    w: &GpuMat,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    residual: GemvResidual<'_>,
    partial: &mut CudaSlice<f32>,
    partial_stride: usize,
) -> Result<()> {
    let n_split = gemv_n_split(w.n_rows);
    if n_split <= 1 {
        return gemv_baseline(stream, k, w, x, y, bias, residual);
    }
    if w.n_cols > partial_stride {
        // Safety fallback if a mat is wider than the arena (should not happen).
        return gemv_baseline(stream, k, w, x, y, bias, residual);
    }
    gemv_splitk(stream, k, w, x, y, bias, residual, partial, n_split)
}

fn gemv_baseline(
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
    let use_res: i32 = match residual {
        GemvResidual::None => 0,
        GemvResidual::Add(_) => 1,
        GemvResidual::InPlace => 2,
    };
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

fn gemv_splitk(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    w: &GpuMat,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    residual: GemvResidual<'_>,
    partial: &mut CudaSlice<f32>,
    n_split: u32,
) -> Result<()> {
    let n_rows = w.n_rows as i32;
    let n_cols = w.n_cols as i32;
    let col_bytes = w.col_bytes as i32;
    let n_split_i = n_split as i32;
    let use_bias: i32 = if bias.is_some() { 1 } else { 0 };
    let use_res: i32 = match residual {
        GemvResidual::None => 0,
        GemvResidual::Add(_) => 1,
        GemvResidual::InPlace => 2,
    };
    let bias_ptr: &CudaSlice<f32> = bias.unwrap_or(x);

    // Max rows any split owns (ceil). Superblock-aligned types use 256; group types 32 —
    // host smem is an upper bound; over-allocating smem is OK, under-alloc is not.
    let block = if matches!(w.wtype, WType::Q4K | WType::Q6K) {
        256usize
    } else {
        32usize
    };
    let n_blocks = w.n_rows / block;
    let max_blocks_per_split = (n_blocks + n_split as usize - 1) / n_split as usize;
    let rows_per_split = (max_blocks_per_split * block).max(block) as u32;

    let f = match w.wtype {
        WType::Q4K => &k.gemv_q4_splitk,
        WType::Q5_0 => &k.gemv_q5_splitk,
        WType::Q6K => &k.gemv_q6_splitk,
        WType::Q8_0 => &k.gemv_q8_splitk,
    };

    unsafe {
        stream
            .launch_builder(f)
            .arg(&w.data)
            .arg(x)
            .arg(&mut *partial)
            .arg(&n_rows)
            .arg(&n_cols)
            .arg(&col_bytes)
            .arg(&n_split_i)
            .launch(lc_gemv_splitk(w.n_cols as u32, n_split, rows_per_split))?;

        let mut lb = stream.launch_builder(&k.gemv_splitk_reduce);
        lb.arg(&*partial)
            .arg(y)
            .arg(&n_cols)
            .arg(&n_split_i)
            .arg(&use_bias)
            .arg(bias_ptr)
            .arg(&use_res);
        match residual {
            GemvResidual::None | GemvResidual::InPlace => {
                lb.arg(x);
            }
            GemvResidual::Add(r) => {
                lb.arg(r);
            }
        }
        lb.launch(LaunchConfig {
            grid_dim: ((w.n_cols as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
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
    partial: &mut CudaSlice<f32>,
    partial_stride: usize,
) -> Result<()> {
    if n_tok == 1 {
        return gemv(
            stream,
            k,
            w,
            x,
            y,
            None,
            GemvResidual::None,
            partial,
            partial_stride,
        );
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
