//! Quantized GEMV / GEMM / embed launches.

use super::model::CudaModel;
use super::types::{GpuMat, Kernels, WType};
use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Must match `GEMV_WARPS` in gemv.cu.
const GEMV_WARPS: u32 = 32;
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

fn lc_gemv_quantized(n_cols: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n_cols + GEMV_WARPS - 1) / GEMV_WARPS, 1, 1),
        block_dim: (GEMV_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn quantize_q8(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    x: &CudaSlice<f32>,
    q8_x: &mut CudaSlice<i8>,
    q8_d: &mut CudaSlice<f32>,
    n_rows: i32,
) -> Result<()> {
    let groups = (n_rows as u32 + 31) / 32;
    unsafe {
        stream
            .launch_builder(&k.quantize_q8)
            .arg(x)
            .arg(q8_x)
            .arg(q8_d)
            .arg(&n_rows)
            .launch(LaunchConfig {
                grid_dim: ((groups + 7) / 8, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok(())
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
    // Four-way split balances tall-matrix occupancy with reduction traffic on T4/consumer GPUs.
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

/// Single-token GEMV that quantizes `x` once globally instead of once per
/// output block. Used for wide decode heads where redundant quantization is costly.
pub(crate) fn try_gemv_global_q8(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    w: &GpuMat,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    residual: GemvResidual<'_>,
    q8_x: &mut CudaSlice<i8>,
    q8_d: &mut CudaSlice<f32>,
    partial: &mut CudaSlice<f32>,
    partial_stride: usize,
) -> Result<bool> {
    let n_split = gemv_n_split(w.n_rows);
    let (f, data, col_bytes) = match w.wtype {
        WType::Q4K => (&k.gemv_q4_global, &w.data, w.col_bytes),
        WType::Q6K if w.decode_data.is_some() => (
            &k.gemv_q6_repack_global,
            w.decode_data.as_ref().unwrap(),
            w.decode_col_bytes,
        ),
        _ => return Ok(false),
    };
    let n_rows = w.n_rows as i32;
    let n_cols = w.n_cols as i32;
    let col_bytes = col_bytes as i32;
    let use_bias = if bias.is_some() { 1i32 } else { 0i32 };
    let use_res = match residual {
        GemvResidual::None => 0i32,
        GemvResidual::Add(_) => 1i32,
        GemvResidual::InPlace => 2i32,
    };
    let bias_ptr = bias.unwrap_or(x);
    let residual_ptr = match residual {
        GemvResidual::Add(r) => r,
        _ => x,
    };
    quantize_q8(stream, k, x, q8_x, q8_d, n_rows)?;
    if n_split > 1 {
        if w.wtype != WType::Q6K || w.decode_data.is_none() || w.n_cols > partial_stride {
            return Ok(false);
        }
        let n_split_i = n_split as i32;
        unsafe {
            stream
                .launch_builder(&k.gemv_q6_repack_global_splitk)
                .arg(w.decode_data.as_ref().unwrap())
                .arg(q8_x)
                .arg(q8_d)
                .arg(&mut *partial)
                .arg(&n_rows)
                .arg(&n_cols)
                .arg(&col_bytes)
                .arg(&n_split_i)
                .launch(LaunchConfig {
                    grid_dim: ((w.n_cols as u32 + GEMV_WARPS - 1) / GEMV_WARPS, n_split, 1),
                    block_dim: (GEMV_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            stream
                .launch_builder(&k.gemv_splitk_reduce)
                .arg(&*partial)
                .arg(y)
                .arg(&n_cols)
                .arg(&n_split_i)
                .arg(&use_bias)
                .arg(bias_ptr)
                .arg(&use_res)
                .arg(residual_ptr)
                .launch(LaunchConfig {
                    grid_dim: ((w.n_cols as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        return Ok(true);
    }
    unsafe {
        stream
            .launch_builder(f)
            .arg(data)
            .arg(q8_x)
            .arg(q8_d)
            .arg(y)
            .arg(&n_rows)
            .arg(&n_cols)
            .arg(&col_bytes)
            .arg(&use_bias)
            .arg(bias_ptr)
            .arg(&use_res)
            .arg(residual_ptr)
            .launch(lc_gemv_quantized(w.n_cols as u32))?;
    }
    Ok(true)
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
    let use_q6_repack = w.wtype == WType::Q6K && w.decode_data.is_some();
    let col_bytes = if use_q6_repack {
        w.decode_col_bytes as i32
    } else {
        w.col_bytes as i32
    };
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
        WType::Q6K if use_q6_repack => &k.gemv_q6_repack,
        WType::Q6K => &k.gemv_q6,
        WType::Q8_0 => &k.gemv_q8,
    };
    unsafe {
        let mut lb = stream.launch_builder(f);
        let data = w.decode_data.as_ref().unwrap_or(&w.data);
        lb.arg(data)
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
    let use_q6_repack = w.wtype == WType::Q6K && w.decode_data.is_some();
    let col_bytes = if use_q6_repack {
        w.decode_col_bytes as i32
    } else {
        w.col_bytes as i32
    };
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
        WType::Q6K if use_q6_repack => &k.gemv_q6_repack_splitk,
        WType::Q6K => &k.gemv_q6_splitk,
        WType::Q8_0 => &k.gemv_q8_splitk,
    };

    unsafe {
        let data = w.decode_data.as_ref().unwrap_or(&w.data);
        stream
            .launch_builder(f)
            .arg(data)
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

/// Decode dual GEMV (Q+K or gate+up): Q5_0 or Q4_K, stage `x` once.
/// Returns true if fused path ran.
pub(crate) fn try_gemv_pair(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    wa: &GpuMat,
    wb: &GpuMat,
    x: &CudaSlice<f32>,
    out_a: &mut CudaSlice<f32>,
    out_b: &mut CudaSlice<f32>,
    ba: Option<&CudaSlice<f32>>,
    bb: Option<&CudaSlice<f32>>,
    q8_x: &mut CudaSlice<i8>,
    q8_d: &mut CudaSlice<f32>,
) -> Result<bool> {
    if try_gemv_q5_pair(stream, k, wa, wb, x, out_a, out_b, ba, bb)? {
        return Ok(true);
    }
    try_gemv_q4_pair(stream, k, wa, wb, x, out_a, out_b, ba, bb, q8_x, q8_d)
}

fn try_gemv_q4_pair(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    wa: &GpuMat,
    wb: &GpuMat,
    x: &CudaSlice<f32>,
    out_a: &mut CudaSlice<f32>,
    out_b: &mut CudaSlice<f32>,
    ba: Option<&CudaSlice<f32>>,
    bb: Option<&CudaSlice<f32>>,
    q8_x: &mut CudaSlice<i8>,
    q8_d: &mut CudaSlice<f32>,
) -> Result<bool> {
    if wa.wtype != WType::Q4K
        || wb.wtype != WType::Q4K
        || wa.n_rows != wb.n_rows
        || wa.col_bytes != wb.col_bytes
    {
        return Ok(false);
    }
    let both_bias = ba.is_some() && bb.is_some();
    let no_bias = ba.is_none() && bb.is_none();
    if !both_bias && !no_bias {
        return Ok(false);
    }
    let n_rows = wa.n_rows as i32;
    let n_a = wa.n_cols as i32;
    let n_b = wb.n_cols as i32;
    let col_bytes = wa.col_bytes as i32;
    let use_bias: i32 = if both_bias { 1 } else { 0 };
    let ba_p: &CudaSlice<f32> = ba.unwrap_or(x);
    let bb_p: &CudaSlice<f32> = bb.unwrap_or(x);
    let n_tot = (wa.n_cols + wb.n_cols) as u32;
    quantize_q8(stream, k, x, q8_x, q8_d, n_rows)?;
    unsafe {
        stream
            .launch_builder(&k.gemv_q4_pair)
            .arg(&wa.data)
            .arg(&wb.data)
            .arg(q8_x)
            .arg(q8_d)
            .arg(out_a)
            .arg(out_b)
            .arg(&n_rows)
            .arg(&n_a)
            .arg(&n_b)
            .arg(&col_bytes)
            .arg(&use_bias)
            .arg(ba_p)
            .arg(bb_p)
            .launch(lc_gemv_quantized(n_tot))?;
    }
    Ok(true)
}

/// Decode dual Q5_0 GEMV (Q+K or gate+up): stage `x` once.
/// Returns true if fused path ran.
fn try_gemv_q5_pair(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    wa: &GpuMat,
    wb: &GpuMat,
    x: &CudaSlice<f32>,
    out_a: &mut CudaSlice<f32>,
    out_b: &mut CudaSlice<f32>,
    ba: Option<&CudaSlice<f32>>,
    bb: Option<&CudaSlice<f32>>,
) -> Result<bool> {
    if wa.wtype != WType::Q5_0
        || wb.wtype != WType::Q5_0
        || wa.n_rows != wb.n_rows
        || wa.col_bytes != wb.col_bytes
    {
        return Ok(false);
    }
    // Mixed bias (one Some, one None) cannot use single use_bias flag safely.
    let both_bias = ba.is_some() && bb.is_some();
    let no_bias = ba.is_none() && bb.is_none();
    if !both_bias && !no_bias {
        return Ok(false);
    }
    let n_rows = wa.n_rows as i32;
    let n_a = wa.n_cols as i32;
    let n_b = wb.n_cols as i32;
    let col_bytes = wa.col_bytes as i32;
    let use_bias: i32 = if both_bias { 1 } else { 0 };
    let ba_p: &CudaSlice<f32> = ba.unwrap_or(x);
    let bb_p: &CudaSlice<f32> = bb.unwrap_or(x);
    let n_tot = (wa.n_cols + wb.n_cols) as u32;
    unsafe {
        stream
            .launch_builder(&k.gemv_q5_qk)
            .arg(&wa.data)
            .arg(&wb.data)
            .arg(x)
            .arg(out_a)
            .arg(out_b)
            .arg(&n_rows)
            .arg(&n_a)
            .arg(&n_b)
            .arg(&col_bytes)
            .arg(&use_bias)
            .arg(ba_p)
            .arg(bb_p)
            .launch(lc_gemv(n_tot, wa.n_rows as u32))?;
    }
    Ok(true)
}

/// Decode Q+K+V in one GEMV for Q4_K or Q5_0 (stage `x` once).
pub(crate) fn try_gemv_qkv(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    wq: &GpuMat,
    wk: &GpuMat,
    wv: &GpuMat,
    x: &CudaSlice<f32>,
    q: &mut CudaSlice<f32>,
    k_out: &mut CudaSlice<f32>,
    v_out: &mut CudaSlice<f32>,
    bq: Option<&CudaSlice<f32>>,
    bk: Option<&CudaSlice<f32>>,
    bv: Option<&CudaSlice<f32>>,
    q8_x: &mut CudaSlice<i8>,
    q8_d: &mut CudaSlice<f32>,
) -> Result<bool> {
    let same_shape = wq.n_rows == wk.n_rows
        && wq.n_rows == wv.n_rows
        && wq.col_bytes == wk.col_bytes
        && wq.col_bytes == wv.col_bytes;
    let all_bias = bq.is_some() && bk.is_some() && bv.is_some();
    let no_bias = bq.is_none() && bk.is_none() && bv.is_none();
    if wq.wtype == WType::Q4K
        && wk.wtype == WType::Q4K
        && wv.wtype == WType::Q4K
        && same_shape
        && (all_bias || no_bias)
    {
        let n_rows = wq.n_rows as i32;
        let n_q = wq.n_cols as i32;
        let n_k = wk.n_cols as i32;
        let n_v = wv.n_cols as i32;
        let col_bytes = wq.col_bytes as i32;
        let use_bias = if all_bias { 1i32 } else { 0i32 };
        let bq_p = bq.unwrap_or(x);
        let bk_p = bk.unwrap_or(x);
        let bv_p = bv.unwrap_or(x);
        let n_tot = (wq.n_cols + wk.n_cols + wv.n_cols) as u32;
        quantize_q8(stream, k, x, q8_x, q8_d, n_rows)?;
        unsafe {
            stream
                .launch_builder(&k.gemv_q4_qkv)
                .arg(&wq.data)
                .arg(&wk.data)
                .arg(&wv.data)
                .arg(q8_x)
                .arg(q8_d)
                .arg(q)
                .arg(k_out)
                .arg(v_out)
                .arg(&n_rows)
                .arg(&n_q)
                .arg(&n_k)
                .arg(&n_v)
                .arg(&col_bytes)
                .arg(&use_bias)
                .arg(bq_p)
                .arg(bk_p)
                .arg(bv_p)
                .launch(lc_gemv_quantized(n_tot))?;
        }
        return Ok(true);
    }
    if wq.wtype != WType::Q5_0
        || wk.wtype != WType::Q5_0
        || wv.wtype != WType::Q5_0
        || wq.n_rows != wk.n_rows
        || wq.n_rows != wv.n_rows
        || wq.col_bytes != wk.col_bytes
        || wq.col_bytes != wv.col_bytes
    {
        return Ok(false);
    }
    if !all_bias && !no_bias {
        return Ok(false);
    }
    let n_rows = wq.n_rows as i32;
    let n_q = wq.n_cols as i32;
    let n_k = wk.n_cols as i32;
    let n_v = wv.n_cols as i32;
    let col_bytes = wq.col_bytes as i32;
    let use_bias: i32 = if all_bias { 1 } else { 0 };
    let bq_p: &CudaSlice<f32> = bq.unwrap_or(x);
    let bk_p: &CudaSlice<f32> = bk.unwrap_or(x);
    let bv_p: &CudaSlice<f32> = bv.unwrap_or(x);
    let n_tot = (wq.n_cols + wk.n_cols + wv.n_cols) as u32;
    unsafe {
        stream
            .launch_builder(&k.gemv_q5_qkv)
            .arg(&wq.data)
            .arg(&wk.data)
            .arg(&wv.data)
            .arg(x)
            .arg(q)
            .arg(k_out)
            .arg(v_out)
            .arg(&n_rows)
            .arg(&n_q)
            .arg(&n_k)
            .arg(&n_v)
            .arg(&col_bytes)
            .arg(&use_bias)
            .arg(bq_p)
            .arg(bk_p)
            .arg(bv_p)
            .launch(lc_gemv(n_tot, wq.n_rows as u32))?;
    }
    Ok(true)
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

    /// Embed from `d_token` (device-side id; CUDA-graph safe).
    pub(crate) fn embed_one_device(&mut self) -> Result<()> {
        let n_rows = self.token_embd.n_rows as i32;
        let col_bytes = self.token_embd.col_bytes as i32;
        let f = match self.token_embd.wtype {
            WType::Q4K => &self.k.embed_q4_one_d,
            WType::Q5_0 => &self.k.embed_q5_one_d,
            WType::Q6K => &self.k.embed_q6_one_d,
            WType::Q8_0 => &self.k.embed_q8_one_d,
        };
        unsafe {
            self.stream
                .launch_builder(f)
                .arg(&self.token_embd.data)
                .arg(&mut self.x)
                .arg(&self.d_token)
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
