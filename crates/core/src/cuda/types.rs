//! GPU tensor / layer / kernel handle types.

use cudarc::driver::{CudaFunction, CudaSlice};
use std::collections::HashMap;

/// Max tokens in one prefill GEMM launch.
pub const MAX_BATCH: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WType {
    Q4K,
    Q5_0,
    Q6K,
    Q8_0,
}

pub struct GpuMat {
    pub data: CudaSlice<u8>,
    pub n_rows: usize,
    pub n_cols: usize,
    pub col_bytes: usize,
    pub wtype: WType,
}

pub struct GpuLayer {
    pub attn_norm: CudaSlice<f32>,
    pub wq: GpuMat,
    pub bq: Option<CudaSlice<f32>>,
    pub wk: GpuMat,
    pub bk: Option<CudaSlice<f32>>,
    pub wv: GpuMat,
    pub bv: Option<CudaSlice<f32>>,
    pub wo: GpuMat,
    pub ffn_norm: CudaSlice<f32>,
    pub gate: GpuMat,
    pub up: GpuMat,
    pub down: GpuMat,
}

pub struct Kernels {
    pub gemv_q4: CudaFunction,
    pub gemv_q5: CudaFunction,
    pub gemv_q6: CudaFunction,
    pub gemv_q8: CudaFunction,
    pub gemv_q4_splitk: CudaFunction,
    pub gemv_q5_splitk: CudaFunction,
    pub gemv_q6_splitk: CudaFunction,
    pub gemv_q8_splitk: CudaFunction,
    pub gemv_splitk_reduce: CudaFunction,
    pub gemm_q4: CudaFunction,
    pub gemm_q5: CudaFunction,
    pub gemm_q6: CudaFunction,
    pub gemm_q8: CudaFunction,
    pub embed_q4: CudaFunction,
    pub embed_q5: CudaFunction,
    pub embed_q6: CudaFunction,
    pub embed_q8: CudaFunction,
    pub embed_q4_one: CudaFunction,
    pub embed_q5_one: CudaFunction,
    pub embed_q6_one: CudaFunction,
    pub embed_q8_one: CudaFunction,
    pub rms_norm: CudaFunction,
    pub silu_mul: CudaFunction,
    pub add: CudaFunction,
    pub add_bias: CudaFunction,
    pub rope: CudaFunction,
    /// Attention symbols from [`crate::cuda::decode::REGISTRY`] (CUDA name → fn).
    pub attn: HashMap<&'static str, CudaFunction>,
    pub copy_kv: CudaFunction,
    pub argmax: CudaFunction,
    pub copy_last: CudaFunction,
}

impl Kernels {
    pub fn attn(&self, symbol: &str) -> anyhow::Result<&CudaFunction> {
        self.attn.get(symbol).ok_or_else(|| {
            anyhow::anyhow!(
                "attention kernel {symbol:?} not loaded — check REGISTRY + kernels/mod.rs includes"
            )
        })
    }
}
