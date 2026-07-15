//! GPU tensor / layer / kernel handle types.

use cudarc::driver::{CudaFunction, CudaSlice};

/// Max tokens in one prefill GEMM launch.
pub const MAX_BATCH: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WType {
    Q4K,
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
    pub gemv_q6: CudaFunction,
    pub gemv_q8: CudaFunction,
    pub gemm_q4: CudaFunction,
    pub gemm_q6: CudaFunction,
    pub gemm_q8: CudaFunction,
    pub embed_q4: CudaFunction,
    pub embed_q6: CudaFunction,
    pub embed_q8: CudaFunction,
    pub rms_norm: CudaFunction,
    pub silu_mul: CudaFunction,
    pub add: CudaFunction,
    pub add_bias: CudaFunction,
    pub rope: CudaFunction,
    pub attn: CudaFunction,
    pub copy_kv: CudaFunction,
    pub argmax: CudaFunction,
    pub copy_last: CudaFunction,
}
