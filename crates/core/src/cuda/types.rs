//! GPU tensor / layer / kernel handle types.

use cudarc::driver::{CudaFunction, CudaSlice};
use std::collections::HashMap;

/// Max tokens in one prefill GEMM launch.
pub const MAX_BATCH: usize = 256;
/// Current token plus up to eight prompt-lookup draft tokens.
pub const MAX_VERIFY_TOKENS: usize = 9;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WType {
    Q4K,
    Q5K,
    Q5_0,
    Q6K,
    Q8_0,
}

pub struct GpuMat {
    pub data: CudaSlice<u8>,
    /// Optional decode-optimized representation; original GGUF data remains
    /// available for batched prefill/GEMM.
    pub decode_data: Option<CudaSlice<u8>>,
    /// Aligned compact Q6_K blocks used by decode GEMVs.
    pub compact_data: Option<CudaSlice<u8>>,
    pub n_rows: usize,
    pub n_cols: usize,
    pub col_bytes: usize,
    pub decode_col_bytes: usize,
    pub compact_col_bytes: usize,
    pub wtype: WType,
}

pub struct GpuLayer {
    pub attn_norm: CudaSlice<f32>,
    /// Qwen3 applies an RMS norm independently to every projected Q head.
    pub attn_q_norm: Option<CudaSlice<f32>>,
    /// Qwen3 applies an RMS norm independently to every projected K head.
    pub attn_k_norm: Option<CudaSlice<f32>>,
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
    pub quantize_q8: CudaFunction,
    pub gemv_q4: CudaFunction,
    pub gemv_q4_global: CudaFunction,
    pub gemv_q5: CudaFunction,
    pub gemv_q5k: CudaFunction,
    pub gemv_q6: CudaFunction,
    pub gemv_q6_repack: CudaFunction,
    pub gemv_q6_repack_global: CudaFunction,
    pub gemv_q6_compact_global: CudaFunction,
    pub gemv_q6_compact_global_4way: CudaFunction,
    pub gemv_q8: CudaFunction,
    pub gemv_q4_splitk: CudaFunction,
    pub gemv_q4_global_splitk: CudaFunction,
    pub gemv_q5_splitk: CudaFunction,
    pub gemv_q5k_splitk: CudaFunction,
    pub gemv_q6_splitk: CudaFunction,
    pub gemv_q6_repack_splitk: CudaFunction,
    pub gemv_q6_repack_global_splitk: CudaFunction,
    pub gemv_q6_compact_global_splitk: CudaFunction,
    pub gemv_q8_splitk: CudaFunction,
    pub gemv_splitk_reduce: CudaFunction,
    /// Fused dual single-token GEMV for Q5_0 (Q+K or gate+up; stage x once).
    pub gemv_q5_qk: CudaFunction,
    /// Fused Q+K+V single-token GEMV for Q5_0 when all three match.
    pub gemv_q5_qkv: CudaFunction,
    /// Fused dual single-token GEMV for Q4_K (gate+up / Q+K on larger Q4_K_M).
    pub gemv_q4_pair: CudaFunction,
    pub gemv_q4_dual: CudaFunction,
    pub gemv_q4_ffn: CudaFunction,
    pub gemv_q4_dual_threads: u32,
    pub gemv_quantized_warps: u32,
    pub gemv_q4_qkv: CudaFunction,
    pub gemm_q4: CudaFunction,
    pub gemm_q5: CudaFunction,
    pub gemm_q5k: CudaFunction,
    pub gemm_q6: CudaFunction,
    pub gemm_q8: CudaFunction,
    pub embed_q4: CudaFunction,
    pub embed_q5: CudaFunction,
    pub embed_q5k: CudaFunction,
    pub embed_q6: CudaFunction,
    pub embed_q8: CudaFunction,
    pub embed_q4_one: CudaFunction,
    pub embed_q5_one: CudaFunction,
    pub embed_q5k_one: CudaFunction,
    pub embed_q6_one: CudaFunction,
    pub embed_q8_one: CudaFunction,
    pub embed_q4_one_d: CudaFunction,
    pub embed_q5_one_d: CudaFunction,
    pub embed_q5k_one_d: CudaFunction,
    pub embed_q6_one_d: CudaFunction,
    pub embed_q8_one_d: CudaFunction,
    pub rms_norm: CudaFunction,
    pub silu_mul: CudaFunction,
    pub add: CudaFunction,
    pub add_bias: CudaFunction,
    pub rope: CudaFunction,
    pub rope_d: CudaFunction,
    pub qk_norm_rope: CudaFunction,
    pub qk_norm_rope_d: CudaFunction,
    /// Attention symbols from [`crate::cuda::decode::REGISTRY`] (CUDA name → fn).
    pub attn: HashMap<&'static str, CudaFunction>,
    pub copy_kv: CudaFunction,
    pub copy_kv_d: CudaFunction,
    pub argmax: CudaFunction,
    pub argmax_rows: CudaFunction,
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
