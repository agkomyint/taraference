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
    /// GGML Q4_0 (18 B / 32 vals) — column-major [n_cols][n_blocks][18].
    Q4_0,
    /// Q4_0 block-major [n_blocks][n_cols][18] for coalesced multi-col expert GEMV.
    Q4_0_BM,
    /// Decode-optimized: Q4_0 expanded to f16 column-major (2 B/elem) at load.
    F16,
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

/// Full softmax-attention weights (Qwen2 / Qwen3 / Qwen3.5 full layers).
pub struct FullAttnWeights {
    pub wq: GpuMat,
    pub bq: Option<CudaSlice<f32>>,
    pub wk: GpuMat,
    pub bk: Option<CudaSlice<f32>>,
    pub wv: GpuMat,
    pub bv: Option<CudaSlice<f32>>,
    pub wo: GpuMat,
    pub attn_q_norm: Option<CudaSlice<f32>>,
    pub attn_k_norm: Option<CudaSlice<f32>>,
    /// Qwen3.5: Q projection is 2× wide (interleaved Q|gate per head).
    pub fused_q_gate: bool,
}

/// Gated DeltaNet linear-attention weights (Qwen3.5).
pub struct LinearAttnWeights {
    pub wqkv: GpuMat,
    /// Output gate z projection (value_dim).
    pub w_gate: GpuMat,
    /// Depthwise conv1d weight [kernel, channels] as f32 flat.
    pub conv1d: CudaSlice<f32>,
    pub conv_kernel: usize,
    pub conv_channels: usize,
    /// Already -exp(A_log); length = n_v_heads.
    pub ssm_a: CudaSlice<f32>,
    pub ssm_dt: CudaSlice<f32>,
    pub ssm_alpha: GpuMat,
    pub ssm_beta: GpuMat,
    pub ssm_norm: CudaSlice<f32>,
    pub ssm_out: GpuMat,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub state_size: usize,
}

pub enum LayerAttn {
    Full(FullAttnWeights),
    Linear(LinearAttnWeights),
}

/// Sparse MoE FFN: router (f32) + packed expert Q8 weights.
/// Experts packed as consecutive column groups so GEMV can index by device top-k id
/// (CUDA-graph safe, no host roundtrip). Inspired by AirLLM sparse weight access.
pub struct MoeFfnWeights {
    /// Row-major `[n_experts, n_embd]` f32.
    pub router: CudaSlice<f32>,
    pub n_experts: usize,
    pub top_k: usize,
    pub expert_ff: usize,
    /// Packed gate: n_rows=n_embd, n_cols=expert_ff * n_experts (Q8_0 columns).
    pub gate_all: GpuMat,
    /// Packed up: same layout as gate_all.
    pub up_all: GpuMat,
    /// Packed down: n_rows=expert_ff, n_cols=n_embd * n_experts.
    pub down_all: GpuMat,
}

pub enum LayerFfn {
    Dense {
        gate: GpuMat,
        up: GpuMat,
        down: GpuMat,
    },
    Moe(MoeFfnWeights),
}

pub struct GpuLayer {
    pub attn_norm: CudaSlice<f32>,
    /// Pre-FFN norm (`ffn_norm` or Qwen3.5 `post_attention_norm`).
    pub ffn_norm: CudaSlice<f32>,
    pub attn: LayerAttn,
    pub ffn: LayerFfn,
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
    pub gemv_q6_compact_global_8way: CudaFunction,
    #[allow(dead_code)]
    pub gemv_q6_compact_global_mcol: CudaFunction,
    pub gemv_q8: CudaFunction,
    pub gemv_q8_global: CudaFunction,
    pub gemv_q4_0: CudaFunction,
    pub gemv_q4_0_global: CudaFunction,
    /// Packed MoE expert GEMV for Q4_0.
    pub gemv_q4_0_expert_slot: CudaFunction,
    pub gemv_q4_0_expert_gate_up: CudaFunction,
    /// Gate+up+SiLU then Q8-quantize 32-col tiles (skips separate quantize before down).
    pub gemv_q4_0_expert_gate_up_q8: CudaFunction,
    pub gemv_q4_0_expert_down_scale: CudaFunction,
    /// f16 expert GEMV (Q4 expanded at load) — hot path for 100M@750.
    pub gemv_f16_expert_gate_up: CudaFunction,
    pub gemv_f16_expert_down_scale: CudaFunction,
    pub gemv_f16_expert_gate_up_4w: CudaFunction,
    pub gemv_f16_expert_down_scale_4w: CudaFunction,
    pub gemv_q4_0_bm_expert_gate_up: CudaFunction,
    pub gemv_q4_0_bm_expert_down_scale: CudaFunction,
    /// Fused Q4_0 decode helpers (MoE packs fall off Q4_K paths without these).
    pub gemv_q4_0_qkv: CudaFunction,
    pub gemv_q4_0_qkv_2w: CudaFunction,
    pub gemv_q4_0_pair: CudaFunction,
    pub gemv_q4_0_ffn: CudaFunction,
    pub gemv_q4_0_expert_gate_up_2w: CudaFunction,
    pub gemv_q4_0_expert_down_scale_2w: CudaFunction,
    pub gemv_q4_0_expert_gate_up_4w: CudaFunction,
    pub gemv_q4_0_expert_down_scale_4w: CudaFunction,
    pub gemv_q4_0_expert_gate_up_f32: CudaFunction,
    pub gemv_q4_0_expert_down_scale_f32: CudaFunction,
    pub gemv_q8_expert_gate_up: CudaFunction,
    pub gemv_q8_expert_down_scale: CudaFunction,
    pub gemv_q4_splitk: CudaFunction,
    pub gemv_q4_global_splitk: CudaFunction,
    pub gemv_q5_splitk: CudaFunction,
    pub gemv_q5k_splitk: CudaFunction,
    pub gemv_q6_splitk: CudaFunction,
    pub gemv_q6_repack_splitk: CudaFunction,
    pub gemv_q6_repack_global_splitk: CudaFunction,
    pub gemv_q6_compact_global_splitk: CudaFunction,
    pub gemv_q8_splitk: CudaFunction,
    pub gemv_q8_global_splitk: CudaFunction,
    pub gemv_splitk_reduce: CudaFunction,
    /// Fused dual single-token GEMV for Q5_0 (Q+K or gate+up; stage x once).
    pub gemv_q5_qk: CudaFunction,
    /// Fused Q+K+V single-token GEMV for Q5_0 when all three match.
    pub gemv_q5_qkv: CudaFunction,
    /// Fused dual single-token GEMV for Q4_K (gate+up / Q+K on larger Q4_K_M).
    pub gemv_q4_pair: CudaFunction,
    pub gemv_q4_dual: CudaFunction,
    pub gemv_q4_ffn: CudaFunction,
    pub gemv_q4_ffn_8way: CudaFunction,
    pub gemv_q4_ffn_mcol: CudaFunction,
    #[allow(dead_code)]
    pub gemv_q4_ffn_smem: CudaFunction,
    pub gemv_q4_dual_threads: u32,
    pub gemv_quantized_warps: u32,
    pub gemv_q4_qkv: CudaFunction,
    pub gemv_q8_qkv: CudaFunction,
    pub gemv_q8_gdn_4way: CudaFunction,
    pub gemv_hybrid_gdn_4way: CudaFunction,
    pub gemv_q8_qkv_splitk: CudaFunction,
    pub gemv_q8_gdn_4way_splitk: CudaFunction,
    pub gemv_hybrid_gdn_4way_splitk: CudaFunction,
    pub gemv_splitk_reduce_qkv: CudaFunction,
    pub gemv_splitk_reduce_gdn_4way: CudaFunction,
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
    pub embed_q4_0: CudaFunction,
    pub embed_q4_one: CudaFunction,
    pub embed_q5_one: CudaFunction,
    pub embed_q5k_one: CudaFunction,
    pub embed_q6_one: CudaFunction,
    pub embed_q8_one: CudaFunction,
    pub embed_q4_0_one: CudaFunction,
    pub embed_q4_one_d: CudaFunction,
    pub embed_q5_one_d: CudaFunction,
    pub embed_q5k_one_d: CudaFunction,
    pub embed_q6_one_d: CudaFunction,
    pub embed_q8_one_d: CudaFunction,
    pub embed_q4_0_one_d: CudaFunction,
    pub rms_norm: CudaFunction,
    /// Fused MoE FFN prep: rms + router top-k + Q8 quant (decode n_tok=1).
    pub moe_ffn_prep: CudaFunction,
    /// Fused attn prep: rms + Q8 quant (decode n_tok=1).
    pub attn_prep: CudaFunction,
    pub silu_mul: CudaFunction,
    pub add: CudaFunction,
    /// `a[i] += scale * b[i]` (MoE expert residual).
    pub scale_add: CudaFunction,
    /// `a[i] += weights[slot] * b[i]`.
    pub scale_add_slot: CudaFunction,
    /// Dense f32 GEMV for MoE router scores.
    pub gemv_f32_rows: CudaFunction,
    /// Device MoE router top-k + softmax.
    pub moe_router_topk: CudaFunction,
    /// Q8 GEMV for packed expert column group selected by device top-k slot.
    pub gemv_q8_expert_slot: CudaFunction,
    pub add_bias: CudaFunction,
    pub rope: CudaFunction,
    pub rope_d: CudaFunction,
    pub qk_norm_rope: CudaFunction,
    pub qk_norm_rope_d: CudaFunction,
    pub qk_norm_partial_rope: CudaFunction,
    pub qk_norm_partial_rope_d: CudaFunction,
    #[allow(dead_code)]
    pub sigmoid: CudaFunction,
    #[allow(dead_code)]
    pub softplus_bias_scale: CudaFunction,
    #[allow(dead_code)]
    pub softplus_bias_scale_rows: CudaFunction,
    pub gdn_prep_decay_beta: CudaFunction,
    pub copy_f32: CudaFunction,
    #[allow(dead_code)]
    pub exp_f: CudaFunction,
    pub l2_norm_heads: CudaFunction,
    pub gated_rms_norm: CudaFunction,
    pub split_q_gate: CudaFunction,
    pub mul_sigmoid: CudaFunction,
    pub causal_conv1d: CudaFunction,
    pub causal_conv1d_one: CudaFunction,
    pub gated_delta_seq: CudaFunction,
    pub gated_delta_one: CudaFunction,
    /// Fused decode: conv1d + SiLU + split Q/K/V + L2(Q,K).
    pub gdn_conv_qkvl2_one: CudaFunction,
    /// Fused decode: prep(α,β) + delta rule + gated RMSNorm.
    pub gdn_delta_gated_one: CudaFunction,
    /// d_state==128 specialized variant.
    pub gdn_delta_gated_one_d128: CudaFunction,
    /// Prefill: split conv buffer + L2(Q,K) in one launch.
    pub gdn_split_l2_seq: CudaFunction,
    /// Prefill: delta rule + gated RMSNorm (token-serial, head-parallel).
    pub gdn_delta_gated_seq: CudaFunction,
    pub split_qkv_conv: CudaFunction,
    pub split_qkv_l2_one: CudaFunction,
    pub zero_f32: CudaFunction,
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
