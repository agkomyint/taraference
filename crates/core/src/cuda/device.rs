//! Shared CUDA device bootstrap for GGUF and MoE pack loaders.
//!
//! Both load paths open the same context, compile the same NVRTC source, and
//! bind the same kernel symbols. Keeping that logic in one place avoids dual
//! edits when a kernel is added or retargeted.

use super::decode::DecodeBackend;
use super::kernels::SOURCE;
use super::types::Kernels;
use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaModule, CudaStream};
use std::sync::Arc;

/// Live GPU + compiled module used for one `CudaModel` lifetime.
pub(crate) struct DeviceBundle {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    pub gpu_name: String,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub nvrtc_arch: String,
    pub kernels: Kernels,
}

/// Tuning knobs derived from compute capability (and optional env override).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemvTuning {
    pub q4_dual_4way: bool,
    pub q4_dual_threads: u32,
    pub gemv_quantized_warps: u32,
}

impl GemvTuning {
    /// Select dual/ffn symbols and warp counts for the live GPU.
    ///
    /// `warps_override` is the parsed `TARAFER_GEMV_WARPS` value when set.
    pub fn from_cc(cc_major: i32, warps_override: Option<u32>) -> Self {
        let q4_dual_4way = cc_major >= 8;
        let q4_dual_threads = if q4_dual_4way { 128 } else { 64 };
        // Ampere+: modest warps for high-register Q4 DP4A; Turing prefers fat blocks.
        let gemv_quantized_warps = warps_override
            .filter(|&w| (1..=32).contains(&w))
            .unwrap_or(if cc_major >= 8 { 16 } else { 32 });
        Self {
            q4_dual_4way,
            q4_dual_threads,
            gemv_quantized_warps,
        }
    }

    pub fn q4_dual_symbol(self) -> &'static str {
        if self.q4_dual_4way {
            "gemv_q4_k_dual_4way"
        } else {
            "gemv_q4_k_dual"
        }
    }

    pub fn q4_ffn_symbol(self) -> &'static str {
        if self.q4_dual_4way {
            "gemv_q4_k_ffn_4way"
        } else {
            "gemv_q4_k_ffn"
        }
    }

    pub fn q4_ffn_smem_symbol(self) -> &'static str {
        if self.q4_dual_4way {
            "gemv_q4_k_ffn_4way_smem"
        } else {
            "gemv_q4_k_ffn"
        }
    }
}

fn gemv_warps_from_env() -> Option<u32> {
    std::env::var("TARAFER_GEMV_WARPS")
        .ok()
        .and_then(|s| s.parse().ok())
}

/// Open device 0, disable event tracking, compile NVRTC, load all kernels.
pub(crate) fn init_device() -> Result<DeviceBundle> {
    let ctx = CudaContext::new(0).context("CudaContext")?;
    // cudarc event-tracking inserts cuStreamWaitEvent around every buffer use.
    // Those waits create cross-stream deps that make CUDA graph capture fail with
    // CUDA_ERROR_STREAM_CAPTURE_ISOLATION. We use one stream only → tracking is
    // unnecessary and must be off *before* any allocations.
    // SAFETY: single inference stream; no concurrent multi-stream buffer sharing.
    unsafe {
        ctx.disable_event_tracking();
    }
    // Non-null stream required for CUDA graph capture (legacy default/null cannot capture).
    let stream = ctx.new_stream().context("CudaStream (non-blocking)")?;
    let (cc_major, cc_minor) = ctx
        .compute_capability()
        .context("device compute capability")?;
    let arch = format!("sm_{cc_major}{cc_minor}");
    let gpu_name = ctx
        .name()
        .unwrap_or_else(|_| format!("CUDA device 0 (sm_{cc_major}{cc_minor})"));
    eprintln!("GPU device | {gpu_name} | compute {cc_major}.{cc_minor} | NVRTC arch={arch}");

    // `CompileOptions::arch` is `&'static str`; pass dynamic target via `options`.
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        SOURCE,
        cudarc::nvrtc::CompileOptions {
            arch: None,
            include_paths: vec![],
            ftz: Some(true),
            prec_div: Some(false),
            prec_sqrt: Some(false),
            fmad: Some(true),
            use_fast_math: Some(true),
            maxrregcount: None,
            options: vec![format!("--gpu-architecture={arch}")],
            name: None,
        },
    )
    .context("nvrtc compile")?;
    let module = ctx.load_module(ptx).context("load_module")?;
    let tuning = GemvTuning::from_cc(cc_major, gemv_warps_from_env());
    eprintln!(
        "tuning | q4_dual={} | threads={} | quantized_warps={}",
        if tuning.q4_dual_4way { "4way" } else { "2way" },
        tuning.q4_dual_threads,
        tuning.gemv_quantized_warps
    );
    let kernels = Kernels::load(&module, tuning)?;

    Ok(DeviceBundle {
        ctx,
        stream,
        module,
        gpu_name,
        compute_major: cc_major,
        compute_minor: cc_minor,
        nvrtc_arch: arch,
        kernels,
    })
}

impl Kernels {
    /// Bind every NVRTC symbol used by the forward path.
    pub(crate) fn load(module: &Arc<CudaModule>, tuning: GemvTuning) -> Result<Self> {
        let q4_dual_symbol = tuning.q4_dual_symbol();
        let q4_ffn_symbol = tuning.q4_ffn_symbol();
        let q4_ffn_smem_symbol = tuning.q4_ffn_smem_symbol();
        Ok(Self {
            quantize_q8: module.load_function("quantize_q8_global")?,
            gemv_q4: module.load_function("gemv_q4_k")?,
            gemv_q4_global: module.load_function("gemv_q4_k_global")?,
            gemv_q5: module.load_function("gemv_q5_0")?,
            gemv_q5k: module.load_function("gemv_q5_k")?,
            gemv_q6: module.load_function("gemv_q6_k")?,
            gemv_q6_repack: module.load_function("gemv_q6_k_repack")?,
            gemv_q6_repack_global: module.load_function("gemv_q6_k_repack_global")?,
            gemv_q6_compact_global: module.load_function("gemv_q6_k_compact_global")?,
            gemv_q6_compact_global_4way: module.load_function("gemv_q6_k_compact_global_4way")?,
            gemv_q6_compact_global_8way: module.load_function("gemv_q6_k_compact_global_8way")?,
            gemv_q6_compact_global_mcol: module.load_function("gemv_q6_k_compact_global_mcol")?,
            gemv_q8: module.load_function("gemv_q8_0")?,
            gemv_q8_global: module.load_function("gemv_q8_0_global")?,
            gemv_q4_0: module.load_function("gemv_q4_0")?,
            gemv_q4_0_global: module.load_function("gemv_q4_0_global")?,
            gemv_q4_0_expert_slot: module.load_function("gemv_q4_0_global_expert_slot")?,
            gemv_q4_0_expert_gate_up: module.load_function("gemv_q4_0_global_expert_gate_up")?,
            gemv_q4_0_expert_gate_up_q8: module
                .load_function("gemv_q4_0_global_expert_gate_up_q8")?,
            gemv_q4_0_expert_down_scale: module
                .load_function("gemv_q4_0_global_expert_down_scale")?,
            gemv_f16_expert_gate_up: module.load_function("gemv_f16_expert_gate_up")?,
            gemv_f16_expert_down_scale: module.load_function("gemv_f16_expert_down_scale")?,
            gemv_f16_expert_gate_up_4w: module.load_function("gemv_f16_expert_gate_up_4w")?,
            gemv_f16_expert_down_scale_4w: module
                .load_function("gemv_f16_expert_down_scale_4w")?,
            gemv_q4_0_bm_expert_gate_up: module.load_function("gemv_q4_0_bm_expert_gate_up")?,
            gemv_q4_0_bm_expert_down_scale: module
                .load_function("gemv_q4_0_bm_expert_down_scale")?,
            gemv_q4_0_qkv: module.load_function("gemv_q4_0_global_qkv")?,
            gemv_q4_0_qkv_2w: module.load_function("gemv_q4_0_global_qkv_2w")?,
            gemv_q4_0_pair: module.load_function("gemv_q4_0_global_pair")?,
            gemv_q4_0_ffn: module.load_function("gemv_q4_0_global_ffn")?,
            gemv_q4_0_expert_gate_up_2w: module
                .load_function("gemv_q4_0_global_expert_gate_up_2w")?,
            gemv_q4_0_expert_down_scale_2w: module
                .load_function("gemv_q4_0_global_expert_down_scale_2w")?,
            gemv_q4_0_expert_gate_up_4w: module
                .load_function("gemv_q4_0_global_expert_gate_up_4w")?,
            gemv_q4_0_expert_down_scale_4w: module
                .load_function("gemv_q4_0_global_expert_down_scale_4w")?,
            gemv_q4_0_expert_gate_up_f32: module.load_function("gemv_q4_0_expert_gate_up_f32")?,
            gemv_q4_0_expert_down_scale_f32: module
                .load_function("gemv_q4_0_expert_down_scale_f32")?,
            gemv_q8_expert_gate_up: module.load_function("gemv_q8_0_global_expert_gate_up")?,
            gemv_q8_expert_down_scale: module
                .load_function("gemv_q8_0_global_expert_down_scale")?,
            gemv_q4_splitk: module.load_function("gemv_q4_k_splitk")?,
            gemv_q4_global_splitk: module.load_function("gemv_q4_k_global_splitk")?,
            gemv_q5_splitk: module.load_function("gemv_q5_0_splitk")?,
            gemv_q5k_splitk: module.load_function("gemv_q5_k_splitk")?,
            gemv_q6_splitk: module.load_function("gemv_q6_k_splitk")?,
            gemv_q6_repack_splitk: module.load_function("gemv_q6_k_repack_splitk")?,
            gemv_q6_repack_global_splitk: module.load_function("gemv_q6_k_repack_global_splitk")?,
            gemv_q6_compact_global_splitk: module
                .load_function("gemv_q6_k_compact_global_splitk")?,
            gemv_q8_splitk: module.load_function("gemv_q8_0_splitk")?,
            gemv_q8_global_splitk: module.load_function("gemv_q8_0_global_splitk")?,
            gemv_splitk_reduce: module.load_function("gemv_splitk_reduce")?,
            gemv_q5_qk: module.load_function("gemv_q5_0_qk")?,
            gemv_q5_qkv: module.load_function("gemv_q5_0_qkv")?,
            gemv_q4_pair: module.load_function("gemv_q4_k_pair")?,
            gemv_q4_dual: module.load_function(q4_dual_symbol)?,
            gemv_q4_ffn: module.load_function(q4_ffn_symbol)?,
            gemv_q4_ffn_8way: module.load_function("gemv_q4_k_ffn_8way")?,
            gemv_q4_ffn_mcol: module.load_function("gemv_q4_k_ffn_mcol")?,
            gemv_q4_ffn_smem: module.load_function(q4_ffn_smem_symbol)?,
            gemv_q4_dual_threads: tuning.q4_dual_threads,
            gemv_quantized_warps: tuning.gemv_quantized_warps,
            gemv_q4_qkv: module.load_function("gemv_q4_k_qkv")?,
            gemv_q8_qkv: module.load_function("gemv_q8_0_qkv")?,
            gemv_q8_gdn_4way: module.load_function("gemv_q8_0_gdn_4way")?,
            gemv_hybrid_gdn_4way: module.load_function("gemv_hybrid_gdn_4way")?,
            gemv_q8_qkv_splitk: module.load_function("gemv_q8_0_qkv_splitk")?,
            gemv_q8_gdn_4way_splitk: module.load_function("gemv_q8_0_gdn_4way_splitk")?,
            gemv_hybrid_gdn_4way_splitk: module.load_function("gemv_hybrid_gdn_4way_splitk")?,
            gemv_splitk_reduce_qkv: module.load_function("gemv_splitk_reduce_qkv")?,
            gemv_splitk_reduce_gdn_4way: module.load_function("gemv_splitk_reduce_gdn_4way")?,
            gemm_q4: module.load_function("gemm_q4_k")?,
            gemm_q5: module.load_function("gemm_q5_0")?,
            gemm_q5k: module.load_function("gemm_q5_k")?,
            gemm_q6: module.load_function("gemm_q6_k")?,
            gemm_q8: module.load_function("gemm_q8_0")?,
            embed_q4: module.load_function("embed_q4_k")?,
            embed_q5: module.load_function("embed_q5_0")?,
            embed_q5k: module.load_function("embed_q5_k")?,
            embed_q6: module.load_function("embed_q6_k")?,
            embed_q8: module.load_function("embed_q8_0")?,
            embed_q4_0: module.load_function("embed_q4_0")?,
            embed_q4_one: module.load_function("embed_q4_k_one")?,
            embed_q5_one: module.load_function("embed_q5_0_one")?,
            embed_q5k_one: module.load_function("embed_q5_k_one")?,
            embed_q6_one: module.load_function("embed_q6_k_one")?,
            embed_q8_one: module.load_function("embed_q8_0_one")?,
            embed_q4_0_one: module.load_function("embed_q4_0_one")?,
            embed_q4_one_d: module.load_function("embed_q4_k_one_d")?,
            embed_q5_one_d: module.load_function("embed_q5_0_one_d")?,
            embed_q5k_one_d: module.load_function("embed_q5_k_one_d")?,
            embed_q6_one_d: module.load_function("embed_q6_k_one_d")?,
            embed_q8_one_d: module.load_function("embed_q8_0_one_d")?,
            embed_q4_0_one_d: module.load_function("embed_q4_0_one_d")?,
            rms_norm: module.load_function("rms_norm_f32")?,
            moe_ffn_prep: module.load_function("moe_ffn_prep_rms_router_quant")?,
            attn_prep: module.load_function("attn_prep_rms_quant")?,
            silu_mul: module.load_function("silu_mul_f32")?,
            add: module.load_function("add_f32")?,
            scale_add: module.load_function("scale_add_f32")?,
            scale_add_slot: module.load_function("scale_add_slot_f32")?,
            gemv_f32_rows: module.load_function("gemv_f32_rows")?,
            moe_router_topk: module.load_function("moe_router_topk_f32")?,
            gemv_q8_expert_slot: module.load_function("gemv_q8_0_global_expert_slot")?,
            add_bias: module.load_function("add_bias_f32")?,
            rope: module.load_function("rope_neox_f32")?,
            rope_d: module.load_function("rope_neox_f32_d")?,
            qk_norm_rope: module.load_function("qk_rms_norm_rope_neox_f32")?,
            qk_norm_rope_d: module.load_function("qk_rms_norm_rope_neox_f32_d")?,
            qk_norm_partial_rope: module.load_function("qk_rms_norm_partial_rope_neox_f32")?,
            qk_norm_partial_rope_d: module.load_function("qk_rms_norm_partial_rope_neox_f32_d")?,
            sigmoid: module.load_function("sigmoid_f32")?,
            softplus_bias_scale: module.load_function("softplus_bias_scale_f32")?,
            softplus_bias_scale_rows: module.load_function("softplus_bias_scale_rows_f32")?,
            gdn_prep_decay_beta: module.load_function("gdn_prep_decay_beta_f32")?,
            copy_f32: module.load_function("copy_f32")?,
            exp_f: module.load_function("exp_f32")?,
            l2_norm_heads: module.load_function("l2_norm_heads_f32")?,
            gated_rms_norm: module.load_function("gated_rms_norm_f32")?,
            split_q_gate: module.load_function("split_q_gate_interleaved_f32")?,
            mul_sigmoid: module.load_function("mul_sigmoid_f32")?,
            causal_conv1d: module.load_function("causal_conv1d_f32")?,
            causal_conv1d_one: module.load_function("causal_conv1d_one_f32")?,
            gated_delta_seq: module.load_function("gated_delta_rule_seq_f32")?,
            gated_delta_one: module.load_function("gated_delta_rule_one_f32")?,
            gdn_conv_qkvl2_one: module.load_function("gdn_conv_qkvl2_one_f32")?,
            gdn_delta_gated_one: module.load_function("gdn_delta_gated_one_f32")?,
            gdn_delta_gated_one_d128: module.load_function("gdn_delta_gated_one_d128_f32")?,
            gdn_split_l2_seq: module.load_function("gdn_split_l2_seq_f32")?,
            gdn_delta_gated_seq: module.load_function("gdn_delta_gated_seq_f32")?,
            split_qkv_conv: module.load_function("split_qkv_from_conv_f32")?,
            split_qkv_l2_one: module.load_function("split_qkv_l2_one_f32")?,
            zero_f32: module.load_function("zero_f32")?,
            attn: {
                let mut map = std::collections::HashMap::new();
                for sym in DecodeBackend::required_kernel_symbols() {
                    map.insert(
                        sym,
                        module.load_function(sym).with_context(|| {
                            format!("load attn kernel {sym} (missing from CUDA source?)")
                        })?,
                    );
                }
                map
            },
            copy_kv: module.load_function("copy_kv_f16")?,
            copy_kv_d: module.load_function("copy_kv_f16_d")?,
            argmax: module.load_function("argmax_f32")?,
            argmax_rows: module.load_function("argmax_rows_f32")?,
            copy_last: module.load_function("copy_last_row")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ampere_uses_4way_and_16_warps() {
        let t = GemvTuning::from_cc(8, None);
        assert!(t.q4_dual_4way);
        assert_eq!(t.q4_dual_threads, 128);
        assert_eq!(t.gemv_quantized_warps, 16);
        assert_eq!(t.q4_dual_symbol(), "gemv_q4_k_dual_4way");
        assert_eq!(t.q4_ffn_symbol(), "gemv_q4_k_ffn_4way");
        assert_eq!(t.q4_ffn_smem_symbol(), "gemv_q4_k_ffn_4way_smem");
    }

    #[test]
    fn turing_uses_2way_and_32_warps() {
        let t = GemvTuning::from_cc(7, None);
        assert!(!t.q4_dual_4way);
        assert_eq!(t.q4_dual_threads, 64);
        assert_eq!(t.gemv_quantized_warps, 32);
        assert_eq!(t.q4_dual_symbol(), "gemv_q4_k_dual");
        assert_eq!(t.q4_ffn_symbol(), "gemv_q4_k_ffn");
        assert_eq!(t.q4_ffn_smem_symbol(), "gemv_q4_k_ffn");
    }

    #[test]
    fn warps_override_clamped_to_valid_range() {
        assert_eq!(GemvTuning::from_cc(8, Some(8)).gemv_quantized_warps, 8);
        assert_eq!(GemvTuning::from_cc(8, Some(32)).gemv_quantized_warps, 32);
        // Out of range → default for arch.
        assert_eq!(GemvTuning::from_cc(8, Some(0)).gemv_quantized_warps, 16);
        assert_eq!(GemvTuning::from_cc(8, Some(64)).gemv_quantized_warps, 16);
        assert_eq!(GemvTuning::from_cc(7, Some(0)).gemv_quantized_warps, 32);
    }
}
