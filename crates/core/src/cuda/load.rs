//! Load GGUF weights onto GPU and compile NVRTC kernels.

use super::decode::DecodeBackend;
use super::kernels::SOURCE;
use super::model::CudaModel;
use super::types::{
    FullAttnWeights, GpuLayer, GpuMat, Kernels, LayerAttn, LayerFfn, LinearAttnWeights, WType,
    MAX_BATCH,
};
use crate::config::LayerKind;
use crate::config::ModelConfig;
use crate::quant::dequant_f32;
use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::CudaContext;
use taraference_gguf::{GgmlType, GgufFile};

fn wtype_of(t: GgmlType) -> Result<WType> {
    Ok(match t {
        GgmlType::Q4_K => WType::Q4K,
        GgmlType::Q4_0 => WType::Q4_0,
        GgmlType::Q5_K => WType::Q5K,
        GgmlType::Q5_0 => WType::Q5_0,
        GgmlType::Q6_K => WType::Q6K,
        GgmlType::Q8_0 => WType::Q8_0,
        other => bail!(
            "unsupported weight type {} (supported: Q4_0, Q4_K, Q5_K, Q5_0, Q6_K, Q8_0)",
            other.name()
        ),
    })
}

const Q5_K_BLOCK_BYTES: usize = 176;
const Q8_0_BLOCK_BYTES: usize = 34;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x03ff) as u32;
    match exponent {
        0 if fraction == 0 => sign * 0.0,
        0 => sign * (fraction as f32) * 2.0f32.powi(-24),
        31 if fraction == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + fraction as f32 / 1024.0) * 2.0f32.powi(exponent - 15),
    }
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x007f_ffff;
    if exponent == 255 {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exponent - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = fraction | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let mut rounded = mantissa >> shift;
        let remainder = mantissa & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }
    let mut rounded = fraction + 0x0000_0fff + ((fraction >> 13) & 1);
    let mut out_exp = half_exp as u16;
    if rounded & 0x0080_0000 != 0 {
        rounded = 0;
        out_exp += 1;
        if out_exp >= 31 {
            return sign | 0x7c00;
        }
    }
    sign | (out_exp << 10) | (rounded >> 13) as u16
}

fn q5_k_scale_min(scales: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3f, scales[j + 4] & 0x3f)
    } else {
        (
            (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

fn dequant_q5_k_block(block: &[u8], out: &mut [f32; 256]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];
    let mut is = 0;
    let mut high_low = 1u8;
    let mut high_high = 2u8;
    for group in 0..4 {
        let (sc0, min0) = q5_k_scale_min(scales, is);
        let (sc1, min1) = q5_k_scale_min(scales, is + 1);
        let scale0 = d * sc0 as f32;
        let scale1 = d * sc1 as f32;
        let offset0 = dmin * min0 as f32;
        let offset1 = dmin * min1 as f32;
        let qbase = group * 32;
        let obase = group * 64;
        for l in 0..32 {
            let low = (qs[qbase + l] & 0x0f) + if qh[l] & high_low != 0 { 16 } else { 0 };
            let high = (qs[qbase + l] >> 4) + if qh[l] & high_high != 0 { 16 } else { 0 };
            out[obase + l] = scale0 * low as f32 - offset0;
            out[obase + 32 + l] = scale1 * high as f32 - offset1;
        }
        is += 2;
        high_low <<= 2;
        high_high <<= 2;
    }
}

/// Compatibility path for mixed GGUFs that store selected tensors as Q5_K.
/// CUDA uses the already complete Q8_0 GEMV/GEMM/embedding paths after this
/// one-time CPU conversion. Each Q5_K block (256 values) becomes 8 Q8_0 blocks.
fn transcode_q5_k_to_q8_0(raw: &[u8]) -> Result<Vec<u8>> {
    if raw.len() % Q5_K_BLOCK_BYTES != 0 {
        bail!("invalid Q5_K tensor byte length {}", raw.len());
    }
    let mut out = Vec::with_capacity(raw.len() / Q5_K_BLOCK_BYTES * 8 * Q8_0_BLOCK_BYTES);
    let mut values = [0.0f32; 256];
    for block in raw.chunks_exact(Q5_K_BLOCK_BYTES) {
        dequant_q5_k_block(block, &mut values);
        for group in values.chunks_exact(32) {
            let amax = group.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
            out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
            for &v in group {
                let q = if d == 0.0 {
                    0
                } else {
                    (v / d).round().clamp(-127.0, 127.0) as i8
                };
                out.push(q as u8);
            }
        }
    }
    Ok(out)
}

fn transcode_q5_1_to_q8_0(raw: &[u8]) -> Result<Vec<u8>> {
    const Q5_1_BLOCK_BYTES: usize = 24;
    if raw.len() % Q5_1_BLOCK_BYTES != 0 {
        bail!("invalid Q5_1 tensor byte length {}", raw.len());
    }
    let mut out = Vec::with_capacity(raw.len() / Q5_1_BLOCK_BYTES * Q8_0_BLOCK_BYTES);
    let mut values = [0.0f32; 32];
    for block in raw.chunks_exact(Q5_1_BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];
        for j in 0..16 {
            let high0 = (((qh >> j) << 4) & 0x10) as u8;
            let high1 = ((qh >> (j + 12)) & 0x10) as u8;
            values[j] = ((qs[j] & 0x0f) | high0) as f32 * d + min;
            values[j + 16] = ((qs[j] >> 4) | high1) as f32 * d + min;
        }
        let amax = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let q8_d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        out.extend_from_slice(&f32_to_f16(q8_d).to_le_bytes());
        for &v in &values {
            let q = if q8_d == 0.0 {
                0
            } else {
                (v / q8_d).round().clamp(-127.0, 127.0) as i8
            };
            out.push(q as u8);
        }
    }
    Ok(out)
}

fn f32_host(gguf: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let t = gguf.tensor(name).ok_or_else(|| anyhow!("missing {name}"))?;
    if t.ggml_type != GgmlType::F32 {
        bail!("{name} not F32");
    }
    let mut v = vec![0f32; t.n_elements() as usize];
    dequant_f32(gguf.tensor_data(t), &mut v);
    Ok(v)
}

const Q6_COMPACT_BLOCK_BYTES: usize = 212;

fn align_q6_decode(raw: &[u8], n_rows: usize, n_cols: usize, col_bytes: usize) -> Vec<u8> {
    let n_blocks = n_rows / 256;
    let compact_col_bytes = n_blocks * Q6_COMPACT_BLOCK_BYTES;
    let mut out = vec![0u8; n_cols * compact_col_bytes];
    for col in 0..n_cols {
        for bi in 0..n_blocks {
            let src = col * col_bytes + bi * 210;
            let dst = col * compact_col_bytes + bi * Q6_COMPACT_BLOCK_BYTES;
            out[dst..dst + 210].copy_from_slice(&raw[src..src + 210]);
        }
    }
    out
}

impl CudaModel {
    pub fn load(gguf: &GgufFile) -> Result<Self> {
        Self::load_with(gguf, DecodeBackend::default())
    }

    pub fn load_with(gguf: &GgufFile, decode: DecodeBackend) -> Result<Self> {
        let cfg = ModelConfig::from_gguf(gguf)?;
        eprintln!(
            "GPU | {} L={} d={} heads={}/{} ff={} | {:.2} GiB Q-weights | decode={}",
            cfg.architecture,
            cfg.n_layer,
            cfg.n_embd,
            cfg.n_head,
            cfg.n_head_kv,
            cfg.n_ff,
            gguf.total_tensor_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
            decode.name()
        );

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
        // Match the live GPU (e.g. sm_86 laptop 3050 Ti, sm_75 Tesla T4).
        // Hardcoding sm_86 breaks PTX load on other arches.
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
        let q4_dual_4way = cc_major >= 8;
        let q4_dual_symbol = if q4_dual_4way {
            "gemv_q4_k_dual_4way"
        } else {
            "gemv_q4_k_dual"
        };
        let q4_ffn_symbol = if q4_dual_4way {
            "gemv_q4_k_ffn_4way"
        } else {
            "gemv_q4_k_ffn"
        };
        let q4_ffn_smem_symbol = if q4_dual_4way {
            "gemv_q4_k_ffn_4way_smem"
        } else {
            "gemv_q4_k_ffn"
        };
        let q4_dual_threads = if q4_dual_4way { 128 } else { 64 };
        // Ampere+: keep modest warps for high-register Q4 DP4A paths; Turing
        // prefers fat blocks. Override with TARAFER_GEMV_WARPS.
        let gemv_quantized_warps = std::env::var("TARAFER_GEMV_WARPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&w: &u32| (1..=32).contains(&w))
            .unwrap_or(if cc_major >= 8 { 16 } else { 32 });
        eprintln!(
            "tuning | q4_dual={} | threads={} | quantized_warps={}",
            if q4_dual_4way { "4way" } else { "2way" },
            q4_dual_threads,
            gemv_quantized_warps
        );
        let k = Kernels {
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
            gemv_q4_dual_threads: q4_dual_threads,
            gemv_quantized_warps,
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
            embed_q4_one: module.load_function("embed_q4_k_one")?,
            embed_q5_one: module.load_function("embed_q5_0_one")?,
            embed_q5k_one: module.load_function("embed_q5_k_one")?,
            embed_q6_one: module.load_function("embed_q6_k_one")?,
            embed_q8_one: module.load_function("embed_q8_0_one")?,
            embed_q4_one_d: module.load_function("embed_q4_k_one_d")?,
            embed_q5_one_d: module.load_function("embed_q5_0_one_d")?,
            embed_q5k_one_d: module.load_function("embed_q5_k_one_d")?,
            embed_q6_one_d: module.load_function("embed_q6_k_one_d")?,
            embed_q8_one_d: module.load_function("embed_q8_0_one_d")?,
            rms_norm: module.load_function("rms_norm_f32")?,
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
                // Load only symbols listed in decode::REGISTRY (easy add/remove).
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
        };

        // Q5_K f32-dot GEMV is much slower than Q8_0 on sm_86 decode.
        // Hybrid Qwen3.5 puts Q5_K on the hot path (attn_qkv / ssm_out) — auto
        // transcode unless the user forces native Q5_K.
        // Override: TARAFER_Q5K_TRANSCODE=1 always; TARAFER_Q5K_NATIVE=1 never.
        let q5k_fallback = if std::env::var_os("TARAFER_Q5K_NATIVE").is_some() {
            false
        } else if std::env::var_os("TARAFER_Q5K_TRANSCODE").is_some() {
            true
        } else {
            cfg.is_hybrid()
        };
        let q5k_transcoded = std::cell::Cell::new(0usize);
        let q5_1_transcoded = std::cell::Cell::new(0usize);
        let upload_mat = |name: &str| -> Result<GpuMat> {
            let t = gguf.tensor(name).ok_or_else(|| anyhow!("missing {name}"))?;
            let n_rows = t.dims[0] as usize;
            let n_cols = *t.dims.get(1).unwrap_or(&1) as usize;
            let source_raw = gguf.tensor_data(t);
            let converted = match t.ggml_type {
                GgmlType::Q5_K if q5k_fallback => {
                    q5k_transcoded.set(q5k_transcoded.get() + 1);
                    Some(
                        transcode_q5_k_to_q8_0(source_raw)
                            .with_context(|| format!("transcode {name} Q5_K -> Q8_0"))?,
                    )
                }
                GgmlType::Q5_1 => {
                    q5_1_transcoded.set(q5_1_transcoded.get() + 1);
                    Some(
                        transcode_q5_1_to_q8_0(source_raw)
                            .with_context(|| format!("transcode {name} Q5_1 -> Q8_0"))?,
                    )
                }
                _ => None,
            };
            let (raw, wtype, col_bytes) = if let Some(ref data) = converted {
                (
                    data.as_slice(),
                    WType::Q8_0,
                    GgmlType::Q8_0.nbytes(n_rows as u64) as usize,
                )
            } else {
                (
                    source_raw,
                    wtype_of(t.ggml_type)?,
                    t.ggml_type.nbytes(n_rows as u64) as usize,
                )
            };
            let decode_data = None;
            let decode_col_bytes = col_bytes;
            let (compact_data, compact_col_bytes) = if wtype == WType::Q6K {
                let compact = align_q6_decode(raw, n_rows, n_cols, col_bytes);
                (
                    Some(stream.clone_htod(&compact)?),
                    (n_rows / 256) * Q6_COMPACT_BLOCK_BYTES,
                )
            } else {
                (None, 0)
            };
            Ok(GpuMat {
                data: stream.clone_htod(raw)?,
                decode_data,
                compact_data,
                n_rows,
                n_cols,
                col_bytes,
                decode_col_bytes,
                compact_col_bytes,
                wtype,
            })
        };
        // Default: plain RMS weights (GGUF stores usable scales for llama.cpp).
        // Set TARAFER_RMS_ONE_PLUS=1 to bake (1+w) if a convert left raw HF weights.
        let one_plus = cfg.is_hybrid() && std::env::var_os("TARAFER_RMS_ONE_PLUS").is_some();
        if one_plus {
            eprintln!("approx | RMSNorm bake (1+w) for qwen35");
        }
        let upload_vec = |name: &str| -> Result<_> {
            let mut v = f32_host(gguf, name)?;
            if one_plus {
                for x in &mut v {
                    *x += 1.0;
                }
            }
            Ok(stream.clone_htod(&v)?)
        };
        let upload_vec_plain = |name: &str| -> Result<_> {
            Ok(stream.clone_htod(&f32_host(gguf, name)?)?)
        };

        let mut token_embd = upload_mat("token_embd.weight")?;
        let output_norm = upload_vec("output_norm.weight")?;
        let mut output = upload_mat("output.weight").ok();
        let mut output_special = None;
        let mut output_special_id = None;
        // Hybrid + huge vocab (Qwen3.5) pays a full-head GEMV every token. Default
        // a generous active shortlist for speed; full vocab with TARAFER_FULL_VOCAB=1.
        let vocab_limit = std::env::var("TARAFER_VOCAB_LIMIT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .or_else(|| {
                if cfg.is_hybrid()
                    && cfg.n_vocab > 100_000
                    && std::env::var_os("TARAFER_FULL_VOCAB").is_none()
                {
                    // Aggressive shortlist for single-stream decode speed on 4GB.
                    Some(49_152)
                } else {
                    None
                }
            });
        if let Some(limit) = vocab_limit.filter(|&n| n >= 1024 && n < cfg.n_vocab) {
            let eos = gguf
                .meta_u32("tokenizer.ggml.eos_token_id")
                .or_else(|| {
                    gguf.meta_u64("tokenizer.ggml.eos_token_id")
                        .map(|v| v as u32)
                });
            if let Some(eos) = eos.filter(|&id| id as usize >= limit && (id as usize) < cfg.n_vocab) {
                let name = if output.is_some() {
                    "output.weight"
                } else {
                    "token_embd.weight"
                };
                let t = gguf.tensor(name).ok_or_else(|| anyhow!("missing {name}"))?;
                let n_rows = t.dims[0] as usize;
                let col_bytes = t.ggml_type.nbytes(n_rows as u64) as usize;
                let raw = gguf.tensor_data(t);
                let start = eos as usize * col_bytes;
                let one_col = &raw[start..start + col_bytes];
                let wtype = wtype_of(t.ggml_type)?;
                let (compact_data, compact_col_bytes) = if wtype == WType::Q6K {
                    let compact = align_q6_decode(one_col, n_rows, 1, col_bytes);
                    (
                        Some(stream.clone_htod(&compact)?),
                        (n_rows / 256) * Q6_COMPACT_BLOCK_BYTES,
                    )
                } else {
                    (None, 0)
                };
                output_special = Some(GpuMat {
                    data: stream.clone_htod(one_col)?,
                    decode_data: None,
                    compact_data,
                    n_rows,
                    n_cols: 1,
                    col_bytes,
                    decode_col_bytes: col_bytes,
                    compact_col_bytes,
                    wtype,
                });
                output_special_id = Some(eos);
            }
            if let Some(ref mut matrix) = output {
                matrix.n_cols = limit;
            } else {
                token_embd.n_cols = limit;
            }
            eprintln!(
                "approximation | active_vocab={limit}/{} (low-id shortlist + eos={:?})",
                cfg.n_vocab, output_special_id
            );
        }

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let ffn_norm = upload_vec(&format!("{p}.ffn_norm.weight"))
                .or_else(|_| upload_vec(&format!("{p}.post_attention_norm.weight")))?;
            let kind = cfg
                .layer_kinds
                .get(i)
                .copied()
                .unwrap_or(LayerKind::FullAttention);
            let attn = match kind {
                LayerKind::FullAttention => LayerAttn::Full(FullAttnWeights {
                    wq: upload_mat(&format!("{p}.attn_q.weight"))?,
                    bq: upload_vec(&format!("{p}.attn_q.bias")).ok(),
                    wk: upload_mat(&format!("{p}.attn_k.weight"))?,
                    bk: upload_vec(&format!("{p}.attn_k.bias")).ok(),
                    wv: upload_mat(&format!("{p}.attn_v.weight"))?,
                    bv: upload_vec(&format!("{p}.attn_v.bias")).ok(),
                    wo: upload_mat(&format!("{p}.attn_output.weight"))?,
                    attn_q_norm: upload_vec(&format!("{p}.attn_q_norm.weight")).ok(),
                    attn_k_norm: upload_vec(&format!("{p}.attn_k_norm.weight")).ok(),
                    fused_q_gate: cfg.fused_q_gate,
                }),
                LayerKind::LinearAttention => {
                    let conv_t = gguf
                        .tensor(&format!("{p}.ssm_conv1d.weight"))
                        .ok_or_else(|| anyhow!("missing {p}.ssm_conv1d.weight"))?;
                    // GGUF dims [kernel, channels] for f32.
                    let conv_kernel = conv_t.dims[0] as usize;
                    let conv_channels = conv_t.dims.get(1).copied().unwrap_or(0) as usize;
                    let conv_raw = gguf.tensor_data(conv_t);
                    let conv_f32: Vec<f32> = conv_raw
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    LayerAttn::Linear(LinearAttnWeights {
                        wqkv: upload_mat(&format!("{p}.attn_qkv.weight"))?,
                        w_gate: upload_mat(&format!("{p}.attn_gate.weight"))?,
                        conv1d: stream.clone_htod(&conv_f32)?,
                        conv_kernel,
                        conv_channels,
                        ssm_a: upload_vec_plain(&format!("{p}.ssm_a"))?,
                        ssm_dt: upload_vec_plain(&format!("{p}.ssm_dt.bias"))?,
                        ssm_alpha: upload_mat(&format!("{p}.ssm_alpha.weight"))?,
                        ssm_beta: upload_mat(&format!("{p}.ssm_beta.weight"))?,
                        // Gated RMSNorm uses plain weight (not 1+w).
                        ssm_norm: upload_vec_plain(&format!("{p}.ssm_norm.weight"))?,
                        ssm_out: upload_mat(&format!("{p}.ssm_out.weight"))?,
                        n_k_heads: cfg.ssm_n_k_heads,
                        n_v_heads: cfg.ssm_n_v_heads,
                        state_size: cfg.ssm_state_size,
                    })
                }
            };
            layers.push(GpuLayer {
                attn_norm: upload_vec(&format!("{p}.attn_norm.weight"))?,
                ffn_norm,
                attn,
                ffn: LayerFfn::Dense {
                    gate: upload_mat(&format!("{p}.ffn_gate.weight"))?,
                    up: upload_mat(&format!("{p}.ffn_up.weight"))?,
                    down: upload_mat(&format!("{p}.ffn_down.weight"))?,
                },
            });
        }
        if q5k_transcoded.get() != 0 || q5_1_transcoded.get() != 0 {
            eprintln!(
                "load fast-path | Q5_K -> Q8_0={} Q5_1 -> Q8_0={} tensors (hybrid decode; set TARAFER_Q5K_NATIVE=1 to keep Q5_K)",
                q5k_transcoded.get(),
                q5_1_transcoded.get()
            );
        }
        let mut type_counts = [0usize; 5];
        let mut type_bytes = [0usize; 5];
        let mut count_mat = |m: &GpuMat| {
            let idx = match m.wtype {
                WType::Q4K => 0,
                WType::Q4_0 => 0,
                WType::Q5K => 1,
                WType::Q5_0 => 2,
                WType::Q6K => 3,
                WType::Q8_0 => 4,
            };
            type_counts[idx] += 1;
            type_bytes[idx] += m.col_bytes * m.n_cols;
        };
        count_mat(&token_embd);
        if let Some(ref m) = output {
            count_mat(m);
        }
        for layer in &layers {
            match &layer.ffn {
                LayerFfn::Dense { gate, up, down } => {
                    count_mat(gate);
                    count_mat(up);
                    count_mat(down);
                }
                LayerFfn::Moe(m) => {
                    count_mat(&m.gate_all);
                    count_mat(&m.up_all);
                    count_mat(&m.down_all);
                }
            }
            match &layer.attn {
                LayerAttn::Full(a) => {
                    count_mat(&a.wq);
                    count_mat(&a.wk);
                    count_mat(&a.wv);
                    count_mat(&a.wo);
                }
                LayerAttn::Linear(a) => {
                    count_mat(&a.wqkv);
                    count_mat(&a.w_gate);
                    count_mat(&a.ssm_alpha);
                    count_mat(&a.ssm_beta);
                    count_mat(&a.ssm_out);
                }
            }
        }
        if cfg.is_hybrid() {
            let n_full = cfg
                .layer_kinds
                .iter()
                .filter(|k| matches!(k, LayerKind::FullAttention))
                .count();
            let n_lin = cfg.n_layer - n_full;
            eprintln!(
                "hybrid | qwen35  full_attn={n_full}  linear_gdn={n_lin}  rope_dim={}  ssm d_k={} heads_k/v={}/{}",
                cfg.rope_dim,
                cfg.ssm_state_size,
                cfg.ssm_n_k_heads,
                cfg.ssm_n_v_heads
            );
        }
        eprintln!(
            "weights | Q4_K={} ({:.2} GiB) Q5_K={} ({:.2} GiB) Q5_0={} ({:.2} GiB) Q6_K={} ({:.2} GiB) Q8_0={} ({:.2} GiB)",
            type_counts[0], type_bytes[0] as f64 / 1073741824.0,
            type_counts[1], type_bytes[1] as f64 / 1073741824.0,
            type_counts[2], type_bytes[2] as f64 / 1073741824.0,
            type_counts[3], type_bytes[3] as f64 / 1073741824.0,
            type_counts[4], type_bytes[4] as f64 / 1073741824.0,
        );
        // Ensure no pending default-stream work before inference/graphs.
        ctx.synchronize().context("ctx sync after load")?;
        eprintln!(
            "ready | decode={} | Q4_K×Q8 DP4A | fused projections | CUDA-graph ready",
            decode.name()
        );

        let n_embd = cfg.n_embd;
        let n_kv = cfg.n_head_kv * cfg.head_dim();
        let head_dim = cfg.head_dim();
        let n_q = cfg.n_head * head_dim;
        let n_q_proj = cfg.q_proj_dim();
        let ssm_qkv = if cfg.is_hybrid() {
            cfg.ssm_conv_channels().max(1)
        } else {
            1
        };
        let ssm_key = if cfg.is_hybrid() {
            cfg.ssm_key_dim().max(1)
        } else {
            1
        };
        let ssm_val = if cfg.is_hybrid() {
            cfg.ssm_value_dim().max(1)
        } else {
            1
        };
        let ssm_nv = cfg.ssm_n_v_heads.max(1);
        let b = MAX_BATCH;
        // Active vocab may be shortlisted (hybrid speed path).
        let active_vocab = output
            .as_ref()
            .map(|m| m.n_cols)
            .unwrap_or(token_embd.n_cols)
            + usize::from(output_special.is_some());
        // Split-K partials: enough for largest single-token GEMV (usually vocab).
        let gemv_partial_stride = active_vocab
            .max(cfg.n_ff)
            .max(n_embd)
            .max(n_kv)
            .max(n_q_proj)
            .max(ssm_qkv)
            .max(ssm_val);
        const GEMV_SPLIT_MAX: usize = 8;
        use super::decode::FLASH_MAX_SPLIT;
        let flash_partial_stride = 2 + head_dim;
        let flash_partial_n = cfg.n_head * FLASH_MAX_SPLIT as usize * flash_partial_stride;
        // Hybrid GDN updates fixed device buffers in-place — CUDA graphs work
        // when capture saves/restores recurrent state (see try_capture_decode_graph).
        if cfg.is_hybrid() {
            eprintln!("CUDA graph | hybrid ok (GDN/conv state fixed pointers; capture saves/restores)");
        }
        Ok(Self {
            x: stream.alloc_zeros(b * n_embd)?,
            // Qwen3 can have n_head*head_dim > n_embd; xb is reused for
            // normalized hidden state and the attention result.
            xb: stream.alloc_zeros(b * n_embd.max(n_q).max(ssm_val))?,
            xb2: stream.alloc_zeros(b * n_embd.max(ssm_val))?,
            q: stream.alloc_zeros(b * n_q_proj.max(ssm_qkv))?,
            k_buf: stream.alloc_zeros(b * n_kv.max(ssm_key))?,
            v_buf: stream.alloc_zeros(b * n_kv.max(ssm_val))?,
            hb: stream.alloc_zeros(b * cfg.n_ff)?,
            hb2: stream.alloc_zeros(b * cfg.n_ff)?,
            x1: stream.alloc_zeros(n_embd)?,
            xb1: stream.alloc_zeros(n_embd)?,
            logits: stream.alloc_zeros(active_vocab)?,
            special_logit: stream.alloc_zeros(1)?,
            logits_batch: stream
                .alloc_zeros(super::types::MAX_VERIFY_TOKENS * active_vocab)?,
            argmax_buf: stream.alloc_zeros(1)?,
            argmax_batch: stream.alloc_zeros(super::types::MAX_VERIFY_TOKENS)?,
            tok_buf: stream.alloc_zeros(MAX_BATCH)?,
            q8_x: stream.alloc_zeros(cfg.n_ff.max(n_embd).max(ssm_qkv).max(ssm_val))?,
            q8_d: stream.alloc_zeros((cfg.n_ff.max(n_embd).max(ssm_qkv).max(ssm_val) + 31) / 32)?,
            gemv_partial: stream.alloc_zeros(GEMV_SPLIT_MAX * gemv_partial_stride)?,
            gemv_partial_stride,
            d_pos0: stream.alloc_zeros(1)?,
            d_token: stream.alloc_zeros(1)?,
            flash_partial: stream.alloc_zeros(flash_partial_n)?,
            gate_buf: stream.alloc_zeros(b * n_q.max(ssm_val))?,
            gdn_q: stream.alloc_zeros(b * ssm_key)?,
            gdn_k: stream.alloc_zeros(b * ssm_key)?,
            gdn_v: stream.alloc_zeros(b * ssm_val)?,
            gdn_z: stream.alloc_zeros(b * ssm_val)?,
            gdn_alpha: stream.alloc_zeros(b * ssm_nv)?,
            gdn_beta: stream.alloc_zeros(b * ssm_nv)?,
            gdn_conv: stream.alloc_zeros(b * ssm_qkv)?,
            gdn_out: stream.alloc_zeros(b * ssm_val)?,
            decode_graph: None,
            graph_tried: false,
            cuda_graph: true,
            graph_active: false,
            moe_idx: stream.alloc_zeros(8)?,
            moe_w: stream.alloc_zeros(8)?,
            cfg,
            decode,
            gpu_name,
            compute_major: cc_major,
            compute_minor: cc_minor,
            nvrtc_arch: arch,
            stream,
            _ctx: ctx,
            _module: module,
            k,
            token_embd,
            output_norm,
            output,
            output_special,
            output_special_id,
            layers,
        })
    }
}
