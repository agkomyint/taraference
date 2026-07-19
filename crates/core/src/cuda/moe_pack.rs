//! Load Tara MoE Q8 pack directories (`meta.json` + per-tensor files).
//!
//! Pack layout is produced by `taraference_750_department/scripts/export_moe_q8_pack.py`.

use super::decode::DecodeBackend;
use super::kernels::SOURCE;
use super::model::CudaModel;
use super::types::{
    FullAttnWeights, GpuLayer, GpuMat, Kernels, LayerAttn, LayerFfn, MoeFfnWeights, WType, MAX_BATCH,
};
use crate::config::{LayerKind, ModelConfig};
use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::CudaContext;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct PackMeta {
    format: String,
    #[serde(default)]
    quant: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    arch: String,
    n_vocab: usize,
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    n_head_kv: usize,
    n_ff: usize,
    n_experts: usize,
    router_top_k: usize,
    expert_ff: usize,
    num_dense_layers: usize,
    rope_theta: f32,
    rms_eps: f32,
    n_ctx: usize,
    head_dim: usize,
    layers: Vec<PackLayerMeta>,
    files: HashMap<String, PackFileMeta>,
}

#[derive(Debug, Deserialize)]
struct PackLayerMeta {
    i: usize,
    kind: String,
}

#[derive(Debug, Deserialize)]
struct PackFileMeta {
    n_rows: usize,
    n_cols: usize,
    #[allow(dead_code)]
    nbytes: usize,
}

const Q8_0_BLOCK: usize = 34;
const Q4_0_BLOCK: usize = 18;
/// Padded Q4_0 block for vectorized loads: f16 d + 2B pad + 16B qs + pad.
const Q4_0_ALIGN_BLOCK: usize = 32;

fn q8_col_bytes(n_rows: usize) -> usize {
    (n_rows / 32) * Q8_0_BLOCK
}

fn q4_col_bytes(n_rows: usize) -> usize {
    (n_rows / 32) * Q4_0_BLOCK
}

fn q4_align_col_bytes(n_rows: usize) -> usize {
    (n_rows / 32) * Q4_0_ALIGN_BLOCK
}

/// IEEE f32 → f16 bits (round-to-nearest-even-ish).
fn f32_to_f16_bits(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mant = x & 0x7f_ffff;
    let exp = ((x >> 23) & 0xff) as i32;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    if exp == 0 {
        return sign; // flush denorm to 0
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00; // inf
    }
    if new_exp <= 0 {
        return sign; // underflow
    }
    let new_mant = mant + 0x1000; // round
    sign | ((new_exp as u16) << 10) | ((new_mant >> 13) as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let bits = if exp == 0 {
        sign
    } else if exp == 31 {
        sign | 0x7f80_0000 | ((mant as u32) << 13)
    } else {
        sign | (((exp as u32) - 15 + 127) << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(bits)
}

/// Column-major Q4_0 [n_cols][n_blocks][18] → block-major [n_blocks][n_cols][18].
fn repack_q4_0_block_major(src: &[u8], n_rows: usize, n_cols: usize) -> Result<Vec<u8>> {
    let n_blocks = n_rows / 32;
    let col_stride = n_blocks * Q4_0_BLOCK;
    if src.len() != col_stride * n_cols {
        bail!("q4 bm size mismatch");
    }
    let mut out = vec![0u8; src.len()];
    for j in 0..n_cols {
        for bi in 0..n_blocks {
            let s = j * col_stride + bi * Q4_0_BLOCK;
            let d = bi * n_cols * Q4_0_BLOCK + j * Q4_0_BLOCK;
            out[d..d + Q4_0_BLOCK].copy_from_slice(&src[s..s + Q4_0_BLOCK]);
        }
    }
    Ok(out)
}

/// Dequant Q4_0 file bytes → column-major f16 (u16 LE bits).
fn dequant_q4_0_to_f16(src: &[u8], n_rows: usize, n_cols: usize) -> Result<Vec<u8>> {
    let n_blocks = n_rows / 32;
    let src_col = n_blocks * Q4_0_BLOCK;
    if src.len() != src_col * n_cols {
        bail!(
            "q4→f16 size mismatch: got {} want {}",
            src.len(),
            src_col * n_cols
        );
    }
    let mut out = vec![0u8; n_cols * n_rows * 2];
    for j in 0..n_cols {
        let sbase = j * src_col;
        let dbase = j * n_rows;
        for bi in 0..n_blocks {
            let s = sbase + bi * Q4_0_BLOCK;
            let d_bits = u16::from_le_bytes([src[s], src[s + 1]]);
            let d = f16_bits_to_f32(d_bits);
            let qs = &src[s + 2..s + 18];
            for t in 0..16 {
                let lo = (qs[t] & 0x0f) as i32 - 8;
                let hi = (qs[t] >> 4) as i32 - 8;
                let i0 = bi * 32 + t;
                let i1 = bi * 32 + 16 + t;
                let b0 = f32_to_f16_bits(lo as f32 * d).to_le_bytes();
                let b1 = f32_to_f16_bits(hi as f32 * d).to_le_bytes();
                let o0 = (dbase + i0) * 2;
                let o1 = (dbase + i1) * 2;
                out[o0] = b0[0];
                out[o0 + 1] = b0[1];
                out[o1] = b1[0];
                out[o1 + 1] = b1[1];
            }
        }
    }
    Ok(out)
}

/// Repack file Q4_0 (18B/block) → 32B aligned blocks for faster GEMV.
fn repack_q4_0_align32(src: &[u8], n_rows: usize, n_cols: usize) -> Result<Vec<u8>> {
    let n_blocks = n_rows / 32;
    let src_col = n_blocks * Q4_0_BLOCK;
    let dst_col = n_blocks * Q4_0_ALIGN_BLOCK;
    if src.len() != src_col * n_cols {
        bail!(
            "q4 repack size mismatch: got {} want {}",
            src.len(),
            src_col * n_cols
        );
    }
    let mut out = vec![0u8; dst_col * n_cols];
    for j in 0..n_cols {
        let sbase = j * src_col;
        let dbase = j * dst_col;
        for bi in 0..n_blocks {
            let s = sbase + bi * Q4_0_BLOCK;
            let d = dbase + bi * Q4_0_ALIGN_BLOCK;
            // f16 scale
            out[d] = src[s];
            out[d + 1] = src[s + 1];
            // bytes 2-3 pad; qs at 4..20 (4-byte aligned)
            out[d + 4..d + 20].copy_from_slice(&src[s + 2..s + 18]);
        }
    }
    Ok(out)
}

impl CudaModel {
    /// Load a Tara MoE Q8 pack directory (`format: tara_moe_q8_v1`).
    pub fn load_tara_moe_pack(dir: &Path, decode: DecodeBackend) -> Result<Self> {
        let meta_path = dir.join("meta.json");
        let meta_raw = fs::read_to_string(&meta_path)
            .with_context(|| format!("read {}", meta_path.display()))?;
        let meta: PackMeta = serde_json::from_str(&meta_raw).context("parse meta.json")?;
        let is_q4 = meta.format == "tara_moe_q4_v1" || meta.quant.as_deref() == Some("q4_0");
        if meta.format != "tara_moe_q8_v1" && !is_q4 {
            bail!(
                "unsupported pack format {:?} (want tara_moe_q8_v1 or tara_moe_q4_v1)",
                meta.format
            );
        }
        if meta.n_embd % 32 != 0 {
            bail!("n_embd={} must be multiple of 32 for Q8_0", meta.n_embd);
        }
        if meta.expert_ff % 32 != 0 {
            bail!("expert_ff={} must be multiple of 32 for Q8_0", meta.expert_ff);
        }

        let n_ff = meta.n_ff.max(meta.expert_ff);
        let cfg = ModelConfig {
            architecture: meta.arch.clone(),
            n_vocab: meta.n_vocab,
            n_embd: meta.n_embd,
            n_layer: meta.n_layer,
            n_head: meta.n_head,
            n_head_kv: meta.n_head_kv,
            attention_head_dim: meta.head_dim,
            no_rope_layer_interval: 0,
            n_ff,
            // Product target for 100M-active MoE: solid ~500 tok/s through ctx≤1024.
            // Default floor 1024 (not 4096) so KV/graph match the speed SKU.
            // Longer chat: TARAFER_N_CTX=4096 (or pass --ctx). Strict: TARAFER_STRICT_CTX=1.
            n_ctx: {
                let env_ctx = std::env::var("TARAFER_N_CTX")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|&n| n >= 256);
                if std::env::var_os("TARAFER_STRICT_CTX").is_some() {
                    env_ctx.unwrap_or(meta.n_ctx)
                } else {
                    // Product default: at least 1024 (500 tok/s band), no forced 4k floor.
                    env_ctx.unwrap_or(meta.n_ctx.max(1024))
                }
            },
            rope_theta: meta.rope_theta,
            rms_eps: meta.rms_eps,
            rope_dim: meta.head_dim,
            fused_q_gate: false,
            ssm_conv_kernel: 0,
            ssm_state_size: 0,
            ssm_n_k_heads: 0,
            ssm_n_v_heads: 0,
            ssm_inner_size: 0,
            layer_kinds: vec![LayerKind::FullAttention; meta.n_layer],
            n_experts: meta.n_experts,
            router_top_k: meta.router_top_k,
            expert_ff: meta.expert_ff,
            num_dense_layers: meta.num_dense_layers,
        };

        let mut weight_bytes = 0u64;
        for e in fs::read_dir(dir).with_context(|| format!("list {}", dir.display()))? {
            let e = e?;
            if e.file_type()?.is_file() {
                weight_bytes += e.metadata()?.len();
            }
        }
        if is_q4 {
            let mode = if std::env::var_os("TARAFER_MOE_F16").is_some() {
                "experts Q4→f16"
            } else if std::env::var_os("TARAFER_MOE_BM").is_some() {
                "experts Q4 block-major"
            } else {
                "experts Q4 column-major"
            };
            eprintln!("decode | {mode}");
        }
        eprintln!(
            "GPU | {} MoE L={} dense_ffn={} moe_ffn={} d={} heads={}/{} ff={} experts={} top_k={} expert_ff={} n_ctx={} | {:.2} GiB pack | decode={}",
            cfg.architecture,
            cfg.n_layer,
            cfg.num_dense_layers,
            cfg.n_layer.saturating_sub(cfg.num_dense_layers),
            cfg.n_embd,
            cfg.n_head,
            cfg.n_head_kv,
            cfg.n_ff,
            cfg.n_experts,
            cfg.router_top_k,
            cfg.expert_ff,
            cfg.n_ctx,
            weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            decode.name()
        );

        let ctx = CudaContext::new(0).context("CudaContext")?;
        unsafe {
            ctx.disable_event_tracking();
        }
        let stream = ctx.new_stream().context("CudaStream (non-blocking)")?;
        let (cc_major, cc_minor) = ctx
            .compute_capability()
            .context("device compute capability")?;
        let arch = format!("sm_{cc_major}{cc_minor}");
        let gpu_name = ctx
            .name()
            .unwrap_or_else(|_| format!("CUDA device 0 (sm_{cc_major}{cc_minor})"));
        eprintln!("GPU device | {gpu_name} | compute {cc_major}.{cc_minor} | NVRTC arch={arch}");
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
        };

        let upload_mat = |stem: &str| -> Result<GpuMat> {
            // Prefer .q4 then .q8 (Q4 packs mix embd.q8 + weight.q4).
            let (name, is_q4_file) = if meta.files.contains_key(&format!("{stem}.q4")) {
                (format!("{stem}.q4"), true)
            } else if meta.files.contains_key(&format!("{stem}.q8")) {
                (format!("{stem}.q8"), false)
            } else {
                bail!("meta.files missing {stem}.q4 or {stem}.q8");
            };
            let fi = meta.files.get(&name).unwrap();
            let path = dir.join(&name);
            let raw = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            if is_q4_file {
                let expect = q4_col_bytes(fi.n_rows) * fi.n_cols;
                if raw.len() != expect {
                    bail!(
                        "{name}: got {} bytes, expected {expect} (rows={} cols={})",
                        raw.len(),
                        fi.n_rows,
                        fi.n_cols
                    );
                }
                // Optional 32B-aligned repack (TARAFER_Q4_ALIGN=1). Default keeps 18B
                // (less weight traffic); align can help some GPUs with vector loads.
                let use_align = std::env::var_os("TARAFER_Q4_ALIGN").is_some();
                if use_align {
                    let aligned = repack_q4_0_align32(&raw, fi.n_rows, fi.n_cols)?;
                    let col_bytes = q4_align_col_bytes(fi.n_rows);
                    Ok(GpuMat {
                        data: stream.clone_htod(&aligned)?,
                        decode_data: None,
                        compact_data: None,
                        n_rows: fi.n_rows,
                        n_cols: fi.n_cols,
                        col_bytes,
                        decode_col_bytes: col_bytes,
                        compact_col_bytes: 0,
                        wtype: WType::Q4_0,
                    })
                } else {
                    let col_bytes = q4_col_bytes(fi.n_rows);
                    Ok(GpuMat {
                        data: stream.clone_htod(&raw)?,
                        decode_data: None,
                        compact_data: None,
                        n_rows: fi.n_rows,
                        n_cols: fi.n_cols,
                        col_bytes,
                        decode_col_bytes: col_bytes,
                        compact_col_bytes: 0,
                        wtype: WType::Q4_0,
                    })
                }
            } else {
                let col_bytes = q8_col_bytes(fi.n_rows);
                let expect = col_bytes * fi.n_cols;
                if raw.len() != expect {
                    bail!(
                        "{name}: got {} bytes, expected {expect} (rows={} cols={})",
                        raw.len(),
                        fi.n_rows,
                        fi.n_cols
                    );
                }
                Ok(GpuMat {
                    data: stream.clone_htod(&raw)?,
                    decode_data: None,
                    compact_data: None,
                    n_rows: fi.n_rows,
                    n_cols: fi.n_cols,
                    col_bytes,
                    decode_col_bytes: col_bytes,
                    compact_col_bytes: 0,
                    wtype: WType::Q8_0,
                })
            }
        };
        let upload_f32_vec = |name: &str, n: usize| -> Result<_> {
            let path = dir.join(name);
            let raw = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            if raw.len() != n * 4 {
                bail!("{name}: got {} bytes, expected {}", raw.len(), n * 4);
            }
            let mut v = vec![0f32; n];
            for (i, chunk) in raw.chunks_exact(4).enumerate() {
                v[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(stream.clone_htod(&v)?)
        };

        let mut token_embd = upload_mat("token_embd")?;
        let output_norm = upload_f32_vec("output_norm.f32", cfg.n_embd)?;
        // Tied embeddings: no separate output head.
        let output = None;
        let output_special = None;
        let output_special_id = None;
        // Vocab head shortlist for 750 path (full vocab: TARAFER_FULL_VOCAB=1).
        // Default 8192 on speed packs (top_k=1 or mode contains "speed").
        let speed_pack = meta.router_top_k <= 1
            || meta
                .mode
                .as_deref()
                .map(|m| m.contains("speed"))
                .unwrap_or(false);
        let vocab_limit = if std::env::var_os("TARAFER_FULL_VOCAB").is_some() {
            None
        } else {
            std::env::var("TARAFER_VOCAB_LIMIT")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .or_else(|| {
                    if std::env::var_os("TARAFER_SPEED").is_some() || speed_pack {
                        Some(8_192)
                    } else {
                        None
                    }
                })
        };
        if let Some(limit) = vocab_limit.filter(|&n| n >= 1024 && n < cfg.n_vocab) {
            token_embd.n_cols = limit;
            eprintln!(
                "approximation | active_vocab={limit}/{} (MoE 750 shortlist; TARAFER_FULL_VOCAB=1 for full)",
                cfg.n_vocab
            );
        }

        let pack_experts = |kind: &str,
                            n_rows: usize,
                            n_cols_e: usize,
                            layer_i: usize|
         -> Result<GpuMat> {
            let mut packed = Vec::new();
            let mut col_bytes = 0usize;
            let mut wtype = WType::Q8_0;
            for e in 0..meta.n_experts {
                let stem = format!("blk.{layer_i}.exp{e}.ffn_{kind}");
                let (name, is_q4_file) = if meta.files.contains_key(&format!("{stem}.q4")) {
                    (format!("{stem}.q4"), true)
                } else {
                    (format!("{stem}.q8"), false)
                };
                let fi = meta
                    .files
                    .get(&name)
                    .ok_or_else(|| anyhow!("meta.files missing {name}"))?;
                if fi.n_rows != n_rows || fi.n_cols != n_cols_e {
                    bail!(
                        "{name}: shape {}x{}, expected {n_rows}x{n_cols_e}",
                        fi.n_rows,
                        fi.n_cols
                    );
                }
                let path = dir.join(&name);
                let raw = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
                if is_q4_file {
                    let expect = q4_col_bytes(n_rows) * n_cols_e;
                    if raw.len() != expect {
                        bail!("{name}: got {} bytes, expected {expect}", raw.len());
                    }
                    // Default: col-major Q4 (best measured on short-K d=640; max CTA fill).
                    // TARAFER_MOE_BM=1 → block-major layout + BM kernels.
                    // TARAFER_MOE_F16=1 → expand f16 (usually slower / more BW).
                    if std::env::var_os("TARAFER_MOE_F16").is_some() {
                        let f16b = dequant_q4_0_to_f16(&raw, n_rows, n_cols_e)?;
                        col_bytes = n_rows * 2;
                        wtype = WType::F16;
                        packed.extend_from_slice(&f16b);
                    } else if std::env::var_os("TARAFER_MOE_BM").is_some() {
                        let bm = repack_q4_0_block_major(&raw, n_rows, n_cols_e)?;
                        col_bytes = q4_col_bytes(n_rows); // logical; kernel uses BM layout
                        wtype = WType::Q4_0_BM;
                        packed.extend_from_slice(&bm);
                    } else {
                        col_bytes = q4_col_bytes(n_rows);
                        wtype = WType::Q4_0;
                        packed.extend_from_slice(&raw);
                    }
                } else {
                    let cb = q8_col_bytes(n_rows);
                    let expect = cb * n_cols_e;
                    if raw.len() != expect {
                        bail!("{name}: got {} bytes, expected {expect}", raw.len());
                    }
                    col_bytes = cb;
                    wtype = WType::Q8_0;
                    packed.extend_from_slice(&raw);
                }
            }
            Ok(GpuMat {
                data: stream.clone_htod(&packed)?,
                decode_data: None,
                compact_data: None,
                n_rows,
                n_cols: n_cols_e * meta.n_experts,
                col_bytes,
                decode_col_bytes: col_bytes,
                compact_col_bytes: 0,
                wtype,
            })
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let kind = meta
                .layers
                .iter()
                .find(|l| l.i == i)
                .map(|l| l.kind.as_str())
                .unwrap_or(if i < meta.num_dense_layers {
                    "dense"
                } else {
                    "moe"
                });
            let attn = LayerAttn::Full(FullAttnWeights {
                wq: upload_mat(&format!("blk.{i}.attn_q"))?,
                bq: None,
                wk: upload_mat(&format!("blk.{i}.attn_k"))?,
                bk: None,
                wv: upload_mat(&format!("blk.{i}.attn_v"))?,
                bv: None,
                wo: upload_mat(&format!("blk.{i}.attn_o"))?,
                attn_q_norm: None,
                attn_k_norm: None,
                fused_q_gate: false,
            });
            let ffn = if kind == "dense" {
                LayerFfn::Dense {
                    gate: upload_mat(&format!("blk.{i}.ffn_gate"))?,
                    up: upload_mat(&format!("blk.{i}.ffn_up"))?,
                    down: upload_mat(&format!("blk.{i}.ffn_down"))?,
                }
            } else {
                let router_elems = meta.n_experts * cfg.n_embd;
                let router = upload_f32_vec(&format!("blk.{i}.router.f32"), router_elems)?;
                let gate_all = pack_experts("gate", cfg.n_embd, meta.expert_ff, i)?;
                let up_all = pack_experts("up", cfg.n_embd, meta.expert_ff, i)?;
                let down_all = pack_experts("down", meta.expert_ff, cfg.n_embd, i)?;
                LayerFfn::Moe(MoeFfnWeights {
                    router,
                    n_experts: meta.n_experts,
                    top_k: meta.router_top_k,
                    expert_ff: meta.expert_ff,
                    gate_all,
                    up_all,
                    down_all,
                })
            };
            layers.push(GpuLayer {
                attn_norm: upload_f32_vec(&format!("blk.{i}.attn_norm.f32"), cfg.n_embd)?,
                ffn_norm: upload_f32_vec(&format!("blk.{i}.ffn_norm.f32"), cfg.n_embd)?,
                attn,
                ffn,
            });
        }

        eprintln!(
            "moe | loaded {} layers ({} dense + {} sparse) top_k={} experts={}",
            cfg.n_layer,
            cfg.num_dense_layers,
            cfg.n_layer.saturating_sub(cfg.num_dense_layers),
            cfg.router_top_k,
            cfg.n_experts
        );
        eprintln!(
            "weights | {:?} tied embedding/head on device (no separate output head)",
            token_embd.wtype
        );
        ctx.synchronize().context("ctx sync after load")?;
        eprintln!(
            "ready | decode={} | MoE device top-k (packed experts, graph-ready)",
            decode.name()
        );

        let n_embd = cfg.n_embd;
        let n_kv = cfg.n_head_kv * cfg.head_dim();
        let head_dim = cfg.head_dim();
        let n_q = cfg.n_head * head_dim;
        let n_q_proj = cfg.q_proj_dim();
        let b = MAX_BATCH;
        let active_vocab = token_embd.n_cols;
        let gemv_partial_stride = active_vocab
            .max(cfg.n_ff)
            .max(n_embd)
            .max(n_kv)
            .max(n_q_proj);
        const GEMV_SPLIT_MAX: usize = 8;
        use super::decode::FLASH_MAX_SPLIT;
        let flash_partial_stride = 2 + head_dim;
        let flash_partial_n = cfg.n_head * FLASH_MAX_SPLIT as usize * flash_partial_stride;

        Ok(Self {
            x: stream.alloc_zeros(b * n_embd)?,
            xb: stream.alloc_zeros(b * n_embd.max(n_q))?,
            xb2: stream.alloc_zeros(b * n_embd)?,
            q: stream.alloc_zeros(b * n_q_proj)?,
            k_buf: stream.alloc_zeros(b * n_kv)?,
            v_buf: stream.alloc_zeros(b * n_kv)?,
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
            q8_x: stream.alloc_zeros(cfg.n_ff.max(n_embd))?,
            q8_d: stream.alloc_zeros((cfg.n_ff.max(n_embd) + 31) / 32)?,
            q8_ff: stream.alloc_zeros(cfg.n_ff.max(32))?,
            q8_ff_d: stream.alloc_zeros((cfg.n_ff.max(32) + 31) / 32)?,
            gemv_partial: stream.alloc_zeros(GEMV_SPLIT_MAX * gemv_partial_stride)?,
            gemv_partial_stride,
            d_pos0: stream.alloc_zeros(1)?,
            d_token: stream.alloc_zeros(1)?,
            flash_partial: stream.alloc_zeros(flash_partial_n)?,
            gate_buf: stream.alloc_zeros(b * n_q)?,
            gdn_q: stream.alloc_zeros(1)?,
            gdn_k: stream.alloc_zeros(1)?,
            gdn_v: stream.alloc_zeros(1)?,
            gdn_z: stream.alloc_zeros(1)?,
            gdn_alpha: stream.alloc_zeros(1)?,
            gdn_beta: stream.alloc_zeros(1)?,
            gdn_conv: stream.alloc_zeros(1)?,
            gdn_out: stream.alloc_zeros(1)?,
            decode_graph: None,
            graph_tried: false,
            // Device-side top-k + packed experts → fixed launch structure → CUDA graphs OK.
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
