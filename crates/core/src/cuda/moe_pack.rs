//! Load Tara MoE Q8 pack directories (`meta.json` + per-tensor files).
//!
//! Pack layout is produced by `taraference_750_department/scripts/export_moe_q8_pack.py`.

use super::decode::DecodeBackend;
use super::device::init_device;
use super::model::CudaModel;
use super::types::{
    FullAttnWeights, GpuLayer, GpuMat, LayerAttn, LayerFfn, MoeFfnWeights, WType, MAX_BATCH,
};
use crate::config::{LayerKind, ModelConfig, RouterWeightMode};
use crate::quant::{f16_bits_to_f32_fast, f32_to_f16_bits_fast};
use anyhow::{anyhow, bail, Context, Result};
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
    #[serde(default)]
    router_weight_mode: Option<String>,
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
            let d = f16_bits_to_f32_fast(d_bits);
            let qs = &src[s + 2..s + 18];
            for t in 0..16 {
                let lo = (qs[t] & 0x0f) as i32 - 8;
                let hi = (qs[t] >> 4) as i32 - 8;
                let i0 = bi * 32 + t;
                let i1 = bi * 32 + 16 + t;
                let b0 = f32_to_f16_bits_fast(lo as f32 * d).to_le_bytes();
                let b1 = f32_to_f16_bits_fast(hi as f32 * d).to_le_bytes();
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
        if meta.n_experts == 0 || meta.n_experts > 64 {
            bail!("n_experts={} must be in 1..=64 for CUDA MoE", meta.n_experts);
        }
        if meta.router_top_k == 0 || meta.router_top_k > meta.n_experts || meta.router_top_k > 8 {
            bail!(
                "router_top_k={} must be in 1..=min(n_experts, 8)",
                meta.router_top_k
            );
        }
        let router_weight_mode = RouterWeightMode::from_metadata(meta.router_weight_mode.as_deref())?;

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
            router_weight_mode,
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
            "GPU | {} MoE L={} dense_ffn={} moe_ffn={} d={} heads={}/{} ff={} experts={} top_k={} router_weights={} expert_ff={} n_ctx={} | {:.2} GiB pack | decode={}",
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
            cfg.router_weight_mode.as_str(),
            cfg.expert_ff,
            cfg.n_ctx,
            weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            decode.name()
        );

        let device = init_device()?;
        let ctx = device.ctx;
        let stream = device.stream;
        let module = device.module;
        let k = device.kernels;
        let gpu_name = device.gpu_name;
        let cc_major = device.compute_major;
        let cc_minor = device.compute_minor;
        let arch = device.nvrtc_arch;

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
