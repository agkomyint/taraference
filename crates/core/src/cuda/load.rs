//! Load GGUF weights onto GPU and compile NVRTC kernels.

use super::decode::DecodeBackend;
use super::kernels::SOURCE;
use super::model::CudaModel;
use super::types::{GpuLayer, GpuMat, Kernels, WType, MAX_BATCH};
use crate::config::ModelConfig;
use crate::quant::dequant_f32;
use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::CudaContext;
use taraference_gguf::{GgmlType, GgufFile};

fn wtype_of(t: GgmlType) -> Result<WType> {
    Ok(match t {
        GgmlType::Q4_K => WType::Q4K,
        GgmlType::Q5_0 => WType::Q5_0,
        GgmlType::Q6_K => WType::Q6K,
        GgmlType::Q8_0 => WType::Q8_0,
        other => bail!(
            "unsupported weight type {} (supported: Q4_K, Q5_0, Q6_K, Q8_0)",
            other.name()
        ),
    })
}

fn f32_host(gguf: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let t = gguf
        .tensor(name)
        .ok_or_else(|| anyhow!("missing {name}"))?;
    if t.ggml_type != GgmlType::F32 {
        bail!("{name} not F32");
    }
    let mut v = vec![0f32; t.n_elements() as usize];
    dequant_f32(gguf.tensor_data(t), &mut v);
    Ok(v)
}

const Q6_REPACK_BLOCK_BYTES: usize = 276;

fn q6_value(base: &[u8], i: usize) -> i8 {
    let half = i >> 7;
    let within = i & 127;
    let segment = within >> 5;
    let lane = within & 31;
    let lo_byte = base[half * 64 + (segment & 1) * 32 + lane];
    let lo = if segment < 2 { lo_byte & 0x0f } else { lo_byte >> 4 };
    let hi = (base[128 + half * 32 + lane] >> (segment * 2)) & 3;
    ((lo | (hi << 4)) as i8) - 32
}

/// Expand Q6_K's split 4+2-bit representation once at load time.  Decode then
/// streams aligned signed bytes + the original per-16 scales and block scale.
fn repack_q6_decode(raw: &[u8], n_rows: usize, n_cols: usize, col_bytes: usize) -> Vec<u8> {
    let n_blocks = n_rows / 256;
    let decode_col_bytes = n_blocks * Q6_REPACK_BLOCK_BYTES;
    let mut out = vec![0u8; n_cols * decode_col_bytes];
    for col in 0..n_cols {
        for bi in 0..n_blocks {
            let src = &raw[col * col_bytes + bi * 210..][..210];
            let dst0 = col * decode_col_bytes + bi * Q6_REPACK_BLOCK_BYTES;
            for i in 0..256 {
                out[dst0 + i] = q6_value(src, i) as u8;
            }
            out[dst0 + 256..dst0 + 274].copy_from_slice(&src[192..210]);
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
        let k = Kernels {
            quantize_q8: module.load_function("quantize_q8_global")?,
            gemv_q4: module.load_function("gemv_q4_k")?,
            gemv_q4_global: module.load_function("gemv_q4_k_global")?,
            gemv_q5: module.load_function("gemv_q5_0")?,
            gemv_q6: module.load_function("gemv_q6_k")?,
            gemv_q6_repack: module.load_function("gemv_q6_k_repack")?,
            gemv_q6_repack_global: module.load_function("gemv_q6_k_repack_global")?,
            gemv_q8: module.load_function("gemv_q8_0")?,
            gemv_q4_splitk: module.load_function("gemv_q4_k_splitk")?,
            gemv_q5_splitk: module.load_function("gemv_q5_0_splitk")?,
            gemv_q6_splitk: module.load_function("gemv_q6_k_splitk")?,
            gemv_q6_repack_splitk: module.load_function("gemv_q6_k_repack_splitk")?,
            gemv_q6_repack_global_splitk: module
                .load_function("gemv_q6_k_repack_global_splitk")?,
            gemv_q8_splitk: module.load_function("gemv_q8_0_splitk")?,
            gemv_splitk_reduce: module.load_function("gemv_splitk_reduce")?,
            gemv_q5_qk: module.load_function("gemv_q5_0_qk")?,
            gemv_q5_qkv: module.load_function("gemv_q5_0_qkv")?,
            gemv_q4_pair: module.load_function("gemv_q4_k_pair")?,
            gemv_q4_qkv: module.load_function("gemv_q4_k_qkv")?,
            gemm_q4: module.load_function("gemm_q4_k")?,
            gemm_q5: module.load_function("gemm_q5_0")?,
            gemm_q6: module.load_function("gemm_q6_k")?,
            gemm_q8: module.load_function("gemm_q8_0")?,
            embed_q4: module.load_function("embed_q4_k")?,
            embed_q5: module.load_function("embed_q5_0")?,
            embed_q6: module.load_function("embed_q6_k")?,
            embed_q8: module.load_function("embed_q8_0")?,
            embed_q4_one: module.load_function("embed_q4_k_one")?,
            embed_q5_one: module.load_function("embed_q5_0_one")?,
            embed_q6_one: module.load_function("embed_q6_k_one")?,
            embed_q8_one: module.load_function("embed_q8_0_one")?,
            embed_q4_one_d: module.load_function("embed_q4_k_one_d")?,
            embed_q5_one_d: module.load_function("embed_q5_0_one_d")?,
            embed_q6_one_d: module.load_function("embed_q6_k_one_d")?,
            embed_q8_one_d: module.load_function("embed_q8_0_one_d")?,
            rms_norm: module.load_function("rms_norm_f32")?,
            silu_mul: module.load_function("silu_mul_f32")?,
            add: module.load_function("add_f32")?,
            add_bias: module.load_function("add_bias_f32")?,
            rope: module.load_function("rope_neox_f32")?,
            rope_d: module.load_function("rope_neox_f32_d")?,
            qk_norm_rope: module.load_function("qk_rms_norm_rope_neox_f32")?,
            qk_norm_rope_d: module.load_function("qk_rms_norm_rope_neox_f32_d")?,
            attn: {
                // Load only symbols listed in decode::REGISTRY (easy add/remove).
                let mut map = std::collections::HashMap::new();
                for sym in DecodeBackend::required_kernel_symbols() {
                    map.insert(sym, module.load_function(sym).with_context(|| {
                        format!("load attn kernel {sym} (missing from CUDA source?)")
                    })?);
                }
                map
            },
            copy_kv: module.load_function("copy_kv_f16")?,
            copy_kv_d: module.load_function("copy_kv_f16_d")?,
            argmax: module.load_function("argmax_f32")?,
            copy_last: module.load_function("copy_last_row")?,
        };

        let upload_mat = |name: &str| -> Result<GpuMat> {
            let t = gguf
                .tensor(name)
                .ok_or_else(|| anyhow!("missing {name}"))?;
            let n_rows = t.dims[0] as usize;
            let n_cols = *t.dims.get(1).unwrap_or(&1) as usize;
            let raw = gguf.tensor_data(t);
            let wtype = wtype_of(t.ggml_type)?;
            let (decode_data, decode_col_bytes) = if wtype == WType::Q6K {
                let packed = repack_q6_decode(raw, n_rows, n_cols, t.ggml_type.nbytes(n_rows as u64) as usize);
                let stride = (n_rows / 256) * Q6_REPACK_BLOCK_BYTES;
                (Some(stream.clone_htod(&packed)?), stride)
            } else {
                (None, t.ggml_type.nbytes(n_rows as u64) as usize)
            };
            Ok(GpuMat {
                data: stream.clone_htod(raw)?,
                decode_data,
                n_rows,
                n_cols,
                col_bytes: t.ggml_type.nbytes(n_rows as u64) as usize,
                decode_col_bytes,
                wtype,
            })
        };
        let upload_vec = |name: &str| -> Result<_> {
            Ok(stream.clone_htod(&f32_host(gguf, name)?)?)
        };

        let token_embd = upload_mat("token_embd.weight")?;
        let output_norm = upload_vec("output_norm.weight")?;
        let output = upload_mat("output.weight").ok();

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            layers.push(GpuLayer {
                attn_norm: upload_vec(&format!("{p}.attn_norm.weight"))?,
                attn_q_norm: upload_vec(&format!("{p}.attn_q_norm.weight")).ok(),
                attn_k_norm: upload_vec(&format!("{p}.attn_k_norm.weight")).ok(),
                wq: upload_mat(&format!("{p}.attn_q.weight"))?,
                bq: upload_vec(&format!("{p}.attn_q.bias")).ok(),
                wk: upload_mat(&format!("{p}.attn_k.weight"))?,
                bk: upload_vec(&format!("{p}.attn_k.bias")).ok(),
                wv: upload_mat(&format!("{p}.attn_v.weight"))?,
                bv: upload_vec(&format!("{p}.attn_v.bias")).ok(),
                wo: upload_mat(&format!("{p}.attn_output.weight"))?,
                ffn_norm: upload_vec(&format!("{p}.ffn_norm.weight"))?,
                gate: upload_mat(&format!("{p}.ffn_gate.weight"))?,
                up: upload_mat(&format!("{p}.ffn_up.weight"))?,
                down: upload_mat(&format!("{p}.ffn_down.weight"))?,
            });
        }
        let mut type_counts = [0usize; 4];
        let mut type_bytes = [0usize; 4];
        let mut count_mat = |m: &GpuMat| {
            let idx = match m.wtype {
                WType::Q4K => 0,
                WType::Q5_0 => 1,
                WType::Q6K => 2,
                WType::Q8_0 => 3,
            };
            type_counts[idx] += 1;
            type_bytes[idx] += m.col_bytes * m.n_cols;
        };
        count_mat(&token_embd);
        if let Some(ref m) = output {
            count_mat(m);
        }
        for layer in &layers {
            for m in [
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
                &layer.gate,
                &layer.up,
                &layer.down,
            ] {
                count_mat(m);
            }
        }
        eprintln!(
            "weights | Q4_K={} ({:.2} GiB) Q5_0={} ({:.2} GiB) Q6_K={} ({:.2} GiB) Q8_0={} ({:.2} GiB)",
            type_counts[0], type_bytes[0] as f64 / 1073741824.0,
            type_counts[1], type_bytes[1] as f64 / 1073741824.0,
            type_counts[2], type_bytes[2] as f64 / 1073741824.0,
            type_counts[3], type_bytes[3] as f64 / 1073741824.0,
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
        let b = MAX_BATCH;
        // Split-K partials: enough for largest single-token GEMV (usually vocab).
        let gemv_partial_stride = cfg
            .n_vocab
            .max(cfg.n_ff)
            .max(n_embd)
            .max(n_kv);
        const GEMV_SPLIT_MAX: usize = 8;
        use super::decode::FLASH_MAX_SPLIT;
        let flash_partial_stride = 2 + head_dim;
        let flash_partial_n =
            cfg.n_head * FLASH_MAX_SPLIT as usize * flash_partial_stride;
        Ok(Self {
            x: stream.alloc_zeros(b * n_embd)?,
            // Qwen3 can have n_head*head_dim > n_embd; xb is reused for
            // normalized hidden state and the attention result.
            xb: stream.alloc_zeros(b * n_embd.max(n_q))?,
            xb2: stream.alloc_zeros(b * n_embd)?,
            q: stream.alloc_zeros(b * n_q)?,
            k_buf: stream.alloc_zeros(b * n_kv)?,
            v_buf: stream.alloc_zeros(b * n_kv)?,
            hb: stream.alloc_zeros(b * cfg.n_ff)?,
            hb2: stream.alloc_zeros(b * cfg.n_ff)?,
            x1: stream.alloc_zeros(n_embd)?,
            xb1: stream.alloc_zeros(n_embd)?,
            logits: stream.alloc_zeros(cfg.n_vocab)?,
            argmax_buf: stream.alloc_zeros(1)?,
            tok_buf: stream.alloc_zeros(MAX_BATCH)?,
            q8_x: stream.alloc_zeros(cfg.n_ff.max(n_embd))?,
            q8_d: stream.alloc_zeros((cfg.n_ff.max(n_embd) + 31) / 32)?,
            gemv_partial: stream.alloc_zeros(GEMV_SPLIT_MAX * gemv_partial_stride)?,
            gemv_partial_stride,
            d_pos0: stream.alloc_zeros(1)?,
            d_token: stream.alloc_zeros(1)?,
            flash_partial: stream.alloc_zeros(flash_partial_n)?,
            decode_graph: None,
            graph_tried: false,
            cuda_graph: true,
            graph_active: false,
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
            layers,
        })
    }
}
