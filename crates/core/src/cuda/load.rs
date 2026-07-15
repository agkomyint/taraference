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
        GgmlType::Q6_K => WType::Q6K,
        GgmlType::Q8_0 => WType::Q8_0,
        other => bail!("unsupported weight type {}", other.name()),
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
        let stream = ctx.default_stream();
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            SOURCE,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_86"),
                include_paths: vec![],
                ftz: Some(true),
                prec_div: Some(false),
                prec_sqrt: Some(false),
                fmad: Some(true),
                use_fast_math: Some(true),
                maxrregcount: None,
                options: vec![],
                name: None,
            },
        )
        .context("nvrtc compile")?;
        let module = ctx.load_module(ptx).context("load_module")?;
        let k = Kernels {
            gemv_q4: module.load_function("gemv_q4_k")?,
            gemv_q6: module.load_function("gemv_q6_k")?,
            gemv_q8: module.load_function("gemv_q8_0")?,
            gemm_q4: module.load_function("gemm_q4_k")?,
            gemm_q6: module.load_function("gemm_q6_k")?,
            gemm_q8: module.load_function("gemm_q8_0")?,
            embed_q4: module.load_function("embed_q4_k")?,
            embed_q6: module.load_function("embed_q6_k")?,
            embed_q8: module.load_function("embed_q8_0")?,
            embed_q4_one: module.load_function("embed_q4_k_one")?,
            embed_q6_one: module.load_function("embed_q6_k_one")?,
            embed_q8_one: module.load_function("embed_q8_0_one")?,
            rms_norm: module.load_function("rms_norm_f32")?,
            silu_mul: module.load_function("silu_mul_f32")?,
            add: module.load_function("add_f32")?,
            add_bias: module.load_function("add_bias_f32")?,
            rope: module.load_function("rope_neox_f32")?,
            attn_fast: module.load_function("attn_f32")?,
            attn_basic: module.load_function("attn_basic_f32")?,
            attn_online: module.load_function("attn_online_f32")?,
            copy_kv: module.load_function("copy_kv_f16")?,
            argmax: module.load_function("argmax_f32")?,
            copy_last: module.load_function("copy_last_row")?,
        };

        let upload_mat = |name: &str| -> Result<GpuMat> {
            let t = gguf
                .tensor(name)
                .ok_or_else(|| anyhow!("missing {name}"))?;
            let n_rows = t.dims[0] as usize;
            let n_cols = *t.dims.get(1).unwrap_or(&1) as usize;
            Ok(GpuMat {
                data: stream.clone_htod(gguf.tensor_data(t))?,
                n_rows,
                n_cols,
                col_bytes: t.ggml_type.nbytes(n_rows as u64) as usize,
                wtype: wtype_of(t.ggml_type)?,
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
        eprintln!("ready");

        let n_embd = cfg.n_embd;
        let n_kv = cfg.n_head_kv * cfg.head_dim();
        let b = MAX_BATCH;
        Ok(Self {
            x: stream.alloc_zeros(b * n_embd)?,
            xb: stream.alloc_zeros(b * n_embd)?,
            xb2: stream.alloc_zeros(b * n_embd)?,
            q: stream.alloc_zeros(b * n_embd)?,
            k_buf: stream.alloc_zeros(b * n_kv)?,
            v_buf: stream.alloc_zeros(b * n_kv)?,
            hb: stream.alloc_zeros(b * cfg.n_ff)?,
            hb2: stream.alloc_zeros(b * cfg.n_ff)?,
            x1: stream.alloc_zeros(n_embd)?,
            xb1: stream.alloc_zeros(n_embd)?,
            logits: stream.alloc_zeros(cfg.n_vocab)?,
            argmax_buf: stream.alloc_zeros(1)?,
            tok_buf: stream.alloc_zeros(MAX_BATCH)?,
            cfg,
            decode,
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
