//! CUDA Qwen2 — all weights quantized on GPU (fits 4GB). Fused Q GEMV.

use crate::cuda_kernels::KERNELS;
use crate::model::ModelConfig;
use crate::quant::dequant_f32;
use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;
use taraference_gguf::{GgmlType, GgufFile};

#[derive(Clone, Copy, PartialEq, Eq)]
enum WType {
    Q4K,
    Q6K,
    Q8_0,
}

struct GpuMat {
    data: CudaSlice<u8>,
    n_rows: usize,
    n_cols: usize,
    col_bytes: usize,
    wtype: WType,
}

struct GpuLayer {
    attn_norm: CudaSlice<f32>,
    wq: GpuMat,
    bq: Option<CudaSlice<f32>>,
    wk: GpuMat,
    bk: Option<CudaSlice<f32>>,
    wv: GpuMat,
    bv: Option<CudaSlice<f32>>,
    wo: GpuMat,
    ffn_norm: CudaSlice<f32>,
    gate: GpuMat,
    up: GpuMat,
    down: GpuMat,
}

struct Kernels {
    gemv_q4: CudaFunction,
    gemv_q6: CudaFunction,
    gemv_q8: CudaFunction,
    embed_q4: CudaFunction,
    embed_q6: CudaFunction,
    embed_q8: CudaFunction,
    rms_norm: CudaFunction,
    silu: CudaFunction,
    add: CudaFunction,
    mul: CudaFunction,
    add_bias: CudaFunction,
    rope: CudaFunction,
    attn: CudaFunction,
    copy_kv: CudaFunction,
    argmax: CudaFunction,
}

pub struct CudaModel {
    pub cfg: ModelConfig,
    stream: Arc<CudaStream>,
    _ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    k: Kernels,
    token_embd: GpuMat,
    output_norm: CudaSlice<f32>,
    output: Option<GpuMat>,
    layers: Vec<GpuLayer>,
    x: CudaSlice<f32>,
    xb: CudaSlice<f32>,
    xb2: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k_buf: CudaSlice<f32>,
    v_buf: CudaSlice<f32>,
    hb: CudaSlice<f32>,
    hb2: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    argmax_buf: CudaSlice<i32>,
    pub bw_ceiling_tps: f64,
}

pub struct CudaKv {
    k: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    pub len: usize,
    pub max_seq: usize,
    n_head_kv: usize,
    head_dim: usize,
}

impl CudaKv {
    fn stride(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

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
        let cfg = ModelConfig::from_gguf(gguf)?;
        let weight_bytes = gguf.total_tensor_bytes() as f64;
        let ceiling = 192e9 / weight_bytes;
        eprintln!(
            "CUDA GPU | {} L={} d={} | Q-weights {:.2} GiB | ceiling≈{:.0} tok/s",
            cfg.architecture,
            cfg.n_layer,
            cfg.n_embd,
            weight_bytes / (1024.0 * 1024.0 * 1024.0),
            ceiling
        );

        let ctx = CudaContext::new(0).context("CudaContext")?;
        let stream = ctx.default_stream();
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            KERNELS,
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
            embed_q4: module.load_function("embed_q4_k")?,
            embed_q6: module.load_function("embed_q6_k")?,
            embed_q8: module.load_function("embed_q8_0")?,
            rms_norm: module.load_function("rms_norm_f32")?,
            silu: module.load_function("silu_f32")?,
            add: module.load_function("add_f32")?,
            mul: module.load_function("mul_f32")?,
            add_bias: module.load_function("add_bias_f32")?,
            rope: module.load_function("rope_neox_f32")?,
            attn: module.load_function("attn_decode_f32")?,
            copy_kv: module.load_function("copy_kv_f32")?,
            argmax: module.load_function("argmax_f32")?,
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
        let upload_vec = |name: &str| -> Result<CudaSlice<f32>> {
            Ok(stream.clone_htod(&f32_host(gguf, name)?)?)
        };

        eprintln!("uploading Q-tensors to GPU…");
        let token_embd = upload_mat("token_embd.weight")?;
        let output_norm = upload_vec("output_norm.weight")?;
        let output = upload_mat("output.weight").ok();

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            eprint!("\r  layer {}/{}", i + 1, cfg.n_layer);
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
        eprintln!("\r  GPU ready.                    ");

        let n_embd = cfg.n_embd;
        let n_kv = cfg.n_head_kv * cfg.head_dim();
        Ok(Self {
            x: stream.alloc_zeros(n_embd)?,
            xb: stream.alloc_zeros(n_embd)?,
            xb2: stream.alloc_zeros(n_embd)?,
            q: stream.alloc_zeros(n_embd)?,
            k_buf: stream.alloc_zeros(n_kv)?,
            v_buf: stream.alloc_zeros(n_kv)?,
            hb: stream.alloc_zeros(cfg.n_ff)?,
            hb2: stream.alloc_zeros(cfg.n_ff)?,
            logits: stream.alloc_zeros(cfg.n_vocab)?,
            argmax_buf: stream.alloc_zeros(1)?,
            cfg,
            stream,
            _ctx: ctx,
            _module: module,
            k,
            token_embd,
            output_norm,
            output,
            layers,
            bw_ceiling_tps: ceiling,
        })
    }

    pub fn alloc_kv(&self, max_seq: usize) -> Result<CudaKv> {
        let stride = self.cfg.n_head_kv * self.cfg.head_dim();
        let slot = max_seq * stride;
        let mut k = Vec::with_capacity(self.cfg.n_layer);
        let mut v = Vec::with_capacity(self.cfg.n_layer);
        for _ in 0..self.cfg.n_layer {
            k.push(self.stream.alloc_zeros(slot)?);
            v.push(self.stream.alloc_zeros(slot)?);
        }
        Ok(CudaKv {
            k,
            v,
            len: 0,
            max_seq,
            n_head_kv: self.cfg.n_head_kv,
            head_dim: self.cfg.head_dim(),
        })
    }

    fn gemv(
        stream: &Arc<CudaStream>,
        k: &Kernels,
        w: &GpuMat,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let n_rows = w.n_rows as i32;
        let n_cols = w.n_cols as i32;
        let col_bytes = w.col_bytes as i32;
        let f = match w.wtype {
            WType::Q4K => &k.gemv_q4,
            WType::Q6K => &k.gemv_q6,
            WType::Q8_0 => &k.gemv_q8,
        };
        unsafe {
            stream
                .launch_builder(f)
                .arg(&w.data)
                .arg(x)
                .arg(y)
                .arg(&n_rows)
                .arg(&n_cols)
                .arg(&col_bytes)
                .launch(LaunchConfig::for_num_elems(w.n_cols as u32))?;
        }
        Ok(())
    }

    fn embed(
        stream: &Arc<CudaStream>,
        k: &Kernels,
        table: &GpuMat,
        token: i32,
        out: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let n_rows = table.n_rows as i32;
        let col_bytes = table.col_bytes as i32;
        let f = match table.wtype {
            WType::Q4K => &k.embed_q4,
            WType::Q6K => &k.embed_q6,
            WType::Q8_0 => &k.embed_q8,
        };
        unsafe {
            stream
                .launch_builder(f)
                .arg(&table.data)
                .arg(out)
                .arg(&token)
                .arg(&n_rows)
                .arg(&col_bytes)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    pub fn forward_greedy(&mut self, tokens: &[u32], cache: &mut CudaKv) -> Result<u32> {
        if tokens.is_empty() {
            bail!("empty tokens");
        }
        if cache.len + tokens.len() > cache.max_seq {
            bail!("context full");
        }

        let start = cache.len;
        let n_embd_u = self.cfg.n_embd;
        let n_ff_u = self.cfg.n_ff;
        let n_head_u = self.cfg.n_head;
        let n_kv_heads = self.cfg.n_head_kv;
        let head_dim = self.cfg.head_dim();
        let n_vocab_u = self.cfg.n_vocab;
        let n_layer = self.layers.len();
        let eps = self.cfg.rms_eps;
        let theta = self.cfg.rope_theta;
        let scale = (head_dim as f32).sqrt().recip();
        let stride_u = cache.stride();

        let n_embd = n_embd_u as i32;
        let n_ff = n_ff_u as i32;
        let n_head = n_head_u as i32;
        let n_kv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let stride = stride_u as i32;

        for (t_i, &tok) in tokens.iter().enumerate() {
            let pos = (start + t_i) as i32;
            let token = tok as i32;

            Self::embed(
                &self.stream,
                &self.k,
                &self.token_embd,
                token,
                &mut self.x,
            )?;

            for li in 0..n_layer {
                unsafe {
                    self.stream
                        .launch_builder(&self.k.rms_norm)
                        .arg(&self.x)
                        .arg(&self.layers[li].attn_norm)
                        .arg(&mut self.xb)
                        .arg(&n_embd)
                        .arg(&eps)
                        .launch(LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }

                Self::gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].wq,
                    &self.xb,
                    &mut self.q,
                )?;
                if let Some(ref b) = self.layers[li].bq {
                    unsafe {
                        self.stream
                            .launch_builder(&self.k.add_bias)
                            .arg(&mut self.q)
                            .arg(b)
                            .arg(&n_embd)
                            .launch(LaunchConfig::for_num_elems(n_embd_u as u32))?;
                    }
                }
                Self::gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].wk,
                    &self.xb,
                    &mut self.k_buf,
                )?;
                if let Some(ref b) = self.layers[li].bk {
                    let n = (n_kv_heads * head_dim) as i32;
                    unsafe {
                        self.stream
                            .launch_builder(&self.k.add_bias)
                            .arg(&mut self.k_buf)
                            .arg(b)
                            .arg(&n)
                            .launch(LaunchConfig::for_num_elems(n as u32))?;
                    }
                }
                Self::gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].wv,
                    &self.xb,
                    &mut self.v_buf,
                )?;
                if let Some(ref b) = self.layers[li].bv {
                    let n = (n_kv_heads * head_dim) as i32;
                    unsafe {
                        self.stream
                            .launch_builder(&self.k.add_bias)
                            .arg(&mut self.v_buf)
                            .arg(b)
                            .arg(&n)
                            .launch(LaunchConfig::for_num_elems(n as u32))?;
                    }
                }

                unsafe {
                    self.stream
                        .launch_builder(&self.k.rope)
                        .arg(&mut self.q)
                        .arg(&n_head)
                        .arg(&hd)
                        .arg(&pos)
                        .arg(&theta)
                        .launch(LaunchConfig {
                            grid_dim: (n_head_u as u32, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    self.stream
                        .launch_builder(&self.k.rope)
                        .arg(&mut self.k_buf)
                        .arg(&n_kv)
                        .arg(&hd)
                        .arg(&pos)
                        .arg(&theta)
                        .launch(LaunchConfig {
                            grid_dim: (n_kv_heads as u32, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    self.stream
                        .launch_builder(&self.k.copy_kv)
                        .arg(&self.k_buf)
                        .arg(&mut cache.k[li])
                        .arg(&pos)
                        .arg(&stride)
                        .launch(LaunchConfig::for_num_elems(stride_u as u32))?;
                    self.stream
                        .launch_builder(&self.k.copy_kv)
                        .arg(&self.v_buf)
                        .arg(&mut cache.v[li])
                        .arg(&pos)
                        .arg(&stride)
                        .launch(LaunchConfig::for_num_elems(stride_u as u32))?;
                }

                let seq_len = pos + 1;
                let smem = (seq_len as u32).max(1) * 4;
                unsafe {
                    self.stream
                        .launch_builder(&self.k.attn)
                        .arg(&self.q)
                        .arg(&cache.k[li])
                        .arg(&cache.v[li])
                        .arg(&mut self.xb)
                        .arg(&n_head)
                        .arg(&n_kv)
                        .arg(&hd)
                        .arg(&seq_len)
                        .arg(&scale)
                        .launch(LaunchConfig {
                            grid_dim: (n_head_u as u32, 1, 1),
                            block_dim: (64, 1, 1),
                            shared_mem_bytes: smem,
                        })?;
                }

                Self::gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].wo,
                    &self.xb,
                    &mut self.xb2,
                )?;
                unsafe {
                    self.stream
                        .launch_builder(&self.k.add)
                        .arg(&mut self.x)
                        .arg(&self.xb2)
                        .arg(&n_embd)
                        .launch(LaunchConfig::for_num_elems(n_embd_u as u32))?;
                    self.stream
                        .launch_builder(&self.k.rms_norm)
                        .arg(&self.x)
                        .arg(&self.layers[li].ffn_norm)
                        .arg(&mut self.xb)
                        .arg(&n_embd)
                        .arg(&eps)
                        .launch(LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }

                Self::gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].gate,
                    &self.xb,
                    &mut self.hb,
                )?;
                Self::gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].up,
                    &self.xb,
                    &mut self.hb2,
                )?;
                unsafe {
                    self.stream
                        .launch_builder(&self.k.silu)
                        .arg(&mut self.hb)
                        .arg(&n_ff)
                        .launch(LaunchConfig::for_num_elems(n_ff_u as u32))?;
                    self.stream
                        .launch_builder(&self.k.mul)
                        .arg(&mut self.hb)
                        .arg(&self.hb2)
                        .arg(&n_ff)
                        .launch(LaunchConfig::for_num_elems(n_ff_u as u32))?;
                }
                Self::gemv(
                    &self.stream,
                    &self.k,
                    &self.layers[li].down,
                    &self.hb,
                    &mut self.xb2,
                )?;
                unsafe {
                    self.stream
                        .launch_builder(&self.k.add)
                        .arg(&mut self.x)
                        .arg(&self.xb2)
                        .arg(&n_embd)
                        .launch(LaunchConfig::for_num_elems(n_embd_u as u32))?;
                }
            }

            if t_i + 1 == tokens.len() {
                unsafe {
                    self.stream
                        .launch_builder(&self.k.rms_norm)
                        .arg(&self.x)
                        .arg(&self.output_norm)
                        .arg(&mut self.xb)
                        .arg(&n_embd)
                        .arg(&eps)
                        .launch(LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
                if let Some(ref ow) = self.output {
                    Self::gemv(&self.stream, &self.k, ow, &self.xb, &mut self.logits)?;
                } else {
                    Self::gemv(
                        &self.stream,
                        &self.k,
                        &self.token_embd,
                        &self.xb,
                        &mut self.logits,
                    )?;
                }
                let n_vocab = n_vocab_u as i32;
                unsafe {
                    self.stream
                        .launch_builder(&self.k.argmax)
                        .arg(&self.logits)
                        .arg(&n_vocab)
                        .arg(&mut self.argmax_buf)
                        .launch(LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
            }
        }

        cache.len = start + tokens.len();
        self.stream.synchronize().context("cuda sync")?;
        let idx = self.stream.clone_dtoh(&self.argmax_buf)?;
        Ok(idx[0] as u32)
    }
}
