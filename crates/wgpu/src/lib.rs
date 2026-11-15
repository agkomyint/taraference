//! Native packed-GGUF Qwen2/Qwen2.5 inference through wgpu compute shaders.
//! No CUDA, ONNX, Python or CPU neural-network fallback in the default build.
use anyhow::{bail, ensure, Context, Result};
use std::{collections::HashMap, sync::mpsc};
use taraference_core::ModelConfig;
use taraference_gguf::{GgmlType, GgufFile};
use wgpu::util::DeviceExt;

pub mod runner;

struct Matrix {
    buffer: wgpu::Buffer,
    n: usize,
    m: usize,
    kind: u32,
}
struct Layer {
    an: Matrix,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    qb: Option<Matrix>,
    kb: Option<Matrix>,
    vb: Option<Matrix>,
    fnorm: Matrix,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
}
#[derive(Clone)]
struct Op {
    entry: &'static str,
    rows: u32,
    cols: u32,
    pipeline: usize,
    bindings: wgpu::BindGroup,
    grid: [u32; 3],
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::ComputePipeline>,
    entries: HashMap<&'static str, usize>,
    dummy: wgpu::Buffer,
    dummy_kv: wgpu::Buffer,
    state: wgpu::Buffer,
    adapter: String,
    max_binding: u64,
    gemm_tile_k: u32,
    expand_q6: bool,
}

impl Gpu {
    fn new() -> Result<Self> {
        let gemm_tile_k = std::env::var("TARAFER_WGPU_GEMM_TILE_K")
            .ok()
            .map(|s| s.parse::<u32>())
            .transpose()
            .context("invalid TARAFER_WGPU_GEMM_TILE_K")?
            .unwrap_or(32);
        ensure!(
            [16, 32, 64].contains(&gemm_tile_k),
            "GEMM tile depth must be 16, 32 or 64"
        );
        let expand_q6 = match std::env::var("TARAFER_WGPU_Q6_SCALES").as_deref() {
            Ok("0") => false,
            Ok("1") | Err(_) => true,
            Ok(_) => bail!("TARAFER_WGPU_Q6_SCALES must be 0 or 1"),
        };
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .context("no native wgpu adapter available (Vulkan/Metal/DX12)")?;
        let info = adapter.get_info();
        ensure!(
            info.device_type != wgpu::DeviceType::Cpu,
            "wgpu selected a software CPU adapter; a hardware GPU is required"
        );
        let mut limits = wgpu::Limits::default();
        let available = adapter.limits();
        // The original Q6_K output head can exceed the WebGPU default 128 MiB.
        limits.max_storage_buffer_binding_size = available.max_storage_buffer_binding_size;
        limits.max_buffer_size = available.max_buffer_size;
        let max_binding = limits.max_storage_buffer_binding_size as u64;
        let required_features = if std::env::var_os("TARAFER_WGPU_PROFILE").is_some() {
            let features =
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
            ensure!(
                adapter.features().contains(features),
                "this adapter does not support per-dispatch GPU timestamps"
            );
            features
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("taraference native text"),
            required_features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .context("request wgpu device")?;
        let mut entries = Vec::new();
        for binding in 0..8 {
            let ty = if binding == 5 || binding == 6 {
                wgpu::BufferBindingType::Uniform
            } else {
                wgpu::BufferBindingType::Storage {
                    read_only: binding < 3 || binding == 7,
                }
            };
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native GGUF kernels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("kernels.wgsl").into()),
        });
        let mut pipelines = Vec::new();
        let mut map = HashMap::new();
        for entry in [
            "embed",
            "gemv",
            "gemv_q4k",
            "gemv_q6k",
            "gemv_q6k_scaled",
            "gate_up_q4k",
            "gemm",
            "gemm_q4k",
            "gemm_q6k",
            "gemm_q6k_scaled",
            "rms",
            "rope",
            "store_kv",
            "attention",
            "add",
            "swiglu",
            "argmax",
            "unpack",
        ] {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(if entry.starts_with("gemm") {
                    "gemm"
                } else {
                    entry
                }),
                // Every shared-memory location read by our kernels is explicitly
                // written before its first barrier/read; padding is never read.
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("TILE_K", gemm_tile_k as f64),
                        (
                            "WEIGHT_KIND",
                            match entry {
                                "gemm_q4k" => 12.,
                                "gemm_q6k" => 14.,
                                "gemm_q6k_scaled" => 114.,
                                _ => 255.,
                            },
                        ),
                    ],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });
            map.insert(entry, pipelines.len());
            pipelines.push(pipeline);
        }
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            bail!("wgpu shader validation: {error}");
        }
        let dummy = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("zero"),
            contents: &[0; 16],
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy_kv = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("token and position"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            device,
            queue,
            layout,
            pipelines,
            entries: map,
            dummy,
            dummy_kv,
            state,
            adapter: format!("{} ({:?})", info.name, info.backend),
            max_binding,
            gemm_tile_k,
            expand_q6,
        })
    }
    fn buffer(&self, label: &str, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * 4).max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
    fn upload(&self, label: &str, bytes: &[u8]) -> Result<wgpu::Buffer> {
        ensure!(
            bytes.len() as u64 <= self.max_binding,
            "tensor {label} needs {} MiB binding, adapter limit is {} MiB",
            bytes.len() / 1048576,
            self.max_binding / 1048576
        );
        let mut padding = Vec::new();
        let data = if bytes.len() % 4 == 0 {
            bytes
        } else {
            padding.extend_from_slice(bytes);
            padding.resize(bytes.len().next_multiple_of(4), 0);
            &padding
        };
        Ok(self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            }))
    }
    #[allow(clippy::too_many_arguments)]
    fn op(
        &self,
        entry: &str,
        w: Option<&wgpu::Buffer>,
        a: Option<&wgpu::Buffer>,
        b: Option<&wgpu::Buffer>,
        out: &wgpu::Buffer,
        kv: Option<&wgpu::Buffer>,
        params: [u32; 16],
        grid: [u32; 3],
    ) -> Op {
        self.op_extra(entry, w, a, b, out, kv, params, grid, None)
    }
    #[allow(clippy::too_many_arguments)]
    fn op_extra(
        &self,
        entry: &str,
        w: Option<&wgpu::Buffer>,
        a: Option<&wgpu::Buffer>,
        b: Option<&wgpu::Buffer>,
        out: &wgpu::Buffer,
        kv: Option<&wgpu::Buffer>,
        params: [u32; 16],
        grid: [u32; 3],
        extra: Option<&wgpu::Buffer>,
    ) -> Op {
        let entry = if entry == "gemv" {
            match params[2] {
                12 => "gemv_q4k",
                14 => "gemv_q6k",
                114 => "gemv_q6k_scaled",
                _ => entry,
            }
        } else {
            entry
        };
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(entry),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let buffers = [
            w.unwrap_or(&self.dummy),
            a.unwrap_or(&self.dummy),
            b.unwrap_or(&self.dummy),
            out,
            kv.unwrap_or(&self.dummy_kv),
            &uniform,
            &self.state,
            extra.unwrap_or(&self.dummy),
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        Op {
            entry: *self.entries.get_key_value(entry).unwrap().0,
            rows: params[1],
            cols: params[0],
            pipeline: self.entries[entry],
            bindings: self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &self.layout,
                entries: &entries,
            }),
            grid,
        }
    }
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, ops: &[Op]) {
        self.encode_count(encoder, ops, 1);
    }
    fn encode_count(&self, encoder: &mut wgpu::CommandEncoder, ops: &[Op], count: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Qwen2 forward"),
            timestamp_writes: None,
        });
        for op in ops {
            let matrix_batch = count > 1 && op.entry.starts_with("gemv");
            let pipeline = if matrix_batch {
                self.entries[match op.entry {
                    "gemv_q4k" => "gemm_q4k",
                    "gemv_q6k" => "gemm_q6k",
                    "gemv_q6k_scaled" => "gemm_q6k_scaled",
                    _ => "gemm",
                }]
            } else {
                op.pipeline
            };
            pass.set_pipeline(&self.pipelines[pipeline]);
            pass.set_bind_group(0, &op.bindings, &[]);
            let grid = if count == 1 {
                op.grid
            } else if matrix_batch {
                [op.rows.div_ceil(16), count.div_ceil(16), 1]
            } else if op.entry == "rms" {
                [count, 1, 1]
            } else {
                [op.grid[0], count, op.grid[2]]
            };
            pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
        }
    }
    fn read(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<u32>> {
        let size = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        self.read_mapped(&staging)
    }
    fn read_mapped(&self, staging: &wgpu::Buffer) -> Result<Vec<u32>> {
        let (tx, rx) = mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait)
            .context("wgpu GPU synchronization")?;
        rx.recv()??;
        let view = staging.slice(..).get_mapped_range();
        let result = view
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        drop(view);
        staging.unmap();
        Ok(result)
    }
}

fn repack_q6_scales(raw: &[u8]) -> Result<Vec<u8>> {
    ensure!(raw.len() % 210 == 0, "incomplete Q6_K block");
    let bytes = (raw.len() / 210)
        .checked_mul(256)
        .context("Q6_K repack size overflow")?;
    let mut out = vec![0u8; bytes];
    for (src, dst) in raw.chunks_exact(210).zip(out.chunks_exact_mut(256)) {
        dst[..192].copy_from_slice(&src[..192]);
        let d = taraference_core::quant::f16_to_f32(u16::from_le_bytes([src[208], src[209]]));
        ensure!(d.is_finite(), "non-finite Q6_K scale");
        for i in 0..16 {
            let coefficient = d * (src[192 + i] as i8 as f32);
            dst[192 + i * 4..196 + i * 4].copy_from_slice(&coefficient.to_le_bytes());
        }
    }
    Ok(out)
}

fn load_matrix(gpu: &Gpu, file: &GgufFile, name: &str, shape: &[usize]) -> Result<Matrix> {
    let tensor = file
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    ensure!(
        tensor
            .dims
            .iter()
            .copied()
            .eq(shape.iter().map(|&n| n as u64)),
        "{name}: incompatible dimensions {:?}, expected {shape:?}",
        tensor.dims
    );
    ensure!(
        matches!(
            tensor.ggml_type,
            GgmlType::F32
                | GgmlType::F16
                | GgmlType::Q4_0
                | GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::Q8_0
        ),
        "{name}: unsupported {}",
        tensor.ggml_type.name()
    );
    ensure!(
        shape[0] % tensor.ggml_type.block_size() == 0,
        "{name}: row is not quant block aligned"
    );
    if shape.len() == 1 {
        ensure!(
            tensor.ggml_type == GgmlType::F32,
            "norm/bias {name} must be fp32"
        );
    }
    let raw = file.tensor_data(tensor);
    let expanded_bytes = (raw.len() / 210) as u64 * 256;
    // Fall back per tensor on adapters whose binding limit cannot hold the
    // expanded coefficients. The original file is never modified.
    let scaled =
        tensor.ggml_type == GgmlType::Q6_K && gpu.expand_q6 && expanded_bytes <= gpu.max_binding;
    let buffer = if scaled {
        gpu.upload(name, &repack_q6_scales(raw)?)?
    } else {
        gpu.upload(name, raw)?
    };
    Ok(Matrix {
        buffer,
        n: shape[0],
        m: shape.get(1).copied().unwrap_or(1),
        kind: if scaled { 114 } else { tensor.ggml_type as u32 },
    })
}

/// Resident packed weights and per-layer packed-fp16 KV cache. Decode activations
/// and sums are fp32; prefill uses tiled GEMM in bounded token batches.
pub struct WgpuModel {
    gpu: Gpu,
    ops: Vec<Op>,
    prefill_ops: Vec<Op>,
    head: Vec<Op>,
    logits: wgpu::Buffer,
    next: wgpu::Buffer,
    readback: wgpu::Buffer,
    tokens: wgpu::Buffer,
    prefill_batch: usize,
    fused_layers: usize,
    q6_scaled_tensors: usize,
    gpu_weight_bytes: u64,
    pub cfg: ModelConfig,
    pub len: usize,
    pub max_seq: usize,
    pub weight_bytes: u64,
}

impl WgpuModel {
    pub fn load(file: &GgufFile, max_seq: usize) -> Result<Self> {
        let cfg = ModelConfig::from_gguf(file)?;
        ensure!(cfg.architecture == "qwen2" && !cfg.is_moe(), "native wgpu text currently supports dense Qwen2/Qwen2.5 GGUF only (got {}); Qwen3.5/DeltaNet is not implemented", cfg.architecture);
        ensure!(
            (1..=2048).contains(&max_seq) && max_seq <= cfg.n_ctx,
            "wgpu context must be 1..={} (maximum 2048)",
            cfg.n_ctx.min(2048)
        );
        ensure!(
            cfg.n_head_kv > 0
                && cfg.n_vocab <= (1 << 24)
                && cfg.n_head % cfg.n_head_kv == 0
                && cfg.attention_head_dim <= 128
                && cfg.attention_head_dim % 2 == 0
                && cfg.n_head * cfg.attention_head_dim == cfg.n_embd
                && (cfg.rope_dim == 0 || cfg.rope_dim == cfg.attention_head_dim),
            "unsupported Qwen2 attention geometry"
        );
        let gpu = Gpu::new()?;
        let fuse_gate_up = match std::env::var("TARAFER_WGPU_FUSE_GATE_UP").as_deref() {
            Ok("0") => false,
            Ok("1") | Err(_) => true,
            Ok(_) => bail!("TARAFER_WGPU_FUSE_GATE_UP must be 0 or 1"),
        };
        let prefill_batch = std::env::var("TARAFER_WGPU_PREFILL_BATCH")
            .ok()
            .map(|s| s.parse::<usize>())
            .transpose()
            .context("invalid TARAFER_WGPU_PREFILL_BATCH")?
            .unwrap_or(32);
        ensure!(
            (1..=32).contains(&prefill_batch),
            "prefill batch must be 1..=32"
        );
        eprintln!(
            "adapter| {} | native WGSL, fp32 activations, packed GGUF, fp16 KV",
            gpu.adapter
        );
        let n = cfg.n_embd;
        let hd = cfg.attention_head_dim;
        let kvn = cfg.n_head_kv * hd;
        let ff = cfg.n_ff;
        let emb = load_matrix(&gpu, file, "token_embd.weight", &[n, cfg.n_vocab])?;
        let norm = load_matrix(&gpu, file, "output_norm.weight", &[n])?;
        let output = if file.tensor("output.weight").is_some() {
            Some(load_matrix(&gpu, file, "output.weight", &[n, cfg.n_vocab])?)
        } else {
            None
        };
        let mut layers = Vec::new();
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let mat = |key: &str, rows, cols| {
                load_matrix(&gpu, file, &format!("{p}.{key}.weight"), &[rows, cols])
            };
            let vec =
                |key: &str, width| load_matrix(&gpu, file, &format!("{p}.{key}.weight"), &[width]);
            let bias = |key: &str, width| -> Result<Option<Matrix>> {
                let name = format!("{p}.{key}.bias");
                if file.tensor(&name).is_some() {
                    Ok(Some(load_matrix(&gpu, file, &name, &[width])?))
                } else {
                    Ok(None)
                }
            };
            layers.push(Layer {
                an: vec("attn_norm", n)?,
                q: mat("attn_q", n, n)?,
                k: mat("attn_k", n, kvn)?,
                v: mat("attn_v", n, kvn)?,
                o: mat("attn_output", n, n)?,
                qb: bias("attn_q", n)?,
                kb: bias("attn_k", kvn)?,
                vb: bias("attn_v", kvn)?,
                fnorm: vec("ffn_norm", n)?,
                gate: mat("ffn_gate", n, ff)?,
                up: mat("ffn_up", n, ff)?,
                down: mat("ffn_down", ff, n)?,
            });
            gpu.device.poll(wgpu::PollType::Wait)?;
        }
        let x = gpu.buffer("hidden", n * 32);
        let xb = gpu.buffer("normalized", n * 32);
        let tmp = gpu.buffer("residual delta", n);
        let q = gpu.buffer("q", n * 32);
        let k = gpu.buffer("k", kvn * 32);
        let v = gpu.buffer("v", kvn * 32);
        let att = gpu.buffer("attention", n * 32);
        let gate = gpu.buffer("gate", ff * 32);
        let up = gpu.buffer("up", ff * 32);
        let act = gpu.buffer("SwiGLU", ff * 32);
        let tokens = gpu.buffer("prefill tokens", 32);
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("next token readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let logits = gpu.buffer("full vocabulary logits", cfg.n_vocab);
        let next = gpu.buffer("argmax", 1);
        let mut base = [0u32; 16];
        base[4] = cfg.rms_eps.to_bits();
        base[5] = cfg.rope_theta.to_bits();
        base[6] = cfg.n_head as u32;
        base[7] = cfg.n_head_kv as u32;
        base[8] = hd as u32;
        base[9] = max_seq as u32;
        let element = |entry: &str,
                       a: &wgpu::Buffer,
                       b: Option<&wgpu::Buffer>,
                       out: &wgpu::Buffer,
                       len: usize| {
            let mut p = base;
            p[0] = len as u32;
            gpu.op(
                entry,
                None,
                Some(a),
                b,
                out,
                None,
                p,
                [(len as u32).div_ceil(128), 1, 1],
            )
        };
        let rms = |weight: &Matrix, a: &wgpu::Buffer, out: &wgpu::Buffer, last: bool| {
            let mut p = base;
            p[0] = n as u32;
            p[3] = u32::from(last);
            gpu.op(
                "rms",
                Some(&weight.buffer),
                Some(a),
                None,
                out,
                None,
                p,
                [1, 1, 1],
            )
        };
        let gemv = |weight: &Matrix,
                    a: &wgpu::Buffer,
                    bias: Option<&Matrix>,
                    out: &wgpu::Buffer,
                    residual: bool| {
            let mut p = base;
            p[0] = weight.n as u32;
            p[1] = weight.m as u32;
            p[2] = weight.kind;
            p[10] = u32::from(bias.is_some());
            p[11] = u32::from(residual);
            gpu.op(
                "gemv",
                Some(&weight.buffer),
                Some(a),
                bias.map(|m| &m.buffer),
                out,
                None,
                p,
                [(weight.m as u32).div_ceil(4), 1, 1],
            )
        };
        let mut ep = base;
        ep[0] = n as u32;
        ep[2] = emb.kind;
        let mut ops = vec![gpu.op(
            "embed",
            Some(&emb.buffer),
            None,
            Some(&tokens),
            &x,
            None,
            ep,
            [(n as u32).div_ceil(128), 1, 1],
        )];
        let mut prefill_replacements = HashMap::<usize, Vec<Op>>::new();
        for layer in &layers {
            // One packed buffer contains K followed by V; each stores two f16s/u32.
            let cache = gpu.buffer("packed KV", max_seq * kvn);
            ops.push(rms(&layer.an, &x, &xb, false));
            ops.push(gemv(&layer.q, &xb, layer.qb.as_ref(), &q, false));
            ops.push(gemv(&layer.k, &xb, layer.kb.as_ref(), &k, false));
            ops.push(gemv(&layer.v, &xb, layer.vb.as_ref(), &v, false));
            for (out, heads) in [(&q, cfg.n_head), (&k, cfg.n_head_kv)] {
                let mut p = base;
                p[6] = heads as u32;
                ops.push(gpu.op(
                    "rope",
                    None,
                    None,
                    None,
                    out,
                    None,
                    p,
                    [((heads * hd / 2) as u32).div_ceil(128), 1, 1],
                ));
            }
            let mut p = base;
            p[0] = kvn as u32;
            ops.push(gpu.op(
                "store_kv",
                None,
                Some(&k),
                Some(&v),
                &tmp,
                Some(&cache),
                p,
                [((kvn / 2) as u32).div_ceil(128), 1, 1],
            ));
            ops.push(gpu.op(
                "attention",
                None,
                Some(&q),
                None,
                &att,
                Some(&cache),
                base,
                [cfg.n_head as u32, 1, 1],
            ));
            ops.push(gemv(&layer.o, &att, None, &x, true));
            ops.push(rms(&layer.fnorm, &x, &xb, false));
            let projections = vec![
                gemv(&layer.gate, &xb, None, &gate, false),
                gemv(&layer.up, &xb, None, &up, false),
                element("swiglu", &gate, Some(&up), &act, ff),
            ];
            if fuse_gate_up && layer.gate.kind == 12 && layer.up.kind == 12 {
                let mut p = base;
                p[0] = n as u32;
                p[1] = ff as u32;
                p[2] = 12;
                prefill_replacements.insert(ops.len(), projections);
                ops.push(gpu.op_extra(
                    "gate_up_q4k",
                    Some(&layer.gate.buffer),
                    Some(&xb),
                    None,
                    &act,
                    None,
                    p,
                    [(ff as u32).div_ceil(4), 1, 1],
                    Some(&layer.up.buffer),
                ));
            } else {
                ops.extend(projections);
            }
            ops.push(gemv(&layer.down, &act, None, &x, true));
        }
        let mut ap = base;
        ap[0] = cfg.n_vocab as u32;
        let head = vec![
            rms(&norm, &x, &xb, true),
            gemv(output.as_ref().unwrap_or(&emb), &xb, None, &logits, false),
            gpu.op(
                "argmax",
                None,
                Some(&logits),
                None,
                &next,
                None,
                ap,
                [1, 1, 1],
            ),
        ];
        let fused_layers = prefill_replacements.len();
        let mut matrices: Vec<&Matrix> = vec![&emb, &norm];
        matrices.extend(output.iter());
        for l in &layers {
            matrices.extend([
                &l.an, &l.q, &l.k, &l.v, &l.o, &l.fnorm, &l.gate, &l.up, &l.down,
            ]);
            matrices.extend(l.qb.iter());
            matrices.extend(l.kb.iter());
            matrices.extend(l.vb.iter());
        }
        let gpu_weight_bytes = matrices.iter().map(|m| m.buffer.size()).sum();
        let q6_scaled_tensors = matrices.iter().filter(|m| m.kind == 114).count();
        let prefill_ops: Vec<Op> = ops
            .iter()
            .enumerate()
            .flat_map(|(i, op)| {
                prefill_replacements
                    .get(&i)
                    .cloned()
                    .unwrap_or_else(|| vec![op.clone()])
            })
            .collect();
        gpu.queue.submit([]);
        gpu.device.poll(wgpu::PollType::Wait)?;
        let weight_bytes = file.total_tensor_bytes();
        Ok(Self {
            gpu,
            ops,
            prefill_ops,
            head,
            logits,
            next,
            readback,
            tokens,
            prefill_batch,
            fused_layers,
            q6_scaled_tensors,
            gpu_weight_bytes,
            cfg,
            len: 0,
            max_seq,
            weight_bytes,
        })
    }
    pub fn adapter(&self) -> &str {
        &self.gpu.adapter
    }
    pub fn prefill_batch(&self) -> usize {
        self.prefill_batch
    }
    pub fn kernel_settings(&self) -> serde_json::Value {
        serde_json::json!({"gemm_tile_k":self.gpu.gemm_tile_k,"fused_feedforward_layers":self.fused_layers,"q6_scaled_tensors":self.q6_scaled_tensors,"gpu_weight_bytes":self.gpu_weight_bytes})
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }
    pub fn sync(&self) -> Result<()> {
        self.gpu.device.poll(wgpu::PollType::Wait)?;
        Ok(())
    }
    /// Cache all input tokens; only the final token runs the full output head.
    pub fn forward_greedy(&mut self, tokens: &[u32]) -> Result<u32> {
        ensure!(!tokens.is_empty(), "empty prompt");
        ensure!(
            tokens.len() <= self.max_seq.saturating_sub(self.len),
            "context full: {} + {} > {}",
            self.len,
            tokens.len(),
            self.max_seq
        );
        ensure!(
            tokens.iter().all(|&id| (id as usize) < self.cfg.n_vocab),
            "token outside vocabulary"
        );
        let batches = tokens.chunks(self.prefill_batch);
        let num_batches = batches.len();
        for (i, batch) in batches.enumerate() {
            let token_ids: Vec<f32> = batch.iter().map(|&id| id as f32).collect();
            self.gpu
                .queue
                .write_buffer(&self.tokens, 0, bytemuck::cast_slice(&token_ids));
            self.gpu.queue.write_buffer(
                &self.gpu.state,
                0,
                bytemuck::cast_slice(&[self.len as u32, batch[0], batch.len() as u32, 0]),
            );
            let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
            self.gpu.encode_count(
                &mut encoder,
                if batch.len() > 1 {
                    &self.prefill_ops
                } else {
                    &self.ops
                },
                batch.len() as u32,
            );
            if i + 1 == num_batches {
                self.gpu.encode(&mut encoder, &self.head);
                encoder.copy_buffer_to_buffer(&self.next, 0, &self.readback, 0, 4);
            }
            self.gpu.queue.submit([encoder.finish()]);
            self.len += batch.len();
        }
        let token = self.gpu.read_mapped(&self.readback)?[0];
        ensure!(
            (token as usize) < self.cfg.n_vocab,
            "invalid GPU argmax output {token}"
        );
        Ok(token)
    }
    pub fn current_logits(&self) -> Result<Vec<f32>> {
        ensure!(self.len > 0, "no current logits");
        let out: Vec<f32> = self
            .gpu
            .read(&self.logits, self.cfg.n_vocab)?
            .into_iter()
            .map(f32::from_bits)
            .collect();
        ensure!(out.iter().all(|v| v.is_finite()), "non-finite wgpu logits");
        Ok(out)
    }

    /// Diagnostic only: GPU timestamps around each dispatch, after caller warmup.
    /// Never enabled during throughput measurements.
    pub fn profile_decode(&mut self, token: u32) -> Result<serde_json::Value> {
        ensure!(
            self.gpu
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
            "set TARAFER_WGPU_PROFILE=1 before loading to enable GPU timestamps"
        );
        ensure!(
            self.len < self.max_seq && (token as usize) < self.cfg.n_vocab,
            "invalid profiling token/context"
        );
        self.gpu.queue.write_buffer(
            &self.gpu.state,
            0,
            bytemuck::cast_slice(&[self.len as u32, token, 1, 0]),
        );
        let count = ((self.ops.len() + self.head.len()) * 2) as u32;
        let queries = self.gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("dispatch timings"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolved = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: count as u64 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("profile decode"),
                timestamp_writes: None,
            });
            for (i, op) in self.ops.iter().chain(&self.head).enumerate() {
                pass.write_timestamp(&queries, (i * 2) as u32);
                pass.set_pipeline(&self.gpu.pipelines[op.pipeline]);
                pass.set_bind_group(0, &op.bindings, &[]);
                pass.dispatch_workgroups(op.grid[0], op.grid[1], op.grid[2]);
                pass.write_timestamp(&queries, (i * 2 + 1) as u32);
            }
        }
        encoder.resolve_query_set(&queries, 0..count, &resolved, 0);
        self.gpu.queue.submit([encoder.finish()]);
        self.len += 1;
        let words = self.gpu.read(&resolved, count as usize * 2)?;
        let times: Vec<u64> = words
            .chunks_exact(2)
            .map(|w| w[0] as u64 | ((w[1] as u64) << 32))
            .collect();
        let mut totals = std::collections::BTreeMap::<String, (usize, f64)>::new();
        for (i, op) in self.ops.iter().chain(&self.head).enumerate() {
            let mut name = self
                .gpu
                .entries
                .iter()
                .find(|(_, &index)| index == op.pipeline)
                .unwrap()
                .0
                .to_string();
            if name.starts_with("gemv") || name.starts_with("gate_up") {
                name = format!("{name}:{}x{}", op.cols, op.rows);
            }
            let entry = totals.entry(name).or_default();
            entry.0 += 1;
            entry.1 += times[i * 2 + 1].wrapping_sub(times[i * 2]) as f64
                * self.gpu.queue.get_timestamp_period() as f64
                / 1e6;
        }
        Ok(
            serde_json::json!({"adapter":self.adapter(), "dispatches":self.ops.len()+self.head.len(), "gpu_ms": totals.values().map(|(_,t)| t).sum::<f64>(), "operations":totals}),
        )
    }
}

#[cfg(test)]
mod tests;
