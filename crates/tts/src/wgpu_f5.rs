//! Native F5-Myanmar-TTS v2 and Vocos forward passes on Burn's wgpu backend.
//! Layout inside this module is [batch, time, channels]. No ONNX sessions are used.
use anyhow::{bail, Context, Result};
use burn::{
    backend::{
        wgpu::{self, FloatElement, WgpuDevice},
        Wgpu,
    },
    tensor::{
        activation, backend::Backend, module::conv1d, ops::ConvOptions, Int, Tensor, TensorData,
    },
};
use half::slice::HalfFloatSliceExt;
use safetensors::{Dtype, SafeTensors};
use std::{collections::HashMap, path::Path};

type B<F> = Wgpu<F, i32>;
type T<F> = Tensor<B<F>, 3>;
type T4<F> = Tensor<B<F>, 4>;
/// Attention rows scored at once; bounds the `[2, 16, rows, n]` score buffers. It equals the
/// frame limit, so attention runs in one piece (splitting it cost about 2% per step).
const ATTENTION_ROWS: usize = 2048;
const EMBEDDING_TABLE: &str = "text_embed.text_embed.weight";

struct Weights<F: FloatElement> {
    values: HashMap<String, T<F>>,
    shapes: HashMap<String, Vec<usize>>,
}

impl<F: FloatElement> Weights<F> {
    fn linear_shape(&self, name: &str, output: usize, input: usize) -> Result<()> {
        self.require(&format!("{name}.weight"), &[output, input])?;
        self.require(&format!("{name}.bias"), &[output])
    }
    fn norm_shape(&self, name: &str, width: usize) -> Result<()> {
        self.require(&format!("{name}.weight"), &[width])?;
        self.require(&format!("{name}.bias"), &[width])
    }
    fn conv_shape(&self, name: &str, output: usize, input: usize, kernel: usize) -> Result<()> {
        self.require(&format!("{name}.weight"), &[output, input, kernel])?;
        self.require(&format!("{name}.bias"), &[output])
    }
    fn load(path: &Path, device: &WgpuDevice, prefix: &str, keep: fn(&str) -> bool) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let mut values = HashMap::new();
        let mut shapes = HashMap::new();
        for (name, view) in tensors.tensors() {
            let Some(name) = name.strip_prefix(prefix) else {
                continue;
            };
            if !keep(name) {
                continue;
            }
            let shape = view.shape().to_vec();
            if shape.is_empty() || shape.len() > 3 {
                bail!("unsupported weight {name}: {shape:?}");
            }
            let data: Vec<f32> = match view.dtype() {
                Dtype::F32
                    if cfg!(target_endian = "little")
                        && bytemuck::try_cast_slice::<u8, f32>(view.data()).is_ok() =>
                {
                    bytemuck::cast_slice::<u8, f32>(view.data()).to_vec()
                }
                Dtype::F16
                    if cfg!(target_endian = "little")
                        && bytemuck::try_cast_slice::<u8, half::f16>(view.data()).is_ok() =>
                {
                    let src = bytemuck::cast_slice::<u8, half::f16>(view.data());
                    let mut dst = vec![0f32; src.len()];
                    src.convert_to_f32_slice(&mut dst);
                    dst
                }
                Dtype::F32 => view
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                Dtype::F16 => view
                    .data()
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes(b.try_into().unwrap()).to_f32())
                    .collect(),
                other => bail!("unsupported weight dtype {other:?}: {name}"),
            };
            // Linear weights are stored transposed, as `[1, in, out]`, so that every matmul
            // gets a contiguous right-hand side; the embedding table keeps its row layout.
            let transpose = shape.len() == 2 && name.ends_with(".weight") && name != EMBEDDING_TABLE;
            let data = if transpose {
                let (rows, cols) = (shape[0], shape[1]);
                let mut turned = vec![0f32; data.len()];
                for r in 0..rows {
                    for c in 0..cols {
                        turned[c * rows + r] = data[r * cols + c];
                    }
                }
                turned
            } else {
                data
            };
            let dims = match shape.as_slice() {
                [c] => [1, 1, *c],
                [r, c] if transpose => [1, *c, *r],
                [r, c] => [1, *r, *c],
                [a, b, c] => [*a, *b, *c],
                _ => unreachable!(),
            };
            values.insert(
                name.to_owned(),
                T::<F>::from_data(TensorData::new(data, dims), device),
            );
            shapes.insert(name.to_owned(), shape);
            if values.len() % 32 == 0 {
                B::<F>::sync(device);
            }
        }
        B::<F>::sync(device);
        Ok(Self { values, shapes })
    }

    fn get(&self, name: &str) -> T<F> {
        self.values[name].clone()
    }
    fn linear(&self, x: T<F>, name: &str) -> T<F> {
        let y = x.matmul(self.get(&format!("{name}.weight")));
        match self.values.get(&format!("{name}.bias")) {
            Some(b) => y + b.clone(),
            None => y,
        }
    }
    fn conv(&self, x: T<F>, name: &str, groups: usize) -> T<F> {
        let w = self.get(&format!("{name}.weight"));
        let [out, _, kernel] = w.dims();
        let b = self.get(&format!("{name}.bias")).reshape([out]);
        conv1d(
            x.swap_dims(1, 2),
            w,
            Some(b),
            ConvOptions::new([1], [kernel / 2], [1], groups),
        )
        .swap_dims(1, 2)
    }
    fn norm(&self, x: T<F>, name: &str) -> T<F> {
        norm(x) * self.get(&format!("{name}.weight")) + self.get(&format!("{name}.bias"))
    }
    fn require(&self, name: &str, shape: &[usize]) -> Result<()> {
        if self.shapes.get(name).map(Vec::as_slice) != Some(shape) {
            bail!("missing or incompatible F5/Vocos tensor {name}, expected {shape:?}");
        }
        Ok(())
    }
}

/// Layer norm without affine terms. F5 hidden rows carry outlier channels in the thousands,
/// whose squares overflow half precision, so each row is first divided by its largest magnitude.
fn norm<F: FloatElement>(x: T<F>) -> T<F> {
    let centered = x.clone() - x.mean_dim(2);
    let centered = centered.clone() / (centered.abs().max_dim(2) + 1e-4);
    let variance = (centered.clone() * centered.clone()).mean_dim(2);
    centered / (variance + 1e-6).sqrt()
}

fn gelu_tanh<F: FloatElement>(x: T<F>) -> T<F> {
    let cube = x.clone() * x.clone() * x.clone();
    x.clone()
        * 0.5
        * (((x + cube * 0.044715) * (2.0f32 / std::f32::consts::PI).sqrt()).tanh() + 1.0)
}

fn read<F: FloatElement, const D: usize>(x: Tensor<B<F>, D>) -> Result<Vec<f32>> {
    let values = x
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("wgpu readback: {e:?}"))?;
    if values.iter().any(|v| !v.is_finite()) {
        bail!("wgpu produced non-finite audio activations");
    }
    Ok(values)
}

/// Per-user cache directory for compiled GPU kernels.
fn cache_dir() -> Option<std::path::PathBuf> {
    let base = if cfg!(windows) {
        std::path::PathBuf::from(std::env::var_os("LOCALAPPDATA")?)
    } else if let Some(cache) = std::env::var_os("XDG_CACHE_HOME") {
        std::path::PathBuf::from(cache)
    } else {
        std::path::PathBuf::from(std::env::var_os("HOME")?).join(".cache")
    };
    Some(base.join("taraference"))
}

/// Burn measures the matmul and reduction variants once per shape class and remembers the
/// winner. Its default location is the `target` directory of whatever project the process
/// starts in, so keep the results next to the kernel cache instead. A new shape class (for
/// example a much longer sentence) is tuned on first use, which can take tens of seconds.
fn pin_autotune_cache(dir: std::path::PathBuf) {
    use cubecl_runtime::config::GlobalConfig;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // The cache location type is private in cubecl-runtime 0.6, so go through serde.
        let config = serde_json::json!({ "autotune": { "cache": { "file": dir } } });
        if let Ok(config) = serde_json::from_value::<GlobalConfig>(config) {
            GlobalConfig::set(config.override_from_env());
        }
    });
}

/// F5-Myanmar-TTS v2 on wgpu: the DiT in half or full precision, text encoder and Vocos in fp32.
pub enum NativeF5 {
    Half(Engine<burn::tensor::f16>),
    Full(Engine<f32>),
}

impl NativeF5 {
    /// Half precision on Vulkan adapters (SPIR-V shaders), full precision elsewhere.
    /// `TARAFER_WGPU_PRECISION=fp16|fp32` overrides the choice.
    pub fn load(dir: &Path) -> Result<Self> {
        // Compiling the SPIR-V kernels takes seconds, so keep them between runs
        // (vendor/cubecl-wgpu/TARAFERENCE.md). Must be set before the runtime starts.
        if std::env::var_os("CUBECL_KERNEL_CACHE").is_none() {
            if let Some(dir) = cache_dir() {
                std::env::set_var("CUBECL_KERNEL_CACHE", dir.join("wgpu-kernels"));
            }
        }
        if let Some(dir) = cache_dir() {
            pin_autotune_cache(dir.join("wgpu-autotune"));
        }
        let device = WgpuDevice::default();
        // Default sliced pages can reserve multi-GiB slabs even for small weights.
        // Exclusive pages keep this model practical on a 4 GiB adapter.
        let options = wgpu::RuntimeOptions {
            memory_config: wgpu::MemoryConfiguration::ExclusivePages,
            ..Default::default()
        };
        let setup = wgpu::init_setup::<wgpu::graphics::AutoGraphicsApi>(&device, options);
        let info = setup.adapter.get_info();
        let backend = format!("{:?}", info.backend);
        let half = match std::env::var("TARAFER_WGPU_PRECISION").as_deref() {
            Ok("fp16") => true,
            Ok("fp32") => false,
            Ok(other) => bail!("TARAFER_WGPU_PRECISION must be fp16 or fp32, got {other:?}"),
            Err(_) => backend == "Vulkan",
        };
        Ok(if half {
            Self::Half(Engine::load(dir, device, format!("{} ({backend}, wgpu fp16)", info.name))?)
        } else {
            Self::Full(Engine::load(dir, device, format!("{} ({backend}, wgpu fp32)", info.name))?)
        })
    }

    /// Adapter, graphics backend and precision, for reporting measurements.
    pub fn adapter(&self) -> &str {
        match self {
            Self::Half(engine) => &engine.adapter,
            Self::Full(engine) => &engine.adapter,
        }
    }

    /// Exposed for fixed-input parity tests against the existing ONNX export.
    pub fn velocity(&self, x: &[f32], cond: &[f32], ids: &[i64], time: f32, cfg: f32) -> Result<Vec<f32>> {
        match self {
            Self::Half(engine) => engine.velocity(x, cond, ids, time, cfg),
            Self::Full(engine) => engine.velocity(x, cond, ids, time, cfg),
        }
    }

    pub fn sample(&self, noise: &[f32], cond: &[f32], ids: &[i64], timesteps: &[f32], cfg: f32) -> Result<Vec<f32>> {
        match self {
            Self::Half(engine) => engine.sample(noise, cond, ids, timesteps, cfg),
            Self::Full(engine) => engine.sample(noise, cond, ids, timesteps, cfg),
        }
    }

    /// Input mel is [1, 100, frames]; outputs are [1, 513, frames].
    pub fn spectrogram(&self, mel: &[f32], n: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        match self {
            Self::Half(engine) => engine.spectrogram(mel, n),
            Self::Full(engine) => engine.spectrogram(mel, n),
        }
    }
}

/// First native speech model. Only the known 1024-wide, 22-block F5 v2 architecture
/// and 512-wide, eight-block Vocos checkpoint are accepted.
pub struct Engine<F: FloatElement> {
    dit: Weights<F>,
    /// Text encoder weights; it runs once per utterance and sums over time, so it stays fp32.
    text: Weights<f32>,
    vocos: Weights<f32>,
    /// Grouped position convolutions as `[1, 16, 31 * 64, 64]` matrices with `[1, 1, 1024]` biases.
    pos_conv: [(T4<F>, T<F>); 2],
    device: WgpuDevice,
    inv_freq: Vec<f32>,
    adapter: String,
}

/// Rotary tables in `[1, n, 1024]` layout; the query tables carry the attention scale.
struct Rotary<F: FloatElement> {
    query: [T<F>; 2],
    key: [T<F>; 2],
    /// Swaps the two channels of each rotary pair.
    swap: Tensor<B<F>, 1, Int>,
}

impl<F: FloatElement> Engine<F> {
    fn load(dir: &Path, device: WgpuDevice, adapter: String) -> Result<Self> {
        eprintln!("adapter| {adapter}");
        let model = dir.join("model.safetensors");
        let dit = Weights::<F>::load(&model, &device, "transformer.", |_| true)?;
        let text = Weights::<f32>::load(&model, &device, "transformer.", |n| n.starts_with("text_embed."))?;
        let vocos = Weights::<f32>::load(&dir.join("vocos.safetensors"), &device, "", |_| true)?;
        eprintln!("weights| F5-Myanmar-TTS v2 + Vocos uploaded");
        dit.require("input_embed.proj.weight", &[1024, 712])?;
        dit.require("text_embed.text_embed.weight", &[2627, 512])?;
        dit.linear_shape("input_embed.proj", 1024, 712)?;
        dit.linear_shape("time_embed.time_mlp.0", 1024, 256)?;
        dit.linear_shape("time_embed.time_mlp.2", 1024, 1024)?;
        dit.linear_shape("norm_out.linear", 2048, 1024)?;
        dit.linear_shape("proj_out", 100, 1024)?;
        for i in [0, 2] {
            dit.conv_shape(
                &format!("input_embed.conv_pos_embed.conv1d.{i}"),
                1024,
                64,
                31,
            )?;
        }
        for i in 0..4 {
            let p = format!("text_embed.text_blocks.{i}");
            dit.conv_shape(&format!("{p}.dwconv"), 512, 1, 7)?;
            dit.norm_shape(&format!("{p}.norm"), 512)?;
            dit.linear_shape(&format!("{p}.pwconv1"), 1024, 512)?;
            dit.linear_shape(&format!("{p}.pwconv2"), 512, 1024)?;
            for field in ["gamma", "beta"] {
                dit.require(&format!("{p}.grn.{field}"), &[1, 1, 1024])?;
            }
        }
        for i in 0..22 {
            let p = format!("transformer_blocks.{i}");
            dit.linear_shape(&format!("{p}.attn_norm.linear"), 6144, 1024)?;
            for field in ["to_q", "to_k", "to_v", "to_out.0"] {
                dit.linear_shape(&format!("{p}.attn.{field}"), 1024, 1024)?;
            }
            dit.linear_shape(&format!("{p}.ff.ff.0.0"), 2048, 1024)?;
            dit.linear_shape(&format!("{p}.ff.ff.2"), 1024, 2048)?;
        }
        if dit.values.len() != 364 {
            bail!(
                "unsupported F5 architecture: expected 364 checkpoint tensors, got {}",
                dit.values.len()
            );
        }
        vocos.conv_shape("backbone.embed", 512, 100, 7)?;
        vocos.norm_shape("backbone.norm", 512)?;
        vocos.norm_shape("backbone.final_layer_norm", 512)?;
        vocos.linear_shape("head.out", 1026, 512)?;
        for i in 0..8 {
            let p = format!("backbone.convnext.{i}");
            vocos.conv_shape(&format!("{p}.dwconv"), 512, 1, 7)?;
            vocos.norm_shape(&format!("{p}.norm"), 512)?;
            vocos.linear_shape(&format!("{p}.pwconv1"), 1536, 512)?;
            vocos.linear_shape(&format!("{p}.pwconv2"), 512, 1536)?;
            vocos.require(&format!("{p}.gamma"), &[512])?;
        }
        dit.require("rotary_embed.inv_freq", &[32])?;
        let inv_freq = read(dit.get("rotary_embed.inv_freq"))?;
        let pos_conv = [0, 2].map(|i| {
            let name = format!("input_embed.conv_pos_embed.conv1d.{i}");
            // [out, in / groups, taps] -> [group, tap, in, out], one matrix per group.
            let weight = dit
                .get(&format!("{name}.weight"))
                .reshape([16, 64, 64, 31])
                .permute([0, 3, 2, 1])
                .reshape([1, 16, 31 * 64, 64]);
            (weight, dit.get(&format!("{name}.bias")))
        });
        Ok(Self {
            dit,
            text,
            vocos,
            pos_conv,
            device,
            inv_freq,
            adapter,
        })
    }

    fn tensor<P: FloatElement>(&self, data: Vec<f32>, shape: [usize; 3]) -> T<P> {
        T::<P>::from_data(TensorData::new(data, shape), &self.device)
    }

    /// Text features `[1, n, 512]`, computed in fp32 and returned on the CPU.
    fn text_embed(&self, ids: &[i64], drop: bool) -> Result<Vec<f32>> {
        let n = ids.len();
        if ids.iter().any(|id| !(0..2627).contains(id)) {
            bail!("F5 token id outside vocabulary");
        }
        let indices = Tensor::<B<f32>, 1, Int>::from_data(
            TensorData::new(
                ids.iter()
                    .map(|&x| if drop { 0 } else { x as i32 })
                    .collect::<Vec<_>>(),
                [n],
            ),
            &self.device,
        );
        let table = self
            .text
            .get("text_embed.text_embed.weight")
            .reshape([2627, 512]);
        let mut pos = vec![0f32; n * 512];
        let mut mask = vec![0f32; n];
        for t in 0..n {
            mask[t] = if ids[t] == 0 { 0.0 } else { 1.0 };
            for j in 0..256 {
                let angle = t as f32 / 10000f32.powf((2 * j) as f32 / 512.0);
                pos[t * 512 + j] = angle.cos();
                pos[t * 512 + j + 256] = angle.sin();
            }
        }
        let mask: T<f32> = self.tensor(mask, [1, n, 1]);
        let mut x = (table.select(0, indices).reshape([1, n, 512]) + self.tensor(pos, [1, n, 512]))
            * mask.clone();
        for i in 0..4 {
            let p = format!("text_embed.text_blocks.{i}");
            let y = self.text.conv(x.clone(), &format!("{p}.dwconv"), 512);
            let y = self.text.norm(y, &format!("{p}.norm"));
            let y = activation::gelu(self.text.linear(y, &format!("{p}.pwconv1")));
            let gx = (y.clone() * y.clone()).sum_dim(1).sqrt();
            let nx = gx.clone() / (gx.mean_dim(2) + 1e-6);
            let y = self.text.get(&format!("{p}.grn.gamma")) * (y.clone() * nx)
                + self.text.get(&format!("{p}.grn.beta"))
                + y;
            x = (x + self.text.linear(y, &format!("{p}.pwconv2"))) * mask.clone();
        }
        read(x)
    }

    /// The 31-tap, 16-group position convolution as one matmul per group over stacked taps;
    /// Burn's general convolution is several times slower for this shape.
    fn pos_conv(&self, x: T<F>, index: usize) -> T<F> {
        let [b, n, _] = x.dims();
        let pad = T::<F>::zeros([b, 15, 1024], &self.device);
        let x = T::cat(vec![pad.clone(), x, pad], 1).reshape([b, n + 30, 16, 64]).swap_dims(1, 2);
        let taps = (0..31).map(|t| x.clone().slice([0..b, 0..16, t..t + n, 0..64])).collect();
        let (weight, bias) = &self.pos_conv[index];
        let y = Tensor::cat(taps, 3).matmul(weight.clone());
        y.swap_dims(1, 2).reshape([b, n, 1024]) + bias.clone()
    }

    fn input_embed(&self, x: T<F>) -> T<F> {
        let x = self.dit.linear(x, "input_embed.proj");
        let pos = activation::mish(self.pos_conv(x.clone(), 0));
        let pos = activation::mish(self.pos_conv(pos, 1));
        x + pos
    }

    fn time_embed(&self, time: f32) -> T<F> {
        let mut data = vec![0f32; 256];
        for i in 0..128 {
            let a = 1000.0 * time * (-(10000f32.ln()) * i as f32 / 127.0).exp();
            data[i] = a.sin();
            data[i + 128] = a.cos();
        }
        let x = self.dit.linear(self.tensor(data, [1, 1, 256]), "time_embed.time_mlp.0");
        self.dit.linear(activation::silu(x), "time_embed.time_mlp.2")
    }

    /// Rotates adjacent channel pairs (the x-transformers layout) and splits into heads.
    fn rope(x: T<F>, tables: &[T<F>; 2], swap: &Tensor<B<F>, 1, Int>) -> T4<F> {
        let [b, n, _] = x.dims();
        let rotated = x.clone().select(2, swap.clone()) * tables[1].clone();
        (x * tables[0].clone() + rotated).reshape([b, n, 16, 64]).swap_dims(1, 2)
    }

    fn rotary_tables(&self, n: usize) -> Rotary<F> {
        let mut cos = vec![0f32; n * 1024];
        let mut sin = cos.clone();
        for t in 0..n {
            for c in 0..1024 {
                let angle = t as f32 * self.inv_freq[c % 64 / 2];
                cos[t * 1024 + c] = angle.cos();
                // rotated[2j] = -x[2j + 1], rotated[2j + 1] = x[2j]
                sin[t * 1024 + c] = if c % 2 == 0 { -angle.sin() } else { angle.sin() };
            }
        }
        let scaled = |v: &[f32]| v.iter().map(|x| x * 0.125).collect::<Vec<_>>();
        let swap: Vec<i32> = (0..1024).map(|c| c ^ 1).collect();
        Rotary {
            query: [self.tensor(scaled(&cos), [1, n, 1024]), self.tensor(scaled(&sin), [1, n, 1024])],
            key: [self.tensor(cos, [1, n, 1024]), self.tensor(sin, [1, n, 1024])],
            swap: Tensor::from_data(TensorData::new(swap, [1024]), &self.device),
        }
    }

    /// Conditional and unconditional velocities, `[2, n, 100]` flattened.
    /// `fixed` is `[2, n, 612]`: reference mel and text features for both branches.
    fn forward(&self, x: &[f32], fixed: &T<F>, rotary: &Rotary<F>, time: f32) -> Result<Vec<f32>> {
        let n = x.len() / 100;
        let x: T<F> = self.tensor([x, x].concat(), [2, n, 100]);
        let mut h = self.input_embed(T::cat(vec![x, fixed.clone()], 2));
        let time = activation::silu(self.time_embed(time));
        let ones = T4::<F>::ones([2, 16, n, 1], &self.device);
        for i in 0..22 {
            let p = format!("transformer_blocks.{i}");
            let ada = self.dit.linear(time.clone(), &format!("{p}.attn_norm.linear"));
            let part = |j| ada.clone().slice([0..1, 0..1, j * 1024..(j + 1) * 1024]);
            let y = norm(h.clone()) * (part(1) + 1.0) + part(0);
            let q = Self::rope(self.dit.linear(y.clone(), &format!("{p}.attn.to_q")), &rotary.query, &rotary.swap);
            let k = Self::rope(self.dit.linear(y.clone(), &format!("{p}.attn.to_k")), &rotary.key, &rotary.swap)
                .swap_dims(2, 3);
            let v = self.dit.linear(y, &format!("{p}.attn.to_v")).reshape([2, n, 16, 64]).swap_dims(1, 2);
            // A trailing column of ones makes one matmul return both the weighted values
            // and the softmax denominator, so the [2,16,n,n] weights are never normalised.
            let v = Tensor::cat(vec![v, ones.clone()], 3);
            let mut pieces = Vec::new();
            for start in (0..n).step_by(ATTENTION_ROWS) {
                let end = (start + ATTENTION_ROWS).min(n);
                let rows = end - start;
                let query = if rows == n { q.clone() } else { q.clone().slice([0..2, 0..16, start..end, 0..64]) };
                let scores = query.matmul(k.clone());
                let weights = (scores.clone() - scores.max_dim(3)).exp();
                let mixed = weights.matmul(v.clone());
                pieces.push(
                    mixed.clone().slice([0..2, 0..16, 0..rows, 0..64]) / mixed.slice([0..2, 0..16, 0..rows, 64..65]),
                );
            }
            let att = if pieces.len() == 1 { pieces.pop().unwrap() } else { Tensor::cat(pieces, 2) };
            let att = att.swap_dims(1, 2).reshape([2, n, 1024]);
            h = h + part(2) * self.dit.linear(att, &format!("{p}.attn.to_out.0"));
            let y = norm(h.clone()) * (part(4) + 1.0) + part(3);
            let y = gelu_tanh(self.dit.linear(y, &format!("{p}.ff.ff.0.0")));
            h = h + part(5) * self.dit.linear(y, &format!("{p}.ff.ff.2"));
        }
        let ada = self.dit.linear(time, "norm_out.linear");
        let scale = ada.clone().slice([0..1, 0..1, 0..1024]);
        let shift = ada.slice([0..1, 0..1, 1024..2048]);
        read(self.dit.linear(norm(h) * (scale + 1.0) + shift, "proj_out"))
    }

    /// Reference mel and text features for the conditional and unconditional branches.
    fn fixed_inputs(&self, cond: &[f32], ids: &[i64]) -> Result<T<F>> {
        let n = ids.len();
        let mut data = Vec::with_capacity(2 * n * 612);
        for (drop, text) in [(false, self.text_embed(ids, false)?), (true, self.text_embed(ids, true)?)] {
            for t in 0..n {
                if drop {
                    data.extend(std::iter::repeat(0f32).take(100));
                } else {
                    data.extend_from_slice(&cond[t * 100..(t + 1) * 100]);
                }
                data.extend_from_slice(&text[t * 512..(t + 1) * 512]);
            }
        }
        Ok(self.tensor(data, [2, n, 612]))
    }

    /// Classifier-free guidance, in fp32 on the CPU.
    fn guided(&self, x: &[f32], fixed: &T<F>, rotary: &Rotary<F>, time: f32, cfg: f32) -> Result<Vec<f32>> {
        let both = self.forward(x, fixed, rotary, time)?;
        let (c, u) = both.split_at(x.len());
        Ok(c.iter().zip(u).map(|(c, u)| c + (c - u) * cfg).collect())
    }

    fn velocity(&self, x: &[f32], cond: &[f32], ids: &[i64], time: f32, cfg: f32) -> Result<Vec<f32>> {
        let n = ids.len();
        validate_inputs(n, x, cond, cfg)?;
        if !time.is_finite() {
            bail!("non-finite F5 timestep");
        }
        self.guided(x, &self.fixed_inputs(cond, ids)?, &self.rotary_tables(n), time, cfg)
    }

    fn sample(&self, noise: &[f32], cond: &[f32], ids: &[i64], timesteps: &[f32], cfg: f32) -> Result<Vec<f32>> {
        let n = ids.len();
        validate_inputs(n, noise, cond, cfg)?;
        if timesteps.len() < 2
            || timesteps.iter().any(|t| !t.is_finite())
            || timesteps.windows(2).any(|w| w[1] <= w[0])
        {
            bail!("F5 requires finite, increasing timesteps");
        }
        let trace = std::env::var_os("TARAFER_WGPU_TRACE").is_some();
        let started = std::time::Instant::now();
        let fixed = self.fixed_inputs(cond, ids)?;
        let tables = self.rotary_tables(n);
        if trace {
            B::<F>::sync(&self.device);
            eprintln!("wgpu   | F5 text features and rotary tables ({:.3} s)", started.elapsed().as_secs_f64());
        }
        // The Euler state stays in fp32 on the CPU, so half precision does not accumulate.
        let mut x = noise.to_vec();
        for (i, ts) in timesteps.windows(2).enumerate() {
            let started = std::time::Instant::now();
            let velocity = self.guided(&x, &fixed, &tables, ts[0], cfg)?;
            for (x, v) in x.iter_mut().zip(&velocity) {
                *x += v * (ts[1] - ts[0]);
            }
            if trace {
                let seconds = started.elapsed().as_secs_f64();
                eprintln!("wgpu   | F5 step {}/{} ({seconds:.3} s)", i + 1, timesteps.len() - 1);
            }
        }
        Ok(x)
    }

    fn spectrogram(&self, mel: &[f32], n: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        if n == 0 || mel.len() != n * 100 {
            bail!("invalid Vocos mel shape");
        }
        let started = std::time::Instant::now();
        let x: T<f32> = self.tensor(mel.to_vec(), [1, 100, n]).swap_dims(1, 2);
        let mut x = self
            .vocos
            .norm(self.vocos.conv(x, "backbone.embed", 1), "backbone.norm");
        for i in 0..8 {
            let p = format!("backbone.convnext.{i}");
            let y = self.vocos.conv(x.clone(), &format!("{p}.dwconv"), 512);
            let y = self.vocos.norm(y, &format!("{p}.norm"));
            let y = activation::gelu(self.vocos.linear(y, &format!("{p}.pwconv1")));
            let y = self.vocos.linear(y, &format!("{p}.pwconv2"));
            x = x + y * self.vocos.get(&format!("{p}.gamma"));
        }
        let x = self.vocos.norm(x, "backbone.final_layer_norm");
        let x = self.vocos.linear(x, "head.out").swap_dims(1, 2);
        let mag = x.clone().slice([0..1, 0..513, 0..n]).exp().clamp_max(100.0);
        let phase = x.slice([0..1, 513..1026, 0..n]);
        let spectrum = (read(mag.clone() * phase.clone().cos())?, read(mag * phase.sin())?);
        if std::env::var_os("TARAFER_WGPU_TRACE").is_some() {
            eprintln!("wgpu   | Vocos ({:.3} s)", started.elapsed().as_secs_f64());
        }
        Ok(spectrum)
    }
}

fn validate_inputs(n: usize, x: &[f32], cond: &[f32], cfg: f32) -> Result<()> {
    if n == 0 || n > 2048 {
        bail!("native wgpu F5 supports 1..=2048 total mel frames including reference audio; shorten the reference or split the text");
    }
    if x.len() != n * 100 || cond.len() != x.len() {
        bail!("invalid DiT input shapes");
    }
    if !cfg.is_finite() || x.iter().chain(cond).any(|v| !v.is_finite()) {
        bail!("non-finite F5 input");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_inputs;
    #[test]
    fn rejects_invalid_gpu_inputs_before_dispatch() {
        let x = vec![0.; 100];
        assert!(validate_inputs(1, &x, &x, 2.).is_ok());
        assert!(validate_inputs(0, &[], &[], 2.).is_err());
        assert!(validate_inputs(2049, &[], &[], 2.).is_err());
        assert!(validate_inputs(1, &x[..99], &x, 2.).is_err());
        assert!(validate_inputs(1, &x, &x, f32::NAN).is_err());
        let mut bad = x.clone();
        bad[0] = f32::INFINITY;
        assert!(validate_inputs(1, &bad, &x, 2.).is_err());
    }
}
