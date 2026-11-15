//! Native F5-Myanmar-TTS v2 and Vocos forward passes on Burn's wgpu backend.
//! Layout inside this module is [batch, time, channels]. No ONNX sessions are used.
use anyhow::{bail, Context, Result};
use burn::{
    backend::{
        wgpu::{self, WgpuDevice},
        Wgpu,
    },
    tensor::{
        activation, backend::Backend, module::conv1d, ops::ConvOptions, Int, Tensor, TensorData,
    },
};
use half::slice::HalfFloatSliceExt;
use safetensors::{Dtype, SafeTensors};
use std::{collections::HashMap, path::Path};

type B = Wgpu<f32, i32>;
type T = Tensor<B, 3>;

struct Weights {
    values: HashMap<String, T>,
    shapes: HashMap<String, Vec<usize>>,
}

impl Weights {
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
    fn load(path: &Path, device: &WgpuDevice, prefix: &str) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let mut values = HashMap::new();
        let mut shapes = HashMap::new();
        for (name, view) in tensors.tensors() {
            let Some(name) = name.strip_prefix(prefix) else {
                continue;
            };
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
            let dims = match shape.as_slice() {
                [c] => [1, 1, *c],
                [r, c] => [1, *r, *c],
                [a, b, c] => [*a, *b, *c],
                _ => unreachable!(),
            };
            values.insert(
                name.to_owned(),
                T::from_data(TensorData::new(data, dims), device),
            );
            shapes.insert(name.to_owned(), shape);
            if values.len() % 32 == 0 {
                B::sync(device);
            }
        }
        B::sync(device);
        Ok(Self { values, shapes })
    }

    fn get(&self, name: &str) -> T {
        self.values[name].clone()
    }
    fn linear(&self, x: T, name: &str) -> T {
        let w = self.get(&format!("{name}.weight")).swap_dims(1, 2);
        let y = x.matmul(w);
        match self.values.get(&format!("{name}.bias")) {
            Some(b) => y + b.clone(),
            None => y,
        }
    }
    fn conv(&self, x: T, name: &str, groups: usize) -> T {
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
    fn norm(&self, x: T, name: &str) -> T {
        norm(x) * self.get(&format!("{name}.weight")) + self.get(&format!("{name}.bias"))
    }
    fn require(&self, name: &str, shape: &[usize]) -> Result<()> {
        if self.shapes.get(name).map(Vec::as_slice) != Some(shape) {
            bail!("missing or incompatible F5/Vocos tensor {name}, expected {shape:?}");
        }
        Ok(())
    }
}

fn norm(x: T) -> T {
    let centered = x.clone() - x.mean_dim(2);
    let variance = (centered.clone() * centered.clone()).mean_dim(2);
    centered / (variance + 1e-6).sqrt()
}

fn gelu_tanh(x: T) -> T {
    let cube = x.clone() * x.clone() * x.clone();
    x.clone()
        * 0.5
        * (((x + cube * 0.044715) * (2.0f32 / std::f32::consts::PI).sqrt()).tanh() + 1.0)
}

/// First native speech model. Only the known 1024-wide, 22-block F5 v2 architecture
/// and 512-wide, eight-block Vocos checkpoint are accepted.
pub struct NativeF5 {
    dit: Weights,
    vocos: Weights,
    device: WgpuDevice,
    inv_freq: Vec<f32>,
    pub adapter: String,
}

impl NativeF5 {
    pub fn load(dir: &Path) -> Result<Self> {
        let device = WgpuDevice::default();
        // Default sliced pages can reserve multi-GiB slabs even for small weights.
        // Exclusive pages keep this fp32 model practical on a 4 GiB adapter.
        let options = wgpu::RuntimeOptions {
            memory_config: wgpu::MemoryConfiguration::ExclusivePages,
            ..Default::default()
        };
        let setup = wgpu::init_setup::<wgpu::graphics::AutoGraphicsApi>(&device, options);
        let info = setup.adapter.get_info();
        let adapter = format!("{} ({:?}, wgpu fp32)", info.name, info.backend);
        eprintln!("adapter| {adapter}");
        let dit = Weights::load(&dir.join("model.safetensors"), &device, "transformer.")?;
        let vocos = Weights::load(&dir.join("vocos.safetensors"), &device, "")?;
        eprintln!("weights| F5-Myanmar-TTS v2 + Vocos uploaded (fp32)");
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
        let inv_freq = Self::read(dit.get("rotary_embed.inv_freq"))?;
        Ok(Self {
            dit,
            vocos,
            device,
            inv_freq,
            adapter,
        })
    }

    fn tensor(&self, data: Vec<f32>, shape: [usize; 3]) -> T {
        T::from_data(TensorData::new(data, shape), &self.device)
    }
    fn read(x: T) -> Result<Vec<f32>> {
        let data = x.into_data();
        let values = data
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("wgpu readback: {e:?}"))?;
        if values.iter().any(|v| !v.is_finite()) {
            bail!("wgpu produced non-finite audio activations");
        }
        Ok(values)
    }

    fn text_embed(&self, ids: &[i64], drop: bool) -> Result<T> {
        let n = ids.len();
        if ids.iter().any(|id| !(0..2627).contains(id)) {
            bail!("F5 token id outside vocabulary");
        }
        let indices = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                ids.iter()
                    .map(|&x| if drop { 0 } else { x as i32 })
                    .collect::<Vec<_>>(),
                [n],
            ),
            &self.device,
        );
        let table = self
            .dit
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
        let mask = self.tensor(mask, [1, n, 1]);
        let mut x = (table.select(0, indices).reshape([1, n, 512]) + self.tensor(pos, [1, n, 512]))
            * mask.clone();
        for i in 0..4 {
            let p = format!("text_embed.text_blocks.{i}");
            let y = self.dit.conv(x.clone(), &format!("{p}.dwconv"), 512);
            let y = self.dit.norm(y, &format!("{p}.norm"));
            let y = activation::gelu(self.dit.linear(y, &format!("{p}.pwconv1")));
            let gx = (y.clone() * y.clone()).sum_dim(1).sqrt();
            let nx = gx.clone() / (gx.mean_dim(2) + 1e-6);
            let y = self.dit.get(&format!("{p}.grn.gamma")) * (y.clone() * nx)
                + self.dit.get(&format!("{p}.grn.beta"))
                + y;
            x = (x + self.dit.linear(y, &format!("{p}.pwconv2"))) * mask.clone();
        }
        Ok(x)
    }

    fn input_embed(&self, x: T, cond: T, text: T) -> T {
        let x = self
            .dit
            .linear(T::cat(vec![x, cond, text], 2), "input_embed.proj");
        let pos = activation::mish(self.dit.conv(
            x.clone(),
            "input_embed.conv_pos_embed.conv1d.0",
            16,
        ));
        let pos = activation::mish(
            self.dit
                .conv(pos, "input_embed.conv_pos_embed.conv1d.2", 16),
        );
        x + pos
    }

    fn time_embed(&self, time: f32) -> T {
        let mut data = vec![0f32; 256];
        for i in 0..128 {
            let a = 1000.0 * time * (-(10000f32.ln()) * i as f32 / 127.0).exp();
            data[i] = a.sin();
            data[i + 128] = a.cos();
        }
        let x = self
            .dit
            .linear(self.tensor(data, [1, 1, 256]), "time_embed.time_mlp.0");
        self.dit
            .linear(activation::silu(x), "time_embed.time_mlp.2")
    }

    fn rope(x: T, tables: &[Tensor<B, 4>; 2]) -> Tensor<B, 4> {
        let [b, n, _] = x.dims();
        let x = x.reshape([b, n, 16, 64]).swap_dims(1, 2);
        // x-transformers uses adjacent pairs, not a split-half rotary layout.
        let pairs = x.clone().reshape([b, 16, n, 32, 2]);
        let a = pairs.clone().slice([0..b, 0..16, 0..n, 0..32, 0..1]);
        let z = pairs.slice([0..b, 0..16, 0..n, 0..32, 1..2]);
        let rotated = Tensor::cat(vec![-z, a], 4).reshape([b, 16, n, 64]);
        x * tables[0].clone() + rotated * tables[1].clone()
    }

    fn rotary_tables(&self, n: usize) -> [Tensor<B, 4>; 2] {
        let mut cos = vec![0f32; n * 64];
        let mut sin = cos.clone();
        for t in 0..n {
            for j in 0..32 {
                let angle = t as f32 * self.inv_freq[j];
                for k in 0..2 {
                    cos[t * 64 + j * 2 + k] = angle.cos();
                    sin[t * 64 + j * 2 + k] = angle.sin();
                }
            }
        }
        [
            self.tensor(cos, [1, n, 64]).reshape([1, 1, n, 64]),
            self.tensor(sin, [1, n, 64]).reshape([1, 1, n, 64]),
        ]
    }

    fn forward(
        &self,
        x: T,
        cond: T,
        text: &[T; 2],
        tables: &[Tensor<B, 4>; 2],
        time: f32,
        cfg: f32,
    ) -> T {
        let [_, n, _] = x.dims();
        let h_c = self.input_embed(x.clone(), cond.clone(), text[0].clone());
        let h_u = self.input_embed(x, T::zeros(cond.shape(), &self.device), text[1].clone());
        let mut h = T::cat(vec![h_c, h_u], 0);
        let time = activation::silu(self.time_embed(time));
        for i in 0..22 {
            let p = format!("transformer_blocks.{i}");
            let ada = self
                .dit
                .linear(time.clone(), &format!("{p}.attn_norm.linear"));
            let part = |j| ada.clone().slice([0..1, 0..1, j * 1024..(j + 1) * 1024]);
            let y = norm(h.clone()) * (part(1) + 1.0) + part(0);
            let q = Self::rope(
                self.dit.linear(y.clone(), &format!("{p}.attn.to_q")),
                tables,
            );
            let k = Self::rope(
                self.dit.linear(y.clone(), &format!("{p}.attn.to_k")),
                tables,
            );
            let v = self
                .dit
                .linear(y, &format!("{p}.attn.to_v"))
                .reshape([2, n, 16, 64])
                .swap_dims(1, 2);
            // Bound score-buffer memory rather than materializing [2,16,n,n].
            let mut pieces = Vec::new();
            for start in (0..n).step_by(128) {
                let end = (start + 128).min(n);
                let query = q.clone().slice([0..2, 0..16, start..end, 0..64]);
                pieces.push(
                    activation::softmax(query.matmul(k.clone().swap_dims(2, 3)) * 0.125, 3)
                        .matmul(v.clone()),
                );
            }
            let att = Tensor::cat(pieces, 2);
            let att = att.swap_dims(1, 2).reshape([2, n, 1024]);
            h = h + part(2) * self.dit.linear(att, &format!("{p}.attn.to_out.0"));
            let y = norm(h.clone()) * (part(4) + 1.0) + part(3);
            let y = gelu_tanh(self.dit.linear(y, &format!("{p}.ff.ff.0.0")));
            h = h + part(5) * self.dit.linear(y, &format!("{p}.ff.ff.2"));
        }
        let ada = self.dit.linear(time, "norm_out.linear");
        let scale = ada.clone().slice([0..1, 0..1, 0..1024]);
        let shift = ada.slice([0..1, 0..1, 1024..2048]);
        let out = self.dit.linear(norm(h) * (scale + 1.0) + shift, "proj_out");
        let c = out.clone().slice([0..1, 0..n, 0..100]);
        let u = out.slice([1..2, 0..n, 0..100]);
        c.clone() + (c - u) * cfg
    }

    /// Exposed for fixed-input parity tests against the existing ONNX export.
    pub fn velocity(
        &self,
        x: &[f32],
        cond: &[f32],
        ids: &[i64],
        time: f32,
        cfg: f32,
    ) -> Result<Vec<f32>> {
        let n = ids.len();
        validate_inputs(n, x, cond, cfg)?;
        if !time.is_finite() {
            bail!("non-finite F5 timestep");
        }
        let text = [self.text_embed(ids, false)?, self.text_embed(ids, true)?];
        Self::read(self.forward(
            self.tensor(x.to_vec(), [1, n, 100]),
            self.tensor(cond.to_vec(), [1, n, 100]),
            &text,
            &self.rotary_tables(n),
            time,
            cfg,
        ))
    }

    pub fn sample(
        &self,
        noise: &[f32],
        cond: &[f32],
        ids: &[i64],
        timesteps: &[f32],
        cfg: f32,
    ) -> Result<Vec<f32>> {
        let n = ids.len();
        validate_inputs(n, noise, cond, cfg)?;
        if timesteps.len() < 2
            || timesteps.iter().any(|t| !t.is_finite())
            || timesteps.windows(2).any(|w| w[1] <= w[0])
        {
            bail!("F5 requires finite, increasing timesteps");
        }
        let text = [self.text_embed(ids, false)?, self.text_embed(ids, true)?];
        let cond = self.tensor(cond.to_vec(), [1, n, 100]);
        let tables = self.rotary_tables(n);
        let mut x = self.tensor(noise.to_vec(), [1, n, 100]);
        for (i, ts) in timesteps.windows(2).enumerate() {
            let started = std::time::Instant::now();
            let velocity = self.forward(x.clone(), cond.clone(), &text, &tables, ts[0], cfg);
            x = x + velocity * (ts[1] - ts[0]);
            B::sync(&self.device);
            eprintln!(
                "wgpu   | F5 step {}/{} ({:.3} s)",
                i + 1,
                timesteps.len() - 1,
                started.elapsed().as_secs_f64()
            );
        }
        Self::read(x)
    }

    /// Input mel is [1, 100, frames]; outputs are [1, 513, frames].
    pub fn spectrogram(&self, mel: &[f32], n: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        if n == 0 || mel.len() != n * 100 {
            bail!("invalid Vocos mel shape");
        }
        let x = self.tensor(mel.to_vec(), [1, 100, n]).swap_dims(1, 2);
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
        Ok((
            Self::read(mag.clone() * phase.clone().cos())?,
            Self::read(mag * phase.sin())?,
        ))
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
