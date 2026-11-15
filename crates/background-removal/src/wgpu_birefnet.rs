//! Native BiRefNet Lite forward pass on Burn's wgpu backend. No ONNX session is used.
//!
//! Loads the upstream `ZhengPeng7/BiRefNet_lite` safetensors checkpoint directly.
//! Layout is NCHW for feature maps and [1, tokens, channels] inside the Swin-T backbone.
//! Batch norms are folded into their convolutions at load time, and the two 1x1
//! projections that upstream applies to wide concatenations are applied per branch,
//! so the widest full-resolution tensor is 64 channels rather than 120.
use anyhow::{anyhow, bail, ensure, Context, Result};
use burn::{
    backend::{
        wgpu::{self, WgpuDevice},
        Wgpu,
    },
    tensor::{
        activation::{gelu, relu, sigmoid, softmax},
        backend::Backend,
        module::{conv2d, deform_conv2d},
        ops::{ConvOptions, DeformConvOptions},
        Tensor, TensorData,
    },
};
use safetensors::{Dtype, SafeTensors};
use std::{collections::HashMap, path::Path};

type B = Wgpu<f32, i32>;
type T3 = Tensor<B, 3>;
type T4 = Tensor<B, 4>;
type Raw = HashMap<String, (Vec<usize>, Vec<f32>)>;

pub const WEIGHTS_REPO: &str = "ZhengPeng7/BiRefNet_lite";
pub const WEIGHTS_REVISION: &str = "aa62cd87eafb9cc43056d08ef3615a14628b831d";
pub const WEIGHTS_FILE: &str = "model.safetensors";
pub const WEIGHTS_SHA256: &str = "4417d89795250e698c3cb0ae8df15743810065f646f48a694fdfa7ca052d0815";

const EMBED: usize = 96;
const DEPTHS: [usize; 4] = [2, 2, 6, 2];
const HEADS: [usize; 4] = [3, 6, 12, 24];
const WINDOW: usize = 7;
const SHIFT: usize = WINDOW / 2;
const TOKENS: usize = WINDOW * WINDOW;
const ASPP_WIDTH: usize = 256;
/// Upper bound for one deformable im2col buffer. The 7x7 branch at 256x256 would
/// otherwise need 64 * 49 * 65536 f32 = 784 MiB in a single allocation.
const COLUMN_BYTES: usize = 192 << 20;
/// Largest softmax batch (rows of 49 attention scores) known to be exact on Burn 0.18 wgpu.
const SOFTMAX_ROWS: usize = 16384;

/// Named activations captured by [`NativeBiRefNet::trace`] for parity tests.
pub type Trace = Vec<(String, Vec<f32>)>;

pub struct NativeBiRefNet {
    values: HashMap<String, (Vec<usize>, Tensor<B, 1>)>,
    device: WgpuDevice,
    pub adapter: String,
}

fn take(raw: &mut Raw, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    raw.remove(name)
        .with_context(|| format!("missing BiRefNet Lite tensor {name}"))
}

/// Folds every inference-mode batch norm into the convolution in front of it.
fn fold_batch_norms(raw: &mut Raw) -> Result<()> {
    let norms: Vec<String> = raw
        .keys()
        .filter_map(|key| key.strip_suffix(".running_mean").map(str::to_owned))
        .collect();
    for norm in norms {
        let conv = if let Some(prefix) = norm.strip_suffix(".bn_in") {
            format!("{prefix}.conv_in")
        } else if let Some(prefix) = norm.strip_suffix(".bn_out") {
            format!("{prefix}.conv_out")
        } else if let Some(prefix) = norm.strip_suffix(".bn1") {
            format!("{prefix}.conv1")
        } else if let Some(prefix) = norm.strip_suffix(".bn") {
            format!("{prefix}.atrous_conv.regular_conv")
        } else if let Some(prefix) = norm.strip_suffix(".global_avg_pool.2") {
            format!("{prefix}.global_avg_pool.1")
        } else if let Some(prefix) = norm
            .strip_suffix(".1")
            .filter(|prefix| prefix.starts_with("decoder.gdt_convs_"))
        {
            format!("{prefix}.0")
        } else {
            bail!("unexpected batch norm {norm} in BiRefNet Lite checkpoint");
        };
        let gamma = take(raw, &format!("{norm}.weight"))?.1;
        let beta = take(raw, &format!("{norm}.bias"))?.1;
        let mean = take(raw, &format!("{norm}.running_mean"))?.1;
        let variance = take(raw, &format!("{norm}.running_var"))?.1;
        let out = gamma.len();
        let mut bias = raw
            .remove(&format!("{conv}.bias"))
            .map_or_else(|| vec![0.0; out], |entry| entry.1);
        let (shape, weight) = raw
            .get_mut(&format!("{conv}.weight"))
            .with_context(|| format!("missing BiRefNet Lite tensor {conv}.weight"))?;
        ensure!(
            shape[0] == out && bias.len() == out,
            "{norm} does not match {conv}"
        );
        let per = weight.len() / out;
        for channel in 0..out {
            let scale = gamma[channel] / (variance[channel] + 1e-5).sqrt();
            for value in &mut weight[channel * per..(channel + 1) * per] {
                *value *= scale;
            }
            bias[channel] = (bias[channel] - mean[channel]) * scale + beta[channel];
        }
        raw.insert(format!("{conv}.bias"), (vec![out], bias));
    }
    Ok(())
}

/// Expands each relative position table into a ready [1, heads, 49, 49] bias and
/// stores linear weights as [input, output] for `x.matmul(w)`.
fn prepare_backbone(raw: &mut Raw) -> Result<()> {
    let tables: Vec<String> = raw
        .keys()
        .filter_map(|key| {
            key.strip_suffix(".relative_position_bias_table")
                .map(str::to_owned)
        })
        .collect();
    let span = 2 * WINDOW - 1;
    for prefix in tables {
        let (shape, table) = take(raw, &format!("{prefix}.relative_position_bias_table"))?;
        ensure!(
            shape.len() == 2 && shape[0] == span * span,
            "{prefix}: unexpected relative position table {shape:?}"
        );
        let heads = shape[1];
        let mut bias = vec![0.0; heads * TOKENS * TOKENS];
        for i in 0..TOKENS {
            for j in 0..TOKENS {
                let dy = i / WINDOW + WINDOW - 1 - j / WINDOW;
                let dx = i % WINDOW + WINDOW - 1 - j % WINDOW;
                for head in 0..heads {
                    bias[(head * TOKENS + i) * TOKENS + j] = table[(dy * span + dx) * heads + head];
                }
            }
        }
        raw.insert(
            format!("{prefix}.rel_bias"),
            (vec![1, heads, TOKENS, TOKENS], bias),
        );
    }
    for (shape, data) in raw.values_mut() {
        if let [out, input] = shape[..] {
            let mut transposed = vec![0.0; data.len()];
            for o in 0..out {
                for i in 0..input {
                    transposed[i * out + o] = data[o * input + i];
                }
            }
            *data = transposed;
            *shape = vec![input, out];
        }
    }
    Ok(())
}

/// `conv_out1` is a 1x1 projection of cat(upsampled decoder features, image branch).
/// A 1x1 convolution commutes with bilinear resizing, so the decoder half is applied
/// at quarter resolution (`decoder.out_low`) and the image half is merged into the
/// preceding 3x3 convolution (`decoder.out_ipt`). Both are exact.
fn fold_output_head(raw: &mut Raw) -> Result<()> {
    let (shape, head) = take(raw, "decoder.conv_out1.0.weight")?;
    let head_bias = take(raw, "decoder.conv_out1.0.bias")?.1;
    let (ipt_shape, ipt) = take(raw, "decoder.ipt_blk1.conv_out.weight")?;
    let ipt_bias = take(raw, "decoder.ipt_blk1.conv_out.bias")?.1;
    ensure!(
        shape[0] == 1 && shape[2..] == [1, 1] && shape[1] > ipt_shape[0] && head_bias.len() == 1,
        "unexpected BiRefNet Lite output head {shape:?} / {ipt_shape:?}"
    );
    let low = shape[1] - ipt_shape[0];
    let per = ipt.len() / ipt_shape[0];
    let mut merged = vec![0.0; per];
    let mut bias = head_bias[0];
    for (channel, scale) in head[low..].iter().enumerate() {
        for (sum, value) in merged.iter_mut().zip(&ipt[channel * per..]) {
            *sum += scale * value;
        }
        bias += scale * ipt_bias[channel];
    }
    raw.insert(
        "decoder.out_low.weight".into(),
        (vec![1, low, 1, 1], head[..low].to_vec()),
    );
    raw.insert(
        "decoder.out_ipt.weight".into(),
        (vec![1, ipt_shape[1], ipt_shape[2], ipt_shape[3]], merged),
    );
    raw.insert("decoder.out_ipt.bias".into(), (vec![1], vec![bias]));
    Ok(())
}

/// Bilinear weights of PyTorch's `align_corners=True`, stored as [1, input, output].
fn resize_matrix(input: usize, output: usize, device: &WgpuDevice) -> T3 {
    let mut weights = vec![0.0f32; input * output];
    let scale = if output > 1 {
        (input - 1) as f64 / (output - 1) as f64
    } else {
        0.0
    };
    for o in 0..output {
        let at = o as f64 * scale;
        let low = (at.floor() as usize).min(input - 1);
        let high = (low + 1).min(input - 1);
        let fraction = (at - low as f64) as f32;
        weights[low * output + o] += 1.0 - fraction;
        weights[high * output + o] += fraction;
    }
    T3::from_data(TensorData::new(weights, [1, input, output]), device)
}

/// Separable bilinear resize, `F.interpolate(..., mode='bilinear', align_corners=True)`.
fn resize(x: T4, [oh, ow]: [usize; 2]) -> T4 {
    let [_, c, h, w] = x.dims();
    let device = x.device();
    let mut x = x;
    if w != ow {
        x = x
            .reshape([1, c * h, w])
            .matmul(resize_matrix(w, ow, &device))
            .reshape([1, c, h, ow]);
    }
    if h != oh {
        x = x
            .swap_dims(2, 3)
            .reshape([1, c * ow, h])
            .matmul(resize_matrix(h, oh, &device))
            .reshape([1, c, ow, oh])
            .swap_dims(2, 3);
    }
    x
}

/// `torch.roll(x, -shift, dim)`: element `i` of the result is element `i + shift`.
fn roll(x: T4, dim: usize, shift: usize) -> T4 {
    let len = x.dims()[dim];
    Tensor::cat(
        vec![
            x.clone().narrow(dim, shift, len - shift),
            x.narrow(dim, 0, shift),
        ],
        dim,
    )
}

fn size(x: &T4) -> [usize; 2] {
    let [_, _, h, w] = x.dims();
    [h, w]
}

fn read(x: T4) -> Result<Vec<f32>> {
    x.into_data()
        .to_vec::<f32>()
        .map_err(|error| anyhow!("read wgpu tensor: {error:?}"))
}

impl NativeBiRefNet {
    /// Loads `model.safetensors` from the pinned upstream BiRefNet Lite revision.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let mut raw = Raw::new();
        for (name, view) in tensors.tensors() {
            // Integer entries are batch-norm counters and the fixed window index table.
            if view.dtype() != Dtype::F32 {
                continue;
            }
            let data = view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            raw.insert(name, (view.shape().to_vec(), data));
        }
        drop(bytes);
        for (name, shape) in [
            ("bb.patch_embed.proj.weight", &[EMBED, 3, 4, 4][..]),
            ("bb.layers.2.blocks.5.attn.qkv.weight", &[1152, 384]),
            ("squeeze_module.0.conv_in.weight", &[64, 2880, 3, 3]),
            ("decoder.decoder_block1.conv_in.weight", &[64, 240, 3, 3]),
            ("decoder.conv_out1.0.weight", &[1, 120, 1, 1]),
        ] {
            ensure!(
                raw.get(name).map(|entry| entry.0.as_slice()) == Some(shape),
                "{} is not a BiRefNet Lite (Swin-T) checkpoint: {name} should be {shape:?}",
                path.display()
            );
        }
        fold_batch_norms(&mut raw)?;
        prepare_backbone(&mut raw)?;
        fold_output_head(&mut raw)?;

        let device = WgpuDevice::default();
        // Exclusive pages return freed feature maps instead of keeping multi-GiB slabs,
        // which is what keeps this fp32 model inside a 4 GiB adapter.
        let options = wgpu::RuntimeOptions {
            memory_config: wgpu::MemoryConfiguration::ExclusivePages,
            ..Default::default()
        };
        let setup = wgpu::init_setup::<wgpu::graphics::AutoGraphicsApi>(&device, options);
        let info = setup.adapter.get_info();
        let adapter = format!("{} ({:?}, wgpu fp32)", info.name, info.backend);
        let mut values = HashMap::new();
        for (name, (shape, data)) in raw {
            let len = data.len();
            let tensor = Tensor::<B, 1>::from_data(TensorData::new(data, [len]), &device);
            values.insert(name, (shape, tensor));
            if values.len() % 32 == 0 {
                B::sync(&device);
            }
        }
        B::sync(&device);
        Ok(Self {
            values,
            device,
            adapter,
        })
    }

    fn t<const D: usize>(&self, name: &str) -> Result<Tensor<B, D>> {
        let (shape, tensor) = self
            .values
            .get(name)
            .with_context(|| format!("missing BiRefNet Lite tensor {name}"))?;
        let dims: [usize; D] = shape
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("{name}: unexpected shape {shape:?}"))?;
        Ok(tensor.clone().reshape(dims))
    }

    fn bias(&self, name: &str) -> Option<Tensor<B, 1>> {
        self.values
            .get(&format!("{name}.bias"))
            .map(|entry| entry.1.clone())
    }

    fn input(&self, data: &[f32], dims: [usize; 4]) -> Result<T4> {
        ensure!(
            data.len() == dims.iter().product::<usize>(),
            "input has {} values, expected {dims:?}",
            data.len()
        );
        Ok(T4::from_data(
            TensorData::new(data.to_vec(), dims),
            &self.device,
        ))
    }

    fn conv(&self, x: T4, name: &str, padding: usize, stride: usize) -> Result<T4> {
        let weight = self.t(&format!("{name}.weight"))?;
        let options = ConvOptions::new([stride; 2], [padding; 2], [1, 1], 1);
        Ok(conv2d(x, weight, self.bias(name), options))
    }

    fn linear(&self, x: T3, name: &str) -> Result<T3> {
        let weight: Tensor<B, 2> = self.t(&format!("{name}.weight"))?;
        let weight: T3 = weight.unsqueeze();
        let y = x.matmul(weight);
        Ok(match self.bias(name) {
            Some(bias) => y + bias.unsqueeze(),
            None => y,
        })
    }

    fn layer_norm(&self, x: T3, name: &str) -> Result<T3> {
        let centered = x.clone() - x.mean_dim(2);
        let variance = (centered.clone() * centered.clone()).mean_dim(2);
        let weight: Tensor<B, 1> = self.t(&format!("{name}.weight"))?;
        let bias: Tensor<B, 1> = self.t(&format!("{name}.bias"))?;
        Ok(centered / (variance + 1e-5).sqrt() * weight.unsqueeze() + bias.unsqueeze())
    }

    /// 0 / -100 attention mask for shifted windows, [windows, 1, 49, 49].
    fn shift_mask(&self, h: usize, w: usize) -> T4 {
        let (hp, wp) = (h.div_ceil(WINDOW) * WINDOW, w.div_ceil(WINDOW) * WINDOW);
        let region = |at: usize, len: usize| match at {
            _ if at < len - WINDOW => 0,
            _ if at < len - SHIFT => 1,
            _ => 2,
        };
        let (rows, cols) = (hp / WINDOW, wp / WINDOW);
        let mut mask = vec![0.0f32; rows * cols * TOKENS * TOKENS];
        for (window, chunk) in mask.chunks_exact_mut(TOKENS * TOKENS).enumerate() {
            let (top, left) = (window / cols * WINDOW, window % cols * WINDOW);
            let id = |token: usize| {
                region(top + token / WINDOW, hp) * 3 + region(left + token % WINDOW, wp)
            };
            for i in 0..TOKENS {
                for j in 0..TOKENS {
                    if id(i) != id(j) {
                        chunk[i * TOKENS + j] = -100.0;
                    }
                }
            }
        }
        T4::from_data(
            TensorData::new(mask, [rows * cols, 1, TOKENS, TOKENS]),
            &self.device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn swin_block(
        &self,
        x: T3,
        name: &str,
        h: usize,
        w: usize,
        dim: usize,
        heads: usize,
        mask: Option<&T4>,
    ) -> Result<T3> {
        let (hp, wp) = (h.div_ceil(WINDOW) * WINDOW, w.div_ceil(WINDOW) * WINDOW);
        let (rows, cols) = (hp / WINDOW, wp / WINDOW);
        let windows = rows * cols;
        let head = dim / heads;
        let shortcut = x.clone();
        let mut y: T4 = self
            .layer_norm(x, &format!("{name}.norm1"))?
            .reshape([1, h, w, dim]);
        if wp > w {
            y = Tensor::cat(vec![y, T4::zeros([1, h, wp - w, dim], &self.device)], 2);
        }
        if hp > h {
            y = Tensor::cat(vec![y, T4::zeros([1, hp - h, wp, dim], &self.device)], 1);
        }
        if mask.is_some() {
            y = roll(roll(y, 1, SHIFT), 2, SHIFT);
        }
        let y: T3 = y
            .reshape([rows, WINDOW, cols, WINDOW, dim])
            .permute([0, 2, 1, 3, 4])
            .reshape([1, windows * TOKENS, dim]);
        let qkv = self
            .linear(y, &format!("{name}.attn.qkv"))?
            .reshape([windows, TOKENS, 3, heads, head])
            .permute([2, 0, 3, 1, 4]);
        let part = |index: usize| -> T4 {
            qkv.clone()
                .narrow(0, index, 1)
                .reshape([windows, heads, TOKENS, head])
        };
        let query = part(0) * (head as f32).powf(-0.5);
        let mut attention = query.matmul(part(1).swap_dims(2, 3))
            + self.t::<4>(&format!("{name}.attn.rel_bias"))?;
        if let Some(mask) = mask {
            attention = attention + mask.clone();
        }
        // Burn 0.18's wgpu softmax leaves the tail of a very large tensor non-finite
        // (seen with 1369 windows x 3 heads at 1024x1024), so normalise in window batches.
        let step = (SOFTMAX_ROWS / (heads * TOKENS)).max(1);
        let sm = if windows <= step {
            softmax(attention, 3)
        } else {
            let parts = (0..windows)
                .step_by(step)
                .map(|start| {
                    softmax(
                        attention
                            .clone()
                            .narrow(0, start, step.min(windows - start)),
                        3,
                    )
                })
                .collect();
            Tensor::cat(parts, 0)
        };
        let av = sm.matmul(part(2));
        let y: T3 = av.swap_dims(1, 2).reshape([1, windows * TOKENS, dim]);
        let mut y: T4 = self
            .linear(y, &format!("{name}.attn.proj"))?
            .reshape([rows, cols, WINDOW, WINDOW, dim])
            .permute([0, 2, 1, 3, 4])
            .reshape([1, hp, wp, dim]);
        if mask.is_some() {
            y = roll(roll(y, 1, hp - SHIFT), 2, wp - SHIFT);
        }
        if hp > h || wp > w {
            y = y.slice([0..1, 0..h, 0..w]);
        }
        let x = shortcut + y.reshape([1, h * w, dim]);
        let hidden = self.linear(
            self.layer_norm(x.clone(), &format!("{name}.norm2"))?,
            &format!("{name}.mlp.fc1"),
        )?;
        Ok(x + self.linear(gelu(hidden), &format!("{name}.mlp.fc2"))?)
    }

    /// Swin-T: four stages at strides 4, 8, 16 and 32.
    fn backbone(&self, x: T4) -> Result<Vec<T4>> {
        let x = self.conv(x, "bb.patch_embed.proj", 0, 4)?;
        let [_, _, mut h, mut w] = x.dims();
        let mut x: T3 = x.reshape([1, EMBED, h * w]).swap_dims(1, 2);
        x = self.layer_norm(x, "bb.patch_embed.norm")?;
        let mut stages = Vec::with_capacity(4);
        for stage in 0..4 {
            let dim = EMBED << stage;
            let mask = self.shift_mask(h, w);
            for block in 0..DEPTHS[stage] {
                x = self.swin_block(
                    x,
                    &format!("bb.layers.{stage}.blocks.{block}"),
                    h,
                    w,
                    dim,
                    HEADS[stage],
                    (block % 2 == 1).then_some(&mask),
                )?;
            }
            stages.push(
                self.layer_norm(x.clone(), &format!("bb.norm{stage}"))?
                    .reshape([1, h, w, dim])
                    .permute([0, 3, 1, 2]),
            );
            if stage < 3 {
                ensure!(h % 2 == 0 && w % 2 == 0, "odd Swin stage size {h}x{w}");
                let name = format!("bb.layers.{stage}.downsample");
                // Channel order of the merge is (row 0, col 0), (1, 0), (0, 1), (1, 1).
                let merged: T3 = x
                    .reshape([1, h / 2, 2, w / 2, 2, dim])
                    .permute([0, 1, 3, 4, 2, 5])
                    .reshape([1, h * w / 4, 4 * dim]);
                x = self.linear(
                    self.layer_norm(merged, &format!("{name}.norm"))?,
                    &format!("{name}.reduction"),
                )?;
                (h, w) = (h / 2, w / 2);
            }
            B::sync(&self.device);
        }
        Ok(stages)
    }

    /// Modulated deformable convolution (torchvision semantics) with its folded batch norm.
    fn deformable(&self, x: T4, name: &str) -> Result<T4> {
        let weight: T4 = self.t(&format!("{name}.regular_conv.weight"))?;
        let [out, channels, kernel, _] = weight.dims();
        let [_, _, h, w] = x.dims();
        let padding = kernel / 2;
        let offset = self.conv(x.clone(), &format!("{name}.offset_conv"), padding, 1)?;
        let mask =
            sigmoid(self.conv(x.clone(), &format!("{name}.modulator_conv"), padding, 1)?) * 2.0;
        let options = DeformConvOptions::new([1, 1], [padding; 2], [1, 1], 1, 1);
        // Offsets are shared by all input channels, so the sum over channel slices is exact.
        let step = (COLUMN_BYTES / (kernel * kernel * h * w * 4)).clamp(1, channels);
        let mut sum: Option<T4> = None;
        for start in (0..channels).step_by(step) {
            let end = (start + step).min(channels);
            let part = deform_conv2d(
                x.clone().slice([0..1, start..end]),
                offset.clone(),
                weight.clone().slice([0..out, start..end]),
                Some(mask.clone()),
                None,
                options.clone(),
            );
            sum = Some(match sum {
                Some(sum) => sum + part,
                None => part,
            });
            if step < channels {
                B::sync(&self.device);
            }
        }
        let y = sum.context("deformable convolution without input channels")?;
        Ok(match self.bias(&format!("{name}.regular_conv")) {
            Some(bias) => y + bias.reshape([1, out, 1, 1]),
            None => y,
        })
    }

    /// ASPPDeformable: 1x1, 1x1, 3x3 and 7x7 deformable branches plus global pooling,
    /// projected by a 1x1 convolution that is applied per branch instead of to the cat.
    fn aspp(&self, x: T4, name: &str) -> Result<T4> {
        let project: T4 = self.t(&format!("{name}.conv1.weight"))?;
        let [out, _, _, _] = project.dims();
        let slice = |branch: usize| {
            project
                .clone()
                .slice([0..out, branch * ASPP_WIDTH..(branch + 1) * ASPP_WIDTH])
        };
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let branches = std::iter::once(format!("{name}.aspp1.atrous_conv"))
            .chain((0..3).map(|i| format!("{name}.aspp_deforms.{i}.atrous_conv")));
        let pooled = x.clone().mean_dim(3).mean_dim(2);
        let pooled = relu(self.conv(pooled, &format!("{name}.global_avg_pool.1"), 0, 1)?);
        let bias = self
            .bias(&format!("{name}.conv1"))
            .context("ASPP projection without folded batch norm")?;
        // The pooled branch is constant over the map: it only shifts the projection bias.
        let mut y = conv2d(pooled, slice(4), Some(bias), options.clone());
        for (index, branch) in branches.enumerate() {
            let features = relu(self.deformable(x.clone(), &branch)?);
            y = conv2d(features, slice(index), None, options.clone()) + y;
        }
        Ok(relu(y))
    }

    /// BasicDecBlk: 3x3 conv, ASPPDeformable, 3x3 conv (batch norms folded).
    fn block(&self, x: T4, name: &str) -> Result<T4> {
        let x = relu(self.conv(x, &format!("{name}.conv_in"), 1, 1)?);
        let x = self.aspp(x, &format!("{name}.dec_att"))?;
        let y = self.conv(x, &format!("{name}.conv_out"), 1, 1)?;
        B::sync(&self.device);
        Ok(y)
    }

    /// Image patches folded into channels at the feature resolution, then two 3x3 convs.
    fn image_branch(&self, image: &T4, [h, w]: [usize; 2], name: &str) -> Result<T4> {
        let [_, channels, ih, iw] = image.dims();
        let (gh, gw) = (ih / h, iw / w);
        let patches: T4 = image
            .clone()
            .reshape([1, channels, gh, h, gw, w])
            .permute([0, 1, 2, 4, 3, 5])
            .reshape([1, channels * gh * gw, h, w]);
        let hidden = self.conv(patches, &format!("{name}.conv1"), 1, 1)?;
        self.conv(hidden, &format!("{name}.conv_out"), 1, 1)
    }

    /// Gradient-attention gate applied to decoder levels 4, 3 and 2.
    fn gate(&self, x: T4, level: usize) -> Result<T4> {
        let hidden = relu(self.conv(x.clone(), &format!("decoder.gdt_convs_{level}.0"), 1, 1)?);
        let gate = self.conv(hidden, &format!("decoder.gdt_convs_attn_{level}.0"), 0, 1)?;
        Ok(x * sigmoid(gate))
    }

    fn forward(&self, image: T4, mut trace: Option<&mut Trace>) -> Result<T4> {
        let mut probe = |name: &str, tensor: &T4| -> Result<()> {
            if let Some(trace) = trace.as_deref_mut() {
                trace.push((name.to_owned(), read(tensor.clone())?));
            }
            Ok(())
        };
        let [_, _, h, w] = image.dims();
        // Encoder: Swin-T at full and half resolution, concatenated per stage.
        let full = self.backbone(image.clone())?;
        for (index, stage) in full.iter().enumerate() {
            probe(&format!("bb_x{}", index + 1), stage)?;
        }
        let half = self.backbone(resize(image.clone(), [h / 2, w / 2]))?;
        let mut stages: Vec<T4> = full
            .into_iter()
            .zip(half)
            .map(|(full, half)| {
                let target = size(&full);
                Tensor::cat(vec![full, resize(half, target)], 1)
            })
            .collect();
        let deepest = size(&stages[3]);
        let context = stages[..3]
            .iter()
            .map(|stage| resize(stage.clone(), deepest))
            .chain([stages[3].clone()])
            .collect();
        stages[3] = Tensor::cat(context, 1);
        probe("enc_x1", &stages[0])?;
        probe("enc_x4", &stages[3])?;
        let [x1, x2, x3, x4]: [T4; 4] = stages
            .try_into()
            .map_err(|_| anyhow!("backbone did not return four stages"))?;
        B::sync(&self.device);

        // Decoder.
        let x4 = self.block(x4, "squeeze_module.0")?;
        probe("squeeze", &x4)?;
        let patches = self.image_branch(&image, size(&x4), "decoder.ipt_blk5")?;
        let p4 = self.block(Tensor::cat(vec![x4, patches], 1), "decoder.decoder_block4")?;
        let p4 = self.gate(p4, 4)?;

        let lateral = self.conv(x3, "decoder.lateral_block4.conv", 0, 1)?;
        let p3 = resize(p4, size(&lateral)) + lateral;
        let patches = self.image_branch(&image, size(&p3), "decoder.ipt_blk4")?;
        let p3 = self.block(Tensor::cat(vec![p3, patches], 1), "decoder.decoder_block3")?;
        let p3 = self.gate(p3, 3)?;

        let lateral = self.conv(x2, "decoder.lateral_block3.conv", 0, 1)?;
        let p2 = resize(p3, size(&lateral)) + lateral;
        let patches = self.image_branch(&image, size(&p2), "decoder.ipt_blk3")?;
        let p2 = self.block(Tensor::cat(vec![p2, patches], 1), "decoder.decoder_block2")?;
        let p2 = self.gate(p2, 2)?;

        let lateral = self.conv(x1, "decoder.lateral_block2.conv", 0, 1)?;
        let p1 = resize(p2, size(&lateral)) + lateral;
        let patches = self.image_branch(&image, size(&p1), "decoder.ipt_blk2")?;
        let p1 = self.block(Tensor::cat(vec![p1, patches], 1), "decoder.decoder_block1")?;

        // Output head, see `fold_output_head`.
        let low = self.conv(p1, "decoder.out_low", 0, 1)?;
        let hidden = self.conv(image, "decoder.ipt_blk1.conv1", 1, 1)?;
        let logits = resize(low, [h, w]) + self.conv(hidden, "decoder.out_ipt", 1, 1)?;
        probe("logits", &logits)?;
        Ok(logits)
    }

    fn image(&self, input: &[f32], size: usize) -> Result<T4> {
        ensure!(
            size % 64 == 0 && (64..=1024).contains(&size),
            "BiRefNet Lite input size must be a multiple of 64 in 64..=1024, got {size}"
        );
        self.input(input, [1, 3, size, size])
    }

    /// Mask logits for one normalised RGB image, `[3, size, size]` in and `[size, size]` out.
    pub fn logits(&self, input: &[f32], size: usize) -> Result<Vec<f32>> {
        read(self.forward(self.image(input, size)?, None)?)
    }

    /// Same as [`Self::logits`], also returning intermediate activations for parity tests.
    pub fn trace(&self, input: &[f32], size: usize) -> Result<Trace> {
        let mut trace = Trace::new();
        self.forward(self.image(input, size)?, Some(&mut trace))?;
        Ok(trace)
    }

    /// One deformable convolution with its batch norm, e.g.
    /// `decoder.decoder_block4.dec_att.aspp_deforms.1.atrous_conv`, for parity tests.
    pub fn deformable_conv(&self, name: &str, input: &[f32], dims: [usize; 4]) -> Result<Vec<f32>> {
        read(self.deformable(self.input(input, dims)?, name)?)
    }
}
