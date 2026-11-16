//! Text-to-image with SDXS-512 (one-step distilled Stable Diffusion 1.5, DreamShaper 8) on ONNX Runtime.
//! Pipeline: CLIP tokenizer -> text encoder -> one UNet pass from Gaussian noise -> VAE decoder -> PNG.

mod clip;

use anyhow::{anyhow, bail, Context, Result};
use half::f16;
use ort::{ep, session::Session, value::Tensor};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub use clip::ClipTokenizer;

pub const MODEL_DIR: &str = "models/image/sdxs-512";
pub const TEXT_ENCODER: &str = "text_encoder_fp16.onnx";
pub const UNET: &str = "unet_fp16.onnx";
pub const DECODER_FAST: &str = "taesd_decoder_fp16.onnx";
pub const DECODER_FULL: &str = "vae_large_decoder_fp16.onnx";
const LATENT: usize = 4 * 64 * 64;
// One DEIS step from t=999 (manifest.json): latent = A * noise + B * eps.
const SAMPLE_COEF: f32 = 14.642_591;
const OUTPUT_COEF: f32 = -14.579_279;
const TIMESTEP: f32 = 999.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Auto,
    Cuda,
    DirectMl,
    Cpu,
}

impl std::str::FromStr for Device {
    type Err = anyhow::Error;
    fn from_str(v: &str) -> Result<Self> {
        match v.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cuda" => Ok(Self::Cuda),
            "directml" | "dml" => Ok(Self::DirectMl),
            "cpu" => Ok(Self::Cpu),
            _ => bail!("unknown device '{v}'; use auto, cuda, directml, or cpu"),
        }
    }
}

pub struct Options {
    pub model_dir: PathBuf,
    pub device: Device,
    /// Use the small TAESD decoder (fast, slightly softer) instead of the full VAE decoder.
    pub fast_decoder: bool,
    pub seed: u64,
}

pub struct Generated {
    pub device: &'static str,
    pub load_ms: f64,
    pub text_ms: f64,
    pub unet_ms: f64,
    pub decode_ms: f64,
    pub seed: u64,
}

fn e<T: ToString>(x: T) -> anyhow::Error {
    anyhow!(x.to_string())
}

fn load(path: &Path, device: Device) -> Result<(Session, &'static str)> {
    if matches!(device, Device::Auto | Device::Cuda) {
        let r = (|| -> Result<Session> {
            let cuda = ep::CUDA::default()
                .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
                .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
                .build()
                .error_on_failure();
            Session::builder()
                .map_err(e)?
                .with_execution_providers([cuda])
                .map_err(e)?
                .commit_from_file(path)
                .map_err(e)
        })();
        match r {
            Ok(s) => return Ok((s, "CUDA")),
            Err(err) if device == Device::Cuda => {
                return Err(err).context("ONNX Runtime CUDA failed to start")
            }
            Err(err) => eprintln!("CUDA unavailable ({err:#}); trying next backend"),
        }
    }
    #[cfg(windows)]
    if matches!(device, Device::Auto | Device::DirectMl) {
        let r = (|| -> Result<Session> {
            let dml = ep::DirectML::default().build().error_on_failure();
            Session::builder()
                .map_err(e)?
                .with_execution_providers([dml])
                .map_err(e)?
                .commit_from_file(path)
                .map_err(e)
        })();
        match r {
            Ok(s) => return Ok((s, "DirectML")),
            Err(err) if device == Device::DirectMl => {
                return Err(err).context("ONNX Runtime DirectML failed to start")
            }
            Err(err) => eprintln!("DirectML unavailable ({err:#}); falling back to CPU"),
        }
    }
    #[cfg(not(windows))]
    if device == Device::DirectMl {
        bail!("DirectML is available only on Windows");
    }
    Ok((Session::builder().map_err(e)?.commit_from_file(path).map_err(e)?, "CPU"))
}

/// SplitMix64 + Box-Muller: deterministic standard-normal noise from a seed.
fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u1, u2) = (next().max(1e-12), next());
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * u2).cos()) as f32);
        out.push((r * (std::f64::consts::TAU * u2).sin()) as f32);
    }
    out.truncate(n);
    out
}

pub fn missing_files(dir: &Path, fast_decoder: bool) -> Vec<String> {
    let dec = if fast_decoder { DECODER_FAST } else { DECODER_FULL };
    [TEXT_ENCODER, UNET, dec, "tokenizer.json"]
        .iter()
        .filter(|f| !dir.join(f).is_file())
        .map(|f| f.to_string())
        .collect()
}

pub fn generate(prompt: &str, output: &Path, opts: &Options) -> Result<Generated> {
    let missing = missing_files(&opts.model_dir, opts.fast_decoder);
    if !missing.is_empty() {
        bail!(
            "SDXS-512 files missing in {}: {}. Run: tarafer models pull sdxs-512",
            opts.model_dir.display(),
            missing.join(", ")
        );
    }
    let t = Instant::now();
    let tok = ClipTokenizer::load(&opts.model_dir.join("tokenizer.json"))?;
    let (mut text_enc, device) = load(&opts.model_dir.join(TEXT_ENCODER), opts.device)?;
    let (mut unet, _) = load(&opts.model_dir.join(UNET), opts.device)?;
    let dec_file = if opts.fast_decoder { DECODER_FAST } else { DECODER_FULL };
    let (mut decoder, _) = load(&opts.model_dir.join(dec_file), opts.device)?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;

    // Text encoder: int32 [1,77] -> f32 [1,77,768].
    let t = Instant::now();
    let ids = tok.encode(prompt);
    let out = text_enc
        .run(ort::inputs!["input_ids" => Tensor::from_array(([1usize, 77], ids)).map_err(e)?])
        .map_err(e)?;
    let (_, hidden) = out[0].try_extract_tensor::<f32>().map_err(e)?;
    let hidden: Vec<f16> = hidden.iter().map(|&x| f16::from_f32(x)).collect();
    drop(out);
    let text_ms = t.elapsed().as_secs_f64() * 1e3;

    // One UNet pass + the scheduler's single step.
    let t = Instant::now();
    let noise = gaussian(opts.seed, LATENT);
    let sample: Vec<f16> = noise.iter().map(|&x| f16::from_f32(x)).collect();
    let out = unet
        .run(ort::inputs![
            "sample" => Tensor::from_array(([1usize, 4, 64, 64], sample)).map_err(e)?,
            "timestep" => Tensor::from_array(([1usize], vec![TIMESTEP])).map_err(e)?,
            "encoder_hidden_states" => Tensor::from_array(([1usize, 77, 768], hidden)).map_err(e)?
        ])
        .map_err(e)?;
    let (_, eps) = out[0].try_extract_tensor::<f16>().map_err(e)?;
    if eps.len() != LATENT {
        bail!("unexpected UNet output length {}", eps.len());
    }
    let latent: Vec<f16> = noise
        .iter()
        .zip(eps)
        .map(|(&n, &p)| f16::from_f32(SAMPLE_COEF * n + OUTPUT_COEF * p.to_f32()))
        .collect();
    drop(out);
    let unet_ms = t.elapsed().as_secs_f64() * 1e3;

    // Decoder: latent -> RGB in [0,1].
    let t = Instant::now();
    let out = decoder
        .run(ort::inputs!["latent" => Tensor::from_array(([1usize, 4, 64, 64], latent)).map_err(e)?])
        .map_err(e)?;
    let (shape, img) = out[0].try_extract_tensor::<f16>().map_err(e)?;
    if shape.len() != 4 || shape[1] != 3 {
        bail!("unexpected decoder output shape {shape:?}");
    }
    let (h, w) = (shape[2] as usize, shape[3] as usize);
    let mut rgb = vec![0u8; h * w * 3];
    for c in 0..3 {
        for i in 0..h * w {
            rgb[i * 3 + c] = (img[c * h * w + i].to_f32().clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    drop(out);
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow!("bad image buffer"))?
        .save(output)
        .with_context(|| format!("write {}", output.display()))?;
    let decode_ms = t.elapsed().as_secs_f64() * 1e3;

    Ok(Generated { device, load_ms, text_ms, unet_ms, decode_ms, seed: opts.seed })
}
