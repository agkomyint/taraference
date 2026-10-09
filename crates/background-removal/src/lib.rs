//! Local BiRefNet Lite inference and transparent PNG output.

use anyhow::{anyhow, bail, Context, Result};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgba};
use ort::{ep, session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const MODEL_FILE: &str = "birefnet-lite.onnx";
pub const MODEL_REPO: &str = "onnx-community/BiRefNet_lite-ONNX";
pub const MODEL_COMMIT: &str = "f82954f";
pub const MODEL_SHA256: &str = "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333";
pub const MODEL_BYTES: u64 = 224_000_000;
const MODEL_SIZE: u32 = 1024;
const MAX_IMAGE_PIXELS: u64 = 100_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Auto,
    Cuda,
    Cpu,
}

impl std::str::FromStr for Device {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cuda" | "gpu" => Ok(Self::Cuda),
            "cpu" => Ok(Self::Cpu),
            _ => bail!("unknown image device {value:?}; choose auto, cuda, or cpu"),
        }
    }
}

#[derive(Debug)]
pub struct RemovalResult {
    pub output: PathBuf,
    pub inference_ms: f64,
    pub device: &'static str,
    pub width: u32,
    pub height: u32,
}

pub fn default_model_path(models_dir: &Path) -> PathBuf {
    models_dir.join("background-removal").join(MODEL_FILE)
}

pub fn download_model(model_path: &Path, force: bool) -> Result<PathBuf> {
    if model_path.is_file() && !force {
        verify_model(model_path)?;
        eprintln!("BiRefNet Lite already downloaded: {}", model_path.display());
        return Ok(model_path.to_path_buf());
    }

    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    let url = format!(
        "https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_COMMIT}/onnx/model.onnx"
    );
    let partial = model_path.with_extension("onnx.partial");
    if partial.exists() {
        fs::remove_file(&partial).with_context(|| format!("remove {}", partial.display()))?;
    }

    eprintln!("Downloading BiRefNet Lite (~224 MB; MIT license)\n  from {url}\n  to {}", model_path.display());
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(3600))
        .build();
    let mut request = agent.get(&url).set("User-Agent", "taraference/0.7.0");
    if let Ok(token) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")) {
        if !token.is_empty() {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
    }
    let response = request.call().with_context(|| format!("download model from {url}"))?;
    if !(200..300).contains(&response.status()) {
        bail!("model download failed with HTTP {}", response.status());
    }

    let total = response
        .header("Content-Length")
        .and_then(|value| value.parse::<u64>().ok());
    let mut reader = response.into_reader();
    let mut file = File::create(&partial).with_context(|| format!("create {}", partial.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 256 * 1024];
    let mut downloaded = 0u64;
    let started = Instant::now();
    let mut last_update = Instant::now();

    loop {
        let count = reader.read(&mut buffer).context("read model download")?;
        if count == 0 {
            break;
        }
        file.write_all(&buffer[..count]).context("write model download")?;
        hasher.update(&buffer[..count]);
        downloaded += count as u64;
        if last_update.elapsed() > Duration::from_millis(500) {
            if let Some(bytes) = total {
                eprint!("\r  {:5.1}%  {:.1}/{:.1} MiB", downloaded as f64 * 100.0 / bytes as f64, downloaded as f64 / 1_048_576.0, bytes as f64 / 1_048_576.0);
            } else {
                eprint!("\r  {:.1} MiB", downloaded as f64 / 1_048_576.0);
            }
            let _ = std::io::stderr().flush();
            last_update = Instant::now();
        }
    }
    file.flush().context("flush model download")?;
    drop(file);

    let actual_hash = format!("{:x}", hasher.finalize());
    if actual_hash != MODEL_SHA256 {
        let _ = fs::remove_file(&partial);
        bail!("downloaded model SHA-256 mismatch; expected {MODEL_SHA256}, got {actual_hash}");
    }
    if downloaded < MODEL_BYTES - 1_000_000 {
        let _ = fs::remove_file(&partial);
        bail!("downloaded model is unexpectedly small ({downloaded} bytes)");
    }
    if model_path.exists() {
        fs::remove_file(model_path).with_context(|| format!("replace {}", model_path.display()))?;
    }
    fs::rename(&partial, model_path)
        .with_context(|| format!("install model at {}", model_path.display()))?;
    eprintln!("\r  downloaded {:.1} MiB in {:.1}s; SHA-256 verified", downloaded as f64 / 1_048_576.0, started.elapsed().as_secs_f64());
    Ok(model_path.to_path_buf())
}

fn verify_model(model_path: &Path) -> Result<()> {
    let mut file = File::open(model_path).with_context(|| format!("open {}", model_path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).context("read model file")?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != MODEL_SHA256 {
        bail!("model checksum does not match the pinned BiRefNet Lite file; re-download with `tarafer image model download --force`");
    }
    Ok(())
}

fn load_session(model_path: &Path, device: Device) -> Result<(Session, &'static str)> {
    if device != Device::Cpu {
        let cuda_result = (|| -> Result<Session> {
            let cuda = ep::CUDA::default()
                .with_memory_limit(2 * 1024 * 1024 * 1024)
                .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
                .build()
                .error_on_failure();
            Ok(Session::builder()?
                .with_execution_providers([cuda])?
                .commit_from_file(model_path)?)
        })();
        match cuda_result {
            Ok(session) => return Ok((session, "CUDA")),
            Err(error) if device == Device::Cuda => {
                return Err(error).context("could not start BiRefNet Lite with ONNX Runtime CUDA; check the NVIDIA driver and CUDA 13 runtime libraries");
            }
            Err(error) => eprintln!("CUDA image inference unavailable ({error:#}); falling back to CPU"),
        }
    }

    let session = Session::builder()?
        .commit_from_file(model_path)
        .context("start BiRefNet Lite with ONNX Runtime CPU")?;
    Ok((session, "CPU"))
}

pub fn remove_background(
    input_path: &Path,
    output_path: &Path,
    model_path: &Path,
    device: Device,
) -> Result<RemovalResult> {
    if !model_path.is_file() {
        bail!(
            "BiRefNet Lite is not installed at {}. Run `tarafer image model download` first",
            model_path.display()
        );
    }
    verify_model(model_path)?;
    if output_path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()) != Some("png".into()) {
        bail!("output must be a PNG file so transparency is preserved");
    }
    if input_path == output_path {
        bail!("input and output paths must be different");
    }

    let image = image::open(input_path).with_context(|| format!("open image {}", input_path.display()))?;
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_IMAGE_PIXELS {
        bail!("image dimensions {width}×{height} exceed the supported 100-megapixel limit");
    }

    let (mut session, selected_device) = load_session(model_path, device)?;
    let input_name = session
        .inputs()
        .first()
        .ok_or_else(|| anyhow!("BiRefNet ONNX model has no inputs"))?
        .name
        .to_owned();

    let rgb = image.to_rgb8();
    let resized = image::imageops::resize(&rgb, MODEL_SIZE, MODEL_SIZE, FilterType::Triangle);
    let means = [0.485f32, 0.456, 0.406];
    let stds = [0.229f32, 0.224, 0.225];
    let plane = (MODEL_SIZE * MODEL_SIZE) as usize;
    let mut input = vec![0f32; 3 * plane];
    for (index, pixel) in resized.pixels().enumerate() {
        for channel in 0..3 {
            let value = f32::from(pixel[channel]) / 255.0;
            input[channel * plane + index] = (value - means[channel]) / stds[channel];
        }
    }

    let tensor = Tensor::from_array(([1usize, 3, MODEL_SIZE as usize, MODEL_SIZE as usize], input))?;
    let started = Instant::now();
    let outputs = session.run(ort::inputs![input_name.as_str() => tensor])?;
    let inference_ms = started.elapsed().as_secs_f64() * 1000.0;
    let output = outputs
        .get(0)
        .ok_or_else(|| anyhow!("BiRefNet ONNX model returned no outputs"))?;
    let (shape, logits) = output.try_extract_tensor::<f32>()?;
    if logits.len() != plane || shape.len() < 2 {
        bail!("unexpected BiRefNet Lite output shape {shape:?}; expected one 1024×1024 mask");
    }

    let mask_small: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_fn(MODEL_SIZE, MODEL_SIZE, |x, y| {
        let index = y as usize * MODEL_SIZE as usize + x as usize;
        let value = 1.0 / (1.0 + (-logits[index]).exp());
        Luma([(value.clamp(0.0, 1.0) * 255.0).round() as u8])
    });
    let mask = image::imageops::resize(&mask_small, width, height, FilterType::CatmullRom);
    let original = image.to_rgba8();
    let mut output_image: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    for (x, y, pixel) in output_image.enumerate_pixels_mut() {
        let source = original.get_pixel(x, y);
        *pixel = Rgba([source[0], source[1], source[2], mask.get_pixel(x, y)[0]]);
    }

    if let Some(parent) = output_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    DynamicImage::ImageRgba8(output_image)
        .save_with_format(output_path, image::ImageFormat::Png)
        .with_context(|| format!("write transparent PNG {}", output_path.display()))?;

    Ok(RemovalResult {
        output: output_path.to_path_buf(),
        inference_ms,
        device: selected_device,
        width,
        height,
    })
}
