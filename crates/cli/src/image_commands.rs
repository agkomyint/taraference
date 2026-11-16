//! Focused CLI for local image tools. Image inference stays separate from GGUF chat.

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use taraference_background_removal::{default_model_path, remove_background, Device, MODEL_FILE};

#[derive(Debug, Parser)]
#[command(name = "tarafer image", about = "Local image tools (text-to-image, background removal)")]
struct ImageCli {
    #[command(subcommand)]
    command: ImageCommand,
}

#[derive(Debug, Subcommand)]
enum ImageCommand {
    /// Remove an image background and save a transparent PNG.
    #[command(alias = "remove-background")]
    RemoveBg {
        /// Input photo (JPEG, PNG, WebP, BMP, or TIFF).
        input: PathBuf,
        /// Output PNG path. Defaults to <input>-no-bg.png.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// BiRefNet ONNX model path. Defaults to models/background-removal/birefnet-lite.onnx.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Inference device: auto (CUDA, DirectML, then CPU), cuda, directml, or cpu.
        #[arg(long, default_value = "auto", value_parser = parse_device)]
        device: Device,
    },
    /// Generate an image from a text prompt (SDXS-512, one step).
    #[command(alias = "generate")]
    Gen {
        /// Text prompt (English).
        prompt: String,
        /// Output PNG path.
        #[arg(short, long, default_value = "out.png")]
        output: PathBuf,
        /// Random seed (default: time-based).
        #[arg(long)]
        seed: Option<u64>,
        /// SDXS-512 ONNX model directory.
        #[arg(long, default_value = taraference_imagegen::MODEL_DIR)]
        model_dir: PathBuf,
        /// Inference device: auto (CUDA, DirectML, then CPU), cuda, directml, or cpu.
        #[arg(long, default_value = "auto")]
        device: String,
        /// Use the tiny TAESD decoder: faster, slightly softer picture.
        #[arg(long)]
        fast: bool,
    },
}

fn parse_device(value: &str) -> Result<Device, String> {
    value.parse::<Device>().map_err(|error| error.to_string())
}

fn default_output(input: &Path) -> Result<PathBuf> {
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow::anyhow!("input path has no valid filename"))?;
    Ok(input.with_file_name(format!("{stem}-no-bg.png")))
}

pub fn run(args: Vec<String>) -> Result<()> {
    let args = std::iter::once("tarafer image".to_string()).chain(args);
    let cli = ImageCli::parse_from(args);

    match cli.command {
        ImageCommand::Gen { prompt, output, seed, model_dir, device, fast } => {
            let seed = seed.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0)
            });
            let opts = taraference_imagegen::Options {
                model_dir,
                device: device.parse()?,
                fast_decoder: fast,
                seed,
            };
            eprintln!("prompt | {prompt}");
            let r = taraference_imagegen::generate(&prompt, &output, &opts)?;
            eprintln!(
                "done   | SDXS-512 1-step | {} | seed {} | load {:.0} ms | text {:.0} ms | unet {:.0} ms | decode {:.0} ms",
                r.device, r.seed, r.load_ms, r.text_ms, r.unet_ms, r.decode_ms
            );
            eprintln!("output | {}", output.display());
        }
        ImageCommand::RemoveBg {
            input,
            output,
            model,
            device,
        } => {
            if !input.is_file() {
                bail!("input image not found: {}", input.display());
            }
            let output = match output {
                Some(path) => path,
                None => default_output(&input)?,
            };
            let model = model.unwrap_or_else(|| default_model_path(Path::new("models")));
            if !model.is_file() {
                bail!("model not installed: {}. Run: tarafer models pull birefnet-lite", model.display());
            }
            eprintln!("input  | {}", input.display());
            eprintln!("model  | {} ({MODEL_FILE})", model.display());
            let result = remove_background(&input, &output, &model, device)?;
            eprintln!(
                "done   | {} × {} | {} | {:.0} ms setup | {:.0} ms inference",
                result.width, result.height, result.device, result.session_ms, result.inference_ms
            );
            eprintln!("output | {}", result.output.display());
        }
    }
    Ok(())
}
