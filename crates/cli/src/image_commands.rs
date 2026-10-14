//! Focused CLI for local image tools. Image inference stays separate from GGUF chat.

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use taraference_background_removal::{
    default_model_path, download_model, remove_background, Device, MODEL_FILE,
};

#[derive(Debug, Parser)]
#[command(name = "tarafer image", about = "Local image tools (background removal)")]
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
    /// Manage the optional BiRefNet Lite model.
    Model {
        #[command(subcommand)]
        command: ImageModelCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ImageModelCommand {
    /// Download the 115 MB FP16 MIT-licensed model from Hugging Face.
    Download {
        /// Model file destination. Defaults to models/background-removal/birefnet-lite.onnx.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Directory used for default model storage.
        #[arg(long, default_value = "models")]
        models_dir: PathBuf,
        /// Replace and verify an existing model file.
        #[arg(long)]
        force: bool,
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
            eprintln!("input  | {}", input.display());
            eprintln!("model  | {} ({MODEL_FILE})", model.display());
            let result = remove_background(&input, &output, &model, device)?;
            eprintln!(
                "done   | {} × {} | {} | {:.0} ms setup | {:.0} ms inference",
                result.width, result.height, result.device, result.session_ms, result.inference_ms
            );
            eprintln!("output | {}", result.output.display());
        }
        ImageCommand::Model {
            command: ImageModelCommand::Download {
                model,
                models_dir,
                force,
            },
        } => {
            let model = model.unwrap_or_else(|| default_model_path(&models_dir));
            let installed = download_model(&model, force)?;
            eprintln!("ready  | {}", installed.display());
            eprintln!("next   | tarafer image remove-bg input.jpg");
        }
    }
    Ok(())
}
