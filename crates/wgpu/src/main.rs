use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use taraference_core::sampler::SamplingOptions;

mod benchmark;

#[derive(Parser)]
#[command(
    name = "tarafer-wgpu",
    about = "Native packed-GGUF Qwen2/Qwen2.5 on wgpu"
)]
struct App {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Text {
        #[command(subcommand)]
        command: Text,
    },
    /// Fixed-length warm benchmark, plus teacher-forced logits for backend parity.
    Bench {
        #[command(flatten)]
        model: Model,
        #[arg(
            long,
            default_value = "Explain why the sky is blue in a short paragraph."
        )]
        prompt: String,
        #[arg(short = 'n', long, default_value_t = 64)]
        max_new: usize,
        #[arg(long, default_value_t = 3)]
        runs: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value = "wgpu", value_parser = ["wgpu", "cuda"])]
        backend: String,
        #[arg(long)]
        report: PathBuf,
    },
}
#[derive(Args)]
struct Model {
    #[arg(short, long, default_value = "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf")]
    model: PathBuf,
    #[arg(long, default_value_t = 512)]
    ctx: usize,
}
#[derive(Subcommand)]
enum Text {
    Gen {
        prompt: String,
        #[command(flatten)]
        model: Model,
        #[arg(short = 'n', long, default_value_t = 128)]
        max_new: usize,
        #[arg(long)]
        raw: bool,
        #[arg(long)]
        json: bool,
    },
}
fn run() -> Result<()> {
    match App::parse().command {
        Command::Text {
            command:
                Text::Gen {
                    prompt,
                    model,
                    max_new,
                    raw,
                    json,
                },
        } => {
            let result = taraference_wgpu::runner::generate_file(
                &model.model,
                &prompt,
                raw,
                max_new,
                model.ctx,
                SamplingOptions::default(),
                42,
                !json,
            )?;
            if json {
                println!("{}", serde_json::to_string(&result)?);
            } else {
                eprintln!(
                    "{} | {} | {:.2} tok/s decode | {:.2} tok/s prefill | {:.0} ms load",
                    result.model,
                    result.adapter,
                    result.decode_tps,
                    result.prefill_tps,
                    result.load_ms
                );
            }
            Ok(())
        }
        Command::Bench {
            model,
            prompt,
            max_new,
            runs,
            warmup,
            backend,
            report,
        } => benchmark::run(
            &model.model,
            model.ctx,
            &prompt,
            max_new,
            runs,
            warmup,
            &backend,
            &report,
        ),
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
