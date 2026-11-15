//! Standalone native speech CLI: no GGUF/CUDA engine or ONNX Runtime required.
use anyhow::{bail, Result};
use clap::Parser;
use std::{io::Read, path::PathBuf};
use taraference_tts::{
    normalize::{normalize, Lexicon},
    write_wav_rate, Device, SynthOptions, Tts, DEFAULT_MODEL_DIR,
};

#[derive(Parser)]
#[command(about = "Native F5-Myanmar-TTS v2 on wgpu (weights: CC BY-NC 4.0)")]
struct Args {
    /// Burmese text, or - for stdin.
    text: String,
    #[arg(short, long, default_value = "speech.wav")]
    output: PathBuf,
    #[arg(long, default_value = DEFAULT_MODEL_DIR)]
    model_dir: PathBuf,
    #[arg(long, default_value_t = 16)]
    steps: usize,
    #[arg(long, default_value_t = 2.0)]
    cfg: f32,
    #[arg(long, default_value_t = 1.0)]
    speed: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    ref_audio: Option<PathBuf>,
    #[arg(long)]
    ref_text: Option<String>,
    #[arg(long)]
    lexicon: Option<PathBuf>,
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.steps == 0 || a.steps > 128 || !a.cfg.is_finite() || !a.speed.is_finite() || a.speed <= 0.0
    {
        bail!("require 1..=128 steps, finite guidance and a positive finite speed");
    }
    if a.output
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
        != Some("wav")
    {
        bail!("output must be a .wav file");
    }
    let mut text = a.text;
    if text == "-" {
        text.clear();
        std::io::stdin().read_to_string(&mut text)?;
    }
    let lexicon = a
        .lexicon
        .map(|p| Lexicon::from_file(&p))
        .transpose()?
        .unwrap_or_default();
    let normalized = normalize(&text, &lexicon);
    if normalized.text.trim().is_empty() {
        bail!("no speakable Burmese text");
    }
    if !normalized.unknown_words.is_empty() {
        eprintln!("skipped| {}", normalized.unknown_words.join(", "));
    }
    let opts = SynthOptions {
        steps: a.steps,
        cfg: a.cfg,
        speed: a.speed,
        seed: a.seed,
        ref_audio: a.ref_audio,
        ref_text: a.ref_text,
    };
    let mut model = Tts::load(&a.model_dir, Device::Wgpu)?;
    let result = model.synthesize(&normalized.text, &opts)?;
    write_wav_rate(&a.output, &result.samples, result.sample_rate)?;
    eprintln!("F5-Myanmar-TTS v2 | wgpu fp32 | {} steps | {:.3} s audio | {:.3} s synthesis | RTF {:.3} | {:.3} s load | {}",
        a.steps, result.seconds(), result.synth_ms / 1000., result.synth_ms / 1000. / result.seconds().max(1e-6), result.load_ms / 1000., a.output.display());
    Ok(())
}
