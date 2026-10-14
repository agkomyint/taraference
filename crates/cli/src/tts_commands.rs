//! Focused CLI for local text-to-speech. TTS inference stays separate from GGUF chat.

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use std::io::Read;
use std::path::PathBuf;
use taraference_tts::{
    mms::{self, Mms, MMS_MODEL_DIR},
    missing_files,
    normalize::{normalize, Lexicon},
    write_wav_rate, Device, SynthOptions, Tts, DEFAULT_MODEL_DIR,
};

#[derive(Debug, Parser)]
#[command(
    name = "tarafer speech",
    about = "Local Burmese text-to-speech: F5-Myanmar-TTS (best quality) and MMS-TTS (fast). Weights are CC BY-NC 4.0, non-commercial."
)]
struct TtsCli {
    #[command(subcommand)]
    command: TtsCommand,
}

#[derive(Debug, Subcommand)]
enum TtsCommand {
    /// Speak Burmese text and save a 24 kHz mono WAV.
    Say {
        /// Text to speak. Use `-` to read stdin, or pass --file.
        text: Option<String>,
        /// Read the text from a UTF-8 file.
        #[arg(long, value_name = "PATH")]
        file: Option<PathBuf>,
        /// Output WAV path.
        #[arg(short, long, default_value = "speech.wav")]
        output: PathBuf,
        /// Engine: f5 (F5-Myanmar-TTS, best quality) or mms (Meta MMS-TTS VITS, fast).
        #[arg(long, default_value = "f5", value_parser = ["f5", "mms"])]
        engine: String,
        /// Model directory (default models/tts/f5-myanmar or models/tts/mms-mya).
        #[arg(long)]
        model_dir: Option<PathBuf>,
        /// Inference device: auto (CUDA, DirectML, then CPU), cuda, directml, or cpu.
        #[arg(long, default_value = "auto", value_parser = parse_device)]
        device: Device,
        /// Flow-matching steps (8 is fast, 32 is the quality default).
        #[arg(long, default_value_t = 32)]
        steps: usize,
        /// Classifier-free guidance strength.
        #[arg(long, default_value_t = 2.0)]
        cfg: f32,
        /// Speaking speed multiplier.
        #[arg(long, default_value_t = 1.0)]
        speed: f32,
        /// Noise seed for reproducible output.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Reference voice WAV (3-10 s) to clone instead of the bundled voice.
        #[arg(long, value_name = "WAV")]
        ref_audio: Option<PathBuf>,
        /// Exact transcript of --ref-audio.
        #[arg(long)]
        ref_text: Option<String>,
        /// Word list (`English=Burmese` per line) for names and loanwords the models cannot read.
        #[arg(long, value_name = "FILE")]
        lexicon: Option<PathBuf>,
        /// Skip number reading and cleanup; send the text to the model as given.
        #[arg(long)]
        no_normalize: bool,
    },
    /// Show which engines are installed and ready.
    Status {
        /// Check only this F5 directory (default: all engines at their default paths).
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
}

fn parse_device(value: &str) -> Result<Device, String> {
    value.parse::<Device>().map_err(|error| error.to_string())
}

pub fn run(args: Vec<String>) -> Result<()> {
    let args = std::iter::once("tarafer speech".to_string()).chain(args);
    match TtsCli::parse_from(args).command {
        TtsCommand::Status { model_dir } => {
            let f5_dir = model_dir.unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
            let engines = [
                ("f5 ", "F5-Myanmar-TTS v2 (best quality, CC BY-NC)", f5_dir.clone(), missing_files(&f5_dir),
                 format!("python tools/tts/export_f5_myanmar.py --out {}", f5_dir.display())),
                ("mms", "MMS-TTS Burmese (fast, CPU ok, CC BY-NC)", PathBuf::from(MMS_MODEL_DIR),
                 mms::missing_files(std::path::Path::new(MMS_MODEL_DIR)),
                 format!("put model.onnx + tokens.txt (willwade/mms-tts-multilingual-models-onnx, mya/) in {MMS_MODEL_DIR}")),
            ];
            let mut ready = 0;
            for (id, name, dir, missing, hint) in &engines {
                if missing.is_empty() {
                    ready += 1;
                    eprintln!("ready   | {id} | {name} | {}", dir.display());
                } else {
                    eprintln!("missing | {id} | {name}");
                    for path in missing {
                        eprintln!("          - {}", path.display());
                    }
                    eprintln!("          setup: {hint}");
                }
            }
            if ready == 0 {
                bail!("no TTS engine installed");
            }
            eprintln!("next    | tarafer speech say \"မင်္ဂလာပါ\" -o hello.wav [--engine f5|mms]");
        }
        TtsCommand::Say {
            text,
            file,
            output,
            engine,
            model_dir,
            device,
            steps,
            cfg,
            speed,
            seed,
            ref_audio,
            ref_text,
            lexicon,
            no_normalize,
        } => {
            let text = match (text.as_deref(), file) {
                (_, Some(path)) => std::fs::read_to_string(&path)
                    .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?,
                (Some("-"), None) | (None, None) => {
                    let mut buffer = String::new();
                    std::io::stdin().read_to_string(&mut buffer)?;
                    buffer
                }
                (Some(text), None) => text.to_string(),
            };
            if text.trim().is_empty() {
                bail!("no text provided");
            }
            if engine == "f5" && (steps == 0 || steps > 128) {
                bail!("--steps must be between 1 and 128");
            }
            if output.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase)
                != Some("wav".into())
            {
                bail!("output must be a .wav file");
            }
            let text = if no_normalize {
                text
            } else {
                let lex = match &lexicon {
                    Some(path) => Lexicon::from_file(path)?,
                    None => Lexicon::default(),
                };
                let n = normalize(&text, &lex);
                if !n.unknown_words.is_empty() {
                    eprintln!(
                        "warning| not speakable, skipped: {} (add `Word=Burmese` lines to a --lexicon file)",
                        n.unknown_words.join(", ")
                    );
                }
                if n.dropped_symbols > 0 {
                    eprintln!("note   | removed {} emoji/symbol character(s)", n.dropped_symbols);
                }
                if n.text.trim().is_empty() {
                    bail!("nothing speakable left after cleanup (is the text Burmese?)");
                }
                n.text
            };
            let options = SynthOptions { steps, cfg, speed, seed, ref_audio, ref_text };
            let result = if engine == "mms" {
                let dir = model_dir.unwrap_or_else(|| PathBuf::from(MMS_MODEL_DIR));
                let mut tts = Mms::load(&dir, device)?;
                eprintln!("model  | MMS-TTS Burmese (VITS) on {} ({:.0} ms load)", tts.device, tts.load_ms);
                tts.synthesize(&text, &options)?
            } else {
                let dir = model_dir.unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
                let mut tts = Tts::load(&dir, device)?;
                eprintln!("model  | F5-Myanmar-TTS on {} ({:.0} ms load)", tts.device, tts.load_ms);
                tts.synthesize(&text, &options)?
            };
            write_wav_rate(&output, &result.samples, result.sample_rate)?;
            eprintln!(
                "done   | {:.1} s audio | {} chunk(s) | {:.1} s synth | RTF {:.2}{}",
                result.seconds(),
                result.chunks,
                result.synth_ms / 1000.0,
                result.synth_ms / 1000.0 / result.seconds().max(1e-6),
                if engine == "f5" { format!(" | {steps} steps") } else { String::new() }
            );
            eprintln!("output | {}", output.display());
        }
    }
    Ok(())
}
