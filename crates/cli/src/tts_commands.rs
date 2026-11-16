//! Focused CLI for local speech: text-to-speech, speech-to-text and the speech benchmark.
//! Speech inference stays separate from GGUF chat.

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use std::io::Read;
use std::path::PathBuf;
use taraference_tts::{
    asr::{self, Asr, ASR_MODEL_DIR, ASR_MODEL_NAME, ASR_SAMPLE_RATE},
    mms::{self, Mms, MMS_MODEL_DIR},
    missing_files,
    normalize::{normalize, Lexicon},
    write_wav_rate, Device, SynthOptions, Tts, DEFAULT_MODEL_DIR,
};

#[derive(Debug, Parser)]
#[command(
    name = "tarafer speech",
    about = "Local Burmese speech. Text-to-speech: F5-Myanmar-TTS (best quality) and MMS-TTS (fast), weights CC BY-NC 4.0, non-commercial. Speech-to-text: freococo/myanmar_asr, Apache-2.0."
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
        /// Inference device: auto (ORT CUDA/DirectML/CPU), cuda, directml, cpu, or native wgpu (F5).
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
    /// Transcribe Burmese speech to text (stdout). WAV is read directly; other formats need ffmpeg.
    Transcribe {
        /// Audio file, any sample rate, mono or stereo.
        audio: PathBuf,
        /// Speech-to-text model.
        #[arg(short, long, default_value = "myanmar-asr", value_parser = ["myanmar-asr"])]
        model: String,
        /// Spoken language; only Burmese (my) has a model so far.
        #[arg(long, default_value = "my", value_parser = ["my"])]
        lang: String,
        /// Model directory (default models/stt/myanmar-asr).
        #[arg(long)]
        model_dir: Option<PathBuf>,
        /// Inference device: auto (ORT CUDA/DirectML/CPU), cuda, directml or cpu.
        #[arg(long, default_value = "auto", value_parser = parse_device)]
        device: Device,
        /// Print syllables separated by spaces instead of running text.
        #[arg(long)]
        syllables: bool,
        /// Also write the transcript to this UTF-8 file.
        #[arg(short, long, value_name = "PATH")]
        output: Option<PathBuf>,
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
                 "tarafer models pull mms-mya".to_string()),
                ("asr", "myanmar_asr Burmese speech-to-text (Apache-2.0)", PathBuf::from(ASR_MODEL_DIR),
                 asr::missing_files(std::path::Path::new(ASR_MODEL_DIR)),
                 "python tools/stt/export_myanmar_asr.py".to_string()),
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
                bail!("no speech engine installed");
            }
            eprintln!("next    | tarafer speech say \"မင်္ဂလာပါ\" -o hello.wav [--engine f5|mms]");
            eprintln!("        | tarafer speech transcribe hello.wav");
        }
        TtsCommand::Transcribe { audio, model: _, lang: _, model_dir, device, syllables, output } => {
            let dir = model_dir.unwrap_or_else(|| PathBuf::from(ASR_MODEL_DIR));
            let mut engine = Asr::load(&dir, device)?;
            eprintln!("model  | {ASR_MODEL_NAME} on {} ({:.0} ms load)", engine.device, engine.load_ms);
            let samples = asr::read_audio(&audio)?;
            let result = engine.transcribe(&samples)?;
            let text = if syllables { result.syllables.join(" ") } else { result.text() };
            println!("{text}");
            if let Some(path) = &output {
                std::fs::write(path, format!("{text}\n"))
                    .map_err(|e| anyhow::anyhow!("write {}: {e}", path.display()))?;
            }
            eprintln!(
                "done   | {:.1} s audio | {} chunk(s) | {} syllables | {:.2} s run | RTF {:.3}",
                result.audio_seconds,
                result.chunks,
                result.syllables.len(),
                result.run_ms / 1000.0,
                result.run_ms / 1000.0 / result.audio_seconds.max(1e-6),
            );
            if result.syllables.is_empty() {
                eprintln!("note   | no speech recognised (clips under 0.5 s are skipped)");
            }
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

const BENCH_TEXT: &str = "မင်္ဂလာပါ။ ဒီနေ့ ရာသီဥတု အရမ်း သာယာပါတယ်။ ကျွန်တော် မြန်မာစကား ပြောနေပါတယ်။";

struct BenchRow {
    task: &'static str,
    model: String,
    device: &'static str,
    load_s: f64,
    first_s: f64,
    warm_s: f64,
    audio_s: f64,
}

/// Times `run` (returning run ms and audio seconds) once cold, then `runs` more times.
fn bench_row(
    task: &'static str,
    model: String,
    device: &'static str,
    load_ms: f64,
    runs: usize,
    mut run: impl FnMut() -> Result<(f64, f64)>,
) -> Result<BenchRow> {
    let (first, audio_s) = run()?;
    let mut warm = f64::INFINITY;
    for _ in 0..runs {
        warm = warm.min(run()?.0);
    }
    Ok(BenchRow { task, model, device, load_s: load_ms / 1e3, first_s: first / 1e3, warm_s: warm / 1e3, audio_s })
}

/// `tarafer bench speech`: load, first run, best warm run and RTF for every installed engine.
/// Speech-to-text is timed on `audio`, or on the benchmark sentence spoken by a TTS engine.
pub fn bench(device: Device, runs: usize, steps: usize, audio: Option<PathBuf>) -> Result<()> {
    use std::path::Path;
    let runs = runs.max(1);
    let options = SynthOptions { steps, ..SynthOptions::default() };
    let text = normalize(BENCH_TEXT, &Lexicon::default()).text;
    let mut rows = Vec::new();
    let mut spoken: Option<(Vec<f32>, String)> = None;

    if mms::missing_files(Path::new(MMS_MODEL_DIR)).is_empty() {
        let mut tts = Mms::load(Path::new(MMS_MODEL_DIR), device)?;
        let name = "MMS-TTS Burmese (facebook/mms-tts-mya, VITS ONNX)".to_string();
        rows.push(bench_row("tts", name, tts.device, tts.load_ms, runs, || {
            let result = tts.synthesize(&text, &options)?;
            let timing = (result.synth_ms, result.seconds());
            let samples = asr::resample(&result.samples, result.sample_rate, ASR_SAMPLE_RATE);
            spoken = Some((samples, "the benchmark sentence spoken by MMS-TTS".into()));
            Ok(timing)
        })?);
    }
    if missing_files(Path::new(DEFAULT_MODEL_DIR)).is_empty() {
        let mut tts = Tts::load(Path::new(DEFAULT_MODEL_DIR), device)?;
        let name = format!("F5-Myanmar-TTS v2 (ONNX, {steps} steps)");
        rows.push(bench_row("tts", name, tts.device, tts.load_ms, runs, || {
            let result = tts.synthesize(&text, &options)?;
            let timing = (result.synth_ms, result.seconds());
            let samples = asr::resample(&result.samples, result.sample_rate, ASR_SAMPLE_RATE);
            spoken = Some((samples, "the benchmark sentence spoken by F5-Myanmar-TTS".into()));
            Ok(timing)
        })?);
    }
    // The native engine reads the safetensors checkpoints, not the ONNX graphs.
    #[cfg(feature = "wgpu")]
    if ["model.safetensors", "vocos.safetensors", "vocab.txt"]
        .iter()
        .all(|f| Path::new(DEFAULT_MODEL_DIR).join(f).is_file())
    {
        let mut tts = Tts::load(Path::new(DEFAULT_MODEL_DIR), Device::Wgpu)?;
        let name = format!("F5-Myanmar-TTS v2 (native wgpu, {steps} steps)");
        rows.push(bench_row("tts", name, tts.device, tts.load_ms, runs, || {
            let result = tts.synthesize(&text, &options)?;
            Ok((result.synth_ms, result.seconds()))
        })?);
    }
    let mut transcript = None;
    if asr::missing_files(Path::new(ASR_MODEL_DIR)).is_empty() {
        let input = match &audio {
            Some(path) => Some((asr::read_audio(path)?, path.display().to_string())),
            None => spoken.take(),
        };
        match input {
            None => eprintln!("skip   | speech-to-text needs --audio FILE or an installed TTS engine"),
            Some((samples, source)) => {
                let mut engine = Asr::load(Path::new(ASR_MODEL_DIR), device)?;
                rows.push(bench_row("stt", ASR_MODEL_NAME.into(), engine.device, engine.load_ms, runs, || {
                    let result = engine.transcribe(&samples)?;
                    let timing = (result.run_ms, result.audio_seconds);
                    transcript = Some(result.text());
                    Ok(timing)
                })?);
                transcript = transcript.map(|text| format!("{text}\nstt input| {source}"));
            }
        }
    }
    if rows.is_empty() {
        bail!("no speech engine installed; see `tarafer speech status`");
    }
    println!("tarafer bench speech | best of {runs} warm run(s) | RTF = warm run time / audio time");
    println!(
        "{:<4} {:<50} {:<9} {:>6} {:>7} {:>7} {:>7} {:>6}",
        "task", "model", "device", "load s", "first s", "warm s", "audio s", "RTF"
    );
    for r in &rows {
        println!(
            "{:<4} {:<50} {:<9} {:>6.2} {:>7.2} {:>7.3} {:>7.1} {:>6.3}",
            r.task, r.model, r.device, r.load_s, r.first_s, r.warm_s, r.audio_s,
            r.warm_s / r.audio_s.max(1e-6)
        );
    }
    println!("tts text | {BENCH_TEXT}");
    if let Some(text) = transcript {
        println!("stt text | {text}");
    }
    Ok(())
}
