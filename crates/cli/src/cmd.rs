//! Command layer: `text`, `speech`, `image`, `serve`, `models`, `bench`, `doctor`, `update`, `install`.

use crate::registry::{self, State, Task, REGISTRY};
use crate::{bench, image_commands, self_update, serve, tts_commands};
use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};
use std::io::Read;
use std::path::{Path, PathBuf};
use taraference_core::{DecodeBackend, InferenceEngine, SessionOptions};

const AFTER_HELP: &str = "\
Model selection: -m PATH or tag, else $TARAFER_MODEL, else the only .gguf in ./models.

Quick start:
  tarafer models pull 4b
  tarafer text gen \"Explain RAII in two sentences\"
  tarafer serve
";

#[derive(Parser, Debug)]
#[command(
    name = "tarafer",
    version,
    about = "Lean local runtime for small specialised AI models",
    after_help = AFTER_HELP
)]
struct App {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Text generation, chat and embeddings (small GGUF models).
    Text {
        #[command(subcommand)]
        cmd: TextCmd,
    },
    /// Burmese speech: `say <text|-> -o out.wav [--engine f5|mms]`, `transcribe <audio>`, `status`.
    #[command(disable_help_flag = true)]
    Speech {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Image tools: `remove-bg <file>` (get the model with `models pull`).
    #[command(disable_help_flag = true)]
    Image {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// OpenAI-compatible server (unauthenticated, all interfaces; keep behind a firewall).
    Serve {
        #[command(flatten)]
        rt: Runtime,
        #[arg(long, default_value_t = 8787)]
        port: u16,
    },
    /// Model registry: list, pull, rm, info.
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
    /// Speed benchmark (multi-turn chat) with a report.
    Bench {
        #[command(flatten)]
        rt: Runtime,
        /// Single custom prompt instead of the built-in multi-turn script.
        #[arg(long)]
        prompt: Option<String>,
        #[command(subcommand)]
        task: Option<BenchTask>,
    },
    /// Download the latest release and replace this binary.
    Update {
        /// Release tag to pin (default: latest).
        tag: Option<String>,
        /// Also copy the result to ~/.local/bin.
        #[arg(long)]
        install: bool,
    },
    /// Copy this binary to ~/.local/bin (or DIR).
    Install { dir: Option<PathBuf> },
    /// Show GPU, installed models and speech engines.
    Doctor {
        #[arg(long, default_value = "models")]
        models_dir: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
enum BenchTask {
    /// Text-to-speech and speech-to-text: load, first run, warm run and RTF per installed engine.
    Speech {
        /// Inference device: auto (ORT CUDA/DirectML/CPU), cuda, directml or cpu.
        #[arg(long, default_value = "auto")]
        device: taraference_tts::Device,
        /// Warm runs per engine; the best one is reported.
        #[arg(long, default_value_t = 3)]
        runs: usize,
        /// F5 flow-matching steps.
        #[arg(long, default_value_t = 32)]
        steps: usize,
        /// Audio for speech-to-text (default: the benchmark sentence spoken by a TTS engine).
        #[arg(long, value_name = "FILE")]
        audio: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum TextCmd {
    /// Generate a reply and exit. `-` reads the prompt from stdin.
    Gen {
        prompt: Option<String>,
        #[command(flatten)]
        rt: Runtime,
        #[command(flatten)]
        sampling: Sampling,
        /// Treat the prompt as a raw completion (no chat template).
        #[arg(long)]
        raw: bool,
        /// Print the reply and token stats as one JSON object.
        #[arg(long)]
        json: bool,
        /// Native compute backend (wgpu currently supports Qwen2/Qwen2.5).
        #[arg(long, default_value = "cuda", value_parser = ["cuda", "wgpu"])]
        backend: String,
    },
    /// Interactive chat (thin REPL).
    Chat {
        #[command(flatten)]
        rt: Runtime,
        #[command(flatten)]
        sampling: Sampling,
    },
    /// Print the embedding vector as JSON (needs an embedding GGUF: `models pull embed-4b`).
    Embed {
        text: Option<String>,
        #[command(flatten)]
        rt: Runtime,
    },
}

#[derive(Subcommand, Debug)]
enum ModelsCmd {
    /// Catalog of every model (text, embed, speech, image) with install state and licence.
    List {
        /// Only one task: text, embed, speech or image.
        #[arg(long, value_parser = parse_task)]
        task: Option<Task>,
        /// Hide non-commercial models.
        #[arg(long)]
        commercial: bool,
        #[command(flatten)]
        root: Root,
    },
    /// Download one or more models by id; shows the licence and verifies SHA-256.
    Pull {
        #[arg(required = true)]
        ids: Vec<String>,
        /// Re-download even if installed and verified.
        #[arg(long)]
        force: bool,
        /// Confirm non-commercial licences without prompting.
        #[arg(long)]
        accept_license: bool,
        #[command(flatten)]
        root: Root,
    },
    /// Re-check installed files against pinned SHA-256 (all installed models if no id).
    Verify {
        id: Option<String>,
        #[command(flatten)]
        root: Root,
    },
    /// Delete an installed model.
    Rm {
        id: String,
        #[command(flatten)]
        root: Root,
    },
    /// Source, revision, licence, files and local state for one model.
    Info {
        id: String,
        #[command(flatten)]
        root: Root,
    },
}

#[derive(Args, Debug, Clone)]
struct Root {
    /// Models directory.
    #[arg(long, default_value = "models", global = true)]
    models_dir: PathBuf,
}

fn parse_task(s: &str) -> Result<Task, String> {
    match s {
        "text" => Ok(Task::Text),
        "embed" => Ok(Task::Embed),
        "speech" => Ok(Task::Speech),
        "image" => Ok(Task::Image),
        _ => Err("expected text, embed, speech or image".into()),
    }
}

#[derive(Args, Debug, Clone)]
struct Runtime {
    /// GGUF path or tag (4b, 0.8b, ...). Default: $TARAFER_MODEL, else the only .gguf in ./models.
    #[arg(short, long, value_name = "MODEL")]
    model: Option<String>,
    /// Max new tokens per reply.
    #[arg(short = 'n', long, default_value_t = 512)]
    max_new: usize,
    /// KV context length.
    #[arg(long, default_value_t = 5000)]
    ctx: usize,
    /// Open the model's <think> block (Qwen3 / Qwen3.5).
    #[arg(long)]
    think: bool,
    /// Speculative decoding via prompt lookup (CUDA, greedy). Same output, faster on repetitive
    /// text (code edits, quoting), slower on free-form text. MODE: smart (default) or aggressive.
    #[arg(long, value_name = "MODE", num_args = 0..=1, default_missing_value = "smart",
          value_parser = ["smart", "aggressive"])]
    speculative: Option<String>,
    #[arg(long, hide = true, default_value_t = DecodeBackend::default(), value_parser = parse_decode)]
    decode: DecodeBackend,
    #[arg(long, hide = true)]
    no_cuda_graph: bool,
    #[arg(long, hide = true)]
    experimental: bool,
}

#[derive(Args, Debug, Clone)]
struct Sampling {
    /// 0 = greedy (fastest).
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 256)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn parse_decode(s: &str) -> Result<DecodeBackend, String> {
    s.parse::<DecodeBackend>().map_err(|e| e.to_string())
}

/// `-m` as a path, a known tag, or the default lookup.
fn resolve_model(arg: Option<&str>) -> Result<PathBuf> {
    let from_env = std::env::var("TARAFER_MODEL").ok();
    if let Some(m) = arg.or(from_env.as_deref()) {
        let as_path = PathBuf::from(m);
        if as_path.exists() {
            return Ok(as_path);
        }
        if let Ok(e) = registry::find(m) {
            if !matches!(e.task, Task::Text | Task::Embed) {
                bail!("'{m}' is a {} model, not a text model", e.task.name());
            }
            let p = e.main_path(Path::new("models"));
            if p.exists() {
                return Ok(p);
            }
            bail!("model '{m}' is not installed. Run: tarafer models pull {}", e.id);
        }
        bail!("model not found: {m}");
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir("models")
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .collect();
    found.sort();
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => bail!("no model installed. Run: tarafer models pull 4b"),
        _ => bail!(
            "several models in ./models; choose one with -m (or set TARAFER_MODEL):\n  {}",
            found.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n  ")
        ),
    }
}

fn load_engine(rt: &Runtime, model: &Path) -> Result<InferenceEngine> {
    // Read once by the session on first use, so set it before anything loads.
    match rt.speculative.as_deref() {
        Some("aggressive") => std::env::set_var("TARAFER_PLD", "1"),
        Some(_) => {
            std::env::set_var("TARAFER_PLD", "1");
            std::env::set_var("TARAFER_PLD_SMART", "1");
        }
        None => {}
    }
    let engine = InferenceEngine::load_with_flags(
        model,
        rt.decode,
        rt.ctx,
        rt.max_new,
        !rt.no_cuda_graph,
        rt.experimental,
    )?;
    if rt.think && !engine.backend_kind.uses_thinking_template() {
        bail!("--think is not supported by model family {}", engine.backend_kind.name());
    }
    Ok(engine)
}

fn session_opts(rt: &Runtime, s: &Sampling) -> SessionOptions {
    let mut o = SessionOptions::interactive(rt.max_new);
    o.enable_thinking = rt.think;
    o.temperature = s.temperature;
    o.top_p = s.top_p;
    o.top_k = s.top_k;
    o.repetition_penalty = s.repetition_penalty;
    o.seed = s.seed;
    o
}

fn read_text(arg: Option<String>) -> Result<String> {
    match arg.as_deref() {
        Some("-") | None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            Ok(s)
        }
        Some(t) => Ok(t.to_string()),
    }
}

pub fn run() -> Result<()> {
    match App::parse().command {
        Cmd::Text { cmd } => run_text(cmd),
        Cmd::Speech { args } => {
            if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
                eprintln!(
                    "tarafer speech say <text|-> [-o out.wav] [--engine f5|mms] [--lexicon FILE]\n\
                     tarafer speech transcribe <audio> [-o out.txt] [--syllables]\n\
                     tarafer speech status\n(`tarafer speech say --help` lists all options)"
                );
                return Ok(());
            }
            tts_commands::run(args)
        }
        Cmd::Image { args } => image_commands::run(args),
        Cmd::Serve { rt, port } => {
            let model = resolve_model(rt.model.as_deref())?;
            run_serve(&rt, &model, port)
        }
        Cmd::Bench { task: Some(BenchTask::Speech { device, runs, steps, audio }), .. } => {
            tts_commands::bench(device, runs, steps, audio)
        }
        Cmd::Bench { rt, prompt, task: None } => {
            let model = resolve_model(rt.model.as_deref())?;
            let mut engine = load_engine(&rt, &model)?;
            bench::run(&mut engine, prompt, rt.max_new)
        }
        Cmd::Models { cmd } => run_models(cmd),
        Cmd::Update { tag, install } => {
            let dest = if install {
                Some(self_update::default_install_dir()?.join(self_update::bin_name()))
            } else {
                None
            };
            self_update::self_update(tag.as_deref(), dest).map(|_| ())
        }
        Cmd::Install { dir } => self_update::install_to_path(dir).map(|_| ()),
        Cmd::Doctor { models_dir } => run_doctor(&models_dir),
    }
}

fn run_text(cmd: TextCmd) -> Result<()> {
    match cmd {
        TextCmd::Gen { prompt, rt, sampling, raw, json, backend } => {
            let prompt = read_text(prompt)?;
            if prompt.trim().is_empty() {
                bail!("no prompt provided");
            }
            let model = resolve_model(rt.model.as_deref())?;
            if backend == "wgpu" {
                #[cfg(not(feature = "wgpu"))]
                bail!("native wgpu text requires a build with --features wgpu");
                #[cfg(feature = "wgpu")]
                {
                    if rt.think || rt.experimental { bail!("--think and --experimental are not supported by the Qwen2 wgpu backend"); }
                    let context = rt.ctx.min(2048);
                    if context != rt.ctx { eprintln!("wgpu   | context {} -> {context} (current backend limit)", rt.ctx); }
                    let options = taraference_core::sampler::SamplingOptions { temperature: sampling.temperature, top_p: sampling.top_p, top_k: sampling.top_k, repetition_penalty: sampling.repetition_penalty };
                    let result = taraference_wgpu::runner::generate_file(&model, &prompt, raw, rt.max_new, context, options, sampling.seed, !json)?;
                    if json { println!("{}", serde_json::to_string(&result)?); }
                    else { eprintln!("wgpu | {} | {:.2} tok/s decode | {:.2} tok/s prefill | {:.0} ms TTFT", result.model, result.decode_tps, result.prefill_tps, result.ttft_ms); }
                    return Ok(());
                }
            }
            let mut engine = load_engine(&rt, &model)?;
            let mut opts = session_opts(&rt, &sampling);
            if json {
                opts.print_stream = false;
                opts.print_stats = false;
            }
            let model_id = engine.model_id.clone();
            let mut session = engine.session(opts);
            let stats = if raw {
                session.complete_prompt(&prompt)?
            } else {
                session.turn(&prompt)?
            };
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "model": model_id,
                        "reply": stats.reply,
                        "prompt_tokens": stats.prompt_tokens,
                        "gen_tokens": stats.gen_tokens,
                        "ttft_ms": stats.ttft_ms,
                        "decode_tps": stats.decode_tps,
                        "prefill_tps": stats.prefill_tps,
                        "wall_ms": stats.wall_ms,
                    })
                );
            }
            Ok(())
        }
        TextCmd::Chat { rt, sampling } => {
            let model = resolve_model(rt.model.as_deref())?;
            let mut engine = load_engine(&rt, &model)?;
            engine.session(session_opts(&rt, &sampling)).run_repl()
        }
        TextCmd::Embed { text, rt } => {
            let text = read_text(text)?;
            let model = resolve_model(rt.model.as_deref())?;
            let mut engine = load_engine(&rt, &model)?;
            let model_id = engine.model_id.clone();
            let (vectors, tokens) = engine.embeddings(&[text], None)?;
            let v = vectors.into_iter().next().unwrap_or_default();
            println!(
                "{}",
                serde_json::json!({ "model": model_id, "dimensions": v.len(), "tokens": tokens, "embedding": v })
            );
            Ok(())
        }
    }
}

fn run_serve(rt: &Runtime, model: &Path, port: u16) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=info,taraference=info".into()),
        )
        .compact()
        .init();
    let engine = load_engine(rt, model)?;
    tracing::info!(
        model_id = %engine.model_id,
        weight_gib = format!("{:.2}", engine.weight_gib),
        max_seq = engine.max_seq,
        port,
        "serving"
    );
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(serve::run(engine, port))
}

fn run_models(cmd: ModelsCmd) -> Result<()> {
    match cmd {
        ModelsCmd::List { task, commercial, root } => {
            registry::list(&root.models_dir, task, commercial);
            Ok(())
        }
        ModelsCmd::Pull { ids, force, accept_license, root } => {
            registry::pull(&root.models_dir, &ids, force, accept_license)?;
            Ok(())
        }
        ModelsCmd::Verify { id, root } => {
            let failures = registry::verify(&root.models_dir, id.as_deref())?;
            if failures > 0 {
                bail!("{failures} file(s) failed verification");
            }
            Ok(())
        }
        ModelsCmd::Rm { id, root } => registry::remove(&root.models_dir, &id),
        ModelsCmd::Info { id, root } => registry::info(&root.models_dir, &id),
    }
}

fn run_doctor(models_dir: &Path) -> Result<()> {
    eprintln!("tarafer {}", env!("CARGO_PKG_VERSION"));
    match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        .output()
    {
        Ok(o) if o.status.success() => {
            eprintln!("gpu      {}", String::from_utf8_lossy(&o.stdout).trim())
        }
        _ => eprintln!("gpu      nvidia-smi not available (text engine needs a CUDA GPU)"),
    }
    for e in REGISTRY {
        match e.state(models_dir) {
            State::Installed => eprintln!("{:8} {:12} installed", e.task.name(), e.id),
            State::Partial => eprintln!("{:8} {:12} PARTIAL (tarafer models pull {} --force)", e.task.name(), e.id, e.id),
            State::Missing => {}
        }
    }
    if REGISTRY.iter().all(|e| e.state(models_dir) == State::Missing) {
        eprintln!("no models installed; try `tarafer models list`, then `tarafer models pull 4b`");
    }
    Ok(())
}
