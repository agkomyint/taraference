//! Command layer: `text`, `speech`, `image`, `serve`, `models`, `bench`, `doctor`, `update`, `install`.

use crate::download::{self, download_models, SUPPORTED_MODELS};
use crate::{bench, image_commands, self_update, serve, tts_commands};
use anyhow::{bail, Context, Result};
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
    /// Burmese text-to-speech: `say <text|-> -o out.wav [--engine f5|mms]`, `status`.
    #[command(disable_help_flag = true)]
    Speech {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Image tools: `remove-bg <file>`, `model download`.
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
    /// Catalog and installed files.
    List {
        #[arg(long, default_value = "models")]
        models_dir: PathBuf,
    },
    /// Download by tag or group (4b, 0.8b, small, everything, ...).
    Pull {
        which: String,
        #[arg(long, default_value = "models")]
        models_dir: PathBuf,
        /// Re-download even if the file exists.
        #[arg(long)]
        force: bool,
    },
    /// Delete an installed model by tag or file name.
    Rm {
        which: String,
        #[arg(long, default_value = "models")]
        models_dir: PathBuf,
    },
    /// Show source, size and local path for one model.
    Info {
        which: String,
        #[arg(long, default_value = "models")]
        models_dir: PathBuf,
    },
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
        if let Some(spec) = SUPPORTED_MODELS.iter().find(|s| s.tag.eq_ignore_ascii_case(m)) {
            let p = Path::new("models").join(spec.file);
            if p.exists() {
                return Ok(p);
            }
            bail!("model '{m}' is not installed. Run: tarafer models pull {}", spec.tag);
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
        Cmd::Bench { rt, prompt } => {
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
        TextCmd::Gen { prompt, rt, sampling, raw, json } => {
            let prompt = read_text(prompt)?;
            if prompt.trim().is_empty() {
                bail!("no prompt provided");
            }
            let model = resolve_model(rt.model.as_deref())?;
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

fn find_spec(which: &str) -> Result<&'static download::ModelSpec> {
    SUPPORTED_MODELS
        .iter()
        .find(|s| s.tag.eq_ignore_ascii_case(which) || s.file.eq_ignore_ascii_case(which))
        .with_context(|| format!("unknown model '{which}' (see `tarafer models list`)"))
}

fn run_models(cmd: ModelsCmd) -> Result<()> {
    match cmd {
        ModelsCmd::List { models_dir } => {
            eprintln!("{:10}  {:9}  {:9}  {}", "tag", "state", "size", "source / note");
            eprintln!("{}", "-".repeat(96));
            for m in SUPPORTED_MODELS {
                let state = if models_dir.join(m.file).exists() { "installed" } else { "-" };
                eprintln!(
                    "{:10}  {:9}  ~{:.1} GiB    huggingface.co/{} - {}",
                    m.tag, state, m.approx_gib, m.repo, m.note
                );
            }
            eprintln!(
                "\nWeights come straight from the upstream repo above. Check its licence before use;\n\
                 taraference does not ship or host model files."
            );
            Ok(())
        }
        ModelsCmd::Pull { which, models_dir, force } => {
            eprintln!("note: weights are fetched from the upstream Hugging Face repo; review its licence terms.");
            download_models(&models_dir, &which, force)?;
            Ok(())
        }
        ModelsCmd::Rm { which, models_dir } => {
            let path = models_dir.join(find_spec(&which)?.file);
            if !path.exists() {
                bail!("{} is not installed", path.display());
            }
            std::fs::remove_file(&path)?;
            eprintln!("removed {}", path.display());
            Ok(())
        }
        ModelsCmd::Info { which, models_dir } => {
            let spec = find_spec(&which)?;
            let path = models_dir.join(spec.file);
            eprintln!("tag      {}", spec.tag);
            eprintln!("file     {}", spec.file);
            eprintln!("source   https://huggingface.co/{}", spec.repo);
            eprintln!("size     ~{:.1} GiB", spec.approx_gib);
            eprintln!("note     {}", spec.note);
            match std::fs::metadata(&path) {
                Ok(md) => eprintln!(
                    "local    {} ({:.2} GiB)",
                    path.display(),
                    md.len() as f64 / (1u64 << 30) as f64
                ),
                Err(_) => eprintln!("local    not installed (tarafer models pull {})", spec.tag),
            }
            Ok(())
        }
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
    let mut any = false;
    for m in SUPPORTED_MODELS {
        if models_dir.join(m.file).exists() {
            any = true;
            eprintln!("text     {} ({})", m.tag, m.file);
        }
    }
    if !any {
        eprintln!("text     no models in {} (tarafer models pull 4b)", models_dir.display());
    }
    for (name, dir, missing) in [
        (
            "speech/f5",
            taraference_tts::DEFAULT_MODEL_DIR,
            taraference_tts::missing_files(Path::new(taraference_tts::DEFAULT_MODEL_DIR)),
        ),
        (
            "speech/mms",
            taraference_tts::mms::MMS_MODEL_DIR,
            taraference_tts::mms::missing_files(Path::new(taraference_tts::mms::MMS_MODEL_DIR)),
        ),
    ] {
        if missing.is_empty() {
            eprintln!("{name:10} ready ({dir})");
        } else {
            eprintln!("{name:10} not installed ({} file(s) missing in {dir})", missing.len());
        }
    }
    Ok(())
}
