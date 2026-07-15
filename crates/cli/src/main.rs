//! CLI entry: download models, interactive chat, profile, OpenAI server, self-update.
//!
//! Binary name: **`tarafer`** (crate package remains `taraference`).
//!
//! Layout:
//! - **inference** — `taraference_core::InferenceEngine` / `Session`
//! - **server** — `serve` module (OpenAI `/v1/*`)
//! - **cli** — this binary (`download` + `profile` + `update` / `install`)

mod download;
mod profile;
mod self_update;
mod serve;

use anyhow::{bail, Result};
use clap::Parser;
use download::download_models;
use profile::{
    enrich_gpu_info_from_smi, GpuProfileInfo, Profiler, ProfileMeta, TurnRow, MULTI_TURN_SCRIPT,
    PROFILE_MAX_NEW,
};
use std::path::PathBuf;
use taraference_core::{DecodeBackend, InferenceEngine, SessionOptions};

const AFTER_HELP: &str = "\
Commands (no model path required):
  tarafer update              Download latest GitHub release and replace this binary
  tarafer update v0.2.0       Pin a release tag
  tarafer install             Copy tarafer to ~/.local/bin (add to PATH)

Fast path on a new Linux GPU box:
  curl -fsSL .../tarafer-linux-x86_64.tar.gz | tar xz
  ./tarafer install
  tarafer --download 0.5b
  tarafer --download 7b          # larger model (~4.7 GiB)
  tarafer --download large       # 7b + 14b
  tarafer --download list        # show all tags
  tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
";

#[derive(Parser, Debug)]
#[command(
    name = "tarafer",
    version,
    about = "CUDA multi-turn GGUF inference (chat / profile / OpenAI server / model download)",
    after_help = AFTER_HELP
)]
struct Cli {
    /// Path to GGUF weights (file stem = OpenAI model id when serving).
    /// Optional if you only pass `--download`, or use `update` / `install`.
    model: Option<PathBuf>,
    /// Download GGUF(s) from Hugging Face into `--models-dir`.
    /// `list` | `all` (0.5b+3b) | `small` | `large` (7b+14b) | `profile` | `7b` | `14b` | …
    /// Skip existing unless `--force`.
    #[arg(long, value_name = "WHICH", num_args = 0..=1, default_missing_value = "all")]
    download: Option<String>,
    /// Directory for `--download` (default: `models` under the current working directory).
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,
    /// Re-download even if the GGUF already exists.
    #[arg(long, default_value_t = false)]
    force: bool,
    /// Max tokens per assistant reply (default sized for full answers on 4GB).
    #[arg(short = 'n', long, default_value_t = 512)]
    max_new: usize,
    /// KV context length (default for RTX 3050 Ti 4GB + 3B Q4).
    #[arg(long, default_value_t = 5000)]
    ctx: usize,
    #[arg(long)]
    prompt: Option<String>,
    /// Attention / decode backend (see registry). Default = marked `is_default` (fastv2).
    /// Use `flash` for long-context flash-decoding (split KV).
    #[arg(long, default_value_t = DecodeBackend::default(), value_parser = parse_decode)]
    decode: DecodeBackend,
    /// Capture CUDA graph for single-token decode after warm-up (default: on).
    #[arg(long, default_value_t = true)]
    cuda_graph: bool,
    /// Disable CUDA graph capture / replay.
    #[arg(long, default_value_t = false)]
    no_cuda_graph: bool,
    /// Prompt Lookup Decoding (n-gram speculative; full-draft verify only).
    #[arg(long, default_value_t = false)]
    pld: bool,
    /// Benchmark: multi-turn chat + CPU/GPU sampling + rich report.
    #[arg(long, default_value_t = false)]
    profile: bool,
    /// Start OpenAI-compatible HTTP server on PORT (default 8787). One GGUF = one model.
    #[arg(long, value_name = "PORT", num_args = 0..=1, default_missing_value = "8787")]
    serve: Option<u16>,
}

fn parse_decode(s: &str) -> Result<DecodeBackend, String> {
    s.parse::<DecodeBackend>().map_err(|e| e.to_string())
}

fn main() -> Result<()> {
    // Lightweight subcommands so `tarafer update` does not conflict with a model path.
    let mut argv: Vec<String> = std::env::args().collect();
    if argv.len() >= 2 {
        match argv[1].as_str() {
            "update" | "--update" => {
                let tag = argv.get(2).map(|s| s.as_str()).filter(|s| !s.starts_with('-'));
                // Optional: tarafer update --install  → also land in ~/.local/bin
                let also_install = argv.iter().any(|a| a == "--install" || a == "install");
                if also_install {
                    let dest = self_update::default_install_dir()?.join("tarafer");
                    self_update::self_update(tag, Some(dest))?;
                } else {
                    self_update::self_update(tag, None)?;
                }
                return Ok(());
            }
            "install" | "--install" => {
                // tarafer install [dir]
                let dir = argv.get(2).map(PathBuf::from);
                self_update::install_to_path(dir)?;
                return Ok(());
            }
            _ => {}
        }
    }
    // Drop the binary name for clap if we ever need re-parse; clap reads env::args itself.
    let _ = &mut argv;

    let cli = Cli::parse();

    if let Some(ref which) = cli.download {
        let paths = download_models(&cli.models_dir, which, cli.force)?;
        // Download-only mode: no model path and no serve/profile/prompt.
        if cli.model.is_none() && cli.serve.is_none() && !cli.profile && cli.prompt.is_none() {
            if paths.is_empty() {
                // e.g. --download list
                return Ok(());
            }
            eprintln!("done. examples:");
            for p in paths.iter().take(5) {
                eprintln!("  tarafer {}", p.display());
                eprintln!("  tarafer {} --profile", p.display());
            }
            return Ok(());
        }
    }

    let model = match &cli.model {
        Some(p) => p.clone(),
        None => bail!(
            "missing model path.\n  \
             update:   tarafer update\n  \
             install:  tarafer install\n  \
             download: tarafer --download 0.5b\n  \
             then run: tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
        ),
    };

    // Rebuild a minimal view for subcommands that need model path as PathBuf.
    let cli = Cli {
        model: Some(model.clone()),
        ..cli
    };

    if let Some(port) = cli.serve {
        if cli.profile {
            bail!("--serve and --profile cannot be used together");
        }
        return run_serve(&cli, &model, port);
    }

    let cuda_graph = cli.cuda_graph && !cli.no_cuda_graph;
    let mut engine = InferenceEngine::load_with_flags(
        &model,
        cli.decode,
        cli.ctx,
        cli.max_new,
        cuda_graph,
        cli.pld,
    )?;

    if cli.profile {
        run_profile(&mut engine, &cli, &model)?;
    } else {
        let mut opts = SessionOptions::interactive(cli.max_new);
        opts.pld = cli.pld;
        let mut session = engine.session(opts);
        if let Some(p) = cli.prompt {
            session.turn(&p)?;
        } else {
            session.run_repl()?;
        }
    }
    Ok(())
}

fn run_serve(cli: &Cli, model: &PathBuf, port: u16) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=info,taraference=info".into()),
        )
        .with_target(true)
        .with_thread_ids(false)
        .compact()
        .init();

    tracing::info!(
        model = %model.display(),
        port,
        ctx = cli.ctx,
        max_new = cli.max_new,
        decode = %cli.decode,
        "starting serve mode"
    );

    let cuda_graph = cli.cuda_graph && !cli.no_cuda_graph;
    let engine = InferenceEngine::load_with_flags(
        model,
        cli.decode,
        cli.ctx,
        cli.max_new,
        cuda_graph,
        cli.pld,
    )?;
    tracing::info!(
        model_id = %engine.model_id,
        weight_gib = format!("{:.2}", engine.weight_gib),
        max_seq = engine.max_seq,
        cuda_graph,
        pld = cli.pld,
        "model loaded"
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(serve::run(engine, port))
}

fn run_profile(engine: &mut InferenceEngine, cli: &Cli, model: &PathBuf) -> Result<()> {
    let max_new = cli.max_new.min(PROFILE_MAX_NEW);
    let script: Vec<String> = if let Some(ref p) = cli.prompt {
        vec![p.clone()]
    } else {
        MULTI_TURN_SCRIPT.iter().map(|s| (*s).to_string()).collect()
    };

    let mode = if script.len() == 1 {
        "single-turn"
    } else {
        "multi-turn"
    };

    let (cc_maj, cc_min) = engine.compute_capability();
    let mut gpu = GpuProfileInfo {
        name: engine.gpu_name().to_string(),
        compute_cap: format!("{cc_maj}.{cc_min}"),
        nvrtc_arch: engine.nvrtc_arch().to_string(),
        ..Default::default()
    };
    enrich_gpu_info_from_smi(&mut gpu);

    let meta = ProfileMeta {
        model_path: model.display().to_string(),
        cfg: engine.cfg().clone(),
        weight_gib: engine.weight_gib,
        max_seq: engine.max_seq,
        max_new,
        sample_interval_ms: 100,
        decode: engine.decode(),
        gpu: gpu.clone(),
    };

    eprintln!(
        "profile mode ({mode}) | decode={} | max_new={max_new} | turns={} | gpu={} ({})",
        engine.decode().name(),
        script.len(),
        gpu.name,
        gpu.nvrtc_arch
    );
    for (i, u) in script.iter().enumerate() {
        eprintln!("  turn {}: {u:?}", i + 1);
    }

    let mut opts = SessionOptions::interactive(max_new);
    opts.pld = engine.pld;
    let mut session = engine.session(opts);
    let mut prof = Profiler::start(100);
    let mut rows = Vec::with_capacity(script.len());

    for (i, user) in script.iter().enumerate() {
        eprintln!("\n—— turn {}/{} ——", i + 1, script.len());
        eprintln!("user: {user}");
        let stats = session.turn(user)?;
        rows.push(TurnRow {
            index: i,
            user: user.clone(),
            stats,
        });
    }

    let _log = prof.stop_and_report(&rows, mode, &meta);
    Ok(())
}
