//! CLI entry: interactive chat, profile, or OpenAI-compatible server.
//!
//! Layout:
//! - **inference** — `taraference_core::InferenceEngine` / `Session`
//! - **server** — `serve` module (OpenAI `/v1/*`)
//! - **cli** — this binary (`profile` + flags)

mod profile;
mod serve;

use anyhow::Result;
use clap::Parser;
use profile::{Profiler, ProfileMeta, TurnRow, MULTI_TURN_SCRIPT, PROFILE_MAX_NEW};
use std::path::PathBuf;
use taraference_core::{DecodeBackend, InferenceEngine, SessionOptions};

#[derive(Parser, Debug)]
#[command(
    name = "taraference",
    about = "CUDA multi-turn GGUF inference (chat / profile / OpenAI server)"
)]
struct Cli {
    /// Path to GGUF weights (also the only OpenAI model id = file stem when serving).
    model: PathBuf,
    /// Max tokens per assistant reply (default sized for full answers on 4GB).
    #[arg(short = 'n', long, default_value_t = 512)]
    max_new: usize,
    /// KV context length (default for RTX 3050 Ti 4GB + 3B Q4).
    #[arg(long, default_value_t = 5000)]
    ctx: usize,
    #[arg(long)]
    prompt: Option<String>,
    /// Attention / decode backend (see registry). Default = marked `is_default` (fastv2).
    #[arg(long, default_value_t = DecodeBackend::default(), value_parser = parse_decode)]
    decode: DecodeBackend,
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
    let cli = Cli::parse();

    if let Some(port) = cli.serve {
        if cli.profile {
            anyhow::bail!("--serve and --profile cannot be used together");
        }
        return run_serve(&cli, port);
    }

    let mut engine = InferenceEngine::load_path(&cli.model, cli.decode, cli.ctx, cli.max_new)?;

    if cli.profile {
        run_profile(&mut engine, &cli)?;
    } else {
        let opts = SessionOptions::interactive(cli.max_new);
        let mut session = engine.session(opts);
        if let Some(p) = cli.prompt {
            session.turn(&p)?;
        } else {
            session.run_repl()?;
        }
    }
    Ok(())
}

fn run_serve(cli: &Cli, port: u16) -> Result<()> {
    // RUST_LOG=debug for verbose tower/http; default = request + completion lines.
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
        model = %cli.model.display(),
        port,
        ctx = cli.ctx,
        max_new = cli.max_new,
        decode = %cli.decode,
        "starting serve mode"
    );

    let engine = InferenceEngine::load_path(&cli.model, cli.decode, cli.ctx, cli.max_new)?;
    tracing::info!(
        model_id = %engine.model_id,
        weight_gib = format!("{:.2}", engine.weight_gib),
        max_seq = engine.max_seq,
        "model loaded"
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(serve::run(engine, port))
}

fn run_profile(engine: &mut InferenceEngine, cli: &Cli) -> Result<()> {
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

    let meta = ProfileMeta {
        model_path: cli.model.display().to_string(),
        cfg: engine.cfg().clone(),
        weight_gib: engine.weight_gib,
        max_seq: engine.max_seq,
        max_new,
        sample_interval_ms: 100,
        decode: engine.decode(),
    };

    eprintln!(
        "profile mode ({mode}) | decode={} | max_new={max_new} | turns={}",
        engine.decode().name(),
        script.len()
    );
    for (i, u) in script.iter().enumerate() {
        eprintln!("  turn {}: {u:?}", i + 1);
    }

    let opts = SessionOptions::interactive(max_new);
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
