mod profile;

use anyhow::{Context, Result};
use clap::Parser;
use profile::{Profiler, ProfileMeta, TurnRow, MULTI_TURN_SCRIPT, PROFILE_MAX_NEW};
use std::path::PathBuf;
use taraference_core::{CudaModel, DecodeBackend, Session, Tokenizer};
use taraference_gguf::GgufFile;

#[derive(Parser, Debug)]
#[command(name = "taraference", about = "CUDA multi-turn GGUF (Qwen2.5-3B)")]
struct Cli {
    model: PathBuf,
    /// Max tokens per assistant reply (default sized for full answers on 4GB).
    #[arg(short = 'n', long, default_value_t = 512)]
    max_new: usize,
    /// KV context length (default for RTX 3050 Ti 4GB + 3B Q4).
    #[arg(long, default_value_t = 5000)]
    ctx: usize,
    #[arg(long)]
    prompt: Option<String>,
    /// Attention / decode backend for A/B tests: fast | basic | online
    #[arg(long, default_value = "fast", value_parser = parse_decode)]
    decode: DecodeBackend,
    /// Benchmark: multi-turn chat + CPU/GPU sampling + rich report.
    #[arg(long, default_value_t = false)]
    profile: bool,
}

fn parse_decode(s: &str) -> Result<DecodeBackend, String> {
    s.parse::<DecodeBackend>().map_err(|e| e.to_string())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    eprintln!("loading {} …", cli.model.display());
    let gguf =
        GgufFile::open(&cli.model).with_context(|| format!("open {}", cli.model.display()))?;
    let weight_gib = gguf.total_tensor_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
    let tok = Tokenizer::from_gguf(&gguf)?;
    let mut model = CudaModel::load_with(&gguf, cli.decode)?;
    let max_seq = cli.ctx.min(model.cfg.n_ctx);

    if cli.profile {
        run_profile(&mut model, &tok, max_seq, weight_gib, &cli)?;
    } else {
        let mut session = Session::new(&mut model, &tok, max_seq, cli.max_new)?;
        if let Some(p) = cli.prompt {
            session.turn(&p)?;
        } else {
            session.run_repl()?;
        }
    }
    Ok(())
}

fn run_profile(
    model: &mut CudaModel,
    tok: &Tokenizer,
    max_seq: usize,
    weight_gib: f64,
    cli: &Cli,
) -> Result<()> {
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
        cfg: model.cfg.clone(),
        weight_gib,
        max_seq,
        max_new,
        sample_interval_ms: 100,
        decode: model.decode,
    };

    eprintln!(
        "profile mode ({mode}) | decode={} | max_new={max_new} | turns={}",
        model.decode.name(),
        script.len()
    );
    for (i, u) in script.iter().enumerate() {
        eprintln!("  turn {}: {u:?}", i + 1);
    }

    let mut session = Session::new(model, tok, max_seq, max_new)?;
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

    prof.stop_and_report(&rows, mode, &meta);
    Ok(())
}
