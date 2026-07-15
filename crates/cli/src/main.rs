use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use taraference_core::{CudaModel, Session, Tokenizer};
use taraference_gguf::GgufFile;

#[derive(Parser, Debug)]
#[command(name = "taraference", about = "CUDA multi-turn GGUF (Qwen2.5-3B)")]
struct Cli {
    model: PathBuf,
    #[arg(short = 'n', long, default_value_t = 64)]
    max_new: usize,
    #[arg(long, default_value_t = 2048)]
    ctx: usize,
    #[arg(long)]
    prompt: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    eprintln!("loading {} …", cli.model.display());
    let gguf =
        GgufFile::open(&cli.model).with_context(|| format!("open {}", cli.model.display()))?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let mut model = CudaModel::load(&gguf)?;
    eprintln!(
        "ready GPU | L={} d={} vocab={} | ceiling≈{:.0} tok/s",
        model.cfg.n_layer, model.cfg.n_embd, model.cfg.n_vocab, model.bw_ceiling_tps
    );

    let max_seq = cli.ctx.min(model.cfg.n_ctx);
    let mut session = Session::new(&mut model, &tok, max_seq, cli.max_new)?;

    if let Some(p) = cli.prompt {
        session.turn(&p)?;
    } else {
        session.run_repl()?;
    }
    Ok(())
}
