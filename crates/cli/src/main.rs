use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use taraference_core::{CudaModel, Session, Tokenizer};
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    eprintln!("loading {} …", cli.model.display());
    let gguf =
        GgufFile::open(&cli.model).with_context(|| format!("open {}", cli.model.display()))?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let mut model = CudaModel::load(&gguf)?;
    let max_seq = cli.ctx.min(model.cfg.n_ctx);
    let mut session = Session::new(&mut model, &tok, max_seq, cli.max_new)?;

    if let Some(p) = cli.prompt {
        session.turn(&p)?;
    } else {
        session.run_repl()?;
    }
    Ok(())
}
