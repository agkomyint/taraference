use anyhow::Result;
use std::path::Path;
use taraference_gguf::GgufFile;
use taraference_wgpu::{runner::prompt_tokens, WgpuModel};
fn main() -> Result<()> {
    let file = GgufFile::open("models/Qwen2.5-3B-Instruct-Q4_K_M.gguf")?;
    let (_, ids) = prompt_tokens(
        &file,
        "Explain why the sky is blue in a short paragraph.",
        false,
    )?;
    let mut model = WgpuModel::load(&file, 512)?;
    let mut next = model.forward_greedy(&ids)?;
    for _ in 0..8 {
        next = model.forward_greedy(&[next])?;
    }
    let profile = model.profile_decode(next)?;
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/wgpu-profile.json".into());
    std::fs::write(Path::new(&output), serde_json::to_vec_pretty(&profile)?)?;
    println!("{}", serde_json::to_string_pretty(&profile)?);
    Ok(())
}
