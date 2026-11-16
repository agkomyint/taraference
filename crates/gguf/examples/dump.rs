//! Print GGUF metadata (compact) and one tensor per layer pattern: `cargo run -p taraference-gguf --example dump -- file.gguf`
use taraference_gguf::GgufFile;

fn main() -> anyhow::Result<()> {
    let p = std::env::args().nth(1).expect("path");
    let g = GgufFile::open(&p)?;
    println!("== {} tensors={} arch={:?}", p, g.tensors.len(), g.architecture());
    let mut keys: Vec<_> = g.metadata.iter().collect();
    keys.sort_by(|a, b| a.0.cmp(b.0));
    for (k, v) in keys {
        let s = v.display_short();
        println!("{k} = {}", &s[..s.len().min(110)]);
    }
    let mut seen = std::collections::HashSet::new();
    for t in &g.tensors {
        let key: String = t.name.split('.').map(|s| if s.chars().all(|c| c.is_ascii_digit()) { "N" } else { s }).collect::<Vec<_>>().join(".");
        if seen.insert(key) {
            println!("  {}  {:?}  {}", t.name, t.dims, t.ggml_type.name());
        }
    }
    Ok(())
}
