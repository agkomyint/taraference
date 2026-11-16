//! List tensors whose name matches a substring: `... --example names -- file.gguf substr`
use taraference_gguf::GgufFile;
fn main() -> anyhow::Result<()> {
    let mut a = std::env::args().skip(1);
    let g = GgufFile::open(a.next().expect("path"))?;
    let pat = a.next().unwrap_or_default();
    for t in &g.tensors {
        if t.name.contains(&pat) { println!("{}  {:?}  {}", t.name, t.dims, t.ggml_type.name()); }
    }
    Ok(())
}
