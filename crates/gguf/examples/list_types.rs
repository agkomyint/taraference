use std::collections::BTreeMap;
use taraference_gguf::GgufFile;

fn main() -> anyhow::Result<()> {
    let p = std::env::args().nth(1).expect("path");
    let g = GgufFile::open(&p)?;
    let mut m: BTreeMap<String, usize> = BTreeMap::new();
    for t in &g.tensors {
        *m.entry(t.ggml_type.name().to_string()).or_default() += 1;
    }
    println!("=== type counts ===");
    for (k, v) in &m {
        println!("{k}: {v}");
    }
    println!("=== non Q4_K / F32 tensors ===");
    for t in &g.tensors {
        let n = t.ggml_type.name();
        if n != "Q4_K" && n != "F32" {
            println!("  {n}  {}  dims={:?}", t.name, t.dims);
        }
    }
    Ok(())
}
