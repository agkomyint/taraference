fn main() {
    let g = taraference_gguf::GgufFile::open("models/Qwen3.5-0.8B-Q4_K_M.gguf").unwrap();
    // dump a few tokens around specials
    if let Some(taraference_gguf::Value::Array { items, .. }) = g.metadata.get("tokenizer.ggml.tokens") {
        for (i, v) in items.iter().enumerate() {
            if let Some(s) = v.as_str() {
                if s.contains("im_") || s.contains("think") || s.contains("end") || s.contains("start") || s.contains("vision") || s.contains("pad") {
                    if i < 300 || s.contains("think") || s.contains("im_") {
                        println!("{i}: {s:?}");
                    }
                }
            }
        }
        // also print last 50 special-ish
        let n = items.len();
        println!("vocab={n}");
        for i in (n.saturating_sub(80))..n {
            if let Some(s) = items[i].as_str() {
                println!("{i}: {s:?}");
            }
        }
    }
    // dequant ssm_a is F32 already
    let t = g.tensor("blk.0.ssm_a").unwrap();
    println!("ssm_a dims={:?} type={:?} offset={}", t.dims, t.ggml_type, t.offset);
}
