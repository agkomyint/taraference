fn main() {
    let path = std::env::args().nth(1).expect("path");
    let g = taraference_gguf::GgufFile::open(&path).expect("open");
    println!("arch={:?}", g.architecture());
    let mut keys: Vec<_> = g.metadata.keys().cloned().collect();
    keys.sort();
    for k in &keys {
        let v = &g.metadata[k];
        let s = format!("{v:?}");
        let short = if s.len() > 200 { format!("{}...", &s[..200]) } else { s };
        if k.starts_with("general.")
            || k.contains("ssm")
            || k.contains("attention")
            || k.contains("rope")
            || k.contains("block")
            || k.contains("embedding")
            || k.contains("context")
            || k.contains("feed_forward")
            || k.contains("full_attention")
            || k.contains("recurrent")
            || k.contains("vocab")
            || k.contains("layer")
            || k.contains("tokenizer.ggml.eos")
            || k.contains("tokenizer.ggml.bos")
        {
            println!("{k} = {short}");
        }
    }
    println!("--- tensors ---");
    let mut names: Vec<_> = g.tensors.iter().map(|t| (t.name.clone(), t.dims.clone(), format!("{:?}", t.ggml_type))).collect();
    names.sort_by(|a,b| a.0.cmp(&b.0));
    for (i,(n,d,ty)) in names.iter().enumerate() {
        if i < 100 || n.contains("ssm") || n.contains("attn") || n.contains("gate") || n.contains("post") || n.contains("ffn") {
            println!("{n} dims={d:?} {ty}");
        }
    }
    println!("total tensors {}", g.tensors.len());
}
