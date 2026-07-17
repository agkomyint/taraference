fn main() {
    let g = taraference_gguf::GgufFile::open("models/Qwen3.5-0.8B-Q4_K_M.gguf").unwrap();
    for k in ["tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id", "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"] {
        println!("{k} = {:?}", g.metadata.get(k));
    }
    if let Some(taraference_gguf::Value::Array { items, .. }) = g.metadata.get("tokenizer.ggml.merges") {
        println!("merges len={}", items.len());
    } else {
        println!("no merges");
    }
    // read ssm_a f32
    let t = g.tensor("blk.0.ssm_a").unwrap();
    let raw = g.tensor_data(t);
    let mut vals = Vec::new();
    for i in 0..16 {
        let b = &raw[i*4..i*4+4];
        vals.push(f32::from_le_bytes([b[0],b[1],b[2],b[3]]));
    }
    println!("ssm_a = {:?}", vals);
}
