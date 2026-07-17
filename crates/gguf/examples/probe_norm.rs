fn main() {
    for path in ["models/Qwen3.5-0.8B-Q4_K_M.gguf", "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"] {
        let g = taraference_gguf::GgufFile::open(path).unwrap();
        let t = g.tensor("blk.0.attn_norm.weight").unwrap();
        let raw = g.tensor_data(t);
        let mut vals = Vec::new();
        for i in 0..8 {
            let b = &raw[i*4..i*4+4];
            vals.push(f32::from_le_bytes([b[0],b[1],b[2],b[3]]));
        }
        println!("{path} attn_norm: {:?}", vals);
        for name in ["blk.0.attn_q_norm.weight", "blk.3.attn_q_norm.weight"] {
            if let Some(t2) = g.tensor(name) {
                let raw = g.tensor_data(t2);
                let mut vals = Vec::new();
                for i in 0..8 {
                    let b = &raw[i*4..i*4+4];
                    vals.push(f32::from_le_bytes([b[0],b[1],b[2],b[3]]));
                }
                println!("  {name}: {:?}", vals);
            }
        }
    }
}
