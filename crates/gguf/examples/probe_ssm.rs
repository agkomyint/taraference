fn main() {
    let g = taraference_gguf::GgufFile::open(r"models\Qwen3.5-4B-Q4_K_M.gguf").unwrap();
    for name in ["blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_norm.weight", "blk.0.attn_norm.weight"] {
        let t = g.tensor(name).unwrap();
        let raw = g.tensor_data(t);
        let v: Vec<f32> = raw.chunks_exact(4).take(8).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
        println!("{name} dims={:?} first8={:?}", t.dims, v);
    }
    // compare 0.8b
    let g2 = taraference_gguf::GgufFile::open(r"models\Qwen3.5-0.8B-Q4_K_M.gguf").unwrap();
    for name in ["blk.0.ssm_a", "blk.0.ssm_dt.bias"] {
        let t = g2.tensor(name).unwrap();
        let raw = g2.tensor_data(t);
        let v: Vec<f32> = raw.chunks_exact(4).take(8).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
        println!("0.8b {name} dims={:?} first8={:?}", t.dims, v);
    }
}
