fn main() {
    let g = taraference_gguf::GgufFile::open("models/Qwen3.5-0.8B-Q4_K_M.gguf").unwrap();
    for n in ["token_embd.weight", "output.weight", "output_norm.weight"] {
        if let Some(t) = g.tensor(n) {
            println!("{n} dims={:?} ty={:?}", t.dims, t.ggml_type);
        } else {
            println!("{n} MISSING");
        }
    }
    // encode check via listing specials already done
}
