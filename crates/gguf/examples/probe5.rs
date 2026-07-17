fn main() {
    let g = taraference_gguf::GgufFile::open("models/Qwen2.5-3B-Instruct-Q4_K_M.gguf").unwrap();
    let t = g.tensor("token_embd.weight").unwrap();
    println!("qwen25 embd dims={:?} ty={:?}", t.dims, t.ggml_type);
}
