fn main() {
  let g = taraference_gguf::GgufFile::open(r"models/Qwen3.5-4B-Q4_K_M.gguf").unwrap();
  for n in g.tensors.iter().map(|t| t.name.clone()).filter(|n| n.starts_with("blk.0.") && (n.contains("ssm") || n.contains("gate") || n.contains("qkv") || n.contains("attn"))) {
    let t = g.tensor(&n).unwrap();
    println!("{n} type={:?} dims={:?}", t.ggml_type, t.dims);
  }
}
