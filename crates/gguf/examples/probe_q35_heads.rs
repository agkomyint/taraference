fn main() {
  let g = taraference_gguf::GgufFile::open(r"models/Qwen3.5-4B-Q4_K_M.gguf").unwrap();
  let mut keys: Vec<_> = g.metadata.keys().cloned().collect();
  keys.sort();
  for k in keys {
    if k.contains("attention") || k.contains("head") || k.contains("key") || k.contains("value") || k.contains("embedding") || k.contains("feed") {
      let v = &g.metadata[&k];
      println!("{k} = {v:?}");
    }
  }
  for n in ["blk.3.attn_q.weight","blk.3.attn_k.weight","blk.3.attn_v.weight","blk.3.attn_output.weight","blk.3.attn_q_norm.weight","blk.3.attn_k_norm.weight"] {
    if let Some(t) = g.tensor(n) {
      println!("tensor {n} type={:?} dims={:?}", t.ggml_type, t.dims);
    } else {
      println!("tensor {n} MISSING");
    }
  }
}
