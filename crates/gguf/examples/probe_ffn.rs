fn main() {
  let g = taraference_gguf::GgufFile::open(r"models/Qwen3.5-4B-Q4_K_M.gguf").unwrap();
  for i in [0usize, 3, 1] {
    for part in ["ffn_gate","ffn_up","ffn_down","attn_qkv","ssm_out","attn_gate"] {
      let n = format!("blk.{i}.{part}.weight");
      if let Some(t) = g.tensor(&n) {
        println!("{n} type={:?} dims={:?}", t.ggml_type, t.dims);
      }
    }
  }
}
