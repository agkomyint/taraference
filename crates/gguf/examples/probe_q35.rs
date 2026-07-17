fn main() {
  let g = taraference_gguf::GgufFile::open(r"models/Qwen3.5-4B-Q4_K_M.gguf").unwrap();
  let keys = [
    "general.architecture",
    "qwen35.block_count",
    "qwen35.embedding_length",
    "qwen35.attention.head_count",
    "qwen35.attention.head_count_kv",
    "qwen35.attention.key_length",
    "qwen35.full_attention_interval",
    "qwen35.ssm.conv_kernel",
    "qwen35.ssm.state_size",
    "qwen35.ssm.group_count",
    "qwen35.ssm.time_step_rank",
    "qwen35.ssm.inner_size",
    "qwen35.vocab_size",
    "qwen35.rope.dimension_count",
  ];
  for k in keys {
    if let Some(v) = g.meta_u32(k) { println!("{k} = u32 {v}"); }
    else if let Some(v) = g.meta_u64(k) { println!("{k} = u64 {v}"); }
    else if let Some(v) = g.meta_f32(k) { println!("{k} = f32 {v}"); }
    else if let Some(v) = g.meta_str(k) { println!("{k} = str {v}"); }
    else { println!("{k} = MISSING"); }
  }
  // tensor type sample for blk.0
  for n in ["blk.0.attn_qkv.weight","blk.0.ssm_out.weight","blk.0.ffn_gate.weight","blk.0.ffn_up.weight","blk.0.ffn_down.weight","output.weight","token_embd.weight","blk.0.ssm_alpha.weight","blk.0.ssm_beta.weight","blk.0.ssm_in.weight"] {
    if let Some(t) = g.tensor(n) {
      println!("tensor {n} type={:?} dims={:?}", t.ggml_type, t.dims);
    } else {
      // try alt names
      println!("tensor {n} MISSING");
    }
  }
  for n in ["blk.0.attn_q.weight","blk.0.attn_k.weight","blk.0.attn_v.weight","blk.0.attn_output.weight"] {
    if let Some(t) = g.tensor(n) { println!("tensor {n} type={:?} dims={:?}", t.ggml_type, t.dims); }
  }
}
