//! `tarafer bench`: short multi-turn chat run with a per-turn table and a summary.

use anyhow::Result;
use taraference_core::{InferenceEngine, SessionOptions};

/// Reply cap per turn so the whole run stays short.
const BENCH_MAX_NEW: usize = 128;

/// Realistic multi-turn script (context grows like a real chat).
const SCRIPT: &[&str] = &[
    "hi, who are you?",
    "what can you help me with in one sentence?",
    "ok give me 3 bullet ideas for a weekend project",
    "expand on the second idea a bit more",
    "summarize everything we talked about so far",
];

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

pub fn run(engine: &mut InferenceEngine, prompt: Option<String>, max_new: usize) -> Result<()> {
    let max_new = max_new.min(BENCH_MAX_NEW);
    let script: Vec<String> = match prompt {
        Some(p) => vec![p],
        None => SCRIPT.iter().map(|s| s.to_string()).collect(),
    };
    let (maj, min) = engine.compute_capability();
    eprintln!(
        "bench | model={} | gpu={} (sm_{maj}{min}) | decode={} | weights={:.2} GiB | max_seq={} | turns={} | max_new={max_new}",
        engine.model_id,
        engine.gpu_name(),
        engine.decode().name(),
        engine.weight_gib,
        engine.max_seq,
        script.len()
    );

    let mut opts = SessionOptions::interactive(max_new);
    opts.print_stream = false;
    opts.print_stats = false;
    let mut session = engine.session(opts);

    eprintln!("\n turn  prompt  prefill  gen   ttft ms  decode tok/s  prefill tok/s  ctx");
    let (mut decode, mut prefill, mut ttft) = (Vec::new(), Vec::new(), Vec::new());
    let (mut gen_total, mut wall_ms) = (0usize, 0.0f64);
    for (i, user) in script.iter().enumerate() {
        let s = session.turn(user)?;
        eprintln!(
            " {:>4}  {:>6}  {:>7}  {:>3}  {:>7.0}  {:>12.1}  {:>13.1}  {}",
            i + 1,
            s.prompt_tokens,
            s.prefilled_prompt_tokens,
            s.gen_tokens,
            s.ttft_ms,
            s.decode_tps,
            s.prefill_tps,
            s.ctx_len
        );
        if s.gen_tokens > 1 {
            decode.push(s.decode_tps);
        }
        prefill.push(s.prefill_tps);
        ttft.push(s.ttft_ms);
        gen_total += s.gen_tokens;
        wall_ms += s.wall_ms;
    }
    eprintln!(
        "\nsummary | decode median {:.1} tok/s | prefill median {:.1} tok/s | ttft median {:.0} ms | {gen_total} tokens in {:.1} s",
        median(&mut decode),
        median(&mut prefill),
        median(&mut ttft),
        wall_ms / 1000.0
    );
    Ok(())
}
