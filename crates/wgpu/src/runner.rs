//! Shared generation and benchmark frontend for the standalone and main CLIs.
use crate::WgpuModel;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::{io::Write, path::Path, time::Instant};
use taraference_core::{
    chat::{format_chatml_for_model, ChatMessage},
    sampler::{sample_logits, SamplingOptions},
    tokenizer::Tokenizer,
};
use taraference_gguf::GgufFile;

#[derive(Debug, Serialize, Deserialize)]
pub struct Generation {
    pub model: String,
    pub backend: String,
    pub adapter: String,
    pub reply: String,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub load_ms: f64,
    pub ttft_ms: f64,
    pub prefill_tps: f64,
    pub decode_tps: f64,
    pub wall_ms: f64,
}

pub fn prompt_tokens(file: &GgufFile, prompt: &str, raw: bool) -> Result<(Tokenizer, Vec<u32>)> {
    let tok = Tokenizer::from_gguf(file)?;
    let text = if raw {
        prompt.to_owned()
    } else {
        format_chatml_for_model(
            &[ChatMessage::user(prompt)],
            Some("You are a helpful assistant."),
            false,
            false,
        )
    };
    let tokens = tok.encode(&text, true);
    ensure!(!tokens.is_empty(), "empty tokenized prompt");
    Ok((tok, tokens))
}

/// Latency includes readback and sampling. Decode throughput excludes the first
/// token, whose computation belongs to prefill/TTFT.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &mut WgpuModel,
    tok: &Tokenizer,
    prompt: &[u32],
    count: usize,
    sampling: SamplingOptions,
    seed: u64,
    stop_eos: bool,
    stream: bool,
) -> Result<Generation> {
    ensure!(
        count > 0 && prompt.len().saturating_add(count - 1) <= model.max_seq,
        "prompt + generation exceeds context (or zero output requested)"
    );
    model.clear();
    let started = Instant::now();
    let mut token = model.forward_greedy(prompt)?;
    let ttft_ms = started.elapsed().as_secs_f64() * 1000.;
    let mut history = prompt.to_vec();
    let mut rng = seed;
    let mut generated = Vec::new();
    let mut reply = String::new();
    let decode_started = Instant::now();
    for i in 0..count {
        if i > 0 {
            token = model.forward_greedy(&[generated[i - 1]])?;
        }
        if sampling.temperature > 0.0 || sampling.repetition_penalty > 1.0 {
            let logits = model
                .current_logits()?
                .into_iter()
                .enumerate()
                .map(|(id, v)| (id as u32, v))
                .collect();
            token = sample_logits(logits, &history, sampling, &mut rng);
        }
        if stop_eos
            && (token == tok.eos_id
                || matches!(
                    tok.tokens.get(token as usize).map(String::as_str),
                    Some("<|im_end|>" | "<|endoftext|>" | "<|im_start|>")
                ))
        {
            break;
        }
        generated.push(token);
        history.push(token);
        // Use the same tokenizer decoding as the CUDA frontend.
        let text = tok.decode(&generated);
        if stream && text.starts_with(&reply) {
            print!("{}", &text[reply.len()..]);
            std::io::stdout().flush()?;
        }
        reply = text;
    }
    let decode_s = decode_started.elapsed().as_secs_f64();
    let wall_ms = started.elapsed().as_secs_f64() * 1000.;
    if stream {
        println!();
    }
    let gen_tokens = generated.len();
    Ok(Generation {
        model: String::new(),
        backend: "wgpu".into(),
        adapter: model.adapter().into(),
        reply,
        token_ids: generated,
        prompt_tokens: prompt.len(),
        gen_tokens,
        load_ms: 0.,
        ttft_ms,
        prefill_tps: prompt.len() as f64 / (ttft_ms / 1000.).max(1e-9),
        decode_tps: gen_tokens.saturating_sub(1) as f64 / decode_s.max(1e-9),
        wall_ms,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn generate_file(
    path: &Path,
    prompt: &str,
    raw: bool,
    count: usize,
    context: usize,
    sampling: SamplingOptions,
    seed: u64,
    stream: bool,
) -> Result<Generation> {
    let load = Instant::now();
    let file = GgufFile::open(path)?;
    let (tok, ids) = prompt_tokens(&file, prompt, raw)?;
    let mut model = WgpuModel::load(&file, context)?;
    let load_ms = load.elapsed().as_secs_f64() * 1000.;
    let mut result = generate(&mut model, &tok, &ids, count, sampling, seed, true, stream)?;
    result.model = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    result.load_ms = load_ms;
    Ok(result)
}
