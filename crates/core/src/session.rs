//! Multi-turn chat session (GPU).

use crate::cuda::{CudaKv, CudaModel};
use crate::sampler::{sample_logits, SamplingOptions};
use crate::tokenizer::Tokenizer;
use anyhow::Result;
use std::io::{self, Write};
use std::time::Instant;

/// Why generation stopped this turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    MaxNew,
    Empty,
}

/// Timing / token counts from one generation.
#[derive(Debug, Clone)]
pub struct TurnStats {
    pub reply: String,
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_tps: f64,
    pub prefill_tps: f64,
    /// Time from start until first assistant token is ready (tokenize + prefill).
    pub ttft_ms: f64,
    pub tokenize_ms: f64,
    pub ctx_before: usize,
    pub ctx_len: usize,
    pub max_seq: usize,
    pub max_new: usize,
    pub wall_ms: f64,
    pub stop: StopReason,
    pub hit_max_new: bool,
    pub first: bool,
}

/// Session behaviour (CLI streams tokens; server stays quiet).
#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub max_new: usize,
    pub system: String,
    /// Print tokens to stdout as they are generated.
    pub print_stream: bool,
    /// Print the `[n tok | …]` line after a turn.
    pub print_stats: bool,
    /// Qwen3 / Qwen3.5 thinking mode. When true, the generation prompt opens
    /// an unclosed `<think>` block so the model writes chain-of-thought.
    /// When false (default, matches Qwen3.5-0.8B/2B/4B/9B), an empty
    /// `<think></think>` pair is prefilled so the model answers directly.
    pub enable_thinking: bool,
    /// Zero means greedy GPU argmax. Positive values enable host quality sampling.
    pub temperature: f32,
    /// Nucleus probability used when sampling.
    pub top_p: f32,
    /// Preselect this many highest-logit candidates before nucleus sampling.
    pub top_k: usize,
    /// Hugging Face-style penalty applied once to tokens already in the context.
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            max_new: 512,
            system: "You are a helpful assistant.".into(),
            print_stream: true,
            print_stats: true,
            enable_thinking: false,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 256,
            repetition_penalty: 1.0,
            seed: 42,
        }
    }
}

impl SessionOptions {
    pub fn quiet(max_new: usize) -> Self {
        Self {
            max_new,
            print_stream: false,
            print_stats: false,
            ..Self::default()
        }
    }

    pub fn interactive(max_new: usize) -> Self {
        Self {
            max_new,
            print_stream: true,
            print_stats: true,
            ..Self::default()
        }
    }
}

/// Generation session over a borrowed model + KV.
pub struct Session<'a> {
    model: &'a mut CudaModel,
    tok: &'a Tokenizer,
    cache: &'a mut CudaKv,
    pub max_new: usize,
    system: String,
    print_stream: bool,
    print_stats: bool,
    enable_thinking: bool,
    sampling: SamplingOptions,
    rng_state: u64,
    primed: bool,
    /// Tokens whose KV entries are committed. Used as the draft source for
    /// prompt-lookup decoding; it never changes model acceptance semantics.
    history: Vec<u32>,
}

impl<'a> Session<'a> {
    /// Bind to an existing KV arena (engine owns allocation).
    pub fn with_kv(
        model: &'a mut CudaModel,
        tok: &'a Tokenizer,
        cache: &'a mut CudaKv,
        opts: SessionOptions,
    ) -> Self {
        let primed = cache.len > 0;
        Self {
            model,
            tok,
            cache,
            max_new: opts.max_new,
            system: opts.system,
            print_stream: opts.print_stream,
            print_stats: opts.print_stats,
            enable_thinking: opts.enable_thinking,
            sampling: SamplingOptions {
                temperature: opts.temperature,
                top_p: opts.top_p,
                top_k: opts.top_k,
                repetition_penalty: opts.repetition_penalty,
            },
            rng_state: opts.seed.max(1),
            primed,
            history: Vec::new(),
        }
    }

    pub fn max_seq(&self) -> usize {
        self.cache.max_seq
    }

    pub fn ctx_len(&self) -> usize {
        self.cache.len
    }

    pub fn reset(&mut self) {
        self.cache.clear();
        self.history.clear();
        self.primed = false;
        // Hybrid (Qwen3.5) keeps GDN/conv state in the KV arena; wipe it too.
        let _ = self.model.zero_recurrent(self.cache);
    }

    /// Find a continuation that followed the longest recent token suffix at an
    /// earlier position. The model still verifies every proposed token.
    fn prompt_lookup_draft(&self, current: u32, limit: usize) -> Vec<u32> {
        const MAX_NGRAM: usize = 8;
        const MAX_DRAFT: usize = 8;
        if limit < 2 || self.history.len() < 4 {
            return Vec::new();
        }
        let mut context = Vec::with_capacity(self.history.len() + 1);
        context.extend_from_slice(&self.history);
        context.push(current);
        for n in (2..=MAX_NGRAM.min(context.len())).rev() {
            let suffix_start = context.len() - n;
            let suffix = &context[suffix_start..];
            if self.history.len() <= n {
                continue;
            }
            for pos in (0..=self.history.len() - n).rev() {
                if &self.history[pos..pos + n] != suffix || pos + n >= self.history.len() {
                    continue;
                }
                let end = (pos + n + MAX_DRAFT.min(limit)).min(self.history.len());
                let mut draft = Vec::new();
                for &id in &self.history[pos + n..end] {
                    let piece = self.tok.decode(&[id]);
                    if id == self.tok.eos_id || piece.starts_with("<|") {
                        break;
                    }
                    draft.push(id);
                }
                if draft.len() >= 2 {
                    return draft;
                }
            }
        }
        Vec::new()
    }

    fn build_user_prompt(&self, user: &str, first: bool) -> String {
        if std::env::var_os("TARAFER_TARA141_SFT").is_some() {
            return format!(
                "<|system|>{}<|/system|>\n<|user|>{user}<|/user|>\n<|assistant|>",
                self.system
            );
        }
        let mut s = String::new();
        if first {
            s.push_str("<|im_start|>system\n");
            s.push_str(&self.system);
            s.push_str("<|im_end|>\n");
        }
        s.push_str("<|im_start|>user\n");
        s.push_str(user);
        s.push_str("<|im_end|>\n");
        s.push_str(crate::chat::assistant_generation_prompt(self.enable_thinking));
        s
    }

    /// Incremental multi-turn user message (CLI chat).
    pub fn turn(&mut self, user: &str) -> Result<TurnStats> {
        // Tara 1.4.1 SFT is a one-shot Alpaca-style model, not ChatML multi-turn.
        if std::env::var_os("TARAFER_TARA141_SFT").is_some() {
            self.reset();
        }
        let first = !self.primed;
        let prompt = self.build_user_prompt(user, first);
        self.primed = true;
        self.generate_with(&prompt, first, |_| {})
    }

    /// One-shot: full ChatML (or other) prompt → assistant continuation.
    /// Clears the KV cache first (stateless request).
    pub fn complete_prompt(&mut self, prompt: &str) -> Result<TurnStats> {
        self.complete_prompt_stream(prompt, |_| {})
    }

    /// Like [`Self::complete_prompt`], calling `on_token` for each decoded piece.
    pub fn complete_prompt_stream<F>(&mut self, prompt: &str, on_token: F) -> Result<TurnStats>
    where
        F: FnMut(&str),
    {
        self.reset();
        self.primed = true;
        self.generate_with(prompt, true, on_token)
    }

    fn generate_with<F>(&mut self, prompt: &str, first: bool, mut on_token: F) -> Result<TurnStats>
    where
        F: FnMut(&str),
    {
        let wall = Instant::now();
        let ctx_before = self.cache.len;

        let t_tok = Instant::now();
        let ids = self.tok.encode(prompt, false);
        let tokenize_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
        if ids.is_empty() {
            anyhow::bail!("tokenizer produced 0 tokens for prompt");
        }
        let prompt_tokens = ids.len();

        let t0 = Instant::now();
        let greedy = self.model.forward_greedy(&ids, self.cache)?;
        self.history.extend_from_slice(&ids);
        let mut next = self.choose_next(greedy)?;
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = wall.elapsed().as_secs_f64() * 1000.0;

        let mut reply_ids = Vec::new();
        let mut reply = String::new();
        if self.print_stream {
            print!("assistant: ");
            let _ = io::stdout().flush();
        }

        let mut stop = StopReason::MaxNew;
        let t1 = Instant::now();
        let mut step = 0usize;
        let mut pld_proposed = 0usize;
        let mut pld_accepted = 0usize;
        let mut pld_passes = 0usize;
        while step < self.max_new {
            let piece = self.tok.decode(&[next]);
            // Speed benches for under-trained Sprint models: TARAFER_IGNORE_EOS=1
            // forces max_new generation (do not use for quality claims).
            let ignore_eos = std::env::var_os("TARAFER_IGNORE_EOS").is_some();
            if !ignore_eos
                && (next == self.tok.eos_id
                    || piece == "<|im_end|>"
                    || piece == "<|endoftext|>"
                    || piece == "<|/assistant|>"
                    || piece == "<|assistant|>"
                    || piece == "<|im_start|>")
            {
                stop = if step == 0 && reply_ids.is_empty() {
                    StopReason::Empty
                } else {
                    StopReason::Eos
                };
                break;
            }
            reply_ids.push(next);
            reply.push_str(&piece);
            if self.print_stream {
                print!("{piece}");
                let _ = io::stdout().flush();
            }
            on_token(&piece);
            step += 1;
            if step >= self.max_new {
                stop = StopReason::MaxNew;
                break;
            }

            let draft = if std::env::var_os("TARAFER_PLD").is_some() {
                self.prompt_lookup_draft(next, self.max_new - step)
            } else {
                Vec::new()
            };
            if draft.is_empty() {
                let greedy = self.model.forward_greedy(&[next], self.cache)?;
                self.history.push(reply_ids[reply_ids.len() - 1]);
                next = self.choose_next(greedy)?;
                continue;
            }

            let old_len = self.cache.len;
            let mut verify = Vec::with_capacity(draft.len() + 1);
            verify.push(next);
            verify.extend_from_slice(&draft);
            let predictions = self.model.forward_greedy_all(&verify, self.cache)?;
            pld_passes += 1;
            pld_proposed += draft.len();
            let mut accepted = 0usize;
            while accepted < draft.len() && predictions[accepted] == draft[accepted] {
                accepted += 1;
            }
            pld_accepted += accepted;
            self.cache.len = old_len + 1 + accepted;
            self.history.push(next);
            self.history.extend_from_slice(&draft[..accepted]);

            for &id in &draft[..accepted] {
                if step >= self.max_new {
                    break;
                }
                let piece = self.tok.decode(&[id]);
                reply_ids.push(id);
                reply.push_str(&piece);
                if self.print_stream {
                    print!("{piece}");
                    let _ = io::stdout().flush();
                }
                on_token(&piece);
                step += 1;
            }
            if step >= self.max_new {
                stop = StopReason::MaxNew;
                break;
            }
            next = predictions[accepted];
        }

        // Decode throughput covers accepted/generated tokens only. The ChatML
        // end marker below is a post-generation KV maintenance pass and must
        // not be charged to decode_ms when its tokens are not counted in `n`.
        let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // Keep multi-turn KV consistent with ChatML (append end marker).
        let end = self.tok.encode("<|im_end|>\n", false);
        if !end.is_empty() && self.cache.len + end.len() < self.cache.max_seq {
            let _ = self.model.forward_greedy(&end, self.cache);
            self.history.extend_from_slice(&end);
        }

        let n = reply_ids.len();
        let gen_s = decode_ms / 1000.0;
        let decode_tps = if n > 0 {
            n as f64 / gen_s.max(1e-6)
        } else {
            0.0
        };
        let prefill_tps = if prompt_tokens > 0 {
            prompt_tokens as f64 / (prefill_ms / 1000.0).max(1e-6)
        } else {
            0.0
        };
        let hit_max_new = stop == StopReason::MaxNew && n > 0;
        if n == 0 {
            stop = StopReason::Empty;
        }

        if self.print_stream {
            println!();
        }
        if self.print_stats {
            if pld_passes != 0 {
                eprintln!(
                    "PLD | passes={pld_passes} proposed={pld_proposed} accepted={pld_accepted} rate={:.0}%",
                    100.0 * pld_accepted as f64 / pld_proposed.max(1) as f64
                );
            }
            eprintln!(
                "[{n} tok | prefill {prefill_ms:.0} ms | decode {decode_tps:.1} tok/s | ctx {} | stop={stop:?}]",
                self.cache.len
            );
        }

        Ok(TurnStats {
            reply,
            prompt_tokens,
            gen_tokens: n,
            prefill_ms,
            decode_ms,
            decode_tps,
            prefill_tps,
            ttft_ms,
            tokenize_ms,
            ctx_before,
            ctx_len: self.cache.len,
            max_seq: self.cache.max_seq,
            max_new: self.max_new,
            wall_ms: wall.elapsed().as_secs_f64() * 1000.0,
            stop,
            hit_max_new,
            first,
        })
    }

    pub fn run_repl(&mut self) -> Result<()> {
        println!("ready — type a message (/quit /reset)");
        let stdin = io::stdin();
        loop {
            print!("\nuser: ");
            let _ = io::stdout().flush();
            let mut line = String::new();
            if stdin.read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line == "/quit" || line == "/exit" {
                break;
            }
            if line == "/reset" {
                self.reset();
                println!("(reset)");
                continue;
            }
            if let Err(e) = self.turn(line) {
                eprintln!("error: {e:#}");
            }
        }
        Ok(())
    }
}

impl Session<'_> {
    fn choose_next(&mut self, greedy: u32) -> Result<u32> {
        if self.sampling.temperature <= 0.0 {
            return Ok(greedy);
        }
        let logits = self.model.current_logits()?;
        Ok(sample_logits(
            logits,
            &self.history,
            self.sampling,
            &mut self.rng_state,
        ))
    }
}
