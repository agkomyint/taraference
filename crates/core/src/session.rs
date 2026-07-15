//! Multi-turn chat session (GPU).

use crate::cuda::{CudaKv, CudaModel};
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
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            max_new: 512,
            system: "You are a helpful assistant.".into(),
            print_stream: true,
            print_stats: true,
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
    primed: bool,
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
            primed,
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
        self.primed = false;
    }

    fn build_user_prompt(&self, user: &str, first: bool) -> String {
        let mut s = String::new();
        if first {
            s.push_str("<|im_start|>system\n");
            s.push_str(&self.system);
            s.push_str("<|im_end|>\n");
        }
        s.push_str("<|im_start|>user\n");
        s.push_str(user);
        s.push_str("<|im_end|>\n");
        s.push_str("<|im_start|>assistant\n");
        s
    }

    /// Incremental multi-turn user message (CLI chat).
    pub fn turn(&mut self, user: &str) -> Result<TurnStats> {
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
        let mut next = self.model.forward_greedy(&ids, self.cache)?;
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
        for step in 0..self.max_new {
            let piece = self.tok.decode(&[next]);
            if next == self.tok.eos_id
                || piece == "<|im_end|>"
                || piece == "<|endoftext|>"
                || piece == "<|im_start|>"
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
            next = self.model.forward_greedy(&[next], self.cache)?;
            if step + 1 == self.max_new {
                stop = StopReason::MaxNew;
            }
        }

        // Keep multi-turn KV consistent with ChatML (append end marker).
        let end = self.tok.encode("<|im_end|>\n", false);
        if !end.is_empty() && self.cache.len + end.len() < self.cache.max_seq {
            let _ = self.model.forward_greedy(&end, self.cache);
        }

        let n = reply_ids.len();
        let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;
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
