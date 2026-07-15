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

/// Timing / token counts from one `turn`.
#[derive(Debug, Clone)]
pub struct TurnStats {
    pub reply: String,
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_tps: f64,
    pub prefill_tps: f64,
    /// Time from start of turn until first assistant token is ready (tokenize + prefill).
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

pub struct Session<'a> {
    model: &'a mut CudaModel,
    tok: &'a Tokenizer,
    cache: CudaKv,
    pub max_new: usize,
    system: String,
    primed: bool,
}

impl<'a> Session<'a> {
    pub fn new(
        model: &'a mut CudaModel,
        tok: &'a Tokenizer,
        max_seq: usize,
        max_new: usize,
    ) -> Result<Self> {
        let cache = model.alloc_kv(max_seq)?;
        Ok(Self {
            model,
            tok,
            cache,
            max_new,
            system: "You are a helpful assistant.".into(),
            primed: false,
        })
    }

    pub fn max_seq(&self) -> usize {
        self.cache.max_seq
    }

    pub fn ctx_len(&self) -> usize {
        self.cache.len
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

    pub fn turn(&mut self, user: &str) -> Result<TurnStats> {
        let wall = Instant::now();
        let first = !self.primed;
        let ctx_before = self.cache.len;

        let t_tok = Instant::now();
        let prompt = self.build_user_prompt(user, first);
        let ids = self.tok.encode(&prompt, false);
        let tokenize_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
        if ids.is_empty() {
            anyhow::bail!("tokenizer produced 0 tokens for prompt");
        }
        let prompt_tokens = ids.len();
        self.primed = true;

        let t0 = Instant::now();
        let mut next = self.model.forward_greedy(&ids, &mut self.cache)?;
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = wall.elapsed().as_secs_f64() * 1000.0;

        let mut reply_ids = Vec::new();
        let mut reply = String::new();
        print!("assistant: ");
        let _ = io::stdout().flush();

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
            print!("{piece}");
            let _ = io::stdout().flush();
            next = self.model.forward_greedy(&[next], &mut self.cache)?;
            if step + 1 == self.max_new {
                stop = StopReason::MaxNew;
            }
        }

        let end = self.tok.encode("<|im_end|>\n", false);
        if !end.is_empty() && self.cache.len + end.len() < self.cache.max_seq {
            let _ = self.model.forward_greedy(&end, &mut self.cache);
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

        println!();
        eprintln!(
            "[{n} tok | prefill {prefill_ms:.0} ms | decode {decode_tps:.1} tok/s | ctx {} | stop={stop:?}]",
            self.cache.len
        );

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
                self.cache.clear();
                self.primed = false;
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
