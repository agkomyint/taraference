//! Multi-turn chat session (GPU).

use crate::cuda_model::{CudaKv, CudaModel};
use crate::tokenizer::Tokenizer;
use anyhow::Result;
use std::io::{self, Write};
use std::time::Instant;

pub struct Session<'a> {
    model: &'a mut CudaModel,
    tok: &'a Tokenizer,
    cache: CudaKv,
    pub max_new: usize,
    system: String,
    primed: bool,
    turn_n: u32,
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
            turn_n: 0,
        })
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

    pub fn turn(&mut self, user: &str) -> Result<String> {
        let wall = Instant::now();
        self.turn_n += 1;
        let turn = self.turn_n;
        let first = !self.primed;

        eprintln!("──────── turn #{turn} ────────");
        eprintln!(
            "[1/5] user said: {:?}",
            if user.len() > 120 {
                format!("{}…", &user[..120])
            } else {
                user.to_string()
            }
        );
        eprintln!(
            "      first_turn={} | ctx_tokens={} / max={} | max_new={}",
            first, self.cache.len, self.cache.max_seq, self.max_new
        );

        let t_build = Instant::now();
        let prompt = self.build_user_prompt(user, first);
        let ids = self.tok.encode(&prompt, false);
        let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;
        if ids.is_empty() {
            anyhow::bail!("tokenizer produced 0 tokens for prompt");
        }
        self.primed = true;

        eprintln!(
            "[2/5] chat template + tokenize: {} prompt tokens in {:.1} ms (includes system on first turn)",
            ids.len(),
            build_ms
        );
        if first {
            eprintln!("      system prompt: {:?}", self.system);
        }

        // Prefill = process all prompt tokens through the model (heavy).
        eprintln!(
            "[3/5] prefill: running {} token(s) on GPU (waiting for first assistant token)…",
            ids.len()
        );
        let t0 = Instant::now();
        let mut next = self.model.forward_greedy(&ids, &mut self.cache)?;
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = wall.elapsed().as_secs_f64() * 1000.0;
        let first_piece = self.tok.decode(&[next]);
        eprintln!(
            "      prefill done in {:.0} ms | time-to-first-token (user→start): {:.0} ms",
            prefill_ms, ttft_ms
        );
        eprintln!(
            "      first_token id={} piece={:?}",
            next, first_piece
        );
        // All stage logs before streaming so they don't mix into assistant text.
        eprintln!(
            "[4/5] decode: streaming up to {} new token(s)…",
            self.max_new
        );
        let _ = io::stderr().flush();

        let mut reply_ids = Vec::new();
        let mut reply = String::new();
        print!("assistant: ");
        let _ = io::stdout().flush();

        let mut stop_reason = String::from("max_new");
        let t1 = Instant::now();
        for _step in 0..self.max_new {
            let piece = self.tok.decode(&[next]);
            if next == self.tok.eos_id
                || piece == "<|im_end|>"
                || piece == "<|endoftext|>"
            {
                stop_reason = format!("end token id={next} piece={piece:?}");
                break;
            }
            if piece == "<|im_start|>" {
                stop_reason = "saw <|im_start|>".into();
                break;
            }

            reply_ids.push(next);
            reply.push_str(&piece);
            print!("{piece}");
            let _ = io::stdout().flush();

            next = self.model.forward_greedy(&[next], &mut self.cache)?;
        }

        // Seal turn for multi-turn KV (best-effort).
        let end = self.tok.encode("<|im_end|>\n", false);
        if !end.is_empty() && self.cache.len + end.len() < self.cache.max_seq {
            let _ = self.model.forward_greedy(&end, &mut self.cache);
        }

        let gen_s = t1.elapsed().as_secs_f64();
        let gen_ms = gen_s * 1000.0;
        let n = reply_ids.len();
        let tps = if n > 0 {
            n as f64 / gen_s.max(1e-6)
        } else {
            0.0
        };
        let total_ms = wall.elapsed().as_secs_f64() * 1000.0;
        println!();
        let _ = io::stdout().flush();

        eprintln!("[5/5] turn complete (stop: {stop_reason})");
        if n == 0 {
            eprintln!(
                "      empty reply — last_id={} eos={} | prefill {:.0} ms | total {:.0} ms | ctx {}",
                next, self.tok.eos_id, prefill_ms, total_ms, self.cache.len
            );
        } else {
            eprintln!(
                "      generated {} token(s) | decode {:.0} ms ({:.1} tok/s)",
                n, gen_ms, tps
            );
            eprintln!(
                "      timing: tokenize {:.1} ms | prefill {:.0} ms | decode {:.0} ms",
                build_ms, prefill_ms, gen_ms
            );
            eprintln!(
                "      time to first token: {:.0} ms | full reply (user→done): {:.0} ms ({:.2} s) | ctx {}",
                ttft_ms,
                total_ms,
                total_ms / 1000.0,
                self.cache.len
            );
            let preview = if reply.len() > 160 {
                format!("{}…", &reply[..160])
            } else {
                reply.clone()
            };
            eprintln!("      reply preview: {preview:?}");
        }
        eprintln!("────────────────────────");
        Ok(reply)
    }

    pub fn run_repl(&mut self) -> Result<()> {
        println!(
            "taraference CUDA | {} | type message, /quit /reset",
            self.model.cfg.architecture
        );
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
                self.turn_n = 0;
                println!("(session reset)");
                continue;
            }
            match self.turn(line) {
                Ok(_) => {}
                Err(e) => eprintln!("error: {e:#}"),
            }
        }
        Ok(())
    }
}
