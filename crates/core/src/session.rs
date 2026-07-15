//! Multi-turn chat session (GPU).

use crate::cuda::{CudaKv, CudaModel};
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
        let first = !self.primed;
        let prompt = self.build_user_prompt(user, first);
        let ids = self.tok.encode(&prompt, false);
        if ids.is_empty() {
            anyhow::bail!("tokenizer produced 0 tokens for prompt");
        }
        self.primed = true;

        let t0 = Instant::now();
        let mut next = self.model.forward_greedy(&ids, &mut self.cache)?;
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut reply_ids = Vec::new();
        let mut reply = String::new();
        print!("assistant: ");
        let _ = io::stdout().flush();

        let t1 = Instant::now();
        for _ in 0..self.max_new {
            let piece = self.tok.decode(&[next]);
            if next == self.tok.eos_id
                || piece == "<|im_end|>"
                || piece == "<|endoftext|>"
                || piece == "<|im_start|>"
            {
                break;
            }
            reply_ids.push(next);
            reply.push_str(&piece);
            print!("{piece}");
            let _ = io::stdout().flush();
            next = self.model.forward_greedy(&[next], &mut self.cache)?;
        }

        let end = self.tok.encode("<|im_end|>\n", false);
        if !end.is_empty() && self.cache.len + end.len() < self.cache.max_seq {
            let _ = self.model.forward_greedy(&end, &mut self.cache);
        }

        let n = reply_ids.len();
        let gen_s = t1.elapsed().as_secs_f64();
        let tps = if n > 0 {
            n as f64 / gen_s.max(1e-6)
        } else {
            0.0
        };
        println!();
        eprintln!(
            "[{n} tok | prefill {prefill_ms:.0} ms | decode {tps:.1} tok/s | ctx {}]",
            self.cache.len
        );
        Ok(reply)
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
