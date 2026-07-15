//! GPT-2 / Qwen2 byte-level BPE from GGUF metadata.

mod bytes;
mod special;

use bytes::{bpe_chars_to_text, text_to_bpe_chars};
use special::split_special;

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use taraference_gguf::{GgufFile, Value};

pub struct Tokenizer {
    pub tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merges: HashMap<(String, String), u32>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub add_bos: bool,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let tokens = match gguf.metadata.get("tokenizer.ggml.tokens") {
            Some(Value::Array { items, .. }) => items
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(|s| s.to_string())
                        .ok_or_else(|| anyhow!("token not string"))
                })
                .collect::<Result<Vec<_>>>()?,
            _ => bail!("missing tokenizer.ggml.tokens"),
        };

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.entry(t.clone()).or_insert(i as u32);
        }

        let merges = match gguf.metadata.get("tokenizer.ggml.merges") {
            Some(Value::Array { items, .. }) => {
                let mut m = HashMap::with_capacity(items.len());
                for (rank, v) in items.iter().enumerate() {
                    let s = v.as_str().ok_or_else(|| anyhow!("merge not string"))?;
                    let (a, b) = s
                        .split_once(' ')
                        .ok_or_else(|| anyhow!("bad merge: {s}"))?;
                    m.insert((a.to_string(), b.to_string()), rank as u32);
                }
                m
            }
            _ => bail!("missing tokenizer.ggml.merges"),
        };

        let bos_id = gguf
            .meta_u32("tokenizer.ggml.bos_token_id")
            .or_else(|| gguf.meta_u64("tokenizer.ggml.bos_token_id").map(|v| v as u32))
            .unwrap_or(0);
        let eos_id = gguf
            .meta_u32("tokenizer.ggml.eos_token_id")
            .or_else(|| gguf.meta_u64("tokenizer.ggml.eos_token_id").map(|v| v as u32))
            .unwrap_or(0);
        let add_bos = gguf
            .metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Self {
            tokens,
            token_to_id,
            merges,
            bos_id,
            eos_id,
            add_bos,
        })
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_special && self.add_bos {
            ids.push(self.bos_id);
        }
        for piece in split_special(text, &self.token_to_id) {
            if piece.starts_with("<|") && piece.ends_with("|>") {
                if let Some(&id) = self.token_to_id.get(&piece) {
                    ids.push(id);
                    continue;
                }
            }
            ids.extend(self.bpe_encode(&piece));
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids {
            if let Some(t) = self.tokens.get(id as usize) {
                if t.starts_with("<|") && t.ends_with("|>") {
                    s.push_str(t);
                } else {
                    s.push_str(&bpe_chars_to_text(t));
                }
            }
        }
        s
    }

    fn bpe_encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let mapped = text_to_bpe_chars(text);
        let mut word: Vec<String> = mapped.chars().map(|c| c.to_string()).collect();

        loop {
            if word.len() < 2 {
                break;
            }
            let mut best: Option<(usize, u32)> = None;
            for i in 0..word.len() - 1 {
                if let Some(&rank) = self.merges.get(&(word[i].clone(), word[i + 1].clone())) {
                    if best.map(|(_, r)| rank < r).unwrap_or(true) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", word[i], word[i + 1]);
            word[i] = merged;
            word.remove(i + 1);
        }

        let mut ids = Vec::with_capacity(word.len());
        for p in &word {
            if let Some(&id) = self.token_to_id.get(p) {
                ids.push(id);
            } else {
                let mut any = false;
                for ch in p.chars() {
                    if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                        ids.push(id);
                        any = true;
                    }
                }
                if !any {
                    ids.push(self.eos_id);
                }
            }
        }
        ids
    }
}
