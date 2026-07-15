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
    scores: Vec<f32>,
    sentencepiece: bool,
    max_piece_bytes: usize,
    pub bos_id: u32,
    pub eos_id: u32,
    pub unk_id: u32,
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

        let (merges, sentencepiece) = match gguf.metadata.get("tokenizer.ggml.merges") {
            Some(Value::Array { items, .. }) => {
                let mut m = HashMap::with_capacity(items.len());
                for (rank, v) in items.iter().enumerate() {
                    let s = v.as_str().ok_or_else(|| anyhow!("merge not string"))?;
                    let (a, b) = s
                        .split_once(' ')
                        .ok_or_else(|| anyhow!("bad merge: {s}"))?;
                    m.insert((a.to_string(), b.to_string()), rank as u32);
                }
                (m, false)
            }
            // Llama/SentencePiece GGUFs store vocabulary scores instead of an
            // explicit GPT-2 merge table.
            _ => (HashMap::new(), true),
        };
        let scores = match gguf.metadata.get("tokenizer.ggml.scores") {
            Some(Value::Array { items, .. }) => items
                .iter()
                .map(|v| v.as_f32().unwrap_or(-100.0))
                .collect(),
            _ => vec![0.0; tokens.len()],
        };
        let max_piece_bytes = tokens.iter().map(|t| t.len()).max().unwrap_or(1);

        let bos_id = gguf
            .meta_u32("tokenizer.ggml.bos_token_id")
            .or_else(|| gguf.meta_u64("tokenizer.ggml.bos_token_id").map(|v| v as u32))
            .unwrap_or(0);
        let eos_id = gguf
            .meta_u32("tokenizer.ggml.eos_token_id")
            .or_else(|| gguf.meta_u64("tokenizer.ggml.eos_token_id").map(|v| v as u32))
            .unwrap_or(0);
        let unk_id = gguf
            .meta_u32("tokenizer.ggml.unknown_token_id")
            .or_else(|| gguf.meta_u64("tokenizer.ggml.unknown_token_id").map(|v| v as u32))
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
            scores,
            sentencepiece,
            max_piece_bytes,
            bos_id,
            eos_id,
            unk_id,
            add_bos,
        })
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_special && self.add_bos {
            ids.push(self.bos_id);
        }
        if self.sentencepiece {
            ids.extend(self.sentencepiece_encode(text));
            return ids;
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
                if self.sentencepiece {
                    if t.starts_with('<') && t.ends_with('>') {
                        continue;
                    }
                    s.push_str(&t.replace('▁', " "));
                } else if t.starts_with("<|") && t.ends_with("|>") {
                    s.push_str(t);
                } else {
                    s.push_str(&bpe_chars_to_text(t));
                }
            }
        }
        s
    }

    /// SentencePiece unigram Viterbi segmentation used by Llama-family GGUFs.
    fn sentencepiece_encode(&self, text: &str) -> Vec<u32> {
        let normalized = format!("▁{}", text.replace(' ', "▁"));
        let n = normalized.len();
        let mut best = vec![f32::NEG_INFINITY; n + 1];
        let mut prev: Vec<Option<(usize, u32)>> = vec![None; n + 1];
        best[0] = 0.0;

        for start in 0..n {
            if !normalized.is_char_boundary(start) || !best[start].is_finite() {
                continue;
            }
            let limit = (start + self.max_piece_bytes).min(n);
            for end in (start + 1)..=limit {
                if !normalized.is_char_boundary(end) {
                    continue;
                }
                if let Some(&id) = self.token_to_id.get(&normalized[start..end]) {
                    let score = *self.scores.get(id as usize).unwrap_or(&0.0);
                    let candidate = best[start] + score;
                    if candidate > best[end] {
                        best[end] = candidate;
                        prev[end] = Some((start, id));
                    }
                }
            }
            // Ensure progress for unusual bytes not represented as pieces.
            let next = normalized[start..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| start + i)
                .unwrap_or(n);
            if best[start] - 100.0 > best[next] {
                best[next] = best[start] - 100.0;
                prev[next] = Some((start, self.unk_id));
            }
        }

        let mut out = Vec::new();
        let mut at = n;
        while at > 0 {
            let Some((p, id)) = prev[at] else {
                return vec![self.unk_id];
            };
            out.push(id);
            at = p;
        }
        out.reverse();
        out
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
