//! Minimal CLIP byte-pair tokenizer, read straight from a Hugging Face `tokenizer.json`.

use anyhow::{anyhow, Context, Result};
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

pub const MAX_TOKENS: usize = 77;
const BOS: i32 = 49406;
const EOS: i32 = 49407;

pub struct ClipTokenizer {
    vocab: HashMap<String, i32>,
    ranks: HashMap<(String, String), usize>,
    split: Regex,
    byte_char: [char; 256],
}

/// GPT-2 / CLIP reversible byte -> printable unicode map.
fn byte_chars() -> [char; 256] {
    let mut keep: Vec<u32> = (b'!' as u32..=b'~' as u32).collect();
    keep.extend(0xA1..=0xAC);
    keep.extend(0xAE..=0xFF);
    let mut out = ['\0'; 256];
    let mut extra = 0u32;
    for b in 0..256u32 {
        out[b as usize] = if keep.contains(&b) {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + extra).unwrap();
            extra += 1;
            c
        };
    }
    out
}

impl ClipTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&text)?;
        let model = &json["model"];
        let vocab = model["vocab"]
            .as_object()
            .ok_or_else(|| anyhow!("tokenizer.json has no vocab"))?
            .iter()
            .map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(0) as i32))
            .collect();
        let mut ranks = HashMap::new();
        for (i, m) in model["merges"].as_array().ok_or_else(|| anyhow!("no merges"))?.iter().enumerate() {
            let (a, b) = match m {
                serde_json::Value::Array(p) => (
                    p[0].as_str().unwrap_or("").to_string(),
                    p[1].as_str().unwrap_or("").to_string(),
                ),
                serde_json::Value::String(s) => {
                    let (a, b) = s.split_once(' ').ok_or_else(|| anyhow!("bad merge {s}"))?;
                    (a.to_string(), b.to_string())
                }
                _ => continue,
            };
            ranks.insert((a, b), i);
        }
        let split = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+")?;
        Ok(Self { vocab, ranks, split, byte_char: byte_chars() })
    }

    fn bpe(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if let Some(last) = parts.last_mut() {
            last.push_str("</w>");
        }
        while parts.len() > 1 {
            let best = (0..parts.len() - 1)
                .filter_map(|i| {
                    self.ranks
                        .get(&(parts[i].clone(), parts[i + 1].clone()))
                        .map(|&r| (r, i))
                })
                .min();
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts[i] = merged;
            parts.remove(i + 1);
        }
        parts
    }

    /// BOS + tokens + EOS, padded with EOS to 77 and truncated if longer.
    pub fn encode(&self, prompt: &str) -> Vec<i32> {
        let normalized = prompt
            .nfc()
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        let mut ids = vec![BOS];
        for m in self.split.find_iter(&normalized) {
            let mapped: String = m.as_str().bytes().map(|b| self.byte_char[b as usize]).collect();
            for piece in self.bpe(&mapped) {
                ids.push(self.vocab.get(&piece).copied().unwrap_or(EOS));
            }
        }
        ids.truncate(MAX_TOKENS - 1);
        ids.push(EOS);
        ids.resize(MAX_TOKENS, EOS);
        ids
    }
}
