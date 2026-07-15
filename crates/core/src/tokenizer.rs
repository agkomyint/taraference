//! GPT-2 / Qwen2 byte-level BPE from GGUF metadata.

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::sync::OnceLock;
use taraference_gguf::{GgufFile, Value};

pub struct Tokenizer {
    pub tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merges: HashMap<(String, String), u32>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub add_bos: bool,
}

/// GPT-2 `bytes_to_unicode`: maps each UTF-8 byte 0..255 to a unique unicode char
/// so BPE never sees raw control characters (space→Ġ, newline→Ċ, …).
fn bytes_to_unicode() -> &'static [char; 256] {
    static TABLE: OnceLock<[char; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut bs: Vec<u8> = (b'!'..=b'~').collect();
        bs.extend(0xA1u8..=0xACu8);
        bs.extend(0xAEu8..=0xFFu8);
        let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
        let mut n = 0u32;
        for b in 0u8..=255 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut table = ['\0'; 256];
        for (i, &b) in bs.iter().enumerate() {
            table[b as usize] = char::from_u32(cs[i]).unwrap();
        }
        table
    })
}

fn unicode_to_bytes() -> &'static HashMap<char, u8> {
    static TABLE: OnceLock<HashMap<char, u8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let b2u = bytes_to_unicode();
        let mut m = HashMap::with_capacity(256);
        for (b, &ch) in b2u.iter().enumerate() {
            m.insert(ch, b as u8);
        }
        m
    })
}

/// Map UTF-8 text → GPT-2 unicode string used by BPE merges / vocab.
fn text_to_bpe_chars(text: &str) -> String {
    let table = bytes_to_unicode();
    text.as_bytes().iter().map(|&b| table[b as usize]).collect()
}

/// Inverse: BPE unicode string → UTF-8 bytes → lossy String.
fn bpe_chars_to_text(s: &str) -> String {
    let u2b = unicode_to_bytes();
    let mut bytes = Vec::with_capacity(s.len());
    for ch in s.chars() {
        if let Some(&b) = u2b.get(&ch) {
            bytes.push(b);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
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

        // Split out special tokens like <|im_start|>, then BPE the rest.
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
                // Special tokens are stored as plain text in the vocab.
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
        // Byte-level: UTF-8 → GPT-2 unicode chars, then greedy BPE merges.
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
                // Rare: unmapped piece → per-char fallback, else eos.
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

fn split_special(text: &str, vocab: &HashMap<String, u32>) -> Vec<String> {
    let mut specials: Vec<&str> = vocab
        .keys()
        .filter(|t| t.starts_with("<|") && t.ends_with("|>"))
        .map(|s| s.as_str())
        .collect();
    specials.sort_by_key(|s| std::cmp::Reverse(s.len()));

    let mut out = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = text.chars().collect();
    while i < chars.len() {
        let rest: String = chars[i..].iter().collect();
        let mut hit = None;
        for s in &specials {
            if rest.starts_with(s) {
                hit = Some(*s);
                break;
            }
        }
        if let Some(s) = hit {
            out.push(s.to_string());
            i += s.chars().count();
        } else {
            let start = i;
            i += 1;
            while i < chars.len() {
                let rest: String = chars[i..].iter().collect();
                if specials.iter().any(|s| rest.starts_with(s)) {
                    break;
                }
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        }
    }
    out
}
