//! Split chat special tokens (`<|…|>`) out of raw text.

use std::collections::HashMap;

pub fn split_special(text: &str, vocab: &HashMap<String, u32>) -> Vec<String> {
    let mut specials: Vec<&str> = vocab
        .keys()
        .filter(|t| {
            (t.starts_with("<|") && t.ends_with("|>"))
                // Qwen3 / Qwen3.5 thinking delimiters (not ChatML-style).
                || *t == "<think>"
                || *t == "</think>"
        })
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
