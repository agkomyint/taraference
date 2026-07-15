//! GPT-2 / Qwen byte ↔ unicode maps for byte-level BPE.

use std::collections::HashMap;
use std::sync::OnceLock;

/// GPT-2 `bytes_to_unicode`: each UTF-8 byte → unique unicode char.
pub fn bytes_to_unicode() -> &'static [char; 256] {
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

/// UTF-8 text → GPT-2 BPE unicode string.
pub fn text_to_bpe_chars(text: &str) -> String {
    let table = bytes_to_unicode();
    text.as_bytes().iter().map(|&b| table[b as usize]).collect()
}

/// BPE unicode string → UTF-8 text.
pub fn bpe_chars_to_text(s: &str) -> String {
    let u2b = unicode_to_bytes();
    let mut bytes = Vec::with_capacity(s.len());
    for ch in s.chars() {
        if let Some(&b) = u2b.get(&ch) {
            bytes.push(b);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}
