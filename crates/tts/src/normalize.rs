//! Text normalisation shared by every TTS engine.
//!
//! The Burmese models only know Burmese characters. This pass makes real-world text speakable:
//! digits become Burmese number words, a user lexicon maps loanwords and names to Burmese
//! spelling, and whatever cannot be spoken (emoji, unknown Latin words) is removed and reported
//! so nothing disappears silently.

use anyhow::{Context, Result};
use regex::Regex;
use std::path::Path;

const DIGITS: [&str; 10] = ["သုည", "တစ်", "နှစ်", "သုံး", "လေး", "ငါး", "ခြောက်", "ခုနစ်", "ရှစ်", "ကိုး"];
// (value, word, form used when more digits follow)
const SCALES: [(u64, &str, &str); 6] = [
    (1_000_000, "သန်း", "သန်း"),
    (100_000, "သိန်း", "သိန်း"),
    (10_000, "သောင်း", "သောင်း"),
    (1_000, "ထောင်", "ထောင့်"),
    (100, "ရာ", "ရာ့"),
    (10, "ဆယ်", "ဆယ့်"),
];

/// User word list: `Key=Burmese` per line, `#` for comments. Latin keys match case-insensitively.
#[derive(Debug, Default, Clone)]
pub struct Lexicon {
    entries: Vec<(String, String)>,
}

impl Lexicon {
    pub fn parse(text: &str) -> Self {
        let mut entries: Vec<(String, String)> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .filter(|(k, _)| !k.is_empty())
            .collect();
        // Longest key first so "Min Buri" wins over "Min".
        entries.sort_by_key(|(k, _)| std::cmp::Reverse(k.chars().count()));
        Self { entries }
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read lexicon {}", path.display()))?;
        Ok(Self::parse(&text))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn apply(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (key, value) in &self.entries {
            let pattern = format!("(?i){}", regex::escape(key));
            if let Ok(re) = Regex::new(&pattern) {
                out = re.replace_all(&out, regex::NoExpand(value)).into_owned();
            }
        }
        out
    }
}

#[derive(Debug, Clone, Default)]
pub struct Normalized {
    pub text: String,
    /// Latin words with no lexicon entry; they were removed from the spoken text.
    pub unknown_words: Vec<String>,
    /// Count of symbols/emoji removed.
    pub dropped_symbols: usize,
}

/// Burmese words for a whole number. Values of 10^9 and above are read digit by digit.
pub fn number_words(n: u64) -> String {
    if n == 0 {
        return DIGITS[0].to_string();
    }
    if n >= 1_000_000_000 {
        return n.to_string().bytes().map(|b| DIGITS[(b - b'0') as usize]).collect();
    }
    if n < 10 {
        return DIGITS[n as usize].to_string();
    }
    for (scale, word, joined) in SCALES {
        if n >= scale {
            let (q, r) = (n / scale, n % scale);
            let head = number_words(q);
            return if r == 0 {
                format!("{head}{word}")
            } else {
                format!("{head}{joined}{}", number_words(r))
            };
        }
    }
    unreachable!("n >= 10 always matches a scale")
}

fn to_ascii_digits(s: &str) -> String {
    s.chars()
        .filter_map(|c| match c {
            '0'..='9' => Some(c),
            '၀'..='၉' => char::from_u32('0' as u32 + (c as u32 - '၀' as u32)),
            _ => None,
        })
        .collect()
}

fn read_number(token: &str) -> String {
    let (int_part, frac_part) = match token.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (token, None),
    };
    let int_digits = to_ascii_digits(int_part);
    let mut out = match int_digits.parse::<u64>() {
        Ok(n) => number_words(n),
        Err(_) => int_digits.bytes().map(|b| DIGITS[(b - b'0') as usize]).collect(),
    };
    if let Some(frac) = frac_part {
        out.push_str("ဒသမ");
        for b in to_ascii_digits(frac).bytes() {
            out.push_str(DIGITS[(b - b'0') as usize]);
        }
    }
    out
}

fn is_burmese(c: char) -> bool {
    ('\u{1000}'..='\u{109F}').contains(&c) || ('\u{AA60}'..='\u{AA7F}').contains(&c)
}

pub fn normalize(text: &str, lexicon: &Lexicon) -> Normalized {
    let mut s = lexicon.apply(text);

    s = s.replace('%', "ရာခိုင်နှုန်း");
    let number = Regex::new(r"[0-9၀-၉]+(?:,[0-9]{3})*(?:\.[0-9၀-၉]+)?").expect("number regex");
    s = number
        .replace_all(&s, |caps: &regex::Captures| read_number(&caps[0].replace(',', "")))
        .into_owned();

    let latin = Regex::new(r"[A-Za-z][A-Za-z'’]*").expect("latin regex");
    let mut unknown_words: Vec<String> = Vec::new();
    for m in latin.find_iter(&s) {
        let w = m.as_str().to_string();
        if !unknown_words.contains(&w) {
            unknown_words.push(w);
        }
    }
    s = latin.replace_all(&s, " ").into_owned();

    let mut dropped_symbols = 0;
    let kept: String = s
        .chars()
        .filter(|&c| {
            let keep = is_burmese(c) || c.is_whitespace() || matches!(c, ',' | '.' | '!' | '?' | ';' | ':' | '-' | '\'' | '"' | '(' | ')');
            if !keep {
                dropped_symbols += 1;
            }
            keep
        })
        .collect();
    let text = kept
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    Normalized { text, unknown_words, dropped_symbols }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_numbers() {
        assert_eq!(number_words(0), "သုည");
        assert_eq!(number_words(7), "ခုနစ်");
        assert_eq!(number_words(10), "တစ်ဆယ်");
        assert_eq!(number_words(11), "တစ်ဆယ့်တစ်");
        assert_eq!(number_words(25), "နှစ်ဆယ့်ငါး");
        assert_eq!(number_words(100), "တစ်ရာ");
        assert_eq!(number_words(2024), "နှစ်ထောင့်နှစ်ဆယ့်လေး");
    }

    #[test]
    fn digits_inside_text_and_burmese_digits() {
        let n = normalize("7နာရီခွဲ ၅ ယောက် 3.5", &Lexicon::default());
        assert_eq!(n.text, "ခုနစ်နာရီခွဲ ငါး ယောက် သုံးဒသမငါး");
    }

    #[test]
    fn lexicon_and_unknown_words() {
        let lex = Lexicon::parse("# c\nclient=ကလိုင်းယင့်\nMin Buri=မင်းဘူရီ");
        let n = normalize("Client တွေ 👻 Min Buri Respect", &lex);
        assert_eq!(n.text, "ကလိုင်းယင့် တွေ မင်းဘူရီ");
        assert_eq!(n.unknown_words, vec!["Respect".to_string()]);
        assert_eq!(n.dropped_symbols, 1);
    }
}
