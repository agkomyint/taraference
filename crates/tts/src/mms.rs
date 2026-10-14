//! Meta MMS-TTS Burmese (`facebook/mms-tts-mya`, VITS, CC BY-NC 4.0) on ONNX Runtime.
//!
//! One non-autoregressive graph: character ids (blank-interleaved) -> 16 kHz waveform.
//! Files: `model.onnx` and `tokens.txt` (`<char> <id>` per line), e.g. from
//! `willwade/mms-tts-multilingual-models-onnx/mya`.

use crate::{load_session, Device, SynthOptions, SynthResult};
use anyhow::{anyhow, bail, Context, Result};
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;

pub const MMS_SAMPLE_RATE: u32 = 16_000;
pub const MMS_MODEL_DIR: &str = "models/tts/mms-mya";
const MODEL_FILE: &str = "model.onnx";
const TOKENS_FILE: &str = "tokens.txt";

pub fn missing_files(dir: &Path) -> Vec<PathBuf> {
    [MODEL_FILE, TOKENS_FILE].map(|f| dir.join(f)).into_iter().filter(|p| !p.is_file()).collect()
}

pub struct Mms {
    session: ort::session::Session,
    vocab: HashMap<char, i64>,
    pub device: &'static str,
    pub load_ms: f64,
}

impl Mms {
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        let missing = missing_files(dir);
        if !missing.is_empty() {
            let list: Vec<_> = missing.iter().map(|p| p.display().to_string()).collect();
            bail!("MMS-TTS files missing: {}", list.join(", "));
        }
        let started = Instant::now();
        let (session, device) = load_session(&dir.join(MODEL_FILE), device)?;
        let text = std::fs::read_to_string(dir.join(TOKENS_FILE)).context("read tokens.txt")?;
        let mut vocab = HashMap::new();
        for line in text.split('\n') {
            let line = line.strip_suffix('\r').unwrap_or(line);
            if let Some((ch, id)) = line.rsplit_once(' ') {
                let mut chars = ch.chars();
                if let (Some(c), None, Ok(id)) = (chars.next(), chars.next(), id.trim().parse::<i64>()) {
                    vocab.insert(c, id);
                }
            }
        }
        if vocab.is_empty() {
            bail!("tokens.txt has no usable entries");
        }
        Ok(Self { session, vocab, device, load_ms: started.elapsed().as_secs_f64() * 1000.0 })
    }

    fn encode(&self, text: &str) -> Vec<i64> {
        // VITS add_blank: id 0 (the pad/blank token) between and around every symbol.
        let mut ids = vec![0i64];
        for c in text.nfc().flat_map(char::to_lowercase) {
            if let Some(&id) = self.vocab.get(&c) {
                ids.push(id);
                ids.push(0);
            }
        }
        ids
    }

    pub fn synthesize(&mut self, text: &str, opts: &SynthOptions) -> Result<SynthResult> {
        let started = Instant::now();
        let sentences: Vec<&str> = text
            .split(|c| c == '။' || c == '\n')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if sentences.is_empty() {
            bail!("no text to speak");
        }
        let pause = vec![0f32; MMS_SAMPLE_RATE as usize / 6];
        let mut out = Vec::new();
        let mut chunks = 0;
        for sentence in sentences {
            let ids = self.encode(sentence);
            if ids.len() <= 1 {
                continue;
            }
            let len = ids.len();
            let outputs = self.session.run(ort::inputs![
                "x" => Tensor::from_array(([1usize, len], ids))?,
                "x_length" => Tensor::from_array(([1usize], vec![len as i64]))?,
                "noise_scale" => Tensor::from_array(([1usize], vec![0.667f32]))?,
                "length_scale" => Tensor::from_array(([1usize], vec![1.0 / opts.speed.max(0.1)]))?,
                "noise_scale_w" => Tensor::from_array(([1usize], vec![0.8f32]))?,
            ])?;
            let (_, wave) = outputs[0].try_extract_tensor::<f32>()?;
            if !out.is_empty() {
                out.extend_from_slice(&pause);
            }
            out.extend_from_slice(wave);
            chunks += 1;
        }
        if out.is_empty() {
            return Err(anyhow!("no speakable Burmese characters in the text"));
        }
        Ok(SynthResult {
            samples: out,
            device: self.device,
            load_ms: self.load_ms,
            synth_ms: started.elapsed().as_secs_f64() * 1000.0,
            chunks,
            sample_rate: MMS_SAMPLE_RATE,
        })
    }
}
