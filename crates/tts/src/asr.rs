//! Burmese speech-to-text: `freococo/myanmar_asr` (Conformer CTC, Apache-2.0) on ONNX Runtime.
//!
//! One graph from `tools/stt/export_myanmar_asr.py`: log-mel `[1, frames, 80]` -> CTC logits over
//! 2,566 Burmese syllables (id 0 = blank). Decoding is greedy. Files: `model.onnx`, `tokens.txt`.

use crate::{load_session, Device};
use anyhow::{bail, ensure, Context, Result};
use ort::value::Tensor;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub const ASR_SAMPLE_RATE: u32 = 16_000;
pub const ASR_MODEL_DIR: &str = "models/stt/myanmar-asr";
pub const ASR_MODEL_NAME: &str = "freococo/myanmar_asr (Conformer CTC, fp32 ONNX)";
const MODEL_FILE: &str = "model.onnx";
const TOKENS_FILE: &str = "tokens.txt";

const N_FFT: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 80;
const BINS: usize = N_FFT / 2 + 1;
/// Upstream skips clips under half a second.
const MIN_SAMPLES: usize = ASR_SAMPLE_RATE as usize / 2;
/// The model was trained on single utterances; longer audio is cut at quiet points.
const CHUNK_SAMPLES: usize = ASR_SAMPLE_RATE as usize * 20;
const CUT_SEARCH: usize = ASR_SAMPLE_RATE as usize * 5;

pub fn missing_files(dir: &Path) -> Vec<PathBuf> {
    [MODEL_FILE, TOKENS_FILE].map(|f| dir.join(f)).into_iter().filter(|p| !p.is_file()).collect()
}

#[derive(Debug)]
pub struct Transcript {
    /// Recognised syllables in order; Burmese text is their concatenation.
    pub syllables: Vec<String>,
    pub audio_seconds: f64,
    pub run_ms: f64,
    pub chunks: usize,
}

impl Transcript {
    pub fn text(&self) -> String {
        self.syllables.concat()
    }
}

pub struct Asr {
    session: ort::session::Session,
    tokens: Vec<String>,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    /// `[N_MELS][BINS]` triangular HTK filters, as torchaudio's `MelSpectrogram` builds them.
    filters: Vec<f32>,
    pub device: &'static str,
    pub load_ms: f64,
}

impl Asr {
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        let missing = missing_files(dir);
        if !missing.is_empty() {
            let list: Vec<_> = missing.iter().map(|p| p.display().to_string()).collect();
            bail!(
                "speech-to-text files missing: {} (run `python tools/stt/export_myanmar_asr.py`)",
                list.join(", ")
            );
        }
        let started = Instant::now();
        let (session, device) = load_session(&dir.join(MODEL_FILE), device)?;
        let tokens: Vec<String> = std::fs::read_to_string(dir.join(TOKENS_FILE))
            .context("read tokens.txt")?
            .lines()
            .map(str::to_string)
            .collect();
        ensure!(
            tokens.first().map(String::as_str) == Some("<blank>"),
            "tokens.txt must start with <blank>"
        );
        let window = (0..N_FFT)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / N_FFT as f32).cos())
            .collect();
        Ok(Self {
            session,
            tokens,
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
            window,
            filters: mel_filters(),
            device,
            load_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Transcribes 16 kHz mono samples in `[-1, 1]`.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<Transcript> {
        let started = Instant::now();
        let mut syllables = Vec::new();
        let mut chunks = 0;
        let mut rest = samples;
        while rest.len() >= MIN_SAMPLES {
            let cut = if rest.len() <= CHUNK_SAMPLES {
                rest.len()
            } else {
                quietest(&rest[CHUNK_SAMPLES - CUT_SEARCH..CHUNK_SAMPLES]) + CHUNK_SAMPLES - CUT_SEARCH
            };
            let (chunk, tail) = rest.split_at(cut);
            rest = tail;
            if chunk.len() < MIN_SAMPLES {
                continue;
            }
            let mel = self.log_mel(chunk);
            let frames = mel.len() / N_MELS;
            let outputs = self
                .session
                .run(ort::inputs!["mel" => Tensor::from_array(([1usize, frames, N_MELS], mel))?])?;
            let (shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
            let vocab = *shape.last().context("logits have no shape")? as usize;
            ensure!(vocab == self.tokens.len(), "model has {vocab} tokens, tokens.txt has {}", self.tokens.len());
            let mut previous = 0;
            for step in logits.chunks_exact(vocab) {
                let best = (0..vocab).fold(0, |best, i| if step[i] > step[best] { i } else { best });
                // CTC: collapse repeats, drop blanks; id 1 is <unk>.
                if best != previous && best > 1 {
                    syllables.push(self.tokens[best].clone());
                }
                previous = best;
            }
            chunks += 1;
        }
        Ok(Transcript {
            syllables,
            audio_seconds: samples.len() as f64 / ASR_SAMPLE_RATE as f64,
            run_ms: started.elapsed().as_secs_f64() * 1000.0,
            chunks,
        })
    }

    /// `log(max(mel power, 1e-5))`, frame-major `[frames, N_MELS]`; centred frames, reflect padding.
    fn log_mel(&self, samples: &[f32]) -> Vec<f32> {
        let n = samples.len() as isize;
        let frames = samples.len() / HOP + 1;
        let mut mel = Vec::with_capacity(frames * N_MELS);
        let mut buffer = vec![Complex32::default(); N_FFT];
        let mut power = [0f32; BINS];
        for frame in 0..frames {
            let start = (frame * HOP) as isize - (N_FFT / 2) as isize;
            for (i, slot) in buffer.iter_mut().enumerate() {
                let mut at = start + i as isize;
                if at < 0 {
                    at = -at;
                }
                if at >= n {
                    at = 2 * (n - 1) - at;
                }
                *slot = Complex32::new(samples[at.clamp(0, n - 1) as usize] * self.window[i], 0.0);
            }
            self.fft.process(&mut buffer);
            for (p, c) in power.iter_mut().zip(&buffer) {
                *p = c.norm_sqr();
            }
            for filter in self.filters.chunks_exact(BINS) {
                let energy: f32 = filter.iter().zip(&power).map(|(f, p)| f * p).sum();
                mel.push(energy.max(1e-5).ln());
            }
        }
        mel
    }
}

fn mel_filters() -> Vec<f32> {
    let to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let to_hz = |mel: f64| 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
    let nyquist = ASR_SAMPLE_RATE as f64 / 2.0;
    let points: Vec<f64> = (0..N_MELS + 2)
        .map(|i| to_hz(to_mel(nyquist) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let mut filters = vec![0f32; N_MELS * BINS];
    for m in 0..N_MELS {
        for bin in 0..BINS {
            let hz = nyquist * bin as f64 / (BINS - 1) as f64;
            let rising = (hz - points[m]) / (points[m + 1] - points[m]);
            let falling = (points[m + 2] - hz) / (points[m + 2] - points[m + 1]);
            filters[m * BINS + bin] = rising.min(falling).max(0.0) as f32;
        }
    }
    filters
}

/// Centre of the quietest 100 ms in `samples`.
fn quietest(samples: &[f32]) -> usize {
    let width = ASR_SAMPLE_RATE as usize / 10;
    let mut energy: f32 = samples[..width].iter().map(|s| s * s).sum();
    let (mut best, mut at) = (energy, 0);
    for i in width..samples.len() {
        energy += samples[i] * samples[i] - samples[i - width] * samples[i - width];
        if energy < best {
            (best, at) = (energy, i + 1 - width);
        }
    }
    at + width / 2
}

/// Reads audio as 16 kHz mono. WAV is decoded here; other formats need `ffmpeg` on PATH.
pub fn read_audio(path: &Path) -> Result<Vec<f32>> {
    let is_wav = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    if !is_wav {
        let output = std::process::Command::new("ffmpeg")
            .args(["-nostdin", "-loglevel", "error", "-i"])
            .arg(path)
            .args(["-f", "f32le", "-ar", "16000", "-ac", "1", "-"])
            .output()
            .with_context(|| format!("{} is not a WAV file and ffmpeg could not be started", path.display()))?;
        ensure!(
            output.status.success(),
            "ffmpeg could not decode {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return Ok(output
            .stdout
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect());
    }
    let mut reader = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?
        }
    };
    let mono: Vec<f32> =
        interleaved.chunks(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect();
    Ok(resample(&mono, spec.sample_rate, ASR_SAMPLE_RATE))
}

/// Hann-windowed sinc resampling, band-limited to the lower of the two rates.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    const ZERO_CROSSINGS: f64 = 16.0;
    let step = from as f64 / to as f64;
    let cutoff = (to as f64 / from as f64).min(1.0);
    let reach = ZERO_CROSSINGS / cutoff;
    let count = (input.len() as u64 * to as u64 / from as u64) as usize;
    (0..count)
        .map(|i| {
            let centre = i as f64 * step;
            let first = (centre - reach).ceil().max(0.0) as usize;
            let last = ((centre + reach).floor() as usize).min(input.len() - 1);
            let mut sum = 0f64;
            for (j, &sample) in input.iter().enumerate().take(last + 1).skip(first) {
                let x = (j as f64 - centre) * cutoff;
                let sinc = if x.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
                let hann = 0.5 + 0.5 * (std::f64::consts::PI * x / ZERO_CROSSINGS).cos();
                sum += sample as f64 * sinc * hann * cutoff;
            }
            sum as f32
        })
        .collect()
}
