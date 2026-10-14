//! Local Burmese text-to-speech with F5-Myanmar-TTS (flow-matching DiT + Vocos) on ONNX Runtime.
//!
//! The DiT denoiser and the Vocos spectrogram head run as ONNX graphs; the text frontend,
//! reference-mel extraction, Euler/CFG sampler and ISTFT live here. Model files are produced
//! by `tools/tts/export_f5_myanmar.py` (weights are CC BY-NC 4.0, non-commercial).

use anyhow::{anyhow, bail, Context, Result};
use ort::{ep, session::Session, value::Tensor};
use regex::Regex;
use rustfft::{num_complex::Complex32, FftPlanner};
use std::path::{Path, PathBuf};
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;

pub mod mms;
pub mod normalize;

pub const SAMPLE_RATE: u32 = 24_000;
pub const DIT_FILE: &str = "dit.onnx";
pub const VOCOS_FILE: &str = "vocos.onnx";
pub const VOCAB_FILE: &str = "vocab.txt";
pub const REF_AUDIO_FILE: &str = "ref_sample.wav";
pub const REF_TEXT: &str = "အပြင် မသွားရလို့ စိတ်ညစ်နေပါတယ် မိုးတွေ တအား ရွာနေပါတယ်";
pub const DEFAULT_MODEL_DIR: &str = "models/tts/f5-myanmar";

const HOP: usize = 256;
const N_FFT: usize = 1024;
const N_MELS: usize = 100;
/// The model's positional table covers 8192 mel frames (~87 s).
const MAX_FRAMES: usize = 8000;
/// Longest single chunk generated in one pass, in seconds of estimated speech.
const CHUNK_SECONDS: f64 = 14.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Auto,
    Cuda,
    DirectMl,
    Cpu,
}

impl std::str::FromStr for Device {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cuda" | "gpu" => Ok(Self::Cuda),
            "directml" | "dml" => Ok(Self::DirectMl),
            "cpu" => Ok(Self::Cpu),
            _ => bail!("unknown tts device {value:?}; choose auto, cuda, directml, or cpu"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynthOptions {
    pub steps: usize,
    pub cfg: f32,
    pub speed: f32,
    pub seed: u64,
    pub ref_audio: Option<PathBuf>,
    pub ref_text: Option<String>,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self { steps: 32, cfg: 2.0, speed: 1.0, seed: 0, ref_audio: None, ref_text: None }
    }
}

#[derive(Debug)]
pub struct SynthResult {
    pub samples: Vec<f32>,
    pub device: &'static str,
    pub load_ms: f64,
    pub synth_ms: f64,
    pub chunks: usize,
    pub sample_rate: u32,
}

impl SynthResult {
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

pub fn model_files(dir: &Path) -> [PathBuf; 4] {
    [DIT_FILE, VOCOS_FILE, VOCAB_FILE, REF_AUDIO_FILE].map(|f| dir.join(f))
}

pub fn missing_files(dir: &Path) -> Vec<PathBuf> {
    model_files(dir).into_iter().filter(|p| !p.is_file()).collect()
}

pub(crate) fn load_session(path: &Path, device: Device) -> Result<(Session, &'static str)> {
    if matches!(device, Device::Auto | Device::Cuda) {
        let attempt = (|| -> Result<Session> {
            let cuda = ep::CUDA::default()
                .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
                .build()
                .error_on_failure();
            Session::builder()
                .map_err(|e| anyhow!(e.to_string()))?
                .with_execution_providers([cuda])
                .map_err(|e| anyhow!(e.to_string()))?
                .commit_from_file(path)
                .map_err(|e| anyhow!(e.to_string()))
        })();
        match attempt {
            Ok(s) => return Ok((s, "CUDA")),
            Err(e) if device == Device::Cuda => {
                return Err(e).context("could not start TTS with ONNX Runtime CUDA")
            }
            Err(e) => eprintln!("CUDA TTS unavailable ({e:#}); trying next provider"),
        }
    }
    #[cfg(windows)]
    if matches!(device, Device::Auto | Device::DirectMl) {
        let attempt = (|| -> Result<Session> {
            let dml = ep::DirectML::default().build().error_on_failure();
            Session::builder()
                .map_err(|e| anyhow!(e.to_string()))?
                .with_execution_providers([dml])
                .map_err(|e| anyhow!(e.to_string()))?
                .commit_from_file(path)
                .map_err(|e| anyhow!(e.to_string()))
        })();
        match attempt {
            Ok(s) => return Ok((s, "DirectML")),
            Err(e) if device == Device::DirectMl => {
                return Err(e).context("could not start TTS with ONNX Runtime DirectML")
            }
            Err(e) => eprintln!("DirectML TTS unavailable ({e:#}); falling back to CPU"),
        }
    }
    #[cfg(not(windows))]
    if device == Device::DirectMl {
        bail!("DirectML is available only on Windows");
    }
    let s = Session::builder().map_err(|e| anyhow!(e.to_string()))?.commit_from_file(path).map_err(|e| anyhow!(e.to_string()))?;
    Ok((s, "CPU"))
}

pub struct Tts {
    dit: Session,
    vocos: Session,
    vocab: std::collections::HashMap<char, i64>,
    ref_wave: Vec<f32>,
    mel_fb: Vec<f32>, // [N_MELS][N_FFT/2+1]
    pub device: &'static str,
    pub load_ms: f64,
}

impl Tts {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let missing = missing_files(model_dir);
        if !missing.is_empty() {
            let list: Vec<_> = missing.iter().map(|p| p.display().to_string()).collect();
            bail!(
                "TTS model files missing: {}\nCreate them with `python tools/tts/export_f5_myanmar.py --out {}`",
                list.join(", "),
                model_dir.display()
            );
        }
        let started = Instant::now();
        let (dit, device_name) = load_session(&model_dir.join(DIT_FILE), device)?;
        // Vocos runs on the provider the denoiser ended up on.
        let vocos_device = match device_name {
            "CUDA" => Device::Cuda,
            "DirectML" => Device::DirectMl,
            _ => Device::Cpu,
        };
        let (vocos, _) = load_session(&model_dir.join(VOCOS_FILE), vocos_device)?;

        let vocab_text = std::fs::read_to_string(model_dir.join(VOCAB_FILE)).context("read vocab.txt")?;
        let mut vocab = std::collections::HashMap::new();
        for (i, line) in vocab_text.split('\n').enumerate() {
            let line = line.strip_suffix('\r').unwrap_or(line);
            let mut chars = line.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                vocab.entry(c).or_insert(i as i64);
            }
        }
        let ref_wave = read_wav_mono_24k(&model_dir.join(REF_AUDIO_FILE))?;
        Ok(Self {
            dit,
            vocos,
            vocab,
            ref_wave,
            mel_fb: mel_filterbank(),
            device: device_name,
            load_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub fn synthesize(&mut self, text: &str, opts: &SynthOptions) -> Result<SynthResult> {
        let started = Instant::now();
        let (ref_wave, ref_text) = match &opts.ref_audio {
            Some(path) => {
                let text = opts
                    .ref_text
                    .clone()
                    .filter(|t| !t.trim().is_empty())
                    .ok_or_else(|| anyhow!("--ref-text is required with --ref-audio"))?;
                (read_wav_mono_24k(path)?, text)
            }
            None => (self.ref_wave.clone(), REF_TEXT.to_string()),
        };
        let ref_wave = trim_silence(&ref_wave);
        if ref_wave.len() < HOP * 4 {
            bail!("reference audio is too short or silent");
        }
        let cond_mel = log_mel(&ref_wave, &self.mel_fb);
        let ref_len = ref_wave.len() / HOP;

        let chunks = split_chunks(text, opts.speed);
        if chunks.is_empty() {
            bail!("no text to synthesize");
        }
        let mut out = Vec::new();
        let mut rng = Rng::new(opts.seed);
        for (index, chunk) in chunks.iter().enumerate() {
            let (clean, target_sec) = clean_burmese(chunk, opts.speed);
            let gen_frames = (target_sec * SAMPLE_RATE as f64 / HOP as f64) as usize;
            let total = ref_len + gen_frames.max(1);
            if total > MAX_FRAMES {
                bail!("chunk {} is too long ({total} frames); split the text into shorter sentences", index + 1);
            }
            let wave = self.generate(&clean, &ref_text, &cond_mel, ref_len, total, opts, &mut rng)?;
            if index > 0 {
                out.extend(std::iter::repeat(0f32).take(SAMPLE_RATE as usize / 8));
            }
            out.extend(wave);
        }
        Ok(SynthResult {
            samples: out,
            device: self.device,
            load_ms: self.load_ms,
            synth_ms: started.elapsed().as_secs_f64() * 1000.0,
            chunks: chunks.len(),
            sample_rate: SAMPLE_RATE,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate(
        &mut self,
        clean: &str,
        ref_text: &str,
        cond_mel: &[f32],
        ref_len: usize,
        total: usize,
        opts: &SynthOptions,
        rng: &mut Rng,
    ) -> Result<Vec<f32>> {
        let cond_frames = (cond_mel.len() / N_MELS).min(total);
        let mut cond = vec![0f32; total * N_MELS];
        cond[..cond_frames * N_MELS].copy_from_slice(&cond_mel[..cond_frames * N_MELS]);

        // Token ids are vocab index + 1 so that 0 stays the filler token.
        let mut text_ids = vec![0i64; total];
        let full = format!("{ref_text} {clean}");
        for (slot, ch) in text_ids.iter_mut().zip(full.chars()) {
            *slot = self.vocab.get(&ch).copied().unwrap_or(0) + 1;
        }

        let mut x: Vec<f32> = match std::env::var("TARAFER_TTS_NOISE") {
            Ok(path) => read_f32_file(Path::new(&path), total * N_MELS)?,
            Err(_) => (0..total * N_MELS).map(|_| rng.gaussian()).collect(),
        };

        let ts = sway_timesteps(opts.steps);
        let scalar = |v: f32| Tensor::from_array(([0usize; 0], vec![v]));
        for i in 0..opts.steps {
            let outputs = self.dit.run(ort::inputs![
                "x" => Tensor::from_array(([1usize, total, N_MELS], x.clone()))?,
                "cond" => Tensor::from_array(([1usize, total, N_MELS], cond.clone()))?,
                "text" => Tensor::from_array(([1usize, total], text_ids.clone()))?,
                "time" => scalar(ts[i])?,
                "cfg" => scalar(opts.cfg)?,
            ])?;
            let (_, v) = outputs[0].try_extract_tensor::<f32>()?;
            if v.len() != x.len() {
                bail!("unexpected DiT output size {} (expected {})", v.len(), x.len());
            }
            let dt = ts[i + 1] - ts[i];
            for (xi, vi) in x.iter_mut().zip(v) {
                *xi += dt * vi;
            }
        }
        // The prompt region stays equal to the reference mel.
        x[..cond_frames * N_MELS].copy_from_slice(&cond[..cond_frames * N_MELS]);

        let n = total - ref_len;
        let mut mel = vec![0f32; N_MELS * n];
        for t in 0..n {
            for m in 0..N_MELS {
                mel[m * n + t] = x[(ref_len + t) * N_MELS + m];
            }
        }
        let outputs = self
            .vocos
            .run(ort::inputs!["mel" => Tensor::from_array(([1usize, N_MELS, n], mel))?])?;
        let (_, re) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, im) = outputs[1].try_extract_tensor::<f32>()?;
        Ok(istft(re, im, n))
    }
}

pub fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    write_wav_rate(path, samples, SAMPLE_RATE)
}

pub fn write_wav_rate(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::create(path, spec).with_context(|| format!("create {}", path.display()))?;
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn read_wav_mono_24k(path: &Path) -> Result<Vec<f32>> {
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
    if mono.is_empty() {
        bail!("{} contains no audio", path.display());
    }
    if spec.sample_rate == SAMPLE_RATE {
        return Ok(mono);
    }
    // Linear resampling is adequate for a conditioning prompt.
    let ratio = spec.sample_rate as f64 / SAMPLE_RATE as f64;
    let n = (mono.len() as f64 / ratio) as usize;
    Ok((0..n)
        .map(|i| {
            let p = i as f64 * ratio;
            let j = p as usize;
            let f = (p - j as f64) as f32;
            let a = mono[j.min(mono.len() - 1)];
            let b = mono[(j + 1).min(mono.len() - 1)];
            a + (b - a) * f
        })
        .collect())
}

fn read_f32_file(path: &Path, count: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() != count * 4 {
        bail!("{} has {} bytes; expected {} f32 values", path.display(), bytes.len(), count);
    }
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn trim_silence(wave: &[f32]) -> Vec<f32> {
    let peak = wave.iter().fold(0f32, |m, v| m.max(v.abs()));
    let th = 10f32.powf(-35.0 / 20.0) * peak;
    let first = wave.iter().position(|v| v.abs() > th);
    let last = wave.iter().rposition(|v| v.abs() > th);
    match (first, last) {
        (Some(s), Some(e)) => {
            let margin = (SAMPLE_RATE as f64 * 0.05) as usize;
            wave[s.saturating_sub(margin)..(e + margin).min(wave.len() - 1) + 1].to_vec()
        }
        _ => wave.to_vec(),
    }
}

/// Burmese text normalisation used by the original package, plus the duration estimate.
pub fn clean_burmese(text: &str, speed: f32) -> (String, f64) {
    let nfc: String = text.nfc().collect();
    let r1 = Regex::new(r"\s+(\x{1004}\x{103a}\x{1037}?)").unwrap();
    let r2 = Regex::new(r"\s+([\x{1000}-\x{1021}]\x{103a}[\x{1037}\x{1038}]?)").unwrap();
    let r3 = Regex::new(r"\s+([\x{102b}-\x{103e}])").unwrap();
    let r4 = Regex::new(r"(\x{1031})([\x{103b}\x{103c}\x{103d}\x{103e}]+)").unwrap();
    let mut s = r1.replace_all(&nfc, "$1").into_owned();
    s = r2.replace_all(&s, "$1").into_owned();
    s = r3.replace_all(&s, "$1").into_owned();
    s = r4.replace_all(&s, "$2$1").into_owned();
    // Space after a Burmese comma/full stop when text follows immediately.
    let mut spaced = String::with_capacity(s.len() + 8);
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        spaced.push(c);
        if (c == '၊' || c == '။') && it.peek().map_or(false, |n| !n.is_whitespace()) {
            spaced.push(' ');
        }
    }
    let clean = spaced.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = clean.chars().collect();
    let syllables = chars
        .iter()
        .enumerate()
        .filter(|(i, &c)| ('\u{1000}'..='\u{1021}').contains(&c) && chars.get(i + 1) != Some(&'\u{103a}'))
        .count()
        .max(1);
    let secs = (syllables as f64 * 0.28 / speed as f64) + 0.45;
    (clean, (secs * 100.0).round() / 100.0)
}

/// Split long input at sentence marks so each pass stays within the model's useful range.
fn split_chunks(text: &str, speed: f32) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut sentence = String::new();
    for c in text.chars() {
        sentence.push(c);
        if c == '။' || c == '\n' {
            if !sentence.trim().is_empty() {
                sentences.push(std::mem::take(&mut sentence));
            }
            sentence.clear();
        }
    }
    if !sentence.trim().is_empty() {
        sentences.push(sentence);
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    for s in sentences {
        let candidate = format!("{current}{s}");
        if !current.is_empty() && clean_burmese(&candidate, speed).1 > CHUNK_SECONDS {
            chunks.push(std::mem::take(&mut current).trim().to_string());
        }
        current.push_str(&s);
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks
}

fn sway_timesteps(steps: usize) -> Vec<f32> {
    let table: &[usize] = match steps {
        5 => &[0, 2, 4, 8, 16, 32],
        6 => &[0, 2, 4, 6, 8, 16, 32],
        7 => &[0, 2, 4, 6, 8, 16, 24, 32],
        10 => &[0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12 => &[0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16 => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        _ => &[],
    };
    let base: Vec<f64> = if table.is_empty() {
        (0..=steps).map(|i| i as f64 / steps as f64).collect()
    } else {
        table.iter().map(|&t| t as f64 / 32.0).collect()
    };
    // sway sampling coefficient -1.0
    base.iter()
        .map(|&t| (t - ((std::f64::consts::FRAC_PI_2 * t).cos() - 1.0 + t)) as f32)
        .collect()
}

fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos()) as f32)
        .collect()
}

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// torchaudio melscale_fbanks(n_freqs=513, 0..12000 Hz, 100 mels, htk, no norm); layout [mel][freq].
fn mel_filterbank() -> Vec<f32> {
    let n_freqs = N_FFT / 2 + 1;
    let f_max = SAMPLE_RATE as f64 / 2.0;
    let (m_min, m_max) = (hz_to_mel(0.0), hz_to_mel(f_max));
    let f_pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let mut fb = vec![0f32; N_MELS * n_freqs];
    for m in 0..N_MELS {
        let (lo, mid, hi) = (f_pts[m], f_pts[m + 1], f_pts[m + 2]);
        for k in 0..n_freqs {
            let f = f_max * k as f64 / (n_freqs - 1) as f64;
            let down = (f - lo) / (mid - lo);
            let up = (hi - f) / (hi - mid);
            fb[m * n_freqs + k] = down.min(up).max(0.0) as f32;
        }
    }
    fb
}

/// Log-mel frames laid out [frame][mel], matching torchaudio MelSpectrogram(center=True, power=1).
fn log_mel(wave: &[f32], fb: &[f32]) -> Vec<f32> {
    let n_freqs = N_FFT / 2 + 1;
    let pad = N_FFT / 2;
    let n = wave.len();
    let mut padded = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        padded.push(wave[(pad - i).min(n - 1)]);
    }
    padded.extend_from_slice(wave);
    for i in 0..pad {
        padded.push(wave[n.saturating_sub(2 + i)]);
    }
    let frames = n / HOP + 1;
    let window = hann_periodic(N_FFT);
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);
    let mut out = vec![0f32; frames * N_MELS];
    let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
    let mut mag = vec![0f32; n_freqs];
    for t in 0..frames {
        for k in 0..N_FFT {
            buf[k] = Complex32::new(padded[t * HOP + k] * window[k], 0.0);
        }
        fft.process(&mut buf);
        for k in 0..n_freqs {
            mag[k] = buf[k].norm();
        }
        for m in 0..N_MELS {
            let row = &fb[m * n_freqs..(m + 1) * n_freqs];
            let v: f32 = row.iter().zip(&mag).map(|(a, b)| a * b).sum();
            out[t * N_MELS + m] = v.max(1e-5).ln();
        }
    }
    out
}

/// Vocos "same"-padded ISTFT. `re`/`im` are [513][frames] row-major.
fn istft(re: &[f32], im: &[f32], frames: usize) -> Vec<f32> {
    let n_freqs = N_FFT / 2 + 1;
    let window = hann_periodic(N_FFT);
    let ifft = FftPlanner::<f32>::new().plan_fft_inverse(N_FFT);
    let len = (frames - 1) * HOP + N_FFT;
    let mut y = vec![0f32; len];
    let mut env = vec![0f32; len];
    let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
    for t in 0..frames {
        for k in 0..n_freqs {
            let edge = k == 0 || k == n_freqs - 1;
            let r = re[k * frames + t];
            let i = if edge { 0.0 } else { im[k * frames + t] };
            buf[k] = Complex32::new(r, i);
            if !edge {
                buf[N_FFT - k] = Complex32::new(r, -i);
            }
        }
        ifft.process(&mut buf);
        for k in 0..N_FFT {
            y[t * HOP + k] += buf[k].re / N_FFT as f32 * window[k];
            env[t * HOP + k] += window[k] * window[k];
        }
    }
    let pad = (N_FFT - HOP) / 2;
    (pad..len - pad).map(|i| y[i] / env[i].max(1e-11)).collect()
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
    }

    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn gaussian(&mut self) -> f32 {
        let (u1, u2) = (self.uniform(), self.uniform());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_spaces_punctuation_and_estimates_duration() {
        let (clean, secs) =
            clean_burmese("မင်္ဂလာပါ၊ကျွန်တော်တို့ တရားဖြေရှင်းချက် စနစ်ကို စမ်းသပ်နေပါတယ်။", 1.0);
        assert!(clean.contains("၊ ကျွန်တော်တို့"));
        assert!(secs > 3.0 && secs < 10.0, "{secs}");
    }

    #[test]
    fn timesteps_span_zero_to_one() {
        for steps in [8, 12, 16, 32] {
            let ts = sway_timesteps(steps);
            assert_eq!(ts.len(), steps + 1);
            assert!(ts[0].abs() < 1e-6 && (ts[steps] - 1.0).abs() < 1e-6);
            assert!(ts.windows(2).all(|w| w[1] > w[0]));
        }
    }

    #[test]
    fn istft_of_empty_spectrum_is_silent() {
        let frames = 8;
        let z = vec![0f32; 513 * frames];
        assert!(istft(&z, &z, frames).iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn long_text_is_split_into_chunks() {
        let sentence = "ကျွန်တော်တို့ တရားဖြေရှင်းချက် စနစ်ကို စမ်းသပ်နေပါတယ်။";
        let text = sentence.repeat(8);
        assert!(split_chunks(&text, 1.0).len() > 1);
    }
}
