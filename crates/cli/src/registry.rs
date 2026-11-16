//! Model registry: one table for every task (text, embeddings, speech, image).
//!
//! Each entry pins an upstream revision and per-file SHA-256, records the licence, and says how
//! it is installed. taraference never hosts weights: `pull` fetches from the upstream repo.
//!
//! Lifecycle: `list` -> `info` -> `pull` (licence shown, NC needs consent, hash verified)
//! -> `verify` -> use (`text`, `speech`, `image`) -> `rm`.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Text,
    Embed,
    Speech,
    Image,
}

impl Task {
    pub fn name(self) -> &'static str {
        match self {
            Task::Text => "text",
            Task::Embed => "embed",
            Task::Speech => "speech",
            Task::Image => "image",
        }
    }
}

pub struct FileSpec {
    /// Path under the entry's local directory.
    pub local: &'static str,
    /// Path inside the upstream repo.
    pub remote: &'static str,
    /// Expected size in bytes (approximate when `sha256` is empty).
    pub bytes: u64,
    /// Lowercase hex SHA-256 of the file at `Entry::rev`.
    pub sha256: &'static str,
}

pub enum Install {
    /// Fetch every file from `https://huggingface.co/<repo>/resolve/<rev>/<remote>`.
    Download,
    /// Weights must be produced locally; the string says how.
    Manual(&'static str),
}

pub struct Entry {
    pub id: &'static str,
    pub task: Task,
    pub runtime: &'static str,
    pub repo: &'static str,
    pub rev: &'static str,
    /// Local directory under the models root ("" = the root itself).
    pub dir: &'static str,
    pub files: &'static [FileSpec],
    pub licence: &'static str,
    pub commercial: bool,
    pub install: Install,
    pub note: &'static str,
}

const fn f(local: &'static str, remote: &'static str, bytes: u64, sha256: &'static str) -> FileSpec {
    FileSpec { local, remote, bytes, sha256 }
}

/// Licences below come from the upstream Hugging Face model metadata (checked 2026-10-14).
pub const REGISTRY: &[Entry] = &[
    Entry {
        id: "0.8b",
        task: Task::Text,
        runtime: "gguf-cuda",
        repo: "unsloth/Qwen3.5-0.8B-GGUF",
        rev: "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
        dir: "",
        files: &[f(
            "Qwen3.5-0.8B-Q4_K_M.gguf",
            "Qwen3.5-0.8B-Q4_K_M.gguf",
            532_517_120,
            "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517",
        )],
        licence: "Apache-2.0",
        commercial: true,
        install: Install::Download,
        note: "Qwen3.5 0.8B Q4_K_M; fastest, 4 GB GPU OK",
    },
    Entry {
        id: "2b",
        task: Task::Text,
        runtime: "gguf-cuda",
        repo: "unsloth/Qwen3.5-2B-GGUF",
        rev: "f6d5376be1edb4d416d56da11e5397a961aca8ae",
        dir: "",
        files: &[f(
            "Qwen3.5-2B-Q4_K_M.gguf",
            "Qwen3.5-2B-Q4_K_M.gguf",
            1_280_835_840,
            "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223",
        )],
        licence: "Apache-2.0",
        commercial: true,
        install: Install::Download,
        note: "Qwen3.5 2B Q4_K_M; 4 GB GPU with context care",
    },
    Entry {
        id: "4b",
        task: Task::Text,
        runtime: "gguf-cuda",
        repo: "unsloth/Qwen3.5-4B-GGUF",
        rev: "e87f176479d0855a907a41277aca2f8ee7a09523",
        dir: "",
        files: &[f(
            "Qwen3.5-4B-Q4_K_M.gguf",
            "Qwen3.5-4B-Q4_K_M.gguf",
            2_740_937_888,
            "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
        )],
        licence: "Apache-2.0",
        commercial: true,
        install: Install::Download,
        note: "Qwen3.5 4B Q4_K_M; profiling model",
    },
    Entry {
        id: "3b-qwen25",
        task: Task::Text,
        runtime: "gguf-cuda",
        repo: "bartowski/Qwen2.5-3B-Instruct-GGUF",
        rev: "f302c64a2269a69fb27b2f9473b362f5bb8e78d8",
        dir: "",
        files: &[f(
            "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            1_929_903_264,
            "9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94",
        )],
        licence: "Qwen Research License (non-commercial)",
        commercial: false,
        install: Install::Download,
        note: "Qwen2.5 3B Q4_K_M; scoreboard baseline",
    },
    Entry {
        id: "embed-4b",
        task: Task::Embed,
        runtime: "gguf-cuda",
        repo: "Qwen/Qwen3-Embedding-4B-GGUF",
        rev: "f4602530db1d980e16da9d7d3a70294cf5c190be",
        dir: "",
        files: &[f(
            "Qwen3-Embedding-4B-Q4_K_M.gguf",
            "Qwen3-Embedding-4B-Q4_K_M.gguf",
            2_496_703_776,
            "2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b",
        )],
        licence: "Apache-2.0",
        commercial: true,
        install: Install::Download,
        note: "Qwen3 embedding 4B, 2560 dims",
    },
    Entry {
        id: "mms-mya",
        task: Task::Speech,
        runtime: "onnx",
        repo: "willwade/mms-tts-multilingual-models-onnx",
        rev: "4f49fc254f11ddb0e35fbf25b6e48b04463a6779",
        dir: "tts/mms-mya",
        files: &[
            f(
                "model.onnx",
                "mya/model.onnx",
                114_031_620,
                "2dd0874209259565df110160dbad655bbb09855d8210882b640f0fc82bd4a42a",
            ),
            f(
                "tokens.txt",
                "mya/tokens.txt",
                385,
                "cd0a6ecfb0c91dde4d8ced1a6f8d18a7bc48086a438a179ae00230b86d5e76e3",
            ),
        ],
        licence: "CC BY-NC 4.0 (non-commercial)",
        commercial: false,
        install: Install::Download,
        note: "Burmese MMS-TTS (VITS), fast, CPU ok",
    },
    Entry {
        id: "f5-myanmar",
        task: Task::Speech,
        runtime: "onnx",
        repo: "freococo/F5-Myanmar-TTS",
        rev: "f636560",
        dir: "tts/f5-myanmar",
        files: &[
            f("dit.onnx", "", 684_000_000, ""),
            f("vocos.onnx", "", 54_000_000, ""),
            f("vocab.txt", "", 11_000, ""),
            f("ref_sample.wav", "", 215_000, ""),
        ],
        licence: "CC BY-NC 4.0 (non-commercial)",
        commercial: false,
        install: Install::Manual(
            "export from the PyTorch weights on this machine:\n    \
             pip install f5-myanmar-tts onnx onnxruntime\n    \
             python tools/tts/export_f5_myanmar.py",
        ),
        note: "Burmese F5-TTS v2, best quality, GPU recommended",
    },
    Entry {
        id: "myanmar-asr",
        task: Task::Speech,
        runtime: "onnx",
        repo: "freococo/myanmar_asr",
        rev: "e40d48e354f6ee1b8e15e4fc513cfba7d624e80a",
        dir: "stt/myanmar-asr",
        files: &[f("model.onnx", "", 77_400_000, ""), f("tokens.txt", "", 36_000, "")],
        licence: "Apache-2.0",
        commercial: true,
        install: Install::Manual(
            "export from the PyTorch weights on this machine:\n    \
             pip install torch safetensors onnx onnxruntime huggingface_hub\n    \
             python tools/stt/export_myanmar_asr.py",
        ),
        note: "Burmese speech-to-text (Conformer CTC, 24M params), CPU ok",
    },
    Entry {
        id: "birefnet-lite",
        task: Task::Image,
        runtime: "onnx",
        repo: "onnx-community/BiRefNet_lite-ONNX",
        rev: "f82954f",
        dir: "background-removal",
        files: &[f(
            "birefnet-lite.onnx",
            "onnx/model_fp16.onnx",
            115_000_000,
            "d39b897ceb16ae654c1731f3dba0cf9b368d9cae74b5a57459b455cc8bfec402",
        )],
        licence: "MIT",
        commercial: true,
        install: Install::Download,
        note: "background removal (BiRefNet Lite FP16)",
    },
    Entry {
        id: "sdxs-512",
        task: Task::Image,
        runtime: "onnx",
        repo: "creador/sdxs-512-dreamshaper-onnx",
        rev: "7a623d0",
        dir: "image/sdxs-512",
        files: &[
            f("unet_fp16.onnx", "unet_fp16.onnx", 631_778_188, "ada19736853c36825524d73eadf4ef91e827e5f5e6a90c2315f52a08bdc4650b"),
            f("text_encoder_fp16.onnx", "text_encoder_fp16.onnx", 246_315_807, "76dae5ad8080f53edda5743a69583d8a446a0bc6b2316c43b13477e41db9ae93"),
            f("vae_large_decoder_fp16.onnx", "vae_large_decoder_fp16.onnx", 99_085_225, "b53bb46ab03cd86aa43bb146c963fa2270c5b66771be1dd2d82462cdbfa00276"),
            f("taesd_decoder_fp16.onnx", "taesd_decoder_fp16.onnx", 2_468_339, "4e3b6894e1d10aa4e3c9cf5426b505e7da42c7dc3455226b9fe7f9c506b245bd"),
            f("tokenizer.json", "tokenizer.json", 3_642_034, "5f120510b5c01ccdb9a7d29d2beca07cac667f76472cab123ffb1c2f0b27cdeb"),
        ],
        licence: "OpenRAIL++ (use restrictions apply)",
        commercial: true,
        install: Install::Download,
        note: "text-to-image (SDXS-512 DreamShaper, 1-step, 512x512): tarafer image gen",
    },
];

pub fn find(id: &str) -> Result<&'static Entry> {
    let key = id.trim().to_ascii_lowercase();
    // A bare file name also resolves, so a path copied from `models/` works.
    REGISTRY
        .iter()
        .find(|e| {
            e.id == key
                || e.files.iter().any(|file| file.local.eq_ignore_ascii_case(id))
        })
        .with_context(|| {
            let ids: Vec<&str> = REGISTRY.iter().map(|e| e.id).collect();
            format!("unknown model '{id}'. Known: {}", ids.join(", "))
        })
}

impl Entry {
    pub fn path(&self, root: &Path, file: &FileSpec) -> PathBuf {
        let mut p = root.to_path_buf();
        p.extend(self.dir.split('/').filter(|d| !d.is_empty()));
        p.join(file.local)
    }

    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|x| x.bytes).sum()
    }

    pub fn state(&self, root: &Path) -> State {
        let present = self.files.iter().filter(|x| self.path(root, x).is_file()).count();
        match present {
            0 => State::Missing,
            n if n == self.files.len() => State::Installed,
            _ => State::Partial,
        }
    }

    /// Main weights file (first listed).
    pub fn main_path(&self, root: &Path) -> PathBuf {
        self.path(root, &self.files[0])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Missing,
    Partial,
    Installed,
}

impl State {
    pub fn label(self) -> &'static str {
        match self {
            State::Missing => "-",
            State::Partial => "partial",
            State::Installed => "installed",
        }
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn human(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GiB", gib(bytes))
    } else if bytes >= 1 << 20 {
        format!("{:.0} MiB", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{:.0} KiB", bytes as f64 / 1024.0)
    }
}

pub fn list(root: &Path, task: Option<Task>, commercial_only: bool) {
    eprintln!(
        "{:12} {:7} {:10} {:9} {:11} {}",
        "id", "task", "state", "size", "commercial", "licence / note"
    );
    eprintln!("{}", "-".repeat(100));
    for e in REGISTRY {
        if task.is_some_and(|t| t != e.task) || (commercial_only && !e.commercial) {
            continue;
        }
        eprintln!(
            "{:12} {:7} {:10} {:9} {:11} {} - {}",
            e.id,
            e.task.name(),
            e.state(root).label(),
            human(e.total_bytes()),
            if e.commercial { "yes" } else { "no" },
            e.licence,
            e.note
        );
    }
    eprintln!("\nWeights are fetched from the upstream repo (see `models info <id>`); none are hosted here.");
}

pub fn info(root: &Path, id: &str) -> Result<()> {
    let e = find(id)?;
    eprintln!("id        {}", e.id);
    eprintln!("task      {} ({})", e.task.name(), e.runtime);
    eprintln!("source    https://huggingface.co/{} @ {}", e.repo, e.rev);
    eprintln!(
        "licence   {} | commercial use: {}",
        e.licence,
        if e.commercial { "yes" } else { "no" }
    );
    eprintln!("note      {}", e.note);
    match &e.install {
        Install::Download => eprintln!("install   tarafer models pull {}", e.id),
        Install::Manual(how) => eprintln!("install   manual: {how}"),
    }
    eprintln!("state     {}", e.state(root).label());
    for file in e.files {
        let path = e.path(root, file);
        let size = fs::metadata(&path).map(|m| human(m.len())).unwrap_or_else(|_| "missing".into());
        let pin = if file.sha256.is_empty() { "unpinned" } else { &file.sha256[..12] };
        eprintln!("  {:30} {:>10}  sha256 {pin}", path.display().to_string(), size);
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify installed files against pinned hashes. Returns the number of failures.
pub fn verify(root: &Path, id: Option<&str>) -> Result<usize> {
    let entries: Vec<&Entry> = match id {
        Some(id) => vec![find(id)?],
        None => REGISTRY.iter().filter(|e| e.state(root) != State::Missing).collect(),
    };
    let mut failures = 0;
    for e in entries {
        for file in e.files {
            let path = e.path(root, file);
            if !path.is_file() {
                eprintln!("missing    {} ({})", e.id, path.display());
                failures += 1;
            } else if file.sha256.is_empty() {
                eprintln!("unpinned   {} ({})", e.id, file.local);
            } else if sha256_file(&path)? == file.sha256 {
                eprintln!("ok         {} ({})", e.id, file.local);
            } else {
                eprintln!("MISMATCH   {} ({})", e.id, path.display());
                failures += 1;
            }
        }
    }
    Ok(failures)
}

pub fn remove(root: &Path, id: &str) -> Result<()> {
    let e = find(id)?;
    if e.state(root) == State::Missing {
        bail!("{} is not installed", e.id);
    }
    for file in e.files {
        let path = e.path(root, file);
        if path.is_file() {
            fs::remove_file(&path)?;
            eprintln!("removed {}", path.display());
        }
    }
    if !e.dir.is_empty() {
        if let Some(parent) = e.main_path(root).parent() {
            let _ = fs::remove_dir(parent); // only if empty
        }
    }
    Ok(())
}

fn confirm_licence(e: &Entry, accept: bool) -> Result<()> {
    eprintln!("licence | {} | commercial use: {}", e.licence, if e.commercial { "yes" } else { "no" });
    eprintln!("source  | https://huggingface.co/{}", e.repo);
    if e.commercial || accept {
        return Ok(());
    }
    if !std::io::stdin().is_terminal() {
        bail!("{} is non-commercial; re-run with --accept-license to confirm", e.id);
    }
    eprint!("This model is non-commercial only. Continue? [y/N] ");
    std::io::stderr().flush().ok();
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        bail!("cancelled");
    }
    Ok(())
}

pub fn pull(root: &Path, ids: &[String], force: bool, accept: bool) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for id in ids {
        let e = find(id)?;
        if let Install::Manual(how) = &e.install {
            eprintln!("{} needs a local build step:\n    {how}", e.id);
            if e.state(root) == State::Installed {
                eprintln!("(already installed at {})", e.main_path(root).display());
                out.push(e.main_path(root));
                continue;
            }
            bail!("{} cannot be downloaded", e.id);
        }
        confirm_licence(e, accept)?;
        for file in e.files {
            let dest = e.path(root, file);
            if dest.is_file() && !force {
                if file.sha256.is_empty() || sha256_file(&dest)? == file.sha256 {
                    eprintln!("skip     {} (installed, verified)", dest.display());
                    continue;
                }
                eprintln!("hash differs for {}; re-downloading", dest.display());
            }
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
            }
            let url = format!("https://huggingface.co/{}/resolve/{}/{}", e.repo, e.rev, file.remote);
            eprintln!("download {} ({})\n  <- {url}", e.id, human(file.bytes));
            download_file(&url, &dest, file.sha256)?;
        }
        out.push(e.main_path(root));
    }
    Ok(out)
}

/// Stream `url` to `dest` through a `.partial` file; verify the hash before the final rename.
fn download_file(url: &str, dest: &Path, expect_sha256: &str) -> Result<()> {
    let partial = PathBuf::from(format!("{}.partial", dest.display()));
    let _ = fs::remove_file(&partial);

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(3600))
        .build();
    let mut req = agent.get(url).set("User-Agent", concat!("taraference/", env!("CARGO_PKG_VERSION")));
    if let Ok(token) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")) {
        if !token.is_empty() {
            req = req.set("Authorization", &format!("Bearer {token}"));
        }
    }
    let resp = req.call().with_context(|| format!("HTTP GET {url}"))?;
    let total = resp.header("Content-Length").and_then(|s| s.parse::<u64>().ok());
    let mut reader = resp.into_reader();
    let mut file = File::create(&partial).with_context(|| format!("create {}", partial.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    let (mut done, t0, mut last) = (0u64, Instant::now(), Instant::now());
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        done += n as u64;
        if last.elapsed().as_millis() > 500 {
            let mbps = done as f64 / 1e6 / t0.elapsed().as_secs_f64().max(1e-6);
            match total {
                Some(t) => eprint!("\r  {:5.1}%  {} / {}  {mbps:.1} MB/s   ", 100.0 * done as f64 / t as f64, human(done), human(t)),
                None => eprint!("\r  {}   ", human(done)),
            }
            last = Instant::now();
        }
    }
    file.flush()?;
    drop(file);
    let actual = format!("{:x}", hasher.finalize());
    if !expect_sha256.is_empty() && actual != expect_sha256 {
        let _ = fs::remove_file(&partial);
        bail!("SHA-256 mismatch for {url}\n  expected {expect_sha256}\n  actual   {actual}");
    }
    fs::rename(&partial, dest).with_context(|| format!("install {}", dest.display()))?;
    eprintln!("\r  done  {} in {:.1}s, sha256 {}   ", human(done), t0.elapsed().as_secs_f64(), &actual[..12]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_pinned_downloads_have_hashes() {
        for (i, a) in REGISTRY.iter().enumerate() {
            assert!(REGISTRY[i + 1..].iter().all(|b| b.id != a.id), "duplicate {}", a.id);
            if matches!(a.install, Install::Download) {
                assert!(a.files.iter().all(|x| x.sha256.len() == 64), "{} unpinned", a.id);
                assert!(!a.rev.is_empty());
            }
        }
    }

    #[test]
    fn find_by_id_and_file_name() {
        assert_eq!(find("4B").unwrap().id, "4b");
        assert_eq!(find("Qwen3.5-4B-Q4_K_M.gguf").unwrap().id, "4b");
        assert!(find("nope").is_err());
    }

    #[test]
    fn non_commercial_entries_are_flagged() {
        for id in ["3b-qwen25", "mms-mya", "f5-myanmar"] {
            assert!(!find(id).unwrap().commercial, "{id}");
        }
        assert!(find("birefnet-lite").unwrap().commercial);
    }

    #[test]
    fn hash_check_detects_change() {
        let dir = std::env::temp_dir().join("tarafer-registry-test");
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.bin");
        fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
