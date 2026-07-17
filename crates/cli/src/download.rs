//! Download supported GGUF weights from Hugging Face into `models/`.

use anyhow::{bail, Context, Result};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Rough size class for download groups / VRAM hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSizeClass {
    /// Fits easily on 4–8 GB cards.
    Small,
    /// Typical 16 GB (e.g. T4) with room for KV.
    Medium,
    /// Needs ~16 GB+ VRAM for comfortable Q4 + ctx.
    Large,
}

/// One first-party supported checkpoint (Q4_K_M).
pub struct ModelSpec {
    /// Local file name under models dir.
    pub file: &'static str,
    /// Hugging Face repo id.
    pub repo: &'static str,
    /// File name inside the HF repo (usually same as `file`).
    pub remote_file: &'static str,
    /// Short label for CLI (`0.8b`, `4b`, …).
    pub tag: &'static str,
    pub size: ModelSizeClass,
    /// Approximate Q4_K_M file size (GiB) for progress hints.
    pub approx_gib: f32,
    /// One-line VRAM / use note.
    pub note: &'static str,
}

/// Models taraference is built/tested against (Unsloth Qwen3.5 Q4_K_M).
pub const SUPPORTED_MODELS: &[ModelSpec] = &[
    ModelSpec {
        file: "Qwen3.5-0.8B-Q4_K_M.gguf",
        repo: "unsloth/Qwen3.5-0.8B-GGUF",
        remote_file: "Qwen3.5-0.8B-Q4_K_M.gguf",
        tag: "0.8b",
        size: ModelSizeClass::Small,
        approx_gib: 0.5,
        note: "fastest Qwen3.5 hybrid; 4GB OK",
    },
    ModelSpec {
        file: "Qwen3.5-2B-Q4_K_M.gguf",
        repo: "unsloth/Qwen3.5-2B-GGUF",
        remote_file: "Qwen3.5-2B-Q4_K_M.gguf",
        tag: "2b",
        size: ModelSizeClass::Small,
        approx_gib: 1.3,
        note: "small step up; 4GB with ctx care",
    },
    ModelSpec {
        file: "Qwen3.5-4B-Q4_K_M.gguf",
        repo: "unsloth/Qwen3.5-4B-GGUF",
        remote_file: "Qwen3.5-4B-Q4_K_M.gguf",
        tag: "4b",
        size: ModelSizeClass::Small,
        approx_gib: 2.6,
        note: "default scoreboard-class size; ~6GB+ recommended",
    },
    ModelSpec {
        file: "Qwen3.5-9B-Q4_K_M.gguf",
        repo: "unsloth/Qwen3.5-9B-GGUF",
        remote_file: "Qwen3.5-9B-Q4_K_M.gguf",
        tag: "9b",
        size: ModelSizeClass::Medium,
        approx_gib: 5.5,
        note: "strong on 16GB; hybrid long-context friendly",
    },
    // Legacy Qwen2.5 still downloadable for A/B vs old scoreboard.
    ModelSpec {
        file: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        repo: "bartowski/Qwen2.5-3B-Instruct-GGUF",
        remote_file: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        tag: "3b-qwen25",
        size: ModelSizeClass::Small,
        approx_gib: 1.9,
        note: "legacy Qwen2.5 scoreboard model",
    },
];

fn hf_url(repo: &str, remote_file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{remote_file}")
}

/// Print catalog to stderr (for `--download list`).
pub fn print_model_catalog() {
    eprintln!("Supported GGUF downloads (Qwen3.5 Q4_K_M via unsloth; + legacy 3b-qwen25):\n");
    eprintln!("  {:10}  {:6}  {:40}  {}", "tag", "class", "file", "note");
    eprintln!("  {}", "-".repeat(90));
    for m in SUPPORTED_MODELS {
        let class = match m.size {
            ModelSizeClass::Small => "small",
            ModelSizeClass::Medium => "medium",
            ModelSizeClass::Large => "large",
        };
        eprintln!(
            "  {:10}  {:6}  {:40}  ~{:.1} GiB — {}",
            m.tag, class, m.file, m.approx_gib, m.note
        );
    }
    eprintln!(
        "\nGroups:\n  \
         all       → 0.8b + 4b only (fast install)\n  \
         small     → 0.8b + 2b + 4b\n  \
         medium    → 9b\n  \
         profile   → 0.8b + 4b + 9b\n  \
         everything / all-sizes → every listed model\n\n\
         Examples:\n  \
         tarafer --download 4b\n  \
         tarafer --download 0.8b\n  \
         tarafer --download list"
    );
}

fn known_tags_hint() -> String {
    let tags: Vec<&str> = SUPPORTED_MODELS.iter().map(|m| m.tag).collect();
    format!(
        "use list | all | small | medium | profile | everything | {}",
        tags.join(" | ")
    )
}

/// Parse `--download` selection.
pub fn select_models(which: &str) -> Result<Vec<&'static ModelSpec>> {
    let key = which.trim().to_ascii_lowercase();
    if key.is_empty() || key == "all" {
        return Ok(SUPPORTED_MODELS
            .iter()
            .filter(|m| m.tag == "0.8b" || m.tag == "4b")
            .collect());
    }
    if key == "list" || key == "help" || key == "?" {
        print_model_catalog();
        return Ok(Vec::new());
    }

    let by_group: Option<Vec<&'static ModelSpec>> = match key.as_str() {
        "small" => Some(
            SUPPORTED_MODELS
                .iter()
                .filter(|m| m.size == ModelSizeClass::Small && !m.tag.contains("qwen25"))
                .collect(),
        ),
        "medium" | "mid" => Some(
            SUPPORTED_MODELS
                .iter()
                .filter(|m| m.size == ModelSizeClass::Medium)
                .collect(),
        ),
        "large" | "big" | "larger" => Some(
            SUPPORTED_MODELS
                .iter()
                .filter(|m| m.size == ModelSizeClass::Medium || m.size == ModelSizeClass::Large)
                .collect(),
        ),
        "profile" | "ladder" => Some(
            SUPPORTED_MODELS
                .iter()
                .filter(|m| matches!(m.tag, "0.8b" | "4b" | "9b"))
                .collect(),
        ),
        "everything" | "all-sizes" | "all_sizes" | "full" => {
            Some(SUPPORTED_MODELS.iter().collect())
        }
        _ => None,
    };
    if let Some(v) = by_group {
        if v.is_empty() {
            bail!("empty model group");
        }
        return Ok(v);
    }

    let mut out = Vec::new();
    for part in key.split([',', ' ', '+']).filter(|s| !s.is_empty()) {
        // Aliases for old tags
        let part = match part {
            "0.5b" | "3b" => {
                // map old names
                if part == "0.5b" {
                    "0.8b"
                } else {
                    "4b"
                }
            }
            other => other,
        };
        let m = SUPPORTED_MODELS.iter().find(|m| {
            m.tag == part
                || m.file.to_ascii_lowercase().contains(part)
                || m.tag.replace('.', "") == part.replace('.', "")
        });
        match m {
            Some(spec) => {
                if !out.iter().any(|x: &&ModelSpec| x.file == spec.file) {
                    out.push(spec);
                }
            }
            None => bail!("unknown download target {part:?}; {}", known_tags_hint()),
        }
    }
    if out.is_empty() {
        bail!("no models selected; {}", known_tags_hint());
    }
    Ok(out)
}

/// Download selected models into `dir` (created if missing). Skip existing unless `force`.
pub fn download_models(dir: &Path, which: &str, force: bool) -> Result<Vec<PathBuf>> {
    let specs = select_models(which)?;
    if specs.is_empty() {
        return Ok(Vec::new());
    }
    fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;

    let mut paths = Vec::new();
    for spec in specs {
        let dest = dir.join(spec.file);
        if dest.is_file() && !force {
            let sz = dest.metadata().map(|m| m.len()).unwrap_or(0);
            eprintln!(
                "skip  {}  (exists, {:.1} MiB) — pass --force to re-download",
                dest.display(),
                sz as f64 / (1024.0 * 1024.0)
            );
            paths.push(dest);
            continue;
        }
        let url = hf_url(spec.repo, spec.remote_file);
        eprintln!(
            "download {}  (~{:.1} GiB, {:?})\n  ← {url}\n  → {}\n  note: {}",
            spec.tag,
            spec.approx_gib,
            spec.size,
            dest.display(),
            spec.note
        );
        download_file(&url, &dest)?;
        paths.push(dest);
    }
    Ok(paths)
}

fn download_file(url: &str, dest: &Path) -> Result<()> {
    let partial = dest.with_extension("gguf.partial");
    if partial.exists() {
        let _ = fs::remove_file(&partial);
    }

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(3600))
        .build();

    let mut req = agent.get(url);
    if let Ok(token) =
        std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
    {
        if !token.is_empty() {
            req = req.set("Authorization", &format!("Bearer {token}"));
        }
    }

    let resp = req.call().with_context(|| format!("HTTP GET {url}"))?;
    if !(200..300).contains(&resp.status()) {
        bail!("download failed HTTP {} for {url}", resp.status());
    }

    let total = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok());

    let mut reader = resp.into_reader();
    let mut file =
        File::create(&partial).with_context(|| format!("create {}", partial.display()))?;

    let mut buf = [0u8; 1024 * 256];
    let mut done = 0u64;
    let t0 = Instant::now();
    let mut last_print = Instant::now();
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        if last_print.elapsed().as_millis() > 500 {
            if let Some(tot) = total {
                let pct = 100.0 * done as f64 / tot as f64;
                let mbps = (done as f64 / 1e6) / t0.elapsed().as_secs_f64().max(1e-6);
                eprint!("\r  {pct:5.1}%  {:.1}/{:.1} MiB  {mbps:.1} MB/s   ", 
                    done as f64 / (1024.0 * 1024.0),
                    tot as f64 / (1024.0 * 1024.0));
            } else {
                eprint!("\r  {:.1} MiB   ", done as f64 / (1024.0 * 1024.0));
            }
            let _ = std::io::stderr().flush();
            last_print = Instant::now();
        }
    }
    file.flush()?;
    drop(file);
    fs::rename(&partial, dest).with_context(|| {
        format!(
            "rename {} → {}",
            partial.display(),
            dest.display()
        )
    })?;
    eprintln!(
        "\r  done  {:.1} MiB in {:.1}s                    ",
        done as f64 / (1024.0 * 1024.0),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
