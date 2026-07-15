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
    /// Needs ~16 GB+ VRAM for comfortable Q4 + ctx (e.g. 14B on T4: keep ctx modest).
    Large,
}

/// One first-party supported checkpoint (Q4_K_M, ChatML-compatible Qwen2.5).
pub struct ModelSpec {
    /// Local file name under models dir.
    pub file: &'static str,
    /// Hugging Face repo id.
    pub repo: &'static str,
    /// File name inside the HF repo (usually same as `file`).
    pub remote_file: &'static str,
    /// Short label for CLI (`0.5b`, `7b`, …).
    pub tag: &'static str,
    pub size: ModelSizeClass,
    /// Approximate Q4_K_M file size (GiB) for progress hints.
    pub approx_gib: f32,
    /// One-line VRAM / use note.
    pub note: &'static str,
}

/// Models taraference is built/tested against (bartowski Q4_K_M).
pub const SUPPORTED_MODELS: &[ModelSpec] = &[
    ModelSpec {
        file: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        repo: "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        remote_file: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        tag: "0.5b",
        size: ModelSizeClass::Small,
        approx_gib: 0.4,
        note: "fastest profile / 4GB OK",
    },
    ModelSpec {
        file: "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        repo: "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        remote_file: "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        tag: "1.5b",
        size: ModelSizeClass::Small,
        approx_gib: 1.0,
        note: "small step up from 0.5B",
    },
    ModelSpec {
        file: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        repo: "bartowski/Qwen2.5-3B-Instruct-GGUF",
        remote_file: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        tag: "3b",
        size: ModelSizeClass::Small,
        approx_gib: 1.9,
        note: "default medium-small; 4GB with ctx care",
    },
    ModelSpec {
        file: "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        repo: "bartowski/Qwen2.5-7B-Instruct-GGUF",
        remote_file: "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        tag: "7b",
        size: ModelSizeClass::Medium,
        approx_gib: 4.7,
        note: "good on T4 16GB; strong profile target",
    },
    ModelSpec {
        file: "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        repo: "bartowski/Qwen2.5-14B-Instruct-GGUF",
        remote_file: "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        tag: "14b",
        size: ModelSizeClass::Large,
        approx_gib: 9.0,
        note: "T4 16GB: OK for Q4; prefer lower --ctx if OOM",
    },
];

fn hf_url(repo: &str, remote_file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{remote_file}")
}

/// Print catalog to stderr (for `--download list`).
pub fn print_model_catalog() {
    eprintln!("Supported GGUF downloads (Qwen2.5 Instruct Q4_K_M via bartowski):\n");
    eprintln!(
        "  {:6}  {:6}  {:40}  {}",
        "tag", "class", "file", "note"
    );
    eprintln!("  {}", "-".repeat(90));
    for m in SUPPORTED_MODELS {
        let class = match m.size {
            ModelSizeClass::Small => "small",
            ModelSizeClass::Medium => "medium",
            ModelSizeClass::Large => "large",
        };
        eprintln!(
            "  {:6}  {:6}  {:40}  ~{:.1} GiB — {}",
            m.tag, class, m.file, m.approx_gib, m.note
        );
    }
    eprintln!(
        "\nGroups:\n  \
         all       → 0.5b + 3b only (fast install; does NOT pull 7B/14B)\n  \
         small     → 0.5b + 1.5b + 3b\n  \
         medium    → 7b\n  \
         large     → 7b + 14b\n  \
         profile   → 0.5b + 3b + 7b  (common speed ladder on 16GB)\n  \
         everything / all-sizes → every listed model\n\n\
         Examples:\n  \
         tarafer --download 7b\n  \
         tarafer --download 14b --models-dir ~/models\n  \
         tarafer --download large\n  \
         tarafer --download 0.5b,7b,14b"
    );
}

fn known_tags_hint() -> String {
    let tags: Vec<&str> = SUPPORTED_MODELS.iter().map(|m| m.tag).collect();
    format!(
        "use list | all | small | medium | large | profile | everything | {}",
        tags.join(" | ")
    )
}

/// Parse `--download` selection.
///
/// - `list` — print catalog, select nothing  
/// - `all` — **small defaults only** (0.5b + 3b) for install scripts  
/// - `large` / `profile` / `everything` — see [`print_model_catalog`]  
/// - tags: `0.5b`, `1.5b`, `3b`, `7b`, `14b` or filename substrings  
pub fn select_models(which: &str) -> Result<Vec<&'static ModelSpec>> {
    let key = which.trim().to_ascii_lowercase();
    if key.is_empty() || key == "all" {
        // Backward-compatible: install.sh should not pull multi‑GB 7B/14B by surprise.
        return Ok(SUPPORTED_MODELS
            .iter()
            .filter(|m| m.tag == "0.5b" || m.tag == "3b")
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
                .filter(|m| m.size == ModelSizeClass::Small)
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
                .filter(|m| {
                    m.size == ModelSizeClass::Medium || m.size == ModelSizeClass::Large
                })
                .collect(),
        ),
        "profile" | "ladder" => Some(
            SUPPORTED_MODELS
                .iter()
                .filter(|m| matches!(m.tag, "0.5b" | "3b" | "7b"))
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
            None => bail!(
                "unknown download target {part:?}; {}",
                known_tags_hint()
            ),
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

    // Large GGUFs (7B/14B) can take a while on slow links.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(3600))
        .build();

    // Optional HF token for rate limits / private repos.
    let mut req = agent.get(url);
    if let Ok(token) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
    {
        if !token.is_empty() {
            req = req.set("Authorization", &format!("Bearer {token}"));
        }
    }

    let resp = req
        .call()
        .with_context(|| format!("HTTP GET {url}"))?;
    if !(200..300).contains(&resp.status()) {
        bail!("download failed HTTP {} for {url}", resp.status());
    }

    let total = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok());

    let mut reader = resp.into_reader();
    let mut file = File::create(&partial).with_context(|| format!("create {}", partial.display()))?;

    let mut buf = vec![0u8; 1024 * 256];
    let mut done: u64 = 0;
    let t0 = Instant::now();
    let mut last_log = Instant::now();

    loop {
        let n = reader.read(&mut buf).context("read body")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("write partial")?;
        done += n as u64;
        if last_log.elapsed().as_millis() >= 500 {
            log_progress(done, total, t0.elapsed().as_secs_f64());
            last_log = Instant::now();
        }
    }
    file.flush()?;
    drop(file);

    fs::rename(&partial, dest)
        .with_context(|| format!("rename {} → {}", partial.display(), dest.display()))?;

    let secs = t0.elapsed().as_secs_f64().max(1e-6);
    eprintln!(
        "\n  ok   {:.1} MiB in {:.1}s ({:.1} MiB/s)",
        done as f64 / (1024.0 * 1024.0),
        secs,
        (done as f64 / (1024.0 * 1024.0)) / secs
    );
    Ok(())
}

fn log_progress(done: u64, total: Option<u64>, secs: f64) {
    let mib = done as f64 / (1024.0 * 1024.0);
    let rate = mib / secs.max(1e-6);
    match total {
        Some(t) if t > 0 => {
            let pct = 100.0 * done as f64 / t as f64;
            eprint!(
                "\r  …    {mib:.1} / {:.1} MiB ({pct:.0}%)  {rate:.1} MiB/s   ",
                t as f64 / (1024.0 * 1024.0)
            );
        }
        _ => {
            eprint!("\r  …    {mib:.1} MiB  {rate:.1} MiB/s   ");
        }
    }
    let _ = std::io::stderr().flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_all_is_small_defaults() {
        let m = select_models("all").unwrap();
        assert_eq!(m.len(), 2);
        assert!(m.iter().any(|x| x.tag == "0.5b"));
        assert!(m.iter().any(|x| x.tag == "3b"));
    }

    #[test]
    fn select_large_has_7_and_14() {
        let m = select_models("large").unwrap();
        assert!(m.iter().any(|x| x.tag == "7b"));
        assert!(m.iter().any(|x| x.tag == "14b"));
    }

    #[test]
    fn select_tags() {
        assert_eq!(select_models("0.5b").unwrap()[0].tag, "0.5b");
        assert_eq!(select_models("7b").unwrap()[0].tag, "7b");
        assert_eq!(select_models("14b").unwrap()[0].tag, "14b");
    }

    #[test]
    fn select_list_empty() {
        assert!(select_models("list").unwrap().is_empty());
    }
}
