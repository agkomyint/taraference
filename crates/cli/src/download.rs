//! Download supported GGUF weights from Hugging Face into `models/`.

use anyhow::{bail, Context, Result};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// One first-party supported checkpoint (Q4_K_M, ChatML-compatible).
pub struct ModelSpec {
    /// Local file name under models dir.
    pub file: &'static str,
    /// Hugging Face repo id.
    pub repo: &'static str,
    /// File name inside the HF repo (usually same as `file`).
    pub remote_file: &'static str,
    /// Short label for CLI (`0.5b`, `3b`).
    pub tag: &'static str,
}

/// Models taraference is built/tested against.
pub const SUPPORTED_MODELS: &[ModelSpec] = &[
    ModelSpec {
        file: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        repo: "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        remote_file: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        tag: "0.5b",
    },
    ModelSpec {
        file: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        repo: "bartowski/Qwen2.5-3B-Instruct-GGUF",
        remote_file: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        tag: "3b",
    },
];

fn hf_url(repo: &str, remote_file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{remote_file}")
}

/// Parse `--download` selection: `all` | `0.5b` | `3b` | comma list.
pub fn select_models(which: &str) -> Result<Vec<&'static ModelSpec>> {
    let key = which.trim().to_ascii_lowercase();
    if key.is_empty() || key == "all" {
        return Ok(SUPPORTED_MODELS.iter().collect());
    }
    let mut out = Vec::new();
    for part in key.split([',', ' ', '+']).filter(|s| !s.is_empty()) {
        let m = SUPPORTED_MODELS
            .iter()
            .find(|m| m.tag == part || m.file.to_ascii_lowercase().contains(part));
        match m {
            Some(spec) => {
                if !out.iter().any(|x: &&ModelSpec| x.file == spec.file) {
                    out.push(spec);
                }
            }
            None => bail!(
                "unknown download target {part:?}; use all, 0.5b, 3b (or a substring of the filename)"
            ),
        }
    }
    if out.is_empty() {
        bail!("no models selected");
    }
    Ok(out)
}

/// Download selected models into `dir` (created if missing). Skip existing unless `force`.
pub fn download_models(dir: &Path, which: &str, force: bool) -> Result<Vec<PathBuf>> {
    let specs = select_models(which)?;
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
        eprintln!("download {}\n  ← {url}\n  → {}", spec.tag, dest.display());
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
        .timeout_read(std::time::Duration::from_secs(600))
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
        "  ok   {:.1} MiB in {:.1}s ({:.1} MiB/s)",
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
    fn select_all() {
        assert_eq!(select_models("all").unwrap().len(), 2);
        assert_eq!(select_models("").unwrap().len(), 2);
    }

    #[test]
    fn select_tags() {
        assert_eq!(select_models("0.5b").unwrap()[0].tag, "0.5b");
        assert_eq!(select_models("3b").unwrap()[0].tag, "3b");
    }
}
