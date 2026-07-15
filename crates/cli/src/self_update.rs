//! Install onto PATH and self-update from GitHub Releases.

use anyhow::{bail, Context, Result};
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_REPO: &str = "agkomyint/taraference";
const ASSET_LINUX: &str = "tarafer-linux-x86_64.tar.gz";
const BIN_NAME: &str = "tarafer";

fn repo() -> String {
    env::var("TARAFER_REPO")
        .or_else(|_| env::var("TARAFERENCE_REPO"))
        .unwrap_or_else(|_| DEFAULT_REPO.to_string())
}

/// Default install dir: `$TARAFER_BIN_DIR` or `~/.local/bin`.
pub fn default_install_dir() -> Result<PathBuf> {
    if let Ok(d) = env::var("TARAFER_BIN_DIR") {
        return Ok(PathBuf::from(d));
    }
    let home = env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .context("HOME/USERPROFILE not set")?;
    Ok(PathBuf::from(home).join(".local").join("bin"))
}

/// Copy the current executable to `dir/tarafer` and ensure it is executable.
pub fn install_to_path(dir: Option<PathBuf>) -> Result<PathBuf> {
    let dir = dir.unwrap_or(default_install_dir()?);
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;

    let src = env::current_exe().context("current_exe")?;
    let src = fs::canonicalize(&src).unwrap_or(src);
    let dest = dir.join(BIN_NAME);

    if src == dest {
        eprintln!("already installed at {}", dest.display());
        ensure_path_hint(&dir);
        return Ok(dest);
    }

    let tmp = dir.join(format!(".{BIN_NAME}.installing"));
    fs::copy(&src, &tmp).with_context(|| format!("copy {} → {}", src.display(), tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&tmp)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&tmp, perms)?;
    }
    fs::rename(&tmp, &dest).with_context(|| format!("install → {}", dest.display()))?;

    eprintln!("installed  {}", dest.display());
    ensure_path_hint(&dir);
    Ok(dest)
}

fn ensure_path_hint(dir: &Path) {
    let dir_s = dir.display().to_string();
    let on_path = env::var_os("PATH")
        .map(|p| env::split_paths(&p).any(|x| x == dir))
        .unwrap_or(false);
    if on_path {
        eprintln!("OK  {BIN_NAME} is on PATH — try: {BIN_NAME} --help");
    } else {
        eprintln!(
            "!!  {dir_s} is not on PATH. Add it, e.g.:\n\
             \n\
               echo 'export PATH=\"{dir_s}:$PATH\"' >> ~/.bashrc && source ~/.bashrc\n\
             \n\
               # zsh:\n\
               echo 'export PATH=\"{dir_s}:$PATH\"' >> ~/.zshrc && source ~/.zshrc"
        );
    }
}

/// Download a release binary and replace the currently running executable (or install path).
///
/// `tag`: `None` / `"latest"` → latest release; else e.g. `"v0.2.0"`.
/// `dest`: if set, write there; else replace `current_exe()`.
pub fn self_update(tag: Option<&str>, dest: Option<PathBuf>) -> Result<PathBuf> {
    if !cfg!(target_os = "linux") {
        bail!(
            "`{BIN_NAME} update` currently ships Linux x86_64 release assets only.\n\
             On this OS, rebuild from source or download a matching release when available."
        );
    }
    self_update_linux(tag, dest)
}

fn self_update_linux(tag: Option<&str>, dest: Option<PathBuf>) -> Result<PathBuf> {
    let tag = tag.unwrap_or("latest");
    let repo = repo();
    let url = if tag == "latest" {
        format!("https://github.com/{repo}/releases/latest/download/{ASSET_LINUX}")
    } else {
        format!("https://github.com/{repo}/releases/download/{tag}/{ASSET_LINUX}")
    };

    let dest = match dest {
        Some(p) => p,
        None => {
            let exe = env::current_exe().context("current_exe")?;
            fs::canonicalize(&exe).unwrap_or(exe)
        }
    };

    eprintln!("==> downloading {url}");
    let bytes = http_get_bytes(&url).with_context(|| format!("download {url}"))?;
    eprintln!(
        "    {:.1} MiB",
        bytes.len() as f64 / (1024.0 * 1024.0)
    );

    let tmp_dir = env::temp_dir().join(format!("tarafer-update-{}", std::process::id()));
    fs::create_dir_all(&tmp_dir)?;
    let archive = tmp_dir.join(ASSET_LINUX);
    fs::write(&archive, &bytes)?;

    let status = Command::new("tar")
        .args(["-xzf"])
        .arg(&archive)
        .arg("-C")
        .arg(&tmp_dir)
        .status()
        .context("run tar")?;
    if !status.success() {
        bail!("tar extract failed (exit {status})");
    }

    let mut new_bin = tmp_dir.join(BIN_NAME);
    if !new_bin.is_file() {
        // Older releases used `taraference` as the binary name inside the tarball.
        let legacy = tmp_dir.join("taraference");
        if legacy.is_file() {
            new_bin = legacy;
        } else {
            bail!(
                "archive did not contain `{BIN_NAME}` (or legacy `taraference`); check {}",
                tmp_dir.display()
            );
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&new_bin)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&new_bin, perms)?;
    }

    let parent = dest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&parent)?;
    let staging = parent.join(format!(".{BIN_NAME}.new"));
    let backup = parent.join(format!(".{BIN_NAME}.old"));

    fs::copy(&new_bin, &staging).with_context(|| format!("stage {}", staging.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&staging)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&staging, perms)?;
    }

    if dest.exists() {
        let _ = fs::remove_file(&backup);
        // On Linux, renaming a running executable works (mapped inode stays).
        fs::rename(&dest, &backup).with_context(|| format!("backup {}", dest.display()))?;
    }
    fs::rename(&staging, &dest).with_context(|| format!("replace {}", dest.display()))?;
    let _ = fs::remove_file(&backup);
    let _ = fs::remove_dir_all(&tmp_dir);

    eprintln!("updated   {}", dest.display());
    if dest.file_name().and_then(|s| s.to_str()) == Some(BIN_NAME) {
        if let Some(parent) = dest.parent() {
            ensure_path_hint(parent);
        }
    }
    eprintln!("OK  re-run: {BIN_NAME} --help");
    if let Ok(out) = Command::new(&dest).arg("--version").output() {
        eprint!("{}", String::from_utf8_lossy(&out.stdout));
    }
    Ok(dest)
}

fn http_get_bytes(url: &str) -> Result<Vec<u8>> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(120))
        .build();
    let mut req = agent.get(url);
    if let Ok(token) = env::var("GITHUB_TOKEN") {
        if !token.is_empty() {
            req = req.set("Authorization", &format!("Bearer {token}"));
        }
    }
    req = req.set("User-Agent", "tarafer-update");
    let resp = req
        .call()
        .map_err(|e| anyhow::anyhow!("HTTP GET {url}: {e}"))?;
    if !(200..300).contains(&resp.status()) {
        bail!("HTTP {} for {url}", resp.status());
    }
    let mut reader = resp.into_reader();
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    if buf.is_empty() {
        bail!("empty response from {url}");
    }
    Ok(buf)
}
