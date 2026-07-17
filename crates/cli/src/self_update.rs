//! Install onto PATH and self-update from GitHub Releases.

use anyhow::{bail, Context, Result};
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_REPO: &str = "agkomyint/taraference";
const ASSET_LINUX: &str = "tarafer-linux-x86_64.tar.gz";
const ASSET_WINDOWS: &str = "tarafer-windows-x86_64.zip";

/// On-disk binary filename for this platform (`tarafer` / `tarafer.exe`).
pub fn bin_name() -> &'static str {
    if cfg!(windows) {
        "tarafer.exe"
    } else {
        "tarafer"
    }
}

fn repo() -> String {
    env::var("TARAFER_REPO")
        .or_else(|_| env::var("TARAFERENCE_REPO"))
        .unwrap_or_else(|_| DEFAULT_REPO.to_string())
}

fn release_asset_name() -> Result<&'static str> {
    if cfg!(target_os = "linux") {
        Ok(ASSET_LINUX)
    } else if cfg!(target_os = "windows") {
        Ok(ASSET_WINDOWS)
    } else {
        bail!(
            "`{} update` currently ships Linux and Windows x86_64 release assets only.\n\
             On this OS, rebuild from source or download a matching release when available.",
            bin_name()
        );
    }
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

/// Copy the current executable to `dir/<bin_name>` and ensure it is executable.
pub fn install_to_path(dir: Option<PathBuf>) -> Result<PathBuf> {
    let dir = dir.unwrap_or(default_install_dir()?);
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;

    let src = env::current_exe().context("current_exe")?;
    let src = fs::canonicalize(&src).unwrap_or(src);
    let dest = dir.join(bin_name());

    if paths_equal(&src, &dest) {
        eprintln!("already installed at {}", dest.display());
        ensure_path_hint(&dir);
        return Ok(dest);
    }

    let tmp = dir.join(format!(".{}.installing", bin_name()));
    fs::copy(&src, &tmp).with_context(|| format!("copy {} → {}", src.display(), tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&tmp)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&tmp, perms)?;
    }
    // On Windows, replacing a locked running image can fail — remove staging then rename.
    let _ = fs::remove_file(&dest);
    fs::rename(&tmp, &dest).with_context(|| format!("install → {}", dest.display()))?;

    eprintln!("installed  {}", dest.display());
    ensure_path_hint(&dir);
    Ok(dest)
}

fn paths_equal(a: &Path, b: &Path) -> bool {
    let ca = fs::canonicalize(a).unwrap_or_else(|_| a.to_path_buf());
    let cb = fs::canonicalize(b).unwrap_or_else(|_| b.to_path_buf());
    ca == cb
}

fn ensure_path_hint(dir: &Path) {
    let dir_s = dir.display().to_string();
    let on_path = env::var_os("PATH")
        .map(|p| env::split_paths(&p).any(|x| x == dir))
        .unwrap_or(false);
    let name = bin_name();
    if on_path {
        eprintln!("OK  {name} is on PATH — try: {name} --help");
    } else if cfg!(windows) {
        eprintln!(
            "!!  {dir_s} is not on PATH. Add it, e.g.:\n\
             \n\
               $env:Path += \";{dir_s}\"\n\
             \n\
               # permanent (User PATH):\n\
               [Environment]::SetEnvironmentVariable(\n\
                 \"Path\",\n\
                 [Environment]::GetEnvironmentVariable(\"Path\",\"User\") + \";{dir_s}\",\n\
                 \"User\")"
        );
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
    let asset = release_asset_name()?;
    let tag = tag.unwrap_or("latest");
    let repo = repo();
    let url = if tag == "latest" {
        format!("https://github.com/{repo}/releases/latest/download/{asset}")
    } else {
        format!("https://github.com/{repo}/releases/download/{tag}/{asset}")
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
    eprintln!("    {:.1} MiB", bytes.len() as f64 / (1024.0 * 1024.0));

    let tmp_dir = env::temp_dir().join(format!("tarafer-update-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir)?;
    let archive = tmp_dir.join(asset);
    fs::write(&archive, &bytes)?;

    // Windows 10+ ships `tar` that handles both .tar.gz and .zip.
    let status = Command::new("tar")
        .args(["-xf"])
        .arg(&archive)
        .arg("-C")
        .arg(&tmp_dir)
        .status()
        .context("run tar (needed to extract the release archive)")?;
    if !status.success() {
        bail!("tar extract failed (exit {status})");
    }

    let mut new_bin = tmp_dir.join(bin_name());
    if !new_bin.is_file() {
        // Older Linux releases used `taraference` as the binary name inside the tarball.
        let legacy = tmp_dir.join("taraference");
        let legacy_exe = tmp_dir.join("taraference.exe");
        if legacy.is_file() {
            new_bin = legacy;
        } else if legacy_exe.is_file() {
            new_bin = legacy_exe;
        } else {
            // Zip may nest one directory — search one level.
            if let Ok(entries) = fs::read_dir(&tmp_dir) {
                for ent in entries.flatten() {
                    let p = ent.path();
                    if p.is_file() {
                        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                        if name == bin_name()
                            || name == "tarafer"
                            || name == "tarafer.exe"
                            || name == "taraference"
                            || name == "taraference.exe"
                        {
                            new_bin = p;
                            break;
                        }
                    }
                }
            }
            if !new_bin.is_file() {
                bail!(
                    "archive did not contain `{}` (or legacy names); check {}",
                    bin_name(),
                    tmp_dir.display()
                );
            }
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&new_bin)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&new_bin, perms)?;
    }

    replace_executable(&new_bin, &dest)?;
    let _ = fs::remove_dir_all(&tmp_dir);

    eprintln!("updated   {}", dest.display());
    if dest.file_name().and_then(|s| s.to_str()) == Some(bin_name())
        || dest.file_name().and_then(|s| s.to_str()) == Some("tarafer")
    {
        if let Some(parent) = dest.parent() {
            ensure_path_hint(parent);
        }
    }
    eprintln!("OK  re-run: {} --help", bin_name());
    if let Ok(out) = Command::new(&dest).arg("--version").output() {
        eprint!("{}", String::from_utf8_lossy(&out.stdout));
    }
    Ok(dest)
}

fn replace_executable(new_bin: &Path, dest: &Path) -> Result<()> {
    let parent = dest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&parent)?;
    let staging = parent.join(format!(".{}.new", bin_name()));
    let backup = parent.join(format!(".{}.old", bin_name()));

    fs::copy(new_bin, &staging).with_context(|| format!("stage {}", staging.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&staging)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&staging, perms)?;
    }

    #[cfg(windows)]
    {
        let replacing_self = env::current_exe()
            .ok()
            .map(|exe| {
                let exe = fs::canonicalize(&exe).unwrap_or(exe);
                paths_equal(&exe, dest)
            })
            .unwrap_or(false);
        if replacing_self {
            // Windows locks the running image — schedule a rename after this process exits.
            return schedule_windows_replace(&staging, dest, &backup);
        }
    }

    if dest.exists() {
        let _ = fs::remove_file(&backup);
        // On Linux, renaming a running executable works (mapped inode stays).
        // On Windows (non-self), this also works when the file is not locked.
        match fs::rename(dest, &backup) {
            Ok(()) => {}
            Err(e) => {
                #[cfg(windows)]
                {
                    // Fallback: deferred replace even when not detected as self.
                    eprintln!("!!  direct replace failed ({e}); scheduling after exit…");
                    return schedule_windows_replace(&staging, dest, &backup);
                }
                #[cfg(not(windows))]
                {
                    return Err(e).with_context(|| format!("backup {}", dest.display()));
                }
            }
        }
    }
    fs::rename(&staging, dest).with_context(|| format!("replace {}", dest.display()))?;
    let _ = fs::remove_file(&backup);
    Ok(())
}

#[cfg(windows)]
fn schedule_windows_replace(staging: &Path, dest: &Path, backup: &Path) -> Result<()> {
    let bat = dest
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(".tarafer-update-{}.bat", std::process::id()));
    let staging_s = staging.display().to_string().replace('/', "\\");
    let dest_s = dest.display().to_string().replace('/', "\\");
    let backup_s = backup.display().to_string().replace('/', "\\");
    let bat_s = bat.display().to_string().replace('/', "\\");

    // Wait for the parent process to release the file lock, then swap binaries.
    let script = format!(
        "@echo off\r\n\
         setlocal\r\n\
         :retry\r\n\
         ping -n 2 127.0.0.1 >nul\r\n\
         if exist \"{dest_s}\" (\r\n\
           move /Y \"{dest_s}\" \"{backup_s}\" >nul 2>&1\r\n\
           if errorlevel 1 goto retry\r\n\
         )\r\n\
         move /Y \"{staging_s}\" \"{dest_s}\" >nul 2>&1\r\n\
         if errorlevel 1 goto retry\r\n\
         if exist \"{backup_s}\" del /F /Q \"{backup_s}\" >nul 2>&1\r\n\
         del /F /Q \"{bat_s}\" >nul 2>&1\r\n"
    );
    fs::write(&bat, script).with_context(|| format!("write {}", bat.display()))?;

    // Detached so this process can exit and unlock the .exe.
    Command::new("cmd")
        .args(["/C", "start", "", "/MIN", &bat_s])
        .spawn()
        .context("spawn Windows update helper")?;

    eprintln!(
        "update staged — binary will be replaced a moment after this process exits:\n  {}",
        dest.display()
    );
    Ok(())
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
