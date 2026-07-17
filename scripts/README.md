# Setup

## Prebuilt binary (fastest)

**Preferred path for remote GPUs** — full docs in the root
[README](../README.md#fast-install-recommended--prebuilt-linux-binary).

CLI name: **`tarafer`**.

```bash
./scripts/get-binary.sh              # → ~/.local/bin/tarafer
./scripts/get-binary.sh v0.2.0       # pin a tag
```

Then:

```bash
tarafer --download 0.5b
tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

### Update / PATH

| Command | Effect |
|---------|--------|
| `tarafer install` | Copy binary to `~/.local/bin/tarafer` |
| `tarafer update` | Download latest GitHub Release over this binary |
| `tarafer update --install` | Latest → `~/.local/bin/tarafer` |

Needs GPU + CUDA 13.x NVRTC (no Rust).

## Build from source

**One command** after clone (no flags required):

### Windows
```powershell
.\scripts\install.ps1
```

### Linux
```bash
./scripts/install.sh
```

Builds `target/release/tarafer`, optionally installs to PATH, downloads models.

Optional:

| Flag | Meaning |
|------|---------|
| `--skip-models` / `-SkipModels` | don’t download GGUFs |
| `--force` / `-Force` | re-download GGUFs |

## Publishing a release

```bash
git tag v0.2.0
git push origin v0.2.0
```

CI builds `tarafer-linux-x86_64.tar.gz` and `tarafer-windows-x86_64.zip` (plus raw binaries + checksums) and attaches them to the GitHub Release.
