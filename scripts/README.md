# Setup

## Prebuilt binary (fastest)

**Preferred path for remote GPUs** — full docs in the root [README](../README.md#fast-install-recommended--prebuilt-linux-binary).

Linux x86_64 only — downloads the latest [GitHub Release](https://github.com/agkomyint/taraference/releases):

```bash
./scripts/get-binary.sh              # → ./taraference
./scripts/get-binary.sh v0.1.2       # pin a tag
```

Then: `./taraference --download 0.5b` and run. Needs GPU + CUDA 13.x NVRTC (no Rust).
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

Optional only:

| Flag | Meaning |
|------|---------|
| `--skip-models` / `-SkipModels` | don’t download GGUFs |
| `--force` / `-Force` | re-download GGUFs |

Installs Rust if needed, builds `target/release/taraference`, downloads models into `models/`.

Inference still needs an **NVIDIA GPU + driver + CUDA toolkit (NVRTC)** on the machine.

## Publishing a release

Push a version tag (or run **Actions → Release** manually):

```bash
git tag v0.1.0
git push origin v0.1.0
```

CI builds `taraference-linux-x86_64.tar.gz` and attaches it to the GitHub Release.
