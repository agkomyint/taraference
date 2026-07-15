# Install scripts (production setup)

Automate **toolchain checks + release build + model download** for a fresh machine.

| Script | Platform |
|--------|----------|
| [`install.ps1`](./install.ps1) | Windows (PowerShell) |
| [`install.sh`](./install.sh) | Linux (NVIDIA) |

## What they do

1. **Rust** — install via rustup if `cargo` is missing  
2. **C++ linker** — MSVC Build Tools (Windows / winget) or `build-essential` (Linux)  
3. **GPU** — require `nvidia-smi`  
4. **CUDA Toolkit** — detect `nvcc` / `CUDA_PATH` / NVRTC (needed for runtime kernel compile)  
5. **`cargo build --release -p taraference`**  
6. **Download GGUFs** via `taraference --download` into `models/`  

They **do not** reboot the machine or force silent CUDA installs when winget/apt cannot complete safely. Missing CUDA/driver is reported with a non-zero exit (`2`).

## Usage

### Windows

```powershell
cd D:\path\to\taraference
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install.ps1

# options
.\scripts\install.ps1 -Models 0.5b
.\scripts\install.ps1 -SkipModels
.\scripts\install.ps1 -SkipBuild
.\scripts\install.ps1 -Force
```

### Linux

```bash
cd /path/to/taraference
chmod +x scripts/install.sh
./scripts/install.sh

# options
./scripts/install.sh --models 0.5b
./scripts/install.sh --skip-models
./scripts/install.sh --skip-build
./scripts/install.sh --force
./scripts/install.sh --models-dir /data/models
```

## Manual prerequisites (if scripts cannot install)

| Component | Why |
|-----------|-----|
| NVIDIA driver | GPU runtime |
| CUDA Toolkit **13.x** (NVRTC) | Matches `cudarc` feature `cuda-13020`; kernels compile at load |
| Visual Studio C++ Build Tools (Windows) | Link Rust native deps |
| `build-essential` (Linux) | Same |

GPU arch note: device code is currently compiled for **`sm_86`** (Ampere, e.g. RTX 30-series).

## After install

```text
target/release/taraference[.exe] models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
target/release/taraference[.exe] models/….gguf --serve 3000
target/release/taraference[.exe] --download    # models only
```
