#!/usr/bin/env bash
# Production-oriented setup for taraference (Linux / macOS-with-NVIDIA).
#
# Checks / installs what can be automated:
#   - Rust (rustup)
#   - C/C++ toolchain (build-essential / Xcode CLT)
#   - NVIDIA driver + CUDA toolkit presence
# Then builds release binary and optionally downloads GGUF models.
#
# Usage:
#   ./scripts/install.sh
#   ./scripts/install.sh --models 0.5b
#   ./scripts/install.sh --skip-models
#   ./scripts/install.sh --skip-build
#   ./scripts/install.sh --force
#
# Env:
#   HF_TOKEN                 optional Hugging Face token for downloads
#   TARAFERENCE_MODELS_DIR  override models directory

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_MODELS=0
SKIP_BUILD=0
MODELS="all"
FORCE=0
MODELS_DIR="${TARAFERENCE_MODELS_DIR:-$ROOT/models}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-models) SKIP_MODELS=1; shift ;;
    --skip-build)  SKIP_BUILD=1; shift ;;
    --models)
      MODELS="${2:-all}"; shift 2 ;;
    --models-dir)
      MODELS_DIR="${2:?}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

step()  { printf '\n\033[36m==> %s\033[0m\n' "$*"; }
ok()    { printf '  \033[32mOK\033[0m  %s\n' "$*"; }
warn()  { printf '  \033[33m!!\033[0m  %s\n' "$*"; }
fail()  { printf '  \033[31mXX\033[0m  %s\n' "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

ensure_rust() {
  step "Rust toolchain"
  if have cargo; then
    ok "$(cargo --version)"
    return
  fi
  warn "cargo not found — installing rustup (default stable)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  source "${CARGO_HOME:-$HOME/.cargo}/env"
  if ! have cargo; then
    fail "cargo still not on PATH; open a new shell and re-run"
    exit 1
  fi
  ok "$(cargo --version)"
}

ensure_c_toolchain() {
  step "C/C++ build tools"
  if have cc || have gcc || have clang; then
    ok "compiler: $(command -v cc || command -v gcc || command -v clang)"
    return
  fi
  if have apt-get; then
    warn "installing build-essential (sudo)"
    sudo apt-get update -y
    sudo apt-get install -y build-essential pkg-config
  elif have dnf; then
    warn "installing Development Tools (sudo)"
    sudo dnf groupinstall -y "Development Tools"
  elif have pacman; then
    sudo pacman -S --needed --noconfirm base-devel
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    warn "installing Xcode Command Line Tools"
    xcode-select --install || true
  else
    fail "No C compiler and no known package manager"
    exit 1
  fi
  ok "C toolchain ready"
}

ensure_nvidia() {
  step "NVIDIA GPU driver"
  if ! have nvidia-smi; then
    fail "nvidia-smi not found — install NVIDIA proprietary drivers"
    warn "Ubuntu: sudo ubuntu-drivers autoinstall   (then reboot)"
    warn "Or: https://www.nvidia.com/Download/index.aspx"
    return 1
  fi
  ok "nvidia-smi present"
  nvidia-smi | head -n 15 || true
  return 0
}

ensure_cuda() {
  step "CUDA Toolkit (driver + NVRTC — required at runtime)"
  if have nvcc; then
    ok "$(nvcc --version | grep -i release | head -n1 | xargs)"
  elif [[ -n "${CUDA_PATH:-}" && -d "$CUDA_PATH/bin" ]]; then
    ok "CUDA_PATH=$CUDA_PATH"
    export PATH="$CUDA_PATH/bin:$PATH"
  elif [[ -d /usr/local/cuda/bin ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
    ok "using /usr/local/cuda"
  else
    warn "CUDA Toolkit not detected (need NVRTC for runtime kernel compile)"
    warn "This project targets CUDA 13.x (cudarc feature cuda-13020)"
    warn "Install: https://developer.nvidia.com/cuda-downloads"
    if have apt-get; then
      warn "Ubuntu example (adjust version to match NVIDIA docs):"
      warn "  wget https://developer.download.nvidia.com/compute/cuda/repos/..."
      warn "  or: sudo apt-get install -y nvidia-cuda-toolkit   (may be older than 13.x)"
    fi
    return 1
  fi

  # NVRTC shared library
  if ldconfig -p 2>/dev/null | grep -q nvrtc; then
    ok "libnvrtc visible to linker"
  elif [[ -n "${CUDA_PATH:-}" ]] && ls "$CUDA_PATH"/lib64/libnvrtc.so* >/dev/null 2>&1; then
    ok "libnvrtc under $CUDA_PATH/lib64"
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
  else
    warn "libnvrtc not found — runtime NVRTC compile may fail"
  fi
  return 0
}

build_release() {
  step "cargo build --release"
  cargo build --release -p taraference
  local bin="$ROOT/target/release/taraference"
  if [[ -x "$bin" ]]; then
    ok "$bin"
  else
    fail "binary missing: $bin"
    exit 1
  fi
}

download_models() {
  step "Download GGUF models → $MODELS_DIR"
  local bin="$ROOT/target/release/taraference"
  local args=(--download "$MODELS" --models-dir "$MODELS_DIR")
  if [[ "$FORCE" -eq 1 ]]; then
    args+=(--force)
  fi
  "$bin" "${args[@]}"
}

# ── main ────────────────────────────────────────────────────────────────────
echo "taraference install"
echo "repo: $ROOT"

ensure_rust
ensure_c_toolchain
GPU_OK=0
CUDA_OK=0
if ensure_nvidia; then GPU_OK=1; fi
if ensure_cuda; then CUDA_OK=1; fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  if [[ "$CUDA_OK" -eq 0 ]]; then
    warn "CUDA incomplete — build may work; runtime needs NVRTC"
  fi
  build_release
fi

if [[ "$SKIP_MODELS" -eq 0 && "$SKIP_BUILD" -eq 0 ]]; then
  download_models
elif [[ "$SKIP_MODELS" -eq 0 && "$SKIP_BUILD" -eq 1 ]]; then
  warn "Skipping models (need binary). Run without --skip-build or: cargo run --release -- --download"
fi

step "Summary"
cat <<EOF
  Build:   cargo build --release
  Binary:  $ROOT/target/release/taraference
  Models:  $MODELS_DIR

  Run chat:
    ./target/release/taraference models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

  Serve API:
    ./target/release/taraference models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --serve 3000

  Profile:
    ./target/release/taraference models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --profile

  Re-download models only:
    ./target/release/taraference --download
EOF

if [[ "$GPU_OK" -eq 0 || "$CUDA_OK" -eq 0 ]]; then
  warn "Fix GPU driver / CUDA Toolkit, then re-run or cargo build --release"
  exit 2
fi

ok "Install finished."
exit 0
