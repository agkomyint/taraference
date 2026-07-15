#!/usr/bin/env bash
# One-command setup for taraference (Linux).
#   git clone … && cd taraference && ./scripts/install.sh
#
# No flags required. Optional:
#   --skip-models   skip GGUF download
#   --force         re-download models

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_MODELS=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --skip-models) SKIP_MODELS=1 ;;
    --force) FORCE=1 ;;
    -h|--help)
      echo "Usage: ./scripts/install.sh [--skip-models] [--force]"
      exit 0
      ;;
  esac
done

MODELS_DIR="$ROOT/models"

step() { printf '\n\033[36m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mOK\033[0m  %s\n' "$*"; }
warn() { printf '  \033[33m!!\033[0m  %s\n' "$*"; }
fail() { printf '  \033[31mXX\033[0m  %s\n' "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

echo "taraference setup"
echo "repo: $ROOT"

# ── Rust ────────────────────────────────────────────────────────────────────
step "Rust"
if have cargo; then
  ok "$(cargo --version)"
else
  warn "installing rustup…"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  source "${CARGO_HOME:-$HOME/.cargo}/env"
  have cargo || { fail "cargo not on PATH — open a new shell and re-run"; exit 1; }
  ok "$(cargo --version)"
fi
# ensure cargo on PATH in this shell
if [[ -f "${CARGO_HOME:-$HOME/.cargo}/env" ]]; then
  # shellcheck disable=SC1091
  source "${CARGO_HOME:-$HOME/.cargo}/env"
fi

# ── C toolchain ─────────────────────────────────────────────────────────────
step "C/C++ tools"
if have cc || have gcc || have clang; then
  ok "compiler present"
else
  if have apt-get; then
    sudo apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential pkg-config curl ca-certificates
  elif have dnf; then
    sudo dnf groupinstall -y "Development Tools"
  else
    fail "install a C compiler (build-essential), then re-run"
    exit 1
  fi
  ok "compiler installed"
fi

# ── GPU / CUDA (informational — not required to *build*) ────────────────────
step "GPU / CUDA (for running models)"
if have nvidia-smi; then
  ok "nvidia-smi found"
  nvidia-smi -L 2>/dev/null || true
else
  warn "no nvidia-smi — you can still build; inference needs an NVIDIA GPU + driver"
fi
if have nvcc; then
  ok "$(nvcc --version 2>/dev/null | grep -i release | head -1 | xargs)"
elif [[ -d /usr/local/cuda ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
  ok "CUDA at /usr/local/cuda"
else
  warn "CUDA toolkit / NVRTC not detected — runtime kernel compile needs CUDA 13.x toolkit on the host"
fi

# ── Build ───────────────────────────────────────────────────────────────────
step "Build (cargo build --release → tarafer)"
cargo build --release -p taraference
BIN="$ROOT/target/release/tarafer"
[[ -x "$BIN" ]] || { fail "missing $BIN"; exit 1; }
ok "$BIN"

step "Install onto PATH (~/.local/bin)"
"$BIN" install || warn "tarafer install failed — binary still at $BIN"

# ── Models ──────────────────────────────────────────────────────────────────
if [[ "$SKIP_MODELS" -eq 0 ]]; then
  step "Download models → models/"
  args=(--download all --models-dir "$MODELS_DIR")
  [[ "$FORCE" -eq 1 ]] && args+=(--force)
  "$BIN" "${args[@]}"
else
  warn "skipped model download"
fi

# ── Done ────────────────────────────────────────────────────────────────────
step "Done"
cat <<EOF
  Binary:  $BIN  (also try: tarafer on PATH after install)
  Models:  $MODELS_DIR

  Chat:
    tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

  Server:
    tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --serve 3000

  Update prebuilt later:
    tarafer update
EOF
ok "setup finished"
