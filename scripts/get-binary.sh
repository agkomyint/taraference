#!/usr/bin/env bash
# Download the latest (or a given) Linux x86_64 prebuilt **tarafer** binary
# from GitHub Releases and install it onto PATH (~/.local/bin by default).
#
#   ./scripts/get-binary.sh                 # latest → ~/.local/bin/tarafer
#   ./scripts/get-binary.sh v0.2.0          # pin a tag
#   ./scripts/get-binary.sh latest /tmp     # custom install dir
#   ./scripts/get-binary.sh latest .        # current directory only
#
# Requires: curl, tar. Optional: sha256sum for checksum verify.
# Env:
#   TARAFER_REPO / TARAFERENCE_REPO   override owner/repo (default agkomyint/taraference)
#   TARAFER_BIN_DIR                   default install dir when OUT_DIR omitted

set -euo pipefail

REPO="${TARAFER_REPO:-${TARAFERENCE_REPO:-agkomyint/taraference}}"
TAG="${1:-latest}"
# Default: put on PATH location
if [[ $# -ge 2 ]]; then
  OUT_DIR="$2"
else
  OUT_DIR="${TARAFER_BIN_DIR:-${HOME}/.local/bin}"
fi
ASSET="tarafer-linux-x86_64.tar.gz"
BIN_NAME="tarafer"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ "$TAG" == "latest" ]]; then
  URL="https://github.com/${REPO}/releases/latest/download/${ASSET}"
else
  URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
fi

echo "==> downloading ${URL}"
if ! curl -fsSL --retry 3 -o "${TMP}/${ASSET}" "$URL"; then
  # Fallback: older asset name from pre-rename releases
  LEGACY="taraference-linux-x86_64.tar.gz"
  if [[ "$TAG" == "latest" ]]; then
    URL="https://github.com/${REPO}/releases/latest/download/${LEGACY}"
  else
    URL="https://github.com/${REPO}/releases/download/${TAG}/${LEGACY}"
  fi
  echo "    retry legacy asset: ${URL}"
  curl -fsSL --retry 3 -o "${TMP}/${ASSET}" "$URL"
  ASSET_IS_LEGACY=1
else
  ASSET_IS_LEGACY=0
fi

SUM_URL="${URL}.sha256"
if curl -fsSL --retry 2 -o "${TMP}/sum.sha256" "$SUM_URL" 2>/dev/null; then
  echo "==> verifying sha256"
  # Normalize filename in checksum file
  (cd "$TMP" && sed "s/ .*//" sum.sha256 | awk '{print $1"  '"${ASSET}"'"}' > check.sha256 && sha256sum -c check.sha256)
else
  echo "  (no .sha256 asset; skipping verify)"
fi

echo "==> extracting → ${OUT_DIR}/${BIN_NAME}"
tar -xzf "${TMP}/${ASSET}" -C "$TMP"
if [[ -f "${TMP}/${BIN_NAME}" ]]; then
  SRC="${TMP}/${BIN_NAME}"
elif [[ -f "${TMP}/taraference" ]]; then
  SRC="${TMP}/taraference"
else
  echo "error: archive missing tarafer/taraference" >&2
  ls -la "$TMP" >&2
  exit 1
fi
install -m 755 "$SRC" "${OUT_DIR}/${BIN_NAME}"

echo "OK  ${OUT_DIR}/${BIN_NAME}"
"${OUT_DIR}/${BIN_NAME}" --version 2>/dev/null || true
"${OUT_DIR}/${BIN_NAME}" --help 2>/dev/null | head -8 || true

# PATH hint
case ":${PATH}:" in
  *":${OUT_DIR}:"*) echo "OK  already on PATH" ;;
  *)
    cat <<EOF
!!  ${OUT_DIR} is not on PATH. Add it:

  echo 'export PATH="${OUT_DIR}:\$PATH"' >> ~/.bashrc && source ~/.bashrc
  # zsh: use ~/.zshrc instead
EOF
    ;;
esac

cat <<EOF

Next (needs NVIDIA GPU + CUDA 13.x NVRTC):
  tarafer --download 0.5b
  tarafer models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

Update later (same machine):
  tarafer update
  # or: tarafer update --install   # refresh ~/.local/bin/tarafer
EOF
