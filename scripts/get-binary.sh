#!/usr/bin/env bash
# Download the latest (or a given) Linux x86_64 prebuilt binary from GitHub Releases.
#
#   ./scripts/get-binary.sh              # latest → ./taraference
#   ./scripts/get-binary.sh v0.1.0       # specific tag
#   ./scripts/get-binary.sh latest /tmp  # install dir
#
# Requires: curl, tar. Optional: sha256sum for checksum verify.

set -euo pipefail

REPO="${TARAFERENCE_REPO:-agkomyint/taraference}"
TAG="${1:-latest}"
OUT_DIR="${2:-.}"
ASSET="taraference-linux-x86_64.tar.gz"

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
curl -fsSL --retry 3 -o "${TMP}/${ASSET}" "$URL"

# Best-effort checksum if published next to the archive
SUM_URL="${URL}.sha256"
if curl -fsSL --retry 2 -o "${TMP}/${ASSET}.sha256" "$SUM_URL" 2>/dev/null; then
  echo "==> verifying sha256"
  (cd "$TMP" && sha256sum -c "${ASSET}.sha256")
else
  echo "  (no .sha256 asset; skipping verify)"
fi

echo "==> extracting → ${OUT_DIR}/taraference"
tar -xzf "${TMP}/${ASSET}" -C "$OUT_DIR"
chmod +x "${OUT_DIR}/taraference"

echo "OK  ${OUT_DIR}/taraference"
"${OUT_DIR}/taraference" --help 2>/dev/null | head -5 || true
cat <<EOF

Next (needs NVIDIA GPU + CUDA 13.x NVRTC):
  ${OUT_DIR}/taraference --download 0.5b
  ${OUT_DIR}/taraference models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
EOF
