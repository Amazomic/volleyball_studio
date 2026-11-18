#!/usr/bin/env bash
# Install ONLY the previously benchmarked RF-DETR Medium ONNX artifact.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
model="$repo/models/rf-detr-medium.onnx"
expected=18250b5a3a9552ae117d592f8e8c27fe52438d70c914a77a8913fc45d8f29e69
url=https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-medium.onnx
mkdir -p "$repo/models"
if [[ -f "$model" ]]; then
  printf '%s  %s\n' "$expected" "$model" | sha256sum --check --status || {
    echo "Existing RF-DETR model does not match tested SHA-256: $model" >&2
    exit 1
  }
  echo "Already installed and verified: $model"
  exit 0
fi
tmp="$(mktemp "$repo/models/.rf-detr-medium.XXXXXXXX.onnx")"
trap 'rm -f "$tmp"' EXIT
curl --fail --location --retry 3 --connect-timeout 20 --output "$tmp" "$url"
printf '%s  %s\n' "$expected" "$tmp" | sha256sum --check --status || {
  echo "Downloaded model checksum differs from tested version; refusing to install" >&2
  exit 1
}
chmod 0644 "$tmp"
mv "$tmp" "$model"
echo "Installed verified experimental RF-DETR Medium at $model"
