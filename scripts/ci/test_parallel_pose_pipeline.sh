#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
source="$VOLLEY_AB_SOURCE"
models="$VOLLEY_AB_MODEL_ROOT"
[[ -s "$source" && -f "$models/models/rf-detr-medium.onnx" ]] ||
  { echo "Missing video or RF-DETR model" >&2; exit 2; }
bundle="$HOME/.local/share/volley-rust/ort-1.22.0"
[[ -s "$bundle/lib/libonnxruntime.so.1.22.0" ]] ||
  { echo "Missing ORT CUDA runtime" >&2; exit 2; }
export PATH="$HOME/.cargo/bin:$PATH"
export ORT_DYLIB_PATH="$bundle/lib/libonnxruntime.so.1.22.0"
export LD_LIBRARY_PATH="$bundle/lib:$(printenv LD_LIBRARY_PATH || true)"
for directory in "$bundle/deps/nvidia/"*/lib; do
  if [[ -d "$directory" ]]; then
    LD_LIBRARY_PATH="$directory:$LD_LIBRARY_PATH"
  fi
done
export LD_LIBRARY_PATH
export VOLLEY_AB_VARIANT="$(printenv VOLLEY_AB_VARIANT || echo pipeline)"
export VOLLEY_AB_SAMPLE_START_SEC="$(printenv VOLLEY_AB_SAMPLE_START_SEC || echo 28)"
export VOLLEY_AB_SAMPLE_DURATION_SEC="$(printenv VOLLEY_AB_SAMPLE_DURATION_SEC || echo 4)"
export VOLLEY_ZERO_COPY_PRIMARY=1
export VOLLEY_PRIMARY_RALLY_PREFILTER=1
export VOLLEY_PERSON_DETECTOR=rfdetr-medium
cd "$repo/rust"
cargo test -p volley-app original_pose_single_decode_benchmark --offline --locked -- --ignored --nocapture
