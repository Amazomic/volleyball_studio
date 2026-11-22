#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bundle="$HOME/.local/share/volley-rust/ort-1.22.0"
export PATH="$HOME/.cargo/bin:$PATH"
export ORT_DYLIB_PATH="$bundle/lib/libonnxruntime.so.1.22.0"
export LD_LIBRARY_PATH="$bundle/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for directory in "$bundle/deps/nvidia/"*/lib; do
  if [[ -d "$directory" ]]; then
    LD_LIBRARY_PATH="$directory:$LD_LIBRARY_PATH"
  fi
done
export LD_LIBRARY_PATH
model_root="${VOLLEY_AB_MODEL_ROOT:?Set VOLLEY_AB_MODEL_ROOT to a checkout with model weights}"
test -f "$model_root/models/rf-detr-medium.onnx"
test -f "$bundle/deps/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
fixture="$(mktemp -d -t volley-nvdec-smoke-XXXXXXXX)"
trap 'rm -rf "$fixture"' EXIT
ffmpeg -hide_banner -loglevel error -nostdin -threads 2 -f lavfi \
  -i 'testsrc2=size=640x360:rate=30:duration=5' \
  -c:v libx264 -threads 2 -g 60 -pix_fmt yuv420p -y "$fixture/synthetic.mp4"
export VOLLEY_AB_MODEL_ROOT="$model_root"
export VOLLEY_AB_SOURCE="$fixture/synthetic.mp4"
export VOLLEY_AB_SAMPLE_START_SEC=0
export VOLLEY_AB_SAMPLE_DURATION_SEC=3
export VOLLEY_AB_PERSON_STRIDE=8
export VOLLEY_ZERO_COPY_PRIMARY=1
cd "$repo/rust"
cargo test -p volley-inference cuda_p016_rotation_preserves_expected_luma_in_device_memory --offline --locked -- --nocapture
cargo test -p volley-media bounded_nvdec_demux_preserves_original_frame_pixels --offline --locked -- --nocapture
cargo test -p volley-app zero_copy_hevc_preflight_proves_gpu_backend_ready --offline --locked -- --ignored --nocapture
cargo test -p volley-app primary_rally_prefilter_cuda_integration_smoke --offline --locked -- --ignored --nocapture
