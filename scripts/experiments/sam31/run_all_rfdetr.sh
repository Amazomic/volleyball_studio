#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
source_repo=/mnt/c/Users/amazomic/Desktop/github/fast-volleyball-tracking-inference
out=/tmp/volley-sam31
venv=/tmp/volley-onnx-venv
video="$out/sample_640x360.mp4"
if [[ ! -f "$video" ]]; then
  mkdir -p "$out"
  ffmpeg -hide_banner -nostdin -y -loglevel error -i "$source_repo/examples/beach_st_lenina_20250622_g1_005.mp4" -vf "scale=640:360,fps=30000/1001" -an -c:v libx264 -crf 18 "$video"
fi
[[ -x "$venv/bin/python" ]] || { echo "Missing isolated ONNX RF-DETR environment: $venv" >&2; exit 2; }
"$venv/bin/python" -c 'import onnxruntime as ort,cv2,numpy; assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA EP unavailable"'
source "$source_repo/scripts/setup_rust_ort_cuda.sh" check
export CARGO_TARGET_DIR=/tmp/volley-sam31-build
if [[ ! -s "$out/base.jsonl" ]]; then
  "$HOME/.cargo/bin/cargo" run --manifest-path "$repo/rust/Cargo.toml" -p volley-app --release --offline --example bench_person_tracks -- "$source_repo" "$video" "$out/base.jsonl"
fi
for size in small medium; do
  echo "=== RF-DETR $size detector ==="
  "$venv/bin/python" "$repo/scripts/experiments/sam31/run_rfdetr_onnx.py" --variant "$size" --model-path "$out/rfdetr-$size.onnx" --video "$video" --output "$out/rfdetr_$size.jsonl"
  echo "=== RF-DETR $size + RTMPose + Rust tracker ==="
  "$HOME/.cargo/bin/cargo" run --manifest-path "$repo/rust/Cargo.toml" -p volley-app --release --offline --example bench_detector_swap -- "$source_repo" "$video" "$out/rfdetr_$size.jsonl" "$out/rfdetr_${size}_pose.jsonl"
done
python3 "$repo/scripts/experiments/sam31/compare_variants.py" \
  --baseline "$out/base.jsonl" \
  --candidate "RF-DETR Small=$out/rfdetr_small_pose.jsonl" \
  --candidate "RF-DETR Medium=$out/rfdetr_medium_pose.jsonl" \
  --output "$out/rfdetr_comparison.json"
echo "Saved: $out/rfdetr_comparison.json"
