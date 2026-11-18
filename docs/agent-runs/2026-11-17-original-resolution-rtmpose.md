# 2026-11-17 — RTMPose: native-resolution source and stable original coordinates

## Scope and source
- Baseline: d2eab30e6ad8b6ea5c967a1978bbbbeeb80f2571, separate worktree agent/pose-original-crop-20261117.
- RF-DETR remains a 576x576 ONNX detector using its current square resize. Its bounding boxes are normalized x/y in [0,1] and are mapped back to original-display coordinates without applying square pixel scaling to pose joints.
- VBallNet remains 512x288 grayscale. The player-detector input stays at 640x360 until a separately benchmarked detector preprocessing change is approved.
- RTMPose now reads RGB24 pixels from original-resolution decoded frames, crops with the original width/height, normalizes the 17 keypoints to that same original-display width/height, and preserves the existing JSONL keypoint/box contract. Portrait and landscape source orientations use video probe display dimensions.
- The existing primary shared NVDEC decoder still handles VBallNet + RF-DETR. RTMPose currently runs a second sequential original-resolution decode, with bounded one-frame memory and per-frame source checks, rather than using the lossy 640x360 RGB branch. A future CUDA ROI crop from shared NVDEC frames should remove the second decoder and CPU transfer after measured benchmarking.
- The pose manifest records pixel_source_version=original_full_resolution_rgb24_v1, pixel_dimensions, coordinate_system=normalized_original_display_v1. The GPU profile identity was bumped to v6, so older 640x360 pose runs cannot be silently reused as original-resolution observations.
- Immutable previous GPU analyses, native Track IDs, CPU replay and downstream UI/stats/montage readers are unchanged. The UI converts normalized joints to display pixels by multiplying by the visible video rectangle.

## Tests
- cargo test --workspace --offline: passed on Windows after generating current Rust/WASM binary.
- cargo check --workspace --offline: passed.
- cargo fmt --all -- --check: passed.
- node scripts/ci/test_volley_web_wasm.mjs: passed.
- node scripts/ci/check_rust_ownership.mjs: passed.
- node scripts/ci/check_rust_roadmap.mjs --check --self-test: passed.
- Synthetic FFmpeg original RGB test and normalized joint coordinate tests for 640x360, 1080p, 4K and portrait: passed.

## Caveats
- Real CUDA/NVDEC end-to-end pose quality has not been benchmarked on a complete match; this change is not a measured accuracy improvement yet.
- Separate native-resolution decode increases memory bandwidth and may increase wall-clock GPU analysis duration; GPU cropping/zero-copy is future optimization.
- Only deploy/restart the main Studio from an integrator after checking active jobs, dirty checkout and compatibility with concurrent edits.
