# 2026-11-18 — native primary preview rotation failure

- **Symptom:** GPU primary analysis finishes decoding, then fails with `Invalid primary preview dimensions, rotation or timeline` while publishing its MP4 cache. The GPU evidence was subsequently discarded by the joint run's pending-folder guard.
- **Root cause (reproduced with a HEVC camera recording):** Source has a -90-degree display matrix. NVDEC performs the physical transpose, while FFmpeg/NVENC preserves the original display matrix even with `-map_metadata -1 -metadata:s:v:0 rotate=0`. Result: correct 720x1280 encoded frames but `ffprobe` still reports rotation=-90, leading to rejection (and potentially double rotation in browser playback).
- **Fix:** Add input-side `-display_rotation:v:0 0` for NVDEC while preserving the explicit pixel transpose. This clears inherited display metadata. Keep strict validation but include observed codec, dimensions, rotation, duration and aspect mismatch in the error. A preview-only publication failure now logs a warning rather than invalidating completed GPU evidence; cancellation and changed original source remain fatal.
- **Isolation:** Implemented in dedicated worktree `fix/20261118-primary-preview-rotation`, without modifying existing in-progress edits in the primary checkout.
- **Verification:** Two 0.5-second NVDEC/NVENC transcodes from the same real rotated source: old command returned rotation=-90; input-side override returned no rotation side-data. Rust unit and hardware integration tests cover the command and output respectively.
- **User assets:** All original videos, projects, results and active jobs left untouched.
## Completed checks

- cargo fmt --all -- --check: passed.
- cargo check -p volley-app --offline: passed (compiled with existing generated local WASM artifact).
- cargo test -p volley-media --offline: 48 passed; 3 ignored (41 + 7).
- NVIDIA-only shared_preview_nvdec_nvenc_preserves_rotation_audio_and_native_fps: passed for 0, 90, -90 and 180 degrees.
- cargo test -p volley-app --offline: passed (354 unit tests).
