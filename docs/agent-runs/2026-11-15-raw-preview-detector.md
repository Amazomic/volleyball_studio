# 2026-11-15 · Raw preview shows saved GPU detector

## Context
The raw preview already used immutable GPU fusion candidates and a model-box visibility checkbox, but the label was tied to the server's currently configured detector. Switching to older analyses could therefore present RTMDet-Nano detections as RF-DETR.

## Implemented
- rust/volley-app/src/native_preview_overlay.rs: inspect each selected fusion's original person GPU job manifest, validate source identity and SHA against fusion metadata, return person_detector in GPU overlay response. Unknown/missing job sources are not silently relabeled; runs without old person_pipeline field predate RF-DETR and map to RTMDet-Nano.
- rust/volley-app/static/studio.js: bind the existing raw player checkbox's label to the saved fusion model and preserve independent on/off and confidence settings when changing analyses. The display shows only raw GPU candidates, never CPU-derived boxes.
- scripts/ci/test_studio_raw_detector.mjs: UI behavior regression covering RF-DETR, RTMDet, unselected fusion and checkbox preservation.

## Checks
- PASS: cargo test -p volley-app --offline native_preview_overlay::tests (13 tests)
- PASS: cargo check --workspace --offline
- PASS: node --check rust/volley-app/static/studio.js
- PASS: node scripts/ci/test_studio_raw_detector.mjs
- PASS: node scripts/ci/test_studio_identity_sync.mjs
- PASS: git diff --check on edited source files
- Not caused by this change: cargo fmt --all -- --check identifies formatting drift elsewhere in the repository; scripts/ci/test_volley_web_wasm.mjs reports pre-existing untranslated UI static text budget overrun. Do not edit unrelated source files to hide these issues.

## Follow-up
The detector is evaluated on sampled source frames (~3 Hz); raw preview intentionally does not synthesize detection rectangles on intervening frames.
## Deployment
- Deployment: offline release build passed, safely restarted idle Studio PID 42137 to PID 43439, and verified HTTP 200 for /studio and /studio/app.js with the new raw model selector.
