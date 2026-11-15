# SAM 3.1 tracking experiment — 2026-11-15

Host: HOME-PC, GeForce RTX 5070 Ti Laptop GPU, 12227 MiB VRAM, WSL2.

Test: examples/beach_st_lenina_20250622_g1_005.mp4 (6.006 s), pre-rendered to 640x360, 29.97 FPS, 180 frames.
Prepared video: /tmp/volley-sam31/sample_640x360.mp4
SHA-256: 8f31e060717e9ff03750fb7200804ac63e1bf5378c5abdbe8dd9f72e378bdb6c

## Real baseline GPU measurement

Source: real RTMDet-Nano + RTMPose PersonEngine and ProvisionalTracker, ONNX Runtime 1.22 GPU. Timing includes video decode/inference/tracking, but excludes model initialization.

| Metric | RTMDet+RTMPose+Rust | SAM 3.1 native | SAM+RTMPose+Rust |
| --- | ---: | ---: | ---: |
| Actual measured FPS | 9.589 | not measured | not measured |
| Frames | 180 | not measured | not measured |
| Unique provisional Track IDs | 8 | not measured | not measured |
| IDs observed in >=12 frames | 6 | not measured | not measured |
| Total boxes | 732 | not measured | not measured |
| Mean boxes per frame | 4.067 | not measured | not measured |
| Frames with >=4 people | 139 | not measured | not measured |
| Frames with zero people | 0 | not measured | not measured |
| Internal track gap events | 30 | not measured | not measured |
| Missing frames inside a track lifespan | 221 | not measured | not measured |
| Maximum intra-track gap | 72 frames | not measured | not measured |

These are unsupervised diagnostics, NOT proven player recall, precision or identity switches. The sample may show bystanders.

## Implemented, verified

- scripts/experiments/sam31/run_sam31.py: official SAM 3.1 predictor wrapper, syntax verified, not executed because checkpoint is gated
- rust/volley-app/examples/bench_person_tracks.rs: Rust baseline, release compiled, real CUDA inference succeeded
- rust/volley-app/examples/bench_detector_swap.rs: SAM bbox -> RTMPose -> same Rust tracking, release compiled. Compatibility tested on 180 frames using COPIED BASELINE BOXES, NOT actual SAM output.
- scripts/experiments/sam31/compare_tracking.py: frame-level diagnostics, three unit tests pass

The mock adapter smoke test needed 4.948 seconds for 732 RTMPose candidates, 180 frames. This is NOT SAM model time or SAM speed, so it is not reported in the comparison table.

## Blocker

Model weights at https://huggingface.co/facebook/sam3.1 (file sam3.1_multiplex.pt) require Hugging Face approval/login. On HOME-PC, no HF token was present; direct request for weights returned HTTP 401. Real SAM 3.1 benchmark and performance claims are blocked.

User action: request/accept gated-model terms and authenticate using hf auth login locally; do not share tokens in chat. Then run the prepared model experiment and compare all three output JSONLs. Production Rust Studio has NOT been changed.
