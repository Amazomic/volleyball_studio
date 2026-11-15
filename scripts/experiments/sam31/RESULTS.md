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


## RF-DETR Small and Medium: actual ONNX CUDA benchmark (2026-11-15)

**Status: both ONNX models actually executed on NVIDIA RTX 5070 Ti Laptop GPU.** The one-command runner and release-built Rust detector-swap adapter both completed successfully. No production Studio code was modified.

### Reproducible sources / conditions

- Models: public, third-party RF-DETR ONNX exports from https://huggingface.co/PierreMarieCurie/rf-detr-onnx , reportedly exported using rfdetr 1.3.0. These are **not official Roboflow-hosted ONNX exports**; validate with labeled athletes before production adoption.
- Small ONNX SHA-256: 0e0817f4cafa479ccba17662a142092932b0b10c98947e7cf60f3badd0f5c219
- Medium ONNX SHA-256: 18250b5a3a9552ae117d592f8e8c27fe52438d70c914a77a8913fc45d8f29e69
- Runtime: ONNX Runtime GPU 1.22.0, CUDA provider, cuDNN 9. Both models use person class slot 1 in sparse COCO logits, confidence threshold 0.28.
- Small 512x512 input, Medium 576x576; both resize the same 640x360 video frames directly to square, with ImageNet mean/std.
- Video SHA-256: 8f31e060717e9ff03750fb7200804ac63e1bf5378c5abdbe8dd9f72e378bdb6c; duration 180 frames / 6.006 seconds at 29.97 FPS.
- Detector model/session initialization excluded from FPS. Full RF-DETR pipeline FPS is calculated from detector stage plus RTMPose+Rust tracker stage, performed serially. Baseline is a fused one-pass native pipeline, so accounting/decode overhead differs slightly.

### Results

| Metric | RTMDet-Nano + RTMPose + Rust | RF-DETR Small + RTMPose + Rust | RF-DETR Medium + RTMPose + Rust |
| --- | ---: | ---: | ---: |
| First real full-pipeline FPS | 9.589 | 10.638 | 12.646 |
| Follow-up full-pipeline FPS after GPU/cache warm-up | 10.989 | 14.775 | 15.812 |
| Standalone warmed ONNX detection FPS | n/a | 47.67 | 46.80 |
| Observed provisional IDs | 8 | 12 | 13 |
| IDs observed on >=12 frames | 6 | 10 | 10 |
| Observed boxes after Rust tracking | 732 | 1289 | 1237 |
| Boxes per frame after Rust | 4.067 | 7.161 | 6.872 |
| Frames with >=4 detected boxes (NOT verified four athletes) | 139 | 180 | 180 |
| Internal track gap events | 30 | 24 | 4 |
| Missing frames *inside same ID lifespan* | 221 | 315 | 14 |
| Maximum same-ID gap (frames) | 72 | 56 | 9 |

At the warmed speeds, RF-DETR Small is ~34% and Medium ~44% faster than the warmed RTMDet control. GPU/cache effects are significant on this short clip; a longer test and more trials are necessary for reliable performance estimates.

The tracker saw 1301 raw Small proposals and 1246 raw Medium proposals. The pose stage received 24 Small and 9 Medium inference errors, plus 638 and 612 proposals without a confident skeleton, respectively. The experiment **retains the original detector rectangle** on pose failures, whereas current RTMDet observations require a confident skeleton. Therefore, results test a detector swap PLUS more permissive pose gating, not just model quality.

**Conclusion:** RF-DETR Medium is promising for uninterrupted person rectangles: 4 internal ID gap events vs 30 in baseline, and maximum gap 9 vs 72 frames. But it also yields 13 vs 8 provisional IDs; this can reflect spectators, duplicate detections or ID fragmentation. An ID can disappear permanently and be replaced without counting as an intra-ID gap. Without per-athlete ground truth, do NOT call these observations ground-truth tracking accuracy or prove that real athlete identities are continuous.

Output artifacts on HOME-PC WSL (not stored in Git):
- /tmp/volley-sam31/rfdetr_small.jsonl, rfdetr_medium.jsonl: raw proposals
- /tmp/volley-sam31/rfdetr_small_pose.jsonl, rfdetr_medium_pose.jsonl: same Rust provisional tracking
- /tmp/volley-sam31/rfdetr_comparison_warm.json: comparison using warmed baseline
- /tmp/volley-sam31/rfdetr_comparison.json: first baseline plus warmed detector runs
- /tmp/volley-sam31/base_warm.jsonl: repeat RTMDet baseline

The SAM 3.1 weight-access blocker remains independent of these finished RF-DETR measurements.
