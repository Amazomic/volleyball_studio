# Primary GPU rally prefilter - experimental (2026-11-21)

The native full/GPU primary path now supports two model passes over each
video's own saved analysis range:

1. Decode the allowed range and run VBallNet at canonical 30 FPS.
2. Preserve all ball evidence and derive stable moving-ball intervals
   (at least **1.5 s**, inclusive).
3. Merge detected intervals only when the uncovered gap is **strictly less
   than 4 s**. Preserve detected and merged intervals separately.
4. Pad scanning intervals by 3 s before and after without altering the
   detected rally boundaries. Short movement fragments (at least 0.5 s)
   outside confirmed intervals are retained as **uncertain** scan candidates,
   not confirmed rallies. When the ball is absent or no reliable interval
   exists, retain the entire allowed video interval.
5. Decode the allowed range a second time and run RF-DETR only for sampled
   frames in the selected mask. RTMPose still receives original-resolution
   video pixels; confirmed touches are not manufactured from the ball mask.
   Saved model evidence keeps the original 9-frame window layout for fusion
   and existing CPU ByteTrack replay.

Saved `native_gpu_jobs/<ball_job_id>/rally_mask.json` contains source identity,
analysis range, detected/merged/uncertain/scanning intervals and selected-frame
count. Pipeline progress identifies the ball scan and the person/pose scan;
pipeline status includes `gpu_rally_mask`. The standalone CPU training preview
still lets the user change the 1.5 s threshold and <4 s merge gap without
rerunning VBallNet; custom values take precedence over defaults.

## Control / safety

`VOLLEY_PRIMARY_RALLY_PREFILTER=0` (or `false`) retains the original
one-pass full-video primary GPU inference. The default is two-pass filtering
on the compatible shared primary path. Other existing GPU paths retain their
previous behavior. The selected mask deliberately includes ambiguous ball
motion to limit missed gameplay.

## Remaining acceptance work

- The primary source contract still normalizes to 30 FPS; native frame PTS/VFR
  end-to-end correspondence and the zero-copy timeline require independent
  real-media validation.
- Ball and people decode separately. The mask is saved but checkpoint-driven
  restart from an already completed ball pass has **not** been implemented.
- Explicit GPU-stage tuning controls and per-video resource statistics in the
  UI are not complete; the environment toggle supports A/B diagnostics.
- The CUDA primary input allocation now keeps its ONNX Runtime Allocator
  alive until the GPU tensor has been released. Previously the allocator was
  released first, triggering SIGSEGV when zero-copy tensor owners were dropped.
  The first VBallNet-only scan no longer sends empty samples or a tail into
  RTMPose; only the second scan advances the pose stage.
- HOME-PC retest: native Windows CUDA preprocessing (5/5) and NVDEC (10/10,
  3 ignored) pass; full `scripts/check-all.ps1` passes Rust, Clippy, WASM and
  91 Node gates. In WSL, explicit ONNX Runtime CUDA integration smokes pass
  both with and without NVDEC zero-copy.
- A repo-tracked volleyball GIF was converted to an isolated H264 fixture;
  180 frames were decoded, 174 selected, 216 player proposals and 214 pose
  results published, and CPU replay completed. This validates execution,
  **not** rally/Touch precision or a performance gain.
- The strict old-vs-new A/B raw ball-detection SHA test still fails when
  CUDA GPU preprocessing replaces FFmpeg pixel preprocessing (the person
  evidence SHA matches). GPU numerical parity and match-level quality need
  independent measurement with agreed tolerances, never by silently relaxing
  the existing SHA gate.
- No independent real-match precision/recall, matched A/B GPU timing,
  peak VRAM or power reduction has been verified.
- No existing user videos, project data or the running Studio instance were
  changed. Do not merge into develop until GPU acceptance is completed.
