# Primary GPU rally prefilter - experimental (2026-11-21)

The native full/GPU primary path now supports two model passes over each
video's own saved analysis range:

1. Decode the allowed range and run VBallNet at canonical 30 FPS.
2. Preserve all ball evidence and derive stable moving-ball intervals
   (at least **1.5 s**, inclusive).
3. Merge detected intervals only when the uncovered gap is **strictly less
   than 4 s**. Preserve detected and merged intervals separately.
4. Pad scanning intervals by 1 s before and after without altering the
   detected rally boundaries. Short movement fragments (at least 0.5 s)
   outside confirmed intervals are retained as **uncertain** scan candidates,
   not confirmed rallies. When the ball is absent or no reliable interval
   exists, retain the entire allowed video interval.
5. Decode the allowed range a second time and run RF-DETR only for sampled
   frames in the selected mask. RTMPose only retains source pixels inside selected intervals (other sampled frames remain empty and time-aligned); it still receives original-resolution
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

## GPU timing audit after stage cleanup (2026-11-21)

Stage transitions now keep one meaningful video-validation stage and one GPU stage
per actual pass, without a fake zero-second "source verified" or "combined GPU"
stage. The saved mask and pipeline event log expose selected/total frame counts
and the fraction eligible for RF-DETR. CUDA RTMPose receives no full-size frame
copy outside the selected mask; the FFmpeg fallback also omits out-of-rally pose
interpolation/crops while preserving sampled blank evidence rows.

Controlled WSL CUDA / NVDEC zero-copy smoke with an isolated 640x360 H.264
synthetic scene (black/play/black; 12-second view, 362 decoded frames):
- Default person sampling interval 1 frame: old one-pass 24.936 seconds;
  filtered two-pass 21.894 seconds (12.2% less GPU stage wall time).
  The mask selected 208/362 frames; both paths yielded 2160 player proposals
  and 2145 pose observations, with CPU replay completing.
- Sampling interval 10 frames, 18-second view: old one-pass 13.946 seconds;
  filtered two-pass 17.293 seconds (24.0% slower). The extra full decode can
  outweigh saved detector calls at low RF-DETR sampling rates.
- A filtered FFmpeg/non-zero-copy integration pass also succeeded with
  208 selected frames, 216 detected-player proposals and 215 poses.
  The source's trimmed 9-frame tail explains replay frame count differences.

These timings are only synthetic, not evidence of faster real-game analysis
or player/rally/Touch precision. The detector defaults to a one-frame
sampling period; custom larger strides may not benefit. The GPU pass is
experimental until real-match precision, GPU input parity and matched A/B
measurements are accepted. The Studio process on 127.0.0.1:8875 was not
running during this audit and no user's active jobs were interrupted.
## 60 FPS / Full HD acceptance and startup crash diagnostic

The failed existing Studio launches were not slow-running GPU jobs: three
persisted pipeline statuses stopped at 8% and the Linux kernel independently
recorded repeated volley-app tokio-worker SIGSEGV signals at the same invalid
code address. The affected 58.17-second source is 1920x1080 H.264 60 FPS.
The deployed binary was older than this PR; its crash is consistent with
the CUDA ONNX Runtime tensor/allocator lifetime defect fixed in this branch.

An isolated, consistent single-encoder H.264 1920x1080/60 FPS 20-second
test contains a stationary background and a volleyball scene:
- Both NVDEC CUDA zero-copy variants complete their GPU, fusion, RTMPose,
  CPU replay, Touch discovery and rally pipeline without SIGSEGV.
- VBallNet observed the ball on 298 of the 600 canonical 30 FPS frames;
  the mask detected 2 motion intervals and selected 312/600 frames
  (52% of the total) without a full-analysis fallback.
- On the same file, pose_stride/person_stride=1 and one GPU: 38.515 s
  for the original one-pass mode versus 31.212 s for the filtered
  two-pass mode (18.96% faster). VBallNet evidence SHA-256 was exactly
  identical between the two runs.
- The full-frame pass returned 3612 RF-DETR candidates and 3606 pose
  crops; filtered pass returned 3288 candidates and 3284 pose crops.
  These intentionally different counts do NOT establish that all
  out-of-rally player observations are irrelevant. Independent labeled
  match-level precision/recall remains NOT TESTED.
- A deliberately concatenated H.264 fixture with incompatible encoder
  sequence parameters failed NVDEC format validation; replacing it with
  a single-encoder fixture resolved that separate test-input issue.

The CUDA FFmpeg-vs-GPU image-preprocessing evidence SHA mismatch in the
older strict legacy-to-shared A/B test is still an open, independent
numerical parity issue. Do not relabel byte-level differences as parity.
This branch provides a documented opt-out using
VOLLEY_PRIMARY_RALLY_PREFILTER=0 and does not change published master.