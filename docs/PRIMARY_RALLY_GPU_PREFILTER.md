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
5. On the native NVDEC path, seek to the preceding keyframe and demux only
   the selected analysis range for the VBallNet pass. For RF-DETR/RTMPose,
   seek to the preceding keyframes of each padded gameplay interval and stop
   demux after the interval (merging overlapping GOPs). Non-game gaps do not
   enter the hardware decoder; empty evidence windows still preserve the
   continuous original analysis timeline. RTMPose retains only sampled source
   pixels inside selected intervals, uses original-resolution video pixels
   and does not invent touches from the ball mask. The existing nine-frame
   evidence and CPU ByteTrack replay contracts remain unchanged.

Saved `native_gpu_runs/<ball_job_id>/rally_mask.json` contains source identity,
analysis range, detected/merged/uncertain/scanning intervals and selected-frame
count. Pipeline progress identifies the ball scan and the person/pose scan;
pipeline status includes `gpu_rally_mask`. The standalone CPU training preview
still lets the user change the 1.5 s threshold and <4 s merge gap without
rerunning VBallNet; custom values take precedence over defaults.

## Stationary ball boundary (2026-11-22)

The primary and Debug continuous-ball rally algorithms share the Rust
movement filter in rust/volley-app/src/cpu_retrack/ball_track.rs. The
new stationary_ball_ms setting defaults to **1000 ms** and accepts
200–10000 ms. When the observed ball remains in a region no larger than
3.5% of normalized frame width and height for the configured duration
(with at least 70% observed samples), the region is classified as a
stationary hold, rather than a rally. Isolated one-frame jumps are ignored.
The same boundary prevents the 4-second interval merge from bridging a
recognized stationary hold. Candidate moving intervals must also cover
at least 10% of a frame dimension in horizontal or vertical displacement.

GPU analysis and Debug have independent controls and both call this
Rust detector, using cached VBallNet/CPU replay evidence respectively.
The primary GPU mask records the threshold, stationary intervals and
movement_filter_version=stationary_ball_v1, and changing this setting
invalidates the prior GPU profile match; old immutable evidence is not
rewritten. The 1-second person-inference scan padding remains separate
from the saved unpadded rally boundaries. The safety fallback to scan
the full permitted range when no reliable moving interval exists is retained.

The threshold is heuristic and depends on detector stability and camera
motion. Real-game verification for ~18:51.2 and ~19:59.2 remains separate
from unit tests and must not be claimed without inspecting the video.

## Postprocessing boundary owner (2026-11-22)

The Studio Full/Post pipeline uses the saved primary VBallNet GPU mask as
the sole source of rally start/end frames. It reads the primary source file
from native_gpu_runs/<ball_job_id>/rally_mask.json and checks video
provenance, analysis range, ball-job ID and mask SHA-256. The merged primary
intervals scope CPU Touch processing. A lightweight adapter publishes the
same primary boundaries as a versioned rally timeline with already-computed
Touch references and stable candidate IDs. It reads the Touch JSONL once;
there is no additional replay-frame pass for rally events, rally segmenter,
court/surface event calculation, or dedicated postprocessing stage. The eager
Track ID index scan is no longer invoked by the GPU rally adapter: ReID may
construct the same verified index later as part of player assignment.
Legacy rally snapshots remain readable. Primary intervals are clipped only for the 0–8 trailing frames
not represented by complete nine-frame GPU fusion/replay windows. The
original GPU boundaries remain saved.

Primary interval starts separate Touch evaluation windows; the Studio CPU
pipeline does not run a leading-ball tracker or its former discovery-rallies
-> refined touches -> discovery-rallies cycle. It applies the VBallNet
confidence threshold and bounded local artifact rejection only, with the
versioned vballnet_local_artifacts_v1 policy and no leading-ball Track IDs.
Preview continues limited per-rally short-gap interpolation without a leading
tracker. Quick Reel no longer performs a second leading-ball refinement pass.
The lightweight snapshot retains rally_run_id, Touch lineage and existing
preview/montage references. It uses materialization_version =
primary_gpu_intervals_direct_v1; ground/net and court event context are
explicitly marked not computed. Per-rally ball-coverage statistics are not
recalculated: the legacy numeric fields remain zero for schema compatibility
and ball_coverage_not_recomputed=true distinguishes unknown coverage from
an observed absence of the ball. Timeline boundaries always reflect the GPU
mask, even when no Touch was found. GPU runs without a saved primary mask cannot
enter the Studio postprocessing path; rerun a compatible primary analysis.
Old saved rally runs keep their original policy. Independent manual/training
CPU endpoints retain their existing snapshot-specific behavior.

## Control / safety

The shared primary GPU path always executes two passes: VBallNet first,
then RF-DETR / RTMPose limited by its verified rally mask. There is no
one-pass opt-out. Other standalone GPU compatibility paths remain separate.
The selected mask deliberately includes ambiguous ball motion to limit
missed gameplay.

## Remaining acceptance work

- The primary source contract still normalizes to 30 FPS; native frame PTS/VFR
  end-to-end correspondence and the zero-copy timeline require independent
  real-media validation.
- Ball and people decode separately. NVDEC GOP preroll is still required,
  but the selected video trim and long pauses between rallies no longer
  incur hardware decoding outside their keyframe-aligned segments.
  Source PTS/keyframe metadata are still scanned once before decoding.
  The FFmpeg RGB fallback retains full-range decoding with an inference
  mask; its sparse demux is a separate optimization. Checkpoint-driven
  restart after an already completed ball pass is not implemented.
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
The original one-pass diagnostic opt-out was removed on 2026-11-22;
historical A/B measurements above remain valid experimental records.

## NVDEC bounded demux acceptance (2026-11-21)

- NVDEC now uses FFmpeg input `-ss` and bounded `-t` stream copy. Keyframe
  PTS maps independently decoded GOPs back to original source-frame indices.
  Gaps remain empty evidence windows; RTMPose and nine-frame fusion alignment
  are preserved.
- Windows CUDA NV12 test: decoded cropped frames match every corresponding
  frame of the full-source GPU decode; bounded demux emits fewer frames.
  Rust unit tests cover keyframe selection and disjoint/coalesced GOP plans.
- Linux GPU/ORT 1.22 test: actual models and nonzero analysis trim from 2 s
  successfully run VBallNet, RF-DETR, RTMPose, fusion and CPU replay.
  On isolated 33.97 s black/game/black/game/black test media trimmed at 1 s,
  VBallNet found 2 moving-ball intervals; **588 of 989** source timeline
  frames were selected for player inference (two separate windows), with
  576 player proposals, 546 poses and successful CPU replay.
- The same short A/B source measured 17.368 s with two-pass filtering and
  17.692 s with one-pass GPU. This does not establish a significant speedup
  on full matches. VBallNet raw evidence SHA was identical in both runs.
- Full Windows check-all passed fmt, clippy, Rust tests and 92 Node gates.
  Windows native model test could not start with outdated ORT DLL 1.17.1;
  model integration succeeded under Linux with ORT 1.22.
- Remaining limits: long-match timing and independent event accuracy are
  not measured; FFmpeg RGB fallback is not yet GOP-sparse; uncommon codec
  timestamp layouts may require that safe fallback.
