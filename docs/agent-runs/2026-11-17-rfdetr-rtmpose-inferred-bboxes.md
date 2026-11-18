# RF-DETR interpolation before RTMPose — 2026-11-17

## Scope
- `rust/volley-app/src/pose_inference/interpolation.rs`: bounded and geometry-gated interpolation between **real** RF-DETR observations; never an athlete identity.
- `rust/volley-app/src/pose_inference.rs`: missing proposal boxes are supplied to RTMPose at configured detector/pose sampling frames; original-resolution pixels are used for inference; inferred crop evidence is explicitly tagged `crop_origin=interpolated_box`.
- `rust/volley-app/src/pose_inference/stream.rs`: the shared Unix pose stream uses the same 30-frame lookahead and tags crop origins independently of detector evidence.
- `rust/volley-app/src/native_preview_overlay.rs`: raw GPU overlay includes `raw.players_interpolated`, separate from preserved `raw.players`.
- `rust/volley-app/static/studio.{html,js}`: show/hide toggle limited to RF-DETR Medium, orange dashed rectangle and “Интерп.” label.

## Invariants
- Original GPU player candidates are never overwritten or falsely labeled as observed.
- No extrapolation. Bounding-box interpolation needs two unambiguous real endpoints no more than 30 frames apart, compatible size/motion/overlap, and no duplicate of an actual measured player.
- Interpolated crop causes **real** RTMPose model inference on original image pixels; this is NOT a synthetic pose output. Existing interpolation of preview keypoints is a different operation.
- The RF-DETR visibility toggle is UI-only; it does not change saved GPU data.
- Not all missed detections can be recovered; original and synthetic source origins are exposed for evaluation.

## Verification
- Windows `cargo fmt --all -- --check`: passed.
- Windows `cargo check --manifest-path rust/Cargo.toml -p volley-app --offline`: passed.
- Windows focused `cargo test -p volley-app pose_inference:: --offline`: 9 passed.
- Windows focused `cargo test -p volley-app native_preview_overlay::tests::raw_gpu_ --offline`: 3 passed.
- `scripts/ci/test_studio_rfdetr_interpolation.mjs`, processed pose overlay, rally-only pose UI: passed.
- Windows full `cargo test -p volley-app --offline`: 339 passed, 8 ignored, 1 unrelated failure `storage::tests::shared_storage_statistics_preserve_missing_exact_bytes_and_http_errors` (stats absent-object contract).
- Linux WSL compile: `cargo check -p volley-app --locked --offline` passed. Linux `cargo test -p volley-app --locked --offline pose_inference::stream::tests`: 5 passed.
- Studio restart: attempted through the guarded Windows WSL launcher only after confirming zero active jobs; see operational result separately.

## Integration warning
The local rust branch was dirty BEFORE this work (26 modified tracked files). Some modified source dependencies (original-resolution RTMPose / NVDEC pipeline) have not been committed by other agents. A safe commit/push needs to isolate the author's hunks and merge pending changes first. Do not blindly stage entire shared files.

## Follow-up: false duplicated raw player rectangle (2026-11-18)

Screenshot at 00:15.6 showed an orange interpolated RF-DETR box partly overlaying a blue 93% observed box for the same person. The former overlap guard rejected boxes only for IoU > 0.35; shifted/cropped real boxes can have IoU below that threshold and still depict the same player.

Single source of truth: `rust/volley-app/src/pose_inference/interpolation.rs`, used both by raw GPU overlays and RTMPose proposals. A candidate overlapping any real observation is now rejected for IoU > 0.20 **or** intersection > 40% of the smaller box's area. Conservative false-suppression is preferred to fabricating duplicate bodies on ambiguous crossings. Independent teammates can still be measured on the same frame while a missing player is interpolated.

Checks: Windows 5 interpolation tests + 3 raw GPU overlay tests; Linux 5 interpolation tests + raw GPU overlay test; `cargo fmt --all -- --check`, `cargo check -p volley-app --offline`, RF-DETR raw UI Node test — all passed.
