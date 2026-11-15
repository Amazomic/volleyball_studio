# 2026-11-15 — Players/Preview ID synchronization

## Context and baseline
Read AGENTS.md, parallel-work protocol, improvement checklist and prior identity handoffs. Existing dirty Rust branch `17c1f79` preserved; single connected HOME-PC, no independent writers launched.

Live project `d6451abac405`, newest CPU replay `21cb611628fa4996a97ed567de7da5ee`, rally run `f45f766f1fa444b7b85c67cae7dec0d5`. Inspected both authenticated local API responses:
- `GET /api/players/native/tracks` and `GET /api/cpu/native/rallies/overlay-player-labels` have the **same** assignment revision `7bb7ee48bd080b62f5dcff4f47bf0b2ee1dd3681a91afa9f007fcb716903290f`, same replay, and matching ID-to-name entries.
- On the UI, Players `loadVideos()` returned early when source ID was unchanged without reloading player rows, even if their revision had changed server-side; this also broke the refresh button's freshness guarantee.
- A separate visual discrepancy: `native_preview_overlay::extract_frames()` emitted interpolated player boxes, but track-photo/identity review excludes interpolation as independent evidence. In the real replay, measured-versus-interpolated ID conflicts were observed, including frame 1053: inferred ID12 and measured ID13 overlapped with IoU~0.553. Other examples include inferred ID2 shadowing observed ID1. Such visual ghosts can display apparent label swaps even while the saved assignments are consistent.

## Changes (only owner files)
- `rust/volley-app/static/studio_tracks.js`: re-open/refresh same replay now refetches canonical tracks and assignments while preserving the cards; if the server identity revision changed, dispatch `volley:track-assignments-updated` so the preview reloads labels.
- `rust/volley-app/src/native_preview_overlay.rs`: extract the player overlay via `overlay_players()`. Always preserve observed detection; exclude a visually interpolated ID only when its box overlaps or nearly coincides with an independently observed box. Preserve inferred continuity elsewhere.
- `scripts/ci/test_studio_identity_sync.mjs`: VM-regression for fresh same-source gallery read, revision-scoped preview invalidation, and no invalidation on identical revision.
- Added Rust regression `inferred_player_labels_never_cover_a_different_measured_player` to the overlay module.

## Validation and constraints
- Server label-vs-photo gallery revision/ID parity verified on live latest analysis.
- JavaScript syntax, new UI VM regression, existing gallery test and scoped `git diff --check` passed.
- `cargo test -p volley-app native_preview_overlay::tests:: --offline --locked`: 12 passed.
- `cargo test -p volley-app studio_web::tests:: --offline --locked`: 5 passed.
- Release build/restart and live endpoint check performed separately and should be recorded when complete.
- No changes to original media, personal photo library, saved replay, player assignments, CPU pipeline selection or historical analysis.
- **Limit**: this is a display/revision sync fix; if an actually measured Track ID follows the wrong physical person, it requires further CPU tracker/ReID improvements. Avoid claiming this patch has corrected all real-world athlete names.

## Deployment and live verification
- Release cargo build succeeded. Verified Studio idle before graceful stop and detached restart. New process serves at `http://127.0.0.1:8875/studio`.
- `GET /studio/tracks.js` confirms new same-source refresh and identity-revision invalidation code is served (HTTP 200).
- Live `GET /api/cpu/native/rallies/overlays` for latest real rally, frame 1053: interpolated ID 12 overlapping independently measured ID 13 is **omitted**, while ID 13 and separate IDs 11, 14, 15 remain visible. Previously both 12 and 13 were emitted, making a label ghost.
- Source photo albums, manual assignments, and saved CPU/GPU frame data remain untouched; overlay filtering is presentation-only.
