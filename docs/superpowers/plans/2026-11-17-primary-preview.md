# Per-video primary preview implementation plan

Goal: Save one browser-compatible MP4 per original video during shared primary decoding, without concatenation or changing model pixels.

Architecture: Extend the existing shared FFmpeg owner with an optional third output before model FPS/size conversion. The preparation owner publishes source-bound, verified preview receipts. The playback adapter requests the matching per-video artifact before using existing fallback playback.

Constraints: Rust owns cache selection and publication. Preserve original files and existing evidence schemas. No comments added to code. Runtime and build artifacts use isolated directories. Baseline: 3730cd8.

Review focus: multiple videos; cancellation; portrait rotation/audio; native FPS versus 30-FPS evidence; old analysis reuse without preview.

- [x] Add failing HTTP regression for read-only per-video preview discovery.
- [x] Add source-bound preparation plan and atomic publication/cleanup tests.
- [x] Extend shared decoder with optional MP4 output; compare every model pixel with preview disabled, test audio, native FPS, tail, failure and cancellation on synthetic fixtures.
- [x] Enable output in joint inference and publish only after successful complete decode and unchanged source.
- [x] Add read-only discovery/media API and adapter selection with stale-load protection; test two videos and missing artifacts.
- [x] Run fmt/check/workspace tests, WASM and ownership/roadmap checks, independent review, release build; journal and commit tested increments.
- [x] Integrate without losing existing changes, check active jobs and safely restart Studio.

Nonshared analysis and old reused evidence retain existing playback fallback; no neural rerun for a missing preview. Prepared copies are derived playback data, not inference inputs.
