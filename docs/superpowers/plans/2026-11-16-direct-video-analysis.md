# Direct video analysis

**Execution:** Implemented under user authorization; verification failures and restart status are recorded in docs/RUST_MIGRATION.md.

**Goal:** Read the original video without creating a prepared MP4 before primary analysis.

**Architecture:** Keep the existing Rust decoder and model input conversions. Project source planning selects the verified original; legacy evidence resolves its recorded prepared source independently. Browser compatibility remains a playback concern. The existing 30 FPS evidence timeline is preserved for this increment; native FPS evidence is a separate schema migration explicitly deferred by the user.

**Constraints:** Preserve original files, saved evidence and active jobs. Do not include existing user edits. No new application engine or code comments.

### Source planning and compatibility

- [x] Add an isolated HTTP regression using a non-30-FPS original with no prepared cache; assert readiness and original media bytes without creating a prepared MP4.
- [x] Run the regression and verify the old preparation requirement causes failure.
- [x] Select original media in project planning, preserve explicit legacy source resolution, and remove the pipeline transcode stage.
- [x] Verify fusion source identity, source-change rejection, project preparation status and existing playback tests.
- [x] Run formatting, workspace check/tests, WASM and ownership/roadmap checks; record exact failures.
- [x] Append the migration journal and commit only this increment.
- [x] Build release and restart through the launcher only after active jobs finish; otherwise report the precise deferred action.

**Review focus:** Non-MP4 codecs, non-30-FPS source timing, changed originals, old prepared-source evidence, Quick Reel source isolation. Native-FPS sampling and full-resolution model ROI quality are outside this increment and must not be claimed as implemented.
