# Browser playback compatibility

**Goal:** Probe original video decoding when a project is added or opened; create H.264/AAC playback media only after an actual browser decode or unsupported-format error. Analysis always reads the original and retains the existing 30 FPS evidence timeline.

**Owners:** Extend preparation.rs and playback.rs; reuse PreparationManager, SHA receipts, cancellation and the existing FFmpeg writer. Browser JavaScript only reports media events and displays Rust responses. Keep compatibility media in browser-playback-v1, separate from analysis evidence.

- [x] Add HTTP regressions: successful decode/network error never encodes, confirmed media errors queue a verified compatible copy, repeated requests reuse it, and analysis still selects the original.
- [x] Add browser-event regressions for loadeddata, decoder failure, network error, timeout and source changes.
- [x] Implement Rust playback plans/status and explicit compatible-media serving through existing preparation endpoints.
- [x] Implement the DOM probe and error adapter, integrate project-ready events, and remove the editor canPlayType shortcut.
- [x] Verify output codecs with a synthetic video containing audio; run focused, workspace, WASM, ownership and roadmap checks.
- [x] Review, append migration journal, commit only task changes, build release and safely restart after checking active jobs.

**Review focus:** Browser-specific decisions must not globally force proxies; stale requests must not replace a newly selected source; network failures must not trigger encoding; partial or modified media must not be served; original files and saved analysis identities must remain unchanged.

**Result:** Implemented and verified; full-check limitations and runtime restart status are recorded in docs/RUST_MIGRATION.md.
