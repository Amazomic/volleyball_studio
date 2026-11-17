# Montage: archived CPU geometry must not block original-video rendering

- Base SHA: `d2eab30e6ad8b6ea5c967a1978bbbbeeb80f2571`.
- Scope: `rust/volley-app/src/native_rally_preview.rs`, `rust/volley-media/src/montage.rs`.
- Fixed: montage material keeps the immutable CPU run's recorded court revision rather than comparing it to today's editable court. Later field edits no longer block exports; immutable original video provenance is still checked.
- Confirmed source: `PreparationPlan::source` resolves the project's canonical original video, while `ready_path()` is only used for analysis validation. FFmpeg receives `OriginalClip::source`, not a prepared 30 FPS copy.
- Tests: `cargo fmt --all -- --check` passed; `cargo check -p volley-app --offline` passed (one preexisting dead_code warning); `cargo test -p volley-media original_montage_reads_source_media_without_analysis_resize --offline` passed (1 test); `node ../scripts/ci/test_volley_web_wasm.mjs` passed; `git diff --check` passed.
- Runtime before correction: GET `/api/cpu/native/rallies/timeline` for the affected CPU version returned 409 `Geometry changed; rebuild CPU interpretation`; source review confirmed call site.
- Verify after deploy: read-only timeline request for the same version must return a material plan (without relaunching analysis); new export uses SHA-verified originals.