# Shared storage I/O implementation plan

**Goal:** remove tracking-owned general I/O and duplicated bounded-read/revision implementations without changing evidence, sidecar schemas or external failure classes.
**Base:** 655a383cf5830d4f0c683ae19bd40fc5b25ae90b.
**Execution:** one writer in codex/audit-cleanup-stage-four, read-only independent review before integration. User approval covers this stage only.

## Contracts and files

- storage.rs owns safe_file/safe_folder, BoundedReader, MAX_JSONL_ROW_BYTES and typed bounded optional byte reads; it remains infrastructure, not another domain engine.
- versioned_store.rs owns revision/CAS and preserves its String error adapter and zero-limit refusal.
- cpu_retrack.rs consumes shared I/O and the shared 512 KiB row cap for both read/write.
- player_statistics.rs consumes the shared byte reader and versioned_store::revision, preserving 400 versus 500 and legacy JSON byte hashes.
- All consumers importing these I/O primitives from cpu_retrack are rewired directly to storage; no compatibility re-export or duplicate implementation remains.
- RUST_CANONICAL_OWNERS.json records the shared infrastructure dependency; no endpoint or persistence schema changes.

## Verification and integration

- [x] Add characterization tests before extraction: exact CRLF/LF/raw bytes and SHA, multi-buffer lines, inclusive size boundary, missing/empty/required inputs, malformed JSON, 400/500 and Unix symlink refusal.
- [x] Run tests on original implementations using isolated WSL build outputs and temporary fixtures.
- [x] Extract helpers and typed read errors; keep domain error adapters and constants equivalent.
- [x] Update direct consumers and run fmt/check/clippy/workspace tests plus WASM/ownership/roadmap checks; distinguish baseline failures and skipped GPU/real-match tests.
- [ ] Independent review; preserve every unrelated dirty change during integration, stage only this increment and own journal entry, commit in rust.
- [ ] Rebuild the combined working Studio and inspect active jobs/shutdown_safe before any restart. Apply automatically only when safe; otherwise record the blocking job and exact follow-up action.

## Exact write paths

- docs/RUST_CANONICAL_OWNERS.json
- rust/volley-app/src/storage.rs
- rust/volley-app/src/versioned_store.rs
- rust/volley-app/src/cpu_retrack.rs
- rust/volley-app/src/player_statistics.rs
- rust/volley-app/src/analysis_management.rs
- rust/volley-app/src/cpu_rallies.rs
- rust/volley-app/src/cpu_touches.rs
- rust/volley-app/src/native_action_review.rs
- rust/volley-app/src/native_player_statistics.rs
- rust/volley-app/src/native_player_tracks.rs
- rust/volley-app/src/native_preview_overlay.rs
- rust/volley-app/src/native_rally_marks.rs
- rust/volley-app/src/native_rally_preview.rs
- rust/volley-app/src/preparation.rs
- rust/volley-app/src/studio_analysis.rs
- rust/volley-app/src/studio_pipeline.rs
- rust/volley-app/src/studio_web.rs
- docs/agent-runs/2026-11-17-audit-stage-four.md
- docs/RUST_MIGRATION.md
