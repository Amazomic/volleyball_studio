# Rally material boundaries implementation plan

**Goal:** remove the renderer/HTTP-preview dependency cycle while preserving source verification, immutable timelines, revisions and montage intervals.
**Base:** 671f09502472ca6e53467bb321c400289fe3c616.
**Execution:** one writer in codex/audit-cleanup-stage-five; independent read-only review before integration. User separately approved this stage.

## Responsibilities

- native_rally_evidence.rs: existing source schemas, inspect_project, SHA-verified generation, finalized touch/rally context, manual interval selection. Nested camera.rs owns existing verified camera extraction; nested tests.rs holds migrated source tests.
- montage_material.rs: existing MaterialBinding, deterministic material normalization/digest, publication, bindings and validation. Calls evidence/identity/action owners, never HTTP preview or job runner.
- native_rally_preview.rs: existing public query/request DTOs and three HTTP handlers, with unchanged routes and responses.
- montage.rs consumes MaterialBinding/validation from the material owner. Other existing consumers use evidence or material directly; no compatibility re-export or duplicate implementation remains.
- main.rs registers the two modules; owner ledger registers adapters and the existing sidecar path remains unchanged.

## Tasks and verification

- [x] Add architectural regression test and observe old renderer -> preview dependency failure; add a stable MaterialBinding serialization test and run baseline owner tests.
- [x] Move existing items without modifying algorithms; migrate callers and tests alongside their owners.
- [x] Verify function bodies mechanically; run fmt/check/workspace tests, WASM and ownership/roadmap plus the new dependency gate. Report baseline lint limitations and skipped GPU/real-match runs.
- [ ] Independent review; preserve unrelated working-tree changes during selective integration and commit only this stage plus its journal entry.
- [ ] Rebuild combined Studio and safely restart only after active-job/shutdown_safe check; verify new process and served assets.

## Exact write paths

- docs/RUST_CANONICAL_OWNERS.json
- rust/volley-app/src/cpu_rallies.rs
- rust/volley-app/src/inference_full.rs
- rust/volley-app/src/main.rs
- rust/volley-app/src/montage.rs
- rust/volley-app/src/native_action_review.rs
- rust/volley-app/src/native_player_highlights.rs
- rust/volley-app/src/native_player_statistics.rs
- rust/volley-app/src/native_player_tracks/source_index.rs
- rust/volley-app/src/native_preview_overlay.rs
- rust/volley-app/src/native_rally_marks.rs
- rust/volley-app/src/native_rally_preview.rs
- rust/volley-app/src/native_report_montage.rs
- rust/volley-app/src/quick_reel.rs
- rust/volley-app/src/quick_reel_pipeline.rs
- rust/volley-app/src/studio_pipeline.rs
- rust/volley-app/src/montage_material.rs
- rust/volley-app/src/native_rally_evidence.rs
- rust/volley-app/src/native_rally_evidence/camera.rs
- rust/volley-app/src/native_rally_evidence/tests.rs
- scripts/ci/test_rally_material_ownership.mjs
- docs/agent-runs/2026-11-18-audit-stage-five.md
- docs/RUST_MIGRATION.md
