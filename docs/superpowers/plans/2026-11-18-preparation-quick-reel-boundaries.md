# Preparation / Quick Reel source boundaries — Implementation Plan

**Goal:** remove preparation -> Quick Reel orchestrator while preserving immutable run/source checks and original-video plans.
**Base:** fbe9552. **Branch:** codex/audit-cleanup-stage-five-preparation.
**Architecture:** quick_reel_source.rs owns the existing Source record, original-file validation and private-run preparation plan. Both preparation and quick_reel consume it; it never calls the job runner or preparation manager.
**Execution:** native sequential implementation in the existing isolated worktree, independent read-only review. User approved this substage; the existing stage procedure remains in force.
**Spec:** AGENTS.md, docs/RUST_CANONICAL_OWNERS.json and approved preparation/quick_reel cycle cleanup.

## Constraints and review focus

No new engine, compatibility re-export, schema, route, source-policy or algorithm changes. No project/results/videos as test fixtures; no inference rerun or active-job interruption. Preserve run UUID, source ID/path/size/mtime, immutable request schema/inference_cache_reuse checks, global prepared root, SHA binding, original-video plan, and rejection of unavailable media.

## Files and interfaces

- Create rust/volley-app/src/quick_reel_source.rs: moved Source fields and MAX_VIDEO_SIZE/QUICK_ROOT; safe_source, mtime, source_for, verify_source and prepared_plan_for_quick_reel with unchanged signatures/bodies. Narrow pub(crate) access for existing consumers.
- Modify quick_reel.rs: import source owner, remove moved implementations, retain endpoint/runner/selection storage and all existing tests. Its pipeline consumes existing parent imports.
- Modify preparation.rs: dispatch internal project to quick_reel_source directly.
- Modify main.rs: register source module. Register adapter in quick_reel ownership ledger.
- Create scripts/ci/test_quick_reel_source_ownership.mjs: reject preparation -> orchestrator and shared source -> either orchestrator or preparation manager; reject duplicate source implementations.
- Create docs/agent-runs/2026-11-18-audit-stage-five-preparation.md; integrator alone appends docs/RUST_MIGRATION.md.

## Task and verification

- [x] Add source JSON/identity characterization tests; run existing quick_reel and preparation tests before extraction.
- [x] Write dependency guard and observe RED on the old cycle.
- [x] Move existing source items; change only imports/visibility/module dispatch. Confirm unchanged bodies against base modulo imports and removed comments.
- [x] GREEN guard, characterization and existing owner tests; fmt/check/full workspace offline, ownership/roadmap/WASM and prior stage dependency guard.
- [x] Independent review, isolated commit, selective integration preserving parallel changes, combined checks, own journal entry and commit.
- [ ] Release rebuild and restart only with active_jobs=[] / shutdown_safe=true; otherwise document deferred application and exact restart action.

Release build PASS; runtime application deferred because active studio_analysis/gpu_inference keeps shutdown_safe=false. After jobs finish use «Перезапустить студию». New process/served assets remain unverified.
