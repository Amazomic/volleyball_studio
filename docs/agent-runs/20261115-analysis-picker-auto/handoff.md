# 2026-11-15 · Primary GPU analysis auto-selection

- **base_sha:** `658025fc249e4278356385f698f46e026dd7cbcb`
- **branch:** `codex/20261115-analysis-picker-auto/latest`
- **write_paths:** `rust/volley-app/static/studio.js`, `rust/volley-app/static/studio_tabs.js`, `scripts/ci/test_studio_analysis_selection.mjs`, `scripts/ci/test_studio_chain_snapshot_cache.mjs`, this handoff
- **contract:** For an existing project, restore a valid explicit GPU source or select the newest real primary analysis by stage timestamp; a deleted selection falls back to the newest surviving analysis. No blank option while any source exists. CPU postprocessing is disabled when no selectable fused GPU source exists.
- **completion:** The existing `volley:pipeline-complete` event selects the exact finished fusion. Regression test covers it.
- **checks passed:** `node --check` for modified JS; `test_studio_analysis_selection.mjs`; `test_studio_chain_snapshot_cache.mjs`; `test_studio_pipeline_history_ui.mjs`; `test_studio_post_defaults.mjs`; `git diff --check`.
- **preexisting test failures:** Repository binary `volley_web.wasm` is not tracked in isolated worktrees and the existing main checkout binary is behind some Rust tests; the standalone DOM-ID and navigation-cache tests fail even on the original checkout. No runtime/project data was modified.
- **integration:** Apply to `rust` only after reviewing local uncommitted changes in that checkout; do not replace overlapping frontend edits.