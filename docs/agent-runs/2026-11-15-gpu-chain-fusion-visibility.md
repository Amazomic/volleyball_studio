# 2026-11-15 · GPU fusion analysis picker deduplication

- Base SHA: `e3bc2fba99e58fc7ef799217cd15966b7532d86d`
- Write paths: `rust/volley-web/src/domain.rs`, this handoff and `docs/RUST_MIGRATION.md`.
- Finding: `studio_pipeline::pipeline` creates one GPU pipeline with two raw model jobs (ball + people) and a fusion. `studio_analysis::stage_summary` exposes `person_job_id` and `vball_job_id`. The Rust/WASM choice builder still checked the obsolete `ravel_job_id`, displaying the raw person-model job independently as `не объединён`.
- Change: group both raw jobs under their fused analysis; preserve genuinely orphaned GPU jobs as separate choices.
- Regression: `fused_person_and_ball_jobs_do_not_appear_as_separate_analyses` asserts exactly one fused analysis plus one orphan.
- Verification: `cargo fmt -p volley-web -- --check` pass; `cargo test -p volley-web --offline --locked` 9 passed; `cargo build -p volley-web --release --target wasm32-unknown-unknown --offline --locked` pass.
- Known unrelated blocker: `node scripts/ci/test_volley_web_wasm.mjs` fails global translation-budget assertion (`Untranslated top-level static UI text exceeds translation budget`), due other existing UI text; GPU fix does not change these translations.
- No real GPU inference run; no user media or persisted analysis changed.
