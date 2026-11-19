# Volleyball Studio — agent contract

Keep shared rules here and details in linked documents. Follow the requested scope.

## Start with the task

1. Run `git status --short --branch`, `git log -5 --oneline`, `git rev-parse HEAD`. Preserve existing changes/untracked files; commit only your paths. Recheck status before committing.
2. State result, acceptance check, owner and write paths briefly. Small tasks need no separate plan/specification/handoff. Persist a plan when dependencies, parallel work or continuation require it.
3. Select the relevant domain in `docs/RUST_CANONICAL_OWNERS.json`; inspect its implementation, callers, source-specific schemas and tests. Extend the owner, never copy an adjacent implementation.
4. Clarify ambiguities affecting results, compatibility, identity or acceptable quality. Choose routine implementation details yourself; continue independent work while awaiting answers.

## Spend context on evidence

- Find paths (`rg --files` or `git ls-files`), scope symbols with `rg -n`, read bounded excerpts. Batch independent reads. Narrow truncated queries; do not repeat full dumps.
- Exclude local data, weights, media, build outputs and `*.bak*`. Read `docs/RUST_MIGRATION.md` by topic or short tail, never in full. Open historical reports only to resolve specific uncertainty.
- Current evidence is code, schemas and fresh checks. Documents state contracts; historical diagnoses, stage counts, owner status and earlier PASS reports do not prove readiness.
- Return relevant matches, totals and failures. Keep large logs outside the repository; cite paths/excerpts. Reuse findings until files, HEAD or assumptions change.

Load only applicable references and sections:

| Task | Reference |
| --- | --- |
| Documentation navigation or missing route | `docs/README.md` |
| Architecture or migration boundary | `docs/RUST_ARCHITECTURE.md`, `docs/RUST_REWRITE_SCOPE.md` |
| Checks and their prerequisites | `docs/TESTING.md`; executable commands in `scripts/check-all.ps1` |
| UI / WASM | `docs/UI_LAYOUT_POLICY.md`; `rust/volley-web/README.md` for WASM builds |
| Analysis quality, performance, personalized montage | Relevant sections of `docs/AGENT_IMPROVEMENT_CHECKLIST.md`; `docs/quality/README.md` for evaluation |
| Launch, dependencies, applying changes | `rust/README.md`, `docs/RUST_DEPLOYMENT.md` |
| Actual concurrent work | `docs/AGENT_PARALLEL_WORKFLOW.md` and its task/run templates |
| Stage acceptance | `docs/RUST_ROADMAP.json`; generate `docs/RUST_ROADMAP.md`, never edit it manually |

## Preserve the architecture and data

- Only executable Studio: `rust/Cargo.toml`. No Python servers/orchestrators, compatibility subprocess engines, JavaScript business state or second Rust-owned implementation. DOM/media integration may use JavaScript/CSS; decisions belong to Rust/Rust-WASM. Retain FFmpeg, ORT/CUDA and weights. Python stays in the protected checkpoint; legacy formats are data adapters.
- Project → immutable GPU evidence → versioned CPU replay → confirmed player links → previews/statistics → original-video montage. CPU changes reuse saved evidence; run new inference only if required observations are absent, with an explicit new evidence version.
- Preserve source SHA, exact cache schemas, revisions, atomic writes, conflict checks and original/analysis time mapping. Version compatibility changes; never relabel old artifacts or overwrite accepted evidence.
- Provisional Track IDs and ReID similarity are not athlete identities. Keep observations, interpolation, algorithmic hypotheses and manual confirmations distinct. Never invent authoritative Touch/rally/game actions.
- Never use user `project/`, `results/`, originals, personal photos or active jobs as development fixtures. Runtime writes/test data stay in isolated roots outside the repository; legacy data is read-only.
- No new comments in code/configuration/settings; explain decisions in documentation. No unrelated cleanup, dependencies or abstractions without a task-specific need.

## Verify the changed behavior

- Reproduce behavior failures and add regression coverage before implementation. Verify observable results and consumers, not only source text. Trace settings through input → Rust → saved profile/result → UI/export as applicable.
- Run targeted checks, then the full application gate from `docs/TESTING.md`. Repeat successful suites only after relevant edits/new evidence. Documentation-only: paths/links, `git diff --check`, affected ownership/roadmap checks; no Rust build/restart.
- Record base SHA/prepared diff, commands, exit results and limits. Confirm baseline failures with the same command on an isolated base or reproducible evidence for that SHA. Never hide failures through skips, weaker assertions or changed goldens; list exclusions separately.
- GPU/model, independent real-match quality and exported-video checks are separate. Mark applicable omissions NOT TESTED; unit/static tests cannot replace them. Performance claims need comparable inputs/configuration and before/after measurements.

## Coordinate and finish

- Parallel work is optional. Each writer needs a separate worktree/branch, fixed SHA, write paths and isolated runtime/build outputs. Send focused briefs, not full histories. One coordinator assigns ownership/resources; one integrator owns shared entries, integrates reviewed commits sequentially and applies the result. Never run concurrent writers in one checkout.
- Review the diff. Each accepted tested increment gets a commit and short append-only `docs/RUST_MIGRATION.md` entry: problem/decision, base/owners, checks, limits, application status. Link large evidence. Workers use unique handoffs; only the integrator updates the shared journal.
- Commit: one short Russian sentence. Work and regular PRs target `develop`. Only update `master` when a tested version is ready for release; never commit routine tasks there. Never rewrite the protected Python checkpoint. Stage only your paths; no repository backup copies.
- Apply automatically when safe. Verify `studio_web.rs` asset mode: debug with `VOLLEY_DEV_UI_ROOT` and matching `static/ui-contract.txt` reads disk HTML/CSS/JS/SVG; refresh and verify served content. Bump the contract for dependent API/WASM ABI changes. Release/embedded assets and Rust/WASM require rebuilding.
- Launch: `start-volley-ui.cmd` → `scripts/start-volley-rust-dev.ps1` → Rust. Check `start-volley-ui.cmd /status` and `/internal/rewrite-status` before restart: no active jobs, `shutdown_safe=true`. Rebuild, use guarded `start-volley-ui.cmd /restart`, verify availability/scenario. Never kill an unrelated process on 8875. If unsafe, retain Studio; report blocker and exact later action.
- Report result, checks, limits and application status briefly. For continuation, keep goal, base/current SHA, paths, decisions, checks and next action in the existing task record; do not copy the conversation.
