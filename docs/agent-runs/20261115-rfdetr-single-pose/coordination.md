# RF-DETR single-pose GPU run coordination
- Run ID: 20261115-rfdetr-single-pose
- Coordinator and integrator: current Studio maintainer session (single writer)
- Revision: 2
- Base SHA: eda3a1a4c2b950a1b2f1b7c9637939721f491cdc
- Branch: codex/20261115-rfdetr/single-pose
- Worktree: C:\Users\amazomic\Desktop\github\fast-volleyball-tracking-inference\.studio-runtime\worktrees\rfdetr-single-pose
- Task: single-pose, status review, grant epoch 1, no parallel writers
- Exclusive write paths: rust/volley-inference/src/person.rs, rust/volley-app/src/inference_full.rs, rust/volley-app/src/studio_pipeline.rs, docs/agent-runs/20261115-rfdetr-single-pose/coordination.md, docs/agent-runs/20261115-rfdetr-single-pose/tasks/single-pose.md, docs/RUST_MIGRATION.md
- Contract: RF-DETR raw evidence remains detector boxes only; pose evidence is generated exclusively by pose_inference::process after box interpolation; RTMDet remains pose-confirmed.
- Cache: bump RF-DETR GPU stack and person-pipeline versions to prevent reusing different evidence.
- Isolation: do not edit original media, saved project data, other worktrees, or active GPU jobs.
- Acceptance: rustfmt, offline compile, detector-only unit tests, existing inference tests, code review, no uncommitted changes to user's UI.
- Benchmark: not measured; no real game GPU run requested.
- Application: integration to branch rust only after tests and active-job check.

- Verified: cargo check volley-app, 4 RF-DETR tests, 5 studio_pipeline tests, 1 preview detector test, 2 Node history/selector tests, rustfmt changed files, git diff --check.
- Baseline known failure: scripts/ci/test_studio_raw_gpu_preview.mjs fails in both unchanged root and isolated worktree on assertion Current fusion loads normally.
