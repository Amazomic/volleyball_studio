# Primary preview handoff

Base: 3730cd8; branch: codex/primary-analysis-preview.
Worktree: C:/Users/amazomic/.codex/worktrees/primary-preview/fast-volleyball-tracking-inference.
Coordinator/writer: current chat; reviewer: read-only preview_review.
Write paths: preparation owner and its new primary_preview submodules; raw_decoder/shared owner, export and tests; inference_full/joint.rs; playback.rs; main route; studio_playback.js; existing untracked studio_fragments.js source-selection extension; API/VM compatibility tests; ownership map; this record, plan and migration appendix.
Runtime/build: /tmp/volley-primary-preview-target; no live project/result/original writes. GPU fixtures synthetic and isolated under temporary directories. No browser used.

Accepted behavior and verification are recorded in the migration appendix. The new fragment adapter file preserves the existing working-copy implementation and adds ready-preview routing; active checkout's pre-existing main/studio_web/script registration changes are not silently included in this commit.

Integration pending: the active rust checkout is dirty, with overlapping main.rs and studio_playback.js. Do not clean/stash/reset it. Apply reviewed changes with explicit authorization, preserving all unrelated changes; rerun combined checks, use the normal launcher only when shutdown_safe and no active jobs. New shared primary analyses produce previews; old or reused analyses do not rerun inference for a missing MP4.
