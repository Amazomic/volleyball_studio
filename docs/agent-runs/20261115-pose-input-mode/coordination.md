# Pose analysis modes / coordination
- Coordinator and single writer: Studio maintainer; no parallel writers
- Worktree: .studio-runtime/worktrees/pose-input-mode
- Branch: codex/20261115-pose-input-mode
- Base SHA: 902208311898d55f7d74ace5451401ea9e9df8f0
- Exclusive files: rust/volley-app/src/pose_inference.rs; rust/volley-app/src/studio_pipeline.rs; rust/volley-app/src/native_preview_pose.rs; rust/volley-media/src/frame_windows.rs; rust/volley-media/src/raw_decoder.rs; rust/volley-app/static/studio.html; rust/volley-app/static/studio_tabs.js; scripts/ci/test_studio_interpolation_flags.mjs; scripts/ci/test_studio_pose_modes.mjs; docs/agent-runs/20261115-pose-input-mode/coordination.md; docs/RUST_MIGRATION.md
- Contract: photo mode performs RTMPose on stride-sampled decoded frames (default 3); video mode runs RTMPose every decoded video frame (stride 1). Same model, no second pose engine.
- Data: mode saved in profile and immutable pose manifest; legacy manifests without mode treated as photo. No external photos or separate model.
- Tests: offline Rust check/test, selected rustfmt, Node tests, before-deploy active-job check.
- Integration: preserve four uncommitted main working tree files by staging clean branch blobs and applying minimal code changes to main worktree.

- Additional consumer: native preview pose source stride bound expanded to 30 for CPU replay and overlay; no new pose inference in preview.

- FrameWindows now reads and skips actual frames when sampling stride exceeds its 9-frame input window; diagnostics remain limited to original 1..9 sampling.
- Verified: offline cargo check volley-app PASS; studio pipeline tests 6/6; stride-30 pose preview test 1/1; sparse frame decoder test 1/1; node pose mode, interpolation, settings, history tests PASS; rustfmt and diff check PASS. Real GPU run: NOT MEASURED.
