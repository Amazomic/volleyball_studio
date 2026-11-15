# Task: single-pose
- Run / Task: 20261115-rfdetr-single-pose / single-pose
- Base SHA: eda3a1a4c2b950a1b2f1b7c9637939721f491cdc; grant epoch 1
- Writer: single isolated maintainer, worktree C:\Users\amazomic\Desktop\github\fast-volleyball-tracking-inference\.studio-runtime\worktrees\rfdetr-single-pose
- Write paths: rust/volley-inference/src/person.rs, rust/volley-app/src/inference_full.rs, rust/volley-app/src/studio_pipeline.rs, docs/agent-runs/20261115-rfdetr-single-pose/coordination.md, docs/agent-runs/20261115-rfdetr-single-pose/tasks/single-pose.md, docs/RUST_MIGRATION.md
- Read paths: rust/volley-inference/src/pose.rs, rust/volley-app/src/pose_inference.rs, rust/volley-app/src/inference_fusion.rs
- Inputs: RF-DETR ONNX raw boxes, independent GPU VBallNet detections, fusion manifest and immutable frame evidence
- Output: RF-DETR detection-only PersonObservation with authoritative box, no invented head/joints; interpolation then standalone pose stage
- Preserved: RTMDet uses RTMPose in detection to confirm players; original GPU caches immutable; CPU replay uses same fusion contract
- Tests: cargo format, compile, rfdetr_detection_returns_boxes_without_recomputing_or_inventing_pose, and pipeline regression tests
- Status: review; commit SHA will be added in the following handoff update

- Verification complete: WSL offline volley-inference RF-DETR 4/4; volley-app studio_pipeline 5/5; preview detector 1/1; compile volley-app PASS; Node history/analysis selection PASS; rustfmt selected files PASS.
- Performance: NOT MEASURED. Native match GPU tests: not run. Known unrelated baseline UI test failure documented in coordination log.
