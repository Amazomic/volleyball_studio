# Trajectory-first touch detection

Goal: detect a discontinuity in ball flight before choosing a player or technique.

Base: f55c3668ce7933dca2cee36ff85d184c50473956. Rust only; immutable GPU evidence and saved versions remain unchanged. No code comments. No parallel writers in this worktree.

Architecture: extend the existing motion owner with a bounded comparison of two fitted flight segments. A smooth quadratic flight, isolated bad coordinate, straight interpolation and an unexplained long gap must not create a candidate. A persistent velocity change with convergent incoming/outgoing flights becomes independent motion evidence. Existing geometry and pose stages associate the candidate and classify it without granting authority to a real athlete or technique.

Write paths: `rust/volley-core/src/motion.rs`, `rust/volley-core/src/motion/trajectory.rs`, its fixture/tests, `rust/volley-core/src/touch_pipeline/geometry.rs`. Integration into app version checks and canonical-owner metadata is sequential after core tests; do not overwrite concurrent pose attribution work.

Review focus: real 33.1-second rebound; smooth apex; single-frame localization error; two nearby independent contacts; missing samples; no crossing rally/video boundaries; player-free detection and ambiguous attribution.

- [x] Add trajectory regressions and real-ball fixture; run red against the current detector.
- [x] Implement bounded flight fitting and deduplicated motion candidates; run core tests.
- [x] Feed motion candidates into existing player geometry, preserving pose confirmation and unknown technique when unsupported.
- [x] Verify recorded-coordinate contact regression and negative cases; inspect full diff and run workspace, WASM and ownership/roadmap checks.
- [ ] Integrate only reviewed paths into rust, record limitations and provenance, commit in Russian, check active jobs, rebuild/restart safely, and create a new CPU version using saved GPU evidence.

Execution: sequential in this chat under the user's implementation request; no worker agents.

Integration is committed to rust and release is built. Runtime restart is blocked by the existing process refusing two graceful stop attempts; real replay rerun is unavailable because both projects lack saved GPU evidence. See docs/TRAJECTORY_TOUCH_DETECTOR.md.
