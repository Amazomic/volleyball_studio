# 2026-11-15 — CPU Track ID 4 cross-person switch in short match

## Scope and baseline
Single integrator on the connected HOME-PC. Read `AGENTS.md`, `docs/AGENT_PARALLEL_WORKFLOW.md`, the installed project skill, and identity handoff from earlier today. Working checkout `C:\Users\amazomic\Desktop\github\fast-volleyball-tracking-inference`, branch `rust`. Existing unrelated dirty changes preserved.

## Reproduction from immutable GPU/CPU evidence
Project `d6451abac405`, replay `8f77b4efa1cb4adba3179ac8e4b97894`, original GPU fusion `ea8def100c5b405d842789c0f7f20dee`.

- Frame 450 (15.0s): CPU Track ID 4 bbox centered x≈0.562 is a light-shirt person on the far side.
- Frame 540 (18.0s): Track ID 4 is still on this light-shirt person, moving right, measured head near x≈0.586.
- Frame 580 (19.33s): After a 40-frame observation gap, CPU tracker assigned ID 4 to another person moving back left, bbox center x≈0.548, head x≈0.546. No matching new RTMPose sample to prove re-identification.
- Frame 603 (20.1s): Track ID 4 now shows a black-shirt person, making a single-track photo cover disagree with preview frames around 15s. Live photo API confirmed the old best crop for ID 4 showed the black-shirt person. Indexed photo samples 2/4 showed the light-shirt person, demonstrating the cross-person track.

Root cause: `volley-core::tracker::score_match` allowed a long occlusion to match a nearby person travelling sharply opposite to the track's last observed horizontal motion without strong fresh pose evidence.

## Source changes
- `rust/volley-core/src/tracker.rs`: with 16+ frame observation gaps and no strong fresh pose match, do not reuse an ID when the previous recent measured horizontal movement is significant and the new candidate reverses direction sufficiently. This intentionally prefers a new provisional ID to a fabricated player identity; preserves the 90-frame grace period for compatible trajectories.
- Regression `crossing_player_cannot_steal_trajectory_after_sparse_pose_gap` replicates observed timestamps/relative motion and verifies compatible right-moving return still keeps the original ID.

## Verification
- `cargo test -p volley-core tracker::tests:: --offline --locked`: 25 tests passed.
- `cargo test -p volley-app cpu_retrack::tests:: --offline --locked`: 9 passed.
- `cargo test -p volley-app native_player_tracks:: --offline --locked`: 49 passed.
- Rustfmt check and scoped git diff whitespace check passed.
- `cargo build -p volley-app --release --offline --locked` succeeded. Studio checked idle, gracefully restarted and detached at `http://127.0.0.1:8875/studio`.
- Ran a **new CPU-only postprocessing version** using the same saved GPU fusion and the precise prior post profile (no new GPU neural inference, no original video modifications):
  pipeline `d8436d2733734068b788fbebc76f48b8`, replay `7b3138d875c447ecbc94185cde0ee252`, rallies `41b5f18de7eb4c3189b44586b5806e9c`.
- Live replay confirms ID 4 now ends on frame 540 and **does not label the black-shirt person** at frame 603. The black-shirt observation is assigned to a different track; visual photo API for new ID 4 now returns the light-shirt player. 13 provisional tracklets, 11 automatically linked, IDs 11 and 12 still unassigned.
- Data provenance caveat: automatic saved photo references still label new Track 4 as `Раиля`, although user referred to its subject as `Рузиля`. The photo-to-preview physical mismatch is fixed, **but athlete name identity review / mislabeled training photo cleanup remains outstanding**. Do not silently swap the two women or rewrite their personal reference libraries without an explicit verified ground truth; two women are observed simultaneously.
- Historical analysis versions, manual links and photo albums remain intact; the new CPU pipeline is separately versioned.
