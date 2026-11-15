# 2026-11-15 · CPU pose/track gap continuity

## Scope and invariants

- Existing GPU RTMPose observations and detector results are immutable, SHA-verified; new pose interpolation is computed on demand from consecutive verified samples only, each separated by the saved sampling stride.
- native_preview_pose::enrich_continuous attaches derived points to the existing CPU Track ID only when both source boxes are unambiguously matched and the interpolated box overlaps that track. Records carry pose_interpolated = true; measured joints carry false. An inferred sample cannot replace a measured one.
- Touch analysis (cpu_touches::pose_frames) now consumes these derived pose samples, including when scoring wrist/body contacts and refining action type. PosePlayer::interpolated keeps provenance; only original measured samples qualify as the before/after independent witnesses for a confirmed trajectory-based pose contact.
- The preexisting confirmed-track box interpolation remains bounded and checks for crossing / duplicate IDs; inserted boxes now also have measurement_source = interpolated_track_bbox. The CPU overlay sends interpolated to render these boxes dashed. Raw overlay may show dashed RTMPose crop boxes where independent detector boxes are missing. Nothing is written back to original GPU / CPU snapshots.
- Other preexisting checkout changes (studio.html, studio_tabs.js, studio_tasks.js, CUDA setup script) are out of scope and must not be staged.

## Tests

- Rust: native pose provenance/gap tests; CPU touches integration replays the same GPU samples into every intermediate frame, verifies existing Track ID, actual/interpolated flags, contact and type; core refuses synthetic-only anchors; CPU overlay inferred/observed ID handling.
- JS: detector label regression plus render-path checks for dashed interpolated boxes.
- Quality impact on real annotated match clips is NOT MEASURED. Synthetic limb interpolation cannot reconstruct a truly unobserved fast wrist motion, so its confidence is capped by both real endpoints; it does not fabricate ball motion evidence.
- Проверки: cargo check --workspace --offline и cargo build --release --locked --offline -p volley-app прошли; Rust workspace: 533 tests passed, 9 ignored, 1 unrelated failing fixture excluded (studio_pipeline::branching_tests::gpu_settings_and_source_are_branch_keys_while_cpu_settings_are_subbranches; isolated rerun also fails). Node syntax, raw detector UI regression, identity sync and git diff --check прошли. Ручная GPU/матчевая проверка не проводилась.
