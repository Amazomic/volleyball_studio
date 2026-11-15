# 2026-11-15 · Touch direction threshold 5°

- Request: replace 35° direction-change threshold with 5° while retaining rejection of the observed false contact at 00:04.9 (1.82°) near the player.
- Modified (narrow scoped): rust/volley-core/src/touch_pipeline/geometry.rs (geometry-only detections), rust/volley-core/src/touch_pipeline.rs (default occluded_min_turn_deg), rust/volley-core/src/event_evidence.rs (motion-pattern classification), rust/volley-app/static/studio.html (default CPU postprocessing UI value).
- Validation: cargo test -p volley-core --lib --offline: 196 passed; synthetic regression confirms straight-line slowdown rejected and >5°/<35° turn eligible with motion/reach evidence.
- Existing saved GPU/CPU runs and source media were not modified. Rerun CPU postprocessing to materialize new touch snapshot.
