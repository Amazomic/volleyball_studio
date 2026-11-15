# 2026-11-15 — Player attribution for CPU touch without a pose landmark

This change supersedes the earlier preview-only filtering change: suppressing plausible touch candidates because their COCO17 joint did not pass the strict temporal intersection was not the intended solution.

Observed short project: d6451abac405, existing CPU replay 658c4a564d4140a38bc818edbacb2766, old Touch candidate run 7a44bd3e3da34a49a2ea487ffa7b0557. All 24 saved motion candidates have provisional Track IDs; 8 had an independently corroborated pose contact. RTMPose sparse measurements can be near a touch even when the full three-sample temporal test fails.

Changed write paths:
- rust/volley-core/src/pose_contact.rs and rust/volley-core/src/touch_pipeline/pose_motion.rs: recovery of bounded nearest sampled pose contact, with same source Track ID, <=3-frame window, confidence/geometry bounds, and a source frame on the same replay.
- rust/volley-core/src/touch_pipeline.rs: optional provenance field contact_pose_sample_frame.
- rust/volley-app/src/cpu_touches.rs: for events without a corroborated body landmark, select the closest observed player by distance from VBallNet ball to original source-frame player boxes; clear automatic action kind; never synthesize a contact landmark. Preserve pose-supported assignment and use body support ahead of raw score when deduplicating.
- rust/volley-app/src/cpu_rallies.rs: accept and validate optional nearest RTMPose sample provenance.
- rust/volley-app/src/native_rally_preview.rs: expose both pose-verified and nearest-player generic touch attribution, deduplicate immediate repeated unverified candidates, show no predicted kind for the generic contacts.
- rust/volley-app/src/native_player_statistics.rs: count source-validated same-frame Track ID candidate within rallies even without a pose landmark; generic contacts have no action kind, so are excluded from attack/serve/block estimates; distinguish generic credits from excluded/duplicate proposals and invalidate prior report algorithm snapshot.
- rust/volley-app/src/court_touch.rs: new optional field in test fixtures.
- rust/volley-app/static/studio.js: render all track-attributed touch markers, neutral untyped labels, and linked player name/ID without the additional ID prefix; bump session preview cache.

No saved GPU pose, replay, prior Touch or rally evidence files are modified. Reprocess creates new immutable Touch and rally source versions. Distinguish pose-confirmed landmarks from nearest-box fallback; this is a motion-supported hypothesis, not proof of physical contact.

Verified in working Studio: 261 volley-app Rust tests passed (five ignored). CPU rally compatibility and synthetic nearest-player/RTMPose regressions passed. Immutable CPU Touch version 92acacdf7df34a4291c6263893b577d1 and rally version eae74ac9f3ed42c9965d33e210426470 were built from the existing saved replay with no GPU run. Real endpoint counts: 24 raw proposals; per-rally cards [14, 7, 2]; 23 attributed touches (11 with COCO17 landmark and 12 nearest-player generic); 23 Report counts assigned to roster players and zero unassigned. All 12 generic touches have an attributed Track ID and null classification kind. Studio HTTP 200, detached PID 643248, zero active jobs.
