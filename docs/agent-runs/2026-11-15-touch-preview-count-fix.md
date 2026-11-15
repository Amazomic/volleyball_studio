# 2026-11-15 — Count confirmed contacts consistently in rally preview

Scope: rust/volley-app/src/native_rally_preview.rs, rust/volley-app/static/studio.js.
Role: single integrator.

Reproduction (read-only snapshot): project d6451abac405, rally version a268d83ac74140e389c5f749acc1d820 (three rallies). The saved CPU timeline has 24 raw Touch proposals, 23 timeline annotations, of which eight have contact_verified=true. The per-rally counts are respectively 14/5, 7/1, 2/2 (candidate/verified). The first preview card displayed 14 as real touches while the Report displayed eight verified athlete contacts across all rallies.

Root cause: rally preview card motion_event_count counted provisional motions with a Track ID, not COCO17 pose-supported contact. UI also had a secondary status path using unfiltered touch_frames.length. Report requires body_contact_supported and contact_landmark inside a rally, plus roster ownership.

Change: compute page card motion_event_count, action tallies and event pills only for candidates with both body-contact flag and contact landmark; retain raw_motion_event_count for diagnostics. Remove per-rally/card previous-Track-ID suppression which could hide a valid next contact after an unverified proposal and was not used by Report. Keep immutable SHA-checked CPU evidence unchanged. UI status always counts timeline contact_verified=true, removes raw candidate fallback, and bumps the preview cache key to v2.

Verification: 12 focused Rust unit tests passed including a new regression for a nonpose candidate followed by a verified touch of the same Track ID; JS rally-cards test passed, JS syntax check passed, focused rustfmt --check and scoped diff whitespace check passed. Real-game inference was not run. No original media or saved CPU snapshots modified.