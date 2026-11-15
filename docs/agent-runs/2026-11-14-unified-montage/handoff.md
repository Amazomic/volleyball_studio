# Unified montage builder handoff - 2026-11-14

Status: implementation checked in shared working tree, not committed; live Studio not restarted.

## Scope
Frontend: rust/volley-app/static/studio.html, studio_montage_builder.js, studio.css, studio.js.
Backend: rust/volley-app/src/montage.rs, native_player_highlights.rs, native_report_montage.rs, native_rally_preview.rs, quick_reel_pipeline.rs, main.rs, studio_web.rs.
Media: rust/volley-media/src/montage.rs.
Tests: rust/volley-media/tests/original_montage.rs; scripts/ci/test_studio_montage_tab.mjs; studio_web.rs test.
Ownership: docs/RUST_CANONICAL_OWNERS.json, added new montage game-report route only.

## Results
One configurable composer with four presets, roster/action filters, contact-only reels,
short clip context, target duration and song fitting, vertical VBallNet ball-follow
crop with zoom, native game-report and 5-sec statistical outro.
SHA verification, false/static ball rejection, original video export.
Player reels require explicitly reviewed player actions and manual athlete links.
Report game contacts use canonical Report-tab counting, with AI action estimates labeled.

## Verification
Windows cargo check volley-app offline: pass.
WSL cargo check volley-app offline: pass.
volley-media tests: 22 unit + 7 integration: pass.
native_player_highlights tests: 2 pass.
studio_web HTTP test: pass.
Montage Node UI test and JS syntax: pass.
FFmpeg card on Windows requires explicit fontfile (fixed).
No real user video rendered for this verification.

## Blocked deployment
Starter /check fails because unrelated route
/api/players/native/tracks/reset-project is not in ownership ledger.
Starter /status currently says Busy (active jobs).
Do NOT restart, kill the process, or modify other agents' data.
Integrator should resolve route ownership through owning agent, wait for idle,
then run start-volley-ui.cmd /check and /restart.


## 2026-11-14 fix: Report attacks missing from player montage
- Root cause: the report counted SHA-verified pose contacts with estimated ActionPrediction and automatic identity links; montage previously required explicit human action review + manual identity links. Thus a report attack could yield zero montage clips.
- Added native_player_statistics::verified_contacts_for_montage: returns frame, Track ID, and estimated type from the same SHA-verified, inside-rally, pose-landmark selector used for Report totals.
- Player-highlights uses native_player_tracks::statistics_owners (including reconciled photo-auto IDs), selects these canonical contacts for a roster player and type; human labels override estimated type only for an exact matching contact.
- Montage fingerprint now includes resolved and excluded Track ID assignments, so re-linking invalidates an old preview. No guessed Track IDs, outside-rally proposals, or unseen contact points.
- UI distinguishes physically verified contacts from estimated action types; automatic attack labels are no longer falsely called human verified.
- Tests: full volley-app Rust suite 240 passed (5 ignored); exact regression assert that a Report-estimated attack becomes a player reel candidate without manual review; ownership/roadmap and UI checks passed.
- Unrelated metadata fixed for safe launcher: canonical registration of already existing Track ID reset route; README roadmap label aligned to 84/85. No unrelated feature source changed.
- Deployment is performed only via safe script /restart when no live jobs are running.


## Live verification after fix
- Safe /check passed after unrelated canonical route metadata + README roadmap pointer were synchronized.
- Live Studio was restarted only after shutdown_safe=true and active_jobs=[]; local port 8875 serves new montage-builder.js, HTTP 200.
- On a persisted long CPU source (no new GPU/CPU analysis), read-only player-highlights PREVIEW returned 14 player-attack contacts and 14 short clips, with zero manually reviewed actions. Confirms regression was fixed on existing project evidence.
- Existing source analysis and user files untouched; no MP4 export was needed for this read-only validation.
