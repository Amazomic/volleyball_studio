# 2026-11-15 — Timestamp label for unassigned Track IDs

## Task
User requested a compact timestamp in the Players section for unassigned Track IDs, e.g. `25,2с`.

## Changes
- `rust/volley-app/static/studio_tracks.js`: in the single-track photo card title, render a time label beside `ID N` only if `assigned_player_id` is absent and the ID is not excluded. The label uses `best_frame / 30` FPS for the initial portrait's best measured frame, rounded to one decimal and using a comma decimal separator. No backend/identity contracts changed.
- `rust/volley-app/static/studio.css`: subtle inline type treatment for timestamp, responsive wrapping and tabular figures.

## Verification
- `node --check rust/volley-app/static/studio_tracks.js` passed.
- Tested conversion for frames 0, 755, 756, 840 to labels `0,0с`, `25,2с`, `25,2с`, `28,0с`.
- `git diff --check` on touched UI files passed.
- Full API/web smoke test and release build tracked separately in this run. Existing user analyses, video and photos untouched.
