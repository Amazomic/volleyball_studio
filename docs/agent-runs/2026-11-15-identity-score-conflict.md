# 2026-11-15 — Player identity score arbitration and reference collisions

Base: Rust migration branch; live data inspected read-only before change. Existing unrelated dirty changes left intact.

## Reproduction

Live project `d6451abac405`, replay `bda1f895ce594fc49c2f25e42ff75ac1`:
- Track 2 → same roster athlete as Track 4, cosine 0.99335, 104 measured frames.
- Track 4 → same roster athlete, cosine 0.98513, 143 measured frames.
- Distinct, real concurrent observations for Tracks 2 and 4 in 74 frames (zero co-location).
- Previous selector weighted log(observations), so Track 4 won despite weaker photo evidence; Track 2 and reacquired Track 11 (cosine 0.93098) were left unassigned.
- Track 12 matched two athletes at virtually equal 0.90604 cosine; one identical JPEG SHA is stored under both athletes' reference albums.

## Changes

- `rust/volley-app/src/native_player_tracks/assignments.rs`: score-first deterministic ordering, observations only break exactly equal cosine results; keep replay-wide real co-visibility rejection and ambiguous top-two gate.
- `rust/volley-app/src/native_player_tracks/photo_comparison.rs`: exclude images whose SHA belongs to different roster owners from photo comparison, without changing user albums or history.
- `rust/volley-app/src/native_player_tracks/tests.rs`: source-score regression for the simultaneous 2-versus-4 conflict and subsequent 11; updated weighted-length regression.
- Photo comparison unit test: identical image shared across two athletes cannot vote for either.

## Verification and limits

- Targeted `cargo test -p volley-app native_player_tracks:: --offline --locked`: 49 passed, 0 failed.
- Scoped Rust formatting check passed.
- No original video, manual assignment, photo album or archived CPU replay modified by source edits.
- Re-evaluate live automatic links separately using the supported API after safe Studio restart; do not overwrite existing manual ground truth.
- A wrong owner in a manually saved reference album may still bias other photo comparisons; explicit user review and reference cleanup may be required.

## Live verification after deploying revised release binary

- Release build passed; Studio restarted in detached mode after confirming no active jobs.
- Ran saved GPU/CPU replay photo re-assignment via the existing `POST /api/players/native/reid-review` endpoint (without running neural GPU inference). Returned 10 automatic links; Track 2 and Track 11 now both link to the same intended athlete, Track 4 remains unassigned rather than claiming the wrong athlete, Track 12 remains ambiguous.
- Applied the user's explicit correction that the incorrectly named Track 4 is the other athlete using the existing revision-checked manual track API; linked Track 4 to the corrected roster entry. The API now reports 11 of 12 assigned and no visibility conflicts.
- Track 12 remains unassigned: after excluding duplicate cross-album photo hashes, its highest similarity is 90.36% with an athlete already represented by co-visible Track 13, so a blind assignment would relabel a different person.
- Existing photos and previous CPU/GPU analysis snapshots remain intact. The latest link sidecar was updated through versioned APIs; saved pipeline history may retain the earlier automatic count until a new pipeline version is created.
