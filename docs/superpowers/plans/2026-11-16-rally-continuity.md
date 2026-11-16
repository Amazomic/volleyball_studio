# Rally continuity implementation plan

**Goal:** Fill every internal player box, skeleton and ball gap inside one saved rally on CPU.

**Architecture:** Keep immutable GPU evidence and saved replay revisions intact. Persist measured pose joints in new CPU replays. The canonical processed overlay derives rally continuity from complete rally context, using one Track ID or manually confirmed athlete links; it never extrapolates or crosses rally boundaries. Synthetic positions retain endpoint provenance and never prove contacts or identity.

**Tech Stack:** Rust, serde_json, existing indexed replay and RTMPose readers.

**Spec:** User request in this chat, including unrestricted internal gaps within a rally.

## Constraints and review focus

- Preserve unrelated worktree changes and all user data.
- No neural inference, code comments, new JavaScript business logic or second implementation.
- Conflicting athlete links must not connect two simultaneously visible bodies.
- Unknown and automatic identities must not join different Track IDs.
- Page boundaries must produce identical coordinates to a whole rally request.
- Missing endpoints and adjacent rallies must stay unfilled.
- Measured joints, boxes and raw evidence must stay unchanged.

## Steps

- [x] Add failing regression tests for linear box/joint interpolation, long ball gaps, identity changes, conflicts, rally boundaries and page context in `rust/volley-app/src/native_preview_overlay/continuity.rs` and existing replay tests.
- [x] Save measured joints in `cpu_retrack.rs`; expose them through `overlay_players` with provenance.
- [x] Implement the continuity pass in the overlay owner's module, reading complete intersecting rallies before trimming the requested page. Preserve independently measured RTMPose poses and enrich missing associated joints for old replays.
- [x] Run targeted tests, WSL workspace checks, web/WASM checks and ownership/roadmap self-tests; document unrelated failures and skipped GPU/real-match checks.
- [x] Append `docs/RUST_MIGRATION.md`, review and commit only this change, rebuild and safely restart Studio after checking active jobs.
