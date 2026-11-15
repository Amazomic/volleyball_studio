# 2026-11-15 — Stable identity links after CPU postprocessing

Base: branch `rust`, commit `17c1f79`; other working-tree edits were retained.
Scope:
- `rust/volley-app/src/native_player_tracks/assignments.rs`
- `rust/volley-app/src/native_player_tracks/photo_comparison.rs`
- `rust/volley-app/src/native_player_tracks/tests.rs`

Root causes:
- Auto-assignment reset previously confirmed Track ID owners, manual exclusions and suppression flags on the same replay.
- Closed-roster fallback could assign a different athlete even when the strongest photo matched an occupied one.
- Near-tied similarity values could produce unstable names in preview.

Behavior:
- Manual owners/exclusions/suppressions remain intact on the exact source-replay sidecar; new replay IDs continue to receive separate identities.
- CPU auto-compare processes only eligible, unassigned tracks; automatic suggestions are refreshed using full replay co-visibility.
- An occupied top match no longer falls back to a weaker athlete. A runner-up within 0.04 cosine of the best keeps a track unassigned.
- ReID evidence and input SHA/revisions remain source-bound. No historical user data is rewritten.

Verification: targeted `native_player_tracks::tests` and scoped `rustfmt --check`; global `cargo fmt --all -- --check` has unrelated existing formatting differences.
Limitations: real-match identity quality and GPU models not re-evaluated. Previously overwritten confirmed links cannot be reconstructed automatically from this fix.
