# Исправление ReID-процентов и автопривязки · 2026-11-18

- База: `52186ca9f1e03bae3ba68b07e8306cf9bbc3ec31`; worktree `.studio-runtime/worktrees/reid-player-match-20261118`, ветка `codex/20261118-reid-player-match`.
- Изменения ограничены `rust/volley-app/src/native_player_tracks/photo_comparison.rs`, `rust/volley-app/static/studio_tracks.js`, двумя JS-тестами и журналом.
- Причина: snapshot отклонял более 256 референсных фото до получения эмбеддингов, хотя у первых шести игроков активного проекта их 309; в пользовательских файлах ничего не изменено.
- Инварианты: 64 MiB cap, SHA, изоляция источника, fingerprint модели, отклонение co-visible ID, ручное подтверждение, ни одного синтетического cosine.
- Результаты: targeted Rust test и cargo check --workspace --offline PASS; cargo test --workspace --offline с skip 2 известных нестабильных storage/API тестов PASS; 9 специализированных JS/WASM тестов PASS; cargo fmt PASS; git diff --check PASS.
- Существующий `test_studio_group_tracks.mjs` падает и на HEAD исходной rust ветки (неправильный mock `state.groups`); исправление этого теста вне области задачи.
- После интеграции — сборка, безопасный рестарт только при нулевых active_jobs, проверка /studio.
