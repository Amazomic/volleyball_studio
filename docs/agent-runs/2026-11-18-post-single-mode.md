# Единственный алгоритм касаний и свёрнутая постобработка — 2026-11-18

## Изменения

- `rust/volley-app/static/studio.html`: удалён селектор `studio-post-touch-algorithm`. Все параметры CPU-постобработки, включая порог касания и сброс, находятся внутри одного `<details>` «Параметры постобработки», по умолчанию закрытого.
- `rust/volley-app/static/studio_tabs.js`: `touches.algorithm` новых запросов всегда `pose_motion`, без зависимости от удалённого DOM-контрола. Ранее сохранённые значения алгоритма не восстанавливаются; другие пользовательские параметры сохраняются.
- `rust/volley-app/src/studio_pipeline.rs`: API для новых запусков не принимает явно выбранный `legacy`; частично заполненные профили нормализуются в `PoseMotion`. Чтение старых сохранённых результатов через `decode_profiles` сохраняет исходный алгоритм.
- `rust/volley-web/src/studio_i18n.tsv`: добавлены переводы новой свёрнутой секции.
- `scripts/ci`: тест закрытого UI-блока, удаления селектора, актуализация кеша настроек и WASM-тестов.

## Проверки

- `cargo fmt --all -- --check` — PASS.
- `cargo check -p volley-app --offline` — PASS.
- `cargo test -p volley-app new_post_requests_have_one_touch_algorithm_without_rewriting_saved_legacy --offline` — PASS.
- `cargo test --workspace --offline -- --skip storage::tests::shared_storage_statistics_preserve_missing_exact_bytes_and_http_errors` — PASS. Известный ранее сбой статистики хранилища исключён, не связан с этой правкой.
- UI tests: `test_studio_post_single_mode.mjs`, `test_studio_post_defaults.mjs`, `test_studio_settings_cache.mjs`, `test_studio_wasm_ui.mjs`, `test_volley_web_wasm.mjs` — PASS; Rust/WASM coverage 284/284 видимых подписей и 77/77 подписей доступности.

Не изменялись физические пороги, алгоритмы обработки архивных результатов или исходные видео. Перезапуск требует безопасного завершения активных заданий.
