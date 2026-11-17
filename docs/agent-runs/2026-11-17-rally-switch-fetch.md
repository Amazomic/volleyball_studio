# Переключение розыгрышей — ограничение загрузок наложений

- Симптом: при частом выборе карточек розыгрышей браузер сообщал `Failed to fetch`.
- Источник: `studio.js` запрашивал сразу два окна CPU overlay при каждом переходе. Дорогие запросы на SHA-проверенный replay накапливались одновременно.
- Изменение: запрос при ручном seek откладывается на 120 мс и объединяется с последующими выборами, загружается только актуальное окно. Соседнее окно по-прежнему загружается по мере приближения воспроизведения к его границе.
- Проверки: `node --check rust/volley-app/static/studio.js`, `node scripts/ci/test_rally_seek_prefetch.mjs`, `node scripts/ci/test_studio_preview_source.mjs`, `node scripts/ci/test_studio_playback_modes.mjs`, `node scripts/ci/test_studio_wasm_ui.mjs`, `node scripts/ci/test_hevc_fragment_playback.mjs`, `node scripts/ci/test_primary_preview_playback.mjs` — PASS.
- Release: `cargo build -p volley-app --release --offline --locked` — PASS; безопасный перезапуск, активных задач нет; `/studio/app.js` содержит новый код.
- Ограничение: первый запрос `/api/cpu/native/rallies/preview` для длинного исходника после старта Rust-процесса всё ещё может ждать более 25 с. Для полного устранения нужна отдельная оптимизация холодной SHA-проверки исходного видео без ослабления проверки целостности.
