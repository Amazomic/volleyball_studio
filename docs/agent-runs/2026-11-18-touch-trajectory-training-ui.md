# 2026-11-18 — единая шкала дообучения и расчёт касаний по VBallNet

- Base: fc03aac, isolated worktree branch codex/20261118-touch-annotations.
- Из html удалена отдельная шкала касаний; добавлены кнопки рассчитать, взять расчёт, сохранить, метод поиска и выключение неактуальных слоёв.
- Редактор создаёт локальный черновик, сохраняет его явно через action=save, не переписывает CPU/GPU evidence.
- Полный SHA-проверенный проход по CPU replay извлекает физические кандидаты только из траектории мяча и использует существующие Rust motion trajectory_breaks + ball_motion_extrema.
- Калибровка на сохранённых примерах подбирает порог сигнала и систематическую временную поправку (±2 кадра), отрицательная выборка учитывает 0 касаний.
- Новый расчёт не копирует ручные правки непосредственно в результат; показатели по той же обучающей выборке не являются независимой валидацией.
- Кнопки начала и конца розыгрыша и 3 видимых строки скроллируемого списка оставлены.

Проверено на изолированной ветке:
- Cargo check volley-app --offline PASS.
- Cargo test volley-app --offline PASS: 351 passed, 9 ignored, 0 failed.
- node --check studio.js PASS.
- test_studio_dom_ids.mjs PASS (242 JS ID из 290 HTML ID).
- test_studio_wasm_ui.mjs PASS.
- test_studio_viewport_layout.mjs PASS.
- test_studio_touch_training_ui.mjs PASS (одна шкала + необходимые кнопки).
- git diff --check PASS.

Не выполнено: end-to-end UI с реальными матчами/GPU, построение и применение новой локальной release-сборки (требует безопасного интеграционного шага без конкурирующих задач).

Изменены rust/volley-app/src/main.rs, rust/volley-app/src/touch_annotation.rs,
rust/volley-app/src/touch_annotation/trajectory.rs, rust/volley-app/src/native_player_tracks/trajectory.rs,
rust/volley-app/static/studio.js, studio.html, studio.css,
docs/TOUCH_ANNOTATION_DATASET.md, scripts/ci/test_studio_touch_training_ui.mjs.