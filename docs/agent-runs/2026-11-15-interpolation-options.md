# 2026-11-15 · Переключатели интерполяции первичного анализа

## Причина
Нельзя было отдельно выключить интерполяцию рамок игроков и суставов RTMPose в настройках первичного GPU-запуска: обе ветви были активны без выбора пользователя. При повторной CPU-постобработке настроенный режим должен наследоваться из выбранной GPU-версии, а не из текущего состояния формы.

## Исправление
- Добавлены два независимых чекбокса «Интерполяция игроков» и «Интерполяция позы», оба включены по умолчанию. Хранение/сброс настроек формы учитывает состояние checked.
- Rust GPU-профиль сохраняет `interpolate_players` и `interpolate_pose` с обратной совместимостью: для старых профилей и манифестов без полей значением считается `true`.
- Выключение интерполяции игроков исключает восстановленные рамки из GPU crop-плана RTMPose, из CPU track replay и из дополнительного сглаживания игроков при анализе касаний. Измерения и исходные GPU-детекции сохраняются.
- Выключение интерполяции позы исключает искусственные суставы между измеренными RTMPose-кадрами из raw/processed preview и CPU анализа касаний; сами GPU измерения RTMPose сохраняются. Флаг записан в версии GPU-позы, а CPU-пересчёт и предпросмотр используют флаг из этой версии.
- GPU-линия и CPU версия проверяют соответствие опций и происхождение; выбранная первичная версия определяет флаги при последующей CPU-постобработке.

## Изменённые файлы
- `rust/volley-app/static/studio.html`, `studio_tabs.js`, `studio.css`
- `rust/volley-app/src/studio_pipeline.rs`
- `rust/volley-app/src/pose_inference.rs`, `pose_inference/box_interpolation.rs`
- `rust/volley-app/src/native_preview_pose.rs`
- `rust/volley-app/src/cpu_retrack.rs`, `cpu_touches.rs`
- `rust/volley-app/src/quick_reel_pipeline.rs`, `inference_full.rs`
- `scripts/ci/test_studio_interpolation_flags.mjs`

## Проверки
- Node: тест поведения чекбоксов, сброса и сохранения формы; JS-синтаксис.
- Rust: тесты двух режимов crop-плана, pose preview, CPU tracking без интерполяции и совместимости GPU-профиля.
- Проверка workspace и релизная сборка.
- Изолированный тест не создаёт результаты в пользовательском хранилище; новое состояние видно после запуска соответствующего анализа.

## Итог проверок

- `cargo check --workspace --locked --offline`: пройден.
- `cargo test --workspace --locked --offline -- --skip studio_pipeline::branching_tests::gpu_settings_and_source_are_branch_keys_while_cpu_settings_are_subbranches`: пройден; в приложении 279 тестов пройдено, 5 игнорируются; единственный исключённый тест ломался ещё до этой задачи.
- `cargo build --release --locked --offline -p volley-app`: пройден.
- `node scripts/ci/test_studio_interpolation_flags.mjs` и `node --check rust/volley-app/static/studio_tabs.js`: пройдены.
- Старые UI-тесты `test_studio_analysis_selection.mjs` и `test_studio_repeat_photo_analysis.mjs` падают на устаревших ожиданиях (в том числе `updateRawDetectorLabel`) в неизменённых участках; новые переключатели ими не затрагиваются.
- Реальный GPU-анализ с новыми опциями не запускался, тестовые результаты в пользовательском хранилище не создавались.