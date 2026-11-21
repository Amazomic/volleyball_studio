# Действующая Rust Studio

Проверено по Rust-маршрутам и схемам запросов 17.11.2026. Перед выполнением сверяй актуальный код: этот файл не заменяет сохранённые манифесты конкретного анализа.

## Запуск и обнаружение

Из корня репозитория в Windows:

~~~bat
start-volley-ui.cmd /status
start-volley-ui.cmd /doctor
start-volley-ui.cmd /daemon
~~~

/status и /doctor не запускают анализ. /daemon запускает Studio при необходимости; действующие задания не прерывай. Интерфейс: http://127.0.0.1:8875/studio. Подробности: [развёртывание](../../../../docs/RUST_DEPLOYMENT.md).

Rust workspace: rust/Cargo.toml. Python-приложение доступно только в историческом Git-чекпойнте. Каталоги с оставшимися __pycache__ не содержат рабочего Python CLI.

## Сохранённый анализ и персональный монтаж

Используй вкладку «Монтаж» с профилями «Избранное», «Игрок», «Атаки», «Отчёт». Действующий браузерный адаптер: [studio_montage_builder.js](../../../../rust/volley-app/static/studio_montage_builder.js).

| Операция | Действующий API | Владелец |
| --- | --- | --- |
| Проекты | GET /api/projects | projects.rs |
| Каталог спортсменов | POST /api/players/library/native, action=read | player_catalog.rs |
| Версии анализа видео | GET /api/studio/analysis-overview с project_id и video | studio_analysis.rs |
| Персональная подборка | POST /api/montage/native/player-highlights, operation=preview или start | native_player_highlights.rs |
| Подборка розыгрышей с отчётом | POST /api/montage/native/game-report, operation=preview или start | native_report_montage.rs |
| Состояние монтажа | GET /api/montage/native/status | montage.rs |
| Отмена монтажа | POST /api/montage/native/cancel с job_id | montage.rs |
| Загрузка музыки | POST /api/montage/native/audio, multipart file | montage.rs |

Параметры персонального запроса определяет [Request](../../../../rust/volley-app/src/native_player_highlights.rs): project_id, player_id, rally_run_ids; явно задай action_kind, target_duration_sec, layout и operation. Поддержаны action_kind all, attack, block, serve, dig, overhead_pass, low_touch, unknown_touch. Возможность передать тип не доказывает наличие надёжных событий этого типа.

Превью возвращает source_fingerprint, selected_clips, selected_verified_player_contacts, selected_manually_reviewed_actions и флаги происхождения. При start требуется expected_source_fingerprint из актуального превью. Поля reviewed_actions_only, manually_linked_athlete_only и action_classification_verified могут быть false: название API и число контактов не означают, что все личности и типы действий проверены человеком. Не исправляй эти флаги на клиенте.

player_id берётся из реального состава, rally_run_ids — из сохранённых CPU-версий. При отсутствии нужной привязки сначала используй существующий интерфейс проверки игроков; не отождествляй спортсмена с номером трека. Источники, интервалы и ревизии проверяет Rust; не подставляй собственные границы клипов вместо серверной подборки.

Отчёт использует отдельную схему ReportRequest без player_id. Избранные розыгрыши обслуживаются существующим путём отбора/монтажа; не превращай их в запрос player-highlights. Параметры и состояния бери из текущего адаптера и [маршрутов](../../../../rust/volley-app/src/main.rs).

## Новый анализ и Quick Reel

Анализ проекта запускается через вкладку «Анализ» и POST /api/studio/pipeline/start; его StartRequest и режимы определены в [studio_pipeline.rs](../../../../rust/volley-app/src/studio_pipeline.rs). Для изменения CPU-настроек используй сохранённые GPU evidence, когда нужные наблюдения уже есть. Не выдавай этот API за команду универсального CLI.

Quick Reel — отдельный сценарий свежего анализа. [quick_reel.rs](../../../../rust/volley-app/src/quick_reel.rs) определяет выбор/загрузку исходников, StartRequest и статус; [quick_reel_pipeline.rs](../../../../rust/volley-app/src/quick_reel_pipeline.rs) — обработку. Используй вкладку «Быстрый рилс» или API:

- POST /api/quick-reel/native/select с paths, либо pick/upload для выбора или загрузки файлов; сохрани выданные сервером selection_ids.
- Загрузи выбранную пользователем музыку через montage/native/audio; возьми подтверждённый audio_file.
- POST /api/quick-reel/native/start с selection_ids и обязательным audio_file.
- GET /api/quick-reel/native/status с id; отмена — POST /api/quick-reel/native/cancel с id. После complete проверь output.file=quick_reel.mp4, output.native_original_clips_verified=true и output.mp4_quality.has_audio=true, затем используй GET /api/quick-reel/native/media с подтверждённым id. При failed, cancelled или interrupted заверши ожидание; MP4 не считается готовым.

Quick Reel не переиспользует инференс проекта. Длительность связана с аудиодорожкой; отдельные target_duration_sec, player_id, count, events, style и VFX не входят в StartRequest и не должны добавляться к нему.

## Модели и ограничения

Текущий детектор игроков — RF-DETR Medium; позы — RTMPose, мяч — VBallNet V4c. Точные веса, шаг наблюдений, входные размеры, FPS и способ декодирования проверяй в профиле и сохранённых манифестах: они зависят от версии и настроек.

В текущем монтажном API нет огненного мяча, шлейфа, библиотеки VFX, произвольного ретайминга или пользовательских подписей. /api/montage/native/library хранит готовые монтажи, а не эффекты. [Требования к событиям и будущим VFX](events-editing-vfx.md) не являются подтверждением реализации.
