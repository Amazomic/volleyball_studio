# Документация Volleyball Studio

Этот указатель разделяет действующие инструкции, версионные заметки и историю. Единственное исполняемое приложение — Rust workspace `rust/Cargo.toml`. Для текущего поведения проверяй владельца, код и сохранённую схему, а для конкретного анализа — его профиль и манифесты.

Действующие правила задают контракт, но датированные разделы архитектуры, инвентаря миграции и журналов не являются автоматической проверкой текущей полноты. Переход основного приложения на Rust состоялся; дальнейшая разработка не означает обязательный перенос каждой строки JavaScript или вспомогательного Python. Перенос вычислений требует профилирования. Приёмка миграции закрыта (85/85) решением владельца 27.11.2026: архитектура изменилась, старый Python-паритет больше не является блокером. Исторический NO-GO этапа 84 сохранён; качество и production оцениваются отдельно. Число этапов не равно проценту готовности. Исследования и результаты относятся только к указанной базе, данным и условиям; старый диагноз окружения не переносится на сегодняшнюю машину без проверки.

## Начало работы и действующие правила

- [О проекте и запуск на русском](../README.ru.md)
- [О проекте и запуск на английском](../README.md)
- [Скачать бету v0.1.0-beta.1 (Windows/Linux), ограничения и результаты проверок](releases/v0.1.0-beta.1.md)
- [Запуск, зависимости и диагностика](RUST_DEPLOYMENT.md)
- [Обзор UI: смешанная версионная заметка](VOLLEY_STUDIO_UI.md)
- [Skill монтажа через действующую Studio](../.agents/skills/volleyball-reels-studio/SKILL.md)
- [Текущий интерфейс проекта, монтажа и Quick Reel](../.agents/skills/volleyball-reels-studio/references/current-runtime.md)
- [Процесс разработки](DEVELOPMENT_WORKFLOW.md)
- [Локальные проверки](TESTING.md)
- [Оценка качества и её ограничения](quality/README.md)
- [Обязательные инварианты UI](UI_LAYOUT_POLICY.md)

## Архитектура и координация

- [Контракт участника](../AGENTS.md)
- [Rust-архитектура и владельцы](RUST_ARCHITECTURE.md)
- [Машиночитаемая карта владельцев](RUST_CANONICAL_OWNERS.json)
- [Статус Rust и исторический инвентарь миграции](RUST_REWRITE_SCOPE.md)
- [Сгенерированная дорожная карта](RUST_ROADMAP.md)
- [Исходник дорожной карты](RUST_ROADMAP.json)
- [Append-only журнал проверенных изменений](RUST_MIGRATION.md)
- [Критерии улучшения анализа и монтажа](AGENT_IMPROVEMENT_CHECKLIST.md)
- [Изоляция и интеграция параллельной работы](AGENT_PARALLEL_WORKFLOW.md)
- [Шаблон задачи](agents/TASK_TEMPLATE.md)
- [Шаблон запуска](agents/RUN_TEMPLATE.md)
- [Rust workspace](../rust/README.md)
- [HTTP-приложение](../rust/volley-app/README.md)
- [WASM](../rust/volley-web/README.md)
- [История переноса алгоритмов](../rust/volley-core/PORTING.md)

## Предметные заметки Rust

Это объяснения отдельных версий, а не обещание совпадения всех чисел и настроек с текущим кодом.

- [Геометрия площадки](RUST_COURT_GEOMETRY.md)
- [Проект контракта метаданных розыгрыша](RALLY_METADATA.md)
- [Цепочка анализа в интерфейсе](RUST_HEADER_CHAIN.md)
- [Диагностика касаний Quick Reel](RUST_QUICK_REEL_TOUCH_DIAGNOSTICS.md)
- [Пилотная разметка](RUST_PILOT_ANNOTATION.md)
- [Трекинг: версионная заметка](native-player-tracking.md)
- [Позы: версионная заметка](RTMPOSE_PIPELINE.md)
- [Постобработка касаний](TOUCH_PROCESSING.md)
- [Контракт Ball Stabilizer](BALL_STABILIZER.md)
- [Траекторный детектор](TRAJECTORY_TOUCH_DETECTOR.md)
- [Архивный алгоритм ведущего мяча](LEADING_BALL_TRACKING.md)
- [Совместимость геометрической интерпретации касаний](TOUCH_DETECTION_RUST_PARITY.md)
- [Регрессия смены личности](IDENTITY_CUT_REGRESSION.md)

## Датированные исследования и свидетельства

- [Аудит GPU-настроек](GPU_DEFAULTS_AUDIT.md)
- [A/B семплирования рамок, 15.11.2026](PLAYER_BOX_SAMPLING_AB_20261115.md)
- [Журнал задач калибровки площадки](COURT_CALIBRATION_AGENTS.md)
- [Проверка миграции, 10.11.2026](RUST_MIGRATION_CHECK_2026-11-10.md)
- [Историческая приёмка этапа 84: NO-GO](RUST_BENCHMARK_84.md)
- [Свидетельства этапа 84](RUST_BENCHMARK_84_EVIDENCE.json)
- [Вердикт gate этапа 84](RUST_BENCHMARK_84_GATE.json)
- [Медиаизмерения этапа 84](RUST_BENCHMARK_84_MEDIA.json)
- [Исторические Python-результаты](RUST_BENCHMARK_84_PYTHON.json)

## Исторические руководства Python

Команды и пути сохранены для истории; в действующей ветке Python-приложения нет. Указанные измерения не переобъявляются текущими.

- [CALC_SPEED.md](../CALC_SPEED.md)
- [PLAYER_CONTACTS.md](../PLAYER_CONTACTS.md)
- [PLAYER_TRACKING.md](../PLAYER_TRACKING.md)
- [ARCHITECTURE_INFERENCE_POSTPROCESSING.md](ARCHITECTURE_INFERENCE_POSTPROCESSING.md)
- [CPU_GOLDEN_DEFAULTS.md](CPU_GOLDEN_DEFAULTS.md)
- [PLAYER_HIGHLIGHTS.md](PLAYER_HIGHLIGHTS.md)
- [inference-postprocessing.md](inference-postprocessing.md)
- [player-id-continuity-audit.md](player-id-continuity-audit.md)
- [refactor-audit-2026-11-05.md](refactor-audit-2026-11-05.md)
- [ui-controls-audit-2026-11-06.md](ui-controls-audit-2026-11-06.md)
- [upstream-impact-2026-11-04.md](upstream-impact-2026-11-04.md)
- [upstream-touch-evaluation-profile.md](upstream-touch-evaluation-profile.md)
- [volley-studio-client-performance.md](volley-studio-client-performance.md)
- [volleyball-natural-language.md](volleyball-natural-language.md)
- [volleyball-reels-studio.md](volleyball-reels-studio.md)
- [volleyball-reels-studio-v2.md](volleyball-reels-studio-v2.md)
- [volleyball-studio-validation.md](volleyball-studio-validation.md)
- [volleyball-studio-validation-v2.md](volleyball-studio-validation-v2.md)
- [cpu-rally-calibration-2026-11-06.md](cpu-rally-calibration-2026-11-06.md)

## Продуктовые и исследовательские записи

- [Продуктовая цель и технический снимок 01.11.2026](../PROJECT_CONTEXT.md)
- [Математика оценки скорости](../speed.md)
- [Исследование отличий розыгрыша](../rally-not-rally.md)

## Планы и отчёты отдельных работ

Каталоги индексируются по своему назначению. План/спецификация не подтверждает реализацию; handoff подтверждает только перечисленные изменения и проверки своего SHA. Общий принятый результат фиксируется в RUST_MIGRATION.md.

- [Handoff и измерения отдельных задач](agent-runs/)
- [Планы реализации](superpowers/plans/)
- [Спецификации и предложения](superpowers/specs/)

## Как поддерживать документацию

Новый постоянный документ связывай с этим индексом и указывай статус: инструкция, версионная заметка, исследование или история. Даты, SHA, входы и ограничения измерений сохраняй. Устаревший путь не заменяй ссылкой на похожий файл с другой семантикой. Исторические материалы не удаляются ради зелёной проверки ссылок. Машиночитаемые owners/roadmap остаются своими источниками; сгенерированный roadmap не редактируется вручную.
