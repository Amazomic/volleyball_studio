# Документация Volleyball Studio

Единственное действующее приложение — Rust workspace `rust/Cargo.toml`. План разработки: 85/85; миграция завершена (решение владельца 27.11.2026); качество анализа и production readiness оцениваются отдельно.

## Пользователю

- [Обзор проекта](../README.ru.md)
- [Установка и запуск](RUST_DEPLOYMENT.md)
- [Руководство по интерфейсу](VOLLEY_STUDIO_UI.md)
- [Автоматизация монтажа через Agent API](../.agents/skills/volleyball-reels-studio/SKILL.md)

## Анализ и данные

- [Архитектура Rust и конвейер анализа](RUST_ARCHITECTURE.md)
- [Работа с исходным видео без деградации](PRIMARY_SOURCE_FIDELITY.md)
- [Ball Stabilizer](BALL_STABILIZER.md)
- [Touch Detector и Touch Attribution](TRAJECTORY_TOUCH_DETECTOR.md)
- [RTMPose и частота наблюдений](RTMPOSE_PIPELINE.md)
- [Геометрия площадки](RUST_COURT_GEOMETRY.md)
- [Метаданные розыгрышей](RALLY_METADATA.md)
- [Связь проекта и версий анализа](RUST_HEADER_CHAIN.md)
- [Разметка касаний и дообучение](TOUCH_ANNOTATION_DATASET.md)
- [Дополнительная модель TCN](VOLLEY3D_TCN_INTEGRATION.md)
- [Оценка качества](quality/README.md)

## Разработчику и агентам

- [Контракт агента](../AGENTS.md)
- [Доставка изменений](AGENT_DELIVERY_POLICY.md)
- [Процесс разработки](DEVELOPMENT_WORKFLOW.md)
- [Проверки](TESTING.md)
- [Правила интерфейса](UI_LAYOUT_POLICY.md)
- [Критерии улучшения анализа и монтажа](AGENT_IMPROVEMENT_CHECKLIST.md)
- [Параллельная работа и шаблоны](AGENT_PARALLEL_WORKFLOW.md)
- [Rust workspace](../rust/README.md) · [HTTP-приложение](../rust/volley-app/README.md) · [WASM](../rust/volley-web/README.md)
- [Машиночитаемые владельцы](RUST_CANONICAL_OWNERS.json)
- [Текущий API и возможности монтажа](../.agents/skills/volleyball-reels-studio/references/current-runtime.md)

## История и обязательные свидетельства

- [Журнал принятых изменений](RUST_MIGRATION.md)
- [Закрытая дорожная карта Rust](RUST_ROADMAP.md) · [машиночитаемый источник](RUST_ROADMAP.json) · [инвентарь миграции](RUST_REWRITE_SCOPE.md)
- [Историческая оценка этапа 84](RUST_BENCHMARK_84.md) и связанные машинные свидетельства
- [Пилотная независимая оценка](RUST_PILOT_ANNOTATION.md)
- [Бета v0.1.0-beta.1](releases/v0.1.0-beta.1.md) — сведения о конкретном релизе, не руководство к текущему develop.

Старые Python-руководства, дублирующие описания, завершённые планы и промежуточные отчёты удалены из рабочего дерева. Их неизменённые версии доступны в [снимке до очистки](https://github.com/Amazomic/volleyball_studio/tree/a36440727d5d9dc103514ece54228283a52df83f). Для исторического пути из журнала используйте `git show a36440727d5d9dc103514ece54228283a52df83f:<path>`. Отчёты, на которые ведут прямые ссылки из неизменяемого журнала, и обязательные свидетельства оставлены на месте.

Действующий код, владельцы и сохранённый профиль анализа имеют приоритет над датированными исследованиями. Новые постоянные документы добавляйте сюда только при наличии самостоятельного актуального контракта; закрытые планы и устаревшие дубли храните в Git-истории.
