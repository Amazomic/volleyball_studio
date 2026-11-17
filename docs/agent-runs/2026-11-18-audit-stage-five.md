# Аудит структуры: этап 5 — материалы и HTTP-предпросмотр

База: 671f09502472ca6e53467bb321c400289fe3c616.
Ветка: codex/audit-cleanup-stage-five. Этап отдельно согласован пользователем; продолжение подтверждено 18.11.2026.
План: docs/superpowers/plans/2026-11-18-rally-material-boundaries.md.

## Границы и совместимость

native_rally_preview.rs оставляет DTO и три HTTP-обработчика: около 190 строк вместо примерно 2500. Схемы и чтение/проверка неизменяемых Rally/Touch/Replay выделены в native_rally_evidence.rs; камера — в camera.rs, существующие тесты — в tests.rs. MaterialBinding, нормализация/hash, публикация, bindings и проверка ревизии/оригинала находятся в montage_material.rs.

Renderer вызывает material owner, не HTTP-preview. Source/material owners не вызывают renderer или HTTP-preview; прочие production-потребители используют evidence/material напрямую. Старых реализаций и re-export нет. Ссылки на preview остались только у router и намеренного вызова HTTP в игнорируемом интеграционном тесте.

Сохранены прежние канонические владельцы, пути sidecar, поля binding, SHA, нормализация чисел, ревизии, порядок/границы клипов, provisional/manual статусы, маршруты и ответы. Это перенос единственной реализации, не новый движок. Новые source-format adapters зарегистрированы в ledger.

## Проверки

RED: архитектурный тест обнаружил исходный renderer -> preview цикл. До извлечения PASS нового теста сериализации binding и 15 прежних source/material-тестов.
GREEN: архитектурная проверка, включая production-потребителей/camera, PASS; 14 перенесённых evidence-тестов, material digest и binding JSON PASS. Механически сравнены 25 функций с базой modulo форматирование и только новые имена владельцев/существующая wrapper-функция. Независимый reviewer отдельно сравнил тела методов, схемы и тесты: замечаний нет.

WSL fmt/check PASS; полный cargo test --workspace --offline --locked -- --test-threads=1: 657 PASS, 0 FAIL, 17 ignored. Worktree target /home/amazomic/.cache/volley-audit-stage-four-build не используется для combined tree. Ownership/roadmap self-tests, существующий WASM и git diff --check PASS.

Strict Clippy имеет baseline-проблемы предыдущего этапа, WSL-компонент отсутствует; здесь отдельно не запускался. Новых cargo check warnings нет. Полный Node-набор с известными сторонними сбоями, GPU-модельные/матчевые прогоны и визуальная оценка NOT RUN. Только изолированные фикстуры.

## Интеграция и применение

Read-only index patch check успешен; working-tree patch одного overlay-контекста не подходит из-за параллельных пользовательских правок. Dirty-файлы требуют точной замены namespace references без подмены их логики старым worktree-текстом, отдельно от собственного index patch. Нужны повторные combined fmt/check/tests и безопасный rebuild/restart по active_jobs/shutdown_safe.

Автоматический reviewer первоначально отклонил staging overlay, приняв его за primary-файл с независимыми правками. Read-only проверка явного worktree/ветки и обратная namespace-нормализация доказали: isolated overlay отличается от фиксированной базы только именами владельца/форматированием, а primary содержит отдельные 115/5 строк. Повторный staging ограничен этим изолированным checkout.

## Combined-проверка 18.11.2026

Интегрированы 23 собственных blob из 9902b52 в rust поверх 8f313cf через отдельный index patch. Для шести dirty Rust-файлов выполнены только namespace-замены/регистрация модулей с raw и rustfmt-canonical обратной проверкой. Индекс побайтово совпал с worker-коммитом; параллельная логика осталась вне индекса.

Combined cargo check PASS; полный workspace: 679 PASS, 0 FAIL, 17 ignored. Dependency gate, ownership/roadmap self-tests, live montage builder, DOM и staged diff-check PASS. Combined fmt FAIL только на двух сторонних pose_inference-файлах; WASM FAIL на сторонней неподключённой к каталогу подписи «Интерполированные рамки» (271/272). Эти независимые изменения не исправлялись и не включались в этап. Изолированный fmt/WASM этапа ранее PASS.

На проверке применения Studio PID252719 выполняет studio_analysis и gpu_inference, shutdown_safe=false. Перезапуск не запрашивался: активные задачи не прерывались.

Release cargo build --release --locked --offline -p volley-app PASS (2m11s), без остановки процесса. Повторная проверка после сборки: тот же PID252719, shutdown_safe=false, studio_analysis/gpu_inference активны. Применение ОТЛОЖЕНО; после завершения задач нажать «Перезапустить студию». Новый процесс/served assets не проверялись, потому что безопасного перезапуска ещё не было. GPU-модельные и реальные матчевые прогоны этапом не запускались.
