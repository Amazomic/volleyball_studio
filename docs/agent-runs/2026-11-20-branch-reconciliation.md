# Согласование старых локальных веток с develop — 2026-11-20

- База интеграции: `a9b033703156a17ea4648038cb08f5fa80579bb7` (`origin/develop`).
- Решение: **историческое слияние стратегией `ours`**, без переноса старых исходников. У веток остаются уникальные Git-коммиты, однако актуальный продукт уже содержит соответствующие реализации или использует заменивший их контракт. Слепое применение изменений из старых баз могло вернуть устаревшее поведение.
- Владельцы актуальных реализаций: `rust/volley-core/src/court_detection.rs`, `rust/volley-app/src/court_detection.rs`, `rust/volley-app/src/native_player_tracks/photo_comparison.rs`, `rust/volley-app/static/studio_tracks.js`, `rust/volley-app/static/studio_tasks.js`, `rust/volley-app/src/preparation/primary_preview.rs`, `rust/volley-app/static/studio_montage_builder.js`, `rust/volley-app/src/native_rally_evidence.rs`, `rust/volley-app/src/quick_reel_source.rs`.

| Ветки | Разрешение |
| --- | --- |
| `codex/20261114-court-fix/integrate`, `codex/20261114-court-partial/integrate` | Действующий детектор содержит консенсус кадров, восстановление частичной разметки, временную медиану и признак предположительных границ. |
| `codex/20261116/touch-types-rally-identity` | Действующая классификация использует пять именованных типов и неопределённое «Касание» с отдельной привязкой игрока; старые правки UI/CPU не накладывались повторно. |
| `codex/20261116/track-appearance-guard` | **Устарело по выбранному алгоритму:** не восстановлен собственный appearance-merge потерянных Track ID; текущий путь использует ByteTrack, старый альтернативный merge не включаем. Исторический файл `cpu_retrack/track_appearance.rs` намеренно не добавлен. |
| `codex/20261118-reid-player-match` | Сохранены вычисления фото-совпадений и кэш в актуальном Rust; лимит референсов уже увеличен до 4096. Расчёт выполняется при постобработке, а не автоматически заново при открытии вкладки. |
| `codex/20261119/sidebar-job-progress` | Боковая панель читает статус монтажа, CPU/GPU и обновляет прогресс; новые обработчики уже присутствуют. |
| `codex/audit-cleanup-stage-one`, `codex/audit-cleanup-stage-two`, `codex/audit-cleanup-stage-three`, `codex/audit-cleanup-stage-four`, `codex/audit-cleanup-stage-five`, `codex/audit-cleanup-stage-five-preparation` | Актуальные модули и документация включают разделённое хранение, Quick Reel source, rally evidence, владельцев и проверки; устаревшие удаления тоже уже учтены. |
| `codex/primary-analysis-preview` | Отдельные предпросмотры по видео и SHA источника, HTTP-получение и декодирование уже присутствуют в актуальном Rust. |
| `feature/montage-type-picker-20261119` | Единый выбор типа монтажа, условные поля и профиль работы присутствуют в интерфейсе. |

**Проверка:** 13 профильных Node-контрактов (касания, таймлайны, боковые задачи, ReID, монтаж, предпросмотр, Quick Reel, дообучение) — PASS на базовом SHA; проверки `git diff --check` и ссылок журнала — PASS. Изменений исполняемого кода нет; Rust/GPU/реальный матч/экспорт в этом проходе NOT TESTED. Работающая Studio не перезапускалась.

**Область сохранения:** пользовательские локальные изменения в главной копии и других рабочих деревьях остаются нетронутыми. Слияние только согласует историю веток, не переносит незакоммиченные файлы и не является подтверждением незаконченных работ в других worktree.
