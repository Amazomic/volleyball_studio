# ByteTrack default, legacy ID association disabled

Дата: 2026-11-18. База: `41b2147482bb5b64991bdbc49cd827f7c237718c`. Изолированная ветка: `codex/20261118/bytetrack-default`.

## Что изменено

- `rust/volley-core/src/tracker/bytetrack.rs`: ByteTrack на Rust: Kalman XYAH для движения, Hungarian one-to-one association, высокий и низкий пороги уверенности, только высокая уверенность создаёт новый ID, низкая может продолжать уже подтверждённый трек.
- `rust/volley-app/src/cpu_retrack.rs`: в обработке GPU evidence теперь создаётся `ByteTracker`, а не `ProvisionalTracker`. Отключены наследованные склейки и разделения ID (`track_merge`, `identity_turnover`), чтобы результаты ByteTrack можно было честно измерить.
- `rust/volley-inference/src/person.rs`: RF-DETR отдаёт сырые подтверждения от 0.10, включая 0.10–0.45 для второго этапа ByteTrack. Рамки с низкой уверенностью не создают Track ID.
- Статусы `track_ids_provisional`, неизменяемые GPU JSONL и разметка измеренных поз остаются совместимыми; отдельная интерполяция в CPU только заполняет интервалы уже существующего ID.
- Версия отслеживания: `bytetrack_xyah_two_stage_v1`. Старый метод сохранён в кодовой базе, но отключён в production.

## TODO(bytetrack-removal)

**УДАЛИТЬ** прежние `ProvisionalTracker`, `track_merge`, `identity_turnover`, `identity_handoff`, геометрические/pose-ассоциации и их оставшиеся тесты **только после** качественной проверки ByteTrack на размеченных видео.

Критерии удаления: сравнить прежнюю стабильную версию и ByteTrack на одинаковой RF-DETR модели и настройках; получить меньше или не больше ID switches, fragmentation, false tracks и пропущенных игроков; проверить co-visible игроков и моменты кратковременных перекрытий, разрывы не более 90 кадров, перемещения по площадке. Задокументировать результаты и подтвердить готовность. До этого TODO и старый код остаются для отката.

## Запуск и проверки

Проверять на изолированном CPU snapshot; не затрагивать пользовательские исходники и активные процессы. Перед применением в общую студию требуется согласованная интеграция ветки и проверка отсутствия активных заданий.

Напоминание хранится в данном файле и в `TODO(bytetrack-removal)` в исходниках. Создание GitHub Issue невозможно: в данном репозитории Issues отключены (GitHub API 410).

Оригинальная реализация алгоритма: https://github.com/FoundationVision/ByteTrack/blob/main/yolox/tracker/byte_tracker.py

## Прозрачность обработки

Manifest `volley-native-cpu-tracking/1` содержит `track_association_version=bytetrack_xyah_two_stage_v1`, `legacy_tracker_enabled=false`, `track_detector_low_score=0.10`, `track_detector_high_score=0.28`, `bytetrack_low_confidence_matches` — число реально присоединённых слабых обнаружений. GPU model stack теперь `v7`, чтобы старый источник с порогом 0.28 не был принят вместо нового; версия `person_pipeline` тоже обновлена. После интеграции выполнить новый первичный анализ и CPU-постобработку для оценки ByteTrack.

## Результаты проверок

Windows: `cargo fmt --all -- --check` и `cargo check --workspace --offline` пройдены. Тесты `volley-app` — 339 проверок (331 выполнена, 8 ignored), 1 давно существующий тест статистики хранилища пропущен из-за `Result::unwrap_err()` при реальном `(Null, \"absent\")`; `volley-core` и `volley-inference` проходят целиком. `node scripts/ci/test_volley_web_wasm.mjs` успешен. Полноценный A/B тест на размеченном волейбольном видео ещё не выполнялся.
