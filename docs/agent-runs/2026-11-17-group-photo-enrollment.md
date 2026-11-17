# 2026-11-17 · ReID-группа: сохранение фото всех ID

База: `64a3525f21b94c8e8719c94a6e4553154cdf4c3a`.
Ветка: `codex/20261117-group-photo-enrollment`.
Файлы: `rust/volley-app/static/studio_tracks.js`,
`scripts/ci/test_studio_track_assignment.mjs`.

## Исправлено
При подтверждении группы Track ID сохраняется выбранное фото каждого ID, а не только первого. Индексы фото фиксируются до обновления карточек. Повторно сохранённые фото по (проект, видео, replay, rally, ID, индекс, SHA) пропускаются. После каждого добавления берётся свежая ревизия альбома. Ошибка отдельного кадра не препятствует сохранению остальных, статус показывает результат и частичные ошибки.

Не меняются исходное видео, CPU/GPU evidence, автоматическая идентификация, межпроектная изоляция и утверждение принадлежности ID игроку.

## Проверено
`node --check rust/volley-app/static/studio_tracks.js` — успешно.
`node scripts/ci/test_studio_track_assignment.mjs` — успешно: 3 ID, разные фотоиндексы, повтор, цепочка ревизий и частичные ошибки.
`test_studio_group_gallery.mjs`, `test_studio_group_tracks.mjs`,
`test_studio_player_album.mjs`, `test_studio_track_assignment_busy.mjs`,
`test_studio_player_gallery_view.mjs`, `test_studio_identity_sync.mjs` — успешно.
Три существовавшие ещё до изменения ошибки в соседних тестах:
`test_studio_player_suggestions.mjs`, `test_studio_player_remove_track.mjs`,
`test_studio_assignment_transition.mjs` — подтверждены на исходной рабочей копии.
Сквозной анализ матчей и GPU-тесты не выполнялись.
