# Привязки игроков на склейке ролика

Проверка 2026-11-16, база `b2b6ce6`, ветка `rust`.

## Причина

В проекте `d6451abac405` на границе кадров 850–855 исчезают три устойчивых трека (1, 2, 7), появляются два (11, 12), а геометрически подходящие ID 4 и 6 продолжаются на других людях. Пользователь подтвердил, что исходник склеен из разных эпизодов.

Детектор смены состава требовал минимум три исчезнувших и три новых устойчивых трека. Граница не распознавалась. Фотоавтопривязка выбирала одно имя для всего длинного ID: 4 → Раиля, 6 → Ирек. На кадре 940 ID 6 уже описывал Влада во время передачи; ID 4 описывал Ирека.

## Исправление

В существующем `cpu_retrack/identity_turnover.rs` разрешены два новых устойчивых трека при трёх исчезнувших. Остальные ограничения сохранены: длительные наблюдаемые треки, компактная непересекающаяся граница, минимум один продолжающийся трек, достаточный контекст с обеих сторон. Версия политики — `measured_roster_turnover_handoff_v3`. Это эвристика смены состава, не универсальный детектор видеосклеек.

## Проверка эпизода

- Исходный GPU fusion: `f8b75f104e4146839c2e836e71aa63a9`.
- Старый replay: `1c7c8f783c134080a9112f7c065fc0e4`, SHA `659c66a9bbbbe80e8ef4620d4304706ae0996d8d71e631a5781f8ce5ec64565c`.
- Старый rally run: `6eebc35ae5e14fa59c03272f5f91839a`.
- Новый запуск CPU: `011032a8b4824da1b11d7f4bfe151b61`, статус `complete`.
- Новый replay: `b4cdd64616cb4f9f86cb30011a9a5530`.
- Новый rally run: `aefb50dd9fdb4a83ab907480b81403ad`.
- Контакт на кадре 940 (31,333 секунды): `overhead_pass`, ID 15 → Влад.
- Дальний правый игрок на том же кадре: ID 14 → Ирек.
- Старые ID 4 → Раиля и 6 → Ирек сохранены для предыдущего участка. Исходные версии и GPU evidence не переписывались; дорогой GPU-инференс не запускался. Фото-ReID повторно выполнен штатной CPU-постобработкой.
- Автопривязка всего ролика остаётся неполной: есть неназначенные треки. Проверка одного эпизода не доказывает качество всех идентичностей.

## Проверки кода

Новая регрессия на фактических диапазонах сначала упала (`[]` вместо `[852]`), затем прошла. Все шесть тестов `identity_turnover` прошли, включая сохранение исходных данных и отсутствие разделения при частичной потере треков. `cargo fmt --all -- --check`, `cargo check --workspace --offline`, тесты остальных пакетов workspace, WASM/Node, ownership и roadmap self-tests прошли.

Полный `cargo test --workspace --offline` остановился на `volley-app`: 303 passed, 13 failed, 7 ignored. Падающие тесты относятся к модулю автопривязок с уже существовавшими незакоммиченными изменениями; в этой задаче модуль не менялся:

- `ambiguous_roster_fallback_remains_unassigned`
- `automatic_links_do_not_fall_back_to_the_wrong_player`
- `automatic_photo_links_enforce_threshold_and_conflicting_tracks`
- `closed_roster_does_not_override_best_photo_match_by_elimination`
- `co_visible_tracks_do_not_fall_back_to_weaker_other_player_match`
- `highest_photo_similarity_outweighs_track_length`
- `ilnur_late_id_13_can_join_id_12_after_measured_track_handoff`
- `photo_refresh_preserves_confirmed_links_and_manual_exclusions`
- `rally_scope_excludes_nonplaying_boundary_overlap_from_auto_assignment`
- `shorter_correct_track_beats_longer_wrong_co_visible_track`
- `six_player_two_rally_replay_links_every_measured_track_without_overlap`
- `strongest_long_lived_track_wins_overlapping_assignment`
- `strongly_matched_player_is_not_stolen_by_weaker_track`

Release пересобран и штатно перезапущен после проверки отсутствия активных заданий. API нового эпизода проверен. Браузерная проверка остановлена Computer Use: инструмент не смог надёжно определить URL окна Windows. Для отображения результата нужны F5 и выбор новой CPU-версии. Независимый размеченный матч, GPU model tests и экспорт видео не проверялись.
