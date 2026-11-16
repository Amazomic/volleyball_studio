# Проверка согласования архитектуры — 2026-11-16

Реализация интегрирована в ветку `rust`, поверх исходного состояния `a7690db`. Пользовательские изменения сохранены; 7 из 10 первоначально изменённых файлов сохранены побайтно. Статистика, оверлей и HTML объединены с архитектурными изменениями; фильтр конфликтующих назначений одного кадра сохранён.

## Применённый результат

- Оригинал имеет проверяемый SHA-256, метаданные и относительную временную шкалу; рабочая шкала — 30 FPS.
- Модели читают оригинал через существующий FFmpeg-декодер с нормализацией 30 FPS. Prepared-превью новых анализов имеют отдельный ключ с SHA оригинала. Прежние schema-2 файлы сохранены; читатели выбирают медиа по версии evidence.
- Ручная и автоматическая геометрия сохраняются одним CAS-сервисом, последняя применённая разметка имеет приоритет. По умолчанию геометрия отсутствует. Сетка учитывается отдельно от гомографии пола, в том числе без углов площадки.
- Финальная атрибуция контакта и гипотеза типа выполняются в контексте принятого розыгрыша; общий selector используют timeline, статистика и монтаж. CPU-интерполяция поз не меняет GPU evidence.
- После физической обработки и ReID сохраняется `volley-montage-material/1`: SHA зависимостей, ревизии геометрии, личности и ручных типов действий, готовность и допустимые интервалы оригинала.
- Renderer потребляет сохранённый материал, проверяет его digest/revision, интервалы и ожидаемый SHA оригинала. Receipt фиксирует material bindings. Подмена оригинала после фиксации материала отклоняется.

## Проверки

- В изолированном checkout стандартный Rust workspace: 591 успешный тест; 11 native/host тестов по умолчанию ignored.
- В объединённом основном checkout: `cargo check --workspace --offline`, `cargo fmt --all -- --check`, тесты core/media/inference/web, проверки браузерного bridge, редактора, интерполяции, поз, выбора источника, ownership и roadmap выполнены.
- Настоящий CUDA smoke на синтетическом видео: Ball + Players → fusion → CPU replay → контакты → розыгрыши → сохранённый material. Проверены повторяемая ревизия и отказ после подмены оригинала. Нулевые игровые события в синтетическом изображении ожидаемы; это проверка исполнения/происхождения, не качества распознавания.
- Настоящий RTMPose CUDA forward: batch и individual inference совпадают в заданном допуске.
- Настоящий Rust-WASM: 270/270 статических надписей и 73/73 accessibility-подписей переведены, 506 RU/EN пар; браузерный bridge прошёл.
- Полный `cargo test --workspace --offline` в основном checkout: 299 app-тестов прошли, 13 упали, 7 ignored. Те же 13 ошибок воспроизведены на отдельной копии исходного `a7690db` с первоначальным пользовательским patch: 33 native-player-track теста прошли, 13 упали. Ошибки относятся к прежним ожиданиям запрета перекрывающихся автоматических привязок; политика из незакоммиченных пользовательских изменений сохранена. Полный набор не объявляется зелёным.
- Проверка качества на размеченном реальном матче не выполнялась. Оригинальные пользовательские видео, фотографии и анализы в development tests не использовались.

## Прежние ошибки полного набора

- `native_player_tracks::tests::ambiguous_roster_fallback_remains_unassigned`
- `native_player_tracks::tests::automatic_links_do_not_fall_back_to_the_wrong_player`
- `native_player_tracks::tests::automatic_photo_links_enforce_threshold_and_conflicting_tracks`
- `native_player_tracks::tests::closed_roster_does_not_override_best_photo_match_by_elimination`
- `native_player_tracks::tests::co_visible_tracks_do_not_fall_back_to_weaker_other_player_match`
- `native_player_tracks::tests::highest_photo_similarity_outweighs_track_length`
- `native_player_tracks::tests::ilnur_late_id_13_can_join_id_12_after_measured_track_handoff`
- `native_player_tracks::tests::photo_refresh_preserves_confirmed_links_and_manual_exclusions`
- `native_player_tracks::tests::rally_scope_excludes_nonplaying_boundary_overlap_from_auto_assignment`
- `native_player_tracks::tests::shorter_correct_track_beats_longer_wrong_co_visible_track`
- `native_player_tracks::tests::six_player_two_rally_replay_links_every_measured_track_without_overlap`
- `native_player_tracks::tests::strongest_long_lived_track_wins_overlapping_assignment`
- `native_player_tracks::tests::strongly_matched_player_is_not_stolen_by_weaker_track`

## Применение

Release пересобран. Штатный launcher проверил `shutdown_safe=true`, `active_jobs=[]`, завершил PID 576408 и запустил PID 668010. Новый процесс отвечает на HTTP, `build_profile=release`, `person_detector=rfdetr-medium`. Проверены новая кнопка ручной сетки и положение CPU-настройки поз в выдаваемом HTML.

Обновление открытой вкладки остановлено защитой Computer Use, которая не смогла надёжно определить текущий URL браузера Windows. Для загрузки обновлённого UI требуется F5 в существующей вкладке Studio. Другие UI-действия после остановки Computer Use не выполнялись.

Stage 85, полная миграция продукта и качество игрового анализа не объявляются завершёнными этим архитектурным инкрементом.
