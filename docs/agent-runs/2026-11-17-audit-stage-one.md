# Аудит структуры: этап 1 — UI и проверки

База: e709708f548f32e68f285732893c83e740a0430d.
Ветка исполнителя: codex/audit-cleanup-stage-one.
Этап согласован пользователем отдельно; следующие этапы требуют нового согласования.

## Изменения

- Удалены неподключённый studio_highlights.js и тест удалённого интерфейса.
- Добавлен test_studio_montage_builder.mjs для действующего интерфейса и реальных HTML ID: состав проекта, профили атак/отчёта/избранного, SHA источника, повторное превью после изменения настроек, музыка, запуск, отмена, запрет запуска при пустом/чужом/невалидном превью и ошибке сервера.
- WSL-набор test_rust_faults.sh переведён на новый тест; test_studio_montage_tab.mjs запрещает возврат старых файлов.
- WASM-фикстура использует person_job_id и vball_job_id; обе GPU-задачи не должны появляться отдельными анализами.
- check-all.ps1 передаёт ownership --self-test и roadmap --check --self-test.

## Проверки

PASS: test_studio_montage_tab.mjs, test_studio_montage_builder.mjs, test_studio_domain_wasm.mjs, check_rust_ownership.mjs --self-test, check_rust_roadmap.mjs --check --self-test.
PASS: фактический foreach из AST check-all.ps1 с перехватом аргументов node: ownership/roadmap получают самопроверки, остальные тесты не получают лишних аргументов.
PASS: bash -n scripts/ci/test_rust_faults.sh; git diff --check; независимое ревью без замечаний.

Расширенная Windows-проверка 70 не-stage84 Node-гейтов: 54 PASS, 16 FAIL. Все 16 сбоев повторены на отдельном git archive исходной базы; новых сбоев нет. Список: test_rust_packaging, test_studio_activity_icon, test_studio_assignment_transition, test_studio_dom_ids, test_studio_editor_ui, test_studio_existing_project, test_studio_group_tracks, test_studio_header_chain, test_studio_navigation_cache, test_studio_overlay_continuity, test_studio_player_remove_track, test_studio_player_suggestions, test_studio_project_create_ui, test_studio_raw_gpu_preview, test_studio_repeat_photo_analysis, test_studio_settings_cache (расширение .mjs).

На чистой базе DOM дополнительно содержит прежнюю ссылку studio-raw-person-model; её исправление уже есть среди независимых незакоммиченных изменений основного дерева и в этот этап не включено. Windows packaging упирается в передачу Windows-пути WSL bash; отдельный запуск packaging внутри WSL доходит до прежнего несовпадения deployment-контракта start|stop|status|restart|doctor|check.

Rust-код не изменён. Rust fmt/check/test, CUDA, реальные матчи и lifecycle не запускались. WASM для браузерных проверок скопирован из существующего локального артефакта основной Studio; сборка не выполнялась. Перезапуск Studio не требуется: удалённый JS не раздавался сервером; остальные правки относятся к тестам и запуску проверок.

## Интеграция

Интегратор переносит только перечисленные правки, добавляет собственную запись в docs/RUST_MIGRATION.md и повторяет целевые проверки на основном дереве. Независимые незакоммиченные изменения не включаются в коммит. Остальные сбои требуют отдельного согласования объёма работ.
