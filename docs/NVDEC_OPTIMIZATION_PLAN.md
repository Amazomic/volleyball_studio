# План оптимизации NVDEC / CUDA первичного анализа

Дата открытия: 2026-11-21. Базовая ветка: `develop`, коммит `1a295ef5`.
Владельцы: `rust/volley-media/src/nvdec.rs`, `rust/volley-app/src/inference_full/joint/`, `rust/volley-inference/src/gpu_preprocess.rs`.

## Реализация и верификация

- [x] P0. Восстановить NVRTC в Ubuntu-25.04 (WSL) и проверить `decode_backend=nvdec_zero_copy` на совместимом видео, не прерывая активный анализ.
- [x] P0. Журналировать конкретную причину zero-copy fallback.
- [x] P1. Заменить избыточный `cuCtxSynchronize` на CUDA streams/events с корректным ожиданием перед ORT и unmap.
- [x] P1. Переиспользовать GPU scratch-буферы RF-DETR/RTMPose.
- [x] P1. Не вызывать `cuvidMapVideoFrame` для кадров GOP-прогрева и исключённых кадров вне выбранных розыгрышей; сохранять индексы/PTS.
- [x] P2. Кэшировать metadata и display timeline по уникальному идентификатору/штампу видео, инвалидировать при изменении.
- [x] P2. Кэшировать успешно скомпилированный NVRTC PTX на жизнь процесса.
- [ ] P2. Проверить возможность переиспользования CUVID decoder между интервалами; не делать, если flush/reset нарушает порядок отображения.
- [ ] P0. Тесты: Rust fmt/clippy/test, frame parity, 24/30/60 FPS, сегменты и отмена, GPU smoke, реальное видео, сравнимые замеры до/после.

## Проверено 21.11.2026

- PASS: CUDA 12.9 NVRTC загружается из существующего ORT-бандла WSL, динамическая компиляция CUDA kernels успешна.
- PASS: WSL RTX 5070 Ti, изолированное синтетическое видео; GPU smoke сохранил decode_backend=nvdec_zero_copy и decode_pipeline=gpu_resident_nvdec_primary_v1 для обоих ONNX-проходов. 92 кадра, 10 окон, 13.241 с на повторной проверке после защиты unmap, без эталона производительности.
- PASS: CUDA P016 rotation, ограниченный seek NVDEC, перенос индексов кадров, проверки FFmpeg pixel parity на Windows и NVDEC H264 в WSL.
- PASS: scripts/check-all.ps1 c RUST_TEST_THREADS=1 (WASM, fmt, Clippy, Rust тесты, 92 Node gate), повторный WSL GPU smoke с проверкой двух манифестов.
- PENDING: сравнительный бенчмарк на одинаковом видео/настройках и проверка точности на реальном матче; запуск новой сборки отложен, поскольку рабочая Studio сообщает shutdown_safe=false, active_jobs=studio_analysis и gpu_inference.
- PENDING: переиспользование CUVID-декодера между несмежными GOP: требуется безопасная стратегия flush/reset, прежде чем внедрять. Соседние пересекающиеся GOP уже объединяются в текущем plan_decode_windows.

## Дальнейшая задача

- [ ] R1. Рассмотреть **Video Codec SDK 13.1**: прямой вывод NVDEC в `CUarray` и асинхронное декодирование. Оценить наличие API на RTX 5070 Ti под Windows/WSL, изменения CUDA kernels и ORT hand-off, B-frame presentation order, fallback и практический прирост на одном видео до внедрения.

## Критерии завершения

1. Выбранный backend и причина fallback видны в диагностике; нельзя выдавать обычный NVDEC с `hwdownload` за zero-copy.
2. Совместимый исходник даёт `nvdec_zero_copy`, сохраняются пиксели и таймлайн.
3. Не возникает use-after-unmap, потери кадров, неограниченного потребления VRAM и изменений контрактов evidence.
4. Без сопоставимого бенчмарка не заявлять прирост FPS.
5. Разработка через `develop`, не нарушая работающие задания; `master` только для релиза.
