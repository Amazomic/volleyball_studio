# RF-DETR Medium — нативный эксперимент для Volley

**Статус:** отдельная ветка, RTMDet-Nano остаётся выбором по умолчанию. Studio пользователя и текущие задания не перезаписываются. SAM 3.1 по-прежнему ожидает доступа к весам.

## Точки интеграции

- `rust/volley-inference/src/person.rs`: RF-DETR Medium ONNX Runtime CUDA/CPU, вход 1×3×576×576, RGB → bilinear resize → ImageNet normalization, выходы `pred_boxes` и `pred_logits`, COCO person = индекс 1, порог 0.28.
- Обнаруженная рамка остаётся наблюдением при неудачной RTMPose; координаты скелета только дополнительная информация. Никакие касания или личности при этом не подтверждаются автоматически.
- `rust/volley-app/src/studio_pipeline.rs`: GPU cache profile отличается между RTMDet и RF-DETR, чтобы избежать ошибочного повторного использования данных.
- `rust/volley-app/src/inference_full.rs`: имя `person_pipeline` в GPU-манифесте различает реализации.

## Вес модели

Доступный без одобрения **сторонний ONNX-экспорт**, не официальный релиз ONNX от Roboflow:
https://huggingface.co/PierreMarieCurie/rf-detr-onnx

Проверенный SHA-256 модели `rf-detr-medium.onnx`:
`18250b5a3a9552ae117d592f8e8c27fe52438d70c914a77a8913fc45d8f29e69`.

Установка в изолированный checkout из WSL:

```bash
bash scripts/experiments/sam31/install_rfdetr_medium.sh
```

Скрипт проверяет SHA-256 и не перезаписывает посторонние веса.

## Переключение

Нативный `PersonEngine::load` понимает переменную окружения WSL-процесса:

- Не задана или `VOLLEY_PERSON_DETECTOR=rtmdet-nano`: исходное поведение.
- `VOLLEY_PERSON_DETECTOR=rfdetr-medium`: альтернативный RF-DETR Medium.

**Переменная должна присутствовать в окружении процесса самого Rust-сервера**; установка переменной в PowerShell не гарантирует автоматической передачи в WSL. UI-переключатель параметров анализа пока не добавлялся. Для теста из WSL:

```bash
source /mnt/c/Users/amazomic/Desktop/github/fast-volleyball-tracking-inference/scripts/setup_rust_ort_cuda.sh check
export VOLLEY_PERSON_DETECTOR=rfdetr-medium
export CARGO_TARGET_DIR=/tmp/volley-sam31-build
cargo run --manifest-path rust/Cargo.toml -p volley-app --release --offline --example bench_person_tracks -- \
  "$PWD" /tmp/volley-sam31/sample_640x360.mp4 /tmp/volley-sam31/rfdetr_native_medium.jsonl
```

Отмена: `unset VOLLEY_PERSON_DETECTOR`. Вернуть прежний детектор можно без удаления данных и модели.

## Проверки и ограничения

- Rust unit tests и `cargo check`.
- Реальный **Rust-native** GPU-прогон 180 кадров на RTX 5070 Ti Laptop: `/tmp/volley-sam31/rfdetr_native_medium_final.jsonl`, 25,42 FPS; 1229 рамок после Rust-трекера; 9 provisional Track ID; 3 разрыва внутри ID (максимум 62 кадра). Для исходного RTMDet: 10,99 FPS, 732 рамки, 8 ID, 30 разрывов (максимум 72 кадра).
- Прежний Python/ONNX + Rust post-processing benchmark **не равен** нативной цепочке. Нативная обработка иначе фильтрует дубликаты, число ID и разрывы могут отличаться.
- Для решения о постоянной замене нужны длинное видео, ручная разметка 4 волейболистов, анализ false positive (зрители), switch IDs, FPS и VRAM.
- Основной рабочий checkout содержит незавершённые правки нескольких агентов. Вливать изменения туда можно только после согласования с интегратором и синхронизации кодовой базы.

**Важное уточнение:** 3 разрыва внутри ID не исключают внезапного назначения *нового* ID тому же игроку. Показатели близки к предварительному ONNX-эксперименту, но не тождественны из-за предобработки, NMS и порядка RTMPose. Постоянное использование требует ручной проверки идентичностей и работы на длинных роликах.
