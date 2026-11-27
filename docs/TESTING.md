# Локальные проверки без GitHub Actions

GitHub Actions отключены (workflow-файлов нет), автоматических проверок при
push или PR не существует. Весь CI запускается на своей машине; платный
hosted runner не предполагается. Python-эпоха (`scripts.ci.run_checks`,
`pytest tests`) удалена — единственный исполняемый Studio теперь Rust
workspace `rust/Cargo.toml` (контракт: `AGENTS.md`).

## Выбор объёма проверки

Во время реализации запускать целевые проверки владельца и затронутых потребителей. Перед коммитом/PR с изменением приложения выполнить полный локальный прогон ниже; успешный прогон не повторять без новых правок или сомнений в результате. GPU, реальные матчи и экспорт проверяются отдельно по матрице.

Для изменений только документации достаточно проверить существование изменённых путей/ссылок и `git diff --check`; при затрагивании контрактов владельцев или roadmap также выполнить соответствующие self-tests. Это исключение не относится к исполняемым скриптам, конфигурации, зависимостям и ресурсам приложения. Сборка Rust/WASM и перезапуск Studio для документации не нужны.

В отчёте указывать базовый SHA или проверенный diff, фактические команды и PASS/FAIL/NOT TESTED. Исходный сбой подтверждать той же командой на изолированной базе либо воспроизводимым свидетельством для этого SHA. Прогон с исключёнными тестами не называть полным; перечислять исключения и причины отдельно.

## Полный локальный прогон (для изменений приложения перед коммитом/PR)

Windows native, требуется Rust (stable, clippy, таргет
`wasm32-unknown-unknown`), Node.js 22+ и git-lfs (модели `models/`, `ov/` хранятся в LFS; после клона — `git lfs install && git lfs pull`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check-all.ps1
```

Гейт выполняет:

1. сборку WASM-модуля UI (`cargo build -p volley-web --target
   wasm32-unknown-unknown --release --offline --locked` + копия в
   `rust/volley-app/static/volley_web.wasm`); wasm больше не коммитится,
   а `volley-app` встраивает его через `include_bytes!`
2. `cargo fmt --all -- --check`
3. `cargo clippy --workspace --all-targets --offline --locked -- -D warnings`
4. `cargo test --workspace --offline --locked` (пропустить: `-SkipRustTests`,
   тогда в журнале обязательно записать **RUST TESTS NOT RUN**)
5. все не-GPU `scripts/ci/*.mjs`: упаковка и CRLF-контракты, ownership и
   roadmap self-tests, WASM-мост, браузерные DOM/UI-контракты, включая
   статическую проверку «каждый id из JS существует в studio.html»
   (`test_studio_dom_ids.mjs`).

Эквивалент в WSL (bash):

```bash
cd rust
cargo build -p volley-web --target wasm32-unknown-unknown --release --offline --locked
cp target/wasm32-unknown-unknown/release/volley_web.wasm volley-app/static/volley_web.wasm
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --offline --locked -- -D warnings
cargo test --workspace --offline --locked
cd ..
for gate in scripts/ci/*.mjs; do
  case "${gate##*/}" in
    test_rust_parity_84.mjs|test_rust_benchmark_84.mjs|benchmark_rust_parity_84.mjs|audit_python_baseline_84.mjs|test_rust_real_media_84.mjs) continue ;;
    check_rust_ownership.mjs) node "$gate" --self-test || exit 1 ;;
    check_rust_roadmap.mjs) node "$gate" --check --self-test || exit 1 ;;
    *) node "$gate" || exit 1 ;;
  esac
done
```

## Отдельные проверки (не входят в check-all)

| Скрипт | Что нужно | Когда запускать |
| --- | --- | --- |
| `scripts/ci/test_rust_lifecycle.sh` (WSL) | свободный репозиторный lock: живой сервер на 8875 должен быть остановлен | изменения lifecycle/lock/shutdown |
| `scripts/ci/test_rust_faults.sh`, `test_rust_shutdown_race.sh` (WSL) | изолированный /tmp, тестовые порты | изменения обработки ошибок/остановки |
| `scripts/ci/test_rust_cuda_runtime.sh`, `smoke_rust_ort.sh` (WSL) | `libonnxruntime.so` + CUDA provider (`scripts/setup_rust_ort_cuda.sh`, `ORT_DYLIB_PATH`) | изменения GPU-инференса |
| `scripts/ci/test_rust_real_media_84.mjs`, `test_rust_benchmark_84.mjs`, `benchmark_rust_parity_84.mjs`, `audit_python_baseline_84.mjs` | реальные видео, ffmpeg, GPU, замороженный Python-чекпойнт | исторические проверки паритета; для текущей архитектуры выбирать актуальные размеченные сцены |
| `scripts/ci/test_rust_parity_84.mjs` | Node.js; синтетические временные фикстуры, без GPU и реального матча | математика и строгие контракты оценщика; не измерение точности |
| `scripts/ci/test_rust_docker_smoke.sh` | Docker | изменения Dockerfile/упаковки |

Прежние приёмочные прогоны фиксировали отсутствие `libonnxruntime.so`
в WSL и «CUDA failure 999» на RTX 5070 Ti. Это исторические наблюдения,
не текущий диагноз машины. Перед GPU-проверкой выполнить
`start-volley-ui.cmd /doctor` и отдельно подтвердить работу CUDA provider. Пропуск GPU/медиа-проверок обязательно
фиксировать в журнале как **NOT TESTED**, а не как PASS.

## Матрица проверки

| Изменения | Обязательные дополнительные проверки |
| --- | --- |
| UI / статика | `node scripts/ci/test_studio_dom_ids.mjs`, WASM-сьюты, ручной F5 и переключение вкладок |
| HTTP API / хранилище | `cargo test -p volley-app api_http_tests` (in-process `/api/*` на временном корне) |
| Touch / траектория / ралли | Rust-тесты владельца + реальные сцены (stage84, отдельно) |
| Игроки / ReID | `cargo test -p volley-core reid`, `cargo test -p volley-app reid` |
| Quick Reel / монтаж | Rust-тесты + просмотр MP4; медиа-гейты отдельно |
| Lifecycle / запуск | `test_rust_lifecycle.sh` в WSL при остановленном живом сервере |
| Модель / GPU | ручной бенчмарк на разрешённом видео, подтверждение CUDA provider |

## Журнал и слияние

Каждый проверенный инкремент = коммит в `develop` + запись в
`docs/RUST_MIGRATION.md` с фактическими командами и результатами
PASS/FAIL/NOT TESTED. Приёмка миграции закрыта на этапе 85 решением владельца; исторический Python-паритет не блокирует изменения. Production и качество текущей архитектуры требуют независимых проверок. Обычные review и merge — в
`develop`; `master` обновляется исключительно при выпуске проверенной версии.
Автоматического merge gate нет.

Документационные исправления уже опубликованной версии не меняют тег и бинарные артефакты; для них достаточно проверки ссылок, путей и `git diff --check`, если не затронуты исполняемые контракты.

### Внешнее медиа для опциональных RF-DETR CUDA-тестов

Укажите `VOLLEY_AB_MODEL_ROOT` (репозиторий с весами моделей) и `VOLLEY_AB_SOURCE` (внешний тестовый клип с видимыми людьми, не менее 11 кадров при 5 fps). Оба ignored-теста в `rust/volley-inference/src/person.rs` требуют эти пути. Бенчмарк `scripts/benchmark_pose_models.py` требует явного `--source <path>`. Исторические отчёты сохраняют исходные пути.
