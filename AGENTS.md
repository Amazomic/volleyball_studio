# Volleyball Studio — agent contract

Keep shared rules here and details in linked documents. Follow the requested scope.

**Обязательный для всех агентов контракт:** [docs/AGENT_DELIVERY_POLICY.md](docs/AGENT_DELIVERY_POLICY.md). Каждая задача делается в новой локальной ветке и отдельном worktree, затем проверяется, сливается интегратором в `develop`, пушится **только** в `origin/develop` и применяется к работающей Studio с проверкой точного SHA сборки. `master` не изменять, не сливать и не пушить без прямого указания пользователя — включая релизы. Ответ **«Выполнил»** возможен только после проверенной пересборки и развёртывания со ссылкой на Studio. Все параллельные агенты работают только в своих task-ветках.

## Windows-native execution (mandatory)

- На HOME-PC сборка, тесты, инференс GPU и запуск Volleyball Studio выполняются **нативно в Windows 11**: PowerShell/CMD, Windows Rust target `x86_64-pc-windows-msvc`, Windows FFmpeg, ONNX Runtime/CUDA DLL и пути Windows. Не использовать WSL/Ubuntu, `wsl.exe` или Linux ELF для этих операций без нового прямого разрешения пользователя.
- Docker Desktop может пользоваться WSL2 как внутренним механизмом Docker, но **не** как средой запуска самой Studio. WSL разрешён для диагностики и обслуживания ранее созданного Linux-диска, если это не прерывает рабочую Studio и пользовательские задания.
- Штатный `start-volley-ui.cmd` вызывает Windows-native `scripts/start-volley-rust-dev.ps1`: собирает Windows `.exe`, запускает и останавливает только верифицированный PID с защищённым HTTP-завершением; финальное развёртывание выполняет `scripts/deploy-volley-queue.ps1`. Временные Linux/WSL-зависимости и старый `scripts/volley-rust.sh` не использовать как fallback. Если порт `8875` по-прежнему обслуживается прежней WSL Studio, не останавливать её, пока данные, CUDA/ORT и запуск новой Studio не проверены; сообщать о блокировке, не заявляя завершённый Windows-deploy.

## Start with the task

1. Run `git status --short --branch`, `git log -5 --oneline`, `git rev-parse HEAD`, `git worktree list --porcelain`; create a fresh local task branch and isolated worktree from current `origin/develop` before editing. Preserve existing changes/untracked files; commit only your paths. Recheck status before committing.
2. State result, acceptance check, owner and write paths briefly. Small tasks need no separate plan/specification/handoff. Persist a plan when dependencies, parallel work or continuation require it.
3. Select the relevant domain in `docs/RUST_CANONICAL_OWNERS.json`; inspect its implementation, callers, source-specific schemas and tests. Extend the owner, never copy an adjacent implementation.
4. Clarify ambiguities affecting results, compatibility, identity or acceptable quality. Choose routine implementation details yourself; continue independent work while awaiting answers.

## Spend context on evidence

- Find paths (`rg --files` or `git ls-files`), scope symbols with `rg -n`, read bounded excerpts. Batch independent reads. Narrow truncated queries; do not repeat full dumps.
- Exclude local data, weights, media, build outputs and `*.bak*`. Read `docs/RUST_MIGRATION.md` by topic or short tail, never in full. Open historical reports only to resolve specific uncertainty.
- Current evidence is code, schemas and fresh checks. Documents state contracts; historical diagnoses, stage counts, owner status and earlier PASS reports do not prove readiness.
- Return relevant matches, totals and failures. Keep large logs outside the repository; cite paths/excerpts. Reuse findings until files, HEAD or assumptions change.

Project-local skills use the canonical discovery root `.agents/skills/`. For a matching task, read `.agents/skills/<skill-name>/SKILL.md` before work and load only its relevant `references/` files. Follow `AGENTS.md` and the delivery policy alongside the skill.

Load only applicable references and sections:

| Task | Reference |
| --- | --- |
| Documentation navigation or missing route | `docs/README.md` |
| Architecture or migration boundary | `docs/RUST_ARCHITECTURE.md`, `docs/RUST_REWRITE_SCOPE.md` |
| Checks and their prerequisites | `docs/TESTING.md`; executable commands in `scripts/check-all.ps1` |
| UI / WASM | `docs/UI_LAYOUT_POLICY.md`; `rust/volley-web/README.md` for WASM builds |
| Analysis quality, performance, personalized montage | Relevant sections of `docs/AGENT_IMPROVEMENT_CHECKLIST.md`; `docs/quality/README.md` for evaluation |
| Launch, dependencies, applying changes | `rust/README.md`, `docs/RUST_DEPLOYMENT.md` |
| Actual concurrent work | `docs/AGENT_PARALLEL_WORKFLOW.md` and its task/run templates |
| Stage acceptance | `docs/RUST_ROADMAP.json`; generate `docs/RUST_ROADMAP.md`, never edit it manually |

## Preserve the architecture and data

- Only executable Studio: `rust/Cargo.toml`. No Python servers/orchestrators, compatibility subprocess engines, JavaScript business state or second Rust-owned implementation. DOM/media integration may use JavaScript/CSS; decisions belong to Rust/Rust-WASM. Retain FFmpeg, ORT/CUDA and weights. Python stays in the protected checkpoint; legacy formats are data adapters.
- Project → immutable GPU evidence → versioned CPU replay → confirmed player links → previews/statistics → original-video montage. CPU changes reuse saved evidence; run new inference only if required observations are absent, with an explicit new evidence version.
- Preserve source SHA, exact cache schemas, revisions, atomic writes, conflict checks and original/analysis time mapping. Version compatibility changes; never relabel old artifacts or overwrite accepted evidence.
- Provisional Track IDs and ReID similarity are not athlete identities. Keep observations, interpolation, algorithmic hypotheses and manual confirmations distinct. Never invent authoritative Touch/rally/game actions.
- Never use user `project/`, `results/`, originals, personal photos or active jobs as development fixtures. Runtime writes/test data stay in isolated roots outside the repository; legacy data is read-only.
- No new comments in code/configuration/settings; explain decisions in documentation. No unrelated cleanup, dependencies or abstractions without a task-specific need.

## Verify the changed behavior

- Reproduce behavior failures and add regression coverage before implementation. Verify observable results and consumers, not only source text. Trace settings through input → Rust → saved profile/result → UI/export as applicable.
- Run targeted checks, then the full application gate from `docs/TESTING.md`. Repeat successful suites only after relevant edits/new evidence. Documentation-only: paths/links, `git diff --check`, affected ownership/roadmap checks; no Rust build/restart.
- Record base SHA/prepared diff, commands, exit results and limits. Confirm baseline failures with the same command on an isolated base or reproducible evidence for that SHA. Never hide failures through skips, weaker assertions or changed goldens; list exclusions separately.
- GPU/model, independent real-match quality and exported-video checks are separate. Mark applicable omissions NOT TESTED; unit/static tests cannot replace them. Performance claims need comparable inputs/configuration and before/after measurements.

## Coordinate and finish

- Parallel work is optional. Each writer needs a separate worktree/branch, fixed SHA, write paths and isolated runtime/build outputs. Send focused briefs, not full histories. One coordinator assigns ownership/resources; one integrator owns shared entries, integrates reviewed commits sequentially and applies the result. Never run concurrent writers in one checkout.
- Review the diff. Each accepted tested increment gets a commit and short append-only `docs/RUST_MIGRATION.md` entry: problem/decision, base/owners, checks, limits, application status. Link large evidence. Workers use unique handoffs; only the integrator updates the shared journal.
- Commit: one short Russian sentence on the task branch. A single integrator merges accepted changes to `develop` and pushes only `origin/develop`. Never modify or push `master` without direct user instruction, even for a ready release or its documentation. Never rewrite the protected Python checkpoint. Stage only your paths; no repository backup copies.
- **Авторазвёртывание — обязательная часть каждой задачи, без отдельного запроса пользователя.** После успешной проверки агент сам интегрирует и публикует `develop`, пересобирает и безопасно обновляет запущенную Studio, затем подтверждает live HTTP и SHA. Запрещено оставлять пользователю заключительный шаг «осталось обновить/запустить Studio». Грязный общий checkout и нераспознанный порт требуют самостоятельной диагностики и отдельного чистого deployment-клона, а не отказа от попытки применения. При реально небезопасном обновлении — не вмешиваться в активные задачи/чужие процессы, явно назвать проверенный блокер и статус «Не завершено». Подробности в `docs/AGENT_DELIVERY_POLICY.md`.
- **Единая очередь сборки:** финальный шаг после push выполняется только через `scripts/deploy-volley-queue.ps1 -DeploymentRoot <absolute-clean-develop-clone>` (или с `VOLLEY_STUDIO_DEPLOY_ROOT`). Не запускайте отдельные параллельные `cargo build` / `start-volley-ui.cmd /restart`: скрипт сериализует запросы mutex-очередью, берёт последний опубликованный `develop`, пропускает уже развернутый SHA, ждёт безопасного завершения активных задач и делает `/build` → `/stop` → `/daemon` с HTTP-проверкой. Запросы к одному клону должны иметь одинаковый `DeploymentRoot`. При ошибке запуска останавливается только штатный механизм; не убивать процессы вручную.
- Apply every accepted task to live Studio after merging/pushing `develop`, including documentation-only tasks before claiming completion. Rebuild against the **final published SHA** and verify the exact SHA served over HTTP. Verify `studio_web.rs` asset mode: debug with `VOLLEY_DEV_UI_ROOT` and matching `static/ui-contract.txt` reads disk HTML/CSS/JS/SVG; refresh and verify served content. Bump the contract for dependent API/WASM ABI changes. Release/embedded assets and Rust/WASM require rebuilding.
- Launch: запуск и развёртывание Studio допускаются только из отдельного чистого клона на ветке `develop` при `HEAD == origin/develop`. После публикации вызывайте `scripts/deploy-volley-queue.ps1` с путём к этому клону; внутри одной очереди он проверяет владельца процесса, `shutdown_safe`, активные задания, собирает, переключает `/stop` → `/daemon` и сверяет live SHA. Не выполнять отдельные `/restart`, `cargo build`, повторные обходы диагностики и сборки вручную в штатном сценарии. При ошибке очереди сохранить работающий сервер, проверить конкретную причину, не убивать чужой PID 8875.
- After successful merge, push and deployment, **always remove your own completed local task branch and its clean worktree; the integrator must also remove its temporary branch/worktree**, verifying deletion before reporting completion. Begin a completed task response with «Выполнил» and link to the **verified live Studio built from exactly the final origin/develop commit**. If restart/build/HTTP validation is blocked, say «Не завершено» with blocker and SHA; do not pretend it is deployed. Report checks, limits and application status briefly. For continuation, keep goal, base/current SHA, paths, decisions, checks and next action in the existing task record; do not copy the conversation.
