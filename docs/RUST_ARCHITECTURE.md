> 2026-11-19: Global athlete photos and model-specific 256-dimensional OpenVINO
> reference descriptors persist under `players/native_reference_photos/<player_id>/`
> (`index.json`, `ref-*.jpg`, `reid-features.json`) independent of projects.
> Photo SHA-256 + model fingerprint invalidate stale descriptors. Native CPU
> postprocessing calculates project Track ID comparisons, saves suggestions and
> may apply thresholded automatic matches; the Players tab does not trigger
> inference on opening. New photos are considered only after the explicit
> refresh action; manual refresh does not create athlete assignments.
> Stage 61: the global athlete album uses only isolated Rust writes in
> players/native_reference_photos/<player_id>/index.json with bounded JPEG
> blobs, manually reviewed source replay metadata, CAS revision, per-athlete
> lock, and explicit undo. The enrollment API refuses unlinked or excluded
> Track IDs, changed SHA-replays and removed project athletes. No photo
> reference triggers an automatic player assignment.
> Stage 60: volley-inference/src/reid_openvino.rs loads bundled 0287 XML/BIN
> through OpenVINO native C ABI. FFmpeg crop BGR24 -> f32 CHW 1x3x256x128 ->
> 256 float32 L2-normalized visual features. Both weights and preprocessing
> are SHA-256 fingerprinted. Native HTTP adapter binds the extraction to
> exact verified replay SHA and remains read-only, without automatic identity.
> Statistics/montage identity contract (2026-11-09):
> rust/volley-core/src/identity_review.rs::resolve_reviewed is the single
> source of confirmed Track ID owners. Saved ReID groups are **suggestions**
> only; group membership never grants an athlete identity or statistic.
> Both native and historical JSON readers use the same resolver, with
> source-specific storage adapters and unchanged immutable evidence.
> Manual Track ID mutation policy owner: rust/volley-core/src/identity_review.rs.
> Native and historical-cache HTTP handlers are only evidence-bound JSON adapters.
> No separate rules for assign, unassign, exclude, restore are permitted.
> 2026-11-09: This branch has a Rust-only executable and no Python application source. The old code is in the frozen Git checkpoint; legacy on-disk evidence remains a read-only compatibility format. Product completion is not implied.

# Rust migration — canonical architecture and anti-duplication contract

**Status (2026-11-27):** binding architecture for develop. The Rust application runtime is established; complete product parity and release acceptance remain open. The old migration branch is historical. Keep Rust/Rust-WASM as the domain owners, preserve necessary browser adapters, and port remaining compute only after profiling demonstrates a worthwhile gain. Preserve master, the pre-Rust checkpoint and all user media.

## Root cause / what went wrong

Stages 14–21 ported algorithms and readers of legacy Python-compatible evidence (players.json, events.json, game-tracks.json and cached NPZ ReID vectors). Stages 38–51 added an independent native pipeline (GPU detections.jsonl → fused frames → tracked_frames.jsonl → touch_candidates.jsonl → rally_candidates.jsonl). New legacy/native APIs for identity, gold marks, rallies and previews were developed without a named common domain owner. Stage counts were incorrectly interpreted as feature completeness. This is a process failure, not a reason to erase functioning adapters.

**Never remove an existing legacy source reader merely because a native counterpart exists.** Preserve user files and on-disk schemas. First put shared policy and storage behind one owner, then replace downstream calls, then retire an adapter only with actual compatibility tests.

## Canonical application pipeline

1. Original recording → SHA-256 bytes, native stream metadata and original-relative time → a single working timeline at 30 FPS. New neural decoding reads the original directly and normalizes to 30 FPS in FFmpeg, without an intermediate lossy encode. Prepared previews use content-bound paths `prepared/<schema2-metadata-id>-<original-sha>.mp4`; schema-2 caches are retained and source adapters resolve historical media by the saved prepared stamp.
2. Immutable GPU evidence: RF-DETR Medium detects player boxes on a 640×360 frame (then ONNX 576×576); VBallNet receives grayscale 512×288; RTMPose-M COCO17 uses player boxes to crop **original-resolution** RGB pixels, never RF-DETR's reduced image. On Unix and Windows, the shared 30-FPS primary path can branch a single FFmpeg decode into all three outputs, with bounded lookahead for interpolation. Unix uses local sockets; Windows uses loopback TCP and enables this path only after a successful NVDEC preflight, retaining independent decoding for unsupported sources. For compatible progressive 8-bit H.264/HEVC sources at 30 FPS without rotation or unsupported color range, the Windows/Linux NVDEC path keeps decoded NV12 frames in CUDA VRAM, performs bilinear/area-aware scaling, RF-DETR normalization and original-resolution RTMPose crops using CUDA kernels, then binds CUDA input tensors directly to ONNX Runtime. Only compact model outputs/coordinates return to CPU. Color matrix is selected from FFprobe metadata, with explicit fallback on unknown transfer modes. NVRTC/CUDA preflight or VOLLEY_ZERO_COPY_PRIMARY=0 selects the established shared FFmpeg RGB24 path; unsupported FPS/codecs continue through that path. For unusually large source frames or an explicitly selected standalone analysis mode, the original-resolution RTMPose fallback remains separate. Original-source fingerprints and raw/pose evidence SHAs are preserved; new fusion evidence only combines RF-DETR players and VBallNet ball observations.
3. Versioned CPU interpretation: provisional Track ID/body interpolation and ball trajectory/interpolation → contact candidates → rally discovery/refinement → final rally-scoped attribution and action hypotheses. Interpolation settings do not rerun neural inference. Synthetic positions and automatic hypotheses retain their provenance.
4. Optional court/net geometry feeds contact interpretation. There is no default markup. Manual and automatic markup share the same applied record; the latest successful CAS save wins. The observed net is an image-space constraint, not a ground-plane homography. A net-only markup remains usable without invented court corners.
5. ReID compares against known athletes after final physical events. Manual assignments/exclusions take precedence. Automatic identities and action hypotheses remain distinct from reviewed facts.
6. `native_rally_preview` publishes content-revisioned `volley-montage-material/1` snapshots under the selected rally run. The material binds final contact attribution, athlete decisions, manual action revision, geometry revision, immutable evidence SHAs, original proof, explicit readiness and original-source clip intervals. Timeline, statistics and athlete montage use the same final contact policy.
7. Native manual/highlight/report/Quick Reel rendering consumes verified material bindings. The renderer checks source and material revision, selected intervals and original SHA before accepting a plan, verifies originals again while rendering, and records material references in its receipt. One renderer reads original media; an absent identity/action type does not block explicit manual montage. Quick Reel keeps independent runtime data and fresh inference.

`material_schema` is additive to the existing timeline response. `montage_ready=false` in old provisional rally evidence is not rewritten: technical montage readiness is recorded in the derived material, separately from authoritative game-event confirmation. Old evidence without an original content proof remains explicitly unverified; a current file hash is never retroactively declared the hash of its historical analysis input.

Every input must declare source_kind (legacy_cpu_cache OR native_gpu_cpu_replay), schema_version, project/video/run, source SHA/fingerprint and frame-rate/coordinate convention. Legacy run_id and native replay_id are not exchangeable; numeric Track IDs are not global. Every result is either immutable evidence or a revisioned human decision, never both.

## Historical ownership inventory and current canonical ledger

Machine-readable ledger: docs/RUST_CANONICAL_OWNERS.json; mandatory audit: scripts/ci/check_rust_ownership.mjs. The **complete numbered roadmap**, including the last recorded stage X / planned total Y, next stage and explicit future acceptance criteria, is docs/RUST_ROADMAP.md (generated from docs/RUST_ROADMAP.json). Its consistency gate is scripts/ci/check_rust_roadmap.mjs --check --self-test. The current 85-stage denominator is a planning estimate, not an indication of finished-feature percentage.

| Domain | Canonical implementation | Adapters; preserve until migrated | Action |
|---|---|---|---|
| Source provenance/Track ID scoping | rust/volley-core/src/evidence_source.rs | player_identity.rs legacy and native_player_tracks.rs native | Source-kind discriminator, legacy unknown FPS/SHA, native full SHA/30 FPS and scope-qualified Track IDs; not a binding algorithm |
| ReID similarity/grouping | rust/volley-core/src/reid.rs | reid_native.rs, reid_npz.rs, reid_group_edits.rs | REUSE existing cosine and clustering; add native embeddings + photo enrollment, never another similarity algorithm |
| Player identity | rust/volley-core/src/identity_review.rs | player_identity.rs legacy; native_player_tracks.rs native | Consolidate assign/unassign/exclude/revise policy; do not mix Track IDs across source versions |
| Touch evidence | rust/volley-core/src/touch.rs, motion.rs | touch_preview.rs legacy; cpu_touches.rs native | Unify policy, retain source parsing |
| Rally policy | rust/volley-core/src/rally_reconstruction.rs native; legacy compatibility via rallies.rs | cpu_rallies.rs, native_rally_preview.rs | Align output contracts, retain provisional/confirmed distinctions |
| Gold bookmarks | common revisioned decision/storage policy | rally_marks.rs legacy; native_rally_marks.rs native | Keep historical identifiers and JSON formats, share storage/CAS |
| Montage | rust/volley-app/src/montage.rs | player_montage.rs, native_rally_preview.rs, rallies.rs | Exactly one renderer, format-aware selection; use original media |
| Shared sidecar I/O | rust/volley-app/src/versioned_store.rs | existing identity and mark adapters | NO independent revision/SHA/CAS helpers |
| GPU inference | rust/volley-inference/ | inference_full.rs, inference_fusion.rs | Immutable detector evidence |
| Rust/WASM UI state, RU/EN and preferences | rust/volley-web/src/lib.rs, compiled volley_web.wasm | rust/volley-app/src/studio_web.rs and static/studio_ui.js plus feature-specific DOM/HTTP controllers | Rust validates navigation, load epochs, settings and text IDs; JS stays only for DOM/media/file and same-origin transport, no server business policies |
| Quick Reel | rust/volley-app/src/quick_reel.rs | preparation, inference, montage services | Fresh independent inference, no project analysis reuse |
| Rust process / deployment | rust/volley-app/src/runtime.rs | scripts/volley-rust.sh, scripts/start-volley-rust-dev.ps1, start-volley-ui.cmd, Dockerfile | Canonical-checkout single-instance lock, report busy work, refuse stop during native jobs, strict loopback 8875, nonroot Docker runtime and external ORT provider |

**Reality:** Legacy gold/rally/identity adapters are distinct file schemas and currently have consumers. Removing their routes today would lose legacy project features. Shared sidecar primitives live in versioned_store.rs; the table and consolidation sequence below describe the historical migration inventory, not a fresh audit of remaining work. Verify current ownership in the machine-readable ledger and code. Renaming modules is NOT deduplication.

## Mandatory workflow for EVERY new stage / every AI handoff

**0. INVENTORY FIRST.** Read this file, docs/RUST_ROADMAP.md (current X / total Y and numbered remaining stages), docs/RUST_REWRITE_SCOPE.md, last 5 sections of docs/RUST_MIGRATION.md and inspect git log --all for the feature. Search all Rust modules, routes, event structs and stores for related functionality. Run both the ownership audit and roadmap consistency gate. Classify the request as EXISTS / LEGACY-ADAPTER-ONLY / PARTIAL-NATIVE / MISSING. If it already exists: extend its canonical owner or create only a typed source adapter. Do not implement a parallel module.

**1. CONTRACT BEFORE CODE.** Identify owner ledger row; input source kind, schema, version and time coordinates; immutable input and mutable sidecar path; exactly which stages are provisional; consumer list, old-project compatibility and tested migrations. Any new endpoint or schema must update the machine ledger in the SAME commit. If functionality crosses two stores, define the adapter boundary explicitly.

**2. IMPLEMENT VERTICALLY.** Extend pure algorithms in volley-core, infrastructure and APIs in volley-app, UI/WASM in volley-web. Centralize validation, no-symlink bounded I/O, atomic persistence, CAS and project locks. Never silently convert provisional Track IDs/Touch/rallies into verified athlete actions. Never copy or mutate Python user data. No second state store without a documented versioned migration plan.

**3. VERIFY.** Run cargo fmt --all -- --check, cargo test --workspace --offline, cargo check --workspace --offline -q, node scripts/ci/check_rust_ownership.mjs, and targeted Rust/WASM + JavaScript checks for UI modifications. For inference changes run real CUDA smoke; for storage changes use an isolated real-MP4 integration fixture. Check source bytes and invariant statuses, not just HTTP 200. Final full match/benchmark comparison is only after whole rewrite per owner decision.

**4. COMMIT AND PUBLISH EVERY COHERENT STAGE.** Commit code + test + ownership ledger + migration diary + updated 1-to-N roadmap together; a recorded stage must show its number out of the planned total (e.g. 53 / 85), the next stage and the remaining tasks. Run node scripts/ci/check_rust_roadmap.mjs --check --self-test. Record SHA and active branch. Verify published Git tree exactly matches local tree before advancing the migration ref. Never alter master, production port 8765 or original user video/cache. Do not let later agents repeat work only because the diary says PARTIAL.

**5. MARK COMPLETENESS HONESTLY.** COMPLETE means canonical service works on new native evidence, preserves previously saved projects with a tested adapter, connects all consumers and UI, and passes focused tests. Otherwise PARTIAL. Handoff must name tested commit SHA, affected paths, open adapters, and next step.

## Automated safety gate

- Checker rejects duplicate API route literals, missing canonical owners or adapters, conflicting canonical persistence paths, and undocumented additions to guarded player/rally/montage API areas.
- Every feature has exactly one owner; new routes or modules require explicit ledger updates and source-format justification.
- Avoid new *-native2.rs, *-v2.rs or *-rewrite.rs escape hatches; extend owner and add an adapter instead.
- Do not create another ReID grouping engine while the missing work is native vector extraction; do not invent another montage exporter or parallel gold-state manager.
- In review, test both legacy adapter and new native route before deleting a duplicate implementation.

## Historical consolidation sequence (not a current backlog)

1. Factor out duplicated SHA-256 revision + bounded no-symlink reads/atomic writes into versioned_store.rs, shared by legacy and native adapters; preserve JSON content exactly.
2. Introduce EvidenceSource = LegacyCpuRun | NativeCpuReplay. Extract the common identity decision semantics from player_identity.rs and native_player_tracks.rs, leaving thin readers/writers.
3. Feed native track/photo embeddings into existing ReID evaluator and manually reviewed owner policy. Implement actual native encoder and photo reference enrollment; never infer identity from query_index.
4. Give provisional native and legacy annotated rally candidates a common typed selection and bookmark interface without converting IDs or marking provisional contacts authoritative.
5. Integrate athlete stats, Quick Reel, end-to-end UI and real-match validation. Switch production only after full parity comparison with immutable Python checkpoint.

**Progress must be measured against missing behaviors and number of duplicate authorities, never number of commits or stages.** For performance work, measure a representative workload before porting and compare equivalent outputs and latency afterwards. The 2026-11-27 preview profiling in RUST_MIGRATION.md did not justify another Rust port for that workload; reuse and browser-native optimizations remain valid.

### Stage 81: compiled Rust/WASM UI, not a JavaScript business-policy fork

The single Rust source rust/volley-web/src/lib.rs is the canonical owner for navigation among 6 tabs, validated locale/theme/density/reduced-motion settings, generation tokens for cancel-safe asynchronous data loading and translation IDs. rust/volley-web/src/studio_i18n.tsv is compiled into volley_web.wasm using include_str!, and the same source hosts the existing stable montage selection engine. The browser adapter rust/volley-app/static/studio_ui.js validates the ABI, loads the actual WASM from the same-origin Rust Axum server, stores only visual preferences in versioned browser localStorage, renders Rust text and updates localized DOM incrementally. All 163 static UI labels and 20 aria-label/title/placeholder values with Russian text are covered by the embedded RU/EN source. Existing native browser feature controllers (analysis, reviewed highlights, players, Quick Reel, legacy archives) continue to use unchanged same-origin native Rust endpoints and must not re-implement evidence or GPU/CPU business logic. Desktop/phone navigation remains accessible.

The GUI-only settings have **zero effect** on analysis profiles or immutable inference artifacts. The separate GPU launch panel exposes explicit per-run stride 1..9, VRAM cap 0..131072 MB (0 means unlimited) and CPU thread count 1..128. Golden request defaults remain stride=9, VRAM cap=0, CPU=8, device=cuda; JS validates before requests and native Rust revalidates. No GPU or CPU run is started on locale/theme changes.

**Evidence and limitations:** actual target wasm32-unknown-unknown release binary is built and executed by Node WebAssembly.instantiate with bounds/translation tests; 6-tab browser DOM simulation covers navigation, deep links, async data loading and stale-response rejection, language settings persistence, resource flags and explicit GPU request flags. Rust Axum HTTP tests verify static asset wiring. Chrome GUI automation and dynamic API error-message localization are not fully verified at this stage; a successful CUDA run and full Python-versus-Rust feature/quality benchmark are NOT implied. Existing JS is a necessary browser DOM/network adapter and is not a second source of truth for domain policies.
### Stage83: SHA cache, request security and shutdown gate

Shared schema-2 media keys remain compatible with Python, but a Rust
 cache hit additionally requires verified SHA-256 output integrity.
 General JSON readers/writers now reject symlinks, oversize documents
 and unsafe atomic-write destinations. The Axum request barrier denies
 foreign Host/Origin/Fetch-Site headers, and refuses SIGTERM when
 HTTP or native workers remain active; otherwise it atomically drains
 requests before graceful shutdown. This does not mark CUDA/VRAM or
 the Docker container as accepted.
