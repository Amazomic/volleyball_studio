# Volleyball Studio — deployment (Rust beta and source builds)

**Current release:** [v0.1.0-beta.1](releases/v0.1.0-beta.1.md) on `master`, built from Rust source commit `437b446` (release-doc commit `ddb2a82`). **Pre-release, not production-ready**: `GET /internal/rewrite-status` reports `production_ready=false`. Ongoing development stays in `develop`; the published release tag is immutable.

## Installed beta archives (no compiler required)

Download both formats from [GitHub Releases](https://github.com/Amazomic/volleyball_studio/releases/tag/v0.1.0-beta.1). Verify them against the attached `SHA256SUMS.txt`.

- **Windows x64:** extract `Volleyball-Studio-v0.1.0-beta.1-windows-x64.zip`, open the extracted `Volleyball-Studio-v0.1.0-beta.1-windows-x64` folder and start `run-studio.cmd` (or run `volley-app.exe` with suitable environment variables). This is a **native Windows executable**, not a WSL launch; Rust and WSL are unnecessary for starting the prebuilt application. The default writable data root is `%LOCALAPPDATA%\VolleyballStudioBeta\data`.
- **Linux x86-64 / WSL2:** extract `Volleyball-Studio-v0.1.0-beta.1-linux-x86_64.tar.gz`, enter the extracted folder and run `bash run-studio.sh`. It is dynamically linked, built with **glibc 2.41** on Ubuntu 25.04 and may not run on older Linux releases. By default, the data root is `${XDG_DATA_HOME:-$HOME/.local/share}/volleyball-studio-beta`.
- **Both:** open `http://127.0.0.1:8875/studio` in a browser. The `/studio/rust` compatibility URL redirects there. Do not expose the unauthenticated API outside localhost; a second instance cannot bind the same port. The beta archives include the Rust server, embedded HTML/JS/WASM, launch scripts, release notes and MIT license, **not** FFmpeg/FFprobe, model weights, ONNX Runtime, CUDA/cuDNN, OpenVINO or original video.
- **External inference requirements:** supply compatible models and native runtimes separately; the UI starting successfully is **not** proof of successful GPU inference. An extracted beta archive is a standalone application root, not the full source repository/model distribution. For model-based work, provision the expected model locations and dependencies in the selected runtime environment. Do not open the only copy of production projects or overwrite existing results.
- **Verified scope:** the published Windows/Linux archives were extracted and their servers started on isolated loopback ports, with `/studio` and embedded WASM returning HTTP 200. The full Rust/Node local gate passed, but real-match accuracy, complete GPU inference and MP4 export were **NOT TESTED** as release acceptance. `production_ready=false` remains correct.

## Обязательная доставка агентских задач

Каждая задача проходит [контракт веток и развёртывания](AGENT_DELIVERY_POLICY.md): отдельные локальные task-ветка и worktree → проверки → последовательная интеграция с актуальным `develop` → push только `origin/develop` → сборка на точном опубликованном SHA → проверенный безопасный restart → HTTP 200, корректный PID и совпадение SHA на странице. Даже изменения только документации не считаются «Выполнено» до подтверждения живой сборки на новом коммите.

`master` не изменять без прямого указания пользователя. Если активен другой агент, не использовать его дерево. Если `/internal/rewrite-status` сообщает активные задания, запуск небезопасен или скомпилированный SHA отличается — не прерывать Studio и сообщить «Не завершено» вместо заявления о готовом результате. Только после успешной проверки дать ссылку [на локальную Studio](http://127.0.0.1:8875/studio), указав commit SHA. Ссылка предназначена для компьютера с запущенной Studio.

## Build from source on Windows / WSL (developer workflow)

The following `start-volley-ui.cmd` commands apply to a **repository checkout**, not a downloaded beta ZIP. Starting, detaching or restarting the source Studio is allowed only from an independent clean Git clone with checked-out `develop` at the exact `origin/develop` SHA. The Rust native entrypoint also enforces develop, origin/develop and clean tracked files for source checkouts, preventing direct-binary bypass. Standalone release archives without Git metadata remain supported. Feature branches, `master`, detached HEAD and outdated develop are rejected before a build or shutdown; `status`, `stop`, `doctor` and `check` remain available for diagnosis and safe recovery. A separate clone avoids disturbing an existing dirty `develop` worktree. The source launcher compiles as needed and uses the isolated WSL sandbox; its defaults differ from the prebuilt Windows launcher.

The Rust launcher never starts/stops Python or changes original videos,
historical results, or the global player cache.

## Windows and WSL — one working instance

Prerequisites: Ubuntu-25.04 in WSL, Rust offline toolchain, FFmpeg and
FFprobe, Node.js only for CI, local NVIDIA GPU and explicitly provisioned
Linux ONNX Runtime for CUDA inference.

Use from the checkout root:

~~~bat
start-volley-ui.cmd
start-volley-ui.cmd /status
start-volley-ui.cmd /stop
start-volley-ui.cmd /restart
start-volley-ui.cmd /doctor
start-volley-ui.cmd /check
~~~

The source-built app stays in the foreground at http://127.0.0.1:8875/studio. The legacy /studio/rust path redirects here.
Legacy Python port 8765 is never automatically touched or killed.
A second launch reuses the running Rust process, and an exclusive file
lock based on the **canonical repository root** prevents a second
instance on another port or with a different Rust data directory.

PowerShell supports optional settings and the same safe actions:

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-volley-rust-dev.ps1 -Action Start
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-volley-rust-dev.ps1 -Action Status
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-volley-rust-dev.ps1 -Action Restart
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-volley-rust-dev.ps1 -Action Doctor
~~~

Optional parameters: -Bind 127.0.0.1:8875, -Distro Ubuntu-25.04,
-DataRoot /absolute/wsl/path, -ReadOnly, -OrtLibrary /absolute/wsl/path/
libonnxruntime.so and -OpenvinoLibrary /absolute/wsl/path/libopenvino_c.so.
Writable Rust data defaults to $HOME/.cache/volley-rust-sandbox outside
the repository. Original models, videos, Python runs and source code
are not modified.

Before sending SIGTERM, the WSL controller verifies the returned PID,
the expected Rust executable path and checkout working directory using
/proc, and GET /internal/rewrite-status must report shutdown_safe=true
and active_jobs=[]. GPU inference, FFmpeg montage, CPU processing, video
preparation, Quick Reel jobs and in-flight HTTP response bodies block restart.
In particular, an open Range video/audio stream remains active until its
response body finishes or the client disconnects. While another response
is open, status reports shutdown_safe=false and active_jobs includes
http_responses. On Linux/WSL, each accepted TCP socket is configured
with TCP_USER_TIMEOUT=120 seconds: if buffered bytes cannot be transmitted
because the client has a zero receive window, the kernel closes the stalled
connection and releases the HTTP shutdown guard. Active progressing video
streams are not subject to a 120-second total-duration limit. An older
running binary must be safely restarted to gain this protection.
Close active browser playback and retry the guarded restart;
do not escalate from a refusal to SIGKILL. SIGINT and SIGTERM both use native
Tokio graceful shutdown; no force kill, Python process kill,
or generic port-based termination is present.

## Native offline Linux source build

~~~bash
cd rust
cargo fmt --all -- --check
cargo test --workspace --offline --locked
cargo check --workspace --offline --locked
cargo build --release --offline --locked -p volley-app
~~~

Output: rust/target/release/volley-app (Linux ELF x86-64, NOT a
Windows executable). The Rust backend embeds its HTML/CSS/JS and the
compiled Rust WebAssembly UI, without a Python application dependency.
FFmpeg/FFprobe and a compatible Linux ONNX Runtime/CUDA provider remain
external native dependencies. A Windows ONNX DLL cannot substitute for
a WSL Linux shared library.

## Docker image and local build

The SAME Dockerfile has a default source-built final stage and an opt-in
prebuilt target. Both share a Debian Trixie (glibc 2.41) FFmpeg runtime, a nonroot
volley user, SIGTERM shutdown and localhost-only port 8875.

~~~bash
docker build -f Dockerfile -t volley-rust:source .
bash scripts/ci/build_rust_docker_local.sh
~~~

The second command compiles the release binary offline and creates a
temporary context containing ONLY the compiled executable plus models
and OV assets. This avoids downloading the large Rust builder image.
The resulting image is called volley-rust-stage82:local.

Run only on a TRUSTED local Linux/WSL host with Docker host networking:

~~~bash
docker run --rm --network host --cpus=8 --memory=12g \
  -v volley-rust-data:/data volley-rust-stage82:local
~~~

For native CUDA inference, provide trusted, compatible Linux ONNX
Runtime + CUDA/cuDNN provider shared libraries and NVIDIA container
support. Example requiring actual host libraries:

~~~bash
docker run --rm --network host --gpus all --cpus=8 --memory=12g \
  -v volley-rust-data:/data \
  -v /host/onnxruntime/lib:/opt/ort:ro \
  -e ORT_DYLIB_PATH=/opt/ort/libonnxruntime.so \
  -e LD_LIBRARY_PATH=/opt/ort \
  volley-rust-stage82:local
~~~

Host networking must be enabled on Docker Desktop. Docker bridge
port publishing cannot access a deliberately loopback-only service.
NEVER expose the unauthenticated local-file API publicly via
0.0.0.0 or public port forwarding.

The Docker context excludes original videos, project/ and results/,
model inference caches, local Rust builds and temporary files. The
runtime does not use Python services. Docker images must NOT be
considered tested until an actual image build and running-container
smoke test succeed.
The stage82 local build initially used Debian Bookworm, whose glibc
was incompatible with this release binary; the corrected runtime is
Debian Trixie. Any old cached Bookworm-based image must be rebuilt. Large initial base images and Debian FFmpeg
dependencies may require an accessible package registry.

## Resources and acceptance

Runtime doctor inspects CPU count, available RAM, FFmpeg, model files,
NVIDIA GPU/VRAM, Docker and optional Linux ORT/CUDA provider links.
When required CUDA libraries are absent it fails explicitly with
exit code 3; **there is no implicit CUDA-to-CPU fallback**.

Verified on connected WSL/Windows in stage82: native Linux release
compilation, real Windows-to-WSL localhost HTTP access, Windows
PowerShell and CMD status/check/stop commands, an isolated
start→same-port reuse→second-port rejection→graceful stop→restart
regression, canonical-lock ownership and an active CPU job causing
shutdown_safe=false. The legacy user Python service is never touched.

~~~bash
bash scripts/ci/test_rust_lifecycle.sh
bash scripts/ci/test_rust_release_smoke.sh
node scripts/ci/test_rust_packaging.mjs
~~~

Actual CUDA completed inference requires provisioning native ONNX
Runtime and passing the optional real-model smoke. Full match GPU,
memory, cancellation, security and Python-vs-Rust parity remain stage
83/84 acceptance; stage85 decides on production or rollback.

**Linux ABI gate:** the WSL Ubuntu-25.04 offline prebuilt ELF requires
 GLIBC_2.39. Debian Bookworm (glibc 2.36) is not suitable for that
 executable even if Docker build succeeds. The image now uses Debian
 Trixie (glibc 2.41) to support both the WSL prebuilt binary and
 the source-built binary. At packaging acceptance, validate the
 actual container entrypoint via a running-container HTTP smoke,
 not just docker build.

After the image is available, verify the REAL container:

~~~bash
bash scripts/ci/test_rust_docker_smoke.sh volley-rust-stage82:local
~~~
The test only starts its own disposable non-root container with
 a read-only root filesystem, tmpfs-backed isolated data and
 a local test port. It verifies HTTP and compiled WASM before
 stopping only its own container. Host-networking support on Docker
 Desktop must be enabled.

## Historical stage83 security and fault-injection acceptance

An Axum lifecycle barrier closes the status-to-SIGTERM race: after
 confirming zero in-flight handlers, unfinished response bodies and active
 native tasks, incoming requests are denied. The guard owns the response
 body rather than only the handler future; incomplete streaming downloads
 and Range responses cannot cause a false-safe shutdown. SIGTERM during
 HTTP/GPU/CPU/FFmpeg work is refused, never force-killed. The global request gate
 rejects foreign Host, Origin and Sec-Fetch-Site headers on all routes.

Prepared MP4 cache reuse now requires a SHA-256 and byte-size manifest
 receipt. Old unsigned Python/Rust entries are rebuilt, not trusted.
 Rehashing uses a 1-MiB streaming buffer; quantify repeated hashing
 cost during stage84. JSON sidecars are limited to 32 MiB and reject
 symlinked path components; new temp files are fsynced before rename.

Run scripts/ci/test_rust_faults.sh for all offline Rust/code fault
 tests. Add --require-gpu-docker for a strict gate: it requires a
 real CUDA model integration and running Docker container. Without
 Linux ONNX Runtime or a verified Docker image it exits nonzero.
 Representative match quality and peak VRAM profiling remain stage84.

## Historical stage83 integrity and release gates

Run bash scripts/ci/test_rust_faults.sh for complete offline fault-injection
and lifecycle regression. Run bash scripts/ci/test_rust_faults.sh
--require-gpu-docker before considering GPU/Docker deployment complete;
it exits nonzero unless genuine CUDA ONNX inference and a running Docker
image have been verified. Windows Git users must keep Bash scripts LF-only;
.gitattributes enforces this for every *.sh file.
## Cross-platform Quick Reel file chooser

Quick Reel source selection on the Rust Studio web page uses a **browser-owned**
HTML file input for both mouse and keyboard activation. It opens the file
chooser on the **browser's device**, whether the Rust server runs on Windows,
WSL or native Linux. It MUST NOT use the server-side
POST /api/quick-reel/native/pick PowerShell dialog as the default UI.

Browser sandbox security never reveals the absolute local path. Therefore a
browser-selected file or dropped file is streamed to isolated Rust temporary
storage via POST /api/quick-reel/native/upload (currently at most 2 GiB per
video and 30 videos per run). A direct path reference without copying is
NOT portable and cannot be promised through browser file input. The original
files remain untouched. Re-selecting a file after cancellation or upload
failure works because the input is reset before opening the chooser.

Regression tests: scripts/ci/test_native_quick_reel_ui.mjs verifies the
browser file input receives direct user gestures, retains selections after
cancel, supports keyboard, repeated selection and drag-drop, and never makes
the Windows-only server dialog request. The Rust quick-reel HTTP tests check
multipart upload safety independently.
## Guarded restart after rebuilding a running Rust executable

On Linux/WSL Cargo can replace the on-disk Rust binary while the previous
instance continues running. /proc/PID/exe then ends in (deleted). The
launcher accepts this exact old executable ONLY if the HTTP status names
the Rust Studio, the reported PID exists, /proc/PID/cwd equals this
repository's Rust workspace, and argv[0] equals the expected debug or
release executable. It still uses SIGTERM, never SIGKILL, and refuses
to stop when a native job is active. This rule is exercised by a real
rebuild-during-server restart test and checked by the packaging guard.
## Local ONNX Runtime 1.22 + CUDA 12 / cuDNN 9 (Rust GPU)

The Rust ort crate 2.0.0-rc.10 dynamically loads ONNX Runtime **1.22**.
The NVIDIA WSL driver does not include the native Linux ORT .so, CUDA
userspace math libraries, or cuDNN. These must be provisioned separately.
For the initial installation under the regular WSL user:

    bash scripts/setup_rust_ort_cuda.sh install
    bash scripts/setup_rust_ort_cuda.sh check
    bash scripts/ci/test_rust_cuda_runtime.sh

The script obtains Microsoft's official ONNX Runtime Linux x64 GPU 1.22
archive and the CUDA 12/cuDNN 9 support wheels in an isolated directory
under ~/.local/share/volley-rust/ort-1.22.0. Python pip is used only at
**installation time** to fetch NVIDIA libraries; the running Studio,
both inference networks and post-processing are native Rust/CUDA, with
no Python background worker or silent CPU fallback.

The Windows/WSL launcher in scripts/volley-rust.sh automatically sets
ORT_DYLIB_PATH and LD_LIBRARY_PATH for that bundle at startup.
Always gracefully restart the Rust server after installation, since
changing a parent shell environment does not affect a running process.
Do not restart while native GPU/CPU/montage jobs are active.
The explicit smoke script executes both ONNX models on **actual GPU**;
only its successful completion proves that the CUDA provider works.

If libraries are unavailable, native Rust Engine::load returns an
actionable error, rather than panicking a Tokio worker on
libonnxruntime.so. A successful ldd/doctor check is not sufficient
evidence of GPU inference until the real model smoke passes.
### Verified on HOME-PC WSL (post-stage84 GPU runtime hotfix)

Official ONNX Runtime 1.22.0 Linux GPU archive SHA-256:
2a19dbfa403672ec27378c3d40a68f793ac7a6327712cd0e8240a86be2b10c55.
CUDA 12.9 wheel versions and cuDNN 9.28.1.4 are explicitly pinned
in the setup script. Windows NVIDIA driver 616.92; GPU detected in WSL:
GeForce RTX 5070 Ti Laptop, 12227 MiB.

Actual diagnostics: ldd CUDA provider resolved all shared objects;
Rust GPU model test passed for VBallNet V4c and the former player detector using
CUDAExecutionProvider (not Python/CPU fallback). The actual running
Rust web server at 127.0.0.1:8875 opened both CUDA ONNX model
sessions through POST /api/inference/native/preflight with
probe_runtime=true. The optional 18-frame generated-video Rust
integration test passed twice GPU inference -> SHA-verified model
results -> fusion -> provisional CPU tracking -> Touch candidates ->
rally proposals. It is synthetic functional validation, NOT human
ground truth or measured real-match parity.

The original stage84 NO-GO verdict remains valid until independent
same-source Python/Rust match quality and Docker acceptance are done.
This hotfix fixes runtime availability, not every possible reel
selection or lack of qualifying game rallies.