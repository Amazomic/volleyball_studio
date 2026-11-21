# Volleyball Studio

**AI-assisted volleyball match analysis and video highlights.**

**English** | [Русский](README.ru.md)

Documentation: [Index](docs/README.md).

**Branches:** `develop` is the main development branch and the target for day-to-day changes and PRs. `master` is reserved for tested version releases only. See the [development and release workflow](docs/DEVELOPMENT_WORKFLOW.md). [Beta v0.1.0-beta.1](https://github.com/Amazomic/volleyball_studio/releases/tag/v0.1.0-beta.1).

План разработки: 84/85; следующий 85/85

---

## About

**Volleyball Studio** is a local application for analyzing recorded volleyball matches. It helps detect the ball and players, reconstruct trajectories and rallies, examine touches, associate tracked identities with known athletes, and assemble video highlights.

## Features

- **Projects and video:** import recordings, select the analysis interval, annotate court geometry, and prepare footage.
- **Primary GPU analysis:** original-video analysis using VBallNet V4c for the ball, RF-DETR Medium for players, and RTMPose for body keypoints.
- **CPU post-processing:** tracking, court-aware geometry, ball trajectories, touch candidates and types, and rally segmentation.
- **Players:** Track IDs, ReID similarity suggestions, manually confirmed player assignments, and a player photo album.
- **Preview and statistics:** rally playback, trajectory/player/keypoint overlays, filters, and human review.
- **Editing:** favorite rallies, video highlights, and a separate Quick Reel workflow.
- **Analysis versions:** saved GPU evidence with independently configurable post-processing variants.

Automated player identification and touch classification are **not guaranteed to be accurate**. ReID results are suggestions, and uncertain events should be reviewed by a human.

## Architecture

| Component | Responsibility |
| --- | --- |
| `rust/volley-app` | Local server, API, jobs, and studio UI |
| `rust/volley-inference` | Models and native inference runtimes |
| `rust/volley-core` | Tracking, ReID, touches, rallies, and CPU logic |
| `rust/volley-media` | Video preparation and FFmpeg-based editing |
| `rust/volley-web` | WebAssembly UI components |

The active application is developed in **Rust**. The original Python implementation remains available in Git history as the migration baseline. ONNX Runtime, OpenVINO, CUDA, and FFmpeg are external dependencies.

## Download beta v0.1.0-beta.1 (pre-release)

Download the tested [Windows x64 ZIP or Linux x86-64 tar.gz](https://github.com/Amazomic/volleyball_studio/releases/tag/v0.1.0-beta.1). Extract it, then run `run-studio.cmd` on Windows or `bash run-studio.sh` on Linux. Open [http://127.0.0.1:8875/studio](http://127.0.0.1:8875/studio). **The Windows archive runs natively; it does not require WSL or a Rust compiler.** The Linux binary was built on Ubuntu 25.04 (glibc 2.41), so older distributions may be incompatible.

These portable archives include the Rust server and embedded WASM UI, **not** model weights, original footage, FFmpeg/FFprobe, ONNX Runtime/CUDA/cuDNN or OpenVINO. AI analysis needs compatible external dependencies and model files. The launchers use a separate data directory (`%LOCALAPPDATA%\VolleyballStudioBeta\data` on Windows; `${XDG_DATA_HOME:-$HOME/.local/share}/volleyball-studio-beta` on Linux). Do not point the beta at the only copy of a project; never expose its unauthenticated localhost API to the Internet. See [beta release notes](docs/releases/v0.1.0-beta.1.md) and [deployment instructions](docs/RUST_DEPLOYMENT.md).

## Run from source on Windows / WSL

For source builds (not for the Windows beta ZIP), requirements include Windows with WSL (`Ubuntu-25.04` by default), a Rust toolchain inside WSL, FFmpeg/FFprobe, and compatible Linux libraries for the selected models. GPU inference requires a compatible NVIDIA/CUDA setup and ONNX Runtime; OpenVINO ReID requires its native runtime.

From the repository root:

```bat
start-volley-ui.cmd
start-volley-ui.cmd /daemon
start-volley-ui.cmd /doctor
start-volley-ui.cmd /status
start-volley-ui.cmd /check
start-volley-ui.cmd /stop
start-volley-ui.cmd /restart
```

| Command | What it does |
| --- | --- |
| `start-volley-ui.cmd` | Builds (if needed) and starts the local studio in the foreground on `127.0.0.1:8875`; closing the terminal stops the server. Reuses an already running instance. |
| `start-volley-ui.cmd /daemon` | Builds and starts a persistent detached Studio process. Does not depend on the initiating terminal or remote-command timeout. Writes the persistent log to `.studio-runtime/studio-server.log` in the repository. |
| `start-volley-ui.cmd /doctor` | Diagnoses dependencies, model files, system resources, GPU, ONNX Runtime and CUDA libraries. **Does not start** the studio. An unavailable CUDA runtime is reported as not ready (exit code `3`); this is not a live GPU inference test. |
| `start-volley-ui.cmd /status` | Reports whether the studio is running, its address and PID, and whether it is idle or executing jobs. **Does not start** the studio. |
| `start-volley-ui.cmd /check` | Runs Rust ownership/roadmap checks and offline `cargo check`; requires the development toolchain. |
| `start-volley-ui.cmd /stop` | Requests a graceful stop only when no active jobs would be interrupted. |
| `start-volley-ui.cmd /restart` | Safely stops and starts the studio; refuses to interrupt running GPU/CPU/video jobs. |

Or use PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-volley-rust-dev.ps1
```

Studio UI: **http://127.0.0.1:8875/studio**. This is a localhost application; do not expose its API directly to the public internet.

Run the complete local gate from the repository root with `powershell -ExecutionPolicy Bypass -File scripts/check-all.ps1`. See the [deployment guide](docs/RUST_DEPLOYMENT.md) for setup and diagnostics.

## Project status

**Public beta v0.1.0-beta.1 from `master`; ongoing development in `develop`.** Both remain experimental and not production-ready. A visible feature is not proof of accuracy on real matches. End-to-end acceptance, comparison with the historical Python implementation, and comprehensive quality measurements are still pending. Original footage and historical results are not intended to be silently overwritten.

**Documentation:** [Architecture](docs/RUST_ARCHITECTURE.md) · [Roadmap](docs/RUST_ROADMAP.md) · [Migration log](docs/RUST_MIGRATION.md) · [Deployment](docs/RUST_DEPLOYMENT.md) · [Developer guide](rust/README.md).
