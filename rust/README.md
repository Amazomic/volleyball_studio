# Volleyball Studio — developer workspace

This is the **only active application** on develop. Rust migration is complete (85/85, owner acceptance on 2026-11-27); product quality and production readiness remain separate.
`volley-app` is the Axum server, `volley-core` contains CPU identity,
motion and rally logic, `volley-inference` loads native ONNX Runtime,
`volley-media` invokes FFmpeg, and `volley-web` provides actual WASM state.

Launch from repository root via `start-volley-ui.cmd` (Windows),
or `scripts/start-volley-rust-dev.ps1` (PowerShell). Default
`127.0.0.1:8875`, with homepage redirect to `/studio`.
Use the Windows CMD actions /status, /stop, /restart, /doctor and /check,
or PowerShell -Action Status/Stop/Restart/Doctor/Check. WSL Rust uses
127.0.0.1:8875 by default, NOT the Python legacy port 8765. A second Rust
process is prevented by a canonical-checkout lock, even on another port;
stop/restart refuses to signal Rust while GPU/CPU/montage/Quick Reel is
active. The Dockerfile can build from sources, or package an offline Linux
release via scripts/ci/build_rust_docker_local.sh. Complete setup and
current CUDA/Docker verification status: ../docs/RUST_DEPLOYMENT.md.

Run `cargo fmt --all -- --check`, `cargo check --workspace --offline`
and `cargo test --workspace --offline` from this directory. GPU smoke:
`bash ../scripts/ci/smoke_rust_ort.sh /path/to/libonnxruntime.so cuda full`.

OpenVINO person ReID uses volley-inference/src/reid_openvino.rs and the
bundled models/openvino/person-reidentification-retail-0287 model. Configure the absolute
WSL path VOLLEY_OPENVINO_C_LIB and native LD_LIBRARY_PATH or pass the Windows
launcher option -OpenvinoLibrary /absolute/wsl/path/libopenvino_c.so.
Run cargo test --workspace --offline with VOLLEY_TEST_OPENVINO_C_LIB set
to the same native .so to execute the real-model smoke test. These
temporary, SHA-bound vectors are only visual suggestions, not identities.

On native Windows, install the runtime once with
`powershell -ExecutionPolicy Bypass -File scripts/setup-openvino-windows.ps1`
from the repository root. The launcher discovers
`%LOCALAPPDATA%\VolleyballStudio\native-deps\openvino\openvino\libs\openvino_c.dll`
or a 64-bit Python OpenVINO installation. `-OpenvinoLibrary` and
`VOLLEY_OPENVINO_C_LIB` override discovery; invalid paths fail at startup
rather than silently disabling ReID.
Manually reviewed native player reference JPEGs have a separate, versioned,
reversible global athlete album under the isolated Rust data root. Use
POST /api/players/native/references (read/enroll/undo) and GET
/api/players/native/reference-photo; enrollment requires an already confirmed
Track ID assignment in that exact SHA-verified CPU replay. References are
photos, not auto-confirmed identities and not yet computed embedding caches.
This is the Rust application, not a Python-backed interim runtime. It does not claim complete product parity or production readiness. Rust/WASM owns migrated client operations; retain browser integration and require measured benefit before porting further computations.
The former Python application is retained in Git history/checkpoint; optional
Python development, training and benchmark tools remain. Legacy media and
snapshots remain readable through Rust data adapters but are
never silently modified. See `../docs/RUST_MIGRATION.md` for historical
features, decisions and test evidence.
## Development UI assets

The local development launcher sets `VOLLEY_DEV_UI_ROOT` to `rust/volley-app/static`. In debug builds, HTML, CSS, SVG and JavaScript are read from that directory on each request with caching disabled. Refresh the browser after editing them. Release builds and debug builds without this variable use embedded assets. WASM and PNG remain embedded and require rebuilding the server.

`static/ui-contract.txt` is the manually maintained UI/server contract version. Increase it whenever changing the backend API or WASM ABI in a way the UI depends on, in the same change as the UI. A running debug server rejects disk assets when the file differs from its compiled version or is missing; rebuild and restart it. This version check depends on developers updating the contract and cannot detect unmarked incompatibilities. The launcher does not restart an active server, and its existing active-job guard remains in effect.
