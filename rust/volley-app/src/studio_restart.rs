//! One-click restart for the localhost development Studio.
//! The external repository-scoped launcher repeats ownership and idle checks
//! before sending SIGTERM. No caller-supplied path or command is executed.
use crate::{runtime, storage, AppState};
use axum::{extract::State, http::HeaderMap, Json};
use serde_json::{json, Value};
use std::{
    env,
    fs::{self, OpenOptions},
    io,
    path::PathBuf,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
static RESTART_PENDING: AtomicBool = AtomicBool::new(false);
static RESTART_FAILED: AtomicBool = AtomicBool::new(false);

pub(crate) fn current_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

pub(crate) fn restart_failed() -> bool {
    RESTART_FAILED.load(Ordering::Acquire)
}

#[cfg(windows)]
fn windows_path(path: &std::path::Path) -> PathBuf {
    let path = path.to_string_lossy();
    PathBuf::from(path.strip_prefix(r"\\?\").unwrap_or(&path))
}

#[cfg(windows)]
fn restart_command(state: &AppState) -> io::Result<Command> {
    use std::os::windows::process::CommandExt;
    let root = windows_path(&state.repository_root);
    let script = root.join("scripts/deploy-volley-queue.ps1");
    if !script.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Studio deployment queue is missing",
        ));
    }
    let system = env::var_os("SystemRoot")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "SystemRoot is missing"))?;
    let mut command =
        Command::new(PathBuf::from(system).join("System32/WindowsPowerShell/v1.0/powershell.exe"));
    command
        .creation_flags(0x08000000)
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
        ])
        .arg(script)
        .arg("-DeploymentRoot")
        .arg(&root)
        .arg("-RestartFromPid")
        .arg(std::process::id().to_string())
        .arg("-Bind")
        .arg(env::var("VOLLEY_RUST_BIND").unwrap_or_else(|_| "127.0.0.1:8875".into()));
    if state.writes_enabled {
        command
            .arg("-RuntimeDataRoot")
            .arg(windows_path(&state.data_root));
    } else {
        command.arg("-RuntimeReadOnly");
    }
    Ok(command)
}

#[cfg(target_os = "linux")]
fn restart_command(state: &AppState) -> io::Result<Command> {
    let root = &state.repository_root;
    let script = root.join("scripts/volley-rust.sh");
    if !script.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Studio launcher is missing",
        ));
    }
    let mut command = Command::new("setsid");
    command
        .arg("-f")
        .arg("-w")
        .arg("bash")
        .arg("-c")
        .arg("sleep 0.8; exec bash \"$@\"")
        .arg("volley-restart")
        .arg(script)
        .arg("restart")
        .arg(root)
        .arg(env::var("VOLLEY_RUST_BIND").unwrap_or_else(|_| "127.0.0.1:8875".into()))
        .arg(if state.writes_enabled {
            state.data_root.clone()
        } else {
            PathBuf::from("__default__")
        })
        .arg(if state.writes_enabled {
            "isolated"
        } else {
            "readonly"
        })
        .arg(env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "__not_configured__".into()))
        .arg(env::var("VOLLEY_OPENVINO_C_LIB").unwrap_or_else(|_| "__not_configured__".into()));
    Ok(command)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn restart_command(_state: &AppState) -> io::Result<Command> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Unsupported Studio restart platform",
    ))
}

fn launch(state: &AppState) -> io::Result<std::process::Child> {
    let mut command = restart_command(state)?;
    let runtime = state.repository_root.join(".studio-runtime");
    fs::create_dir_all(&runtime)?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(runtime.join("studio-restart.log"))?;
    command
        .current_dir(&state.repository_root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()
}

pub(crate) async fn restart(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> storage::ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let active = runtime::active_job_types(&state).await;
    if !active.is_empty() {
        return Err(storage::conflict(format!(
            "Перезапуск запрещён: выполняются задачи {}. Дождитесь завершения.",
            active.join(", ")
        )));
    }
    if RESTART_PENDING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err(storage::conflict("Перезапуск уже запрошен."));
    }
    RESTART_FAILED.store(false, Ordering::Release);
    let mut child = match launch(&state) {
        Ok(child) => child,
        Err(error) => {
            RESTART_PENDING.store(false, Ordering::Release);
            return Err(storage::internal(format!(
                "Cannot launch studio restart: {error}"
            )));
        }
    };
    std::thread::spawn(move || {
        RESTART_FAILED.store(
            !child.wait().is_ok_and(|status| status.success()),
            Ordering::Release,
        );
        RESTART_PENDING.store(false, Ordering::Release);
    });
    Ok(Json(
        json!({"scheduled":true,"previous_pid":std::process::id()}),
    ))
}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    #[test]
    fn windows_launcher_paths_use_native_drive_syntax() {
        use super::*;
        assert_eq!(
            windows_path(std::path::Path::new(r"\\?\C:\Studio clone")),
            PathBuf::from(r"C:\Studio clone")
        );
    }

    #[test]
    fn launch_never_accepts_user_defined_command_or_remote_target() {
        let source = include_str!("studio_restart.rs");
        assert!(source.contains("scripts/volley-rust.sh"));
        assert!(source.contains("scripts/deploy-volley-queue.ps1"));
        assert!(source.contains("active_job_types"));
        assert!(source.contains("storage::validate_origin"));
        assert!(source.contains("sleep 0.8; exec bash"));
        assert!(source.contains(".arg(\"volley-restart\")"));
    }
}
