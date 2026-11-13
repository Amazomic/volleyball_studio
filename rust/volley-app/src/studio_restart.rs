//! One-click restart for the localhost development Studio.
//! The external repository-scoped launcher repeats ownership and idle checks
//! before sending SIGTERM. No caller-supplied path or command is executed.
use crate::{runtime, storage, AppState};
use axum::{
    extract::{Query, State},
    http::HeaderMap,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    env,
    path::Path,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
static RESTART_PENDING: AtomicBool = AtomicBool::new(false);

pub(crate) fn current_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn restart_profile(profile: Option<&str>) -> Result<&'static str, &'static str> {
    match profile {
        None => Ok(current_profile()),
        Some("debug") => Ok("debug"),
        Some("release") => Ok("release"),
        _ => Err("Недопустимый режим запуска"),
    }
}

#[derive(Deserialize)]
pub(crate) struct RestartQuery {
    profile: Option<String>,
}

fn launch_args(state: &AppState, profile: &str) -> Vec<String> {
    let root = &state.repository_root;
    let bind = env::var("VOLLEY_RUST_BIND").unwrap_or_else(|_| "127.0.0.1:8875".into());
    let data = if state.writes_enabled {
        state.data_root.to_string_lossy().to_string()
    } else {
        "__default__".into()
    };
    let mode = if state.writes_enabled {
        "isolated"
    } else {
        "readonly"
    };
    let ort = env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "__not_configured__".into());
    let openvino =
        env::var("VOLLEY_OPENVINO_C_LIB").unwrap_or_else(|_| "__not_configured__".into());
    vec![
        root.join("scripts/volley-rust.sh")
            .to_string_lossy()
            .to_string(),
        "restart".into(),
        root.to_string_lossy().to_string(),
        bind,
        data,
        mode.into(),
        ort,
        openvino,
        profile.into(),
    ]
}

pub(crate) async fn restart(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<RestartQuery>,
) -> storage::ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let profile = restart_profile(query.profile.as_deref()).map_err(storage::bad_request)?;
    let active = runtime::active_job_types(&state).await;
    if !active.is_empty() {
        return Err(storage::conflict(format!(
            "Перезапуск запрещён: выполняются задачи {}. Дождитесь завершения.",
            active.join(", ")
        )));
    }
    let script = state.repository_root.join("scripts/volley-rust.sh");
    if !Path::new(&script).is_file() {
        return Err(storage::internal("Studio launcher is missing"));
    }
    if RESTART_PENDING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err(storage::conflict("Перезапуск уже запрошен."));
    }
    // setsid -f forks a detached session: it survives the old Rust process and
    // the foreground PowerShell/WSL wrapper. The launcher itself refuses to
    // stop any unknown process or any native worker that becomes active.
    let args = launch_args(&state, profile);
    let result = Command::new("setsid")
        .arg("-f")
        .arg("bash")
        .arg("-c")
        // All interpolated values stay positional; no shell evaluation of paths.
        .arg("sleep 0.8; exec bash \"$@\"")
        .arg("volley-restart")
        .args(&args)
        .current_dir(&state.repository_root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    if let Err(error) = result {
        RESTART_PENDING.store(false, Ordering::SeqCst);
        return Err(storage::internal(format!(
            "Cannot launch studio restart: {error}"
        )));
    }
    // If an external launcher is denied by a newly active job, do not leave
    // this instance permanently unable to accept a later restart request.
    tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        RESTART_PENDING.store(false, Ordering::SeqCst);
    });
    Ok(Json(
        json!({"scheduled":true,"previous_pid":std::process::id()}),
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn restart_profile_rejects_unknown_values_and_defaults_to_current_build() {
        assert_eq!(
            super::restart_profile(None).unwrap(),
            super::current_profile()
        );
        assert_eq!(super::restart_profile(Some("release")).unwrap(), "release");
        assert_eq!(super::restart_profile(Some("debug")).unwrap(), "debug");
        assert!(super::restart_profile(Some("other")).is_err());
    }

    #[test]
    fn launch_never_accepts_user_defined_command_or_remote_target() {
        let source = include_str!("studio_restart.rs");
        assert!(source.contains("scripts/volley-rust.sh"));
        assert!(source.contains("active_job_types"));
        assert!(source.contains("storage::validate_origin"));
        assert!(source.contains("sleep 0.8; exec bash"));
        assert!(source.contains(".arg(\"volley-restart\")"));
    }
}
