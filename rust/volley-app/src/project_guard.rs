//! Per-project cross-process write guard for GPU/CPU and metadata updates.
//! This is independent of any removed archive or legacy import feature.
use crate::{
    projects,
    storage::{self, ApiError},
    AppState,
};
use fs2::FileExt;
use std::{
    fs::{self, File, OpenOptions},
    io, thread,
    time::{Duration, Instant},
};

fn retry_project_lock<T>(
    timeout: Duration,
    mut attempt: impl FnMut() -> Result<T, ApiError>,
) -> Result<T, ApiError> {
    let deadline = Instant::now() + timeout;
    loop {
        match attempt() {
            Err(error)
                if error.0 == axum::http::StatusCode::CONFLICT
                    && error.1 .0["error"] == "Проект занят другой операцией студии"
                    && Instant::now() < deadline =>
            {
                thread::sleep(Duration::from_millis(100));
            }
            result => return result,
        }
    }
}

pub(crate) fn write_guard_for_postprocessing(state: &AppState, id: &str) -> Result<File, ApiError> {
    retry_project_lock(Duration::from_secs(30), || write_guard(state, id))
}

pub(crate) fn write_guard(state: &AppState, id: &str) -> Result<File, ApiError> {
    if !state.writes_enabled {
        return Err(storage::conflict(
            "Изменения проекта требуют отдельного рабочего хранилища",
        ));
    }
    storage::project_id(id)?;
    let root = projects::project_root(state);
    fs::create_dir_all(&root).map_err(|e| storage::internal(format!("Project root: {e}")))?;
    let root_metadata = fs::symlink_metadata(&root)
        .map_err(|e| storage::internal(format!("Check project root: {e}")))?;
    let canonical = root
        .canonicalize()
        .map_err(|e| storage::internal(format!("Resolve project root: {e}")))?;
    if root_metadata.file_type().is_symlink()
        || !root_metadata.is_dir()
        || !canonical.starts_with(
            &state
                .data_root
                .canonicalize()
                .map_err(|e| storage::internal(format!("Resolve data root: {e}")))?,
        )
    {
        return Err(storage::bad_request(
            "Каталог проектов вне рабочего хранилища",
        ));
    }
    let locks = root.join(".project-locks");
    if let Ok(meta) = fs::symlink_metadata(&locks) {
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err(storage::bad_request("Папка блокировок проекта небезопасна"));
        }
    } else {
        fs::create_dir(&locks)
            .map_err(|e| storage::internal(format!("Project lock directory: {e}")))?;
    }
    let path = locks.join(format!("{id}.lock"));
    if fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(storage::bad_request("Файл блокировки проекта небезопасен"));
    }
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(|e| storage::internal(format!("Open project lock: {e}")))?;
    match lock.try_lock_exclusive() {
        Ok(()) => Ok(lock),
        Err(e)
            if e.kind() == io::ErrorKind::WouldBlock
                || e.raw_os_error() == fs2::lock_contended_error().raw_os_error() =>
        {
            Err(storage::conflict("Проект занят другой операцией студии"))
        }
        Err(e) => Err(storage::internal(format!("Lock project: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deferred_postprocessing_waits_for_busy_writer_only() {
        let mut calls = 0;
        let result = retry_project_lock(Duration::from_secs(2), || {
            calls += 1;
            if calls < 3 {
                Err(storage::conflict("Проект занят другой операцией студии"))
            } else {
                Ok(42)
            }
        });
        assert_eq!(result.unwrap(), 42);
        assert_eq!(calls, 3);

        let mut non_busy_calls = 0;
        let invalid = retry_project_lock(Duration::from_secs(2), || -> Result<(), ApiError> {
            non_busy_calls += 1;
            Err(storage::conflict("Timeline identity source changed"))
        });
        assert_eq!(
            invalid.unwrap_err().1 .0["error"],
            "Timeline identity source changed"
        );
        assert_eq!(non_busy_calls, 1);

        let mut exhausted_calls = 0;
        let exhausted = retry_project_lock(Duration::ZERO, || -> Result<(), ApiError> {
            exhausted_calls += 1;
            Err(storage::conflict("Проект занят другой операцией студии"))
        });
        assert!(exhausted.is_err());
        assert_eq!(exhausted_calls, 1);
    }
}
