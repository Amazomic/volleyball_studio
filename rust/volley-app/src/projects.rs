//! Rust-native project management and compatibility with existing project JSON.
use crate::{
    storage::{self, ApiError, ApiResult},
    AppState,
};
use axum::{extract::State, http::HeaderMap, Json};
use fs2::FileExt;
use serde_json::{json, Map, Value};
use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

pub(crate) fn project_root(state: &AppState) -> PathBuf {
    state.data_root.join("projects")
}

pub(crate) fn registry_path(state: &AppState) -> PathBuf {
    project_root(state).join("projects.json")
}

pub(crate) fn load_projects(state: &AppState) -> Result<Vec<Value>, ApiError> {
    if let Ok(meta) = fs::symlink_metadata(registry_path(state)) {
        if meta.file_type().is_symlink() || !meta.is_file() || meta.len() > 8 * 1024 * 1024 {
            return Err(storage::bad_request("Unsafe project registry"));
        }
    }
    let Some(loaded) = storage::read_json(&registry_path(state))? else {
        return Ok(Vec::new());
    };
    let rows = loaded
        .get("projects")
        .and_then(Value::as_array)
        .ok_or_else(|| storage::internal("Project registry lacks projects array"))?;
    Ok(rows.clone())
}

fn find_project(state: &AppState, id: &str) -> Result<Value, ApiError> {
    storage::project_id(id)?;
    load_projects(state)?
        .into_iter()
        .find(|row| row["id"].as_str() == Some(id))
        .ok_or_else(|| storage::not_found("Проект не найден"))
}

pub(crate) fn uploaded_video_root(state: &AppState) -> PathBuf {
    state.data_root.join("uploads").join("project_sources")
}

/// Resolve either an immutable original in repository/project or a validated
/// browser upload in the isolated Rust data root, never a caller-supplied path.
fn linked_source_root(state: &AppState) -> PathBuf {
    state.data_root.join("linked_sources")
}

/// Only metadata minted by the native picker can grant access to an external file.
fn linked_source(state: &AppState, name: &str) -> Result<Option<PathBuf>, ApiError> {
    let Some(tail) = name.strip_prefix("linked_") else {
        return Ok(None);
    };
    let Some((id, _extension)) = tail.rsplit_once('.') else {
        return Ok(None);
    };
    if id.len() != 32 || !id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Ok(None);
    }
    let record = linked_source_root(state).join(format!("{name}.json"));
    let metadata = match fs::symlink_metadata(&record) {
        Ok(metadata) => metadata,
        Err(_) => return Err(storage::bad_request("Linked video is no longer registered")),
    };
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 8192 {
        return Err(storage::bad_request("Unsafe linked video record"));
    }
    let value = storage::read_json(&record)?
        .ok_or_else(|| storage::bad_request("Linked video record missing"))?;
    let path = value["path"]
        .as_str()
        .ok_or_else(|| storage::bad_request("Invalid video link"))?;
    let original = Path::new(path);
    if !original.is_absolute() {
        return Err(storage::bad_request("Invalid video link path"));
    }
    let target = original
        .canonicalize()
        .map_err(|_| storage::bad_request("Original video is unavailable"))?;
    if !target.is_file() || target.metadata().is_ok_and(|m| m.len() < 1024) {
        return Err(storage::bad_request("Original video is unavailable"));
    }
    Ok(Some(target))
}

/// Native picker directly links original videos: no upload or SHA-256.
pub async fn native_pick(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let Json(picked) = crate::quick_reel::native_pick(State(state.clone()), headers).await?;
    let rows = picked["videos"].as_array().cloned().unwrap_or_default();
    if rows.is_empty() {
        return Ok(Json(json!({"videos":[],"cancelled":true})));
    }
    let state2 = state.clone();
    let linked = tokio::task::spawn_blocking(move || {
        let folder = linked_source_root(&state2);
        fs::create_dir_all(&folder)
            .map_err(|e| storage::internal(format!("Linked source dir: {e}")))?;
        let mut result = Vec::new();
        for row in rows {
            let token = row["id"]
                .as_str()
                .ok_or_else(|| storage::bad_request("Missing selection token"))?;
            let path = crate::quick_reel::picked_path(&state2, token)?;
            let extension = path
                .extension()
                .and_then(|s| s.to_str())
                .ok_or_else(|| storage::bad_request("Video extension missing"))?
                .to_ascii_lowercase();
            let name = format!("linked_{}.{}", Uuid::new_v4().simple(), extension);
            storage::video_name(&name)?;
            let display_name = path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| storage::bad_request("Invalid video filename"))?;
            let duration_sec = video_duration_secs(&path)?;
            storage::write_json_atomic(
                &folder.join(format!("{name}.json")),
                &json!({"path":path.to_string_lossy().to_string()}),
            )?;
            result.push(json!({"name":name,"display_name":display_name,
                "duration_sec":duration_sec,"size_gb":row["size_gb"],"origin":"linked"}));
        }
        Ok::<_, ApiError>(result)
    })
    .await
    .map_err(|e| storage::internal(format!("File picker task: {e}")))??;
    Ok(Json(json!({"videos":linked})))
}

pub(crate) fn canonical_source(state: &AppState, name: &str) -> Result<PathBuf, ApiError> {
    storage::video_name(name)?;
    if let Some(path) = linked_source(state, name)? {
        return Ok(path);
    }
    let original = state.repository_root.join("project").join(name);
    let uploaded_root = uploaded_video_root(state);
    let isolated = uploaded_root.join(name);
    for candidate in [&original, &isolated] {
        if candidate == &isolated
            && fs::symlink_metadata(candidate)
                .is_ok_and(|m| m.file_type().is_symlink() || !m.is_file())
        {
            continue;
        }
        let Ok(path) = candidate.canonicalize() else {
            continue;
        };
        if candidate == &isolated {
            let Some(root) = uploaded_root.canonicalize().ok() else {
                continue;
            };
            let Some(data) = state.data_root.canonicalize().ok() else {
                continue;
            };
            if !root.starts_with(data) || !path.starts_with(&root) {
                continue;
            }
        }
        if path.is_file() && path.metadata().is_ok_and(|m| m.len() >= 1024) {
            return Ok(path);
        }
    }
    Err(storage::bad_request(format!("Видео не найдено: {name}")))
}

pub(crate) fn event_source_matches(
    state: &AppState,
    events: &Path,
    name: &str,
    resolve: bool,
) -> bool {
    let Ok(Some(data)) = storage::read_json(events) else {
        return false;
    };
    let Some(recorded) = data.get("source").and_then(Value::as_str) else {
        return false;
    };
    let normalized = recorded.replace('\\', "/");
    if !resolve {
        return Path::new(&normalized).file_name().and_then(|s| s.to_str()) == Some(name);
    }
    let path = Path::new(&normalized);
    let source = if path.is_absolute() {
        path.to_path_buf()
    } else {
        state.repository_root.join(path)
    };
    let expected = canonical_source(state, name);
    matches!((source.canonicalize(), expected), (Ok(a), Ok(b)) if a == b)
}

fn latest_cache(state: &AppState, name: &str, project_id: Option<&str>) -> Option<String> {
    let scoped = project_id.is_some();
    let dir = if let Some(id) = project_id {
        project_root(state).join(id).join("analysis")
    } else {
        state.data_root.join("analysis")
    };
    let mut latest: Option<(SystemTime, String)> = None;
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let events = path.join("analysis").join("events.json");
        if !event_source_matches(state, &events, name, scoped) {
            continue;
        }
        let stamp = events
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let Some(id) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if latest.as_ref().is_none_or(|(seen, _)| stamp > *seen) {
            latest = Some((stamp, id.to_owned()));
        }
    }
    latest.map(|(_, name)| name)
}

fn project_cache_available(state: &AppState, name: &str, id: &str) -> bool {
    latest_cache(state, name, Some(id)).is_some()
}
pub(crate) fn safe_video_available(state: &AppState, project: &Value, video: &str) -> bool {
    if storage::video_name(video).is_err() {
        return false;
    }
    let Ok(source) = canonical_source(state, video) else {
        return false;
    };
    if !source.is_file() || source.metadata().map_or(true, |m| m.len() < 1024) {
        return false;
    }
    if let Some(expected) = project
        .get("source_paths")
        .and_then(|s| s.get(video))
        .and_then(Value::as_str)
    {
        // Old imported absolute source paths are authoritative; Rust never
        // guesses a different source just because its filename matches.
        let expected = Path::new(expected);
        if expected.canonicalize().ok().as_ref() != Some(&source) {
            return false;
        }
    }
    true
}

fn project_summary(state: &AppState, project: &Value) -> Result<Value, ApiError> {
    let id = project
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| storage::internal("Project record has no ID"))?;
    storage::project_id(id)?;
    let videos = project
        .get("videos")
        .and_then(Value::as_array)
        .ok_or_else(|| storage::internal("Project record has no videos"))?;

    let mut analyses = Vec::with_capacity(videos.len());
    let mut missing = Vec::new();
    for video in videos {
        let Some(name) = video.as_str() else {
            continue;
        };
        let available = safe_video_available(state, project, name);
        if !available {
            missing.push(name.to_owned());
        }
        analyses.push(json!({
            "video": name,
            "available": available,
            "analyzed": available && project_cache_available(state, name, id)
        }));
    }

    let montage = project_root(state).join(id).join("highlights.mp4");
    let highlight_ready = montage.metadata().is_ok_and(|m| m.is_file() && m.len() > 0);
    let mut summary = project
        .as_object()
        .cloned()
        .ok_or_else(|| storage::internal("Project record is not object"))?;
    summary.entry("ranges").or_insert_with(|| json!({}));
    summary.insert("analyses".into(), json!(analyses));
    summary.insert("missing_videos".into(), json!(missing));
    summary.insert("highlight_ready".into(), json!(highlight_ready));
    Ok(Value::Object(summary))
}

pub async fn get_projects(State(state): State<Arc<AppState>>) -> ApiResult<Value> {
    let projects = tokio::task::spawn_blocking(move || {
        load_projects(&state)?
            .iter()
            .map(|project| project_summary(&state, project))
            .collect::<Result<Vec<_>, _>>()
    })
    .await
    .map_err(|error| storage::internal(format!("Project task failed: {error}")))??;

    // The Python UI reads {"projects": [summary...]}, never a bare list.
    Ok(Json(json!({"projects": projects})))
}

pub(crate) fn existing_project(state: &AppState, id: &str) -> Result<Value, ApiError> {
    find_project(state, id)
}

pub(crate) fn read_profiles(state: &AppState) -> Result<Vec<Value>, ApiError> {
    let path = state.data_root.join("analysis_profiles.json");
    let Some(contents) = storage::read_json(&path)? else {
        return Ok(Vec::new());
    };
    let Some(profiles) = contents.get("profiles").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    Ok(profiles
        .iter()
        .filter(|row| {
            row.is_object()
                && row.get("id").is_some_and(Value::is_string)
                && row.get("values").is_some_and(Value::is_object)
        })
        .cloned()
        .collect())
}

fn normalized_name(name: &str) -> Result<String, ApiError> {
    let clean: String = name
        .nfkc()
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if (1..=100).contains(&clean.chars().count()) {
        Ok(clean)
    } else {
        Err(storage::bad_request(
            "Название проекта должно содержать от 1 до 100 символов",
        ))
    }
}

fn player_ids(state: &AppState, value: Option<&Value>) -> Result<Vec<String>, ApiError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let ids = value
        .as_array()
        .ok_or_else(|| storage::bad_request("Выберите не более 200 игроков"))?;
    if ids.len() > 200 {
        return Err(storage::bad_request("Выберите не более 200 игроков"));
    }
    let catalog = state.data_root.join("players/registry.json");
    let players = storage::read_json(&catalog)?
        .and_then(|doc| doc.get("players").and_then(Value::as_array).cloned())
        .unwrap_or_default();
    let known: HashSet<&str> = players.iter().filter_map(|p| p["id"].as_str()).collect();

    let mut unique = Vec::new();
    let mut selected = HashSet::new();
    for id in ids {
        let id = id
            .as_str()
            .ok_or_else(|| storage::bad_request("Некорректный идентификатор игрока"))?;
        if !known.contains(id) {
            return Err(storage::bad_request(
                "Один из выбранных игроков не найден в общей базе",
            ));
        }
        if selected.insert(id) {
            unique.push(id.to_owned());
        }
    }
    Ok(unique)
}

fn list_video_names(state: &AppState, value: &Value) -> Result<Vec<String>, ApiError> {
    let videos = value
        .as_array()
        .ok_or_else(|| storage::bad_request("Выберите видео для проекта"))?;
    if videos.is_empty() || videos.len() > 30 {
        return Err(storage::bad_request("Выберите от 1 до 30 видео"));
    }
    let mut names = Vec::new();
    let mut seen = HashSet::new();
    for item in videos {
        let name = item
            .as_str()
            .ok_or_else(|| storage::bad_request("Некорректное имя видео"))?;
        canonical_source(state, name)?;
        if seen.insert(name.to_owned()) {
            names.push(name.to_owned());
        }
    }
    Ok(names)
}

fn video_duration_secs(path: &Path) -> Result<f64, ApiError> {
    let duration = volley_media::probe(path)
        .map_err(storage::bad_request)?
        .duration_sec;
    if duration.is_finite() && duration > 0.0 {
        Ok(duration)
    } else {
        Err(storage::bad_request("Некорректная длительность видео"))
    }
}

fn rounded_secs(time: f64) -> f64 {
    // Python round precision matches for normal timestamps (ties unimportant
    // except at exact .0005; use same recorded three-decimal contract).
    (time * 1000.0).round() / 1000.0
}

fn validate_ranges(
    state: &AppState,
    videos: &[String],
    ranges: Option<&Value>,
) -> Result<Value, ApiError> {
    let explicit = match ranges {
        None | Some(Value::Null) => None,
        Some(Value::Object(object)) => Some(object),
        _ => return Err(storage::bad_request("Диапазоны должны быть объектом")),
    };
    if let Some(ranges) = explicit {
        if ranges.keys().any(|name| !videos.contains(name)) {
            return Err(storage::bad_request(
                "Диапазоны должны относиться к выбранным видео",
            ));
        }
    }
    let mut result = Map::new();
    for name in videos {
        let path = canonical_source(state, name)?;
        let duration = video_duration_secs(&path)?;
        let (start, end) = match explicit.and_then(|v| v.get(name)) {
            None | Some(Value::Null) => (0.0, duration),
            Some(Value::Object(selected)) => {
                if selected.len() != 2
                    || !selected.contains_key("start")
                    || !selected.contains_key("end")
                {
                    return Err(storage::bad_request("Неправильный диапазон видео"));
                }
                let start = selected["start"]
                    .as_f64()
                    .ok_or_else(|| storage::bad_request("Некорректная граница начала"))?;
                let end = selected["end"]
                    .as_f64()
                    .ok_or_else(|| storage::bad_request("Некорректная граница конца"))?;
                if !start.is_finite()
                    || !end.is_finite()
                    || start < 0.0
                    || start >= end
                    || end > duration + 0.01
                    || end - start < 1.0
                {
                    return Err(storage::bad_request(
                        "Диапазон должен быть в пределах видео (не менее 1 с)",
                    ));
                }
                (start, end.min(duration))
            }
            _ => return Err(storage::bad_request("Неправильный диапазон видео")),
        };
        result.insert(
            name.clone(),
            json!({"start": rounded_secs(start), "end": rounded_secs(end)}),
        );
    }
    Ok(Value::Object(result))
}

fn ensure_writable(state: &AppState) -> Result<(), ApiError> {
    if !state.writes_enabled {
        Err(storage::conflict(
            "Текущий источник доступен только для чтения. Для записи настройте отдельное рабочее хранилище",
        ))
    } else {
        Ok(())
    }
}

/// Registry lock is scoped to the isolated Rust data directory. This never
/// takes the Python app's real .registry.lock unless the sandbox guard fails.
pub(crate) fn with_registry_lock<T>(
    state: &AppState,
    operation: impl FnOnce() -> Result<T, ApiError>,
) -> Result<T, ApiError> {
    ensure_writable(state)?;
    let root = project_root(state);
    fs::create_dir_all(&root)
        .map_err(|err| storage::internal(format!("Create dev project root: {err}")))?;
    let root_metadata = fs::symlink_metadata(&root)
        .map_err(|e| storage::internal(format!("Registry root check: {e}")))?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err(storage::bad_request(
            "Project registry root cannot be a symlink",
        ));
    }
    for target in [root.join(".registry.lock"), registry_path(state)] {
        if fs::symlink_metadata(&target).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(storage::bad_request(
                "Project registry file is an unsafe symlink",
            ));
        }
    }
    let file: File = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(".registry.lock"))
        .map_err(|err| storage::internal(format!("Open dev registry lock: {err}")))?;
    file.lock_exclusive()
        .map_err(|err| storage::internal(format!("Lock dev registry: {err}")))?;
    let outcome = operation();
    // File lock automatically releases after this call. Explicitly drop it
    // after every operation rather than keep a lock in shared application state.
    drop(file);
    outcome
}

fn create_record(
    state: &AppState,
    request: Value,
    #[cfg(test)] before_registry_lock: impl FnOnce(),
) -> Result<Value, ApiError> {
    let raw_name = request["name"]
        .as_str()
        .ok_or_else(|| storage::bad_request("Укажите название проекта"))?;
    let name = normalized_name(raw_name)?;
    let videos = list_video_names(state, &request["videos"])?;
    let ranges = validate_ranges(state, &videos, request.get("ranges"))?;
    let ids = player_ids(state, request.get("player_ids"))?;
    #[cfg(test)]
    before_registry_lock();
    with_registry_lock(state, || {
        let mut sources = Map::new();
        for video in &videos {
            sources.insert(
                video.to_owned(),
                json!(canonical_source(state, video)?
                    .to_string_lossy()
                    .to_string()),
            );
        }
        let mut projects = load_projects(state)?;
        if projects.iter().any(|p| {
            p["name"]
                .as_str()
                .is_some_and(|other| other.to_lowercase() == name.to_lowercase())
        }) {
            return Err(storage::bad_request(
                "Проект с таким названием уже существует",
            ));
        }
        let id = (0..20)
            .map(|_| Uuid::new_v4().simple().to_string()[..12].to_owned())
            .find(|id| !project_root(state).join(id).exists())
            .ok_or_else(|| storage::internal("Unable to allocate unique project ID"))?;
        let project = json!({
            "id": id,
            "name": name,
            "videos": videos,
            "ranges": ranges,
            "source_paths": sources,
            "player_ids": ids,
        });
        let directory = project_root(state).join(&id);
        fs::create_dir(&directory)
            .map_err(|err| storage::internal(format!("Create dev project: {err}")))?;
        if let Err(error) = storage::write_json_atomic(&directory.join("project.json"), &project) {
            let _ = fs::remove_dir(&directory);
            return Err(error);
        }
        projects.push(project.clone());
        if let Err(error) = storage::write_json_atomic(
            &registry_path(state),
            &json!({"schema":1,"projects":projects}),
        ) {
            let _ = fs::remove_file(directory.join("project.json"));
            let _ = fs::remove_dir(&directory);
            return Err(error);
        }
        Ok(project)
    })
}

fn update_record(state: &AppState, request: Value) -> Result<Value, ApiError> {
    let id = request["id"]
        .as_str()
        .ok_or_else(|| storage::bad_request("Укажите идентификатор проекта"))?
        .to_owned();
    storage::project_id(&id)?;
    with_registry_lock(state, || {
        let mut projects = load_projects(state)?;
        let index = projects
            .iter()
            .position(|row| row["id"].as_str() == Some(&id))
            .ok_or_else(|| storage::not_found("Проект не найден"))?;
        let before = projects[index].clone();
        let mut item = before
            .as_object()
            .cloned()
            .ok_or_else(|| storage::internal("Invalid project record"))?;
        if let Some(name) = request.get("name").filter(|v| !v.is_null()) {
            let name = normalized_name(
                name.as_str()
                    .ok_or_else(|| storage::bad_request("Некорректное имя проекта"))?,
            )?;
            if projects.iter().enumerate().any(|(idx, p)| {
                idx != index
                    && p["name"]
                        .as_str()
                        .is_some_and(|other| other.to_lowercase() == name.to_lowercase())
            }) {
                return Err(storage::bad_request("Название проекта уже занято"));
            }
            item.insert("name".into(), json!(name));
        }
        if let Some(ids) = request.get("player_ids").filter(|v| !v.is_null()) {
            item.insert("player_ids".into(), json!(player_ids(state, Some(ids))?));
        }
        let changing_videos = request.get("videos").is_some_and(|v| !v.is_null());
        let changing_ranges = request.get("ranges").is_some_and(|v| !v.is_null());
        if changing_videos || changing_ranges {
            let source_videos = if changing_videos {
                list_video_names(state, &request["videos"])?
            } else {
                list_video_names(state, &item["videos"])?
            };
            let target_ranges = if changing_ranges {
                request.get("ranges")
            } else {
                Some(&item["ranges"])
            };
            let existing_ranges = if changing_videos && !changing_ranges {
                // Python drops range values for removed sources.
                let previous = target_ranges.and_then(Value::as_object);
                let mut reduced = Map::new();
                for name in &source_videos {
                    if let Some(old) = previous.and_then(|p| p.get(name)) {
                        reduced.insert(name.clone(), old.clone());
                    }
                }
                Some(Value::Object(reduced))
            } else {
                None
            };
            let range_ref = existing_ranges.as_ref().or(target_ranges);
            item.insert(
                "ranges".into(),
                validate_ranges(state, &source_videos, range_ref)?,
            );
            let mut sources = Map::new();
            for name in &source_videos {
                sources.insert(
                    name.to_owned(),
                    json!(canonical_source(state, name)?.to_string_lossy().to_string()),
                );
            }
            item.insert("videos".into(), json!(source_videos));
            item.insert("source_paths".into(), Value::Object(sources));
        }
        let updated = Value::Object(item);
        projects[index] = updated.clone();
        let detail = project_root(state).join(&id).join("project.json");
        // Both files are atomically swapped individually; if the registry
        // update fails, restore the old per-project document.
        storage::write_json_atomic(&detail, &updated)?;
        if let Err(error) = storage::write_json_atomic(
            &registry_path(state),
            &json!({"schema": 1, "projects": projects}),
        ) {
            let _ = storage::write_json_atomic(&detail, &before);
            return Err(error);
        }
        Ok(updated)
    })
}

/// The registry and folder rename are the user-visible commit. Large file
/// deletion happens later, without holding project/catalog locks.
fn delete_from_root(root: &Path, id: &str) -> Result<(PathBuf, Vec<String>), ApiError> {
    storage::project_id(id)?;
    let registry = root.join("projects.json");
    let mut data =
        storage::read_json(&registry)?.ok_or_else(|| storage::not_found("Проект не найден"))?;
    let rows = data["projects"]
        .as_array_mut()
        .ok_or_else(|| storage::internal("Invalid project registry"))?;
    let index = rows
        .iter()
        .position(|row| row["id"].as_str() == Some(id))
        .ok_or_else(|| storage::not_found("Проект не найден"))?;
    let source = root.join(id);
    let metadata = fs::symlink_metadata(&source)
        .map_err(|_| storage::not_found("Каталог проекта не найден"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(storage::bad_request("Небезопасный каталог проекта"));
    }
    let staged = root.join(format!(".deleted-{}", Uuid::new_v4().simple()));
    fs::rename(&source, &staged)
        .map_err(|e| storage::internal(format!("Stage project deletion: {e}")))?;
    let removed = rows.remove(index);
    let shared: HashSet<String> = rows
        .iter()
        .filter_map(|row| row["videos"].as_array())
        .flat_map(|videos| videos.iter().filter_map(Value::as_str))
        .map(str::to_owned)
        .collect();
    let exclusive = removed["videos"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|video| storage::video_name(video).is_ok() && !shared.contains(*video))
        .map(str::to_owned)
        .collect();
    if let Err(error) = storage::write_json_atomic(&registry, &data) {
        fs::rename(&staged, &source)
            .map_err(|e| storage::internal(format!("Restore project after registry error: {e}")))?;
        return Err(error);
    }
    Ok((staged, exclusive))
}

/// Remove only Studio-owned uploads and link records that no surviving project
/// references. An original video in repository/project or an external linked
/// file is never removed by project deletion.
fn cleanup_staged_project(state: &AppState, staged: PathBuf, candidates: Vec<String>) {
    if let Err(error) = fs::remove_dir_all(&staged) {
        eprintln!("Background project cleanup {}: {error}", staged.display());
    }
    if let Err(error) = with_registry_lock(state, || {
        let referenced: HashSet<String> = load_projects(state)?
            .iter()
            .filter_map(|p| p["videos"].as_array())
            .flat_map(|videos| videos.iter().filter_map(Value::as_str))
            .map(str::to_owned)
            .collect();
        for video in candidates {
            if referenced.contains(&video) {
                continue;
            }
            let uploaded = uploaded_video_root(state);
            let mut files = vec![
                uploaded.join(&video),
                uploaded.join(format!("{video}.json")),
            ];
            if video.starts_with("linked_") {
                files.push(linked_source_root(state).join(format!("{video}.json")));
            }
            for file in files {
                match fs::symlink_metadata(&file) {
                    Ok(meta) if meta.is_file() && !meta.file_type().is_symlink() => {
                        if let Err(error) = fs::remove_file(&file) {
                            eprintln!("Background video cleanup {}: {error}", file.display());
                        }
                    }
                    Ok(_) => eprintln!("Skipping unsafe video cleanup {}", file.display()),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => eprintln!("Video cleanup check {}: {error}", file.display()),
                }
            }
        }
        Ok(())
    }) {
        eprintln!("Background project source cleanup: {error:?}");
    }
}

pub async fn delete_project(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let id = request["id"]
        .as_str()
        .ok_or_else(|| storage::bad_request("Укажите идентификатор проекта"))?;
    existing_project(&state, id)?;
    // Only jobs using THIS project block its deletion. Independent runs
    // must remain accessible, even if they are still processing.
    if crate::studio_pipeline::has_active_for_project(id).await
        || state.full_inference.has_active_for_project(id).await
        || state.montage.has_active_for_project(id).await
        || state.preparation.has_active_for_project(id).await
    {
        return Err(storage::conflict(
            "Этот проект используется выполняющейся задачей",
        ));
    }
    let _guard = crate::project_guard::write_guard(&state, id)?;
    let _catalog = crate::analysis_management::catalog_guard(&state, id)?;
    let (staged, exclusive) =
        with_registry_lock(&state, || delete_from_root(&project_root(&state), id))?;
    let cleanup_state = state.clone();
    tokio::task::spawn_blocking(move || {
        cleanup_staged_project(&cleanup_state, staged, exclusive);
    });
    Ok(Json(json!({"ok":true})))
}

/// Retry committed trash after an interrupted process. Nothing in these
/// folders is visible in the registry, and a source still referenced by
/// another project is rechecked before its Studio-owned copy is removed.
pub(crate) fn cleanup_orphaned_deletions(state: &AppState) {
    if !state.writes_enabled {
        return;
    }
    let root = project_root(state);
    let Ok(entries) = fs::read_dir(&root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(token) = name.strip_prefix(".deleted-") else {
            continue;
        };
        if Uuid::parse_str(token).is_err()
            || !fs::symlink_metadata(entry.path())
                .is_ok_and(|m| m.is_dir() && !m.file_type().is_symlink())
        {
            continue;
        }
        let staged = entry.path();
        let Some(record) = storage::read_json(&staged.join("project.json"))
            .ok()
            .flatten()
        else {
            eprintln!(
                "Skipping project trash without a valid project.json: {}",
                staged.display()
            );
            continue;
        };
        let Some(id) = record["id"].as_str() else {
            continue;
        };
        if storage::project_id(id).is_err() {
            continue;
        }
        // A crash between folder staging and registry commit must RESTORE the
        // project, not silently erase data still listed in the registry.
        let restored = with_registry_lock(state, || {
            if load_projects(state)?
                .iter()
                .any(|project| project["id"].as_str() == Some(id))
            {
                let target = root.join(id);
                if target.exists() {
                    return Err(storage::conflict(
                        "Original project folder already exists during deletion recovery",
                    ));
                }
                fs::rename(&staged, &target)
                    .map_err(|e| storage::internal(format!("Restore interrupted delete: {e}")))?;
                return Ok(true);
            }
            Ok(false)
        });
        match restored {
            Ok(true) => continue,
            Err(error) => {
                eprintln!("Project deletion recovery {}: {error:?}", staged.display());
                continue;
            }
            Ok(false) => {}
        }
        let candidates = record["videos"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter(|video| storage::video_name(video).is_ok())
            .map(str::to_owned)
            .collect();
        cleanup_staged_project(state, staged, candidates);
    }
}

pub async fn create_project(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let writing = state.clone();
    let project = tokio::task::spawn_blocking(move || {
        create_record(
            &writing,
            request,
            #[cfg(test)]
            || {},
        )
    })
    .await
    .map_err(|err| storage::internal(format!("Create task failed: {err}")))??;
    // The project record is already durably committed: a preparation queue
    // failure must not trigger an HTTP retry that creates another project.
    if let Some(id) = project["id"].as_str() {
        let id = id.to_owned();
        tokio::spawn(async move {
            if let Err(error) = crate::preparation::enqueue_project(state, &id, false).await {
                eprintln!("Project created, but preparation queue failed: {error:?}");
            }
        });
    }
    Ok(Json(json!({"project": project})))
}

pub async fn update_project(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let writing = state.clone();
    let project = tokio::task::spawn_blocking(move || update_record(&writing, request))
        .await
        .map_err(|err| storage::internal(format!("Update task failed: {err}")))??;
    if let Some(id) = project["id"].as_str() {
        if let Err(error) = crate::preparation::enqueue_project(state, id, false).await {
            eprintln!("Project updated, but preparation queue failed: {error:?}");
        }
    }
    Ok(Json(json!({"project": project})))
}

pub(crate) fn scan_videos(state: &AppState) -> Result<Vec<Value>, ApiError> {
    let mut out = Vec::new();
    for (root, uploaded) in [
        (state.repository_root.join("project"), false),
        (uploaded_video_root(state), true),
    ] {
        let Ok(folder) = fs::read_dir(&root) else {
            continue;
        };
        let mut paths = folder
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file()
                    && fs::symlink_metadata(path).is_ok_and(|m| !m.file_type().is_symlink())
                    && path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .is_some_and(|name| storage::video_name(name).is_ok())
            })
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Ok(source) = canonical_source(state, name) else {
                continue;
            };
            if source != path.canonicalize().unwrap_or_default() {
                continue;
            }
            let size = source.metadata().map(|m| m.len()).unwrap_or(0);
            let display_name = if uploaded {
                storage::read_json(&root.join(format!("{name}.json")))?
                    .and_then(|meta| {
                        (meta["schema"] == "volley-project-upload/1"
                            && meta["name"] == name
                            && meta["size"] == size)
                            .then(|| meta["display_name"].as_str().map(str::to_owned))
                            .flatten()
                    })
                    .unwrap_or_else(|| name.to_owned())
            } else {
                name.to_owned()
            };
            let cache = latest_cache(state, name, None);
            out.push(json!({
                "name":name,
                "display_name":display_name,
                "origin":if uploaded {"uploaded"} else {"local"},
                "size_gb":((size as f64/1e9)*100.0).round()/100.0,
                "analyzed":cache.is_some(),
                "cache_id":cache
            }));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_rejects_video_removed_after_validation() {
        use std::process::Command;

        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let root = std::env::temp_dir().join(format!("volley-create-race-{}", Uuid::new_v4()));
        let state = AppState {
            repository_root: root.join("original"),
            data_root: root.join("sandbox"),
            writes_enabled: true,
            montage: crate::montage::MontageManager::new(),
            preparation: crate::preparation::PreparationManager::new(),
            preview_limit: tokio::sync::Semaphore::new(1),
            touch_limit: tokio::sync::Semaphore::new(1),
            continuity_limit: tokio::sync::Semaphore::new(1),
            reid_limit: tokio::sync::Semaphore::new(1),
            identity_limit: tokio::sync::Semaphore::new(1),
            statistics_limit: tokio::sync::Semaphore::new(1),
            marks_limit: tokio::sync::Semaphore::new(1),
            quick_reel: crate::quick_reel::QuickReelManager::new(),
            inference_limit: tokio::sync::Semaphore::new(1),
            full_inference: crate::inference_full::FullInferenceManager::new(),
        };
        let id = "123456abcdef";
        let video = "shared.mp4";
        let projects = project_root(&state);
        fs::create_dir_all(projects.join(id)).unwrap();
        fs::create_dir_all(uploaded_video_root(&state)).unwrap();
        let path = uploaded_video_root(&state).join(video);
        assert!(Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30:duration=0.5",
                "-c:v",
                "libx264",
                "-threads",
                "1"
            ])
            .arg(&path)
            .status()
            .unwrap()
            .success());
        storage::write_json_atomic(
            &registry_path(&state),
            &json!({"schema":1,"projects":[{"id":id,"name":"Old project","videos":[video]}]}),
        )
        .unwrap();

        let result = create_record(
            &state,
            json!({"name":"New project","videos":[video]}),
            || {
                let (staged, exclusive) =
                    with_registry_lock(&state, || delete_from_root(&projects, id)).unwrap();
                cleanup_staged_project(&state, staged, exclusive);
                assert!(!path.exists());
            },
        );
        let remaining = load_projects(&state).unwrap();
        fs::remove_dir_all(root).unwrap();

        assert!(result.is_err());
        assert!(remaining.is_empty());
    }

    #[test]
    fn delete_project_preserves_other_data() {
        let root = std::env::temp_dir().join(format!("volley-delete-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("123456abcdef")).unwrap();
        std::fs::create_dir_all(root.join("other")).unwrap();
        crate::storage::write_json_atomic(
            &root.join("projects.json"),
            &serde_json::json!({"schema":1,"projects":[{"id":"123456abcdef"},{"id":"other"}]}),
        )
        .unwrap();
        let (staged, _) = super::delete_from_root(&root, "123456abcdef").unwrap();
        assert!(!root.join("123456abcdef").exists());
        assert!(root.join("other").exists());
        let data = crate::storage::read_json(&root.join("projects.json"))
            .unwrap()
            .unwrap();
        assert_eq!(data["projects"].as_array().unwrap().len(), 1);
        std::fs::remove_dir_all(staged).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn deletion_returns_only_unshared_video_names() {
        let root =
            std::env::temp_dir().join(format!("volley-delete-links-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("123456abcdef")).unwrap();
        std::fs::create_dir_all(root.join("fedcba654321")).unwrap();
        crate::storage::write_json_atomic(
            &root.join("projects.json"),
            &serde_json::json!({"schema":1,"projects":[
                {"id":"123456abcdef","videos":["shared.mp4","exclusive.mp4"]},
                {"id":"fedcba654321","videos":["shared.mp4"]}
            ]}),
        )
        .unwrap();
        let (staged, exclusive) = super::delete_from_root(&root, "123456abcdef").unwrap();
        assert_eq!(exclusive, vec!["exclusive.mp4"]);
        assert!(staged.is_dir());
        assert!(root.join("fedcba654321").is_dir());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn normalized_project_name_collapses_whitespace_and_nfkc() {
        assert_eq!(normalized_name("  A  B  ").unwrap(), "A B");
        assert_eq!(normalized_name("Ｆｏｏ").unwrap(), "Foo");
        assert!(normalized_name(" \t ").is_err());
    }
}
