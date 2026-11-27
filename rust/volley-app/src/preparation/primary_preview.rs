use crate::{
    playback, projects,
    storage::{self, ApiError},
    AppState,
};
use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, HeaderMap, HeaderValue, Response},
    response::IntoResponse,
    Json,
};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc, Mutex, OnceLock},
};
use volley_media::preparation::primary_preview::{scope_root, PrimaryPreviewPlan};

static ENCODING: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();

fn active_previews() -> &'static Mutex<HashSet<PathBuf>> {
    ENCODING.get_or_init(|| Mutex::new(HashSet::new()))
}

pub(crate) fn has_active() -> bool {
    active_previews()
        .lock()
        .map_or(true, |guard| !guard.is_empty())
}

struct ActivePreview(PathBuf);

impl Drop for ActivePreview {
    fn drop(&mut self) {
        if let Ok(mut active) = active_previews().lock() {
            active.remove(&self.0);
        }
    }
}

pub(crate) fn enqueue(
    state: Arc<AppState>,
    project: String,
    video: String,
    expected_source: Value,
) {
    if !state.writes_enabled {
        return;
    }
    let key = scope_root(&state.data_root, &project, &video);
    let Ok(mut running) = active_previews().lock() else {
        eprintln!("Primary preview queue unavailable: lock poisoned");
        return;
    };
    if !running.insert(key.clone()) {
        return;
    }
    drop(running);
    tokio::task::spawn_blocking(move || {
        let _guard = ActivePreview(key);
        let result = (|| -> Result<(), String> {
            let saved = projects::existing_project(&state, &project)
                .map_err(|error| format!("Preview project missing: {error:?}"))?;
            if !saved["videos"]
                .as_array()
                .is_some_and(|rows| rows.iter().any(|row| row.as_str() == Some(&video)))
            {
                return Err("Preview video no longer belongs to the project".into());
            }
            let source = projects::canonical_source(&state, &video)
                .map_err(|error| format!("Preview source missing: {error:?}"))?;
            let verified_source = volley_media::preparation::source_provenance(&source)?;
            if verified_source != expected_source {
                return Err("Preview source differs from completed GPU analysis".into());
            }
            let plan = PrimaryPreviewPlan::new(&source, &state.data_root, &project, &video)?;
            volley_media::preparation::primary_preview::encode_separately(
                &plan,
                &AtomicBool::new(false),
            )
        })();
        if let Err(error) = result {
            eprintln!("Independent primary preview unavailable: {error}");
        }
    });
}

pub(crate) async fn ready(
    state: Arc<AppState>,
    project: &str,
    video: &str,
) -> Result<Option<(PathBuf, String, String)>, ApiError> {
    let project = project.to_owned();
    let video = video.to_owned();
    tokio::task::spawn_blocking(move || {
        let saved = projects::existing_project(&state, &project)?;
        if !saved["videos"]
            .as_array()
            .is_some_and(|rows| rows.iter().any(|row| row.as_str() == Some(&video)))
        {
            return Err(storage::bad_request("Video does not belong to project"));
        }
        if !scope_root(&state.data_root, &project, &video).is_dir() {
            return Ok(None);
        }
        let source = projects::canonical_source(&state, &video)?;
        let plan = PrimaryPreviewPlan::new(&source, &state.data_root, &project, &video)
            .map_err(storage::conflict)?;
        Ok(plan
            .ready_media()
            .map(|(path, hash)| (path.to_path_buf(), plan.id.clone(), hash)))
    })
    .await
    .map_err(|e| storage::internal(format!("Primary preview lookup: {e}")))?
}

pub async fn get(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HashMap<String, String>>,
    headers: HeaderMap,
) -> Result<Response<Body>, ApiError> {
    let project = query
        .get("project_id")
        .ok_or_else(|| storage::bad_request("Missing project_id"))?;
    let video = query
        .get("video")
        .ok_or_else(|| storage::bad_request("Missing video"))?;
    let found = ready(state, project, video).await?;
    if query.get("media").is_some_and(|value| value == "1") {
        let (path, id, hash) =
            found.ok_or_else(|| storage::conflict("Primary preview is not ready"))?;
        if query.get("preview_id") != Some(&id)
            || query.get("media_sha").is_some_and(|sha| sha != &hash)
        {
            return Err(storage::conflict("Primary preview version changed"));
        }
        let mut response = playback::serve_mp4(path, &headers).await?;
        if query.get("media_sha") == Some(&hash) && response.status().is_success() {
            response.headers_mut().insert(
                header::CACHE_CONTROL,
                HeaderValue::from_static("private, max-age=60"),
            );
            response.headers_mut().insert(
                header::ETAG,
                HeaderValue::from_str(&format!("\"{hash}\""))
                    .map_err(|error| storage::internal(error.to_string()))?,
            );
        }
        return Ok(response);
    }
    let media_url = found.map(|(_, id, hash)| {
        format!(
            "/api/video-primary-preview?project_id={}&video={}&media=1&preview_id={id}&media_sha={hash}",
            percent_encoding::utf8_percent_encode(project, percent_encoding::NON_ALPHANUMERIC),
            percent_encoding::utf8_percent_encode(video, percent_encoding::NON_ALPHANUMERIC)
        )
    });
    Ok(Json(
        json!({"schema":"volley-primary-playback-status/1","project_id":project,"video":video,
        "state":if media_url.is_some(){"ready"}else{"missing"},"media_url":media_url}),
    )
    .into_response())
}
