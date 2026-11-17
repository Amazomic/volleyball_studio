use crate::{
    playback, projects,
    storage::{self, ApiError},
    AppState,
};
use axum::{
    body::Body,
    extract::{Query, State},
    http::{HeaderMap, Response},
    response::IntoResponse,
    Json,
};
use serde_json::json;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use volley_media::preparation::primary_preview::{scope_root, PrimaryPreviewPlan};

pub(crate) async fn ready(
    state: Arc<AppState>,
    project: &str,
    video: &str,
) -> Result<Option<(PathBuf, String)>, ApiError> {
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
            .ready_path()
            .map(|path| (path.to_path_buf(), plan.id.clone())))
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
        let (path, id) = found.ok_or_else(|| storage::conflict("Primary preview is not ready"))?;
        if query.get("preview_id") != Some(&id) {
            return Err(storage::conflict("Primary preview version changed"));
        }
        return playback::serve_mp4(path, &headers).await;
    }
    let media_url = found.map(|(_, id)| {
        format!(
            "/api/video-primary-preview?project_id={}&video={}&media=1&preview_id={id}",
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
