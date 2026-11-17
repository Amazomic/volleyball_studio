use crate::{
    montage,
    montage_material::{material_bindings, montage_material},
    native_rally_evidence::{
        generate, inspect_project, validate_ids, verified_camera_tracks, verified_manual_clips,
        Choice,
    },
    project_guard,
    storage::{self, ApiError, ApiResult},
    AppState,
};
use axum::{
    extract::{Query, State},
    http::HeaderMap,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewQuery {
    pub project_id: String,
    pub rally_run_id: String,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_gold_filter")]
    pub gold_filter: String,
    #[serde(default = "default_sort_by")]
    pub sort_by: String,
}
fn default_gold_filter() -> String {
    "all".into()
}
fn default_sort_by() -> String {
    "time".into()
}
fn default_limit() -> usize {
    24
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimelineQuery {
    pub project_id: String,
    pub rally_run_id: String,
}

/// Complete, compact whole-video index, without paged cards or frame overlays.
pub async fn timeline(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TimelineQuery>,
) -> ApiResult<Value> {
    let result = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        let run = inspect_project(&state, &query.project_id, &query.rally_run_id)?;
        montage_material(&state, &run)
    })
    .await
    .map_err(|e| storage::internal(format!("Native rally timeline task: {e}")))??;
    Ok(Json(result))
}

pub async fn preview(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PreviewQuery>,
) -> ApiResult<Value> {
    let result = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        // Immutable SHA-verified CPU snapshots are safe to read while another
        // operation holds the project write lock; preview must never acquire it.
        let run = inspect_project(&state, &query.project_id, &query.rally_run_id)?;
        let marked = crate::native_rally_marks::marked_for_run(
            &state,
            &query.project_id,
            &query.rally_run_id,
        )?;
        Ok(generate(
            &run,
            Choice::Page {
                offset: query.offset,
                limit: query.limit,
                filter: query.gold_filter,
                sort_by: query.sort_by,
                marked,
            },
        )
        .map_err(storage::bad_request)?
        .value)
    })
    .await
    .map_err(|e| storage::internal(format!("Native rally preview task: {e}")))??;
    Ok(Json(result))
}

/// Explicit manual-only choice from a *specific* immutable provisional
/// rally run. This does not enable automatic montage, and never accepts
/// client-provided trim offsets or arbitrary path strings.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManualMontageRequest {
    pub project_id: String,
    pub rally_run_id: String,
    pub candidate_ids: Vec<String>,
    #[serde(default)]
    pub audio_file: Option<String>,
    #[serde(default = "default_volume")]
    pub audio_volume: f64,
    #[serde(default)]
    pub encoder: Option<String>,
    #[serde(default)]
    pub layout: volley_media::montage::VideoLayout,
    #[serde(default)]
    pub target_duration_sec: Option<f64>,
    #[serde(default = "default_camera_zoom")]
    pub camera_zoom: f64,
}
fn default_camera_zoom() -> f64 {
    1.25
}
fn default_volume() -> f64 {
    40.0
}

pub async fn start_manual_montage(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ManualMontageRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    if !state.writes_enabled {
        return Err(storage::conflict(
            "Manual montage requires isolated project storage",
        ));
    }
    if !request.audio_volume.is_finite() || !(0.0..=100.0).contains(&request.audio_volume) {
        return Err(storage::bad_request("Audio volume outside 0–100 percent"));
    }
    validate_ids(&request.candidate_ids).map_err(storage::bad_request)?;
    let project_id = request.project_id.clone();
    let run_id = request.rally_run_id.clone();
    let ids = request.candidate_ids;
    let preparing = state.clone();
    let project_for_worker = project_id.clone();
    let run_for_worker = run_id.clone();
    let layout = request.layout;
    let (clips, camera_tracks) = tokio::task::spawn_blocking(move || -> Result<_, ApiError> {
        let _guard = project_guard::write_guard(&preparing, &project_for_worker)?;
        let run = inspect_project(&preparing, &project_for_worker, &run_for_worker)?;
        let clips = verified_manual_clips(&run, ids).map_err(storage::bad_request)?;
        let tracks = if layout == volley_media::montage::VideoLayout::VerticalFollow {
            Some(verified_camera_tracks(&preparing, &run, &clips).map_err(storage::bad_request)?)
        } else {
            None
        };
        Ok((clips, tracks))
    })
    .await
    .map_err(|e| storage::internal(format!("Manual rally planning task: {e}")))??;
    let materials =
        material_bindings(state.clone(), project_id.clone(), vec![run_id.clone()]).await?;
    let output = montage::start_native_montage(
        State(state),
        headers,
        Json(montage::MontageRequest {
            materials,
            project_id,
            clips,
            audio_file: request.audio_file,
            audio_volume: request.audio_volume,
            encoder: request.encoder,
            layout: request.layout,
            target_duration_sec: request.target_duration_sec,
            camera_zoom: request.camera_zoom,
            camera_tracks,
            verified_highlights: None,
            report_outro: None,
        }),
    )
    .await?;
    let Json(mut body) = output;
    body["source_provisional_rally_run_id"] = json!(run_id);
    body["manual_rally_selection"] = json!(true);
    body["authoritative_action_classification"] = json!(false);
    Ok(Json(body))
}
