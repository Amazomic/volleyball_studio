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

fn fill_legacy_track_counts(
    state: &AppState,
    run: &crate::native_rally_evidence::ValidatedRun,
    page: &mut Value,
) -> Result<(), ApiError> {
    let cards = page["items"]
        .as_array_mut()
        .ok_or_else(|| storage::bad_request("Invalid rally preview items"))?;
    if cards.iter().all(|card| card["stable_track_count"].is_u64()) {
        return Ok(());
    }
    let folder = crate::projects::project_root(state)
        .join(&run.project_id)
        .join("native_cpu_replays")
        .join(&run.replay_id);
    storage::safe_folder(&folder).map_err(storage::bad_request)?;
    let replay = storage::safe_file(&folder, "tracked_frames.jsonl", 64 * 1024 * 1024 * 1024)
        .map_err(storage::bad_request)?;
    let counts =
        crate::native_player_tracks::replay_track_counts(&replay, &run.replay_sha, run.frames)
            .map_err(storage::bad_request)?;
    for card in cards
        .iter_mut()
        .filter(|card| !card["stable_track_count"].is_u64())
    {
        let start = card["detected_start_frame"]
            .as_u64()
            .ok_or_else(|| storage::bad_request("Invalid rally start"))?;
        let end = card["detected_end_frame"]
            .as_u64()
            .ok_or_else(|| storage::bad_request("Invalid rally end"))?;
        let sample = counts
            .get(start as usize..=end as usize)
            .ok_or_else(|| storage::bad_request("Rally track count outside source"))?;
        card["stable_track_count"] = json!(volley_core::rally_reconstruction::stable_track_count(
            sample
        ));
    }
    Ok(())
}

fn fill_legacy_metadata(
    state: &AppState,
    run: &crate::native_rally_evidence::ValidatedRun,
    page: &mut Value,
) -> Result<(), ApiError> {
    page["source_replay_sha256"] = json!(run.replay_sha);
    let cards = page["items"]
        .as_array_mut()
        .ok_or_else(|| storage::bad_request("Invalid rally preview items"))?;
    let first_floor = |card: &Value| {
        card["surface_event_candidates"]
            .as_array()
            .and_then(|events| events.iter().find(|event| event["surface"] == "floor"))
            .cloned()
    };
    if cards.iter().all(|card| {
        card["metadata"].is_object()
            && first_floor(card).is_none_or(|event| event["outcome"] != "line_uncertain")
    }) {
        return Ok(());
    }
    let Some(snapshot) = crate::video_edits::court_snapshot(state, &run.project_id, &run.video)?
    else {
        return Ok(());
    };
    let Some(court) = snapshot.transform.as_ref() else {
        return Ok(());
    };
    let rally_folder = run
        .rally_file
        .parent()
        .ok_or_else(|| storage::bad_request("Missing rally directory"))?;
    let manifest_file = storage::safe_file(rally_folder, "manifest.json", 16 * 1024)
        .map_err(storage::bad_request)?;
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(manifest_file).map_err(|e| storage::internal(e.to_string()))?,
    )
    .map_err(|e| storage::bad_request(e.to_string()))?;
    if manifest["court_revision"].as_str() != Some(snapshot.revision.as_str()) {
        return Ok(());
    }
    for card in cards.iter_mut().filter(|card| card["metadata"].is_object()) {
        if let Some(event) = first_floor(card) {
            let position = serde_json::from_value(event["position"].clone()).ok();
            if let Some(outcome) = event["outcome"].as_str() {
                crate::cpu_rallies::rally_metadata::refresh_line_winner(
                    &mut card["metadata"],
                    outcome,
                    position,
                    court,
                );
            }
        }
    }
    if cards.iter().all(|card| card["metadata"].is_object()) {
        return Ok(());
    }
    let folder = crate::projects::project_root(state)
        .join(&run.project_id)
        .join("native_cpu_replays")
        .join(&run.replay_id);
    storage::safe_folder(&folder).map_err(storage::bad_request)?;
    let replay = storage::safe_file(&folder, "tracked_frames.jsonl", 64 * 1024 * 1024 * 1024)
        .map_err(storage::bad_request)?;
    for card in cards
        .iter_mut()
        .filter(|card| !card["metadata"].is_object())
    {
        let start = card["detected_start_frame"]
            .as_u64()
            .ok_or_else(|| storage::bad_request("Invalid rally start"))?;
        let end = card["detected_end_frame"]
            .as_u64()
            .ok_or_else(|| storage::bad_request("Invalid rally end"))?;
        let sides = crate::native_preview_overlay::team_sides::rally_sides(
            (&replay, &run.replay_sha, run.frames),
            (start, end),
            court,
            &snapshot.revision,
        )
        .map_err(storage::bad_request)?;
        let roster = if card["method"] == "primary_gpu_rally_mask_v1" {
            card["provisional_track_ids"]
                .as_array()
                .ok_or_else(|| storage::bad_request("Missing stable rally roster"))?
                .iter()
                .filter_map(Value::as_u64)
                .collect::<Vec<_>>()
        } else {
            sides.keys().copied().collect()
        };
        let floor = card["surface_event_candidates"]
            .as_array()
            .and_then(|events| events.iter().find(|event| event["surface"] == "floor"))
            .and_then(|event| {
                Some((
                    event["frame_index"].as_u64()?,
                    volley_core::rally_score::scoring_landing(
                        event["outcome"].as_str()?,
                        serde_json::from_value(event["position"].clone()).ok(),
                        Some(court),
                    ),
                ))
            });
        let mut observed = card["motion_events"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|event| {
                Some((
                    event["frame_index"].as_u64()?,
                    event["provisional_track_id"].as_u64(),
                    event["rally_context"]["estimated_kind"]
                        .as_str()
                        .or_else(|| event["action_prediction"]["kind"].as_str())
                        .unwrap_or("unknown_touch"),
                ))
            })
            .collect::<Vec<_>>();
        if let Some((at, "out")) = floor {
            if card["motion_event_preview_truncated"] == true
                || card["raw_motion_event_count"].as_u64() != card["motion_event_count"].as_u64()
            {
                observed.retain(|(frame, _, _)| *frame < at);
                observed.push((at.saturating_sub(1), None, "unknown_touch"));
            }
        }
        let mut metadata = crate::cpu_rallies::rally_metadata::build(
            start,
            end,
            &sides,
            &roster,
            &observed,
            floor,
            run.analysis_range.map_or(0.0, |range| range.start_sec),
        );
        metadata["rally_id"] = card["candidate_id"].clone();
        card["metadata"] = metadata;
    }
    Ok(())
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
        let mut page = generate(
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
        .value;
        if fill_legacy_track_counts(&state, &run, &mut page).is_err() {
            page["stable_track_count_status"] = json!("unavailable");
        }
        if fill_legacy_metadata(&state, &run, &mut page).is_err() {
            page["rally_metadata_status"] = json!("unavailable");
        }
        Ok(page)
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
    #[serde(default)]
    pub include_recognition_data: bool,
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
            preferred_rally_run_id: Some(run_id.clone()),
            project_id,
            clips,
            audio_file: request.audio_file,
            audio_volume: request.audio_volume,
            encoder: request.encoder,
            layout: request.layout,
            target_duration_sec: request.target_duration_sec,
            camera_zoom: request.camera_zoom,
            include_recognition_data: request.include_recognition_data,
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
