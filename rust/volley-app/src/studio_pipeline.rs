//! Complete Studio analysis: one fresh pair of GPU runs followed by native CPU stages.
//! The profile and immutable output IDs are retained for each launch. No fake results.
use crate::{
    cpu_rallies, cpu_retrack, cpu_touches, inference_full, inference_fusion, preparation,
    project_guard, projects,
    storage::{self, ApiError, ApiResult},
    AppState,
};
use axum::{
    extract::{Query, State},
    http::HeaderMap,
    Json,
};
use inference_full::stage_queue::{acquire_gpu, video_turns, VideoTurn};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::Mutex;
use uuid::Uuid;
use volley_core::{
    rally_reconstruction::RallySettings,
    touch_pipeline::{TouchAlgorithm, TouchSettings},
    tracker::TrackingOptions,
};
use volley_inference::person::DetectorVariant;
#[cfg(test)]
use volley_inference::ModelKind;

const SCHEMA: &str = "volley-studio-pipeline/1";
const MAX_LOG_BYTES: u64 = 2 * 1024 * 1024;
const MAX_LOG_ROW: usize = 8192;
const MAX_LOG_PAGE: usize = 150;
const MAX_LOG_ENTRIES: usize = 4000;

/// Append-only structured diagnostics. Function calls are not shell commands;
/// external processes are explicitly labelled with their real executable/argv.
fn log_entry(
    path: &Path,
    level: &str,
    kind: &str,
    stage: &str,
    message: &str,
    call: Option<Value>,
) -> Result<(), String> {
    let file_path = path.with_file_name("events.jsonl");
    let before = match fs::symlink_metadata(&file_path) {
        Ok(meta) => {
            if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > MAX_LOG_BYTES {
                return Err("Unsafe or oversized pipeline log".into());
            }
            meta.len()
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => return Err(format!("Read pipeline log metadata: {e}")),
    };
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("Clock: {e}"))?
        .as_millis();
    let record = json!({
        "timestamp_ms":timestamp_ms,
        "level":level,"kind":kind,"stage":stage,"message":message,
        "call":call
    });
    let bytes = serde_json::to_vec(&record).map_err(|e| format!("Serialize log entry: {e}"))?;
    if bytes.len() > MAX_LOG_ROW || before + bytes.len() as u64 + 1 > MAX_LOG_BYTES {
        return Err("Pipeline log capacity exceeded".into());
    }
    let mut output = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)
        .map_err(|e| format!("Open pipeline log: {e}"))?;
    let mut line = bytes;
    line.push(b'\n');
    output
        .write_all(&line)
        .map_err(|e| format!("Write pipeline log: {e}"))
}
fn log_native(path: &Path, stage: &str, function: &str, arguments: Value) -> Result<(), String> {
    log_entry(
        path,
        "info",
        "invocation",
        stage,
        "Нативный вызов Rust",
        Some(json!({"type":"native_rust","name":function,"arguments":arguments})),
    )
}
fn log_event(path: &Path, stage: &str, level: &str, message: &str) -> Result<(), String> {
    log_entry(path, level, "event", stage, message, None)
}
#[derive(Deserialize)]
pub struct LogQuery {
    project_id: String,
    id: String,
    #[serde(default)]
    after: usize,
}
fn log_rows(state: &AppState, query: &LogQuery) -> ApiResult<Value> {
    if query.after > MAX_LOG_ENTRIES {
        return Err(storage::bad_request("Недопустимый курсор журнала"));
    }
    let root = folder(state, &query.project_id, false)?;
    let status_file = status_path(&root, &query.id)?;
    let status = read(&status_file)?;
    if status["schema"] != SCHEMA
        || status["project_id"] != query.project_id
        || status["id"] != query.id
    {
        return Err(storage::bad_request("Журнал принадлежит другому запуску"));
    }
    let log_file = status_file.with_file_name("events.jsonl");
    let bytes = match fs::symlink_metadata(&log_file) {
        Ok(info) => {
            if info.file_type().is_symlink() || !info.is_file() || info.len() > MAX_LOG_BYTES {
                return Err(storage::bad_request("Журнал анализа повреждён"));
            }
            fs::read(&log_file).map_err(|e| storage::internal(format!("Read pipeline log: {e}")))?
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(storage::internal(format!("Check pipeline log: {e}"))),
    };
    let mut entries = Vec::new();
    let mut total = 0usize;
    // An observer may read while the writer is flushing a final row.
    let complete_len = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
    for row in bytes[..complete_len].split(|byte| *byte == b'\n') {
        if row.is_empty() {
            continue;
        }
        if total >= MAX_LOG_ENTRIES {
            return Err(storage::bad_request("Слишком длинный журнал"));
        }
        if total >= query.after && entries.len() < MAX_LOG_PAGE {
            let value = serde_json::from_slice::<Value>(row)
                .map_err(|_| storage::bad_request("Нарушена структура журнала"))?;
            entries.push(value);
        }
        total += 1;
    }
    let offset = query.after.min(total) + entries.len();
    Ok(Json(json!({
        "schema":"volley-studio-pipeline-log/1","project_id":query.project_id,
        "id":query.id,"entries":entries,"next_offset":offset,
        "has_more":offset<total
    })))
}

pub async fn log(
    State(state): State<Arc<AppState>>,
    Query(query): Query<LogQuery>,
) -> ApiResult<Value> {
    tokio::task::spawn_blocking(move || log_rows(&state, &query))
        .await
        .map_err(|error| storage::internal(format!("Read pipeline log worker: {error}")))?
}

#[derive(Deserialize)]
pub struct HistoryQuery {
    project_id: String,
    #[serde(default)]
    offset: usize,
}

fn hidden_history(root: &Path) -> Result<BTreeSet<String>, ApiError> {
    let path = root.join("history_hidden.json");
    let meta = match fs::symlink_metadata(&path) {
        Ok(meta) => meta,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeSet::new()),
        Err(error) => return Err(storage::internal(format!("Read history metadata: {error}"))),
    };
    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 65536 {
        return Err(storage::bad_request("Повреждены настройки истории"));
    }
    let ids: BTreeSet<String> = serde_json::from_slice(
        &fs::read(path).map_err(|e| storage::internal(format!("Read history: {e}")))?,
    )
    .map_err(|_| storage::bad_request("Повреждены настройки истории"))?;
    if ids.len() > 1500 || ids.iter().any(|id| !valid_id(id)) {
        return Err(storage::bad_request("Повреждены настройки истории"));
    }
    Ok(ids)
}

fn collect_history_rows(
    state: &AppState,
    project_id: &str,
) -> Result<(PathBuf, BTreeSet<String>), ApiError> {
    let root = folder(state, project_id, false)?;
    if !root.exists() {
        return Ok((root, BTreeSet::new()));
    }
    let mut ids = BTreeSet::new();
    for entry in fs::read_dir(&root).map_err(|e| storage::internal(format!("Read history: {e}")))? {
        let entry = entry.map_err(|e| storage::internal(format!("Read history entry: {e}")))?;
        let id = entry.file_name().to_string_lossy().into_owned();
        if !valid_id(&id) {
            continue;
        }
        let meta = fs::symlink_metadata(entry.path())
            .map_err(|e| storage::internal(format!("Read history entry: {e}")))?;
        if meta.is_dir() && !meta.file_type().is_symlink() {
            ids.insert(id);
        }
        if ids.len() > 1500 {
            return Err(storage::bad_request("Слишком много запусков анализа"));
        }
    }
    Ok((root, ids))
}

fn persist_hidden_history(
    root: &Path,
    mut ids: BTreeSet<String>,
    active: Option<&str>,
) -> Result<usize, ApiError> {
    if !root.exists() {
        return Ok(0);
    }
    if let Some(active) = active {
        ids.remove(active);
    }
    storage::write_json_atomic(&root.join("history_hidden.json"), &json!(ids))
        .map_err(|e| storage::internal(format!("Save history visibility: {e:?}")))?;
    Ok(ids.len())
}

#[cfg(test)]
fn clear_history_rows(
    state: &AppState,
    project_id: &str,
    active: Option<&str>,
) -> Result<usize, ApiError> {
    let (root, ids) = collect_history_rows(state, project_id)?;
    persist_hidden_history(&root, ids, active)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClearHistoryRequest {
    project_id: String,
}

pub async fn clear_history(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ClearHistoryRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    if !state.writes_enabled {
        return Err(storage::conflict(
            "Очистка истории доступна только в рабочем хранилище",
        ));
    }
    storage::project_id(&request.project_id)?;
    let scan_state = state.clone();
    let scan_project = request.project_id.clone();
    let (root, mut ids) =
        tokio::task::spawn_blocking(move || collect_history_rows(&scan_state, &scan_project))
            .await
            .map_err(|e| storage::internal(format!("Scan pipeline history worker: {e}")))??;
    let slot = manager().lock().await;
    let _guard = project_guard::write_guard(&state, &request.project_id)?;
    let active = slot
        .as_ref()
        .filter(|run| run.project_id == request.project_id)
        .map(|run| run.id.as_str());
    if let Some(run) = slot
        .as_ref()
        .filter(|run| run.project_id == request.project_id)
    {
        ids.retain(|id| !run.contains_run(&request.project_id, id));
    }
    let count = persist_hidden_history(&root, ids, active)?;
    Ok(Json(
        json!({"project_id":request.project_id,"hidden":count}),
    ))
}

fn history_rows(state: &AppState, project_id: &str, offset: usize) -> Result<Value, ApiError> {
    if offset > 1500 {
        return Err(storage::bad_request("Недопустимая страница истории"));
    }
    let root = folder(state, project_id, false)?;
    if !root.exists() {
        return Ok(
            json!({"schema":"volley-studio-pipeline-history/1","project_id":project_id,"runs":[],"next_offset":0,"has_more":false}),
        );
    }
    let hidden = hidden_history(&root)?;
    let mut rows = Vec::new();
    let mut candidates = Vec::new();
    for entry in
        fs::read_dir(&root).map_err(|e| storage::internal(format!("List pipeline history: {e}")))?
    {
        let entry = entry.map_err(|e| storage::internal(format!("Read pipeline history: {e}")))?;
        let id = entry.file_name().to_string_lossy().into_owned();
        if !valid_id(&id) || hidden.contains(&id) {
            continue;
        }
        let Ok(meta) = fs::symlink_metadata(entry.path()) else {
            continue;
        };
        if meta.file_type().is_symlink() || !meta.is_dir() {
            continue;
        }
        let created = meta.created().or_else(|_| meta.modified()).ok();
        candidates.push((created, id));
        if candidates.len() > 1500 {
            return Err(storage::bad_request("Слишком много запусков анализа"));
        }
    }
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    let total = candidates.len();
    let next_offset = (offset + 25).min(total);
    for (_, id) in candidates.into_iter().skip(offset).take(25) {
        let Ok(path) = status_path(&root, &id) else {
            continue;
        };
        let Ok(status) = read(&path) else { continue };
        if status["schema"] != SCHEMA || status["project_id"] != project_id || status["id"] != id {
            continue;
        }
        rows.push(history_row(&path, &id, &status));
    }
    rows.sort_by(|a, b| {
        b["started_ms"]
            .as_u64()
            .cmp(&a["started_ms"].as_u64())
            .then_with(|| b["id"].as_str().cmp(&a["id"].as_str()))
    });
    let has_more = next_offset < total;
    Ok(
        json!({"schema":"volley-studio-pipeline-history/1","project_id":project_id,"runs":rows,
        "next_offset":next_offset,"has_more":has_more}),
    )
}

fn history_row(path: &Path, id: &str, status: &Value) -> Value {
    let log_path = path.with_file_name("events.jsonl");
    let records: Vec<Value> = match fs::symlink_metadata(&log_path) {
        Ok(meta)
            if meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= MAX_LOG_BYTES =>
        {
            fs::read(&log_path)
                .ok()
                .map(|bytes| {
                    let end = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
                    bytes[..end]
                        .split(|b| *b == b'\n')
                        .take(MAX_LOG_ENTRIES)
                        .filter_map(|line| serde_json::from_slice(line).ok())
                        .collect()
                })
                .unwrap_or_default()
        }
        _ => Vec::new(),
    };
    let started = records.first().and_then(|row| row["timestamp_ms"].as_u64());
    let finished =
        if ["complete", "failed", "cancelled"].contains(&status["status"].as_str().unwrap_or("")) {
            records.last().and_then(|row| row["timestamp_ms"].as_u64())
        } else {
            None
        };
    let starts: Vec<_> = records
        .iter()
        .filter(|row| row["kind"] == "stage_start")
        .filter_map(|row| Some((row["stage"].as_str()?, row["timestamp_ms"].as_u64()?)))
        .collect();
    let stages: Vec<Value> = starts
        .iter()
        .enumerate()
        .map(|(index, (name, start))| {
            let end = starts.get(index + 1).map(|(_, time)| *time).or(finished);
            json!({"name":name,"started_ms":start,"finished_ms":end,
                "duration_ms":end.and_then(|time| time.checked_sub(*start)),
                "active":end.is_none() && ["queued","running","cancelling"].contains(
                    &status["status"].as_str().unwrap_or(""))})
        })
        .collect();
    let stages = status["stages"]
        .as_array()
        .filter(|items| !items.is_empty())
        .cloned()
        .unwrap_or(stages);
    json!({"id":id,"video":status["video"],"mode":status["mode"],
        "status":status["status"],"stage":status["stage"],"error":status["error"],
        "started_ms":started,"duration_ms":started.zip(finished).and_then(|(a,b)| b.checked_sub(a)),
        "gpu_reused":status["gpu_reused"],"result_reused":status["result_reused"],
        "stage_id":status["stage_id"],"analysis_id":status["analysis_id"],
        "gpu_jobs":status["gpu_jobs"],"cpu_runs":status["cpu_runs"],
        "primary_tracking":status["primary_tracking"],
        "gpu_rally_count":status["gpu_rally_mask"]["merged"].as_array().map(Vec::len),
        "touch_detector_count":status["touch_detector_count"],
        "shared_pose":status["shared_pose"],
        "pose_frame_progress":status["pose_frame_progress"],
        "rally_stage_progress":status["rally_stage_progress"],
        "player_tracking_stats":status["player_tracking_stats"],
        "ball_stabilizer":status["ball_stabilizer"],
        "completion_quality":status["completion_quality"],
        "automatic_identity_available":status["automatic_identity"].is_object(),
        "profile":{"gpu":{
            "person_detector":status["profile"]["gpu"]["person_detector"],
            "pose_enabled":status["profile"]["gpu"]["pose_enabled"]
        }},
        "stages":stages})
}

fn archived_history_rows(
    state: &AppState,
    project_id: &str,
    offset: usize,
    active_ids: &BTreeSet<String>,
) -> Result<Value, ApiError> {
    if offset > 1500 {
        return Err(storage::bad_request("Invalid history page"));
    }
    let (root, ids) = collect_history_rows(state, project_id)?;
    let hidden = hidden_history(&root)?;
    let mut launches = Vec::new();
    for id in ids {
        if hidden.contains(&id) || active_ids.contains(&id) {
            continue;
        }
        let Ok(path) = status_path(&root, &id) else {
            continue;
        };
        let Ok(head) = read(&path) else { continue };
        if head["schema"] != SCHEMA || head["project_id"] != project_id || head["id"] != id {
            continue;
        }
        let members = if let Some(queue_id) = head["queue_id"].as_str() {
            if queue_id != id {
                continue;
            }
            let Ok(members) = read_queue_rows(&root, project_id, queue_id) else {
                continue;
            };
            if members
                .iter()
                .any(|row| row["id"].as_str().is_some_and(|id| active_ids.contains(id)))
            {
                continue;
            }
            members
        } else {
            vec![head]
        };
        let started = members
            .iter()
            .filter_map(|row| row["started_ms"].as_u64())
            .min();
        launches.push((started, id, members));
    }
    launches.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    let total = launches.len();
    let next_offset = (offset + 25).min(total);
    let mut rows = Vec::new();
    for (started, id, members) in launches.into_iter().skip(offset).take(25) {
        let mut videos = Vec::new();
        let mut finished = None;
        for mut status in members {
            if matches!(
                status["status"].as_str(),
                Some("queued" | "running" | "cancelling")
            ) {
                status["status"] = json!("interrupted");
            }
            if let Some(end) = status["finished_ms"].as_u64() {
                finished = Some(finished.map_or(end, |previous: u64| previous.max(end)));
            }
            let Some(member_id) = status["id"].as_str() else {
                continue;
            };
            let Ok(path) = status_path(&root, member_id) else {
                continue;
            };
            let mut row = history_row(&path, member_id, &status);
            if let Some(start) = status["started_ms"].as_u64() {
                row["started_ms"] = json!(start);
                if let Some(end) = status["finished_ms"].as_u64() {
                    row["duration_ms"] = json!(end.checked_sub(start));
                }
            }
            videos.push(row);
        }
        let Some(mut row) = videos.first().cloned() else {
            continue;
        };
        if row["id"].as_str() != Some(id.as_str()) {
            continue;
        }
        if videos.len() > 1 {
            let outcome = ["failed", "interrupted", "cancelled"]
                .into_iter()
                .find_map(|kind| videos.iter().find(|row| row["status"] == kind));
            if let Some(outcome) = outcome {
                row["status"] = outcome["status"].clone();
                row["error"] = outcome["error"].clone();
            }
            for quality in ["identity_failed", "identity_review_required"] {
                if videos
                    .iter()
                    .any(|row| row["completion_quality"] == quality)
                {
                    row["completion_quality"] = json!(quality);
                    break;
                }
            }
            row["started_ms"] = json!(started.or_else(|| videos
                .iter()
                .filter_map(|row| row["started_ms"].as_u64())
                .min()));
            row["duration_ms"] = json!(started
                .zip(finished)
                .and_then(|(start, end)| end.checked_sub(start)));
            row["result_reused"] = json!(videos.iter().all(|row| row["result_reused"] == true));
            row["gpu_reused"] = json!(videos.iter().all(|row| row["gpu_reused"] == true));
            row["videos"] = json!(videos);
        }
        rows.push(row);
    }
    Ok(
        json!({"schema":"volley-studio-pipeline-history/1","project_id":project_id,"runs":rows,
        "next_offset":next_offset,"has_more":next_offset < total}),
    )
}

pub async fn history(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HistoryQuery>,
) -> ApiResult<Value> {
    let slot = manager().lock().await;
    let active_ids = slot
        .as_ref()
        .filter(|entry| entry.project_id == query.project_id)
        .map(|entry| entry.queue_runs.iter().cloned().collect::<BTreeSet<_>>())
        .unwrap_or_default();
    let result = tokio::task::spawn_blocking(move || {
        archived_history_rows(&state, &query.project_id, query.offset, &active_ids)
    })
    .await
    .map_err(|e| storage::internal(format!("Pipeline history worker: {e}")))??;
    drop(slot);
    Ok(Json(result))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct GpuProfile {
    person_detector: DetectorVariant,
    parallel_rally_poses: bool,
    stride: usize,
    gpu_memory_limit_mb: u64,
    cpu_threads: usize,
    /// Run independent RTMPose GPU pose evidence for this fusion.
    pose_enabled: bool,
    /// RTMPose reads selected still frames or every video frame.
    #[serde(default)]
    pose_mode: crate::pose_inference::PoseInputMode,
    /// Shared sampling interval for person detection and RTMPose at 30 FPS.
    /// The historical API name pose_stride is retained for saved profiles.
    pose_stride: usize,
    touch_pose_stride: u64,
    background_pose_stride: u64,
    #[serde(default)]
    primary_rallies: inference_full::PrimaryRallySettings,
}
const fn default_interpolation_enabled() -> bool {
    true
}
impl Default for GpuProfile {
    fn default() -> Self {
        Self {
            person_detector: DetectorVariant::default(),
            parallel_rally_poses: true,
            stride: 9,
            gpu_memory_limit_mb: 0,
            cpu_threads: 0,
            pose_enabled: true,
            pose_mode: crate::pose_inference::PoseInputMode::Photo,
            pose_stride: volley_inference::person::PERSON_SAMPLE_PERIOD_FRAMES,
            touch_pose_stride: cpu_retrack::PoseSampling::default().touch_stride,
            background_pose_stride: cpu_retrack::PoseSampling::default().background_stride,
            primary_rallies: inference_full::PrimaryRallySettings::default(),
        }
    }
}
impl GpuProfile {
    fn pose_sampling(&self) -> cpu_retrack::PoseSampling {
        cpu_retrack::PoseSampling {
            touch_stride: self.touch_pose_stride,
            background_stride: self.background_pose_stride,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct PostProfile {
    #[serde(default = "default_interpolation_enabled")]
    interpolate_players: bool,
    #[serde(default = "default_interpolation_enabled")]
    interpolate_pose: bool,
    tracking: TrackingOptions,
    #[serde(default)]
    ball_stabilizer: volley_core::ball_interpolation::BallStabilizerSettings,
    touches: TouchSettings,
    rallies: RallySettings,
    #[serde(default = "default_pose_classification")]
    pose_classification: bool,
    #[serde(default = "default_auto_assign_threshold_percent")]
    auto_assign_threshold_percent: u8,
    #[serde(default)]
    reid_device: volley_inference::reid_openvino::ReidDevice,
    reid_buffer_mib: u32,
}
fn default_pose_classification() -> bool {
    true
}
fn default_auto_assign_threshold_percent() -> u8 {
    65
}
impl Default for PostProfile {
    fn default() -> Self {
        Self {
            interpolate_players: true,
            interpolate_pose: true,
            tracking: TrackingOptions::default(),
            ball_stabilizer: volley_core::ball_interpolation::BallStabilizerSettings::default(),
            touches: TouchSettings {
                algorithm: TouchAlgorithm::PoseMotion,
                require_pose: true,
                ..TouchSettings::default()
            },
            rallies: RallySettings::default(),
            pose_classification: true,
            auto_assign_threshold_percent: default_auto_assign_threshold_percent(),
            reid_device: Default::default(),
            reid_buffer_mib: crate::native_player_tracks::DEFAULT_REID_BUFFER_MIB,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PipelineMode {
    #[default]
    Full,
    Gpu,
    Post,
}
impl PipelineMode {
    /// CPU-only runs are allowed to consume saved GPU evidence, never to start a model.
    const fn allows_gpu_execution(self) -> bool {
        matches!(self, Self::Full | Self::Gpu)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Full => "Полный анализ",
            Self::Gpu => "Первичный GPU-анализ",
            Self::Post => "CPU-постобработка",
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(try_from = "StartRequestWire")]
pub struct StartRequest {
    project_id: String,
    video: String,
    #[serde(default)]
    gpu: GpuProfile,
    #[serde(default)]
    post: PostProfile,
    #[serde(default)]
    mode: PipelineMode,
    #[serde(default)]
    source_analysis_id: Option<String>,
    #[serde(default)]
    client_request_id: Option<String>,
    #[serde(default)]
    analysis_range: Option<volley_media::raw_decoder::AnalysisRange>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    video_profiles: BTreeMap<String, VideoAnalysisProfile>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StartRequestWire {
    project_id: String,
    video: String,
    #[serde(default = "empty_profile")]
    gpu: Value,
    #[serde(default = "empty_profile")]
    post: Value,
    #[serde(default)]
    mode: PipelineMode,
    #[serde(default)]
    source_analysis_id: Option<String>,
    #[serde(default)]
    client_request_id: Option<String>,
    #[serde(default)]
    video_profiles: BTreeMap<String, VideoProfileWire>,
}

#[derive(Clone, Debug, Serialize)]
struct VideoAnalysisProfile {
    gpu: GpuProfile,
    post: PostProfile,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VideoProfileWire {
    #[serde(default = "empty_profile")]
    gpu: Value,
    #[serde(default = "empty_profile")]
    post: Value,
}

fn decode_start_profile(gpu: Value, post: Value) -> Result<VideoAnalysisProfile, String> {
    let (gpu, mut post) = decode_profiles(gpu, post)?;
    gpu.primary_rallies.validate()?;
    post.ball_stabilizer.validate().map_err(str::to_owned)?;
    if !(1..=99).contains(&post.auto_assign_threshold_percent) {
        return Err("Порог сходства для автопривязки должен быть от 1 до 99%".into());
    }
    post.touches.algorithm = TouchAlgorithm::BallTrajectory;
    post.touches.require_pose = false;
    Ok(VideoAnalysisProfile { gpu, post })
}

impl StartRequest {
    fn scoped_to_video(&self, video: &str) -> Self {
        let mut scoped = self.clone();
        scoped.video = video.to_owned();
        if let Some(profile) = self.video_profiles.get(video) {
            scoped.gpu = profile.gpu.clone();
            scoped.post = profile.post.clone();
        }
        scoped.video_profiles.clear();
        scoped
    }

    fn queue_profiles_digest(&self) -> Option<String> {
        (!self.video_profiles.is_empty()).then(|| {
            format!(
                "{:x}",
                Sha256::digest(json!(self.video_profiles).to_string().as_bytes())
            )
        })
    }
}

fn validate_queue_profiles(
    request: &StartRequest,
    videos: &[String],
    all_videos: bool,
) -> Result<(), ApiError> {
    if request.video_profiles.is_empty() {
        return Ok(());
    }
    if !all_videos
        || request.video_profiles.len() != videos.len()
        || videos
            .iter()
            .any(|video| !request.video_profiles.contains_key(video))
    {
        return Err(storage::bad_request(
            "Укажите отдельный профиль каждого видео проекта",
        ));
    }
    let primary = request
        .video_profiles
        .get(&request.video)
        .ok_or_else(|| storage::bad_request("Нет профиля выбранного видео"))?;
    if json!(primary.gpu) != json!(request.gpu) || json!(primary.post) != json!(request.post) {
        return Err(storage::bad_request(
            "Профиль выбранного видео не совпадает с профилем запроса",
        ));
    }
    for video in videos {
        let scoped = request.scoped_to_video(video);
        validate_start_profile(&scoped)?;
    }
    Ok(())
}

fn empty_profile() -> Value {
    json!({})
}

fn decode_profiles(mut gpu: Value, mut post: Value) -> Result<(GpuProfile, PostProfile), String> {
    let gpu_fields = gpu.as_object_mut().ok_or("GPU profile must be an object")?;
    let post_fields = post
        .as_object_mut()
        .ok_or("CPU profile must be an object")?;
    for key in ["interpolate_players", "interpolate_pose"] {
        if let Some(value) = gpu_fields.remove(key) {
            if !value.is_boolean() {
                return Err(format!("Invalid legacy interpolation flag: {key}"));
            }
            post_fields.entry(key).or_insert(value);
        }
    }
    let post: PostProfile = serde_json::from_value(post).map_err(|e| e.to_string())?;
    if post.reid_buffer_mib > crate::native_player_tracks::MAX_REID_BUFFER_MIB {
        return Err("ReID crop buffer must be between 0 and 4096 MiB".into());
    }
    Ok((
        serde_json::from_value(gpu).map_err(|e| e.to_string())?,
        post,
    ))
}

impl TryFrom<StartRequestWire> for StartRequest {
    type Error = String;

    fn try_from(wire: StartRequestWire) -> Result<Self, Self::Error> {
        let VideoAnalysisProfile { gpu, post } = decode_start_profile(wire.gpu, wire.post)?;
        if wire.video_profiles.len() > 100 {
            return Err("Очередь проекта ограничена 100 видео".into());
        }
        let video_profiles = wire
            .video_profiles
            .into_iter()
            .map(|(video, profile)| {
                decode_start_profile(profile.gpu, profile.post).map(|profile| (video, profile))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(Self {
            project_id: wire.project_id,
            video: wire.video,
            gpu,
            post,
            mode: wire.mode,
            source_analysis_id: wire.source_analysis_id,
            client_request_id: wire.client_request_id,
            analysis_range: None,
            video_profiles,
        })
    }
}

#[derive(Deserialize)]
pub struct StatusQuery {
    project_id: String,
    id: Option<String>,
    queue_id: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CancelRequest {
    project_id: String,
    id: String,
}
struct Running {
    project_id: String,
    id: String,
    cancel: Arc<AtomicBool>,
    queue_runs: Vec<String>,
}
impl Running {
    fn contains_run(&self, project_id: &str, id: &str) -> bool {
        self.project_id == project_id
            && (self.id == id || self.queue_runs.iter().any(|run| run == id))
    }
}
fn manager() -> &'static Mutex<Option<Running>> {
    static MANAGER: OnceLock<Mutex<Option<Running>>> = OnceLock::new();
    MANAGER.get_or_init(|| Mutex::new(None))
}

pub(crate) async fn admission_guard() -> tokio::sync::MutexGuard<'static, ()> {
    static ADMISSION: OnceLock<Mutex<()>> = OnceLock::new();
    ADMISSION.get_or_init(|| Mutex::new(())).lock().await
}
pub(crate) async fn has_active() -> bool {
    manager().lock().await.is_some()
}

pub(crate) async fn has_active_for_project(project_id: &str) -> bool {
    manager()
        .lock()
        .await
        .as_ref()
        .is_some_and(|job| job.project_id == project_id)
}

/// Protect only artifacts used by the running studio pipeline; unrelated saved
/// analyses can be renamed or deleted without interrupting GPU/CPU processing.
fn status_uses_analysis(status: &Value, video: &str, artifact_id: &str) -> bool {
    if status["video"] != video {
        return false;
    }
    status["primary_tracking"]["replay_id"].as_str() == Some(artifact_id)
        || ["source_analysis_id", "analysis_id", "reused_from_id"]
            .iter()
            .any(|key| status[*key].as_str() == Some(artifact_id))
        || ["gpu_jobs", "cpu_runs"]
            .iter()
            .filter_map(|key| status[*key].as_object())
            .flat_map(|values| values.values())
            .any(|value| value.as_str() == Some(artifact_id))
}

pub(crate) async fn uses_analysis(
    state: &AppState,
    project_id: &str,
    video: &str,
    artifact_id: &str,
) -> Result<bool, ApiError> {
    let running_ids = {
        let slot = manager().lock().await;
        slot.as_ref()
            .filter(|job| job.project_id == project_id)
            .map(|job| {
                if job.queue_runs.is_empty() {
                    vec![job.id.clone()]
                } else {
                    job.queue_runs.clone()
                }
            })
    };
    let Some(running_ids) = running_ids else {
        return Ok(false);
    };
    let root = folder(state, project_id, false)?;
    let ids = running_ids.clone();
    let video = video.to_owned();
    let artifact = artifact_id.to_owned();
    let used = tokio::task::spawn_blocking(move || {
        for id in ids {
            let status = read(&status_path(&root, &id)?)?;
            if status_uses_analysis(&status, &video, &artifact) {
                return Ok::<_, ApiError>(true);
            }
        }
        Ok(false)
    })
    .await
    .map_err(|e| storage::internal(format!("Read active pipeline worker: {e}")))??;
    let slot = manager().lock().await;
    if !slot.as_ref().is_some_and(|job| {
        job.project_id == project_id
            && (job.queue_runs == running_ids
                || (job.queue_runs.is_empty() && running_ids == [job.id.clone()]))
    }) {
        return Ok(true);
    }
    Ok(used)
}

fn reason(error: ApiError) -> String {
    error.1 .0["error"]
        .as_str()
        .unwrap_or("Ошибка нативной обработки")
        .to_owned()
}
pub(crate) use volley_core::contracts::job_id as valid_id;
fn folder(state: &AppState, project_id: &str, create: bool) -> Result<PathBuf, ApiError> {
    storage::project_id(project_id)?;
    projects::existing_project(state, project_id)?;
    let project = projects::project_root(state).join(project_id);
    storage::safe_folder(&project).map_err(storage::bad_request)?;
    let root = project.join("native_studio_pipelines");
    match fs::symlink_metadata(&root) {
        Ok(info) => {
            if info.file_type().is_symlink() || !info.is_dir() {
                return Err(storage::bad_request("Каталог запусков небезопасен"));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && create => {
            fs::create_dir(&root)
                .map_err(|e| storage::internal(format!("Создание каталога запусков: {e}")))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(storage::internal(format!(
                "Проверка каталога запусков: {error}"
            )))
        }
    }
    Ok(root)
}
fn save(path: &Path, status: &Value) -> Result<(), String> {
    storage::write_json_atomic(path, status).map_err(|e| format!("Сохранение статуса: {e:?}"))
}
fn read(path: &Path) -> Result<Value, ApiError> {
    let meta =
        fs::symlink_metadata(path).map_err(|_| storage::not_found("Статус анализа не найден"))?;
    if meta.file_type().is_symlink() || !meta.is_file() {
        return Err(storage::bad_request("Файл статуса анализа повреждён"));
    }
    storage::read_json(path)?.ok_or_else(|| storage::not_found("Пустой статус анализа"))
}
fn status_path(root: &Path, id: &str) -> Result<PathBuf, ApiError> {
    if !valid_id(id) {
        return Err(storage::bad_request("Неверный идентификатор запуска"));
    }
    let run = root.join(id);
    storage::safe_folder(&run).map_err(storage::bad_request)?;
    Ok(run.join("status.json"))
}
fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn finish_timed_stage(status: &mut Value) {
    let now = epoch_ms();
    if let Some(last) = status["stages"]
        .as_array_mut()
        .and_then(|rows| rows.last_mut())
    {
        if last["finished_ms"].is_null() {
            if let Some(start) = last["started_ms"].as_u64() {
                last["duration_ms"] = json!(now.saturating_sub(start));
                last["finished_ms"] = json!(now);
            }
        }
    }
    status["finished_ms"] = json!(now);
}

fn pipeline_stage_id(name: &str) -> &'static str {
    if name.starts_with("Ожидание очереди") {
        "other"
    } else if name.contains("Проверка исходного видео")
        || name.contains("Подготовка исходного видео")
    {
        "video"
    } else if name.contains("Ball Stabilizer") || name == "ball_stabilizer" {
        "ball_stabilizer"
    } else if matches!(
        name,
        "Ground Detector" | "ground_detector" | "Rally Detector"
    ) {
        "rallies"
    } else if name == "Touch Detector" {
        "touch_detector"
    } else if name.contains("Поиск розыгрышей") {
        "rallies"
    } else if name.contains("VBallNet") && name.contains("GPU") {
        "ball"
    } else if name.contains("Связывание рамок игроков") {
        "boxes"
    } else if name.contains("RTMPose") && !name.contains("RF-DETR") {
        "pose"
    } else if name.contains("RF-DETR") || name.contains("рамки игроков") {
        if name.starts_with("Объединение") {
            "fusion"
        } else {
            "players"
        }
    } else if name.contains("общий поток") {
        "ball"
    } else if name.contains("Объединение") || name.contains("GPU-анализ завершён")
    {
        "fusion"
    } else if name.contains("Track ID") {
        "tracking"
    } else if matches!(name, "Touch Attribution" | "Pose Geometry")
        || name.contains("касания в первичных")
        || name.contains("физические касания")
    {
        "touches"
    } else if name.contains("Публикация розыгрышей") {
        "publish"
    } else if name.contains("автоматическая привязка") {
        "identity"
    } else if name.contains("Монтажная временная разметка") {
        "material"
    } else if name.contains("сохранённые версии")
        || name.contains("Сохранённый GPU-анализ")
        || name.contains("сохранённый GPU-анализ")
    {
        "cache"
    } else {
        "other"
    }
}

fn progress(path: &Path, status: &mut Value, stage: &str, percent: f64) -> Result<(), String> {
    progress_with_frames(path, status, stage, percent, None)
}

fn progress_with_frames(
    path: &Path,
    status: &mut Value,
    stage: &str,
    percent: f64,
    frames: Option<(u64, u64)>,
) -> Result<(), String> {
    let stage_name = stage.split_once(": ").map_or(stage, |(name, _)| name);
    let stage_id = pipeline_stage_id(stage_name);
    let previous = status["stage"].as_str().unwrap_or("");
    let previous_name = previous.split_once(": ").map_or(previous, |(name, _)| name);
    let now = epoch_ms();
    if stage_id != pipeline_stage_id(previous_name)
        || (stage_id == "other" && stage_name != previous_name)
    {
        log_entry(
            path,
            "info",
            "stage_start",
            stage_name,
            "Начало этапа",
            None,
        )?;
        if !status["stages"].is_array() {
            status["stages"] = json!([]);
        }
        if let Some(rows) = status["stages"].as_array_mut() {
            if let Some(last) = rows.last_mut() {
                if last["finished_ms"].is_null() {
                    let start = last["started_ms"].as_u64().unwrap_or(now);
                    last["finished_ms"] = json!(now);
                    last["duration_ms"] = json!(now.saturating_sub(start));
                }
            }
            rows.push(json!({"name":stage_name,"id":stage_id,"started_ms":now,
                "finished_ms":null,"duration_ms":null}));
        }
        status["stage_samples"] = json!([]);
    }
    // A bounded rate window is useful for GPU stages; do not project an
    // entire mixed CPU/GPU pipeline from its global percentage.
    let samples = status["stage_samples"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let previous_sample = samples.last();
    if previous_sample.is_none_or(|last| {
        now.saturating_sub(last["timestamp_ms"].as_u64().unwrap_or(0)) >= 2500
            && (percent - last["percent"].as_f64().unwrap_or(percent)).abs() >= 0.05
    }) {
        let mut updated = samples;
        updated.push(json!({"timestamp_ms":now,"percent":percent}));
        if updated.len() > 45 {
            updated.drain(..updated.len() - 45);
        }
        status["stage_samples"] = json!(updated);
    }
    status["status"] = json!("running");
    status["stage"] = json!(stage);
    status["stage_id"] = json!(stage_id);
    status["progress_percent"] = json!(percent.clamp(0.0, 99.9));
    if let Some((processed, total)) = frames.filter(|(_, total)| *total > 0) {
        if let Some(last) = status["stages"]
            .as_array_mut()
            .and_then(|stages| stages.last_mut())
        {
            last["frames_processed"] = json!(processed.min(total));
            last["frames_total"] = json!(total);
            last["progress_percent"] = json!(100.0 * processed.min(total) as f64 / total as f64);
        }
    }
    save(path, status)
}
fn tracking_progress(
    path: &Path,
    status: &mut Value,
    mode: PipelineMode,
    event: cpu_retrack::TrackingProgress,
) -> Result<(), String> {
    match event {
        cpu_retrack::TrackingProgress::BoxesReady(total) => {
            status["player_tracking_stats"]["total_track_ids"] = json!(total);
            save(path, status)
        }
        cpu_retrack::TrackingProgress::PoseSelection => {
            progress(path, status, "Touch Detector", 87.9)
        }
        cpu_retrack::TrackingProgress::PoseCached(total) => {
            status["pose_frame_progress"] =
                json!({"processed":total,"total":total,"basis":"stable_crop_frames_v1"});
            save(path, status)
        }
        cpu_retrack::TrackingProgress::PoseFrames { processed, total } => {
            status["pose_frame_progress"] =
                json!({"processed":processed,"total":total,"basis":"stable_crop_frames_v1"});
            progress_with_frames(
                path,
                status,
                "GPU · RTMPose по стабилизированным рамкам",
                88.0 + 0.9 * processed as f64 / total.max(1) as f64,
                Some((processed, total)),
            )
        }
        cpu_retrack::TrackingProgress::FinalPass => {
            if mode == PipelineMode::Full {
                progress(path, status, "CPU · Track ID", 89.0)
            } else {
                Ok(())
            }
        }
    }
}

fn shared_pose_metrics(status: &Value, pose: &Value) -> Value {
    let player = status["stages"]
        .as_array()
        .and_then(|stages| stages.iter().rev().find(|stage| stage["id"] == "players"))
        .cloned()
        .unwrap_or(Value::Null);
    json!({
        "source_stage_id": "players",
        "sampled_frames": pose["sampled_frames"],
        "eligible_pose_frames": pose["eligible_pose_frames"],
        "inferred_pose_frames": pose["inferred_pose_frames"],
        "player_poses": pose["player_poses"],
        "started_ms": player["started_ms"],
        "finished_ms": player["finished_ms"],
        "duration_ms": player["duration_ms"]
    })
}

fn publish_primary_rally_mask(
    path: &Path,
    status: &mut Value,
    mask: &Value,
    decoded_frames: u64,
) -> Result<(), String> {
    let detected = mask["detected"]
        .as_array()
        .ok_or("GPU rally mask has no detected intervals")?
        .len();
    let merged = mask["merged"]
        .as_array()
        .ok_or("GPU rally mask has no merged intervals")?
        .len();
    let selected = mask["selected_frames"]
        .as_u64()
        .filter(|frames| *frames <= decoded_frames)
        .ok_or("GPU rally mask has invalid selected frame count")?;
    let fallback = mask["fallback_reason"].as_str().unwrap_or("none");
    progress_with_frames(
        path,
        status,
        "Поиск розыгрышей",
        8.0 + 72.0 * 0.3999995,
        Some((decoded_frames, decoded_frames)),
    )?;
    status["gpu_rally_mask"] = mask.clone();
    status["analysis_frames_processed"] = json!(decoded_frames);
    status["ground_detector"] = mask["ground_detector"].clone();
    save(path, status)?;
    log_event(
        path,
        "Поиск розыгрышей",
        "success",
        &format!(
            "Исходных={detected}, объединённых={merged}, выбранных кадров={selected}/{decoded_frames}, доля для RF-DETR={:.1}%, fallback={fallback}",
            100.0 * selected as f64 / decoded_frames.max(1) as f64,
        ),
    )
}
fn parse_primary_rally_scope(
    bytes: &[u8],
    ball_id: &str,
    status: &Value,
) -> Result<(cpu_touches::PrimaryRallyScope, Value), String> {
    let evidence: Value = serde_json::from_slice(bytes)
        .map_err(|_| "Повреждена сохранённая маска розыгрышей первичного анализа")?;
    let mask = &evidence["mask"];
    let frames = mask["source_frame_count"]
        .as_u64()
        .filter(|frames| *frames > 0 && *frames <= 48 * 3600 * 30)
        .ok_or("Некорректная длина первичного анализа розыгрышей")?;
    let intervals: Vec<(u64, u64)> = serde_json::from_value(
        mask.get("rally_intervals")
            .unwrap_or(&mask["merged"])
            .clone(),
    )
    .map_err(|_| "Некорректные границы первичных розыгрышей")?;
    let short_minimum = if mask["ground_boundary_version"] == "ground_ends_rally_v6"
        && mask["serve_flight_geometry_available"] == true
    {
        crate::cpu_retrack::ball_track::MIN_CONFIRMED_GROUNDED_RALLY_FRAMES
    } else {
        crate::cpu_retrack::ball_track::MIN_RALLY_FRAMES
    };
    if intervals
        .iter()
        .any(|&(start, end)| start > end || end >= frames || end - start + 1 < short_minimum)
        || intervals.windows(2).any(|pair| pair[0].1 >= pair[1].0)
        || !crate::cpu_retrack::ball_track::movement_filter_supported(
            mask["movement_filter_version"].as_str(),
        )
        || mask["min_rally_seconds"] != 5.0
        || evidence["schema"] != "volley-primary-rally-mask-evidence/1"
        || evidence["ball_job_id"] != ball_id
        || evidence["original_source"] != status["original_source"]
        || evidence["analysis_range"] != status["analysis_range"]
        || evidence["selected_frames"] != mask["selected_frames"]
        || mask["schema"] != "volley-primary-rally-mask/1"
    {
        return Err("Первичные розыгрыши не соответствуют сохранённому GPU-анализу".into());
    }
    Ok((
        cpu_touches::PrimaryRallyScope {
            intervals,
            source_frames: frames,
            ball_job_id: ball_id.to_owned(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
        },
        mask.clone(),
    ))
}

fn load_primary_rally_scope(
    app: &AppState,
    project_id: &str,
    status: &Value,
) -> Result<(cpu_touches::PrimaryRallyScope, Value), String> {
    let ball_id = status["gpu_jobs"]["ball"]
        .as_str()
        .filter(|id| valid_id(id))
        .ok_or("Сохранённый первичный анализ не содержит VBallNet ID")?;
    let root = projects::project_root(app)
        .join(project_id)
        .join("native_gpu_runs")
        .join(ball_id);
    let file = storage::safe_file(&root, "rally_mask.json", 64 * 1024 * 1024)
        .map_err(|_| "Для этого GPU-анализа нет первичных розыгрышей: повторите первичный анализ с включённым поиском по мячу")?;
    let bytes = fs::read(file).map_err(|e| format!("Чтение первичных розыгрышей: {e}"))?;
    parse_primary_rally_scope(&bytes, ball_id, status)
}

fn require_id(response: &Value, field: &str) -> Result<String, String> {
    response[field]
        .as_str()
        .filter(|value| valid_id(value))
        .map(str::to_owned)
        .ok_or_else(|| format!("Этап не вернул корректный {field}"))
}
const MAX_SOURCE_MANIFEST: u64 = 32 * 1024;
const MAX_SOURCE_EVIDENCE: u64 = 64 * 1024 * 1024 * 1024;
const MAX_PIPELINES: usize = 1000;

/// Version branching is based on the entire GPU profile AND the immutable
/// prepared-media identity. A change confined to CPU settings only branches
/// under the existing fused GPU source. Metadata is never sufficient for
/// claims of valid raw evidence: CPU processors verify SHA-256 when reading.
#[derive(Default)]
struct ExistingVersions {
    gpu: Option<Value>,
    full: Option<Value>,
}
fn saved_stage(project: &Path, directory: &str, id: &str, file: &str) -> Option<Value> {
    if !valid_id(id) {
        return None;
    }
    let parent = project.join(directory);
    storage::safe_folder(&parent).ok()?;
    let dir = parent.join(id);
    storage::safe_folder(&dir).ok()?;
    let manifest_file = storage::safe_file(&dir, "manifest.json", MAX_SOURCE_MANIFEST).ok()?;
    let manifest: Value = serde_json::from_slice(&fs::read(manifest_file).ok()?).ok()?;
    let evidence = storage::safe_file(&dir, file, MAX_SOURCE_EVIDENCE).ok()?;
    if fs::metadata(evidence).ok()?.len() == 0 || !valid_sha(&manifest["sha256"]) {
        return None;
    }
    Some(manifest)
}
fn valid_sha(value: &Value) -> bool {
    value.as_str().is_some_and(volley_core::contracts::sha256)
}

fn shared_pose_source(pose: &Value) -> bool {
    matches!(
        pose["decode_pipeline"].as_str(),
        Some("gpu_resident_nvdec_primary_v1" | "shared_primary_native_rgb_v2")
    )
}
fn gpu_lineage_valid(
    project: &Path,
    request: &StartRequest,
    identity: &Value,
    saved: &Value,
) -> bool {
    let Some(fusion_id) = saved["analysis_id"].as_str().filter(|id| valid_id(id)) else {
        return false;
    };
    let Some(ball_id) = saved["gpu_jobs"]["ball"].as_str().filter(|id| valid_id(id)) else {
        return false;
    };
    if saved["gpu_rally_mask"]["schema"] != "volley-primary-rally-mask/1"
        || saved["gpu_rally_mask"]["fallback_reason"]
            .as_str()
            .is_some()
    {
        return false;
    }
    if saved["gpu_rally_mask"]["schema"] == "volley-primary-rally-mask/1"
        && (saved["gpu_rally_mask"]["min_trajectory_seconds"]
            != json!(request.gpu.primary_rallies.min_trajectory_seconds)
            || saved["gpu_rally_mask"]["merge_gap_seconds"]
                != json!(request.gpu.primary_rallies.merge_gap_seconds)
            || saved["gpu_rally_mask"]["stationary_ball_ms"]
                != json!(request.gpu.primary_rallies.stationary_ball_ms)
            || (request.mode.allows_gpu_execution()
                && (saved["gpu_rally_mask"]["padding_before_ms"]
                    != json!(request.gpu.primary_rallies.padding_before_ms)
                    || saved["gpu_rally_mask"]["padding_after_ms"]
                        != json!(request.gpu.primary_rallies.padding_after_ms)
                    || !saved["gpu_rally_mask"]["rally_intervals"].is_array()))
            || !crate::cpu_retrack::ball_track::movement_filter_supported(
                saved["gpu_rally_mask"]["movement_filter_version"].as_str(),
            )
            || (request.mode.allows_gpu_execution()
                && saved["gpu_rally_mask"]["movement_filter_version"]
                    != crate::cpu_retrack::ball_track::MOVEMENT_FILTER_VERSION)
            || saved["gpu_rally_mask"]["min_rally_seconds"] != 5.0)
    {
        return false;
    }
    if request.mode.allows_gpu_execution()
        && (saved["gpu_rally_mask"]["ball_stabilizer"]["algorithm"]
            != volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION
            || saved["gpu_rally_mask"]["ball_stabilizer"]["settings"]
                != json!(request.post.ball_stabilizer)
            || saved["gpu_rally_mask"]["serve_flight_version"]
                != volley_core::rally_reconstruction::serve_flight::SERVE_FLIGHT_VERSION
            || saved["gpu_rally_mask"]["ground_boundary_version"] != "ground_ends_rally_v6"
            || saved["gpu_rally_mask"]["ground_detector"]["version"]
                != volley_core::motion::GROUND_DETECTOR_VERSION
            || !valid_sha(&saved["gpu_rally_mask"]["ground_detector"]["sha256"])
            || !saved["gpu_rally_mask"]["ball_stabilizer"]["trajectory_sha256"]
                .as_str()
                .is_some_and(volley_core::contracts::sha256))
    {
        return false;
    }
    let Some(players_id) = saved["gpu_jobs"]["players"]
        .as_str()
        .filter(|id| valid_id(id))
    else {
        return false;
    };
    let Some(fusion) = saved_stage(project, "native_gpu_fusions", fusion_id, "frames.jsonl") else {
        return false;
    };
    if fusion["schema"] != "volley-native-gpu-fusion/1"
        || fusion["state"] != "raw_candidates_fused"
        || fusion["fusion_id"] != fusion_id
        || fusion["project_id"] != request.project_id
        || fusion["video"] != request.video
        || fusion["source_identity"] != *identity
        || fusion["analysis_range"] != json!(request.analysis_range)
        || fusion["sources"]["vballnet"]["job_id"] != ball_id
        || fusion["sources"]["person_pose"]["job_id"] != players_id
        || fusion["player_sampling_period_frames"] != request.gpu.pose_stride
    {
        return false;
    }
    for (id, model, source) in [
        (ball_id, "ball", "vballnet"),
        (players_id, "players", "person_pose"),
    ] {
        let Some(raw) = saved_stage(project, "native_gpu_runs", id, "detections.jsonl") else {
            return false;
        };
        if !volley_core::contracts::gpu_manifest(
            &raw,
            &request.project_id,
            &request.video,
            id,
            model,
        ) || raw["source_identity"] != *identity
            || raw["analysis_range"] != json!(request.analysis_range)
            || raw["stride"] != request.gpu.stride
            || (model == "players"
                && raw["person_sampling_period_frames"] != request.gpu.pose_stride)
            || raw["sha256"] != fusion["sources"][source]["sha256"]
        {
            return false;
        }
    }
    if request.gpu.pose_enabled
        && saved["pose_pipeline_policy"] != cpu_retrack::POSE_REFINEMENT_POLICY
    {
        let Some(pose_id) = saved["gpu_jobs"]["pose"].as_str().filter(|id| valid_id(id)) else {
            return false;
        };
        let Some(pose) = saved_stage(project, "native_gpu_pose_runs", pose_id, "poses.jsonl")
        else {
            return false;
        };
        if pose["schema"] != "volley-native-gpu-pose/1"
            || pose["state"] != "pose_evidence_complete"
            || pose["pose_run_id"] != pose_id
            || pose["fusion_id"] != fusion_id
            || pose["model_sha256"] != volley_inference::pose::MODEL_SHA256
            || pose["stride"]
                != request
                    .gpu
                    .pose_mode
                    .effective_stride(request.gpu.pose_stride)
            || pose["pose_mode"].as_str().unwrap_or("photo") != request.gpu.pose_mode.name()
            || pose["project_id"] != request.project_id
            || pose["video"] != request.video
            || pose["source_identity"] != *identity
            || pose["source_fusion_sha256"] != fusion["sha256"]
            || !shared_pose_source(&pose)
        {
            return false;
        }
        if saved["gpu_model_stack"] == requested_gpu_model_stack(request.gpu.person_detector)
            && saved["zero_copy_request"].as_bool().unwrap_or(false) == zero_copy_requested()
            && (!matches!(
                pose["box_source_version"].as_str(),
                Some(
                    "detector_observed_sampling_v1"
                        | "detector_observed_and_bounded_interpolated_v1"
                )
            ) || pose["source_person_sampling_period_frames"] != request.gpu.pose_stride
                || pose["interpolate_players"] != false
                || pose["stride"] != request.gpu.pose_stride
                || !pose["crop_plan_file"].is_null()
                || pose["pixel_source_version"]
                    != if saved["zero_copy_active"] == true {
                        "original_nv12_cuda_v1"
                    } else {
                        "original_full_resolution_rgb24_v1"
                    }
                || pose["coordinate_system"] != "normalized_original_display_v1")
        {
            return false;
        }
    }
    true
}
fn cpu_lineage_valid(
    project: &Path,
    request: &StartRequest,
    saved: &Value,
    tcn_sha: Option<&str>,
) -> bool {
    let (Some(fusion), Some(replay), Some(touch), Some(rally)) = (
        saved["analysis_id"].as_str().filter(|id| valid_id(id)),
        saved["cpu_runs"]["tracking"]
            .as_str()
            .filter(|id| valid_id(id)),
        saved["cpu_runs"]["touches"]
            .as_str()
            .filter(|id| valid_id(id)),
        saved["postprocessing_id"]
            .as_str()
            .filter(|id| valid_id(id)),
    ) else {
        return false;
    };
    if saved["cpu_runs"]["rallies"] != rally {
        return false;
    }
    let Some(track) = saved_stage(
        project,
        "native_cpu_replays",
        replay,
        "tracked_frames.jsonl",
    ) else {
        return false;
    };
    let Some(touches) = saved_stage(
        project,
        "native_cpu_touch_runs",
        touch,
        "touch_candidates.jsonl",
    ) else {
        return false;
    };
    let Some(rallies) = saved_stage(
        project,
        "native_cpu_rallies",
        rally,
        "rally_candidates.jsonl",
    ) else {
        return false;
    };
    let Some(artifact_id) = touches["ball_stabilizer"]["artifact_id"]
        .as_str()
        .filter(|id| valid_id(id))
    else {
        return false;
    };
    let Some(trajectory) = saved_stage(
        project,
        "native_cpu_ball_stabilizer",
        artifact_id,
        "trajectory.jsonl",
    ) else {
        return false;
    };
    if trajectory["schema"] != "volley-native-stabilized-ball-trajectory/1"
        || trajectory["algorithm_version"]
            != volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION
        || trajectory["state"] != "complete"
        || trajectory["artifact_id"] != artifact_id
        || trajectory["file"] != "trajectory.jsonl"
        || trajectory["sha256"] != touches["ball_stabilizer"]["trajectory_sha256"]
        || trajectory["source_replay_id"] != replay
        || trajectory["source_replay_sha256"] != touches["source_replay_sha256"]
        || trajectory["settings"] != touches["ball_stabilizer"]["settings"]
        || trajectory["frames"] != touches["ball_stabilizer"]["frames"]
        || touches["ball_stabilizer"]["trajectory_file"] != "trajectory.jsonl"
    {
        return false;
    }

    track["schema"] == "volley-native-cpu-tracking/1"
        && track["state"] == "provisional_tracklets_rebuilt"
        && track["interpolated_positions_version"] == (if request.post.interpolate_players {
            "bounded_same_track_3s_v2"
        } else {
            "disabled_v1"
        })
        // A saved replay from the disabled legacy tracker must never be
        // silently reused when ByteTrack is now the default.
        && track["track_association_version"] == cpu_retrack::TRACK_ASSOCIATION_VERSION
        && track["court_revision"] == saved["court_revision"]
        && (!request.gpu.pose_enabled
            || (track["pose_refinement_policy"] == cpu_retrack::POSE_REFINEMENT_POLICY
                && track["pose_sampling"] == cpu_retrack::pose_sampling_contract(request.post.ball_stabilizer, request.gpu.pose_sampling())))
        && track["replay_id"] == replay
        && track["interpolate_pose"].as_bool().unwrap_or(true) == request.post.interpolate_pose
        && track["source_fusion_id"] == fusion
        && track["ball_selection_policy"] == "vballnet_only_v1"
        && track["project_id"] == request.project_id
        && track["video"] == request.video
        && touches["schema"] == "volley-native-cpu-touch-proposals/1"
        && touches["state"] == "motion_gated_touch_proposals"
        && touches["body_trajectory_intersection_required"] == true
        && touches["settings"]["require_pose"] == true
        && touches["automatic_action_prediction_method"]
            == volley_core::pose_classification::ACTION_PREDICTION_POLICY_VERSION
        && touches["preliminary_rally_scope_version"] == "primary_gpu_rally_mask_v1"
        && touches["primary_ball_job_id"] == saved["gpu_jobs"]["ball"]
        && touches["primary_rally_mask_sha256"] == saved["gpu_rally_mask_sha256"]
        && touches["preliminary_rally_settings"] == json!(request.post.rallies)
        && touches["preliminary_rally_scope_applied"].as_bool().is_some()
        && rallies["preliminary_rally_scope_version"] == "primary_gpu_rally_mask_v1"
        && rallies["primary_ball_job_id"] == touches["primary_ball_job_id"]
        && rallies["primary_rally_mask_sha256"] == touches["primary_rally_mask_sha256"]
        && rallies["preliminary_rally_scope_applied"] == touches["preliminary_rally_scope_applied"]
        && touches["touch_id"] == touch
        && touches["source_replay_id"] == replay
        && touches["ball_stabilizer"]["schema"] == "volley-ball-stabilizer/1"
        && touches["ball_stabilizer"]["version"] == volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION
        && touches["ball_stabilizer"]["settings"]
            == if saved["gpu_rally_mask"]["ball_stabilizer"]["trajectory_sha256"].is_string() {
                saved["gpu_rally_mask"]["ball_stabilizer"]["settings"].clone()
            } else {
                json!(request.post.ball_stabilizer)
            }
        && touches["ball_stabilizer"]["source_replay_sha256"] == touches["source_replay_sha256"]
        && saved["ball_stabilizer"] == touches["ball_stabilizer"]
        && touches["project_id"] == request.project_id
        && touches["video"] == request.video
        && touches["court_revision"] == saved["court_revision"]
        // The new strict body/ball intersection must never reuse a saved
        // PoseMotion v1 run just because its sliders happen to match.
        && (match request.post.touches.algorithm {
            TouchAlgorithm::PoseMotion => touches["method"] == "vballnet_trajectory_body_contact_v12",
            TouchAlgorithm::Legacy => touches["method"] == "vballnet_gravity_player_geometry_pose_contact_v11",
            TouchAlgorithm::TrajectoryTcn => tcn_sha.is_some()
                && touches["method"] == "vballnet_volley3d_tcn_v1"
                && touches["tcn_model_sha256"] == serde_json::json!(tcn_sha),
            TouchAlgorithm::BallTrajectory => touches["method"] == "vballnet_touch_detector_v5",
        })
        && rallies["schema"] == "volley-native-cpu-rally-proposals/1"
        && rallies["state"] == "provisional_rally_segments"
        && rallies["court_contact_context_version"] == "not_computed_primary_gpu_v1"
        && touches["initial_serve_attribution_version"] == crate::cpu_touches::INITIAL_SERVE_ATTRIBUTION_VERSION
        && touches["touch_side_attribution_version"] == crate::court_touch::TOUCH_SIDE_ATTRIBUTION_VERSION
        && touches["trajectory_detector_version"] == volley_core::touch_pipeline::TRAJECTORY_DETECTOR_VERSION
        && touches["ground_detector_version"] == volley_core::motion::GROUND_DETECTOR_VERSION
        && rallies["ground_boundary_version"] == volley_core::motion::GROUND_DETECTOR_VERSION
        && rallies["materialization_version"] == "primary_gpu_intervals_direct_v2"
        && rallies["rally_score_version"] == volley_core::rally_score::RALLY_SCORE_VERSION
        && touches["ball_tracking_policy"] == crate::cpu_retrack::ball_track::LOCAL_POLICY
        && rallies["ball_tracking_policy"] == crate::cpu_retrack::ball_track::LOCAL_POLICY
        && touches["leading_ball_rally_starts"] == json!([])
        && touches["leading_ball_rally_starts"] == rallies["leading_ball_rally_starts"]
        && touches["leading_ball_min_score"] == rallies["leading_ball_min_score"]
        && rallies["rally_run_id"] == rally
        && rallies["source_replay_id"] == replay
        && rallies["source_touch_id"] == touch
        && rallies["court_revision"] == touches["court_revision"]
        && rallies["project_id"] == request.project_id
        && rallies["video"] == request.video
}
// Changing the original-video 30 FPS source and detector-to-pose alignment
// together must invalidate both earlier GPU evidence versions.
const RFDETR_GPU_MODEL_STACK: &str =
    "vballnet_rfdetr_medium_f6b2a35e_adaptive_batch_fp16_fp32_original30_native_pose_pixels_bytetrack_lowconf_v9";

fn zero_copy_requested() -> bool {
    matches!(
        std::env::var("VOLLEY_ZERO_COPY_PRIMARY").ok().as_deref(),
        None | Some("1") | Some("true")
    )
}

fn requested_gpu_model_stack(detector: DetectorVariant) -> &'static str {
    match detector {
        DetectorVariant::RfDetrMedium => RFDETR_GPU_MODEL_STACK,
        DetectorVariant::RfDetrLarge => {
            "vballnet_rfdetr_large_da7f1bc1_original30_native_pose_pixels_bytetrack_lowconf_v7"
        }
    }
}

fn rally_pipeline_enabled(request: &StartRequest) -> bool {
    request.gpu.pose_enabled && request.mode.allows_gpu_execution()
}
fn compatible_version(saved: &Value, request: &StartRequest, identity: &Value) -> bool {
    saved["schema"] == SCHEMA
        && saved["analysis_range"] == json!(request.analysis_range)
        && saved["gpu_model_stack"] == requested_gpu_model_stack(request.gpu.person_detector)
        && (!request.gpu.pose_enabled || request.mode == PipelineMode::Post || saved["gpu_jobs"]["pose"].is_null())
        && saved["project_id"] == request.project_id
        && saved["video"] == request.video
        && decode_profiles(saved["profile"]["gpu"].clone(), empty_profile())
            .is_ok_and(|(mut gpu, _)| {
                gpu.parallel_rally_poses = request.gpu.parallel_rally_poses;
                json!(gpu) == json!(request.gpu)
            })
        // Older pipeline status files can predate the source_identity field;
        // the fusion and both GPU manifests are always validated separately.
        && (saved["source_identity"].is_null() || saved["source_identity"] == *identity)
}
fn deleted_launch(names: &Value, id: &str, saved: &Value) -> bool {
    names[format!("hidden_launch:{id}")] == true
        || saved["queue_id"]
            .as_str()
            .is_some_and(|queue| names[format!("hidden_launch:{queue}")] == true)
}

fn latest_deleted_video_start(
    root: &Path,
    project: &str,
    video: &str,
    names: &Value,
) -> Option<u64> {
    names
        .as_object()?
        .iter()
        .filter(|(_, value)| **value == true)
        .filter_map(|(key, _)| key.strip_prefix("hidden_launch:").filter(|id| valid_id(id)))
        .filter_map(|id| read_queue_rows(root, project, id).ok())
        .flatten()
        .filter(|row| row["video"] == video)
        .filter_map(|row| row["started_ms"].as_u64())
        .max()
}

pub(crate) fn unfinished_pipeline_videos(
    project_root: &Path,
    project: &str,
    deleting: &str,
) -> Result<BTreeSet<String>, ApiError> {
    let root = project_root.join("native_studio_pipelines");
    let names = storage::read_json(&project_root.join("native_analysis_names.json"))?
        .unwrap_or_else(|| json!({}));
    let mut videos = BTreeSet::new();
    let mut checked = 0;
    for entry in fs::read_dir(&root)
        .map_err(|e| storage::internal(format!("Read checkpoint owners: {e}")))?
    {
        let entry = entry.map_err(|e| storage::internal(format!("Read checkpoint owner: {e}")))?;
        let id = entry.file_name().to_string_lossy().into_owned();
        if !valid_id(&id) || id == deleting {
            continue;
        }
        checked += 1;
        if checked > MAX_PIPELINES {
            return Err(storage::conflict("Too many checkpoint owners"));
        }
        let Ok(path) = status_path(&root, &id) else {
            continue;
        };
        let Ok(row) = read(&path) else { continue };
        if row["schema"] != SCHEMA
            || row["project_id"] != project
            || row["id"] != id
            || row["queue_id"] == deleting
            || row["status"] == "complete"
            || deleted_launch(&names, &id, &row)
        {
            continue;
        }
        if let Some(video) = row["video"]
            .as_str()
            .filter(|video| storage::video_name(video).is_ok())
        {
            videos.insert(video.to_owned());
        }
    }
    Ok(videos)
}

fn find_reusable(
    state: &AppState,
    request: &StartRequest,
    identity: &Value,
    current_id: &str,
    rally_blocks: bool,
) -> ExistingVersions {
    let mut matches = ExistingVersions::default();
    // A newly trained or replaced TCN must never reuse a cached CPU Touch run
    // produced by different weights, even if all sliders are identical.
    let tcn_sha = if request.post.touches.algorithm == TouchAlgorithm::TrajectoryTcn {
        volley_inference::trajectory_tcn::checkpoint_sha256(
            &state
                .repository_root
                .join("models")
                .join(volley_inference::trajectory_tcn::MODEL_FILENAME),
        )
        .ok()
    } else {
        None
    };
    let Ok(root) = folder(state, &request.project_id, false) else {
        return matches;
    };
    let Ok(directory) = fs::read_dir(&root) else {
        return matches;
    };
    let mut paths = Vec::new();
    for item in directory {
        let Ok(item) = item else {
            return ExistingVersions::default();
        };
        let id = item.file_name().to_string_lossy().into_owned();
        if !valid_id(&id) || id == current_id {
            continue;
        }
        if paths.len() >= MAX_PIPELINES {
            return ExistingVersions::default();
        }
        let modified = item.metadata().and_then(|m| m.modified()).ok();
        paths.push((modified, id));
    }
    paths.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    let project = projects::project_root(state).join(&request.project_id);
    let Ok(names) = storage::read_json(&project.join("native_analysis_names.json")) else {
        return matches;
    };
    let names = names.unwrap_or_else(|| json!({}));
    let deleted_before =
        latest_deleted_video_start(&root, &request.project_id, &request.video, &names);
    // Real execution validates the court snapshot before attempting reuse.
    // In synthetic version tests, the optional source store is absent.
    let court_revision =
        crate::video_edits::court_snapshot(state, &request.project_id, &request.video)
            .ok()
            .flatten()
            .map(|court| court.revision);
    for (_, id) in paths {
        let Ok(file) = status_path(&root, &id) else {
            continue;
        };
        let Ok(saved) = read(&file) else {
            continue;
        };
        if deleted_launch(&names, &id, &saved)
            || deleted_before.is_some_and(|cutoff| {
                saved["started_ms"]
                    .as_u64()
                    .is_none_or(|start| start <= cutoff)
            })
        {
            continue;
        }
        let complete_gpu = gpu_lineage_valid(&project, request, identity, &saved);
        if rally_blocks
            && (!complete_gpu
                || saved["status"] != "complete"
                || !decode_profiles(
                    saved["profile"]["gpu"].clone(),
                    saved["profile"]["post"].clone(),
                )
                .is_ok_and(|(_, post)| {
                    json!(post.tracking) == json!(request.post.tracking)
                        && post.interpolate_players == request.post.interpolate_players
                        && json!(post.ball_stabilizer) == json!(request.post.ball_stabilizer)
                }))
        {
            continue;
        }
        let recoverable_pose = request.gpu.pose_enabled
            && saved["gpu_jobs"]["pose"].is_null()
            && matches!(
                saved["status"].as_str(),
                Some("failed" | "cancelled" | "interrupted")
            )
            && {
                let mut without_pose = request.clone();
                without_pose.gpu.pose_enabled = false;
                gpu_lineage_valid(&project, &without_pose, identity, &saved)
            };
        if saved["id"] != id
            || (saved["rally_block_pipeline"].as_bool().unwrap_or(false) != rally_blocks)
            || !compatible_version(&saved, request, identity)
            || !(complete_gpu || recoverable_pose)
            || saved["gpu_rally_mask"]["ground_detector"]["court_revision"] != json!(court_revision)
        {
            continue;
        }
        if complete_gpu
            && saved["status"] == "complete"
            && saved["pipeline_complete"] == true
            && decode_profiles(
                saved["profile"]["gpu"].clone(),
                saved["profile"]["post"].clone(),
            )
            .is_ok_and(|(_, post)| json!(post) == json!(request.post))
            && saved["court_revision"] == json!(court_revision)
            && cpu_lineage_valid(&project, request, &saved, tcn_sha.as_deref())
        {
            matches.gpu = Some(saved.clone());
            matches.full = Some(saved);
            return matches;
        }
        if matches.gpu.is_none() {
            matches.gpu = Some(saved);
        }
    }
    matches
}

/// Resolve a user-selected fusion against the valid GPU pipeline that
/// created it. CPU runs must never silently choose a different GPU source.
fn selected_gpu(
    state: &AppState,
    request: &StartRequest,
    identity: &Value,
    analysis_id: &str,
) -> Result<Value, String> {
    let root = folder(state, &request.project_id, false).map_err(reason)?;
    let project = projects::project_root(state).join(&request.project_id);
    if let Ok(entries) = fs::read_dir(&root) {
        let mut checked = 0_usize;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Чтение GPU-версии: {e}"))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if !valid_id(&name) {
                continue;
            }
            checked += 1;
            if checked > MAX_PIPELINES {
                return Err("Слишком много сохранённых GPU-версий".into());
            }
            let Ok(file) = status_path(&root, &name) else {
                continue;
            };
            let Ok(saved) = read(&file) else {
                continue;
            };
            if saved["schema"] != SCHEMA
                || saved["project_id"] != request.project_id
                || saved["video"] != request.video
                || saved["analysis_id"] != analysis_id
                || saved["source_identity"] != *identity
            {
                continue;
            }
            let Ok((gpu, _)) = decode_profiles(saved["profile"]["gpu"].clone(), empty_profile())
            else {
                continue;
            };
            let mut scoped = request.clone();
            scoped.gpu = gpu;
            scoped.analysis_range = serde_json::from_value(saved["analysis_range"].clone())
                .ok()
                .flatten();
            if gpu_lineage_valid(&project, &scoped, identity, &saved) {
                return Ok(saved);
            }
        }
    }
    // Fusion snapshots can also originate from the standalone native GPU API,
    // without a Studio pipeline status. Accept that exact source only after
    // validating every immutable GPU manifest and its SHA-256 references.
    let fusion = saved_stage(&project, "native_gpu_fusions", analysis_id, "frames.jsonl")
        .ok_or("Выбранный GPU-анализ не найден")?;
    if fusion["schema"] != "volley-native-gpu-fusion/1"
        || fusion["state"] != "raw_candidates_fused"
        || fusion["fusion_id"] != analysis_id
        || fusion["project_id"] != request.project_id
        || fusion["video"] != request.video
        || fusion["source_identity"] != *identity
        || fusion["fps"] != 30
    {
        return Err("Fusion не принадлежит выбранному видео или исходник изменился".into());
    }
    let ball = require_id(&fusion["sources"]["vballnet"], "job_id")?;
    let players = require_id(&fusion["sources"]["person_pose"], "job_id")?;
    let mut stride = None;
    let mut person_detector = DetectorVariant::default();
    let player_stride = fusion["player_sampling_period_frames"]
        .as_u64()
        .filter(|step| (1..=30).contains(step))
        .ok_or("Неверный шаг определения игроков в сохранённом GPU-анализе")?;
    for (model, id, source) in [
        ("ball", ball.as_str(), "vballnet"),
        ("players", players.as_str(), "person_pose"),
    ] {
        let raw = saved_stage(&project, "native_gpu_runs", id, "detections.jsonl")
            .ok_or("Не найден исходный GPU-результат для Fusion")?;
        let current_stride = raw["stride"]
            .as_u64()
            .filter(|s| (1..=9).contains(s))
            .ok_or("Неверный шаг исходного GPU-анализа")?;
        if !volley_core::contracts::gpu_manifest(
            &raw,
            &request.project_id,
            &request.video,
            id,
            model,
        ) || raw["source_identity"] != *identity
            || raw["analysis_range"] != fusion["analysis_range"]
            || raw["sha256"] != fusion["sources"][source]["sha256"]
            || (model == "players" && raw["person_sampling_period_frames"] != player_stride)
            || stride.is_some_and(|s| s != current_stride)
        {
            return Err("GPU-источник выбранной версии повреждён или не совпадает".into());
        }
        if model == "players" {
            person_detector = raw["person_pipeline"]
                .as_str()
                .and_then(DetectorVariant::from_pipeline_tag)
                .unwrap_or_default();
        }
        stride = Some(current_stride);
    }
    Ok(json!({
        "analysis_id": analysis_id,
        "gpu_jobs": {"ball":ball,"players":players},
        "analysis_range":fusion["analysis_range"],
        "id": analysis_id,
        // Standalone GPU manifests preserve the player sampling period.
        "profile": {"gpu": {"stride":stride.unwrap_or(9),
            "person_detector":person_detector,"pose_stride":player_stride,
            "gpu_memory_limit_mb":0,"cpu_threads":8}}
    }))
}

#[cfg(any(unix, windows))]
fn shared_native_pose_supported(enabled: bool, width: u32, height: u32) -> bool {
    enabled
        && width >= 2
        && height >= 2
        && (width as u64)
            .saturating_mul(height as u64)
            .saturating_mul(3)
            <= volley_media::raw_decoder::MAX_NATIVE_RGB_FRAME_BYTES as u64
}

/// The existing fusion evidence uses disjoint 9-frame model windows.
#[cfg(any(unix, windows))]
fn shared_primary_requested(
    flag: Option<&str>,
    stride: usize,
    pose_enabled: bool,
    pose_mode: crate::pose_inference::PoseInputMode,
) -> bool {
    // Shared NVDEC (+ verified software fallback) is the production default
    // for the normal nine-frame GPU pipeline. VOLLEY_SHARED_PRIMARY=0 is
    // an explicit rollback, preserving old standalone GPU artifacts.
    // Custom model-window strides and continuous video-pose requests still
    // use the established individual model path; never silently resample.
    matches!(flag, None | Some("1") | Some("true"))
        && stride == 9
        && (!pose_enabled || pose_mode == crate::pose_inference::PoseInputMode::Photo)
}

// GPU stage coordinates (progress window, status document, cancellation)
// stay explicit; a stage-context struct is planned for stage E.
#[expect(clippy::too_many_arguments)]
#[cfg(test)]
async fn gpu(
    app: &Arc<AppState>,
    request: &StartRequest,
    model: ModelKind,
    cancel: &AtomicBool,
    path: &Path,
    status: &mut Value,
    from: f64,
    share: f64,
) -> Result<String, String> {
    // Defense in depth: a CPU-post request must never reach native GPU model
    // startup, even if the caller's branching is later changed.
    if !request.mode.allows_gpu_execution() {
        return Err("CPU-постобработка не имеет права запускать GPU-модели".into());
    }
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let model_name = if model == ModelKind::Ball {
        "VballNet"
    } else {
        request.gpu.person_detector.label()
    };
    let total = status["gpu_total_frames"].as_u64().unwrap_or(0);
    progress_with_frames(
        path,
        status,
        &format!("GPU · {model_name}"),
        from,
        Some((0, total)),
    )?;
    log_event(
        path,
        &format!("GPU · {model_name}"),
        "info",
        "Подготовка вызова нативной модели",
    )?;
    let payload = json!({
        "project_id": request.project_id,
        "video": request.video,
        "model": model,
        "device": "cuda",
        "gpu_device_id": 0,
        "stride": request.gpu.stride,
        "person_stride": request.gpu.pose_stride,
        "person_detector": request.gpu.person_detector,
        "gpu_memory_limit_mb": request.gpu.gpu_memory_limit_mb,
        "cpu_threads": request.gpu.cpu_threads,
        "analysis_range":request.analysis_range
    });
    let config: inference_full::StartRequest = serde_json::from_value(payload.clone())
        .map_err(|e| format!("Неверный GPU-профиль: {e}"))?;
    log_native(
        path,
        &format!("GPU · {model_name}"),
        "inference_full::start",
        payload,
    )?;
    let Json(start) =
        inference_full::start_integrated(State(app.clone()), HeaderMap::new(), Json(config))
            .await
            .map_err(reason)?;
    let id = require_id(&start, "job_id")?;
    let mut last_phase = String::new();
    let mut last_bucket = -1i32;
    crate::gpu_poll::wait(
        app,
        &request.project_id,
        &id,
        || cancel.load(Ordering::Acquire),
        |gpu_status| {
            status["current_gpu_job_id"] = json!(id);
            let phase = gpu_status["phase"].as_str().unwrap_or("unknown");
            let pct = gpu_status["progress_percent"].as_f64().unwrap_or(0.0);
            let bucket = (pct.clamp(0.0, 100.0) / 10.0).floor() as i32;
            if phase != last_phase || bucket > last_bucket {
                log_event(
                    path,
                    &format!("GPU · {model_name}"),
                    "info",
                    &format!("job_id={id} · phase={phase} · прогресс={pct:.1}%"),
                )?;
                last_phase = phase.to_owned();
                last_bucket = bucket;
            }
            if phase == "complete" {
                status["current_gpu_job_id"] = Value::Null;
            }
            let done = gpu_status["decoded_frames"].as_u64().unwrap_or(0);
            let total = status["gpu_total_frames"].as_u64().unwrap_or(done).max(1);
            progress_with_frames(
                path,
                status,
                &format!("GPU · {model_name}: {phase}"),
                from + share * pct.clamp(0.0, 100.0) / 100.0,
                Some((done, total)),
            )
        },
    )
    .await
}

async fn cpu<T: Send + 'static>(
    app: Arc<AppState>,
    request: T,
    handler: fn(&AppState, &T) -> Result<Value, ApiError>,
) -> Result<Value, String> {
    tokio::task::spawn_blocking(move || handler(&app, &request))
        .await
        .map_err(|e| format!("CPU worker: {e}"))?
        .map_err(reason)
}
async fn cpu_cancel<T: Send + 'static>(
    app: Arc<AppState>,
    request: T,
    cancel: Arc<AtomicBool>,
    handler: impl FnOnce(&AppState, &T, &AtomicBool) -> Result<Value, ApiError> + Send + 'static,
) -> Result<Value, String> {
    tokio::task::spawn_blocking(move || handler(&app, &request, &cancel))
        .await
        .map_err(|e| format!("CPU worker: {e}"))?
        .map_err(reason)
}

async fn begin_publication(turn: &mut VideoTurn, cancel: &AtomicBool) -> Result<(), String> {
    turn.wait_detector(cancel).await?;
    turn.finish_detector();
    turn.wait_publish(cancel).await
}

async fn pipeline(
    app: &Arc<AppState>,
    request: &StartRequest,
    cancel: &Arc<AtomicBool>,
    path: &Path,
    status: &mut Value,
    turn: &mut VideoTurn,
) -> Result<(), String> {
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    #[cfg(any(unix, windows))]
    let mut pending_shared_pose: Option<crate::pose_inference::PendingPose>;
    let mut early_poses = None;
    let shared_reid_crops =
        crate::native_player_tracks::SharedReidCrops::new(request.post.reid_buffer_mib)?;
    log_native(
        path,
        "Проверка исходного видео",
        "preparation::plan_for_analysis_async",
        json!({"project_id":request.project_id,"video":request.video}),
    )?;
    progress(path, status, "Проверка исходного видео", 0.0)?;
    let ready = preparation::plan_for_analysis_async(
        app.clone(),
        &request.project_id,
        &request.video,
        if request.mode == PipelineMode::Post {
            request.source_analysis_id.as_deref()
        } else {
            None
        },
    )
    .await
    .map_err(reason)?;
    let prepared = preparation::ready_path_async(&ready)
        .await?
        .ok_or("Источник видео недоступен")?;
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    log_event(
        path,
        "Проверка исходного видео",
        "success",
        if request.mode == PipelineMode::Post {
            "Источник сохранённого анализа проверен"
        } else {
            "Источник проверен; первичный анализ читает оригинал без промежуточного MP4"
        },
    )?;
    status["gpu_total_frames"] = json!((request
        .analysis_range
        .map_or(ready.duration_sec, |range| range.end_sec - range.start_sec)
        * 30.0)
        .round()
        .max(0.0) as u64);
    progress(path, status, "Проверка исходного видео", 8.0)?;
    let (bytes, mtime_ns) = crate::inference_stream::source_stamp(&prepared)?;
    let identity = json!({"bytes":bytes,"mtime_ns":mtime_ns});
    let original_source = volley_media::preparation::source_provenance(&ready.source)?;
    status["original_source"] = original_source.clone();
    status["source_identity"] = identity.clone();
    let court_revision = Some(
        crate::video_edits::require_court_revision(app, &request.project_id, &request.video)
            .map_err(reason)?,
    );
    if status["queued_court_revision"] != json!(court_revision) {
        return Err("Разметка видео изменилась после постановки в очередь".into());
    }
    if !status["queue_id"].is_null()
        && ({
            let (bytes, mtime_ns) = crate::inference_stream::source_stamp(&ready.source)?;
            status["queued_source_identity"] != json!({"bytes":bytes,"mtime_ns":mtime_ns})
        } || json!(crate::video_edits::analysis_range(
            app,
            &request.project_id,
            &request.video
        )
        .map_err(reason)?)
            != status["analysis_range"])
    {
        return Err(
            "Источник, отрезок или разметка видео изменились после постановки в очередь".into(),
        );
    }
    status["court_revision"] = json!(court_revision);
    save(path, status)?;
    let current_id = status["id"].as_str().unwrap_or_default();
    log_event(
        path,
        "Выбор версии",
        "info",
        "Проверка GPU/CPU профилей, подготовленного видео и подписанных идентификаторов источника",
    )?;
    let reusable = if request.mode.allows_gpu_execution() {
        let reusable = find_reusable(
            app,
            request,
            &identity,
            current_id,
            rally_pipeline_enabled(request),
        );
        let matches = |saved: &Value| saved["original_source"] == original_source;
        ExistingVersions {
            full: reusable.full.filter(|saved| matches(saved)),
            gpu: reusable.gpu.filter(|saved| matches(saved)),
        }
    } else {
        ExistingVersions::default()
    };
    if request.mode == PipelineMode::Post {
        let source_id = request
            .source_analysis_id
            .as_deref()
            .ok_or("Не выбран первичный GPU-анализ в верхней панели")?;
        let source = selected_gpu(app, request, &identity, source_id)?;
        let (gpu, _) = decode_profiles(source["profile"]["gpu"].clone(), empty_profile())?;
        if gpu.pose_enabled || request.post.touches.require_pose {
            let project = projects::project_root(app).join(&request.project_id);
            let fusion = saved_stage(&project, "native_gpu_fusions", source_id, "frames.jsonl")
                .ok_or("Выбранный GPU-анализ не найден")?;
            let mut scope_source = source.clone();
            scope_source["original_source"] = original_source.clone();
            let (primary, _) = load_primary_rally_scope(app, &request.project_id, &scope_source)?;
            if !cpu_retrack::has_refined_pose(
                &project,
                source_id,
                fusion["sha256"].as_str().ok_or("Нет SHA-256 Fusion")?,
                Some(&primary.sha256),
                court_revision.as_deref(),
            )? {
                return Err("Для пересчёта нужны позы RTMPose, измеренные по стабильным рамкам. Выполните первую фазу анализа.".into());
            }
        }
        status["analysis_id"] = source["analysis_id"].clone();
        status["analysis_range"] = source["analysis_range"].clone();
        status["gpu_jobs"] = source["gpu_jobs"].clone();
        status["touch_detector_count"] = source["touch_detector_count"].clone();
        status["shared_pose"] = source["shared_pose"].clone();
        status["rally_block_pipeline"] = source["rally_block_pipeline"].clone();
        status["pose_frame_progress"] = source["pose_frame_progress"].clone();
        status["profile"]["gpu"] = json!(gpu);
        status["gpu_model_stack"] = source["gpu_model_stack"].clone();
        status["gpu_reused"] = json!(true);
        status["reused_from_id"] = source["id"].clone();
        save(path, status)?;
        log_event(
            path,
            "CPU",
            "info",
            &format!("Создаётся новая CPU-версия для выбранного fusion_id={source_id}"),
        )?;
        progress(
            path,
            status,
            "Выбран сохранённый GPU-анализ · новый CPU-пересчёт",
            82.0,
        )?;
    } else if let Some(existing) = reusable.full.filter(|v| {
        request.mode == PipelineMode::Full
            && (v["primary_tracking"]["pose_ready"] != true
                || v["pose_frame_progress"]["basis"] == "stable_crop_frames_v1")
            && v["automatic_identity"]["decision_policy"] == "track_photo_consensus_v1"
            && v["cpu_runs"]["tracking"]
                .as_str()
                .and_then(|replay| {
                    crate::native_player_tracks::automatic_links_revision(
                        app,
                        &request.project_id,
                        &request.video,
                        replay,
                    )
                    .ok()
                    .flatten()
                })
                .is_some_and(|current| v["automatic_identity"]["identity_revision"] == current)
    }) {
        status["analysis_id"] = existing["analysis_id"].clone();
        status["gpu_jobs"] = existing["gpu_jobs"].clone();
        status["shared_pose"] = existing["shared_pose"].clone();
        status["pose_frame_progress"] = existing["pose_frame_progress"].clone();
        status["gpu_rally_mask"] = existing["gpu_rally_mask"].clone();
        status["touch_detector_count"] = existing["touch_detector_count"].clone();
        status["analysis_frames_processed"] = json!(pipeline_frames_processed(&existing));
        status["gpu_rally_mask_sha256"] = existing["gpu_rally_mask_sha256"].clone();
        status["cpu_runs"] = existing["cpu_runs"].clone();
        status["primary_tracking"] = existing["primary_tracking"].clone();
        status["player_tracking_stats"] = existing["player_tracking_stats"].clone();
        status["postprocessing_id"] = existing["postprocessing_id"].clone();
        status["rallies_published"] = existing["rallies_published"].clone();
        status["gpu_reused"] = json!(true);
        status["result_reused"] = json!(true);
        status["reused_from_id"] = existing["id"].clone();
        status["ball_stabilizer"] = existing["ball_stabilizer"].clone();
        status["ground_detector"] = existing["ground_detector"].clone();
        status["automatic_identity"] = existing["automatic_identity"].clone();
        status["automatic_identity_complete"] = existing["automatic_identity_complete"].clone();
        status["automatic_identity_error"] = existing["automatic_identity_error"].clone();
        log_event(
            path,
            "Повторное использование",
            "success",
            &format!(
                "Настройки совпали: GPU и CPU результатов повторно использованы из запуска {}",
                existing["id"].as_str().unwrap_or("?")
            ),
        )?;
        progress(
            path,
            status,
            "Настройки совпали: выбраны сохранённые версии GPU и CPU",
            99.9,
        )?;
        begin_publication(turn, cancel).await?;
        return Ok(());
    }
    if request.mode.allows_gpu_execution() {
        if let Some(existing) = reusable.gpu {
            status["analysis_id"] = existing["analysis_id"].clone();
            status["gpu_jobs"] = existing["gpu_jobs"].clone();
            status["shared_pose"] = existing["shared_pose"].clone();
            status["pose_frame_progress"] = existing["pose_frame_progress"].clone();
            status["gpu_rally_mask"] = existing["gpu_rally_mask"].clone();
            status["touch_detector_count"] = existing["touch_detector_count"].clone();
            status["analysis_frames_processed"] = json!(pipeline_frames_processed(&existing));
            status["gpu_rally_mask_sha256"] = existing["gpu_rally_mask_sha256"].clone();
            status["gpu_reused"] = json!(true);
            status["reused_from_id"] = existing["id"].clone();
            log_event(
                path,
                "Повторное использование GPU",
                "info",
                &format!(
                    "GPU источник из запуска {} сохранён; запускаются только новые стадии CPU",
                    existing["id"].as_str().unwrap_or("?")
                ),
            )?;
            progress(
                path,
                status,
                "Сохранённый GPU-анализ найден · новая CPU-постобработка",
                82.0,
            )?;
        } else {
            log_event(
                path,
                "Первичный анализ",
                "info",
                &format!(
                    "Совпадающего GPU-профиля нет, VBallNet и {}",
                    request.gpu.person_detector.label()
                ),
            )?;
            // Primary shared GPU path has passed real-match evidence/CPU
            // parity; legacy independent inference remains a rollback.
            #[cfg(any(unix, windows))]
            let wants_shared = shared_primary_requested(
                std::env::var("VOLLEY_SHARED_PRIMARY").ok().as_deref(),
                request.gpu.stride,
                request.gpu.pose_enabled,
                request.gpu.pose_mode,
            );
            if !wants_shared {
                return Err("Для анализа игроков и поз нужен общий GPU-конвейер: шаг VBallNet 9, режим поз «Фото», VOLLEY_SHARED_PRIMARY не должен быть отключён".into());
            }
            #[cfg(any(unix, windows))]
            let shared_plan = if wants_shared {
                let plan = preparation::plan_for_analysis_async(
                    app.clone(),
                    &request.project_id,
                    &request.video,
                    None,
                )
                .await
                .map_err(reason)?;
                let source = preparation::ready_path_async(&plan)
                    .await?
                    .ok_or("Original video not ready for shared GPU analysis")?;
                if source != plan.source {
                    return Err("Shared GPU analysis requires the original video".into());
                }
                Some(plan)
            } else {
                None
            };
            #[cfg(any(unix, windows))]
            let shared = if let Some(plan) = shared_plan {
                log_native(
                    path,
                    "Общий GPU",
                    "inference_full::run_joint_pair",
                    json!({
                        "project_id":request.project_id,
                        "video":request.video,
                        "stride":request.gpu.stride,
                        "person_stride":request.gpu.pose_stride,
                        "mode":"shared_nvdec_primary_v1",
                        "pose_enabled":request.gpu.pose_enabled,
                        "primary_rallies":request.gpu.primary_rallies,
                    }),
                )?;
                let queued = !status["queue_id"].is_null();
                if queued {
                    progress(path, status, "Ожидание очереди VBallNet", 8.0)?;
                }
                turn.wait_detector(cancel).await?;
                let independent = turn.independent.clone();
                let permit = acquire_gpu(
                    independent
                        .as_ref()
                        .map_or(&app.inference_limit, |q| &q.joint),
                    cancel,
                )
                .await?;
                let total = status["gpu_total_frames"].as_u64().unwrap_or(0);
                progress_with_frames(
                    path,
                    status,
                    "GPU · VBallNet: траектория мяча и розыгрыши",
                    8.0,
                    Some((0, total)),
                )?;
                let rally_progress = Arc::new(std::sync::Mutex::new(
                    crate::rally_handoff::RallyStageProgress::default(),
                ));
                let block_mode = rally_pipeline_enabled(request);
                let result = tokio::task::block_in_place(|| {
                    inference_full::run_joint_pair(
                        app,
                        &request.project_id,
                        &request.video,
                        &plan.source,
                        request
                            .analysis_range
                            .map_or(plan.duration_sec, |range| range.end_sec - range.start_sec),
                        request.analysis_range,
                        request.gpu.pose_stride,
                        false,
                        request.gpu.person_detector,
                        request.gpu.gpu_memory_limit_mb,
                        request.gpu.cpu_threads,
                        request.gpu.primary_rallies,
                        request.post.ball_stabilizer,
                        block_mode.then(|| cpu_retrack::EarlyPoseOptions {
                            independent: independent.clone(),
                            sampling: request.gpu.pose_sampling(),
                            tracking: request.post.tracking,
                            court: None,
                            interpolate_players: request.post.interpolate_players,
                            threads: request.gpu.cpu_threads,
                            cancelled: cancel.clone(),
                            progress: rally_progress.clone(),
                            checkpoint_root: path
                                .parent()
                                .and_then(Path::parent)
                                .map(Path::to_path_buf),
                            reid_crops: (request.mode == PipelineMode::Full
                                && request.post.reid_buffer_mib > 0)
                                .then(|| shared_reid_crops.clone()),
                        }),
                        cancel,
                        |fraction, processed, total, data| {
                            if block_mode && fraction >= 0.4 {
                                status["rally_stage_progress"] =
                                    crate::rally_handoff::update_progress(&rally_progress, |p| {
                                        serde_json::to_value(p)
                                    })?
                                    .map_err(|e| e.to_string())?;
                            }
                            match data {
                                Some(inference_full::JointProgressData::RallyMask(
                                    mask,
                                    decoded_frames,
                                )) => {
                                    return publish_primary_rally_mask(
                                        path,
                                        status,
                                        mask,
                                        decoded_frames,
                                    );
                                }
                                Some(inference_full::JointProgressData::BallFinished) => {
                                    if independent.is_some() {
                                        turn.finish_detector();
                                    }
                                }
                                Some(inference_full::JointProgressData::TouchCount(count)) => {
                                    if status["touch_detector_count"].as_u64() == Some(count) {
                                        return Ok(());
                                    }
                                    status["touch_detector_count"] = json!(count);
                                    return save(path, status);
                                }
                                None => {}
                            }
                            let label = if fraction == 0.399999 {
                                "ball_stabilizer"
                            } else if fraction == 0.3999995 {
                                "Rally Detector"
                            } else if fraction == 0.39999975 {
                                "Touch Detector"
                            } else if fraction < 0.4 {
                                "GPU · VBallNet: траектория мяча и розыгрыши"
                            } else {
                                "GPU · RF-DETR: игровые интервалы"
                            };
                            progress_with_frames(
                                path,
                                status,
                                label,
                                8.0 + 72.0 * fraction,
                                Some((processed, total)),
                            )
                        },
                    )
                })?;
                drop(permit);
                crate::preparation::primary_preview::enqueue(
                    app.clone(),
                    request.project_id.clone(),
                    request.video.clone(),
                    result.source_provenance.clone(),
                );
                log_event(
                    path,
                    "Общий GPU",
                    "success",
                    &format!(
                        "GPU-проходов=2: {:?}, окон={}, кадров={}, zero_copy={}, fallback={}",
                        result.backend,
                        result.windows,
                        result.decoded_frames,
                        result.zero_copy,
                        result.fallback_reason.as_deref().unwrap_or("none")
                    ),
                )?;
                status["phase1_checkpoint_key"] = json!(result.checkpoint_key);
                status["zero_copy_active"] = json!(result.zero_copy);
                status["gpu_rally_mask"] = result.rally_filter.unwrap_or(Value::Null);
                save(path, status)?;
                pending_shared_pose = result.pending_pose;
                early_poses = result.early_poses;
                status["early_pose_groups"] =
                    json!(early_poses.as_ref().map_or(0, |cache| cache.group_count()));
                if let Some(error) = result.early_pose_error {
                    log_event(
                        path,
                        "RTMPose",
                        "warning",
                        &format!("Rally pose prepass: {error}"),
                    )?;
                }
                save(path, status)?;
                Some((result.ball_job_id, result.player_job_id))
            } else {
                return Err(
                    "Общий GPU-конвейер не был выбран; отдельная детекция игроков запрещена".into(),
                );
            };
            let (ball, players) = shared.ok_or("Общий GPU-конвейер не вернул результаты")?;
            status["gpu_jobs"]["ball"] = json!(ball);
            status["gpu_jobs"]["players"] = json!(players);
            save(path, status)?;
            if cancel.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }

            if !status["queue_id"].is_null() {
                progress(path, status, "Ожидание очереди публикации", 80.0)?;
            }
            begin_publication(turn, cancel).await?;
            progress(path, status, "Объединение · VBallNet и рамки игроков", 82.0)?;
            log_native(
                path,
                "Объединение GPU",
                "inference_fusion::do_fusion",
                json!({"project_id":request.project_id,"video":request.video,"vball_job_id":ball,"person_job_id":players}),
            )?;
            let fusion = cpu(
                app.clone(),
                inference_fusion::FuseRequest {
                    project_id: request.project_id.clone(),
                    video: request.video.clone(),
                    vball_job_id: ball,
                    person_job_id: players,
                },
                inference_fusion::do_fusion,
            )
            .await?;
            let fusion_id = require_id(&fusion, "fusion_id")?;
            status["analysis_id"] = json!(fusion_id);
            save(path, status)?;
            log_event(
                path,
                "Объединение GPU",
                "success",
                &format!("GPU объединён: fusion_id={fusion_id}"),
            )?;
            #[cfg(any(unix, windows))]
            if let Some(pending) = pending_shared_pose.take() {
                if cancel.load(Ordering::Acquire) {
                    return Err("cancelled".into());
                }
                // Pose evidence is invisible to reuse until its detector
                // source and SHA-bound fusion are both immutable and verified.
                let pose = pending.publish(&fusion, cancel)?;
                status["gpu_jobs"]["pose"] = pose["pose_run_id"].clone();
                status["shared_pose"] = shared_pose_metrics(status, &pose);
                save(path, status)?;
                log_event(
                    path,
                    "GPU · RTMPose-M",
                    "success",
                    &format!(
                        "Поза из общего RGB-потока: samples={}, players={}, fusion={fusion_id}",
                        pose["sampled_frames"], pose["player_poses"]
                    ),
                )?;
            }
            if cancel.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }
        }
    }

    begin_publication(turn, cancel).await?;
    let (primary, primary_mask) = load_primary_rally_scope(app, &request.project_id, status)?;
    if request.mode == PipelineMode::Post {
        for key in ["padding_before_ms", "padding_after_ms"] {
            status["profile"]["gpu"]["primary_rallies"][key] =
                json!(primary_mask[key].as_u64().unwrap_or(0));
        }
    }
    status["gpu_rally_mask"] = primary_mask;
    status["gpu_rally_mask_sha256"] = json!(primary.sha256);
    status["primary_rally_source"] =
        json!({"ball_job_id":primary.ball_job_id,"sha256":primary.sha256});
    save(path, status)?;
    let fusion_id = require_id(status, "analysis_id")?;
    log_native(
        path,
        "Track ID",
        "cpu_retrack::process",
        json!({"project_id":request.project_id,"video":request.video,"fusion_id":fusion_id,"settings":request.post.tracking}),
    )?;
    let refine_pose = (request.mode.allows_gpu_execution() && request.gpu.pose_enabled).then_some(
        cpu_retrack::PoseRefinement {
            threads: request.gpu.cpu_threads,
            memory_mb: request.gpu.gpu_memory_limit_mb,
            ball_stabilizer: request.post.ball_stabilizer,
            sampling: request.gpu.pose_sampling(),
            early: early_poses,
        },
    );
    let pose_permit = if let Some(refinement) = &refine_pose {
        let plan =
            preparation::plan_for(app, &request.project_id, &request.video).map_err(reason)?;
        let native = volley_media::probe(&plan.source)?;
        if !shared_native_pose_supported(true, native.width, native.height) {
            return Err("RTMPose: исходное разрешение превышает предел декодера".into());
        }
        if refinement.early.is_none() {
            progress(path, status, "Ожидание очереди RTMPose", 86.0)?;
            Some(if let Some(queues) = &turn.independent {
                queues.reid(cancel).await?
            } else {
                acquire_gpu(&app.inference_limit, cancel).await?
            })
        } else {
            None
        }
    } else {
        None
    };
    if request.mode == PipelineMode::Post {
        progress(path, status, "CPU · Track ID", 86.0)?;
    }
    let replay_request = cpu_retrack::ReplayRequest {
        project_id: request.project_id.clone(),
        video: request.video.clone(),
        fusion_id,
        settings: Some(request.post.tracking),
        interpolate_pose: request.post.interpolate_pose,
        interpolate_players: request.post.interpolate_players,
    };
    let replay = tokio::task::block_in_place(|| {
        cpu_retrack::process_with_cancel(app, &replay_request, cancel, refine_pose, |event| {
            tracking_progress(path, status, request.mode, event)
        })
    })
    .map_err(reason)?;
    drop(pose_permit);
    let replay_id = require_id(&replay, "replay_id")?;
    if replay["court_revision"] != status["court_revision"] {
        return Err("Court calibration changed during ByteTrack".into());
    }
    if replay["player_frame_scope_version"] != "primary_gpu_selected_intervals_v1"
        || replay["player_frame_scope_sha256"] != status["gpu_rally_mask_sha256"]
        || replay["player_frame_scope_ball_job_id"] != status["gpu_jobs"]["ball"]
    {
        return Err("Трекинг CPU вышел за границы проверенной GPU-маски розыгрышей".into());
    }
    let pose_ready = replay["pose_refinement_policy"] == cpu_retrack::POSE_REFINEMENT_POLICY;
    if (status["profile"]["gpu"]["pose_enabled"] == true
        || (request.mode != PipelineMode::Gpu && request.post.touches.require_pose))
        && !pose_ready
    {
        return Err(
            "Для выбранных настроек трекинга нет совместимых поз RTMPose. Выполните первую фазу анализа."
                .into(),
        );
    }
    status["primary_tracking"] = json!({
        "replay_id":replay_id,"pose_ready":pose_ready,"pose_sampling":replay["pose_sampling"],
    });
    if request.mode != PipelineMode::Gpu {
        status["cpu_runs"]["tracking"] = json!(replay_id);
    }
    if pose_ready {
        log_event(
            path,
            "RTMPose",
            "success",
            &format!(
                "Позы уточнены по RGB оригинала и стабилизированным рамкам: {}",
                replay["refined_pose_count"].as_u64().unwrap_or_default()
            ),
        )?;
    }
    status["player_tracking_stats"] = json!({
        "raw_detections": replay["raw_player_detections"],
        "total_track_ids": replay["tentative_track_count"],
        "short_track_ids": replay["discarded_short_track_ids"],
        "track_ids": replay["provisional_track_count"],
        "refined_poses": replay["refined_pose_count"],
    });
    status["cpu_player_frame_coverage"] = json!({
        "processed_frames":replay["player_frames_processed"],
        "eligible_frames":replay["player_frame_scope_frames"],
        "total_frames":replay["frames"],
    });
    save(path, status)?;
    log_event(
        path,
        "Track ID",
        "success",
        &format!(
            "Трекинг завершён: replay_id={replay_id} · ID подтверждено={} ·              коротких ID отброшено={} · обнаружений отброшено={}",
            replay["provisional_track_count"].as_u64().unwrap_or(0),
            replay["discarded_short_track_ids"].as_u64().unwrap_or(0),
            replay["discarded_short_track_detections"].as_u64().unwrap_or(0),
        ),
    )?;
    log_event(
        path,
        "Track ID",
        "info",
        &format!(
            "CPU обработал {} из {} кадров выбранных игровых интервалов (всего кадров мяча: {})",
            replay["player_frames_processed"],
            replay["player_frame_scope_frames"],
            replay["frames"],
        ),
    )?;
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }

    if request.mode == PipelineMode::Gpu {
        return Ok(());
    }
    if status["gpu_rally_mask"]["ball_stabilizer"]["trajectory_sha256"].is_null() {
        progress(path, status, "ball_stabilizer", 90.0)?;
    }
    let stabilized = tokio::task::block_in_place(|| {
        cpu_touches::stabilize_primary_replay(
            app,
            &request.project_id,
            &request.video,
            &replay_id,
            request.post.ball_stabilizer,
            cancel,
            |done, total| {
                progress_with_frames(
                    path,
                    status,
                    "ball_stabilizer",
                    90.0 + done as f64 / total.max(1) as f64,
                    Some((done, total)),
                )
            },
        )
    })?;
    status["ball_stabilizer"] = stabilized.summary();
    save(path, status)?;
    log_event(
        path,
        "ball_stabilizer",
        "success",
        &format!(
            "Принято={} · отклонено={} · интерполировано={}",
            stabilized.motion.observed_frames,
            stabilized.motion.rejected_frames.len(),
            stabilized
                .motion
                .interpolated
                .iter()
                .filter(|&&v| v)
                .count(),
        ),
    )?;
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let touch_request = cpu_touches::TouchRequest {
        project_id: request.project_id.clone(),
        video: request.video.clone(),
        replay_id: replay_id.clone(),
        settings: Some(request.post.touches),
        pose_classification: request.post.pose_classification,
    };
    progress(path, status, "Touch Attribution", 91.0)?;
    let ground = tokio::task::block_in_place(|| {
        cpu_touches::ground_stage::prepare_touch_detection(
            app,
            &touch_request,
            &stabilized,
            &primary,
            cancel,
        )
    })?;
    if ground.summary["court_revision"] != status["court_revision"] {
        return Err("Court calibration changed during Ground Detector".into());
    }
    status["ground_detector"] = ground.summary.clone();
    save(path, status)?;
    log_event(
        path,
        "Touch Attribution",
        "success",
        &format!(
            "Изломы траектории: {}; кандидаты пола в розыгрышах: {}",
            ground.breaks.len(),
            ground.summary["scoped_candidate_count"]
        ),
    )?;
    progress(path, status, "Touch Attribution", 91.5)?;
    log_native(
        path,
        "Touch Attribution",
        "cpu_touches::attribute_primary_with_progress_cancel",
        json!({"project_id":request.project_id,"video":request.video,"replay_id":replay_id,"settings":request.post.touches,"primary_ball_job_id":primary.ball_job_id}),
    )?;
    let touches = tokio::task::block_in_place(|| {
        cpu_touches::attribute_primary_with_progress_cancel(
            app,
            &touch_request,
            request.post.rallies,
            &primary,
            &stabilized,
            &ground,
            &mut |done, total| {
                progress_with_frames(
                    path,
                    status,
                    "Touch Attribution",
                    91.5 + 3.5 * done as f64 / total.max(1) as f64,
                    Some((done, total)),
                )
            },
            cancel,
        )
    })
    .map_err(reason)?;
    let touch_id = require_id(&touches, "touch_id")?;
    if touches["court_revision"] != status["court_revision"] {
        return Err("Court calibration changed during CPU processing".into());
    }
    status["cpu_runs"]["touches"] = json!(touch_id);
    save(path, status)?;
    log_event(
        path,
        "Touch Attribution",
        "success",
        &format!("Track ID и типы касаний сохранены: touch_id={touch_id}"),
    )?;
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }

    progress(path, status, "Публикация розыгрышей", 96.0)?;
    log_native(
        path,
        "Первичные розыгрыши",
        "cpu_rallies::publish_primary_with_cancel",
        json!({"project_id":request.project_id,"video":request.video,"primary_ball_job_id":primary.ball_job_id,"touch_id":touch_id}),
    )?;
    let rallies = cpu_cancel(
        app.clone(),
        cpu_rallies::RallyRequest {
            project_id: request.project_id.clone(),
            video: request.video.clone(),
            replay_id: replay_id.clone(),
            touch_id,
            settings: Some(request.post.rallies),
        },
        cancel.clone(),
        cpu_rallies::publish_primary_with_cancel,
    )
    .await?;
    let rally_id = require_id(&rallies, "rally_run_id")?;
    status["cpu_runs"]["rallies"] = json!(rally_id);
    status["postprocessing_id"] = json!(rally_id);
    status["rallies_published"] = rallies["rally_candidates"].clone();
    save(path, status)?;
    log_event(
        path,
        "Первичные розыгрыши",
        "success",
        &format!("Розыгрыши VBallNet доступны в предпросмотре: rally_run_id={rally_id}"),
    )?;
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let replay_sha = replay["sha256"]
        .as_str()
        .ok_or("У треков отсутствует SHA-256")?;
    let reid_permit = if !status["queue_id"].is_null()
        && request.post.reid_device != volley_inference::reid_openvino::ReidDevice::Cpu
    {
        progress(path, status, "Ожидание очереди ReID", 98.0)?;
        Some(if let Some(queues) = &turn.independent {
            queues.reid(cancel).await?
        } else {
            acquire_gpu(&app.inference_limit, cancel).await?
        })
    } else {
        None
    };
    progress(
        path,
        status,
        "Постобработка · автоматическая привязка игроков",
        98.0,
    )?;
    let identity_result = crate::native_player_tracks::reid_review_with_crops(
        State(app.clone()),
        HeaderMap::new(),
        Json(crate::native_player_tracks::ReidReviewRequest {
            project_id: request.project_id.clone(),
            video: request.video.clone(),
            replay_id,
            expected_source_sha256: replay_sha.to_owned(),
            // Only assign athletes actually observed in this CPU rally version.
            rally_run_id: Some(rally_id.clone()),
            action: "postprocess_auto_assign".into(),
            track_ids: Vec::new(),
            expected_revision: None,
            threshold_percent: 30,
            auto_threshold_percent: Some(request.post.auto_assign_threshold_percent),
            reid_device: request.post.reid_device,
        }),
        Some(&shared_reid_crops),
    )
    .await;
    drop(reid_permit);
    status["reid_shared_crops"] = shared_reid_crops.stats();
    let identities = match identity_result {
        Ok(Json(value)) => value,
        Err(error) => {
            // Missing references or ReID runtime must not invalidate saved
            // tracking, touches and rallies. Expose the failure in the log.
            let message = reason(error);
            status["automatic_identity_complete"] = json!(false);
            status["automatic_identity_error"] = json!(message);
            save(path, status)?;
            log_event(
                path,
                "Игроки",
                "warning",
                &format!("Автопривязка не выполнена: {message}"),
            )?;
            return Ok(());
        }
    };
    status["automatic_identity"] = identities.clone();
    status["automatic_identity"]["decision_policy"] = json!("track_photo_consensus_v1");
    let remaining = identities["unassigned_track_ids"]
        .as_array()
        .ok_or("Некорректный результат привязки")?;
    status["automatic_identity_complete"] = json!(remaining.is_empty());
    save(path, status)?;
    if !remaining.is_empty() {
        let reason_count = |key: &str| {
            identities["unassigned_reasons"][key]
                .as_array()
                .map_or(0, Vec::len)
        };
        log_event(
            path,
            "Игроки",
            "warning",
            &format!(
                "Автопривязка: назначено {} ID, непривязанных {} ID (первые 12: {:?}). Порог {}%. Причины: нет совпадений {}, мало наблюдений {}, ниже порога {}, конфликт/неоднозначность {}, запрещено вручную {}.",
                identities["automatic_count"],remaining.len(),
                remaining.iter().take(12).collect::<Vec<_>>(),
                request.post.auto_assign_threshold_percent,
                reason_count("no_photo_match"),
                reason_count("insufficient_observations"),
                reason_count("below_similarity_threshold"),
                reason_count("identity_conflict_or_ambiguity"),
                reason_count("manually_suppressed")
            ),
        )?;
    } else {
        log_event(
            path,
            "Игроки",
            "success",
            &format!(
                "Предварительно привязано ID: {}",
                identities["automatic_count"]
            ),
        )?;
    }
    Ok(())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CompletionQuality {
    Complete,
    IdentityFailed,
    IdentityReviewRequired,
}

fn completion_quality(mode: PipelineMode, status: &Value) -> CompletionQuality {
    if mode == PipelineMode::Gpu {
        return CompletionQuality::Complete;
    }
    if status["automatic_identity_error"]
        .as_str()
        .is_some_and(|error| !error.is_empty())
    {
        CompletionQuality::IdentityFailed
    } else if status["automatic_identity_complete"] == false {
        CompletionQuality::IdentityReviewRequired
    } else {
        CompletionQuality::Complete
    }
}

fn cancelled_outcome(outcome: Result<(), String>, cancelled: bool) -> Result<(), String> {
    if cancelled {
        Err("cancelled".into())
    } else {
        outcome
    }
}

async fn perform(
    state: Arc<AppState>,
    job: QueuedRun,
    cancel: Arc<AtomicBool>,
    mut turn: VideoTurn,
) {
    let QueuedRun {
        request,
        path,
        mut status,
        ..
    } = job;
    let _ = log_event(
        &path,
        request.mode.label(),
        "info",
        "Запуск создан, настройки и выбранный источник сохранены.",
    );
    let mut outcome = pipeline(&state, &request, &cancel, &path, &mut status, &mut turn).await;
    if outcome.is_ok() && request.mode != PipelineMode::Gpu && !cancel.load(Ordering::Acquire) {
        let _ = progress(&path, &mut status, "Монтажная временная разметка", 99.5);
        let rally = status["postprocessing_id"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        let app = state.clone();
        let project = request.project_id.clone();
        match tokio::task::spawn_blocking(move || {
            crate::montage_material::publish_material(&app, &project, &rally)
        })
        .await
        {
            Ok(Ok(material)) => {
                status["montage_material"] = material;
                if let Err(error) = log_event(
                    &path,
                    "Монтажная разметка",
                    "success",
                    "Финальная временная разметка и план монтажа сохранены",
                ) {
                    outcome = Err(error);
                }
            }
            Ok(Err(error)) => outcome = Err(reason(error)),
            Err(error) => outcome = Err(format!("Timeline worker: {error}")),
        }
    }
    let _slot = manager().lock().await;
    outcome = cancelled_outcome(outcome, cancel.load(Ordering::Acquire));
    match outcome {
        Ok(()) => {
            status["status"] = json!("complete");
            let quality = completion_quality(request.mode, &status);
            status["completion_quality"] = json!(quality);
            status["stage"] = json!(if quality == CompletionQuality::IdentityFailed {
                "Анализ завершён · автопривязка игроков не выполнена"
            } else if quality == CompletionQuality::IdentityReviewRequired {
                "Анализ завершён · остались непривязанные ID"
            } else {
                match request.mode {
                    PipelineMode::Full => "GPU и CPU завершены",
                    PipelineMode::Gpu => "Первичный GPU-анализ завершён",
                    PipelineMode::Post => "Новая CPU-постобработка завершена",
                }
            });
            status["progress_percent"] = json!(100.0);
            status["pipeline_complete"] = json!(request.mode != PipelineMode::Gpu);
            let _ = log_event(
                &path,
                request.mode.label(),
                "success",
                if status["result_reused"] == true {
                    "Готовый GPU+CPU результат повторно использован"
                } else {
                    match request.mode {
                        PipelineMode::Full => "Все стадии полного анализа успешно завершены",
                        PipelineMode::Gpu => {
                            "Первичный анализ, стабильные рамки и выбранные позы сохранены"
                        }
                        PipelineMode::Post => "Новая CPU-версия сохранена без запуска GPU",
                    }
                },
            );
        }
        Err(error) => {
            let cancelled = error == "cancelled" || cancel.load(Ordering::Acquire);
            status["status"] = json!(if cancelled { "cancelled" } else { "failed" });
            status["stage"] = json!(if cancelled {
                "Обработка отменена"
            } else {
                "Ошибка анализа"
            });
            status["error"] = if cancelled { Value::Null } else { json!(error) };
            status["pipeline_complete"] = json!(false);
            let _ = log_event(
                &path,
                request.mode.label(),
                if cancelled { "warn" } else { "error" },
                if cancelled {
                    "Анализ отменён пользователем"
                } else {
                    &error
                },
            );
        }
    }
    status["current_gpu_job_id"] = Value::Null;
    finish_timed_stage(&mut status);
    let saved = save(&path, &status).is_ok();
    if saved && status["status"] == "complete" {
        turn.finish_publish();
        if let (Some(key), Some(root)) = (
            status["phase1_checkpoint_key"].as_str(),
            path.parent().and_then(Path::parent),
        ) {
            if key.len() == 64 && key.bytes().all(|b| b.is_ascii_hexdigit()) {
                let folder = root.join("phase1_checkpoints").join(key);
                if storage::safe_folder(&folder).is_ok() {
                    let _ = fs::remove_dir_all(folder);
                }
            }
        }
    }
}

fn record_pipeline_worker_failure(
    path: &Path,
    project_id: &str,
    id: &str,
    error: &str,
) -> Result<(), String> {
    let mut status = read(path).map_err(|e| format!("Чтение аварийного статуса анализа: {e:?}"))?;
    if status["schema"] != SCHEMA || status["project_id"] != project_id || status["id"] != id {
        return Err("Аварийный статус принадлежит другому анализу".into());
    }
    if ["complete", "failed", "cancelled"].contains(&status["status"].as_str().unwrap_or("")) {
        return Ok(());
    }
    let failed_stage = status["stage"].as_str().unwrap_or("Анализ").to_owned();
    status["status"] = json!("failed");
    status["stage"] = json!("Ошибка анализа");
    status["failed_stage"] = json!(failed_stage);
    status["error"] = json!(error);
    status["pipeline_complete"] = json!(false);
    status["current_gpu_job_id"] = Value::Null;
    finish_timed_stage(&mut status);
    save(path, &status)?;
    log_event(path, &failed_stage, "error", error)
}

async fn supervise_pipeline_worker(
    worker: tokio::task::JoinHandle<()>,
    active: &Mutex<Option<Running>>,
    path: &Path,
    project_id: &str,
    id: &str,
    retain_slot: bool,
) {
    let Err(join_error) = worker.await else {
        return;
    };
    let message = if join_error.is_panic() {
        let panic = join_error.into_panic();
        let detail = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("неизвестная ошибка");
        format!("Аварийное завершение обработчика анализа: {detail}")
    } else {
        format!("Фоновая задача анализа неожиданно прервана: {join_error}")
    };
    let mut slot = active.lock().await;
    if !slot
        .as_ref()
        .is_some_and(|run| run.contains_run(project_id, id))
    {
        return;
    }
    if let Err(error) = record_pipeline_worker_failure(path, project_id, id, &message) {
        eprintln!("Pipeline worker {id} failed: {message}; status write: {error}");
    }
    if !retain_slot {
        *slot = None;
    }
}
pub async fn start(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<StartRequest>,
) -> ApiResult<Value> {
    start_batch(state, headers, request, false).await
}

pub async fn start_project(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<StartRequest>,
) -> ApiResult<Value> {
    start_batch(state, headers, request, true).await
}

fn validate_start_profile(request: &StartRequest) -> Result<(), ApiError> {
    if request.mode == PipelineMode::Full
        && request.post.touches.require_pose
        && !request.gpu.pose_enabled
    {
        return Err(storage::bad_request(
            "Для алгоритма касаний по движению точек тела включите GPU-анализ позы",
        ));
    }
    if request
        .client_request_id
        .as_deref()
        .is_some_and(|id| !valid_id(id))
    {
        return Err(storage::bad_request("Некорректный идентификатор запроса"));
    }
    if !(1..=9).contains(&request.gpu.stride)
        || request.gpu.cpu_threads > 128
        || request.gpu.gpu_memory_limit_mb > 131072
        || !(1..=30).contains(&request.gpu.pose_stride)
        || !(1..=30).contains(&request.gpu.touch_pose_stride)
        || !(1..=30).contains(&request.gpu.background_pose_stride)
    {
        return Err(storage::bad_request(
            "GPU-профиль вне допустимых диапазонов",
        ));
    }
    if request.mode.allows_gpu_execution()
        && request.gpu.pose_mode != crate::pose_inference::PoseInputMode::Photo
    {
        return Err(storage::bad_request(
            "Рамки игроков и RTMPose должны использовать общий шаг кадров",
        ));
    }
    request
        .post
        .tracking
        .validate()
        .map_err(storage::bad_request)?;
    request
        .post
        .touches
        .validate()
        .map_err(storage::bad_request)?;
    request
        .post
        .rallies
        .validate()
        .map_err(storage::bad_request)?;
    Ok(())
}

async fn start_batch(
    state: Arc<AppState>,
    headers: HeaderMap,
    mut request: StartRequest,
    all_videos: bool,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    if all_videos && request.mode != PipelineMode::Full {
        return Err(storage::bad_request(
            "Очередь проекта требует полного анализа",
        ));
    }
    if !state.writes_enabled {
        return Err(storage::conflict(
            "Запуск анализа возможен только в рабочем хранилище",
        ));
    }
    if request.post.touches.algorithm == TouchAlgorithm::TrajectoryTcn {
        volley_inference::trajectory_tcn::checkpoint_sha256(
            &state
                .repository_root
                .join("models")
                .join(volley_inference::trajectory_tcn::MODEL_FILENAME),
        )
        .map_err(storage::bad_request)?;
    }
    storage::project_id(&request.project_id)?;
    storage::video_name(&request.video)?;
    if request.mode.allows_gpu_execution()
        && projects::canonical_source(&state, &request.video).is_ok()
    {
        request.analysis_range =
            crate::video_edits::analysis_range(&state, &request.project_id, &request.video)?;
    }
    if request.mode == PipelineMode::Post
        && request
            .source_analysis_id
            .as_deref()
            .is_none_or(|id| !valid_id(id))
    {
        return Err(storage::bad_request(
            "Для CPU-постобработки выберите GPU-анализ в шапке",
        ));
    }
    if request.mode != PipelineMode::Post && request.source_analysis_id.is_some() {
        return Err(storage::bad_request(
            "GPU-источник указывается только при CPU-пересчёте",
        ));
    }
    validate_start_profile(&request)?;
    let project = projects::existing_project(&state, &request.project_id)?;
    if !project["videos"].as_array().is_some_and(|videos| {
        videos
            .iter()
            .any(|video| video.as_str() == Some(&request.video))
    }) {
        return Err(storage::bad_request(
            "Видео не принадлежит выбранному проекту",
        ));
    }
    let videos = if all_videos {
        queue_video_order(&project, &request.video)?
    } else {
        vec![request.video.clone()]
    };
    validate_queue_profiles(&request, &videos, all_videos)?;
    let queue_profiles_sha256 = request.queue_profiles_digest();
    let _admission = admission_guard().await;
    let mut slot = manager().lock().await;
    // If the HTTP response was lost after scheduling, replaying the same
    // client token must recover the saved run rather than launch a duplicate.
    if let Some(token) = request.client_request_id.as_deref() {
        let existing_root = folder(&state, &request.project_id, false)?;
        let latest_path = existing_root.join("latest.json");
        if latest_path.exists() {
            let latest = read(&latest_path)?;
            if let Some(previous_id) = latest["id"].as_str().filter(|id| valid_id(id)) {
                let previous = read(&status_path(&existing_root, previous_id)?)?;
                if previous["client_request_id"] == token {
                    if previous["schema"] != SCHEMA
                        || previous["project_id"] != request.project_id
                        || (if all_videos {
                            previous["queue_start_video"] != request.video
                        } else {
                            previous["video"] != request.video || !previous["queue_id"].is_null()
                        })
                        || previous["queue_profiles_sha256"] != json!(queue_profiles_sha256)
                        || previous["mode"] != json!(request.mode)
                        || previous["source_analysis_id"] != json!(request.source_analysis_id)
                        || !decode_profiles(
                            previous["profile"]["gpu"].clone(),
                            previous["profile"]["post"].clone(),
                        )
                        .is_ok_and(|(gpu, post)| {
                            json!(gpu) == json!(request.gpu) && json!(post) == json!(request.post)
                        })
                    {
                        return Err(storage::conflict("Идентификатор запроса уже использован"));
                    }
                    return Ok(Json(json!({"schema":SCHEMA,"id":previous_id,
                        "status":previous["status"],"queue_id":previous["queue_id"],"recovered":true})));
                }
            }
        }
    }
    if slot.is_some() {
        return Err(storage::conflict("Другой запуск анализа уже выполняется"));
    }
    if request.mode != PipelineMode::Post
        && (state.full_inference.has_active().await || state.quick_reel.has_active().await)
    {
        return Err(storage::conflict(
            "GPU занят отдельным запуском или быстрым рилсом",
        ));
    }
    let _guard = project_guard::write_guard(&state, &request.project_id)?;
    let project_id = request.project_id.clone();
    let start_video = request.video.clone();
    if all_videos
        && queue_video_order(
            &projects::existing_project(&state, &project_id)?,
            &start_video,
        )? != videos
    {
        return Err(storage::conflict(
            "Список видео изменился после выбора профилей",
        ));
    }
    let court_revisions = videos
        .iter()
        .map(|video| crate::video_edits::require_court_revision(&state, &project_id, video))
        .collect::<Result<Vec<_>, _>>()?;
    let root = folder(&state, &project_id, true)?;
    let ids: Vec<String> = videos
        .iter()
        .map(|_| Uuid::new_v4().simple().to_string())
        .collect();
    let id = ids[0].clone();
    let queue_id = all_videos.then_some(id.clone());
    let mut jobs = Vec::with_capacity(videos.len());
    for (index, (video, run_id)) in videos.into_iter().zip(ids.iter()).enumerate() {
        let mut scoped = request.scoped_to_video(&video);
        let mut queue_snapshot = json!({"court":court_revisions[index]});
        if all_videos {
            storage::video_name(&scoped.video)?;
            let source = projects::canonical_source(&state, &scoped.video)?;
            scoped.analysis_range =
                crate::video_edits::analysis_range(&state, &project_id, &scoped.video)?;
            let duration = match scoped.analysis_range {
                Some(range) => range.end_sec - range.start_sec,
                None => {
                    volley_media::probe(&source)
                        .map_err(storage::bad_request)?
                        .duration_sec
                }
            };
            if !duration.is_finite() || duration <= 0.0 {
                return Err(storage::bad_request(
                    "Некорректная длительность видео очереди",
                ));
            }
            let (bytes, mtime_ns) =
                crate::inference_stream::source_stamp(&source).map_err(storage::bad_request)?;
            queue_snapshot = json!({"weight":duration,"source":{"bytes":bytes,"mtime_ns":mtime_ns},"court":court_revisions[index]});
        }
        let mut status = json!({
            "schema":SCHEMA,"id":run_id,"project_id":project_id,"video":scoped.video,
            "status":"queued","stage":"В очереди","progress_percent":0.0,
            "started_ms":if index == 0 { Some(epoch_ms()) } else { None },
            "finished_ms":null,"stages":[],"stage_samples":[],
            "profile":{"gpu":scoped.gpu,"post":scoped.post},
            "mode":scoped.mode,"source_analysis_id":scoped.source_analysis_id,
            "client_request_id":scoped.client_request_id,
            "analysis_range":scoped.analysis_range,
            "gpu_model_stack":requested_gpu_model_stack(scoped.gpu.person_detector),
            "pose_pipeline_policy":cpu_retrack::POSE_REFINEMENT_POLICY,
            "zero_copy_request":zero_copy_requested(),"zero_copy_active":false,
            "rally_block_pipeline":rally_pipeline_enabled(&scoped),
            "gpu_jobs":{},"cpu_runs":{},"analysis_id":null,
            "gpu_rally_mask":null,"gpu_rally_mask_sha256":null,
            "postprocessing_id":null,"current_gpu_job_id":null,
            "source_identity":null,"gpu_reused":false,"result_reused":false,
            "reused_from_id":null,"pipeline_complete":false,"completion_quality":null,"error":null,
            "queue_id":queue_id,"queue_runs":if all_videos { &ids[..] } else { &[] },
            "queue_start_video":if all_videos { Some(&start_video) } else { None },
            "queue_weight_sec":queue_snapshot["weight"],"queued_source_identity":queue_snapshot["source"],
            "queued_court_revision":queue_snapshot["court"]
        });
        status["queue_profiles_sha256"] = json!(queue_profiles_sha256);
        jobs.push(QueuedRun {
            id: run_id.clone(),
            request: scoped,
            path: root.join(run_id).join("status.json"),
            status,
        });
    }
    let mut created = Vec::new();
    let persisted = (|| -> Result<(), String> {
        for job in &jobs {
            let dir = job.path.parent().ok_or("Missing run directory")?;
            fs::create_dir(dir).map_err(|e| format!("Создание запуска: {e}"))?;
            created.push(dir.to_owned());
            save(&job.path, &job.status)?;
        }
        log_queued_start(&jobs[0])?;
        save(&root.join("latest.json"), &json!({"id":id}))
    })();
    if let Err(error) = persisted {
        for dir in created {
            let _ = fs::remove_dir_all(dir);
        }
        return Err(storage::internal(error));
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    *slot = Some(Running {
        project_id: project_id.clone(),
        id: id.clone(),
        cancel: cancelled.clone(),
        queue_runs: ids,
    });
    drop(_guard);
    tokio::spawn(run_project_queue(state, project_id, jobs, cancelled));
    Ok(Json(
        json!({"schema":SCHEMA,"id":id,"status":"queued","queue_id":queue_id}),
    ))
}

fn queue_video_order(project: &Value, first: &str) -> Result<Vec<String>, ApiError> {
    let videos = project["videos"]
        .as_array()
        .ok_or_else(|| storage::bad_request("Нет видео проекта"))?;
    if !videos.iter().any(|video| video.as_str() == Some(first)) {
        return Err(storage::bad_request("Видео не принадлежит проекту"));
    }
    let mut ordered = vec![first.to_owned()];
    for video in videos {
        let video = video
            .as_str()
            .ok_or_else(|| storage::bad_request("Некорректное видео проекта"))?;
        if !ordered.iter().any(|item| item == video) {
            ordered.push(video.to_owned());
        }
    }
    if ordered.len() > 100 {
        return Err(storage::bad_request("Очередь проекта ограничена 100 видео"));
    }
    Ok(ordered)
}

struct QueuedRun {
    id: String,
    request: StartRequest,
    path: PathBuf,
    status: Value,
}

fn log_queued_start(job: &QueuedRun) -> Result<(), String> {
    log_native(
        &job.path,
        "Запуск",
        "studio_pipeline::start",
        json!({
            "project_id":job.request.project_id,"video":job.request.video,"gpu":job.request.gpu,
            "post":job.request.post,"mode":job.request.mode,"source_analysis_id":job.request.source_analysis_id,
            "client_request_id":job.request.client_request_id,"queue_id":job.status["queue_id"]
        }),
    )
}

fn mark_unstarted(job: &mut QueuedRun, cancelled: bool, error: Option<&str>) {
    job.status["status"] = json!(if cancelled {
        "cancelled"
    } else {
        "interrupted"
    });
    job.status["stage"] = json!(if cancelled {
        "Очередь отменена"
    } else {
        "Не запущен: ошибка предыдущего видео"
    });
    job.status["error"] = json!(error);
    job.status["finished_ms"] = json!(epoch_ms());
    if let Err(error) = save(&job.path, &job.status) {
        eprintln!("Queue status {}: {error}", job.id);
    }
}

fn independent_queues_requested(value: Option<&str>) -> bool {
    matches!(value, None | Some("1") | Some("true"))
}

fn independent_profile(request: &StartRequest) -> bool {
    request.mode == PipelineMode::Full
        && rally_pipeline_enabled(request)
        && request.gpu.person_detector == DetectorVariant::RfDetrMedium
        && request.gpu.gpu_memory_limit_mb == 0
}

async fn run_project_queue(
    state: Arc<AppState>,
    project: String,
    jobs: Vec<QueuedRun>,
    cancel: Arc<AtomicBool>,
) {
    let first = jobs[0].id.clone();
    let independent_requested = independent_queues_requested(
        std::env::var("VOLLEY_INDEPENDENT_GPU_QUEUES")
            .ok()
            .as_deref(),
    ) && jobs.len() > 1
        && jobs.iter().all(|job| independent_profile(&job.request));
    let queue_permit = if independent_requested {
        acquire_gpu(&state.inference_limit, &cancel).await.ok()
    } else {
        None
    };
    let mut turns = video_turns(jobs.len());
    if queue_permit.is_some() {
        let queues = Arc::new(inference_full::stage_queue::IndependentStages::default());
        for turn in &mut turns {
            turn.independent = Some(queues.clone());
        }
    }
    let mut pending = jobs.into_iter().zip(turns).enumerate();
    let mut workers = tokio::task::JoinSet::new();
    let mut failed = false;
    loop {
        while workers.len() < 2 && !failed && !cancel.load(Ordering::Acquire) {
            let Some((index, (mut job, turn))) = pending.next() else {
                break;
            };
            if index > 0 {
                job.status["started_ms"] = json!(epoch_ms());
                if let Err(error) =
                    save(&job.path, &job.status).and_then(|()| log_queued_start(&job))
                {
                    mark_unstarted(&mut job, false, Some(&error));
                    failed = true;
                    cancel.store(true, Ordering::Release);
                    break;
                }
            }
            let app = state.clone();
            let cancellation = cancel.clone();
            let project = project.clone();
            workers.spawn(async move {
                let path = job.path.clone();
                let id = job.id.clone();
                let worker = tokio::spawn(perform(app, job, cancellation, turn));
                supervise_pipeline_worker(worker, manager(), &path, &project, &id, true).await;
                read(&path).is_ok_and(|status| status["status"] == "complete")
            });
        }
        let Some(outcome) = workers.join_next().await else {
            break;
        };
        if !matches!(outcome, Ok(true)) && !cancel.load(Ordering::Acquire) {
            failed = true;
            cancel.store(true, Ordering::Release);
        }
    }
    for (_, (mut job, _)) in pending {
        mark_unstarted(
            &mut job,
            !failed && cancel.load(Ordering::Acquire),
            failed.then_some("Предыдущий анализ не завершился успешно"),
        );
    }
    let mut slot = manager().lock().await;
    if slot
        .as_ref()
        .is_some_and(|active| active.contains_run(&project, &first))
    {
        *slot = None;
    }
}

// A pipeline's comparable completed runs cannot change while it is active.
// Compute these once per run rather than rereading up to 25 log files on every
// 1.8-second status poll. The key includes the run ID for automatic invalidation.
fn eta_reference_history(state: &AppState, project: &str, run: &str) -> Result<Value, ApiError> {
    type Cached = Option<(String, String, Value)>;
    static CACHE: OnceLock<std::sync::Mutex<Cached>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(None));
    if let Ok(guard) = cache.lock() {
        if let Some((saved_project, saved_run, history)) = guard.as_ref() {
            if saved_project == project && saved_run == run {
                return Ok(history.clone());
            }
        }
    }
    let history = history_rows(state, project, 0)?;
    if let Ok(mut guard) = cache.lock() {
        *guard = Some((project.to_owned(), run.to_owned(), history.clone()));
    }
    Ok(history)
}

// Stage-aware ETA from comparable completed runs on the same source video.
// We use observed GPU throughput only inside the current stage; CPU stages
// have no reliable fractional progress and use their measured durations.
fn stage_aware_remaining(current: &Value, history: &Value, now: u64) -> Option<u64> {
    let active = current["stages"].as_array()?.last()?;
    let name = active["name"].as_str()?;
    let elapsed = now.saturating_sub(active["started_ms"].as_u64()?);
    let mut this_stage = Vec::new();
    let mut future = Vec::new();
    for run in history["runs"]
        .as_array()?
        .iter()
        .filter(|run| {
            run["status"] == "complete"
                && run["mode"] == current["mode"]
                && run["video"] == current["video"]
                && run["gpu_reused"] == current["gpu_reused"]
        })
        .take(20)
    {
        let Some(stages) = run["stages"].as_array() else {
            continue;
        };
        let Some(index) = stages.iter().position(|item| item["name"] == name) else {
            continue;
        };
        let Some(duration) = stages[index]["duration_ms"].as_u64() else {
            continue;
        };
        let tail = stages[index + 1..]
            .iter()
            .map(|item| item["duration_ms"].as_u64())
            .collect::<Option<Vec<_>>>();
        let Some(tail) = tail else { continue };
        this_stage.push(duration);
        future.push(tail.into_iter().sum::<u64>());
    }
    fn median(mut values: Vec<u64>) -> Option<u64> {
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        Some(values[values.len() / 2])
    }
    let historical_stage = median(this_stage)?;
    let historical_future = median(future)?;
    let mut stage_left = historical_stage.saturating_sub(elapsed);
    let target = if name == "Подготовка исходного видео" {
        Some(8.0)
    } else if name == "GPU · VballNet" {
        Some(44.0)
    } else if name.starts_with("GPU · RF-DETR ") {
        Some(80.0)
    } else if name == "GPU · RTMPose-M: позы игроков" || name == "GPU · RTMPose-M" {
        Some(84.5)
    } else {
        None
    };
    if let (Some(target), Some(samples)) = (target, current["stage_samples"].as_array()) {
        if samples.len() >= 2 {
            let newest = samples.last()?;
            let oldest = samples.iter().find(|sample| {
                newest["timestamp_ms"]
                    .as_u64()
                    .unwrap_or(0)
                    .saturating_sub(sample["timestamp_ms"].as_u64().unwrap_or(0))
                    >= 12_000
            });
            if let Some(oldest) = oldest {
                let change = newest["percent"].as_f64()? - oldest["percent"].as_f64()?;
                let span = newest["timestamp_ms"]
                    .as_u64()?
                    .saturating_sub(oldest["timestamp_ms"].as_u64()?);
                if change >= 0.25 && span >= 12_000 {
                    let projected = ((target - newest["percent"].as_f64()?).max(0.0) * span as f64
                        / change) as u64;
                    let bounded = projected.min(historical_stage.saturating_mul(5).max(30_000));
                    stage_left = ((stage_left as f64 * 0.35) + (bounded as f64 * 0.65)) as u64;
                }
            }
        }
    }
    let remaining = stage_left.saturating_add(historical_future);
    (remaining <= 7 * 24 * 3_600_000).then_some(remaining.max(1000))
}

pub(crate) fn read_queue_rows(
    root: &Path,
    project: &str,
    queue_id: &str,
) -> Result<Vec<Value>, ApiError> {
    let first = read(&status_path(root, queue_id)?)?;
    let ids = first["queue_runs"]
        .as_array()
        .ok_or_else(|| storage::bad_request("Нет списка видео очереди"))?;
    if ids.is_empty() || ids.len() > 100 || ids[0].as_str() != Some(queue_id) {
        return Err(storage::bad_request("Некорректный список очереди"));
    }
    let mut seen = BTreeSet::new();
    ids.iter()
        .map(|value| {
            let id = value
                .as_str()
                .filter(|id| valid_id(id))
                .ok_or_else(|| storage::bad_request("Некорректный ID видео очереди"))?;
            if !seen.insert(id) {
                return Err(storage::bad_request("Повтор ID видео очереди"));
            }
            let row = read(&status_path(root, id)?)?;
            if row["schema"] != SCHEMA
                || row["project_id"] != project
                || row["id"] != id
                || row["queue_id"] != queue_id
                || row["queue_runs"] != first["queue_runs"]
            {
                return Err(storage::bad_request("Видео принадлежит другой очереди"));
            }
            Ok(row)
        })
        .collect()
}

pub(crate) fn project_launches(project_root: &Path, project: &str) -> Result<Vec<Value>, ApiError> {
    let root = project_root.join("native_studio_pipelines");
    if !root.exists() {
        return Ok(Vec::new());
    }
    storage::safe_folder(&root).map_err(storage::bad_request)?;
    let names = storage::read_json(&project_root.join("native_analysis_names.json"))?
        .unwrap_or_else(|| json!({}));
    let mut launches = Vec::new();
    let mut count = 0;
    for entry in fs::read_dir(&root)
        .map_err(|error| storage::internal(format!("Read project launches: {error}")))?
    {
        let entry = entry.map_err(|error| storage::internal(format!("Read launch: {error}")))?;
        let id = entry.file_name().to_string_lossy().into_owned();
        if !valid_id(&id) {
            continue;
        }
        count += 1;
        if count > 1500 {
            return Err(storage::bad_request("Too many project launch records"));
        }
        let Ok(path) = status_path(&root, &id) else {
            continue;
        };
        let Ok(head) = read(&path) else { continue };
        if head["schema"] != SCHEMA
            || head["project_id"] != project
            || head["id"] != id
            || head["mode"] != "full"
            || head["queue_id"] != id
        {
            continue;
        }
        let Ok(rows) = read_queue_rows(&root, project, &id) else {
            continue;
        };
        let mut videos = BTreeSet::new();
        if rows.iter().any(|row| {
            row["mode"] != "full"
                || !row["video"]
                    .as_str()
                    .is_some_and(|video| storage::video_name(video).is_ok() && videos.insert(video))
        }) {
            continue;
        }
        let members: Vec<_> = rows
            .iter()
            .map(|row| {
                let jobs: BTreeSet<_> = [
                    &row["gpu_jobs"]["ball"],
                    &row["gpu_jobs"]["players"],
                    &row["current_gpu_job_id"],
                ]
                .into_iter()
                .filter_map(Value::as_str)
                .filter(|id| valid_id(id))
                .collect();
                json!({"video":row["video"],"analysis_id":row["analysis_id"],
                "postprocessing_id":row["postprocessing_id"],"gpu_job_ids":jobs})
            })
            .collect();
        if !members.iter().any(|row| {
            row["analysis_id"].as_str().is_some_and(valid_id)
                || row["gpu_job_ids"]
                    .as_array()
                    .is_some_and(|ids| !ids.is_empty())
        }) {
            continue;
        }
        let created_at = head["started_ms"]
            .as_u64()
            .map(|ms| ms / 1000)
            .or_else(|| {
                entry
                    .metadata()
                    .ok()?
                    .created()
                    .ok()?
                    .duration_since(UNIX_EPOCH)
                    .ok()
                    .map(|t| t.as_secs())
            })
            .unwrap_or_default();
        launches.push(json!({
            "id":id,"created_at":created_at,"members":members,
            "display_name":names[format!("launch:{id}")],
            "hidden":names[format!("hidden_launch:{id}")] == true
        }));
    }
    launches.sort_by(|a, b| {
        a["created_at"]
            .as_u64()
            .cmp(&b["created_at"].as_u64())
            .then_with(|| a["id"].as_str().cmp(&b["id"].as_str()))
    });
    Ok(launches)
}

fn pipeline_frames_processed(row: &Value) -> Option<u64> {
    row["analysis_frames_processed"].as_u64().or_else(|| {
        row["stages"].as_array()?.iter().rev().find_map(|stage| {
            if matches!(
                stage["id"].as_str(),
                Some("ball" | "ball_stabilizer" | "rallies")
            ) {
                stage["frames_processed"].as_u64()
            } else {
                None
            }
        })
    })
}

fn progress_remaining(elapsed_ms: u64, progress_percent: f64) -> Option<u64> {
    if elapsed_ms < 30_000 || !(1.0..100.0).contains(&progress_percent) {
        return None;
    }
    let estimate = elapsed_ms as f64 * (100.0 - progress_percent) / progress_percent;
    (estimate.is_finite() && estimate <= (7 * 24 * 3_600_000) as f64)
        .then(|| (estimate.round() as u64).max(1000))
}

fn queue_progress(rows: &[Value], active: bool, now: u64) -> Value {
    let mut weighted_progress = 0.0;
    let mut total_weight = 0.0;
    let mut completed = 0;
    let mut frames_processed = 0_u64;
    let mut rallies_total = 0_u64;
    let mut touch_detector_count = None::<u64>;
    let videos: Vec<Value> = rows
        .iter()
        .map(|row| {
            let weight = row["queue_weight_sec"]
                .as_f64()
                .filter(|weight| weight.is_finite() && *weight > 0.0)
                .unwrap_or(1.0);
            let state = row["status"].as_str().unwrap_or("interrupted");
            let state = if !active && matches!(state, "queued" | "running" | "cancelling") {
                "interrupted"
            } else {
                state
            };
            let percent = if state == "complete" {
                completed += 1;
                100.0
            } else {
                row["progress_percent"]
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .unwrap_or(0.0)
                    .clamp(0.0, 100.0)
            };
            weighted_progress += weight * percent;
            total_weight += weight;
            let rallies = row["rallies_published"]
                .as_u64()
                .or_else(|| row["rally_stage_progress"]["pose"]["rallies_total"].as_u64())
                .or_else(|| row["rally_stage_progress"]["boxes"]["rallies_total"].as_u64())
                .or_else(|| {
                    row["gpu_rally_mask"]["merged"]
                        .as_array()
                        .map(|rallies| rallies.len() as u64)
                });
            let frames = pipeline_frames_processed(row);
            let ball_frames_total = if state == "running" && row["stage_id"] == "ball" {
                row["stages"]
                    .as_array()
                    .and_then(|stages| stages.last())
                    .filter(|stage| stage["id"] == "ball")
                    .and_then(|stage| stage["frames_total"].as_u64())
            } else {
                None
            };
            frames_processed += frames.unwrap_or(0);
            rallies_total += rallies.unwrap_or(0);
            if let Some(count) = row["touch_detector_count"].as_u64() {
                touch_detector_count = Some(touch_detector_count.unwrap_or(0).saturating_add(count));
            }
            json!({"id":row["id"],"video":row["video"],"status":state,"progress_percent":percent,
            "frames_processed":frames,"ball_frames_total":ball_frames_total,
            "touch_detector_count":row["touch_detector_count"],
            "stage":row["stage"],"stage_id":row["stage_id"],
            "rally_stage_progress":if state == "running" { row["rally_stage_progress"].clone() } else { Value::Null },
            "profile":{"gpu":{"person_detector":row["profile"]["gpu"]["person_detector"]}},
            "rallies_completed":row["rally_stage_progress"]["pose"]["rallies_completed"],
            "analysis_range":row["analysis_range"],"duration_sec":weight,
            "court_revision":row["queued_court_revision"],"rallies_total":rallies,
            "analysis_id":row["analysis_id"],"postprocessing_id":row["postprocessing_id"],
            "error":row["error"]})
        })
        .collect();
    let progress_percent = if total_weight > 0.0 {
        weighted_progress / total_weight
    } else {
        0.0
    };
    let started = rows
        .iter()
        .filter_map(|row| row["started_ms"].as_u64())
        .min();
    let elapsed = if active {
        started.map(|start| now.saturating_sub(start))
    } else {
        None
    };
    let remaining = if active
        && rows.iter().all(|row| {
            matches!(
                row["status"].as_str(),
                Some("queued" | "running" | "complete")
            )
        }) {
        elapsed.and_then(|ms| progress_remaining(ms, progress_percent))
    } else {
        None
    };
    json!({"active":active,"completed_videos":completed,"total_videos":rows.len(),
        "progress_percent":progress_percent,"started_ms":started,"elapsed_ms":elapsed,
        "remaining_ms":remaining,
        "frames_processed":frames_processed,"rallies_total":rallies_total,
        "touch_detector_count":touch_detector_count,"videos":videos})
}

fn queue_focus(rows: &[Value]) -> Option<&Value> {
    let started_active = |row: &&Value| {
        row["started_ms"].as_u64().is_some()
            && matches!(
                row["status"].as_str(),
                Some("queued" | "running" | "cancelling")
            )
    };
    rows.iter()
        .filter(started_active)
        .find(|row| {
            row["status"] == "running"
                && !row["stage"]
                    .as_str()
                    .is_some_and(|stage| stage.starts_with("Ожидание очереди"))
        })
        .or_else(|| rows.iter().find(started_active))
        .or_else(|| rows.iter().find(|row| row["status"] == "failed"))
        .or_else(|| rows.iter().find(|row| row["status"] == "cancelled"))
        .or_else(|| {
            rows.iter()
                .rfind(|row| row["started_ms"].as_u64().is_some())
        })
}

fn status_snapshot(
    state: &AppState,
    query: &StatusQuery,
) -> Result<Option<(String, Value)>, ApiError> {
    let root = folder(state, &query.project_id, false)?;
    if let Some(queue_id) = query.queue_id.as_deref() {
        if query.id.is_some() {
            return Err(storage::bad_request("Укажите запуск или очередь"));
        }
        let rows = read_queue_rows(&root, &query.project_id, queue_id)?;
        let row = queue_focus(&rows)
            .cloned()
            .ok_or_else(|| storage::bad_request("Очередь не запущена"))?;
        let id = row["id"]
            .as_str()
            .ok_or_else(|| storage::bad_request("Нет ID запуска"))?
            .to_owned();
        return Ok(Some((id, row)));
    }
    let id = match query.id.as_deref() {
        Some(id) => id.to_owned(),
        None => {
            if !root.join("latest.json").exists() {
                return Ok(None);
            }
            let latest = read(&root.join("latest.json"))?;
            latest["id"]
                .as_str()
                .filter(|id| valid_id(id))
                .ok_or_else(|| storage::bad_request("Неверный указатель на последний запуск"))?
                .to_owned()
        }
    };
    let result = read(&status_path(&root, &id)?)?;
    if result["schema"] != SCHEMA || result["project_id"] != query.project_id || result["id"] != id
    {
        return Err(storage::bad_request("Статус принадлежит другому проекту"));
    }
    if query.id.is_none() {
        if let Some(queue_id) = result["queue_id"].as_str() {
            let rows = read_queue_rows(&root, &query.project_id, queue_id)?;
            if let Some(row) = queue_focus(&rows) {
                let id = row["id"]
                    .as_str()
                    .ok_or_else(|| storage::bad_request("Нет ID запуска"))?
                    .to_owned();
                return Ok(Some((id, row.clone())));
            }
        }
    }
    Ok(Some((id, result)))
}

pub async fn status(
    State(state): State<Arc<AppState>>,
    Query(query): Query<StatusQuery>,
) -> ApiResult<Value> {
    let project_id = query.project_id.clone();
    let read_state = state.clone();
    let snapshot = tokio::task::spawn_blocking(move || status_snapshot(&read_state, &query))
        .await
        .map_err(|e| storage::internal(format!("Read pipeline status worker: {e}")))??;
    let Some((id, mut result)) = snapshot else {
        return Ok(Json(
            json!({"schema":SCHEMA,"status":"idle","project_id":project_id}),
        ));
    };
    let slot = manager().lock().await;
    if let Some(queue_id) = result["queue_id"].as_str().map(str::to_owned) {
        let active = slot
            .as_ref()
            .is_some_and(|entry| entry.contains_run(&project_id, &queue_id));
        let root = folder(&state, &project_id, false)?;
        let queue_project = project_id.clone();
        let rows =
            tokio::task::spawn_blocking(move || read_queue_rows(&root, &queue_project, &queue_id))
                .await
                .map_err(|e| storage::internal(format!("Read queue worker: {e}")))??;
        if let Some(current) = rows.iter().find(|row| row["id"] == id) {
            result = current.clone();
        }
        result["project_queue"] = queue_progress(&rows, active, epoch_ms());
    }
    if result["status"] == "running" || result["status"] == "queued" {
        match slot
            .as_ref()
            .filter(|entry| entry.contains_run(&project_id, &id))
        {
            Some(entry) if entry.cancel.load(Ordering::Acquire) => {
                result["status"] = json!("cancelling");
                result["stage"] = json!("Отмена запрошена · остановка текущей операции");
                result["cancel_requested"] = json!(true);
            }
            Some(_) => {}
            None => {
                result["status"] = json!("interrupted");
                result["stage"] = json!("Сервер перезапущен");
                result["error"] = json!("Анализ не завершён до перезапуска сервера");
            }
        }
    }
    drop(slot);
    // Presentation fields are computed at read time, never persisted as fake
    // progress. Existing (pre-timeline) runs still retain their log history.
    let now = epoch_ms();
    if let Some(start) = result["started_ms"].as_u64() {
        let end = result["finished_ms"].as_u64().unwrap_or(now);
        result["elapsed_ms"] = json!(end.saturating_sub(start));
    }
    if ["queued", "running", "cancelling"].contains(&result["status"].as_str().unwrap_or(""))
        && result["stages"]
            .as_array()
            .is_some_and(|items| !items.is_empty())
    {
        let app = state.clone();
        let project = project_id.clone();
        let run = id.clone();
        if let Ok(Ok(history)) =
            tokio::task::spawn_blocking(move || eta_reference_history(&app, &project, &run)).await
        {
            result["remaining_ms"] =
                stage_aware_remaining(&result, &history, now).map_or(Value::Null, |ms| json!(ms));
        }
    }
    if matches!(result["status"].as_str(), Some("queued" | "running"))
        && result["remaining_ms"].is_null()
    {
        result["remaining_ms"] = json!(result["elapsed_ms"]
            .as_u64()
            .zip(result["progress_percent"].as_f64())
            .and_then(|(elapsed, progress)| progress_remaining(elapsed, progress)));
    }
    Ok(Json(result))
}
pub async fn cancel(
    State(_state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<CancelRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    storage::project_id(&request.project_id)?;
    if !valid_id(&request.id) {
        return Err(storage::bad_request("Неверный ID анализа"));
    }
    let slot = manager().lock().await;
    let running = slot
        .as_ref()
        .filter(|entry| entry.contains_run(&request.project_id, &request.id))
        .ok_or_else(|| storage::not_found("Нет активного запуска для отмены"))?;
    let active_id = running.id.clone();
    running.cancel.store(true, Ordering::Release);
    drop(slot);
    // Promptly signal the active GPU job; never wait for the pipeline's next
    // poll cycle before forwarding a user cancellation.
    let path = folder(&_state, &request.project_id, false)?
        .join(&active_id)
        .join("status.json");
    let gpu_job_id = read(&path).ok().and_then(|snapshot| {
        snapshot["current_gpu_job_id"]
            .as_str()
            .filter(|id| valid_id(id))
            .map(str::to_owned)
    });
    let mut gpu_notified = false;
    if let Some(job_id) = gpu_job_id {
        let payload = json!({"project_id":request.project_id,"job_id":job_id});
        if let Ok(config) = serde_json::from_value::<inference_full::CancelRequest>(payload) {
            gpu_notified =
                inference_full::cancel(State(_state.clone()), HeaderMap::new(), Json(config))
                    .await
                    .is_ok();
        }
    }
    // The acknowledgement and UI must not depend on journal writes.
    let _ = log_event(&path, "Анализ", "warn", "Получен запрос на отмену");
    Ok(Json(json!({"id":request.id,"cancel_requested":true,
        "gpu_cancel_forwarded":gpu_notified})))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_focus_does_not_hide_running_or_failed_video() {
        let mut rows = vec![
            json!({"id":"first","started_ms":1,"status":"running"}),
            json!({"id":"second","started_ms":2,"status":"running"}),
            json!({"id":"third","status":"queued"}),
        ];
        assert_eq!(queue_focus(&rows).unwrap()["id"], "first");
        rows[0]["stage"] = json!("Ожидание очереди игроков");
        rows[1]["stage"] = json!("GPU · VBallNet: траектория мяча и розыгрыши");
        rows[1]["stage_id"] = json!("ball");
        assert_eq!(queue_focus(&rows).unwrap()["id"], "second");
        rows[0]["stage"] = json!("GPU · RF-DETR: игровые интервалы");
        rows[1]["stage"] = json!("Ожидание очереди VBallNet");
        assert_eq!(queue_focus(&rows).unwrap()["id"], "first");
        rows[0]["status"] = json!("complete");
        assert_eq!(queue_focus(&rows).unwrap()["id"], "second");
        rows[0]["status"] = json!("failed");
        rows[1]["status"] = json!("cancelled");
        assert_eq!(queue_focus(&rows).unwrap()["id"], "first");
    }

    #[test]
    fn project_queue_profiles_are_independent_and_validated() {
        let input = json!({
            "project_id":"abc123def456","video":"a.mp4",
            "gpu":{"pose_stride":1},
            "video_profiles":{
                "a.mp4":{"gpu":{"pose_stride":1}},
                "b.mp4":{"gpu":{"pose_stride":5,"person_detector":"rfdetr-large"},
                    "post":{"interpolate_players":false}}
            }
        });
        let request: StartRequest = serde_json::from_value(input.clone()).unwrap();
        let videos = vec!["a.mp4".to_owned(), "b.mp4".to_owned()];
        assert!(validate_queue_profiles(&request, &videos, true).is_ok());
        let a = request.scoped_to_video("a.mp4");
        let b = request.scoped_to_video("b.mp4");
        assert_eq!(a.gpu.pose_stride, 1);
        assert_eq!(b.gpu.pose_stride, 5);
        assert_eq!(b.gpu.person_detector, DetectorVariant::RfDetrLarge);
        assert!(a.post.interpolate_players);
        assert!(!b.post.interpolate_players);
        assert!(a.video_profiles.is_empty() && b.video_profiles.is_empty());
        assert!(validate_queue_profiles(&request, &videos, false).is_err());
        let mut missing = input.clone();
        missing["video_profiles"]
            .as_object_mut()
            .unwrap()
            .remove("b.mp4");
        let missing: StartRequest = serde_json::from_value(missing).unwrap();
        assert!(validate_queue_profiles(&missing, &videos, true).is_err());
        let mut invalid = input.clone();
        invalid["video_profiles"]["b.mp4"]["gpu"]["pose_stride"] = json!(0);
        let invalid: StartRequest = serde_json::from_value(invalid).unwrap();
        assert!(validate_queue_profiles(&invalid, &videos, true).is_err());
        let mut wrong_primary = input;
        wrong_primary["video_profiles"]["a.mp4"]["gpu"]["pose_stride"] = json!(9);
        let wrong_primary: StartRequest = serde_json::from_value(wrong_primary).unwrap();
        assert!(validate_queue_profiles(&wrong_primary, &videos, true).is_err());
    }

    #[test]
    fn project_launch_catalog_keeps_hundred_video_members_under_one_id() {
        let project = "abc123def456";
        let directory = std::env::temp_dir().join(format!("volley-launch-{}", Uuid::new_v4()));
        let root = directory.join("native_studio_pipelines");
        fs::create_dir_all(&root).unwrap();
        let ids: Vec<_> = (1..=100).map(|id| format!("{id:032x}")).collect();
        let queue = &ids[0];
        for (index, id) in ids.iter().enumerate() {
            let path = root.join(id);
            fs::create_dir(&path).unwrap();
            save(&path.join("status.json"), &json!({
                "schema":SCHEMA,"project_id":project,"id":id,"queue_id":queue,"queue_runs":ids,
                "mode":"full","video":format!("video-{index}.mp4"),"started_ms":1700000000000_u64,
                "analysis_id":format!("{:032x}",1000+index),
                "postprocessing_id":format!("{:032x}",2000+index)
            })).unwrap();
        }
        let launches = project_launches(&directory, project).unwrap();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0]["id"], queue.as_str());
        assert_eq!(launches[0]["members"].as_array().unwrap().len(), 100);
        assert_eq!(launches[0]["members"][99]["video"], "video-99.mp4");
        storage::write_json_atomic(
            &directory.join("native_analysis_names.json"),
            &json!({format!("launch:{queue}"):"Saved name",format!("hidden_launch:{queue}"):true}),
        )
        .unwrap();
        let launches = project_launches(&directory, project).unwrap();
        assert_eq!(launches[0]["hidden"], true);
        assert_eq!(launches[0]["display_name"], "Saved name");
        let bad_path = root.join(&ids[99]).join("status.json");
        let mut bad = read(&bad_path).unwrap();
        bad["queue_id"] = json!("f".repeat(32));
        save(&bad_path, &bad).unwrap();
        assert!(project_launches(&directory, project).unwrap().is_empty());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn project_queue_orders_selected_video_first_without_duplicates() {
        let project = json!({"videos":["second.mp4","first.mp4","second.mp4"]});
        assert_eq!(
            queue_video_order(&project, "first.mp4").unwrap(),
            vec!["first.mp4", "second.mp4"]
        );
        assert!(queue_video_order(&project, "missing.mp4").is_err());
    }

    #[test]
    fn queue_remaining_uses_whole_project_elapsed_and_includes_pending_videos() {
        let rows = [
            json!({"status":"complete","started_ms":1000,"finished_ms":31000,
                "queue_weight_sec":25.0,"progress_percent":100.0}),
            json!({"status":"running","started_ms":31000,
                "queue_weight_sec":75.0,"progress_percent":33.333333333333336}),
            json!({"status":"queued","queue_weight_sec":100.0,"progress_percent":0.0}),
        ];
        let summary = queue_progress(&rows, true, 61000);
        assert_eq!(summary["started_ms"], 1000);
        assert_eq!(summary["elapsed_ms"], 60000);
        assert_eq!(summary["progress_percent"], 25.0);
        assert_eq!(summary["remaining_ms"], 180000);
        assert!(queue_progress(&rows, false, 61000)["remaining_ms"].is_null());
        assert!(queue_progress(&rows, true, 2000)["remaining_ms"].is_null());
        let mut failed = rows.clone();
        failed[0]["status"] = json!("failed");
        assert!(queue_progress(&failed, true, 61000)["remaining_ms"].is_null());
        assert_eq!(progress_remaining(60000, 50.0), Some(60000));
        assert_eq!(progress_remaining(1000, 50.0), None);
        assert_eq!(progress_remaining(60000, 0.0), None);
        assert_eq!(progress_remaining(60000, 100.0), None);
        assert_eq!(progress_remaining(60000, f64::NAN), None);
    }

    #[test]
    fn queue_progress_uses_video_duration_and_never_completes_pending_video() {
        let rows = vec![
            json!({"id":"first","video":"a.mp4","status":"complete","progress_percent":100.0,
                "queue_weight_sec":20.0,"analysis_frames_processed":100,"rallies_published":30,"touch_detector_count":12,"gpu_rally_mask":{"merged":[[0,30]]}}),
            json!({"id":"second","video":"b.mp4","status":"running","progress_percent":25.0,
                "queue_weight_sec":80.0,"stage_id":"players","touch_detector_count":7,"stages":[{"id":"ball","frames_processed":200},{"id":"players","frames_processed":40}],"rally_stage_progress":{"pose":{"rallies_total":20,"state":"running"}}}),
            json!({"id":"third","video":"c.mp4","status":"queued","progress_percent":0.0,
                "queue_weight_sec":100.0}),
        ];
        let summary = queue_progress(&rows, true, 0);
        assert_eq!(summary["progress_percent"], 20.0);
        assert_eq!(summary["completed_videos"], 1);
        assert_eq!(summary["frames_processed"], 300);
        assert_eq!(summary["touch_detector_count"], 19);
        assert_eq!(summary["videos"][0]["touch_detector_count"], 12);
        assert_eq!(summary["videos"][1]["touch_detector_count"], 7);
        assert!(summary["videos"][2]["touch_detector_count"].is_null());
        assert_eq!(summary["rallies_total"], 50);
        assert_eq!(summary["total_videos"], 3);
        assert_eq!(summary["videos"][0]["rallies_total"], 30);
        assert_eq!(summary["videos"][1]["rallies_total"], 20);
        assert!(summary["videos"][1]["ball_frames_total"].is_null());
        assert!(summary["videos"][2]["ball_frames_total"].is_null());
        assert_eq!(
            summary["videos"][1]["rally_stage_progress"]["pose"]["state"],
            "running"
        );
        assert!(summary["videos"][2]["rallies_total"].is_null());
        let interrupted = queue_progress(&rows, false, 0);
        assert_eq!(interrupted["videos"][1]["status"], "interrupted");
        assert!(interrupted["videos"][1]["rally_stage_progress"].is_null());
        assert_eq!(interrupted["videos"][2]["status"], "interrupted");
        assert_eq!(interrupted["progress_percent"], 20.0);
    }
    #[test]
    fn queue_current_ball_input_exposes_its_own_frame_budget() {
        let rows = [
            json!({"status":"running","stage_id":"boxes",
                "stages":[{"id":"ball","frames_processed":100,"frames_total":100}]}),
            json!({"status":"running","stage_id":"ball","stages":[{"id":"ball",
                "frames_processed":45,"frames_total":250}]}),
        ];
        let summary = queue_progress(&rows, true, 0);
        assert!(summary["videos"][0]["ball_frames_total"].is_null());
        assert_eq!(summary["videos"][1]["frames_processed"], 45);
        assert_eq!(summary["videos"][1]["ball_frames_total"], 250);
        assert!(queue_progress(&rows, false, 0)["videos"][1]["ball_frames_total"].is_null());
    }

    #[test]
    fn active_analysis_sources_match_only_the_requested_video_and_artifact() {
        let status = json!({
            "video":"match.mp4",
            "source_analysis_id":"primary",
            "analysis_id":"fusion",
            "reused_from_id":"previous",
            "gpu_jobs":{"ball":"ball-run","players":"person-run"},
            "cpu_runs":{"tracking":"track-run"}
        });
        for artifact in [
            "primary",
            "fusion",
            "previous",
            "ball-run",
            "person-run",
            "track-run",
        ] {
            assert!(status_uses_analysis(&status, "match.mp4", artifact));
            assert!(!status_uses_analysis(&status, "other.mp4", artifact));
        }
        assert!(!status_uses_analysis(&status, "match.mp4", "unrelated"));
    }

    #[tokio::test]
    async fn crashed_analysis_worker_records_failure_and_frees_run_slot() {
        let root = std::env::temp_dir().join(format!("volley-pipeline-panic-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("status.json");
        let run_id = "panic-worker";
        let project_id = "isolated-test";
        save(
            &path,
            &json!({
                "schema": SCHEMA,
                "id": run_id,
                "project_id": project_id,
                "status": "running",
                "stage": "Ball Stabilizer",
                "pipeline_complete": false,
                "stages": [{"started_ms": epoch_ms(), "finished_ms": null}],
                "finished_ms": null,
                "current_gpu_job_id": null,
            }),
        )
        .unwrap();
        let slot = Mutex::new(Some(Running {
            id: run_id.into(),
            project_id: project_id.into(),
            cancel: Arc::new(AtomicBool::new(false)),
            queue_runs: Vec::new(),
        }));
        let worker = tokio::spawn(async { panic!("segment end is observed") });
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            supervise_pipeline_worker(worker, &slot, &path, project_id, run_id, false),
        )
        .await
        .unwrap();
        assert!(slot.lock().await.is_none());
        let result = read(&path).unwrap();
        assert_eq!(result["status"], "failed");
        assert_eq!(result["failed_stage"], "Ball Stabilizer");
        assert!(result["error"]
            .as_str()
            .unwrap()
            .contains("segment end is observed"));
        assert!(result["finished_ms"].as_u64().is_some());
        assert!(result["stages"][0]["finished_ms"].as_u64().is_some());
        let events = fs::read_to_string(path.with_file_name("events.jsonl")).unwrap();
        assert!(events.contains("segment end is observed"));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn crashed_analysis_worker_releases_slot_even_if_status_is_unreadable() {
        let slot = Mutex::new(Some(Running {
            id: "panic-worker".into(),
            project_id: "isolated-test".into(),
            cancel: Arc::new(AtomicBool::new(false)),
            queue_runs: Vec::new(),
        }));
        let missing = std::env::temp_dir().join(format!("missing-pipeline-{}", Uuid::new_v4()));
        let worker = tokio::spawn(async { panic!("unrecoverable worker") });
        supervise_pipeline_worker(
            worker,
            &slot,
            &missing,
            "isolated-test",
            "panic-worker",
            false,
        )
        .await;
        assert!(slot.lock().await.is_none());
    }
    #[test]
    fn clear_history_excludes_active_run_after_background_scan() {
        let root =
            std::env::temp_dir().join(format!("volley-history-lock-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let mut ids = BTreeSet::new();
        ids.insert("a".repeat(32));
        ids.insert("b".repeat(32));
        let active = "b".repeat(32);
        assert_eq!(
            persist_hidden_history(&root, ids, Some(&active)).unwrap(),
            1
        );
        assert_eq!(
            hidden_history(&root).unwrap(),
            BTreeSet::from(["a".repeat(32)])
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn only_shared_rally_scoped_pose_sources_can_be_reused() {
        for decoder in [
            "gpu_resident_nvdec_primary_v1",
            "shared_primary_native_rgb_v2",
        ] {
            assert!(shared_pose_source(&json!({"decode_pipeline":decoder})));
        }
        for decoder in ["standalone_original_video", "", "original_rgb24_full_video"] {
            assert!(!shared_pose_source(&json!({"decode_pipeline":decoder})));
        }
        assert!(!shared_pose_source(&Value::Null));
    }
    #[test]
    fn shared_pose_reports_published_counts_and_shared_gpu_stage_duration() {
        let pipeline = json!({"stages":[
            {"id":"players","started_ms":1000,"finished_ms":37000,"duration_ms":36000},
            {"id":"fusion","started_ms":37000,"finished_ms":38000,"duration_ms":1000}
        ]});
        let pose = json!({"sampled_frames":175,"eligible_pose_frames":50,"inferred_pose_frames":42,"player_poses":612});
        let metrics = shared_pose_metrics(&pipeline, &pose);
        assert_eq!(metrics["source_stage_id"], "players");
        assert_eq!(metrics["sampled_frames"], 175);
        assert_eq!(metrics["eligible_pose_frames"], 50);
        assert_eq!(metrics["inferred_pose_frames"], 42);
        assert_eq!(metrics["player_poses"], 612);
        assert_eq!(metrics["duration_ms"], 36000);
        assert_eq!(metrics["started_ms"], 1000);
        assert_eq!(metrics["finished_ms"], 37000);
    }

    #[test]
    fn pipeline_stage_ids_are_stable() {
        assert_eq!(pipeline_stage_id("Проверка исходного видео"), "video");
        assert_eq!(pipeline_stage_id("Ожидание очереди RTMPose"), "other");
        assert_eq!(pipeline_stage_id("Ожидание очереди ReID"), "other");
        assert_eq!(
            pipeline_stage_id("GPU · VBallNet: траектория мяча и розыгрыши"),
            "ball"
        );
        assert_eq!(
            pipeline_stage_id("Интерполяция траектории мяча · Ball Stabilizer"),
            "ball_stabilizer"
        );
        assert_eq!(pipeline_stage_id("Ground Detector"), "rallies");
        assert_eq!(pipeline_stage_id("Touch Detector"), "touch_detector");
        assert_eq!(pipeline_stage_id("Touch Attribution"), "touches");
        assert_eq!(pipeline_stage_id("Pose Geometry"), "touches");
        assert_eq!(pipeline_stage_id("Rally Detector"), "rallies");
        assert_eq!(pipeline_stage_id("Поиск розыгрышей"), "rallies");
        assert_eq!(
            pipeline_stage_id("GPU · RF-DETR + RTMPose: игровые интервалы"),
            "players"
        );
        assert_eq!(pipeline_stage_id("GPU · RTMPose-M: позы игроков"), "pose");
        assert_eq!(
            pipeline_stage_id("Объединение · VBallNet и рамки игроков"),
            "fusion"
        );
        assert_eq!(pipeline_stage_id("CPU · Связывание рамок игроков"), "boxes");
        assert_eq!(pipeline_stage_id("Постобработка · Track ID"), "tracking");
        assert_eq!(
            pipeline_stage_id("Постобработка · касания в первичных розыгрышах"),
            "touches"
        );
        assert_eq!(pipeline_stage_id("Публикация розыгрышей"), "publish");
        assert_eq!(
            pipeline_stage_id("Постобработка · автоматическая привязка игроков"),
            "identity"
        );
        assert_eq!(
            pipeline_stage_id("Монтажная временная разметка"),
            "material"
        );
    }

    #[test]
    fn completion_quality_distinguishes_identity_failure_from_pending_review() {
        assert_eq!(json!(CompletionQuality::IdentityFailed), "identity_failed");
        assert_eq!(
            json!(CompletionQuality::IdentityReviewRequired),
            "identity_review_required"
        );
        assert_eq!(
            completion_quality(
                PipelineMode::Full,
                &json!({"automatic_identity_error":"OpenVINO unavailable"})
            ),
            CompletionQuality::IdentityFailed
        );
        assert_eq!(
            completion_quality(
                PipelineMode::Post,
                &json!({"automatic_identity_complete":false})
            ),
            CompletionQuality::IdentityReviewRequired
        );
        assert_eq!(
            completion_quality(
                PipelineMode::Post,
                &json!({"automatic_identity_complete":true})
            ),
            CompletionQuality::Complete
        );
        assert_eq!(
            completion_quality(
                PipelineMode::Gpu,
                &json!({"automatic_identity_error":"OpenVINO unavailable"})
            ),
            CompletionQuality::Complete
        );
    }

    #[tokio::test]
    async fn admission_serializes_incompatible_run_starts() {
        let first = admission_guard().await;
        let next = tokio::spawn(async move {
            let _second = admission_guard().await;
        });
        tokio::task::yield_now().await;
        assert!(!next.is_finished());
        drop(first);
        tokio::time::timeout(std::time::Duration::from_secs(5), next)
            .await
            .expect("second admission must proceed after first releases")
            .expect("waiting admission must finish");
    }

    #[test]
    fn accepted_late_cancel_overrides_success_and_failure() {
        assert_eq!(cancelled_outcome(Ok(()), true), Err("cancelled".into()));
        assert_eq!(
            cancelled_outcome(Err("cpu error".into()), true),
            Err("cancelled".into())
        );
        assert_eq!(cancelled_outcome(Ok(()), false), Ok(()));
        assert_eq!(
            cancelled_outcome(Err("cpu error".into()), false),
            Err("cpu error".into())
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "requires an isolated video, model root and local CUDA"]
    async fn real_touch_guided_pose_full_analysis() {
        let base = PathBuf::from(std::env::var("VOLLEY_AB_DATA_ROOT").unwrap());
        let source = PathBuf::from(std::env::var("VOLLEY_AB_SOURCE").unwrap());
        assert!(!base.exists(), "A/B requires a fresh data root");
        let project_id = "abc123def456";
        let video = "experiment.mp4";
        let uploads = base.join("uploads/project_sources");
        fs::create_dir_all(&uploads).unwrap();
        fs::copy(source, uploads.join(video)).unwrap();
        fs::create_dir_all(base.join("projects").join(project_id)).unwrap();
        storage::write_json_atomic(
            &base.join("projects/projects.json"),
            &json!({"schema":1,"projects":[{"id":project_id,"videos":[video]}]}),
        )
        .unwrap();
        let state = Arc::new(AppState {
            repository_root: PathBuf::from(std::env::var("VOLLEY_AB_MODEL_ROOT").unwrap()),
            data_root: base.clone(),
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
        });
        let request = serde_json::from_value::<StartRequest>(json!({
            "project_id":project_id,"video":video,"mode":"full",
            "gpu":{"pose_stride":1,"pose_mode":"photo","person_detector":"rfdetr-medium"}
        }))
        .unwrap();
        let started = std::time::Instant::now();
        let Json(queued) = start(State(state), HeaderMap::new(), Json(request))
            .await
            .unwrap();
        let path = base
            .join("projects")
            .join(project_id)
            .join("native_studio_pipelines")
            .join(queued["id"].as_str().unwrap())
            .join("status.json");
        tokio::time::timeout(std::time::Duration::from_secs(900), async {
            while has_active().await {
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            }
        })
        .await
        .unwrap();
        let result = read(&path).unwrap();
        println!(
            "POSE_AB full_ms={} status_path={}",
            started.elapsed().as_millis(),
            path.display()
        );
        assert_eq!(result["status"], "complete", "{result}");
        let stages = result["stages"].as_array().unwrap();
        let position = |id| stages.iter().position(|stage| stage["id"] == id).unwrap();
        assert!(position("rallies") < position("touch_detector"));
        assert!(position("touch_detector") < position("players"));
        assert!(position("players") < position("touches"));
        let progress = &result["rally_stage_progress"];
        assert_eq!(progress["boxes"]["state"], "complete");
        assert_eq!(progress["pose"]["state"], "complete");
        assert!(
            progress["boxes"]["started_ms"].as_u64().unwrap()
                <= progress["pose"]["started_ms"].as_u64().unwrap()
        );
        assert_eq!(
            result["primary_tracking"]["pose_sampling"],
            cpu_retrack::pose_sampling_contract(Default::default(), Default::default()),
        );
    }

    #[test]
    fn independent_queue_defaults_on_for_medium_full_analysis_with_auto_reid() {
        let mut req: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"first.mp4","mode":"full"
        }))
        .unwrap();
        assert!(independent_profile(&req));
        assert!(independent_queues_requested(None));
        assert!(independent_queues_requested(Some("1")));
        assert!(!independent_queues_requested(Some("0")));
        req.gpu.gpu_memory_limit_mb = 4096;
        assert!(!independent_profile(&req));
        req.gpu.gpu_memory_limit_mb = 0;
        req.gpu.person_detector = DetectorVariant::RfDetrLarge;
        assert!(!independent_profile(&req));
        req.gpu.person_detector = DetectorVariant::RfDetrMedium;
        req.gpu.pose_enabled = false;
        assert!(!independent_profile(&req));
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "requires isolated example and native CUDA/ORT"]
    async fn project_queue_two_videos_cuda_stage_handoff() {
        let base = PathBuf::from(std::env::var("VOLLEY_AB_DATA_ROOT").unwrap());
        let source = PathBuf::from(std::env::var("VOLLEY_AB_SOURCE").unwrap());
        assert!(!base.exists(), "A/B requires a fresh data root");
        let project_id = "abc123def456";
        let videos = ["first.mp4", "second.mp4"];
        let uploads = base.join("uploads/project_sources");
        fs::create_dir_all(&uploads).unwrap();
        for video in videos {
            fs::copy(&source, uploads.join(video)).unwrap();
            write_test_court(
                &base,
                project_id,
                video,
                &json!({"court":[[0.08,0.2],[0.92,0.2],[0.97,0.9],[0.03,0.9]],
                    "net":[[0.08,0.48],[0.92,0.48]]}),
            );
        }
        fs::create_dir_all(base.join("projects").join(project_id)).unwrap();
        storage::write_json_atomic(
            &base.join("projects/projects.json"),
            &json!({"schema":1,"projects":[{"id":project_id,"videos":videos}]}),
        )
        .unwrap();
        let state = Arc::new(AppState {
            repository_root: PathBuf::from(std::env::var("VOLLEY_AB_MODEL_ROOT").unwrap()),
            data_root: base.clone(),
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
        });
        let reid = std::env::var("VOLLEY_AB_REID_DEVICE").unwrap_or_else(|_| "cpu".into());
        let request = serde_json::from_value::<StartRequest>(json!({
            "project_id":project_id,"video":videos[0],"mode":"full",
            "gpu":{"pose_stride":1,"parallel_rally_poses":false,"primary_rallies":{"min_trajectory_seconds":1.0}},
            "post":{"reid_device":reid,"reid_buffer_mib":0},
            "video_profiles":{
                "first.mp4":{"gpu":{"pose_stride":1,"parallel_rally_poses":false,"primary_rallies":{"min_trajectory_seconds":1.0}},
                    "post":{"reid_device":reid,"reid_buffer_mib":0}},
                "second.mp4":{"gpu":{"pose_stride":3,"primary_rallies":{"min_trajectory_seconds":1.0}},
                    "post":{"reid_device":reid,"reid_buffer_mib":0}}
            }
        })).unwrap();
        let Json(queued) = start_project(State(state.clone()), HeaderMap::new(), Json(request))
            .await
            .unwrap();
        let queue_id = queued["queue_id"].as_str().unwrap();
        let root = folder(&state, project_id, false).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(180), async {
            while has_active().await {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        })
        .await
        .unwrap();
        let rows = read_queue_rows(&root, project_id, queue_id).unwrap();
        assert_eq!(rows.len(), 2);
        for row in &rows {
            println!(
                "VIDEO {} state={} error={} stages={}",
                row["video"], row["status"], row["error"], row["stages"]
            );
            assert_eq!(row["status"], "complete", "{}", row["error"]);
            assert!(row["phase1_checkpoint_key"].as_str().is_some());
        }
        let stage_start = |row: &Value, id: &str| {
            row["stages"]
                .as_array()
                .unwrap()
                .iter()
                .find(|s| s["id"] == id)
                .unwrap()["started_ms"]
                .as_u64()
                .unwrap()
        };
        for row in &rows {
            let first_phase_stages = row["stages"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|stage| stage["id"].as_str())
                .filter(|id| matches!(*id, "rallies" | "touch_detector" | "players"))
                .collect::<Vec<_>>();
            assert_eq!(first_phase_stages, ["rallies", "touch_detector", "players"]);
        }
        assert!(stage_start(&rows[0], "ball") <= stage_start(&rows[1], "ball"));
        let detector_finished = rows[0]["stages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|stage| stage["id"] == "players")
            .unwrap()["finished_ms"]
            .as_u64()
            .unwrap();
        let independent = independent_queues_requested(
            std::env::var("VOLLEY_INDEPENDENT_GPU_QUEUES")
                .ok()
                .as_deref(),
        );
        if independent {
            assert!(
                stage_start(&rows[1], "ball") < detector_finished,
                "Independent ball queue did not overlap RF-DETR"
            );
        } else {
            assert!(stage_start(&rows[1], "ball") >= detector_finished);
        }
        let elapsed =
            rows[1]["finished_ms"].as_u64().unwrap() - rows[0]["started_ms"].as_u64().unwrap();
        assert!(rows
            .iter()
            .all(|row| row["gpu_reused"] == false && row["result_reused"] == false));
        println!("BENCH independent={independent} elapsed_ms={elapsed} first_players_finished={detector_finished} second_ball_started={}", stage_start(&rows[1], "ball"));
        assert!(stage_start(&rows[1], "ball") < rows[0]["finished_ms"].as_u64().unwrap());
        for row in &rows {
            let progress = &row["rally_stage_progress"];
            assert!(progress["boxes"]["rallies_total"].as_u64().unwrap() > 0);
            assert!(
                progress["boxes"]["started_ms"].as_u64().unwrap()
                    < progress["detector_finished_ms"].as_u64().unwrap()
            );
        }
        assert_eq!(rows[0]["profile"]["gpu"]["parallel_rally_poses"], false);
        assert_eq!(rows[1]["profile"]["gpu"]["parallel_rally_poses"], true);
        assert_eq!(rows[0]["profile"]["gpu"]["pose_stride"], 1);
        assert_eq!(rows[1]["profile"]["gpu"]["pose_stride"], 3);
        assert_ne!(
            rows[0]["phase1_checkpoint_key"],
            rows[1]["phase1_checkpoint_key"]
        );
        assert_ne!(rows[0]["analysis_id"], rows[1]["analysis_id"]);
        assert_ne!(rows[0]["postprocessing_id"], rows[1]["postprocessing_id"]);
        for (row, video) in rows.iter().zip(videos) {
            for model in ["ball", "players"] {
                let job = row["gpu_jobs"][model].as_str().unwrap();
                let file = base
                    .join("projects")
                    .join(project_id)
                    .join("native_gpu_runs")
                    .join(job)
                    .join("manifest.json");
                let manifest: Value = serde_json::from_slice(&fs::read(file).unwrap()).unwrap();
                assert_eq!(manifest["video"], video);
            }
        }
        println!("PROJECT STAGE HANDOFF: independent profiles, checkpoints, artifacts and overlapping video execution with an ordered GPU handoff");
    }

    fn write_test_court(base: &Path, project_id: &str, video: &str, geometry: &Value) {
        let dir = base.join("projects").join(project_id).join("video_edits");
        fs::create_dir_all(&dir).unwrap();
        let name = format!("{:x}.json", Sha256::digest(video.as_bytes()));
        storage::write_json_atomic(
            &dir.join(name),
            &json!({"schema":"volley-video-edit/1","project_id":project_id,"video":video,
                "trim_start_sec":0.0,"trim_end_sec":null,"court":geometry["court"],
                "net":geometry["net"],"geometry_source":"manual"}),
        )
        .unwrap();
    }

    #[tokio::test]
    async fn start_queues_before_source_validation_and_records_validation_failure() {
        let base = std::env::temp_dir().join(format!("studio-queue-{}", Uuid::new_v4()));
        let project_id = "abc123def456";
        fs::create_dir_all(base.join("projects").join(project_id)).unwrap();
        storage::write_json_atomic(
            &base.join("projects/projects.json"),
            &json!({
                "projects":[{"id":project_id,"videos":["missing.mp4","unmarked.mp4"]}]
            }),
        )
        .unwrap();
        let state = Arc::new(AppState {
            repository_root: base.clone(),
            data_root: base.clone(),
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
        });
        let uploads = base.join("uploads/project_sources");
        fs::create_dir_all(&uploads).unwrap();
        let source = uploads.join("missing.mp4");
        assert!(std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=size=128x128:rate=30",
                "-t",
                "0.5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p"
            ])
            .arg(&source)
            .status()
            .unwrap()
            .success());
        let source_bytes = fs::read(&source).unwrap();
        fs::copy(&source, uploads.join("unmarked.mp4")).unwrap();
        let request = || {
            serde_json::from_value::<StartRequest>(json!({
                "project_id":project_id,"video":"missing.mp4","mode":"gpu",
                "client_request_id":"12345678901234567890123456789012"
            }))
            .unwrap()
        };
        for mode in ["gpu", "full", "post"] {
            let unmarked: StartRequest = serde_json::from_value(json!({
                "project_id":project_id,"video":"missing.mp4","mode":mode,
                "source_analysis_id":if mode == "post" {Some("a".repeat(32))} else {None}
            }))
            .unwrap();
            let error = start(State(state.clone()), HeaderMap::new(), Json(unmarked))
                .await
                .unwrap_err();
            assert!(
                reason(error).contains("размет"),
                "missing court must be rejected"
            );
        }
        write_test_court(
            &base,
            project_id,
            "missing.mp4",
            &json!({"court":[],"net":[[0.1,0.5],[0.9,0.5]]}),
        );
        let error = start(State(state.clone()), HeaderMap::new(), Json(request()))
            .await
            .unwrap_err();
        assert!(reason(error).contains("размет"));
        write_test_court(
            &base,
            project_id,
            "missing.mp4",
            &json!({"court":[[0.1,0.2],[0.9,0.2],[0.9,0.8],[0.1,0.8]],"net":null}),
        );
        let queue: StartRequest = serde_json::from_value(json!({
            "project_id":project_id,"video":"missing.mp4","mode":"full",
            "video_profiles":{"missing.mp4":{"gpu":{},"post":{}},"unmarked.mp4":{"gpu":{},"post":{}}}
        })).unwrap();
        let error = start_project(State(state.clone()), HeaderMap::new(), Json(queue))
            .await
            .unwrap_err();
        assert!(reason(error).contains("unmarked.mp4"));
        assert!(!base
            .join("projects")
            .join(project_id)
            .join("native_studio_pipelines")
            .exists());
        let Json(queued) = start(State(state.clone()), HeaderMap::new(), Json(request()))
            .await
            .unwrap();
        assert_eq!(queued["status"], "queued");
        fs::remove_file(&source).unwrap();
        let Json(recovered) = start(State(state.clone()), HeaderMap::new(), Json(request()))
            .await
            .unwrap();
        assert_eq!(recovered["id"], queued["id"]);
        assert_eq!(recovered["recovered"], true);
        let path = base
            .join("projects")
            .join(project_id)
            .join("native_studio_pipelines")
            .join(queued["id"].as_str().unwrap())
            .join("status.json");
        assert_eq!(read(&path).unwrap()["status"], "queued");
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while has_active().await {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let failed = read(&path).unwrap();
        assert_eq!(failed["status"], "failed");
        assert!(failed["error"].as_str().is_some_and(|s| !s.is_empty()));
        let events = fs::read_to_string(path.with_file_name("events.jsonl")).unwrap();
        assert!(events
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .any(|event| event["kind"] == "stage_start"
                && event["stage"] == "Проверка исходного видео"));
        let mut legacy = failed.clone();
        for key in ["interpolate_players", "interpolate_pose"] {
            legacy["profile"]["post"]
                .as_object_mut()
                .unwrap()
                .remove(key);
            legacy["profile"]["gpu"][key] = json!(true);
        }
        storage::write_json_atomic(&path, &legacy).unwrap();
        let Json(recovered) = start(State(state.clone()), HeaderMap::new(), Json(request()))
            .await
            .unwrap();
        assert_eq!(recovered["id"], queued["id"]);
        assert_eq!(recovered["recovered"], true);
        let mut different = request();
        different.post.interpolate_pose = false;
        assert!(
            start(State(state.clone()), HeaderMap::new(), Json(different))
                .await
                .is_err()
        );
        let mut unowned = request();
        unowned.video = "unowned.mp4".into();
        unowned.client_request_id = None;
        assert!(start(State(state.clone()), HeaderMap::new(), Json(unowned))
            .await
            .is_err());
        fs::write(&source, &source_bytes).unwrap();
        let mut cancellation_request = request();
        cancellation_request.client_request_id = None;
        let Json(new_job) = start(
            State(state.clone()),
            HeaderMap::new(),
            Json(cancellation_request),
        )
        .await
        .unwrap();
        fs::remove_file(&source).unwrap();
        let cancelled_id = new_job["id"].as_str().unwrap().to_owned();
        let cancelled_path = base
            .join("projects")
            .join(project_id)
            .join("native_studio_pipelines")
            .join(&cancelled_id)
            .join("status.json");
        fs::write(&cancelled_path, b"{").unwrap();
        let Json(ack) = cancel(
            State(state.clone()),
            HeaderMap::new(),
            Json(CancelRequest {
                project_id: project_id.into(),
                id: cancelled_id,
            }),
        )
        .await
        .unwrap();
        assert_eq!(ack["cancel_requested"], true);
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while has_active().await {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(read(&cancelled_path).unwrap()["status"], "cancelled");
        fs::remove_dir_all(base).unwrap();
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn shared_gpu_is_primary_and_retains_explicit_rollback() {
        use crate::pose_inference::PoseInputMode::{Photo, Video};
        assert!(shared_primary_requested(None, 9, true, Photo));
        assert!(shared_primary_requested(None, 9, false, Video));
        assert!(shared_primary_requested(Some("1"), 9, true, Photo));
        assert!(shared_primary_requested(Some("true"), 9, true, Photo));
        for override_value in ["0", "false", "legacy", "invalid"] {
            assert!(!shared_primary_requested(
                Some(override_value),
                9,
                true,
                Photo
            ));
        }
        assert!(!shared_primary_requested(None, 1, true, Photo));
        assert!(!shared_primary_requested(None, 3, true, Photo));
        assert!(!shared_primary_requested(None, 9, true, Video));
        assert!(!shared_primary_requested(Some("1"), 9, true, Video));
    }
    #[cfg(any(unix, windows))]
    #[test]
    fn original_resolution_pose_is_default_on_supported_primary_frames() {
        assert!(shared_native_pose_supported(true, 1920, 1080));
        assert!(shared_native_pose_supported(true, 2560, 1440));
        assert!(shared_native_pose_supported(true, 3840, 2160));
        assert!(!shared_native_pose_supported(false, 1920, 1080));
        assert!(!shared_native_pose_supported(true, 0, 1080));
        assert!(!shared_native_pose_supported(true, 7680, 4320));
    }

    #[test]
    fn completed_pipeline_status_over_32_kib_is_readable() {
        let dir = std::env::temp_dir().join(format!("studio-status-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("status.json");
        let detected = vec![json!({"start": 1.1, "end": 2.2, "confidence": 0.8}); 1_000];
        let expected = json!({"schema": SCHEMA, "status": "complete",
            "gpu_rally_mask": {"detected": detected}});
        save(&path, &expected).unwrap();
        assert!(fs::metadata(&path).unwrap().len() > 32 * 1024);
        assert_eq!(read(&path).unwrap(), expected);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn progress_records_stage_transitions_for_history() {
        let dir = std::env::temp_dir().join(format!("studio-history-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("status.json");
        let mut status = json!({"stage":"В очереди"});
        progress(&path, &mut status, "GPU", 1.0).unwrap();
        let log = fs::read_to_string(dir.join("events.jsonl")).unwrap_or_default();
        let records: Vec<Value> = log
            .lines()
            .map(|row| serde_json::from_str(row).unwrap())
            .collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["kind"], "stage_start");
        assert_eq!(records[0]["stage"], "GPU");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn consecutive_gpu_progress_updates_keep_one_stage_per_real_pass() {
        let dir = std::env::temp_dir().join(format!("studio-gpu-stages-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("status.json");
        let mut status = json!({"stage":"В очереди"});
        progress(&path, &mut status, "Проверка исходного видео", 0.0).unwrap();
        progress(&path, &mut status, "Проверка исходного видео", 8.0).unwrap();
        progress(
            &path,
            &mut status,
            "GPU · VBallNet: траектория мяча и розыгрыши",
            8.0,
        )
        .unwrap();
        progress(
            &path,
            &mut status,
            "GPU · VBallNet: траектория мяча и розыгрыши",
            36.8,
        )
        .unwrap();
        progress(
            &path,
            &mut status,
            "GPU · RF-DETR + RTMPose: игровые интервалы",
            36.9,
        )
        .unwrap();
        progress(
            &path,
            &mut status,
            "GPU · RF-DETR + RTMPose: игровые интервалы",
            80.0,
        )
        .unwrap();
        let stages = status["stages"].as_array().unwrap();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0]["name"], "Проверка исходного видео");
        assert_eq!(stages[1]["name"], "GPU · VBallNet");
        assert_eq!(stages[2]["name"], "GPU · RF-DETR + RTMPose");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn saved_primary_rallies_keep_gpu_boundaries_and_reject_other_sources() {
        let status = json!({"original_source":{"sha256":"original"},
            "analysis_range":null});
        let evidence = json!({
            "schema":"volley-primary-rally-mask-evidence/1",
            "ball_job_id":"ball-job-123",
            "original_source":{"sha256":"original"},
            "analysis_range":null,
            "selected_frames":400,
            "mask":{
                "schema":"volley-primary-rally-mask/1",
                "source_frame_count":600,
                "merged":[[12,161],[320,469]],
                "selected_frames":400,
                "movement_filter_version":"stationary_ball_min_rally_5s_v3",
                "min_rally_seconds":5.0,
                "fallback_reason":null
            }
        });
        let bytes = serde_json::to_vec(&evidence).unwrap();
        let (scope, mask) = parse_primary_rally_scope(&bytes, "ball-job-123", &status).unwrap();
        assert_eq!(scope.intervals, vec![(12, 161), (320, 469)]);
        assert_eq!(scope.source_frames, 600);
        assert_eq!(
            cpu_touches::align_primary_intervals(&[(12, 33), (90, 99)], 100, 99).unwrap(),
            [(12, 33), (90, 98)]
        );
        assert!(cpu_touches::align_primary_intervals(&[(12, 33)], 120, 99).is_err());
        assert_eq!(scope.ball_job_id, "ball-job-123");
        assert_eq!(scope.sha256, format!("{:x}", Sha256::digest(&bytes)));
        assert_eq!(mask["merged"], json!([[12, 161], [320, 469]]));
        let mut padded = evidence.clone();
        padded["mask"]["rally_intervals"] = json!([[0, 176], [305, 484]]);
        let (padded_scope, _) = parse_primary_rally_scope(
            &serde_json::to_vec(&padded).unwrap(),
            "ball-job-123",
            &status,
        )
        .unwrap();
        assert_eq!(padded_scope.intervals, vec![(0, 176), (305, 484)]);
        assert!(parse_primary_rally_scope(&bytes, "other-job", &status).is_err());
        let mut altered = evidence.clone();
        altered["mask"]["merged"] = json!([[320, 469], [12, 161]]);
        assert!(parse_primary_rally_scope(
            &serde_json::to_vec(&altered).unwrap(),
            "ball-job-123",
            &status
        )
        .is_err());
        let mut short = evidence.clone();
        short["mask"]["merged"] = json!([[12, 160], [320, 469]]);
        assert!(parse_primary_rally_scope(
            &serde_json::to_vec(&short).unwrap(),
            "ball-job-123",
            &status
        )
        .is_err());
        let mut old = evidence.clone();
        old["mask"]["movement_filter_version"] = json!("stationary_ball_v1");
        assert!(parse_primary_rally_scope(
            &serde_json::to_vec(&old).unwrap(),
            "ball-job-123",
            &status
        )
        .is_err());
        let mut foreign = status.clone();
        foreign["original_source"] = json!({"sha256":"different"});
        assert!(parse_primary_rally_scope(&bytes, "ball-job-123", &foreign).is_err());
    }

    #[test]
    fn primary_rally_mask_is_logged_before_person_inference() {
        let dir = std::env::temp_dir().join(format!("studio-rally-log-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("status.json");
        let mut status = json!({"stage":"В очереди"});
        progress_with_frames(
            &path,
            &mut status,
            "GPU · VBallNet: траектория мяча и розыгрыши",
            36.8,
            Some((900, 900)),
        )
        .unwrap();
        let mask = json!({
            "detected":[[50, 100], [500, 550]],
            "merged":[[50, 100], [500, 550]],
            "selected":[[20, 130], [470, 580]],
            "selected_frames":222,
            "fallback_reason":null
        });
        publish_primary_rally_mask(&path, &mut status, &mask, 900).unwrap();
        assert_eq!(read(&path).unwrap()["gpu_rally_mask"], mask);
        assert_eq!(status["stage"], "Поиск розыгрышей");
        assert_eq!(status["stages"][1]["frames_processed"], 900);
        assert_eq!(status["stages"][1]["frames_total"], 900);
        let log_before_players = fs::read_to_string(dir.join("events.jsonl")).unwrap();
        assert!(log_before_players.contains("выбранных кадров=222/900"));
        assert!(log_before_players.contains("доля для RF-DETR=24.7%"));
        assert!(!log_before_players.contains("GPU · RF-DETR"));
        progress_with_frames(
            &path,
            &mut status,
            "GPU · RF-DETR + RTMPose: игровые интервалы",
            36.8,
            Some((0, 222)),
        )
        .unwrap();
        let entries: Vec<Value> = fs::read_to_string(dir.join("events.jsonl"))
            .unwrap()
            .lines()
            .map(|row| serde_json::from_str(row).unwrap())
            .collect();
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0]["stage"], "GPU · VBallNet");
        assert_eq!(entries[1]["stage"], "Поиск розыгрышей");
        assert_eq!(entries[1]["kind"], "stage_start");
        assert_eq!(entries[2]["stage"], "Поиск розыгрышей");
        assert_eq!(entries[2]["level"], "success");
        assert_eq!(entries[3]["stage"], "GPU · RF-DETR + RTMPose");
        assert!(status["stages"][1]["finished_ms"].is_number());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn primary_rally_mask_reports_zero_selected_frames_on_fallback() {
        let dir = std::env::temp_dir().join(format!("studio-rally-fallback-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("status.json");
        let mut status = json!({"stage":"GPU · VBallNet"});
        let mask = json!({
            "detected":[], "merged":[], "selected_frames":0,
            "fallback_reason":"insufficient_ball_observations"
        });
        publish_primary_rally_mask(&path, &mut status, &mask, 600).unwrap();
        let log = fs::read_to_string(dir.join("events.jsonl")).unwrap();
        assert!(log.contains("fallback=insufficient_ball_observations"));
        assert!(log.contains("доля для RF-DETR=0.0%"));
        assert_eq!(read(&path).unwrap()["gpu_rally_mask"], mask);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn eta_uses_matching_stage_durations_not_global_percent() {
        let now = 200_000;
        let current = json!({"mode":"post","video":"match.mp4","gpu_reused":true,
            "stages":[{"name":"Постобработка · физические касания","started_ms":180_000}],
            "stage_samples":[]});
        let history = json!({"runs":[
            {"status":"complete","mode":"post","video":"match.mp4","gpu_reused":true,
             "stages":[{"name":"Постобработка · Track ID","duration_ms":30000},
                {"name":"Постобработка · физические касания","duration_ms":80000},
                {"name":"Постобработка · автоматическая привязка игроков","duration_ms":10000}]},
            {"status":"complete","mode":"gpu","video":"match.mp4","gpu_reused":false,
             "stages":[{"name":"Постобработка · физические касания","duration_ms":500}]}
        ]});
        assert_eq!(stage_aware_remaining(&current, &history, now), Some(70_000));
        let other = json!({"mode":"post","video":"other.mp4","gpu_reused":true,
            "stages":[{"name":"Постобработка · физические касания","started_ms":180_000}]});
        assert_eq!(stage_aware_remaining(&other, &history, now), None);
    }

    #[test]
    fn interpolation_profile_migrates_legacy_and_prefers_explicit_cpu_values() {
        for (input, players, pose) in [
            (
                json!({"gpu":{"interpolate_players":false,"interpolate_pose":false}}),
                false,
                false,
            ),
            (
                json!({"gpu":{"interpolate_players":false,"interpolate_pose":false},
                    "post":{"interpolate_players":true}}),
                true,
                false,
            ),
            (
                json!({"post":{"interpolate_players":false,"interpolate_pose":true}}),
                false,
                true,
            ),
            (json!({}), true, true),
        ] {
            let mut input = input;
            input["project_id"] = json!("abc123def456");
            input["video"] = json!("match.mp4");
            let request: StartRequest = serde_json::from_value(input).unwrap();
            let output = json!(request);
            assert_eq!(output["post"]["interpolate_players"], players);
            assert_eq!(output["post"]["interpolate_pose"], pose);
            assert!(output["gpu"].get("interpolate_players").is_none());
            assert!(output["gpu"].get("interpolate_pose").is_none());
        }
    }

    #[test]
    fn new_post_requests_have_one_touch_algorithm_without_rewriting_saved_legacy() {
        let base = json!({"project_id":"abc123def456","video":"match.mp4"});
        let fresh: StartRequest = serde_json::from_value(base.clone()).unwrap();
        assert_eq!(fresh.post.touches.algorithm, TouchAlgorithm::BallTrajectory);
        assert!(!fresh.post.touches.require_pose);

        let mut partial = base.clone();
        partial["post"] = json!({"touches":{"threshold":0.62}});
        let active: StartRequest = serde_json::from_value(partial).unwrap();
        assert_eq!(
            active.post.touches.algorithm,
            TouchAlgorithm::BallTrajectory
        );
        assert_eq!(active.post.touches.threshold, 0.62);
        assert!(!active.post.touches.require_pose);

        for algorithm in ["legacy", "pose_motion", "trajectory_tcn"] {
            let request: StartRequest = serde_json::from_value(json!({
                "project_id":"abc123def456","video":"match.mp4",
                "post":{"touches":{"algorithm":algorithm,"require_pose":true}}
            }))
            .unwrap();
            assert_eq!(
                request.post.touches.algorithm,
                TouchAlgorithm::BallTrajectory
            );
            assert!(!request.post.touches.require_pose);
        }
        let (_, archived) =
            decode_profiles(json!({}), json!({"touches":{"algorithm":"legacy"}})).unwrap();
        assert_eq!(archived.touches.algorithm, TouchAlgorithm::Legacy);
    }
    #[test]
    fn interpolation_profiles_reject_invalid_flags_and_unknown_fields() {
        for fields in [
            json!({"gpu":{"interpolate_pose":"false"}}),
            json!({"post":{"interpolate_players":null}}),
            json!({"post":{"interpolate_pose":0}}),
            json!({"gpu":{"unknown":true}}),
            json!({"post":{"unknown":true}}),
            json!({"unknown":true}),
        ] {
            let mut fields = fields;
            fields["project_id"] = json!("abc123def456");
            fields["video"] = json!("match.mp4");
            assert!(serde_json::from_value::<StartRequest>(fields).is_err());
        }
    }

    #[test]
    fn cpu_interpolation_options_default_on_and_remain_independent() {
        let original = PostProfile::default();
        assert!(original.interpolate_players);
        assert!(original.interpolate_pose);
        let changed: PostProfile = serde_json::from_value(json!({
            "interpolate_players":false,"interpolate_pose":true
        }))
        .unwrap();
        assert!(!changed.interpolate_players);
        assert!(changed.interpolate_pose);
        let other: PostProfile = serde_json::from_value(json!({
            "interpolate_players":true,"interpolate_pose":false
        }))
        .unwrap();
        assert!(other.interpolate_players);
        assert!(!other.interpolate_pose);
    }

    #[test]
    fn large_detector_profile_is_distinct_from_medium() {
        let input = json!({"project_id":"project","video":"video.mp4"});
        let medium: StartRequest = serde_json::from_value(input.clone()).unwrap();
        let mut large_input = input;
        large_input["gpu"] = json!({"person_detector":"rfdetr-large"});
        let large: StartRequest = serde_json::from_value(large_input).unwrap();
        let saved = json!({
            "schema":SCHEMA,"project_id":medium.project_id,"video":medium.video,
            "gpu_model_stack":RFDETR_GPU_MODEL_STACK,"profile":{"gpu":medium.gpu}
        });
        assert!(compatible_version(&saved, &medium, &Value::Null));
        let mut previous = saved.clone();
        previous["profile"]["gpu"]["parallel_rally_poses"] = json!(false);
        assert!(compatible_version(&previous, &medium, &Value::Null));
        assert!(!compatible_version(&saved, &large, &Value::Null));
    }

    #[test]
    fn gpu_defaults_match_person_detector_frame_sampling() {
        let profile = GpuProfile::default();
        assert_eq!(
            profile.pose_stride,
            volley_inference::person::PERSON_SAMPLE_PERIOD_FRAMES
        );
        assert_eq!(
            profile.pose_mode,
            crate::pose_inference::PoseInputMode::Photo
        );
    }

    #[test]
    fn primary_rally_gpu_settings_are_validated_versioned_and_defaulted() {
        let identity = json!({"bytes":1234,"mtime_ns":5678});
        let mut request: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4"
        }))
        .unwrap();
        assert_eq!(request.gpu.primary_rallies.min_trajectory_seconds, 1.5);
        assert_eq!(request.gpu.primary_rallies.merge_gap_seconds, 4.0);
        assert_eq!(request.gpu.primary_rallies.stationary_ball_ms, 1000);
        assert_eq!(request.gpu.primary_rallies.padding_before_ms, 500);
        assert_eq!(request.gpu.primary_rallies.padding_after_ms, 500);
        let saved = json!({
            "schema":SCHEMA,
            "gpu_model_stack":requested_gpu_model_stack(request.gpu.person_detector),
            "project_id":request.project_id,
            "video":request.video,
            "profile":{"gpu":request.gpu},
            "source_identity":identity
        });
        assert!(compatible_version(&saved, &request, &identity));

        request.gpu.primary_rallies.min_trajectory_seconds = 2.5;
        assert!(!compatible_version(&saved, &request, &identity));
        request.gpu.primary_rallies.min_trajectory_seconds = 1.5;
        request.gpu.primary_rallies.merge_gap_seconds = 2.0;
        assert!(!compatible_version(&saved, &request, &identity));
        request.gpu.primary_rallies.merge_gap_seconds = 4.0;
        request.gpu.primary_rallies.stationary_ball_ms = 1500;
        assert!(!compatible_version(&saved, &request, &identity));
        request.gpu.primary_rallies.stationary_ball_ms = 1000;
        request.gpu.primary_rallies.padding_before_ms = 750;
        assert!(!compatible_version(&saved, &request, &identity));
        request.gpu.primary_rallies.padding_before_ms = 500;
        request.gpu.primary_rallies.padding_after_ms = 0;
        assert!(!compatible_version(&saved, &request, &identity));
        for invalid in [
            json!({"padding_before_ms":60001}),
            json!({"padding_after_ms":-1}),
            json!({"min_trajectory_seconds":0.0,"merge_gap_seconds":4.0}),
            json!({"min_trajectory_seconds":1.5,"merge_gap_seconds":65.0}),
            json!({"min_trajectory_seconds":"x","merge_gap_seconds":4.0}),
            json!({"min_trajectory_seconds":1.5,"merge_gap_seconds":4.0,"unknown":true}),
        ] {
            assert!(serde_json::from_value::<StartRequest>(json!({
                "project_id":"abc123def456","video":"match.mp4",
                "gpu":{"primary_rallies":invalid}
            }))
            .is_err());
        }
    }

    #[test]
    fn configurable_pose_sampling_changes_gpu_profile_and_evidence_identity() {
        let identity = json!({"bytes":1234,"mtime_ns":5678});
        let mut request: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4"
        }))
        .unwrap();
        let saved = json!({
            "schema":SCHEMA,"gpu_model_stack":requested_gpu_model_stack(request.gpu.person_detector),
            "project_id":request.project_id,"video":request.video,
            "profile":{"gpu":request.gpu},"source_identity":identity
        });
        let default_sampling = request.gpu.pose_sampling();
        assert_eq!(default_sampling.touch_stride, 1);
        assert_eq!(default_sampling.background_stride, 10);
        assert!(compatible_version(&saved, &request, &identity));
        request.gpu.touch_pose_stride = 2;
        assert!(!compatible_version(&saved, &request, &identity));
        let contract =
            cpu_retrack::pose_sampling_contract(Default::default(), request.gpu.pose_sampling());
        assert_eq!(contract["touch_period_frames"], 2);
        request.gpu.touch_pose_stride = 1;
        request.gpu.background_pose_stride = 15;
        assert!(!compatible_version(&saved, &request, &identity));
        let contract =
            cpu_retrack::pose_sampling_contract(Default::default(), request.gpu.pose_sampling());
        assert_eq!(contract["background_period_frames"], 15);
    }

    #[test]
    fn shared_player_pose_stride_changes_gpu_cache_identity() {
        let identity = json!({"bytes":1234,"mtime_ns":5678});
        let mut request: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4"
        }))
        .unwrap();
        let saved = json!({
            "schema":SCHEMA,
            "gpu_model_stack":requested_gpu_model_stack(request.gpu.person_detector),
            "project_id":request.project_id,
            "video":request.video,
            "profile":{"gpu":request.gpu},
            "source_identity":identity
        });
        assert!(compatible_version(&saved, &request, &identity));

        for step in [2, 3, 5, 9, 10, 30] {
            request.gpu.pose_stride = step;
            assert!(!compatible_version(&saved, &request, &identity));
        }
        request.gpu.pose_stride = 1;
        assert!(compatible_version(&saved, &request, &identity));
    }

    #[test]
    fn cpu_player_interpolation_is_not_a_gpu_cache_key() {
        let identity = json!({"bytes":1234,"mtime_ns":5678});
        let mut request: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4"
        }))
        .unwrap();
        let saved = json!({
            "schema":SCHEMA,
            "gpu_model_stack":requested_gpu_model_stack(request.gpu.person_detector),
            "project_id":request.project_id,
            "video":request.video,
            "profile":{"gpu":request.gpu},
            "source_identity":identity
        });
        assert!(compatible_version(&saved, &request, &identity));

        request.post.interpolate_players = false;
        assert!(compatible_version(&saved, &request, &identity));

        request.post.interpolate_pose = false;
        assert!(compatible_version(&saved, &request, &identity));
    }

    #[test]
    fn gpu_pose_mode_distinguishes_video_from_sampled_photos() {
        use crate::pose_inference::PoseInputMode;
        let video: GpuProfile = serde_json::from_value(json!({
            "pose_mode":"video", "pose_stride":30
        }))
        .unwrap();
        assert_eq!(video.pose_mode, PoseInputMode::Video);
        assert_eq!(video.pose_mode.effective_stride(video.pose_stride), 1);
        let photos: GpuProfile = serde_json::from_value(json!({
            "pose_mode":"photo", "pose_stride":10
        }))
        .unwrap();
        assert_eq!(photos.pose_mode, PoseInputMode::Photo);
        assert_eq!(photos.pose_mode.effective_stride(photos.pose_stride), 10);
        assert!(serde_json::from_value::<GpuProfile>(json!({"pose_mode":"unknown"})).is_err());
        assert_eq!(serde_json::to_value(video).unwrap()["pose_mode"], "video");
    }

    #[test]
    fn stage_frame_metrics_preserve_previous_stage_and_cap_counts() {
        let root = std::env::temp_dir().join(format!("volley-stage-frames-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("status.json");
        let mut status = json!({"stage":"В очереди","stages":[]});
        progress_with_frames(&path, &mut status, "GPU · VBallNet", 9.0, Some((25, 100))).unwrap();
        progress_with_frames(&path, &mut status, "GPU · VBallNet", 12.0, Some((140, 100))).unwrap();
        progress_with_frames(&path, &mut status, "GPU · RF-DETR", 50.0, Some((3, 20))).unwrap();
        assert_eq!(status["stages"][0]["frames_processed"], 100);
        assert_eq!(status["stages"][0]["frames_total"], 100);
        assert_eq!(status["stages"][0]["progress_percent"], 100.0);
        assert_eq!(status["stages"][1]["frames_processed"], 3);
        assert_eq!(status["stages"][1]["frames_total"], 20);
        assert_eq!(status["stages"][1]["progress_percent"], 15.0);
        assert_eq!(read(&path).unwrap()["stages"], status["stages"]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tracking_progress_publishes_boxes_and_pose_before_final_track_stage() {
        use cpu_retrack::TrackingProgress;
        let root = std::env::temp_dir().join(format!("volley-track-progress-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("status.json");
        let mut status = json!({"stage":"В очереди","stages":[]});
        progress(&path, &mut status, "CPU · Связывание рамок игроков", 86.0).unwrap();
        tracking_progress(
            &path,
            &mut status,
            PipelineMode::Full,
            TrackingProgress::BoxesReady(12),
        )
        .unwrap();
        tracking_progress(
            &path,
            &mut status,
            PipelineMode::Full,
            TrackingProgress::PoseFrames {
                processed: 0,
                total: 40,
            },
        )
        .unwrap();
        assert_eq!(status["stage_id"], "pose");
        assert_eq!(status["player_tracking_stats"]["total_track_ids"], 12);
        assert!(status["player_tracking_stats"]["track_ids"].is_null());
        assert!(status["stages"][0]["finished_ms"].is_number());
        tracking_progress(
            &path,
            &mut status,
            PipelineMode::Full,
            TrackingProgress::PoseFrames {
                processed: 30,
                total: 40,
            },
        )
        .unwrap();
        assert_eq!(status["stages"][1]["frames_processed"], 30);
        assert_eq!(status["stages"][1]["progress_percent"], 75.0);
        tracking_progress(
            &path,
            &mut status,
            PipelineMode::Full,
            TrackingProgress::PoseFrames {
                processed: 40,
                total: 40,
            },
        )
        .unwrap();
        tracking_progress(
            &path,
            &mut status,
            PipelineMode::Full,
            TrackingProgress::FinalPass,
        )
        .unwrap();
        let saved = read(&path).unwrap();
        assert_eq!(saved["stage_id"], "tracking");
        assert_eq!(saved["stages"][1]["frames_processed"], 40);
        assert_eq!(
            saved["pose_frame_progress"],
            json!({"processed":40,"total":40,"basis":"stable_crop_frames_v1"})
        );
        tracking_progress(
            &path,
            &mut status,
            PipelineMode::Full,
            TrackingProgress::PoseCached(32),
        )
        .unwrap();
        let cached = read(&path).unwrap();
        assert_eq!(cached["stage_id"], "tracking");
        assert_eq!(
            cached["pose_frame_progress"],
            json!({"processed":32,"total":32,"basis":"stable_crop_frames_v1"})
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stage_modes_reject_unknown_values_and_default_to_full() {
        let request: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4"
        }))
        .unwrap();
        assert_eq!(request.mode, PipelineMode::Full);
        assert!(request.mode.allows_gpu_execution());
        for mode in ["gpu", "post"] {
            let request: StartRequest = serde_json::from_value(json!({
                "project_id":"abc123def456", "video":"match.mp4", "mode":mode
            }))
            .unwrap();
            assert_eq!(request.mode.allows_gpu_execution(), mode == "gpu");
            assert_eq!(
                request.mode.label(),
                if mode == "gpu" {
                    "Первичный GPU-анализ"
                } else {
                    "CPU-постобработка"
                }
            );
        }
        assert!(serde_json::from_value::<StartRequest>(json!({
            "project_id":"abc123def456","video":"match.mp4","mode":"unknown"
        }))
        .is_err());
    }

    #[test]
    fn profiles_have_golden_defaults_and_reject_invalid_cpu_thresholds() {
        let profile: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4"
        }))
        .unwrap();
        assert_eq!(profile.gpu.stride, 9);
        assert!(profile.gpu.parallel_rally_poses);
        let sequential: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4",
            "gpu":{"parallel_rally_poses":false}
        }))
        .unwrap();
        assert!(!sequential.gpu.parallel_rally_poses);
        assert_eq!(json!(sequential.gpu)["parallel_rally_poses"], false);
        assert_eq!(profile.gpu.cpu_threads, 0);
        assert_eq!(profile.gpu.gpu_memory_limit_mb, 0);
        let manual: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4",
            "gpu":{"cpu_threads":12,"gpu_memory_limit_mb":4096}
        }))
        .unwrap();
        assert_eq!(manual.gpu.cpu_threads, 12);
        assert_eq!(manual.gpu.gpu_memory_limit_mb, 4096);
        assert_eq!(profile.post.auto_assign_threshold_percent, 65);
        assert_eq!(profile.post.reid_buffer_mib, 512);
        for value in [0, 528, 4096] {
            let custom: StartRequest = serde_json::from_value(json!({
                "project_id":"abc123def456","video":"match.mp4",
                "post":{"reid_buffer_mib":value}
            }))
            .unwrap();
            assert_eq!(custom.post.reid_buffer_mib, value);
        }
        for value in [json!(-1), json!(4097), json!(512.5), json!("512")] {
            assert!(serde_json::from_value::<StartRequest>(json!({
                "project_id":"abc123def456","video":"match.mp4",
                "post":{"reid_buffer_mib":value}
            }))
            .is_err());
        }
        assert_eq!(
            profile.post.reid_device,
            volley_inference::reid_openvino::ReidDevice::Auto
        );
        let manual: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4",
            "post":{"reid_device":"cpu"}
        }))
        .unwrap();
        assert_eq!(
            manual.post.reid_device,
            volley_inference::reid_openvino::ReidDevice::Cpu
        );
        assert!(serde_json::from_value::<StartRequest>(json!({
            "project_id":"abc123def456","video":"match.mp4",
            "post":{"reid_device":"invalid"}
        }))
        .is_err());
        let custom: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4",
            "post":{"auto_assign_threshold_percent":87}
        }))
        .unwrap();
        assert_eq!(custom.post.auto_assign_threshold_percent, 87);
        for threshold in [0, 100] {
            assert!(serde_json::from_value::<StartRequest>(json!({
                "project_id":"abc123def456","video":"match.mp4",
                "post":{"auto_assign_threshold_percent":threshold}
            }))
            .is_err());
        }
        assert_eq!(profile.post.tracking.max_gap_frames, 90);
        assert_eq!(profile.post.touches.threshold, 0.48);
        assert_eq!(
            profile.post.touches.algorithm,
            TouchAlgorithm::BallTrajectory
        );
        assert_eq!(profile.post.rallies.min_rally_frames, 45);
        assert!(profile.post.tracking.validate().is_ok());
        assert!(profile.post.touches.validate().is_ok());
        assert!(profile.post.rallies.validate().is_ok());
        let bad: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4",
            "post":{"tracking":{
                "min_player_score":0.30,"nms_iou":0.88,"max_gap_frames":500,
                "min_assignment_score":0.48,"ambiguity_margin":0.11
            }}
        }))
        .unwrap();
        assert!(bad.post.tracking.validate().is_err());
    }
}

#[cfg(test)]
mod branching_tests {
    use super::*;

    fn emit(project: &Path, directory: &str, id: &str, filename: &str, data: &Value) {
        let folder = project.join(directory).join(id);
        fs::create_dir_all(&folder).unwrap();
        fs::write(folder.join(filename), b"example evidence").unwrap();
        storage::write_json_atomic(&folder.join("manifest.json"), data).unwrap();
    }

    #[tokio::test]
    async fn gpu_settings_and_source_are_branch_keys_while_cpu_settings_are_subbranches() {
        let base = std::env::temp_dir().join(format!("studio-branches-{}", Uuid::new_v4()));
        fs::create_dir_all(&base).unwrap();
        let project_dir = base.join("projects").join("abc123def456");
        fs::create_dir_all(&project_dir).unwrap();
        storage::write_json_atomic(
            &base.join("projects").join("projects.json"),
            &json!({"schema":1,"projects":[{"id":"abc123def456","name":"Test",
                "videos":["match.mp4"]}]}),
        )
        .unwrap();
        let gpu_ball = "a".repeat(32);
        let gpu_players = "b".repeat(32);
        let analysis = "c".repeat(32);
        let replay = "d".repeat(32);
        let touch = "e".repeat(32);
        let rally = "f".repeat(32);
        let trajectory_id = "9".repeat(32);
        let identity = json!({"bytes":2048,"mtime_ns":123456789u64});
        let mut request: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4"
        }))
        .unwrap();
        // The historic fixture contains no RTMPose run, so its GPU profile
        // must explicitly disable the newly introduced pose stage.
        request.gpu.pose_enabled = false;
        request.post.touches.algorithm = TouchAlgorithm::Legacy;
        let sha = "a".repeat(64);
        for (model, id) in [("ball", &gpu_ball), ("players", &gpu_players)] {
            emit(
                &project_dir,
                "native_gpu_runs",
                id,
                "detections.jsonl",
                &json!({
                    "schema":"volley-native-gpu-model-detections/1",
                    "status":"model_raw_detections_complete","job_id":id,
                    "project_id":request.project_id,"video":request.video,
                    "model":model,"stride":9,"source_identity":identity,
                    "person_sampling_period_frames":request.gpu.pose_stride,
                    "raw_candidates_only":true,"studio_gpu_run_complete":false,
                    "file":"detections.jsonl","fps":30,
                    "sha256":sha,
                    "person_sampling_period_frames":request.gpu.pose_stride
                }),
            );
        }
        emit(
            &project_dir,
            "native_gpu_fusions",
            &analysis,
            "frames.jsonl",
            &json!({
                "schema":"volley-native-gpu-fusion/1",
                "state":"raw_candidates_fused","fusion_id":analysis,
                "project_id":request.project_id,"video":request.video,
                "sources":{"vballnet":{"job_id":gpu_ball,"sha256":sha},
                           "person_pose":{"job_id":gpu_players,"sha256":sha}},
                "source_identity":identity,"sha256":sha,"fps":30,
                "player_sampling_period_frames":request.gpu.pose_stride
            }),
        );
        emit(
            &project_dir,
            "native_cpu_replays",
            &replay,
            "tracked_frames.jsonl",
            &json!({
                "schema":"volley-native-cpu-tracking/1","state":"provisional_tracklets_rebuilt","replay_id":replay,
                "project_id":request.project_id,"video":request.video,
                "source_fusion_id":analysis,"ball_selection_policy":"vballnet_only_v1",
                 "interpolated_positions_version":"bounded_same_track_3s_v2",
                "track_association_version":cpu_retrack::TRACK_ASSOCIATION_VERSION,"sha256":sha
            }),
        );
        emit(
            &project_dir,
            "native_cpu_ball_stabilizer",
            &trajectory_id,
            "trajectory.jsonl",
            &json!({
                "schema":"volley-native-stabilized-ball-trajectory/1","state":"complete",
                "algorithm_version":volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION,
                "artifact_id":trajectory_id,"project_id":request.project_id,"video":request.video,
                "source_replay_id":replay,"source_replay_sha256":sha,
                "frames":100,"settings":request.post.ball_stabilizer,
                "file":"trajectory.jsonl","sha256":sha
            }),
        );
        emit(
            &project_dir,
            "native_cpu_touch_runs",
            &touch,
            "touch_candidates.jsonl",
            &json!({
                "schema":"volley-native-cpu-touch-proposals/1","state":"motion_gated_touch_proposals","touch_id":touch,
                "project_id":request.project_id,"video":request.video,
                "source_replay_id":replay,"method":"vballnet_gravity_player_geometry_pose_contact_v11","sha256":sha,
                "body_trajectory_intersection_required":true,
                "settings":{"require_pose":true},
                "source_replay_sha256":sha,
                "ball_stabilizer":{
                    "schema":"volley-ball-stabilizer/1",
                    "version":volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION,
                    "source_replay_sha256":sha,
                    "source_replay_id":replay,
                    "artifact_id":trajectory_id,
                    "trajectory_file":"trajectory.jsonl",
                    "trajectory_sha256":sha,
                    "frames":100,
                    "settings":request.post.ball_stabilizer
                },
                "automatic_action_prediction_method":volley_core::pose_classification::ACTION_PREDICTION_POLICY_VERSION,
                "initial_serve_attribution_version":crate::cpu_touches::INITIAL_SERVE_ATTRIBUTION_VERSION,
                "touch_side_attribution_version":crate::court_touch::TOUCH_SIDE_ATTRIBUTION_VERSION,
                "trajectory_detector_version":volley_core::touch_pipeline::TRAJECTORY_DETECTOR_VERSION,
                "ground_detector_version":volley_core::motion::GROUND_DETECTOR_VERSION,
                "preliminary_rally_scope_version":"primary_gpu_rally_mask_v1",
                "preliminary_rally_settings":request.post.rallies,
                "primary_ball_job_id":gpu_ball,"primary_rally_mask_sha256":sha,
                "preliminary_rally_scope_applied":false,
                "ball_tracking_policy":crate::cpu_retrack::ball_track::LOCAL_POLICY,
                "leading_ball_rally_starts":[],"leading_ball_min_score":0.28
            }),
        );
        emit(
            &project_dir,
            "native_cpu_rallies",
            &rally,
            "rally_candidates.jsonl",
            &json!({
                "schema":"volley-native-cpu-rally-proposals/1","state":"provisional_rally_segments","rally_run_id":rally,
                "court_contact_context_version":"not_computed_primary_gpu_v1",
                "ground_boundary_version":volley_core::motion::GROUND_DETECTOR_VERSION,
                "materialization_version":"primary_gpu_intervals_direct_v2",
                "rally_score_version":volley_core::rally_score::RALLY_SCORE_VERSION,
                "preliminary_rally_scope_version":"primary_gpu_rally_mask_v1",
                "preliminary_rally_scope_applied":false,
                "primary_ball_job_id":gpu_ball,"primary_rally_mask_sha256":sha,
                "ball_tracking_policy":crate::cpu_retrack::ball_track::LOCAL_POLICY,
                "project_id":request.project_id,"video":request.video,
                "source_replay_id":replay,"source_touch_id":touch,"sha256":sha,
                "leading_ball_rally_starts":[],"leading_ball_min_score":0.28
            }),
        );
        let saved = json!({
            "schema":SCHEMA,"project_id":request.project_id,"video":request.video,
            "status":"complete","pipeline_complete":true,"source_identity":identity,
            "gpu_model_stack":requested_gpu_model_stack(request.gpu.person_detector),
            "profile":{"gpu":request.gpu,"post":request.post},
            "gpu_jobs":{"ball":gpu_ball,"players":gpu_players},
            "gpu_rally_mask":{
                "schema":"volley-primary-rally-mask/1",
                "min_trajectory_seconds":request.gpu.primary_rallies.min_trajectory_seconds,
                "merge_gap_seconds":request.gpu.primary_rallies.merge_gap_seconds,
                "stationary_ball_ms":request.gpu.primary_rallies.stationary_ball_ms,
                "padding_before_ms":request.gpu.primary_rallies.padding_before_ms,
                "padding_after_ms":request.gpu.primary_rallies.padding_after_ms,
                "rally_intervals":[],
                "movement_filter_version":crate::cpu_retrack::ball_track::MOVEMENT_FILTER_VERSION,
                "min_rally_seconds":5.0,
                "serve_flight_version":volley_core::rally_reconstruction::serve_flight::SERVE_FLIGHT_VERSION,
                "ground_boundary_version":"ground_ends_rally_v6",
                "ground_detector":{"version":volley_core::motion::GROUND_DETECTOR_VERSION,"sha256":sha},
                "ball_stabilizer":{
                    "algorithm":volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION,
                    "trajectory_sha256":sha,
                    "settings":volley_core::ball_interpolation::BallStabilizerSettings::default()
                }
            },
            "gpu_rally_mask_sha256":sha,
            "ball_stabilizer":{
                "schema":"volley-ball-stabilizer/1",
                "version":volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION,
                "source_replay_sha256":sha,
                "source_replay_id":replay,
                "artifact_id":trajectory_id,
                "trajectory_file":"trajectory.jsonl",
                "trajectory_sha256":sha,
                "frames":100,
                "settings":request.post.ball_stabilizer
            },
            "analysis_id":analysis,"cpu_runs":{"tracking":replay,"touches":touch,"rallies":rally},
            "postprocessing_id":rally
        });
        let mut saved = saved;
        let run_id = "1".repeat(32);
        saved["id"] = json!(run_id);
        let directory = project_dir.join("native_studio_pipelines").join(&run_id);
        fs::create_dir_all(&directory).unwrap();
        storage::write_json_atomic(&directory.join("status.json"), &saved).unwrap();
        let state = AppState {
            repository_root: base.clone(),
            data_root: base.clone(),
            writes_enabled: false,
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
        let matching = find_reusable(&state, &request, &identity, &"2".repeat(32), false);
        assert!(
            matching.full.is_some(),
            "unchanged settings must reuse both versions"
        );
        assert!(matching.gpu.is_some());
        let names = project_dir.join("native_analysis_names.json");
        let queue = "a".repeat(32);
        let older_id = "b".repeat(32);
        let head_dir = project_dir.join("native_studio_pipelines").join(&queue);
        let older_dir = project_dir.join("native_studio_pipelines").join(&older_id);
        fs::create_dir(&head_dir).unwrap();
        fs::create_dir(&older_dir).unwrap();
        let mut older = saved.clone();
        older["id"] = json!(older_id);
        older["started_ms"] = json!(1);
        storage::write_json_atomic(&older_dir.join("status.json"), &older).unwrap();
        saved["queue_id"] = json!(queue);
        saved["queue_runs"] = json!([queue, run_id]);
        saved["started_ms"] = json!(2);
        let mut head = saved.clone();
        head["id"] = json!(queue);
        head["video"] = json!("other.mp4");
        storage::write_json_atomic(&head_dir.join("status.json"), &head).unwrap();
        storage::write_json_atomic(&directory.join("status.json"), &saved).unwrap();
        storage::write_json_atomic(&names, &json!({format!("hidden_launch:{queue}"):true}))
            .unwrap();
        let deleted = find_reusable(&state, &request, &identity, &"2".repeat(32), false);
        assert!(
            deleted.gpu.is_none() && deleted.full.is_none(),
            "deleted launch and older results must not supply cache"
        );
        older["started_ms"] = json!(3);
        storage::write_json_atomic(&older_dir.join("status.json"), &older).unwrap();
        let newer = find_reusable(&state, &request, &identity, &"2".repeat(32), false);
        assert!(
            newer.gpu.is_some() && newer.full.is_some(),
            "a newer visible analysis remains reusable"
        );
        fs::remove_dir_all(head_dir).unwrap();
        fs::remove_dir_all(older_dir).unwrap();
        fs::remove_file(names).unwrap();
        for field in ["queue_id", "queue_runs", "started_ms"] {
            saved.as_object_mut().unwrap().remove(field);
        }
        storage::write_json_atomic(&directory.join("status.json"), &saved).unwrap();

        let new_pipeline = find_reusable(&state, &request, &identity, &"2".repeat(32), true);
        assert!(new_pipeline.gpu.is_none() && new_pipeline.full.is_none());
        assert!(compatible_version(&saved, &request, &identity));

        let mut failed_pose = saved.clone();
        failed_pose["status"] = json!("failed");
        failed_pose["pipeline_complete"] = json!(false);
        failed_pose["profile"]["gpu"]["pose_enabled"] = json!(true);
        storage::write_json_atomic(&directory.join("status.json"), &failed_pose).unwrap();
        request.gpu.pose_enabled = true;
        request.mode = PipelineMode::Gpu;
        let recovered = find_reusable(&state, &request, &identity, &"2".repeat(32), false);
        assert!(recovered.full.is_none());
        assert!(
            recovered.gpu.is_some(),
            "failed RTMPose must reuse valid raw GPU evidence"
        );
        failed_pose["gpu_jobs"]["pose"] = json!("9".repeat(32));
        storage::write_json_atomic(&directory.join("status.json"), &failed_pose).unwrap();
        assert!(
            find_reusable(&state, &request, &identity, &"2".repeat(32), false)
                .gpu
                .is_none()
        );
        request.gpu.pose_enabled = false;
        request.mode = PipelineMode::Full;
        storage::write_json_atomic(&directory.join("status.json"), &saved).unwrap();
        let mut old_profile = saved.clone();
        for key in ["interpolate_players", "interpolate_pose"] {
            old_profile["profile"]["post"]
                .as_object_mut()
                .unwrap()
                .remove(key);
            old_profile["profile"]["gpu"][key] = json!(true);
        }
        storage::write_json_atomic(&directory.join("status.json"), &old_profile).unwrap();
        assert!(
            find_reusable(&state, &request, &identity, &"2".repeat(32), false)
                .full
                .is_some()
        );
        assert!(selected_gpu(&state, &request, &identity, &analysis).is_ok());
        for (players, pose) in [(false, true), (true, false), (false, false)] {
            request.post.interpolate_players = players;
            request.post.interpolate_pose = pose;
            let changed = find_reusable(&state, &request, &identity, &"2".repeat(32), false);
            assert!(changed.full.is_none());
            assert!(changed.gpu.is_some());
            assert!(!cpu_lineage_valid(&project_dir, &request, &saved, None));
        }
        request.post.interpolate_players = true;
        request.post.interpolate_pose = true;
        request.post.ball_stabilizer.max_gap_frames = 8;
        let changed = find_reusable(&state, &request, &identity, &"2".repeat(32), false);
        assert!(
            changed.full.is_none(),
            "changed Ball Stabilizer must invalidate CPU results"
        );
        assert!(
            changed.gpu.is_none(),
            "changing the primary trajectory invalidates dependent player detections"
        );
        assert!(cpu_lineage_valid(&project_dir, &request, &saved, None));
        request.post.ball_stabilizer = Default::default();
        storage::write_json_atomic(&directory.join("status.json"), &saved).unwrap();
        let mut legacy = saved.clone();
        legacy["gpu_model_stack"] = json!("vballnet_rtmdet_rtmpose_tiled_coco17_v3");
        assert!(!compatible_version(&legacy, &request, &identity));
        assert!(gpu_lineage_valid(&project_dir, &request, &identity, &saved));
        assert!(cpu_lineage_valid(&project_dir, &request, &saved, None));
        for version in [
            "ball_stabilizer_v1",
            "ball_stabilizer_v2",
            "ball_stabilizer_v3",
            "ball_stabilizer_v4",
            "ball_stabilizer_v5",
        ] {
            let mut old_version = saved.clone();
            old_version["ball_stabilizer"]["version"] = json!(version);
            assert!(
                !cpu_lineage_valid(&project_dir, &request, &old_version, None),
                "Previous Ball Stabilizer algorithm must invalidate CPU reuse"
            );
        }
        let trajectory_manifest = project_dir
            .join("native_cpu_ball_stabilizer")
            .join(&trajectory_id)
            .join("manifest.json");
        let current_trajectory: Value =
            serde_json::from_slice(&fs::read(&trajectory_manifest).unwrap()).unwrap();
        let mut previous_trajectory = current_trajectory.clone();
        previous_trajectory["algorithm_version"] = json!("ball_stabilizer_v1");
        storage::write_json_atomic(&trajectory_manifest, &previous_trajectory).unwrap();
        assert!(
            !cpu_lineage_valid(&project_dir, &request, &saved, None),
            "An old saved trajectory must invalidate CPU reuse"
        );
        storage::write_json_atomic(&trajectory_manifest, &current_trajectory).unwrap();
        assert!(cpu_lineage_valid(&project_dir, &request, &saved, None));
        let touch_manifest = project_dir
            .join("native_cpu_touch_runs")
            .join(&touch)
            .join("manifest.json");
        let original_touch: Value =
            serde_json::from_slice(&fs::read(&touch_manifest).unwrap()).unwrap();
        for (field, old_value) in [
            (
                "trajectory_detector_version",
                json!("piecewise_ballistic_motion_v8"),
            ),
            ("initial_serve_attribution_version", Value::Null),
            ("body_trajectory_intersection_required", json!(false)),
            (
                "automatic_action_prediction_method",
                json!("pose_keypoints_motion_v1_or_unknown_touch"),
            ),
            (
                "automatic_action_prediction_method",
                json!("pose_keypoints_v1_with_geometry_fallback"),
            ),
        ] {
            let mut incompatible = original_touch.clone();
            incompatible[field] = old_value;
            storage::write_json_atomic(&touch_manifest, &incompatible).unwrap();
            assert!(
                !cpu_lineage_valid(&project_dir, &request, &saved, None),
                "Old CPU evidence must not be reused after pose policy changes: {field}"
            );
            let reusable = find_reusable(&state, &request, &identity, &"2".repeat(32), false);
            assert!(reusable.full.is_none());
            assert!(reusable.gpu.is_some());
        }
        storage::write_json_atomic(&touch_manifest, &original_touch).unwrap();
        assert!(cpu_lineage_valid(&project_dir, &request, &saved, None));
        request.post.touches.algorithm = TouchAlgorithm::PoseMotion;
        let obsolete_pose = find_reusable(&state, &request, &identity, &"2".repeat(32), false);
        assert!(
            obsolete_pose.full.is_none(),
            "PoseMotion v1 must not be reused after the body-contact gate is upgraded"
        );
        assert!(
            obsolete_pose.gpu.is_some(),
            "The existing GPU source is still reusable"
        );
        request.post.touches.algorithm = TouchAlgorithm::Legacy;

        // Reconstructing touches differently reuses the same GPU analysis.
        request.post.touches.threshold = 0.61;
        let changed_cpu = find_reusable(&state, &request, &identity, &"2".repeat(32), false);
        assert!(changed_cpu.full.is_none());
        assert!(
            changed_cpu.gpu.is_some(),
            "CPU change must reuse the GPU fusion"
        );
        assert!(compatible_version(&saved, &request, &identity));

        let mut full_scan_fallback = saved.clone();
        full_scan_fallback["gpu_rally_mask"]["fallback_reason"] =
            json!("no_reliable_moving_ball_intervals");
        assert!(!gpu_lineage_valid(
            &project_dir,
            &request,
            &identity,
            &full_scan_fallback
        ));
        assert!(gpu_lineage_valid(&project_dir, &request, &identity, &saved));
        assert_ne!(
            saved["profile"]["post"],
            serde_json::to_value(&request.post).unwrap()
        );

        // Changing inference parameters or video identity cannot reuse the GPU branch.
        request.gpu.stride = 8;
        let changed_gpu = find_reusable(&state, &request, &identity, &"2".repeat(32), false);
        assert!(changed_gpu.full.is_none() && changed_gpu.gpu.is_none());
        assert!(!compatible_version(&saved, &request, &identity));
        request.gpu.stride = 9;
        let other_source = json!({"bytes":2048,"mtime_ns":123456790u64});
        let replaced_source =
            find_reusable(&state, &request, &other_source, &"2".repeat(32), false);
        assert!(replaced_source.gpu.is_none() && replaced_source.full.is_none());
        assert!(!compatible_version(&saved, &request, &other_source));
        assert!(!gpu_lineage_valid(
            &project_dir,
            &request,
            &other_source,
            &saved
        ));

        // Forged lineage cannot be selected even when a profile matches.
        let mut tampered = saved.clone();
        tampered["gpu_jobs"]["ball"] = json!("7".repeat(32));
        assert!(!gpu_lineage_valid(
            &project_dir,
            &request,
            &identity,
            &tampered
        ));
        tampered = saved.clone();
        tampered["cpu_runs"]["tracking"] = json!("8".repeat(32));
        assert!(!cpu_lineage_valid(&project_dir, &request, &tampered, None));
        // Diagnostics persist the exact native arguments and process argv.
        let logfile = directory.join("status.json");
        log_native(
            &logfile,
            "GPU",
            "inference_full::start",
            json!({"model":"ball","stride":9,"cpu_threads":8}),
        )
        .unwrap();
        log_entry(
            &logfile,
            "info",
            "invocation",
            "Подготовка",
            "FFmpeg Command::spawn",
            Some(json!({"type":"process","name":"ffmpeg",
                "argv":["ffmpeg","-i","match.mp4","-y","output.mp4"]})),
        )
        .unwrap();
        let gpu_selected = selected_gpu(&state, &request, &identity, &analysis).unwrap();
        assert_eq!(gpu_selected["analysis_id"], analysis);
        assert_eq!(gpu_selected["gpu_jobs"]["ball"], gpu_ball);
        assert!(selected_gpu(&state, &request, &identity, &"7".repeat(32)).is_err());
        assert!(selected_gpu(
            &state,
            &request,
            &json!({"bytes":42,"mtime_ns":123456789u64}),
            &analysis
        )
        .is_err());
        let handler_state = Arc::new(state);
        let Json(first) = log(
            State(handler_state.clone()),
            Query(LogQuery {
                project_id: "abc123def456".into(),
                id: run_id.clone(),
                after: 0,
            }),
        )
        .await
        .unwrap();
        assert_eq!(first["schema"], "volley-studio-pipeline-log/1");
        assert_eq!(first["entries"].as_array().unwrap().len(), 2);
        assert_eq!(first["entries"][0]["call"]["name"], "inference_full::start");
        assert_eq!(first["entries"][0]["call"]["arguments"]["stride"], 9);
        assert_eq!(first["entries"][1]["call"]["argv"][1], "-i");
        assert_eq!(first["next_offset"], 2);
        let Json(second) = log(
            State(handler_state.clone()),
            Query(LogQuery {
                project_id: "abc123def456".into(),
                id: run_id.clone(),
                after: 1,
            }),
        )
        .await
        .unwrap();
        assert_eq!(second["entries"].as_array().unwrap().len(), 1);
        assert_eq!(second["entries"][0]["call"]["type"], "process");
        assert!(log(
            State(handler_state.clone()),
            Query(LogQuery {
                project_id: "abc123def456".into(),
                id: "7".repeat(32),
                after: 0
            })
        )
        .await
        .is_err());
        fs::write(
            directory.join("events.jsonl"),
            concat!(
                "{\"timestamp_ms\":1000,\"kind\":\"stage_start\",\"stage\":\"GPU\"}\n",
                "{\"timestamp_ms\":3000,\"kind\":\"stage_start\",\"stage\":\"CPU\"}\n",
                "{\"timestamp_ms\":4500,\"kind\":\"event\",\"stage\":\"Анализ\"}\n"
            ),
        )
        .unwrap();
        let history = history_rows(&handler_state, "abc123def456", 0).unwrap();
        assert_eq!(history["runs"][0]["id"], run_id);
        assert_eq!(history["runs"][0]["duration_ms"], 3500);
        assert_eq!(history["runs"][0]["analysis_id"], analysis);
        assert_eq!(history["runs"][0]["gpu_jobs"], saved["gpu_jobs"]);
        assert_eq!(history["runs"][0]["cpu_runs"], saved["cpu_runs"]);
        assert_eq!(
            history["runs"][0]["profile"]["gpu"]["pose_enabled"],
            saved["profile"]["gpu"]["pose_enabled"]
        );
        assert_eq!(history["runs"][0]["automatic_identity_available"], false);
        assert_eq!(history["runs"][0]["stages"][0]["duration_ms"], 2000);
        assert_eq!(history["runs"][0]["stages"][1]["duration_ms"], 1500);
        // CPU can use a standalone GPU fusion with no Studio pipeline record.
        fs::remove_file(directory.join("status.json")).unwrap();
        let player_manifest = project_dir
            .join("native_gpu_runs")
            .join(&gpu_players)
            .join("manifest.json");
        let mut raw: Value = serde_json::from_slice(&fs::read(&player_manifest).unwrap()).unwrap();
        raw["person_pipeline"] = json!(
            "rfdetr_large_da7f1bc1_detector_boxes_pose_deferred_sampled_bytetrack_lowconf_v4"
        );
        storage::write_json_atomic(&player_manifest, &raw).unwrap();
        let source = selected_gpu(&handler_state, &request, &identity, &analysis).unwrap();
        assert_eq!(source["gpu_jobs"]["ball"], gpu_ball);
        assert_eq!(source["gpu_jobs"]["players"], gpu_players);
        assert_eq!(source["profile"]["gpu"]["stride"], 9);
        assert_eq!(source["profile"]["gpu"]["person_detector"], "rfdetr-large");
        assert!(selected_gpu(
            &handler_state,
            &request,
            &json!({"bytes":42,"mtime_ns":123456789u64}),
            &analysis
        )
        .is_err());
        // Regression: the user-selected GPU source is valid even when the
        // browser sends a different/default GPU profile for CPU-only runs.
        // Such a run must never invoke a model or write a GPU stage status.
        let mut post = request.clone();
        post.mode = PipelineMode::Post;
        post.source_analysis_id = Some(analysis.clone());
        post.gpu.stride = 3;
        let selected = selected_gpu(&handler_state, &post, &identity, &analysis).unwrap();
        assert_eq!(selected["analysis_id"], analysis);
        assert_eq!(selected["gpu_jobs"]["ball"], gpu_ball);
        assert!(!post.mode.allows_gpu_execution());
        let untouched = directory.join("post-must-not-write-gpu.json");
        let mut post_status = json!({"stage":"CPU-only"});
        let refusal = gpu(
            &handler_state,
            &post,
            ModelKind::Ball,
            &AtomicBool::new(false),
            &untouched,
            &mut post_status,
            8.0,
            36.0,
        )
        .await
        .unwrap_err();
        assert!(refusal.contains("не имеет права запускать GPU"));
        assert!(!untouched.exists());
        assert_eq!(post_status["stage"], "CPU-only");
        for index in 0..26 {
            let id = format!("{index:032x}");
            let run = project_dir.join("native_studio_pipelines").join(&id);
            fs::create_dir_all(&run).unwrap();
            fs::write(
                run.join("status.json"),
                serde_json::to_vec(&json!({
                    "schema":SCHEMA,"id":id,"project_id":"abc123def456","status":"complete"
                }))
                .unwrap(),
            )
            .unwrap();
        }
        let indexed_status = status_snapshot(
            &handler_state,
            &StatusQuery {
                project_id: "abc123def456".into(),
                id: Some(format!("{:032x}", 0)),
                queue_id: None,
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(indexed_status.0, format!("{:032x}", 0));
        assert_eq!(indexed_status.1["status"], "complete");
        let page = history_rows(&handler_state, "abc123def456", 0).unwrap();
        assert_eq!(page["runs"].as_array().unwrap().len(), 25);
        assert_eq!(page["next_offset"], 25);
        assert_eq!(page["has_more"], true);
        let page2 = history_rows(&handler_state, "abc123def456", 25).unwrap();
        assert_eq!(page2["runs"].as_array().unwrap().len(), 1);
        assert_eq!(page2["has_more"], false);
        let ids = vec!["a".repeat(32), "b".repeat(32)];
        let queue_root = project_dir.join("native_studio_pipelines");
        for (index, id) in ids.iter().enumerate() {
            let dir = queue_root.join(id);
            fs::create_dir(&dir).unwrap();
            save(&dir.join("status.json"), &json!({
                "schema":SCHEMA,"id":id,"project_id":"abc123def456","queue_id":ids[0],
                "queue_runs":ids,"mode":"full","video":format!("{index}.mp4"),
                "status":if index == 0 {"complete"} else {"running"},
                "started_ms":1000 + index * 1000,"finished_ms":if index == 0 {Some(4000)} else {None}
            })).unwrap();
        }
        let active = ids.iter().cloned().collect();
        let page = archived_history_rows(&handler_state, "abc123def456", 0, &active).unwrap();
        assert_eq!(page["runs"].as_array().unwrap().len(), 25);
        assert!(page["runs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["id"] != ids[0] && row["id"] != ids[1]));
        let path = queue_root.join(&ids[1]).join("status.json");
        let mut second = read(&path).unwrap();
        second["status"] = json!("failed");
        second["error"] = json!("test failure");
        second["finished_ms"] = json!(5000);
        save(&path, &second).unwrap();
        let page =
            archived_history_rows(&handler_state, "abc123def456", 0, &BTreeSet::new()).unwrap();
        let row = &page["runs"][0];
        assert_eq!(row["id"], ids[0]);
        assert_eq!(row["status"], "failed");
        assert_eq!(row["duration_ms"], 4000);
        assert_eq!(row["videos"].as_array().unwrap().len(), 2);
        second["status"] = json!("complete");
        second["error"] = Value::Null;
        save(&path, &second).unwrap();
        let page =
            archived_history_rows(&handler_state, "abc123def456", 0, &BTreeSet::new()).unwrap();
        assert_eq!(page["runs"][0]["status"], "complete");
        assert_eq!(
            archived_history_rows(&handler_state, "abc123def456", 25, &BTreeSet::new()).unwrap()
                ["runs"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        for id in ids {
            fs::remove_dir_all(queue_root.join(id)).unwrap();
        }
        clear_history_rows(&handler_state, "abc123def456", None).unwrap();
        assert!(
            history_rows(&handler_state, "abc123def456", 0).unwrap()["runs"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(project_dir
            .join("native_studio_pipelines")
            .join(format!("{:032x}", 0))
            .join("status.json")
            .exists());
        let new_id = "f".repeat(32);
        let new_run = project_dir.join("native_studio_pipelines").join(&new_id);
        fs::create_dir_all(&new_run).unwrap();
        fs::write(new_run.join("status.json"), serde_json::to_vec(&json!({"schema":SCHEMA,"id":new_id,"project_id":"abc123def456","status":"complete"})).unwrap()).unwrap();
        assert_eq!(
            history_rows(&handler_state, "abc123def456", 0).unwrap()["runs"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        clear_history_rows(&handler_state, "abc123def456", Some(&new_id)).unwrap();
        assert_eq!(
            history_rows(&handler_state, "abc123def456", 0).unwrap()["runs"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        clear_history_rows(&handler_state, "abc123def456", None).unwrap();
        assert!(
            history_rows(&handler_state, "abc123def456", 0).unwrap()["runs"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        fs::remove_dir_all(base).unwrap();
    }
}

#[cfg(test)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "explicit isolated project benchmark with native CUDA models"]
async fn isolated_project_rally_pipeline_benchmark() {
    let data_root = PathBuf::from(std::env::var("VOLLEY_BENCH_DATA").unwrap());
    let repository_root = PathBuf::from(std::env::var("VOLLEY_BENCH_REPO").unwrap());
    let request: StartRequest =
        serde_json::from_slice(&fs::read(data_root.join("benchmark_request.json")).unwrap())
            .unwrap();
    let project_id = request.project_id.clone();
    let post_only = request.mode == PipelineMode::Post;
    let state = Arc::new(AppState {
        repository_root,
        data_root: data_root.clone(),
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
    });
    let started = std::time::Instant::now();
    let Json(queued) = start(State(state), HeaderMap::new(), Json(request))
        .await
        .unwrap();
    let path = data_root
        .join("projects")
        .join(&project_id)
        .join("native_studio_pipelines")
        .join(queued["id"].as_str().unwrap())
        .join("status.json");
    let mut last = Value::Null;
    let outcome = loop {
        let status = read(&path).unwrap();
        if status["stage"] != last {
            eprintln!(
                "BENCH_STAGE {:.3}s {}",
                started.elapsed().as_secs_f64(),
                status["stage"]
            );
            last = status["stage"].clone();
        }
        if matches!(
            status["status"].as_str(),
            Some("complete" | "failed" | "cancelled")
        ) {
            break status;
        }
        assert!(
            started.elapsed().as_secs() < 300,
            "Benchmark exceeded 300 seconds"
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    };
    eprintln!(
        "BENCH_RESULT {}",
        json!({
            "wall_seconds": started.elapsed().as_secs_f64(),
            "status":outcome["status"], "error":outcome["error"], "path":path,
            "early_pose_groups":outcome["early_pose_groups"],
            "player_tracking_stats":outcome["player_tracking_stats"],
            "cpu_runs":outcome["cpu_runs"], "gpu_jobs":outcome["gpu_jobs"],
            "stages":outcome["stages"], "automatic_identity_complete":outcome["automatic_identity_complete"],
            "identity_count":outcome["automatic_identity"]["automatic_count"],
            "identity_error":outcome["automatic_identity_error"], "zero_copy":outcome["zero_copy_active"],
            "gpu_reused":outcome["gpu_reused"], "result_reused":outcome["result_reused"]
        })
    );
    assert_eq!(outcome["status"], "complete", "{outcome}");
    assert_eq!(outcome["gpu_reused"], post_only);
    assert_ne!(outcome["result_reused"], true);
}
