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
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
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
    let (root, ids) =
        tokio::task::spawn_blocking(move || collect_history_rows(&scan_state, &scan_project))
            .await
            .map_err(|e| storage::internal(format!("Scan pipeline history worker: {e}")))??;
    let slot = manager().lock().await;
    let _guard = project_guard::write_guard(&state, &request.project_id)?;
    let active = slot
        .as_ref()
        .filter(|run| run.project_id == request.project_id)
        .map(|run| run.id.as_str());
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
        let log_path = path.with_file_name("events.jsonl");
        let records: Vec<Value> = match fs::symlink_metadata(&log_path) {
            Ok(meta)
                if meta.is_file()
                    && !meta.file_type().is_symlink()
                    && meta.len() <= MAX_LOG_BYTES =>
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
        let finished = if ["complete", "failed", "cancelled"]
            .contains(&status["status"].as_str().unwrap_or(""))
        {
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
        rows.push(json!({"id":id,"video":status["video"],"mode":status["mode"],
            "status":status["status"],"stage":status["stage"],"error":status["error"],
            "started_ms":started,"duration_ms":started.zip(finished).and_then(|(a,b)| b.checked_sub(a)),
            "gpu_reused":status["gpu_reused"],"result_reused":status["result_reused"],
            "stages":stages}));
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

pub async fn history(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HistoryQuery>,
) -> ApiResult<Value> {
    let result =
        tokio::task::spawn_blocking(move || history_rows(&state, &query.project_id, query.offset))
            .await
            .map_err(|e| storage::internal(format!("Pipeline history worker: {e}")))??;
    let mut result = result;
    let project_id = result["project_id"].as_str().unwrap_or("").to_owned();
    let slot = manager().lock().await;
    if let Some(runs) = result["runs"].as_array_mut() {
        for run in runs {
            if ["queued", "running", "cancelling"].contains(&run["status"].as_str().unwrap_or(""))
                && !slot.as_ref().is_some_and(|active| {
                    Some(active.id.as_str()) == run["id"].as_str()
                        && active.project_id == project_id
                })
            {
                run["status"] = json!("interrupted");
            }
        }
    }
    Ok(Json(result))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct GpuProfile {
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
    #[serde(default)]
    primary_rallies: inference_full::PrimaryRallySettings,
}
const fn default_interpolation_enabled() -> bool {
    true
}
impl Default for GpuProfile {
    fn default() -> Self {
        Self {
            stride: 9,
            gpu_memory_limit_mb: 0,
            cpu_threads: 0,
            pose_enabled: true,
            pose_mode: crate::pose_inference::PoseInputMode::Photo,
            pose_stride: volley_inference::person::PERSON_SAMPLE_PERIOD_FRAMES,
            primary_rallies: inference_full::PrimaryRallySettings::default(),
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
}
fn default_pose_classification() -> bool {
    true
}
fn default_auto_assign_threshold_percent() -> u8 {
    75
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
    Ok((
        serde_json::from_value(gpu).map_err(|e| e.to_string())?,
        serde_json::from_value(post).map_err(|e| e.to_string())?,
    ))
}

impl TryFrom<StartRequestWire> for StartRequest {
    type Error = String;

    fn try_from(wire: StartRequestWire) -> Result<Self, Self::Error> {
        // Historical CPU snapshots may still deserialize the old algorithm,
        // but new Studio requests have one default plus a separate opt-in TCN.
        if wire
            .post
            .pointer("/touches/algorithm")
            .and_then(Value::as_str)
            == Some("legacy")
        {
            return Err(
                "Legacy не поддерживается для новых запусков; выберите PoseMotion или TCN".into(),
            );
        }
        let (gpu, mut post) = decode_profiles(wire.gpu, wire.post)?;
        gpu.primary_rallies.validate()?;
        post.ball_stabilizer.validate().map_err(str::to_owned)?;
        if !(1..=99).contains(&post.auto_assign_threshold_percent) {
            return Err("Порог сходства для автопривязки должен быть от 1 до 99%".into());
        }
        // TouchSettings::default() remains legacy for old saved data. An API
        // request with partial touch parameters uses PoseMotion unless TCN is explicit.
        if post.touches.algorithm != TouchAlgorithm::TrajectoryTcn {
            post.touches.algorithm = TouchAlgorithm::PoseMotion;
        }
        post.touches.require_pose = true;
        Ok(Self {
            project_id: wire.project_id,
            video: wire.video,
            gpu,
            post,
            mode: wire.mode,
            source_analysis_id: wire.source_analysis_id,
            client_request_id: wire.client_request_id,
            analysis_range: None,
        })
    }
}

#[derive(Deserialize)]
pub struct StatusQuery {
    project_id: String,
    id: Option<String>,
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
}
fn manager() -> &'static Mutex<Option<Running>> {
    static MANAGER: OnceLock<Mutex<Option<Running>>> = OnceLock::new();
    MANAGER.get_or_init(|| Mutex::new(None))
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
    ["source_analysis_id", "analysis_id", "reused_from_id"]
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
    let running_id = {
        let slot = manager().lock().await;
        slot.as_ref()
            .filter(|job| job.project_id == project_id)
            .map(|job| job.id.clone())
    };
    let Some(running_id) = running_id else {
        return Ok(false);
    };
    let root = folder(state, project_id, false)?;
    let running_path = status_path(&root, &running_id)?;
    let status = tokio::task::spawn_blocking(move || read(&running_path))
        .await
        .map_err(|e| storage::internal(format!("Read active pipeline worker: {e}")))??;
    let slot = manager().lock().await;
    if !slot
        .as_ref()
        .is_some_and(|job| job.project_id == project_id && job.id == running_id)
    {
        return Ok(true);
    }
    Ok(status_uses_analysis(&status, video, artifact_id))
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
    if name.contains("Проверка исходного видео") || name.contains("Подготовка исходного видео")
    {
        "video"
    } else if name.contains("Ball Stabilizer") {
        "ball_stabilizer"
    } else if name.contains("Поиск розыгрышей") {
        "rallies"
    } else if name.contains("VBallNet") && name.contains("GPU") {
        "ball"
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
    } else if name.contains("касания в первичных") || name.contains("физические касания")
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
        8.0 + 72.0 * 0.399999,
        Some((decoded_frames, decoded_frames)),
    )?;
    status["gpu_rally_mask"] = mask.clone();
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
    let intervals: Vec<(u64, u64)> = serde_json::from_value(mask["merged"].clone())
        .map_err(|_| "Некорректные границы первичных розыгрышей")?;
    if intervals.iter().any(|&(start, end)| {
        start > end
            || end >= frames
            || end - start + 1 < crate::cpu_retrack::ball_track::MIN_RALLY_FRAMES
    }) || intervals.windows(2).any(|pair| pair[0].1 >= pair[1].0)
        || mask["movement_filter_version"] != "stationary_ball_min_rally_5s_v3"
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
            || saved["gpu_rally_mask"]["movement_filter_version"]
                != "stationary_ball_min_rally_5s_v3"
            || saved["gpu_rally_mask"]["min_rally_seconds"] != 5.0)
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
    if request.gpu.pose_enabled {
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
        if saved["gpu_model_stack"] == requested_gpu_model_stack()
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
        && matches!(
            touches["automatic_action_prediction_method"].as_str(),
            Some("pose_keypoints_v1_or_unknown_touch" | "pose_keypoints_motion_v1_or_unknown_touch")
        )
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
        && touches["ball_stabilizer"]["settings"] == json!(request.post.ball_stabilizer)
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
        })
        && rallies["schema"] == "volley-native-cpu-rally-proposals/1"
        && rallies["state"] == "provisional_rally_segments"
        && rallies["court_contact_context_version"] == "not_computed_primary_gpu_v1"
        && rallies["ground_boundary_version"] == "not_computed_primary_gpu_v1"
        && rallies["materialization_version"] == "primary_gpu_intervals_direct_v1"
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
    "vballnet_rfdetr_medium_18250b5a_original30_native_pose_pixels_bytetrack_lowconf_v7";

/// The GPU-evidence cache must not be shared by different detector weights
/// or detector-to-pose box policies.
fn person_detector_label() -> &'static str {
    match volley_inference::person::DetectorVariant::from_env() {
        Ok(volley_inference::person::DetectorVariant::RfDetrMedium) => "RF-DETR Medium",
        Err(_) => "Детектор игроков + RTMPose",
    }
}

fn zero_copy_requested() -> bool {
    matches!(
        std::env::var("VOLLEY_ZERO_COPY_PRIMARY").ok().as_deref(),
        None | Some("1") | Some("true")
    )
}

fn requested_gpu_model_stack() -> &'static str {
    match volley_inference::person::DetectorVariant::from_env() {
        Ok(volley_inference::person::DetectorVariant::RfDetrMedium) => RFDETR_GPU_MODEL_STACK,
        Err(_) => "invalid_person_detector_selection_no_gpu_cache_reuse",
    }
}

fn compatible_version(saved: &Value, request: &StartRequest, identity: &Value) -> bool {
    saved["schema"] == SCHEMA
        && saved["analysis_range"] == json!(request.analysis_range)
        && saved["gpu_model_stack"] == requested_gpu_model_stack()
        && saved["project_id"] == request.project_id
        && saved["video"] == request.video
        && decode_profiles(saved["profile"]["gpu"].clone(), empty_profile())
            .is_ok_and(|(gpu, _)| json!(gpu) == json!(request.gpu))
        // Older pipeline status files can predate the source_identity field;
        // the fusion and both GPU manifests are always validated separately.
        && (saved["source_identity"].is_null() || saved["source_identity"] == *identity)
}
fn find_reusable(
    state: &AppState,
    request: &StartRequest,
    identity: &Value,
    current_id: &str,
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
        let complete_gpu = gpu_lineage_valid(&project, request, identity, &saved);
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
            || !compatible_version(&saved, request, identity)
            || !(complete_gpu || recoverable_pose)
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
        stride = Some(current_stride);
    }
    Ok(json!({
        "analysis_id": analysis_id,
        "gpu_jobs": {"ball":ball,"players":players},
        "analysis_range":fusion["analysis_range"],
        "id": analysis_id,
        // Standalone GPU manifests preserve the player sampling period.
        "profile": {"gpu": {"stride":stride.unwrap_or(9),
            "pose_stride":player_stride,"gpu_memory_limit_mb":0,"cpu_threads":8}}
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
        person_detector_label()
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
    let Json(start) = inference_full::start(State(app.clone()), HeaderMap::new(), Json(config))
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
    handler: fn(&AppState, &T, &AtomicBool) -> Result<Value, ApiError>,
) -> Result<Value, String> {
    tokio::task::spawn_blocking(move || handler(&app, &request, &cancel))
        .await
        .map_err(|e| format!("CPU worker: {e}"))?
        .map_err(reason)
}

async fn pipeline(
    app: &Arc<AppState>,
    request: &StartRequest,
    cancel: &Arc<AtomicBool>,
    path: &Path,
    status: &mut Value,
) -> Result<(), String> {
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    #[cfg(any(unix, windows))]
    let mut pending_shared_pose: Option<crate::pose_inference::PendingPose>;
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
    let court_revision =
        crate::video_edits::court_snapshot(app, &request.project_id, &request.video)
            .map_err(reason)?
            .map(|court| court.revision);
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
        let reusable = find_reusable(app, request, &identity, current_id);
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
        if request.post.touches.require_pose
            && crate::native_preview_pose::source_info(
                &projects::project_root(app).join(&request.project_id),
                source_id,
            )?
            .is_none()
        {
            return Err("Для пересчёта по движению точек тела в выбранном GPU-анализе нужна сохранённая поза".into());
        }
        status["analysis_id"] = source["analysis_id"].clone();
        status["analysis_range"] = source["analysis_range"].clone();
        status["gpu_jobs"] = source["gpu_jobs"].clone();
        status["shared_pose"] = source["shared_pose"].clone();
        let (gpu, _) = decode_profiles(source["profile"]["gpu"].clone(), empty_profile())?;
        status["profile"]["gpu"] = json!(gpu);
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
            && v["automatic_identity"].is_object()
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
        status["gpu_rally_mask"] = existing["gpu_rally_mask"].clone();
        status["gpu_rally_mask_sha256"] = existing["gpu_rally_mask_sha256"].clone();
        status["cpu_runs"] = existing["cpu_runs"].clone();
        status["player_tracking_stats"] = existing["player_tracking_stats"].clone();
        status["postprocessing_id"] = existing["postprocessing_id"].clone();
        status["gpu_reused"] = json!(true);
        status["result_reused"] = json!(true);
        status["reused_from_id"] = existing["id"].clone();
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
        return Ok(());
    }
    if request.mode.allows_gpu_execution() {
        if let Some(existing) = reusable.gpu {
            status["analysis_id"] = existing["analysis_id"].clone();
            status["gpu_jobs"] = existing["gpu_jobs"].clone();
            status["shared_pose"] = existing["shared_pose"].clone();
            status["gpu_rally_mask"] = existing["gpu_rally_mask"].clone();
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
                    person_detector_label()
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
                // A single permit covers both persistent model sessions. It
                // also prevents a standalone GPU job from starting concurrently.
                let _permit = app
                    .inference_limit
                    .acquire()
                    .await
                    .map_err(|e| format!("Shared GPU inference limit: {e}"))?;
                let full_resolution_pose = if request.gpu.pose_enabled {
                    let native_info = volley_media::probe(&plan.source)?;
                    shared_native_pose_supported(true, native_info.width, native_info.height)
                } else {
                    false
                };
                if request.gpu.pose_enabled && !full_resolution_pose {
                    return Err("RTMPose не может использовать исходные кадры этого разрешения в общем конвейере; отдельный полный проход запрещён".into());
                }
                let total = status["gpu_total_frames"].as_u64().unwrap_or(0);
                progress_with_frames(
                    path,
                    status,
                    "GPU · VBallNet: траектория мяча и розыгрыши",
                    8.0,
                    Some((0, total)),
                )?;
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
                        full_resolution_pose,
                        request.gpu.gpu_memory_limit_mb,
                        request.gpu.cpu_threads,
                        request.gpu.primary_rallies,
                        cancel,
                        |fraction, processed, total, rally_mask| {
                            if let Some((mask, decoded_frames)) = rally_mask {
                                publish_primary_rally_mask(path, status, mask, decoded_frames)?;
                            }
                            let label = if fraction < 0.4 {
                                "GPU · VBallNet: траектория мяча и розыгрыши"
                            } else if full_resolution_pose {
                                "GPU · RF-DETR + RTMPose: игровые интервалы"
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
                status["zero_copy_active"] = json!(result.zero_copy);
                status["gpu_rally_mask"] = result.rally_filter.unwrap_or(Value::Null);
                save(path, status)?;
                pending_shared_pose = result.pending_pose;
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

    if request.mode.allows_gpu_execution()
        && request.gpu.pose_enabled
        && status["gpu_jobs"]["pose"].as_str().is_none()
    {
        return Err(
            "Не получены позы из общего GPU-конвейера; отдельный повторный проход RTMPose запрещён"
                .into(),
        );
    }
    if request.mode == PipelineMode::Gpu {
        progress(path, status, "Первичный GPU-анализ завершён", 99.9)?;
        return Ok(());
    }
    let (primary, primary_mask) = load_primary_rally_scope(app, &request.project_id, status)?;
    status["gpu_rally_mask"] = primary_mask;
    status["gpu_rally_mask_sha256"] = json!(primary.sha256);
    status["primary_rally_source"] =
        json!({"ball_job_id":primary.ball_job_id,"sha256":primary.sha256});
    save(path, status)?;
    let fusion_id = require_id(status, "analysis_id")?;
    progress(path, status, "Постобработка · Track ID", 86.0)?;
    log_native(
        path,
        "Track ID",
        "cpu_retrack::process",
        json!({"project_id":request.project_id,"video":request.video,"fusion_id":fusion_id,"settings":request.post.tracking}),
    )?;
    let replay = cpu_cancel(
        app.clone(),
        cpu_retrack::ReplayRequest {
            project_id: request.project_id.clone(),
            video: request.video.clone(),
            fusion_id,
            settings: Some(request.post.tracking),
            interpolate_pose: request.post.interpolate_pose,
            interpolate_players: request.post.interpolate_players,
        },
        cancel.clone(),
        cpu_retrack::process_with_cancel,
    )
    .await?;
    let replay_id = require_id(&replay, "replay_id")?;
    if replay["player_frame_scope_version"] != "primary_gpu_selected_intervals_v1"
        || replay["player_frame_scope_sha256"] != status["gpu_rally_mask_sha256"]
        || replay["player_frame_scope_ball_job_id"] != status["gpu_jobs"]["ball"]
    {
        return Err("Трекинг CPU вышел за границы проверенной GPU-маски розыгрышей".into());
    }
    status["cpu_runs"]["tracking"] = json!(replay_id);
    status["player_tracking_stats"] = json!({
        "raw_detections": replay["raw_player_detections"],
        "total_track_ids": replay["tentative_track_count"],
        "short_track_ids": replay["discarded_short_track_ids"],
        "track_ids": replay["provisional_track_count"],
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

    progress(path, status, "Постобработка · Ball Stabilizer", 90.0)?;
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
                    "Постобработка · Ball Stabilizer",
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
        "Ball Stabilizer",
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
    progress(
        path,
        status,
        "Постобработка · касания в первичных розыгрышах",
        91.0,
    )?;
    log_native(
        path,
        "Касания",
        "cpu_touches::process_primary_with_progress_cancel",
        json!({"project_id":request.project_id,"video":request.video,"replay_id":replay_id,"settings":request.post.touches,"primary_ball_job_id":primary.ball_job_id}),
    )?;
    let touches = tokio::task::block_in_place(|| {
        cpu_touches::process_primary_with_progress_cancel(
            app,
            &cpu_touches::TouchRequest {
                project_id: request.project_id.clone(),
                video: request.video.clone(),
                replay_id: replay_id.clone(),
                settings: Some(request.post.touches),
                pose_classification: request.post.pose_classification,
            },
            request.post.rallies,
            &primary,
            &stabilized,
            &mut |done, total| {
                progress_with_frames(
                    path,
                    status,
                    "Постобработка · касания в первичных розыгрышах",
                    91.0 + 4.0 * done as f64 / total.max(1) as f64,
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
        "Касания",
        "success",
        &format!("Касания сохранены: touch_id={touch_id}"),
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
    progress(
        path,
        status,
        "Постобработка · автоматическая привязка игроков",
        98.0,
    )?;
    let replay_sha = replay["sha256"]
        .as_str()
        .ok_or("У треков отсутствует SHA-256")?;
    let identity_result = crate::native_player_tracks::reid_review(
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
        }),
    )
    .await;
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
fn cancelled_outcome(outcome: Result<(), String>, cancelled: bool) -> Result<(), String> {
    if cancelled {
        Err("cancelled".into())
    } else {
        outcome
    }
}

async fn perform(
    state: Arc<AppState>,
    request: StartRequest,
    id: String,
    cancel: Arc<AtomicBool>,
    path: PathBuf,
    mut status: Value,
) {
    let _ = log_event(
        &path,
        request.mode.label(),
        "info",
        "Запуск создан, настройки и выбранный источник сохранены.",
    );
    let mut outcome = pipeline(&state, &request, &cancel, &path, &mut status).await;
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
    let mut slot = manager().lock().await;
    outcome = cancelled_outcome(outcome, cancel.load(Ordering::Acquire));
    match outcome {
        Ok(()) => {
            status["status"] = json!("complete");
            status["stage"] = json!(match request.mode {
                PipelineMode::Full => "GPU и CPU завершены",
                PipelineMode::Gpu => "Первичный GPU-анализ завершён",
                PipelineMode::Post => "Новая CPU-постобработка завершена",
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
                        PipelineMode::Gpu => "Новый GPU-анализ и слияние сохранены",
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
    let _ = save(&path, &status);
    if slot.as_ref().is_some_and(|current| current.id == id) {
        *slot = None;
    }
}

pub async fn start(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut request): Json<StartRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
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
                        || previous["video"] != request.video
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
                        "status":previous["status"],"recovered":true})));
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
    let root = folder(&state, &request.project_id, true)?;
    let id = Uuid::new_v4().simple().to_string();
    let dir = root.join(&id);
    fs::create_dir(&dir).map_err(|e| storage::internal(format!("Создание запуска: {e}")))?;
    let path = dir.join("status.json");
    let status = json!({
        "schema":SCHEMA,"id":id,"project_id":request.project_id,"video":request.video,
        "status":"queued","stage":"В очереди","progress_percent":0.0,
        "started_ms":epoch_ms(),"finished_ms":null,"stages":[],"stage_samples":[],
        "profile":{"gpu":request.gpu,"post":request.post},
        "mode":request.mode,"source_analysis_id":request.source_analysis_id,
        "client_request_id":request.client_request_id,
        "analysis_range":request.analysis_range,
        "gpu_model_stack":requested_gpu_model_stack(),
        "zero_copy_request":zero_copy_requested(),"zero_copy_active":false,
        "gpu_jobs":{},"cpu_runs":{},"analysis_id":null,
        "gpu_rally_mask":null,"gpu_rally_mask_sha256":null,
        "postprocessing_id":null,"current_gpu_job_id":null,
        "source_identity":null,"gpu_reused":false,"result_reused":false,
        "reused_from_id":null,"pipeline_complete":false,"error":null
    });
    save(&path, &status).map_err(storage::internal)?;
    log_native(
        &path,
        "Запуск",
        "studio_pipeline::start",
        json!({
            "project_id":request.project_id,"video":request.video,
            "gpu":request.gpu,"post":request.post,"mode":request.mode,
            "source_analysis_id":request.source_analysis_id,
            "client_request_id":request.client_request_id
        }),
    )
    .map_err(storage::internal)?;
    save(&root.join("latest.json"), &json!({"id":id})).map_err(storage::internal)?;
    let cancelled = Arc::new(AtomicBool::new(false));
    *slot = Some(Running {
        project_id: request.project_id.clone(),
        id: id.clone(),
        cancel: cancelled.clone(),
    });
    drop(_guard);
    tokio::spawn(perform(state, request, id.clone(), cancelled, path, status));
    Ok(Json(json!({"schema":SCHEMA,"id":id,"status":"queued"})))
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
    } else if name == "GPU · RF-DETR Medium + RTMPose" || name == "GPU · RF-DETR Medium" {
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

fn status_snapshot(
    state: &AppState,
    query: &StatusQuery,
) -> Result<Option<(String, Value)>, ApiError> {
    let root = folder(state, &query.project_id, false)?;
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
    if result["status"] == "running" || result["status"] == "queued" {
        let slot = manager().lock().await;
        match slot
            .as_ref()
            .filter(|entry| entry.id == id && entry.project_id == project_id)
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
        .filter(|entry| entry.project_id == request.project_id && entry.id == request.id)
        .ok_or_else(|| storage::not_found("Нет активного запуска для отмены"))?;
    running.cancel.store(true, Ordering::Release);
    drop(slot);
    // Promptly signal the active GPU job; never wait for the pipeline's next
    // poll cycle before forwarding a user cancellation.
    let path = folder(&_state, &request.project_id, false)?
        .join(&request.id)
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
        assert_eq!(
            pipeline_stage_id("GPU · VBallNet: траектория мяча и розыгрыши"),
            "ball"
        );
        assert_eq!(
            pipeline_stage_id("Интерполяция траектории мяча · Ball Stabilizer"),
            "ball_stabilizer"
        );
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

    #[tokio::test]
    async fn start_queues_before_source_validation_and_records_validation_failure() {
        let base = std::env::temp_dir().join(format!("studio-queue-{}", Uuid::new_v4()));
        let project_id = "abc123def456";
        fs::create_dir_all(base.join("projects").join(project_id)).unwrap();
        storage::write_json_atomic(
            &base.join("projects/projects.json"),
            &json!({
                "projects":[{"id":project_id,"videos":["missing.mp4"]}]
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
        let request = || {
            serde_json::from_value::<StartRequest>(json!({
                "project_id":project_id,"video":"missing.mp4","mode":"gpu",
                "client_request_id":"12345678901234567890123456789012"
            }))
            .unwrap()
        };
        let Json(queued) = start(State(state.clone()), HeaderMap::new(), Json(request()))
            .await
            .unwrap();
        assert_eq!(queued["status"], "queued");
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
        let mut cancellation_request = request();
        cancellation_request.client_request_id = None;
        let Json(new_job) = start(
            State(state.clone()),
            HeaderMap::new(),
            Json(cancellation_request),
        )
        .await
        .unwrap();
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
        assert_eq!(fresh.post.touches.algorithm, TouchAlgorithm::PoseMotion);
        assert!(fresh.post.touches.require_pose);

        let mut partial = base.clone();
        partial["post"] = json!({"touches":{"threshold":0.62}});
        let active: StartRequest = serde_json::from_value(partial).unwrap();
        assert_eq!(active.post.touches.algorithm, TouchAlgorithm::PoseMotion);
        assert_eq!(active.post.touches.threshold, 0.62);
        assert!(active.post.touches.require_pose);
        let ignored_optional: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4",
            "post":{"touches":{"require_pose":false}}
        }))
        .unwrap();
        assert!(ignored_optional.post.touches.require_pose);
        let tcn: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4",
            "post":{"touches":{"algorithm":"trajectory_tcn"}}
        }))
        .unwrap();
        assert_eq!(tcn.post.touches.algorithm, TouchAlgorithm::TrajectoryTcn);
        assert!(tcn.post.touches.require_pose);

        let mut previous_choice = base;
        previous_choice["post"] = json!({"touches":{"algorithm":"legacy"}});
        assert!(serde_json::from_value::<StartRequest>(previous_choice).is_err());

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
        let saved = json!({
            "schema":SCHEMA,
            "gpu_model_stack":requested_gpu_model_stack(),
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
        for invalid in [
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
    fn shared_player_pose_stride_changes_gpu_cache_identity() {
        let identity = json!({"bytes":1234,"mtime_ns":5678});
        let mut request: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4"
        }))
        .unwrap();
        let saved = json!({
            "schema":SCHEMA,
            "gpu_model_stack":requested_gpu_model_stack(),
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
            "gpu_model_stack":requested_gpu_model_stack(),
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
        assert_eq!(profile.gpu.cpu_threads, 0);
        assert_eq!(profile.gpu.gpu_memory_limit_mb, 0);
        let manual: StartRequest = serde_json::from_value(json!({
            "project_id":"abc123def456","video":"match.mp4",
            "gpu":{"cpu_threads":12,"gpu_memory_limit_mb":4096}
        }))
        .unwrap();
        assert_eq!(manual.gpu.cpu_threads, 12);
        assert_eq!(manual.gpu.gpu_memory_limit_mb, 4096);
        assert_eq!(profile.post.auto_assign_threshold_percent, 75);
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
        assert_eq!(profile.post.touches.algorithm, TouchAlgorithm::PoseMotion);
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
                "automatic_action_prediction_method":"pose_keypoints_v1_or_unknown_touch",
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
                "ground_boundary_version":"not_computed_primary_gpu_v1",
                "materialization_version":"primary_gpu_intervals_direct_v1",
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
            "gpu_model_stack":requested_gpu_model_stack(),
            "profile":{"gpu":request.gpu,"post":request.post},
            "gpu_jobs":{"ball":gpu_ball,"players":gpu_players},
            "gpu_rally_mask":{
                "schema":"volley-primary-rally-mask/1",
                "min_trajectory_seconds":request.gpu.primary_rallies.min_trajectory_seconds,
                "merge_gap_seconds":request.gpu.primary_rallies.merge_gap_seconds,
                "stationary_ball_ms":request.gpu.primary_rallies.stationary_ball_ms,
                "movement_filter_version":"stationary_ball_min_rally_5s_v3",
                "min_rally_seconds":5.0
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
        let matching = find_reusable(&state, &request, &identity, &"2".repeat(32));
        assert!(
            matching.full.is_some(),
            "unchanged settings must reuse both versions"
        );
        assert!(matching.gpu.is_some());
        assert!(compatible_version(&saved, &request, &identity));

        let mut failed_pose = saved.clone();
        failed_pose["status"] = json!("failed");
        failed_pose["pipeline_complete"] = json!(false);
        failed_pose["profile"]["gpu"]["pose_enabled"] = json!(true);
        storage::write_json_atomic(&directory.join("status.json"), &failed_pose).unwrap();
        request.gpu.pose_enabled = true;
        request.mode = PipelineMode::Gpu;
        let recovered = find_reusable(&state, &request, &identity, &"2".repeat(32));
        assert!(recovered.full.is_none());
        assert!(
            recovered.gpu.is_some(),
            "failed RTMPose must reuse valid raw GPU evidence"
        );
        failed_pose["gpu_jobs"]["pose"] = json!("9".repeat(32));
        storage::write_json_atomic(&directory.join("status.json"), &failed_pose).unwrap();
        assert!(find_reusable(&state, &request, &identity, &"2".repeat(32))
            .gpu
            .is_none());
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
        assert!(find_reusable(&state, &request, &identity, &"2".repeat(32))
            .full
            .is_some());
        assert!(selected_gpu(&state, &request, &identity, &analysis).is_ok());
        for (players, pose) in [(false, true), (true, false), (false, false)] {
            request.post.interpolate_players = players;
            request.post.interpolate_pose = pose;
            let changed = find_reusable(&state, &request, &identity, &"2".repeat(32));
            assert!(changed.full.is_none());
            assert!(changed.gpu.is_some());
            assert!(!cpu_lineage_valid(&project_dir, &request, &saved, None));
        }
        request.post.interpolate_players = true;
        request.post.interpolate_pose = true;
        request.post.ball_stabilizer.max_gap_frames = 8;
        let changed = find_reusable(&state, &request, &identity, &"2".repeat(32));
        assert!(
            changed.full.is_none(),
            "changed Ball Stabilizer must invalidate CPU results"
        );
        assert!(
            changed.gpu.is_some(),
            "changed Ball Stabilizer must reuse VBallNet GPU"
        );
        assert!(!cpu_lineage_valid(&project_dir, &request, &saved, None));
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
            ("body_trajectory_intersection_required", json!(false)),
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
            let reusable = find_reusable(&state, &request, &identity, &"2".repeat(32));
            assert!(reusable.full.is_none());
            assert!(reusable.gpu.is_some());
        }
        storage::write_json_atomic(&touch_manifest, &original_touch).unwrap();
        assert!(cpu_lineage_valid(&project_dir, &request, &saved, None));
        request.post.touches.algorithm = TouchAlgorithm::PoseMotion;
        let obsolete_pose = find_reusable(&state, &request, &identity, &"2".repeat(32));
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
        let changed_cpu = find_reusable(&state, &request, &identity, &"2".repeat(32));
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
        let changed_gpu = find_reusable(&state, &request, &identity, &"2".repeat(32));
        assert!(changed_gpu.full.is_none() && changed_gpu.gpu.is_none());
        assert!(!compatible_version(&saved, &request, &identity));
        request.gpu.stride = 9;
        let other_source = json!({"bytes":2048,"mtime_ns":123456790u64});
        let replaced_source = find_reusable(&state, &request, &other_source, &"2".repeat(32));
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
        assert_eq!(history["runs"][0]["stages"][0]["duration_ms"], 2000);
        assert_eq!(history["runs"][0]["stages"][1]["duration_ms"], 1500);
        // CPU can use a standalone GPU fusion with no Studio pipeline record.
        fs::remove_file(directory.join("status.json")).unwrap();
        let source = selected_gpu(&handler_state, &request, &identity, &analysis).unwrap();
        assert_eq!(source["gpu_jobs"]["ball"], gpu_ball);
        assert_eq!(source["gpu_jobs"]["players"], gpu_players);
        assert_eq!(source["profile"]["gpu"]["stride"], 9);
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
