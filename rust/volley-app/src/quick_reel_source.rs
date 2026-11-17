use crate::{
    storage::{self, ApiError},
    AppState,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};
use uuid::Uuid;
use volley_core::contracts::job_id as is_id;
use volley_media::preparation::PreparationPlan;

const MAX_VIDEO_SIZE: u64 = 500 * 1024 * 1024 * 1024;
pub(crate) const QUICK_ROOT: &str = "quick_reels";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Source {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) source: String,
    pub(crate) size: u64,
    pub(crate) mtime_ns: u128,
    #[serde(default)]
    pub(crate) display_name: Option<String>,
}
fn safe_source(path: &str) -> Result<(PathBuf, fs::Metadata), ApiError> {
    if path.is_empty() || path.len() > 4096 || path.contains('\0') {
        return Err(storage::bad_request(
            "Выберите реальный локальный видеофайл",
        ));
    }
    let initial = Path::new(path);
    if !initial.is_absolute() {
        return Err(storage::bad_request(
            "Ожидается абсолютный путь, выбранный проводником",
        ));
    }
    let canonical = initial
        .canonicalize()
        .map_err(|_| storage::bad_request("Выбранный видеофайл не найден"))?;
    if !canonical.is_file() {
        return Err(storage::bad_request(
            "Указанный путь не является видеофайлом",
        ));
    }
    let name = canonical
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| storage::bad_request("Некорректное имя видео"))?;
    storage::video_name(name)?;
    let meta = fs::metadata(&canonical)
        .map_err(|_| storage::bad_request("Нет доступа к метаданным видео"))?;
    if meta.len() == 0 || meta.len() > MAX_VIDEO_SIZE {
        return Err(storage::bad_request(
            "Размер видео должен быть от 1 байта до 500 ГиБ",
        ));
    }
    Ok((canonical, meta))
}
pub(crate) fn mtime(meta: &fs::Metadata) -> Result<u128, ApiError> {
    meta.modified()
        .ok()
        .and_then(|v| v.duration_since(UNIX_EPOCH).ok())
        .map(|v| v.as_nanos())
        .ok_or_else(|| storage::bad_request("Не удалось прочитать время изменения файла"))
}
pub(crate) fn source_for(path: &str) -> Result<Source, ApiError> {
    let (canonical, meta) = safe_source(path)?;
    let probe = volley_media::probe(&canonical).map_err(storage::bad_request)?;
    if probe.duration_sec < 0.1 || !probe.duration_sec.is_finite() {
        return Err(storage::bad_request(
            "Видео не содержит воспроизводимых кадров",
        ));
    }
    Ok(Source {
        id: Uuid::new_v4().simple().to_string(),
        name: canonical
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
        source: canonical.to_string_lossy().into_owned(),
        size: meta.len(),
        mtime_ns: mtime(&meta)?,
        display_name: None,
    })
}
pub(crate) fn verify_source(item: &Source) -> Result<PathBuf, ApiError> {
    if !is_id(&item.id) {
        return Err(storage::bad_request("Неправильный ID выбранного видео"));
    }
    let (canonical, meta) = safe_source(&item.source)?;
    if canonical.to_string_lossy() != item.source
        || meta.len() != item.size
        || mtime(&meta)? != item.mtime_ns
        || canonical.file_name().and_then(|s| s.to_str()) != Some(item.name.as_str())
    {
        return Err(storage::conflict("Исходное видео изменилось после выбора"));
    }
    Ok(canonical)
}
pub(crate) fn prepared_plan_for_quick_reel(
    state: &AppState,
    project: &Value,
    video: &str,
) -> Result<PreparationPlan, ApiError> {
    storage::video_name(video)?;
    let scoped = state
        .data_root
        .canonicalize()
        .map_err(|_| storage::bad_request("Quick Reel pipeline root missing"))?;
    if scoped.file_name().and_then(|s| s.to_str()) != Some("pipeline") {
        return Err(storage::bad_request(
            "Not an internal Quick Reel pipeline namespace",
        ));
    }
    let run_dir = scoped
        .parent()
        .ok_or_else(|| storage::bad_request("Quick Reel has no run root"))?;
    let run_id = run_dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let quick_root = run_dir
        .parent()
        .ok_or_else(|| storage::bad_request("Quick Reel root missing"))?;
    if !is_id(run_id)
        || quick_root.file_name().and_then(|s| s.to_str()) != Some(QUICK_ROOT)
        || project["quick_reel_run_id"].as_str() != Some(run_id)
    {
        return Err(storage::bad_request(
            "Quick Reel virtual project does not belong to this run",
        ));
    }
    let global_root = quick_root
        .parent()
        .ok_or_else(|| storage::bad_request("Project storage missing"))?;
    let original = project["source_paths"][video]
        .as_str()
        .ok_or_else(|| storage::bad_request("Missing original Quick Reel source mapping"))?;
    let expected_id = project["source_ids"][video]
        .as_str()
        .ok_or_else(|| storage::bad_request("Missing Quick Reel source token"))?;
    let request = storage::read_json(&run_dir.join("request.json"))?
        .ok_or_else(|| storage::bad_request("Missing original Quick Reel request"))?;
    if request["id"].as_str() != Some(run_id)
        || request["inference_cache_reuse"] != false
        || request["schema"] != "volley-rust-quick-reel/1"
    {
        return Err(storage::conflict("Quick Reel request identity changed"));
    }
    let saved = request["videos"]
        .as_array()
        .and_then(|rows| {
            rows.iter().find(|row| {
                row["id"].as_str() == Some(expected_id) && row["source"].as_str() == Some(original)
            })
        })
        .ok_or_else(|| storage::conflict("Quick Reel input not in immutable run request"))?;
    let source: Source = serde_json::from_value(saved.clone())
        .map_err(|_| storage::bad_request("Damaged Quick Reel source identity"))?;
    let canonical = verify_source(&source)?;
    let plan = PreparationPlan::new(&canonical, global_root, None)
        .and_then(PreparationPlan::bind_source)
        .map(PreparationPlan::read_original)
        .map_err(storage::bad_request)?;
    if plan.ready_path().is_none() {
        return Err(storage::conflict(
            "Quick Reel prepared media not ready; no inference reuse allowed",
        ));
    }
    Ok(plan)
}
