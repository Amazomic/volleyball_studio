//! Human-reviewed temporal labels, independent of immutable GPU/CPU evidence.
//! Compare predictions BEFORE any manual correction to avoid inflated scores.
use crate::{
    native_rally_evidence::{generate, inspect_project, Choice, ValidatedRun},
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
    fs,
    sync::{Arc, Mutex, OnceLock},
};
use uuid::Uuid;

const SCHEMA: &str = "volley-touch-annotations/1";
const MAX_BYTES: u64 = 32 * 1024 * 1024;
const MAX_VIDEOS: usize = 256;
const MAX_TOUCHES: usize = 4096;
const MAX_RALLIES: usize = 1024;
static LABEL_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Mark {
    id: String,
    frame: u64,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RallyMark {
    id: String,
    start: u64,
    end: u64,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VideoLabels {
    project_id: String,
    video: String,
    source_sha256: String,
    rally_run_id: String,
    total_frames: u64,
    revision: u64,
    reviewed: bool,
    #[serde(default)]
    quality: Option<Accuracy>,
    #[serde(default)]
    baseline_touches: Vec<u64>,
    #[serde(default)]
    baseline_rallies: Vec<[u64; 2]>,
    #[serde(default)]
    baseline_touch_sha: String,
    #[serde(default)]
    baseline_rally_sha: String,
    #[serde(default)]
    trajectory_replay_sha: String,
    #[serde(default)]
    trajectory_candidates: Vec<trajectory::TrajectoryCandidate>,
    touches: Vec<Mark>,
    rallies: Vec<RallyMark>,
}
#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Library {
    schema: String,
    videos: Vec<VideoLabels>,
}
fn file(state: &AppState) -> std::path::PathBuf {
    state
        .data_root
        .join("training")
        .join("touch_annotations.json")
}
fn lock<T>(f: impl FnOnce() -> Result<T, ApiError>) -> Result<T, ApiError> {
    let _guard = LABEL_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| storage::internal("Touch annotation lock poisoned"))?;
    f()
}
fn valid(v: &VideoLabels) -> bool {
    v.source_sha256.len() == 64
        && v.source_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        && !v.project_id.is_empty()
        && !v.video.is_empty()
        && v.total_frames > 0
        && v.total_frames <= 48 * 3600 * 120
        && v.touches.len() <= MAX_TOUCHES
        && v.rallies.len() <= MAX_RALLIES
        && v.trajectory_candidates.len() <= 32768
        && v.trajectory_candidates.iter().all(|c| {
            c.frame < v.total_frames && c.score.is_finite() && (0.0..=10000.0).contains(&c.score)
        })
        && v.touches
            .iter()
            .all(|t| Uuid::parse_str(&t.id).is_ok() && t.frame < v.total_frames)
        && v.rallies
            .iter()
            .all(|r| Uuid::parse_str(&r.id).is_ok() && r.start <= r.end && r.end < v.total_frames)
}
fn read(state: &AppState) -> Result<Library, ApiError> {
    let p = file(state);
    match fs::symlink_metadata(&p) {
        Ok(m) if !m.is_file() || m.file_type().is_symlink() || m.len() > MAX_BYTES => {
            return Err(storage::bad_request(
                "Unsafe or oversized Touch annotations",
            ))
        }
        Ok(_) => (),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Library {
                schema: SCHEMA.into(),
                videos: vec![],
            })
        }
        Err(e) => return Err(storage::internal(format!("Touch annotation metadata: {e}"))),
    }
    let b = fs::read(&p).map_err(|e| storage::internal(format!("Read Touch annotations: {e}")))?;
    let library: Library = serde_json::from_slice(&b)
        .map_err(|_| storage::bad_request("Malformed Touch annotations"))?;
    if library.schema != SCHEMA
        || library.videos.len() > MAX_VIDEOS
        || library.videos.iter().any(|v| !valid(v))
    {
        return Err(storage::bad_request("Invalid Touch annotation library"));
    }
    Ok(library)
}
fn save(state: &AppState, library: &Library) -> Result<(), ApiError> {
    storage::write_json_atomic(&file(state), &json!(library))
}
fn source_sha(run: &ValidatedRun) -> Result<String, ApiError> {
    let evidence = volley_media::preparation::source_provenance(&run.project_source)
        .map_err(storage::bad_request)?;
    let bytes = serde_json::to_vec(&evidence)
        .map_err(|e| storage::internal(format!("Source provenance: {e}")))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
type AutoTimeline = (Vec<u64>, Vec<(u64, u64)>);

fn auto(run: &ValidatedRun) -> Result<AutoTimeline, ApiError> {
    let v = generate(run, Choice::Timeline)
        .map_err(storage::bad_request)?
        .value;
    let mut touches = v["touch_annotations"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|e| e["touch_attributed"] == true || e["contact_verified"] == true)
        .filter_map(|e| e["frame_index"].as_u64())
        .collect::<Vec<_>>();
    touches.sort_unstable();
    touches.dedup();
    let rallies = v["rallies"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|e| Some((e["start_frame"].as_u64()?, e["end_frame"].as_u64()?)))
        .collect();
    Ok((touches, rallies))
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MatchScore {
    hits: usize,
    distance: u64,
    max_distance: u64,
}
fn better(a: MatchScore, b: MatchScore) -> bool {
    a.hits > b.hits || a.hits == b.hits && a.distance < b.distance
}
#[derive(Clone, Serialize, Deserialize, Default)]
struct Accuracy {
    matched: usize,
    extra: usize,
    missed: usize,
    mean_error_ms: f64,
    max_error_ms: f64,
    precision: f64,
    recall: f64,
    f1: f64,
}
fn accuracy(expected: &[u64], actual: &[u64], tolerance: u64, fps: f64) -> Accuracy {
    // Order-preserving maximum-cardinality matching with minimum frame error.
    // No prediction or label can be matched more than once.
    let mut prev = vec![
        MatchScore {
            hits: 0,
            distance: 0,
            max_distance: 0
        };
        actual.len() + 1
    ];
    for &reference in expected {
        let mut next = vec![prev[0]; actual.len() + 1];
        for (j, &prediction) in actual.iter().enumerate() {
            let mut best = if better(prev[j + 1], next[j]) {
                prev[j + 1]
            } else {
                next[j]
            };
            let delta = reference.abs_diff(prediction);
            if delta <= tolerance {
                let from = prev[j];
                let pair = MatchScore {
                    hits: from.hits + 1,
                    distance: from.distance + delta,
                    max_distance: from.max_distance.max(delta),
                };
                if better(pair, best) {
                    best = pair;
                }
            }
            next[j + 1] = best;
        }
        prev = next;
    }
    let score = prev[actual.len()];
    let matched = score.hits;
    let extra = actual.len().saturating_sub(matched);
    let missed = expected.len().saturating_sub(matched);
    let precision = if actual.is_empty() {
        if expected.is_empty() {
            1.0
        } else {
            0.0
        }
    } else {
        matched as f64 / actual.len() as f64
    };
    let recall = if expected.is_empty() {
        if actual.is_empty() {
            1.0
        } else {
            0.0
        }
    } else {
        matched as f64 / expected.len() as f64
    };
    Accuracy {
        matched,
        extra,
        missed,
        mean_error_ms: if matched == 0 {
            0.0
        } else {
            score.distance as f64 * 1000.0 / (fps * matched as f64)
        },
        max_error_ms: score.max_distance as f64 * 1000.0 / fps,
        precision,
        recall,
        f1: if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        },
    }
}
fn outcome(labels: &VideoLabels, run: &ValidatedRun) -> Result<Value, ApiError> {
    let (predicted, rally_preds) = if labels.rally_run_id == run.run_id
        && labels.baseline_touch_sha == run.touch_sha
        && labels.baseline_rally_sha == run.rally_sha
    {
        (
            labels.baseline_touches.clone(),
            labels
                .baseline_rallies
                .iter()
                .map(|row| (row[0], row[1]))
                .collect::<Vec<_>>(),
        )
    } else {
        auto(run)?
    };
    let mut truth = labels.touches.iter().map(|m| m.frame).collect::<Vec<_>>();
    truth.sort_unstable();
    let score = accuracy(&truth, &predicted, 1, 30.0);
    let diagnostic = accuracy(&truth, &predicted, 3, 30.0);
    let mut boundary_differences = Vec::new();
    for a in &labels.rallies {
        if let Some((start, end)) = rally_preds
            .iter()
            .filter(|(start, end)| a.start.abs_diff(*start) <= 90 || a.end.abs_diff(*end) <= 90)
            .min_by_key(|(start, end)| a.start.abs_diff(*start) + a.end.abs_diff(*end))
        {
            boundary_differences.push(a.start.abs_diff(*start));
            boundary_differences.push(a.end.abs_diff(*end));
        }
    }
    let boundary_mean_ms = if boundary_differences.is_empty() {
        Value::Null
    } else {
        json!(
            boundary_differences.iter().copied().sum::<u64>() as f64 * 1000.0
                / (30.0 * boundary_differences.len() as f64)
        )
    };
    Ok(
        json!({"touches":score,"rallies":{"labeled":labels.rallies.len(),
        "predicted":rally_preds.len(),"boundary_mean_error_ms":boundary_mean_ms},
        "tolerance_frames":1,"tolerance_ms":33.33,"diagnostic_100ms":diagnostic,"reviewed":labels.reviewed}),
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryLabels {
    project_id: String,
    rally_run_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogQuery {
    project_id: String,
}
fn catalog_value(library: &Library, project_id: &str) -> Value {
    let videos: Vec<Value> = library
        .videos
        .iter()
        .filter(|v| v.project_id == project_id && v.reviewed)
        .map(|v| {
            json!({
                "video":v.video,
                "rally_run_id":v.rally_run_id,
                "rallies":v.rallies.len(),
                "touches":v.touches.len()
            })
        })
        .collect();
    json!({"schema":"volley-touch-annotation-catalog/1","project_id":project_id,"videos":videos})
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditLabels {
    project_id: String,
    rally_run_id: String,
    action: String,
    #[serde(default)]
    revision: Option<u64>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    frame: Option<u64>,
    #[serde(default)]
    start: Option<u64>,
    #[serde(default)]
    end: Option<u64>,
    #[serde(default)]
    touches: Option<Vec<Mark>>,
    #[serde(default)]
    rallies: Option<Vec<RallyMark>>,
}
pub(crate) mod trajectory;
pub(crate) mod types;
use trajectory::{scan_trajectory, trajectory_profile};
fn aggregate(lib: &Library) -> Value {
    // Only explicitly reviewed videos contribute to the global benchmark.
    // Metrics are snapshots against each video's saved comparison CPU run.
    let mut hits = 0usize;
    let mut extra = 0usize;
    let mut missed = 0usize;
    let mut weighted_error = 0.0f64;
    let mut max_error = 0.0f64;
    let mut rows = Vec::new();
    for v in &lib.videos {
        if !v.reviewed {
            continue;
        }
        let Some(score) = &v.quality else {
            continue;
        };
        hits += score.matched;
        extra += score.extra;
        missed += score.missed;
        weighted_error += score.mean_error_ms * score.matched as f64;
        max_error = max_error.max(score.max_error_ms);
        rows.push(json!({"project_id":v.project_id,"video":v.video,
            "rally_run_id":v.rally_run_id,"metrics":score,"revision":v.revision}));
    }
    let precision = if hits + extra == 0 {
        if missed == 0 {
            1.0
        } else {
            0.0
        }
    } else {
        hits as f64 / (hits + extra) as f64
    };
    let recall = if hits + missed == 0 {
        if extra == 0 {
            1.0
        } else {
            0.0
        }
    } else {
        hits as f64 / (hits + missed) as f64
    };
    json!({"reviewed_videos":rows.len(),"matched":hits,"extra":extra,"missed":missed,
        "mean_error_ms":if hits==0{0.0}else{weighted_error/hits as f64},
        "max_error_ms":max_error,"precision":precision,"recall":recall,
        "f1":if precision+recall==0.0{0.0}else{2.0*precision*recall/(precision+recall)},
        "videos":rows,"calculation":"weighted_snapshot_of_reviewed_videos"})
}
fn response(state: &AppState, run: &ValidatedRun, source: &str) -> Result<Value, ApiError> {
    lock(|| {
        let lib = read(state)?;
        let selected = lib
            .videos
            .iter()
            .find(|v| v.project_id == run.project_id && v.video == run.video);
        if selected.is_some_and(|v| v.source_sha256 != source) {
            return Err(storage::conflict(
                "Source video changed: old labels cannot be used",
            ));
        }
        let reviewed_videos = lib.videos.iter().filter(|v| v.reviewed).count();
        let labels = selected.cloned();
        let metrics = labels.as_ref().map(|v| outcome(v, run)).transpose()?;
        Ok(
            json!({"schema":SCHEMA,"project_id":run.project_id,"video":run.video,
            "rally_run_id":run.run_id,"total_frames":run.frames,"fps":30,
            "revision":labels.as_ref().map_or(0,|v|v.revision),
            "reviewed":labels.as_ref().is_some_and(|v|v.reviewed),
            "annotated_videos":lib.videos.len(),"reviewed_videos":reviewed_videos,
            "overall":aggregate(&lib),
            "touches":labels.as_ref().map(|v|&v.touches),"rallies":labels.as_ref().map(|v|&v.rallies),
            "metrics":metrics,"trajectory_profile":trajectory_profile(&lib),
            "trajectory_cached":labels.as_ref().is_some_and(|v|v.trajectory_replay_sha==run.replay_sha),
            "model_retrained":false}),
        )
    })
}
pub(crate) fn compare_touch_frames(reference: &[u64], predicted: &[u64]) -> Value {
    let mut reference = reference.to_vec();
    let mut predicted = predicted.to_vec();
    reference.sort_unstable();
    predicted.sort_unstable();
    json!(accuracy(&reference, &predicted, 3, 30.0))
}

type ReviewedComparison = (Vec<(u64, u64)>, Vec<u64>, bool);

pub(crate) fn comparison_truth(
    state: &AppState,
    run: &ValidatedRun,
) -> Result<Option<ReviewedComparison>, ApiError> {
    let source = source_sha(run)?;
    lock(|| {
        let library = read(state)?;
        let selected = library
            .videos
            .iter()
            .find(|v| v.project_id == run.project_id && v.video == run.video);
        if selected.is_some_and(|v| v.source_sha256 != source) {
            return Err(storage::conflict(
                "Source video changed: old labels cannot be used",
            ));
        }
        Ok(selected.map(|v| {
            (
                v.rallies.iter().map(|r| (r.start, r.end)).collect(),
                v.touches.iter().map(|t| t.frame).collect(),
                v.reviewed,
            )
        }))
    })
}

pub async fn catalog(
    State(state): State<Arc<AppState>>,
    Query(q): Query<CatalogQuery>,
) -> ApiResult<Value> {
    if q.project_id.trim().is_empty() {
        return Err(storage::bad_request("Project ID is required"));
    }
    let result = tokio::task::spawn_blocking(move || {
        lock(|| {
            let library = read(&state)?;
            Ok(catalog_value(&library, &q.project_id))
        })
    })
    .await
    .map_err(|e| storage::internal(format!("Annotation catalog task: {e}")))??;
    Ok(Json(result))
}
pub async fn list(
    State(state): State<Arc<AppState>>,
    Query(q): Query<QueryLabels>,
) -> ApiResult<Value> {
    let result = tokio::task::spawn_blocking(move || {
        let run = inspect_project(&state, &q.project_id, &q.rally_run_id)?;
        let source = source_sha(&run)?;
        response(&state, &run, &source)
    })
    .await
    .map_err(|e| storage::internal(format!("Annotation read task: {e}")))??;
    Ok(Json(result))
}
pub async fn edit(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<EditLabels>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    if !state.writes_enabled {
        return Err(storage::conflict("Touch labels are read-only"));
    }
    let result = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        let run = inspect_project(&state, &req.project_id, &req.rally_run_id)?;
        let source = source_sha(&run)?;
        // Hash and decode saved predictions only once per selected CPU version.
        // Manual edits then operate on the small, source/SHA-bound baseline.
        let stale_baseline = lock(|| {
            let library = read(&state)?;
            Ok(library
                .videos
                .iter()
                .find(|v| v.project_id == run.project_id && v.video == run.video)
                .is_some_and(|v| {
                    v.rally_run_id != run.run_id
                        || v.baseline_touch_sha != run.touch_sha
                        || v.baseline_rally_sha != run.rally_sha
                }))
        })?;
        let scanned = if req.action == "save" {
            let cached = lock(|| {
                let lib = read(&state)?;
                Ok(lib
                    .videos
                    .iter()
                    .find(|v| {
                        v.project_id == run.project_id
                            && v.video == run.video
                            && v.trajectory_replay_sha == run.replay_sha
                    })
                    .map(|v| v.trajectory_candidates.clone()))
            })?;
            Some(match cached {
                Some(events) => events,
                None => scan_trajectory(&run)?,
            })
        } else {
            None
        };
        let predictions = if req.action == "seed" || req.action == "save" || stale_baseline {
            Some(auto(&run)?)
        } else {
            None
        };
        lock(|| {
            let mut lib = read(&state)?;
            let position = lib
                .videos
                .iter()
                .position(|v| v.project_id == run.project_id && v.video == run.video);
            if let Some(i) = position {
                if lib.videos[i].source_sha256 != source {
                    return Err(storage::conflict("Source changed since annotation"));
                }
                if req.revision != Some(lib.videos[i].revision) {
                    return Err(storage::conflict(
                        "Annotations changed. Refresh before editing",
                    ));
                }
            } else if req.revision != Some(0) {
                return Err(storage::conflict("Annotations revision mismatch"));
            }
            if position.is_none() {
                if req.action != "seed" && req.action != "save" {
                    return Err(storage::bad_request(
                        "Start annotation from current CPU marks",
                    ));
                }
                if lib.videos.len() >= MAX_VIDEOS {
                    return Err(storage::bad_request("Annotation library full"));
                }
                lib.videos.push(VideoLabels {
                    project_id: run.project_id.clone(),
                    video: run.video.clone(),
                    source_sha256: source.clone(),
                    rally_run_id: run.run_id.clone(),
                    total_frames: run.frames,
                    revision: 0,
                    reviewed: false,
                    quality: None,
                    baseline_touches: vec![],
                    baseline_rallies: vec![],
                    baseline_touch_sha: String::new(),
                    baseline_rally_sha: String::new(),
                    trajectory_replay_sha: String::new(),
                    trajectory_candidates: Vec::new(),
                    touches: vec![],
                    rallies: vec![],
                });
            }
            let i = position.unwrap_or(lib.videos.len() - 1);
            let v = &mut lib.videos[i];
            if let Some((touches, rallies)) = &predictions {
                v.baseline_touches = touches.clone();
                v.baseline_rallies = rallies.iter().map(|(a, b)| [*a, *b]).collect();
                v.baseline_touch_sha = run.touch_sha.clone();
                v.baseline_rally_sha = run.rally_sha.clone();
            }
            if v.total_frames != run.frames {
                return Err(storage::conflict("Video frame count changed"));
            }
            let valid_frame = |frame: Option<u64>| -> Result<u64, ApiError> {
                frame
                    .filter(|&n| n < run.frames)
                    .ok_or_else(|| storage::bad_request("Frame outside source video"))
            };
            match req.action.as_str() {
                "save" => {
                    let touches = req
                        .touches
                        .as_ref()
                        .ok_or_else(|| storage::bad_request("Missing Touch labels"))?;
                    let rallies = req
                        .rallies
                        .as_ref()
                        .ok_or_else(|| storage::bad_request("Missing Rally labels"))?;
                    if touches.len() > MAX_TOUCHES
                        || rallies.len() > MAX_RALLIES
                        || touches
                            .iter()
                            .any(|t| t.frame >= run.frames || Uuid::parse_str(&t.id).is_err())
                        || rallies.iter().any(|r| {
                            r.start >= r.end
                                || r.end >= run.frames
                                || Uuid::parse_str(&r.id).is_err()
                        })
                    {
                        return Err(storage::bad_request("Invalid saved Touch/Rally labels"));
                    }
                    let touch_frames = touches
                        .iter()
                        .map(|t| t.frame)
                        .collect::<std::collections::BTreeSet<_>>();
                    let touch_ids = touches
                        .iter()
                        .map(|t| &t.id)
                        .collect::<std::collections::BTreeSet<_>>();
                    let rally_ids = rallies
                        .iter()
                        .map(|r| &r.id)
                        .collect::<std::collections::BTreeSet<_>>();
                    if touch_frames.len() != touches.len()
                        || touch_ids.len() != touches.len()
                        || rally_ids.len() != rallies.len()
                    {
                        return Err(storage::bad_request("Duplicate saved Touch/Rally labels"));
                    }
                    v.touches = touches.clone();
                    v.rallies = rallies.clone();
                    v.trajectory_candidates = scanned
                        .as_ref()
                        .ok_or_else(|| storage::internal("Missing measured trajectory"))?
                        .clone();
                    v.trajectory_replay_sha = run.replay_sha.clone();
                }
                "seed" => {
                    let (touches, rallies) = predictions.as_ref().expect("seed predictions");
                    v.touches = touches
                        .iter()
                        .take(MAX_TOUCHES)
                        .map(|&frame| Mark {
                            id: Uuid::new_v4().to_string(),
                            frame,
                        })
                        .collect();
                    v.rallies = rallies
                        .iter()
                        .take(MAX_RALLIES)
                        .map(|&(start, end)| RallyMark {
                            id: Uuid::new_v4().to_string(),
                            start,
                            end,
                        })
                        .collect();
                }
                "add_touch" => {
                    let frame = valid_frame(req.frame)?;
                    if v.touches.len() >= MAX_TOUCHES || v.touches.iter().any(|t| t.frame == frame)
                    {
                        return Err(storage::bad_request(
                            "Too many labels or touch already at this frame",
                        ));
                    }
                    v.touches.push(Mark {
                        id: Uuid::new_v4().to_string(),
                        frame,
                    });
                }
                "move_touch" | "delete_touch" => {
                    let id = req
                        .id
                        .as_deref()
                        .ok_or_else(|| storage::bad_request("Missing Touch ID"))?;
                    let index = v
                        .touches
                        .iter()
                        .position(|t| t.id == id)
                        .ok_or_else(|| storage::bad_request("Touch label not found"))?;
                    if req.action == "delete_touch" {
                        v.touches.remove(index);
                    } else {
                        let frame = valid_frame(req.frame)?;
                        if v.touches
                            .iter()
                            .enumerate()
                            .any(|(j, t)| j != index && t.frame == frame)
                        {
                            return Err(storage::bad_request("Another Touch exists at this frame"));
                        }
                        v.touches[index].frame = frame;
                    }
                }
                "add_rally" | "move_rally" => {
                    let start = valid_frame(req.start)?;
                    let end = valid_frame(req.end)?;
                    if start >= end {
                        return Err(storage::bad_request("Rally start must precede end"));
                    }
                    if req.action == "add_rally" {
                        if v.rallies.len() >= MAX_RALLIES {
                            return Err(storage::bad_request("Too many rallies"));
                        }
                        v.rallies.push(RallyMark {
                            id: Uuid::new_v4().to_string(),
                            start,
                            end,
                        });
                    } else {
                        let id = req
                            .id
                            .as_deref()
                            .ok_or_else(|| storage::bad_request("Missing Rally ID"))?;
                        let row = v
                            .rallies
                            .iter_mut()
                            .find(|r| r.id == id)
                            .ok_or_else(|| storage::bad_request("Rally not found"))?;
                        row.start = start;
                        row.end = end;
                    }
                }
                "delete_rally" => {
                    let id = req
                        .id
                        .as_deref()
                        .ok_or_else(|| storage::bad_request("Missing Rally ID"))?;
                    let index = v
                        .rallies
                        .iter()
                        .position(|r| r.id == id)
                        .ok_or_else(|| storage::bad_request("Rally not found"))?;
                    v.rallies.remove(index);
                }
                "review" => {
                    if v.rallies.is_empty() {
                        return Err(storage::bad_request(
                            "Mark at least one rally before review",
                        ));
                    }
                    v.reviewed = true;
                    let score = outcome(v, &run)?;
                    v.quality = Some(
                        serde_json::from_value(score["touches"].clone())
                            .map_err(|e| storage::internal(format!("Calibration metrics: {e}")))?,
                    );
                }
                _ => return Err(storage::bad_request("Unknown annotation action")),
            }
            if req.action != "review" {
                v.reviewed = false;
                v.quality = None;
            }
            v.rally_run_id = run.run_id.clone();
            v.revision += 1;
            if !valid(v) {
                return Err(storage::bad_request("Invalid annotation data"));
            }
            save(&state, &lib)?;
            Ok(())
        })?;
        response(&state, &run, &source)
    })
    .await
    .map_err(|e| storage::internal(format!("Annotation write task: {e}")))??;
    Ok(Json(result))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn existing_saved_annotations_accept_optional_trajectory_evidence() {
        let old = json!({
            "project_id":"test","video":"match.mp4",
            "source_sha256":"a".repeat(64),"rally_run_id":"b".repeat(24),
            "total_frames":300,"revision":1,"reviewed":false,
            "touches":[],"rallies":[]
        });
        let parsed: VideoLabels = serde_json::from_value(old).unwrap();
        assert!(valid(&parsed));
        assert!(parsed.trajectory_candidates.is_empty());
        assert!(parsed.trajectory_replay_sha.is_empty());
        let roundtrip: VideoLabels =
            serde_json::from_value(serde_json::to_value(&parsed).unwrap()).unwrap();
        assert!(valid(&roundtrip));
    }

    #[test]
    fn reference_catalog_includes_only_reviewed_videos_from_selected_project() {
        let mk = |project: &str, video: &str, reviewed: bool| {
            serde_json::from_value::<VideoLabels>(json!({
                "project_id":project,"video":video,"source_sha256":"a".repeat(64),
                "rally_run_id":"b".repeat(24),"total_frames":300,
                "revision":1,"reviewed":reviewed,"touches":[],"rallies":[]
            }))
            .unwrap()
        };
        let library = Library {
            schema: SCHEMA.into(),
            videos: vec![
                mk("one", "reviewed.mp4", true),
                mk("one", "draft.mp4", false),
                mk("two", "private.mp4", true),
            ],
        };
        let result = catalog_value(&library, "one");
        assert_eq!(result["videos"].as_array().unwrap().len(), 1);
        assert_eq!(result["videos"][0]["video"], "reviewed.mp4");
        assert_eq!(result["videos"][0]["rallies"], 0);
    }
    #[test]
    fn duplicate_candidates_cannot_both_match_one_touch() {
        let result = accuracy(&[100], &[99, 100, 102], 3, 30.0);
        assert_eq!((result.matched, result.extra, result.missed), (1, 2, 0));
        assert_eq!(result.mean_error_ms, 0.0);
    }
    #[test]
    fn positional_errors_and_misses_count() {
        let result = accuracy(&[30, 60, 90], &[31, 90, 120], 3, 30.0);
        assert_eq!((result.matched, result.extra, result.missed), (2, 1, 1));
        assert!((result.mean_error_ms - 16.6666667).abs() < 0.01);
    }
    #[test]
    fn empty_detection_is_not_perfect() {
        let result = accuracy(&[1], &[], 3, 30.0);
        assert_eq!((result.matched, result.extra, result.missed), (0, 0, 1));
        assert_eq!(result.f1, 0.0);
    }
}
