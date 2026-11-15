//! Versioned, auditable human-reviewed pose and camera calibration examples.
//! Stored separately from immutable GPU evidence and CPU replay snapshots.
use crate::{
    native_preview_overlay, projects,
    storage::{self, ApiError, ApiResult},
    video_edits, AppState,
};
use axum::{
    extract::{Query, State},
    http::HeaderMap,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs,
    process::Stdio,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};
use uuid::Uuid;
use volley_core::{
    action_prediction::{ActionKind, ActionPrediction, PredictionMethod},
    pose_classification::PoseEvidence,
};

const SCHEMA: &str = "volley-human-training-library/1";
const MAX_FILE: u64 = 16 * 1024 * 1024;
const MAX_EXAMPLES: usize = 2048;
static LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MotionSample {
    pub(crate) offset: i16,
    pub(crate) keypoints: [[f64; 3]; 17],
    pub(crate) bbox: [f64; 4],
    pub(crate) ball: Option<[f64; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Example {
    Touch {
        id: String,
        project_id: String,
        video: String,
        rally_run_id: String,
        replay_sha256: String,
        frame_index: u64,
        track_id: u64,
        label: ActionKind,
        keypoints: [[f64; 3]; 17],
        bbox: [f64; 4],
        ball: [f64; 2],
        width: u32,
        height: u32,
        contact_joint_index: u8,
        motion: Vec<MotionSample>,
    },
    Court {
        id: String,
        project_id: String,
        video: String,
        frame_index: u64,
        corners: [[f64; 2]; 4],
        net: Option<[[f64; 2]; 2]>,
        camera_signature: Vec<u8>,
        aspect_ratio: f64,
    },
}
impl Example {
    fn id(&self) -> &str {
        match self {
            Self::Touch { id, .. } | Self::Court { id, .. } => id,
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Library {
    schema: String,
    revision: u64,
    examples: Vec<Example>,
}
impl Default for Library {
    fn default() -> Self {
        Self {
            schema: SCHEMA.into(),
            revision: 0,
            examples: Vec::new(),
        }
    }
}
fn path(state: &AppState) -> std::path::PathBuf {
    state.data_root.join("training").join("examples.json")
}
fn read(state: &AppState) -> Result<Library, ApiError> {
    let file = path(state);
    match fs::symlink_metadata(&file) {
        Ok(m) if !m.is_file() || m.file_type().is_symlink() || m.len() > MAX_FILE => {
            return Err(storage::bad_request("Unsafe training library"))
        }
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Library::default()),
        Err(e) => return Err(storage::internal(format!("Training library metadata: {e}"))),
    }
    let library: Library = serde_json::from_slice(
        &fs::read(file).map_err(|e| storage::internal(format!("Read training library: {e}")))?,
    )
    .map_err(|_| storage::bad_request("Invalid training library data"))?;
    if library.schema != SCHEMA
        || library.examples.len() > MAX_EXAMPLES
        || library.examples.iter().any(|e| !valid_example(e))
    {
        return Err(storage::bad_request("Incompatible training library schema"));
    }
    Ok(library)
}
fn valid_keypoint(p: &[f64; 3]) -> bool {
    p[0].is_finite()
        && p[1].is_finite()
        && (-0.5..=1.5).contains(&p[0])
        && (-0.5..=1.5).contains(&p[1])
        && p[2].is_finite()
        && (0.0..=1.0).contains(&p[2])
}
fn valid_example(e: &Example) -> bool {
    match e {
        Example::Touch {
            id,
            project_id,
            video,
            rally_run_id,
            replay_sha256,
            frame_index,
            track_id,
            keypoints,
            bbox,
            ball,
            width,
            height,
            contact_joint_index,
            motion,
            ..
        } => {
            Uuid::parse_str(id).is_ok()
                && !project_id.is_empty()
                && !video.is_empty()
                && !rally_run_id.is_empty()
                && replay_sha256.len() == 64
                && *frame_index < 48 * 3600 * 30
                && *track_id > 0
                && *width >= 64
                && *height >= 64
                && bbox
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                && bbox[0] < bbox[2]
                && bbox[1] < bbox[3]
                && ball
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                && keypoints.iter().all(valid_keypoint)
                && *contact_joint_index < 17
                && !motion.is_empty()
                && motion.len() <= 13
                && motion.iter().any(|m| m.offset == 0)
                && motion.iter().all(|m| {
                    m.offset.abs() <= 6
                        && m.keypoints.iter().all(valid_keypoint)
                        && m.bbox
                            .iter()
                            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                        && m.bbox[0] < m.bbox[2]
                        && m.bbox[1] < m.bbox[3]
                        && m.ball.is_none_or(|b| {
                            b.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                        })
                })
        }
        Example::Court {
            id,
            project_id,
            video,
            frame_index,
            corners,
            net,
            camera_signature,
            aspect_ratio,
        } => {
            Uuid::parse_str(id).is_ok()
                && !project_id.is_empty()
                && !video.is_empty()
                && *frame_index < 48 * 3600 * 30
                && aspect_ratio.is_finite()
                && *aspect_ratio > 0.3
                && *aspect_ratio < 5.0
                && camera_signature.len() == 432
                && video_edits::court_is_valid(corners)
                && net.as_ref().is_none_or(|line| {
                    line.iter()
                        .flatten()
                        .all(|v| v.is_finite() && (-0.5..=1.5).contains(v))
                })
        }
    }
}
fn write(state: &AppState, library: &Library) -> Result<(), ApiError> {
    let file = path(state);
    let dir = file
        .parent()
        .ok_or_else(|| storage::bad_request("Invalid training directory"))?;
    if fs::symlink_metadata(dir).is_ok_and(|m| !m.is_dir() || m.file_type().is_symlink()) {
        return Err(storage::bad_request("Unsafe training directory"));
    }
    fs::create_dir_all(dir)
        .map_err(|e| storage::internal(format!("Create training directory: {e}")))?;
    if fs::symlink_metadata(&file).is_ok_and(|m| !m.is_file() || m.file_type().is_symlink()) {
        return Err(storage::bad_request("Unsafe training file"));
    }
    storage::write_json_atomic(&file, &json!(library))
}
fn locked<T>(callback: impl FnOnce() -> Result<T, ApiError>) -> Result<T, ApiError> {
    let _lock = LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| storage::internal("Training lock poisoned"))?;
    callback()
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EditRequest {
    action: String,
    project_id: String,
    video: Option<String>,
    rally_run_id: Option<String>,
    frame_index: Option<u64>,
    track_id: Option<u64>,
    label: Option<ActionKind>,
    corners: Option<[[f64; 2]; 4]>,
    net: Option<[[f64; 2]; 2]>,
    id: Option<String>,
}
fn source(state: &AppState, project_id: &str, video: &str) -> Result<(), ApiError> {
    let project = projects::existing_project(state, project_id)?;
    storage::video_name(video)?;
    if !project["videos"]
        .as_array()
        .is_some_and(|v| v.iter().any(|n| n.as_str() == Some(video)))
    {
        return Err(storage::bad_request("Training video not in project"));
    }
    Ok(())
}
fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = w * h;
    intersection
        / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection).max(1e-9)
}
/// Select exactly one observed RTMPose skeleton corresponding to the CPU Track ID.
fn observed_movement(row: &Value, track: u64) -> Option<MotionSample> {
    if row["pose_points_interpolated"] == true {
        return None;
    }
    let player = row["players"]
        .as_array()?
        .iter()
        .find(|p| p["id"].as_u64() == Some(track))?;
    let bbox: [f64; 4] = serde_json::from_value(player["box"].clone()).ok()?;
    let mut candidates = row["pose_points"]
        .as_array()?
        .iter()
        .filter_map(|p| {
            let pb: [f64; 4] = serde_json::from_value(p["box"].clone()).ok()?;
            let pose: [[f64; 3]; 17] = serde_json::from_value(p["keypoints"].clone()).ok()?;
            Some((iou(bbox, pb), pose))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    if candidates.first()?.0 < 0.40
        || candidates
            .get(1)
            .is_some_and(|next| candidates[0].0 - next.0 < 0.10)
    {
        return None;
    }
    let ball = (row["ball_origin"] == "vballnet")
        .then(|| serde_json::from_value(row["ball"].clone()).ok())
        .flatten();
    Some(MotionSample {
        offset: 0,
        keypoints: candidates[0].1,
        bbox,
        ball,
    })
}
async fn capture_touch(
    state: Arc<AppState>,
    request: &EditRequest,
    video: &str,
) -> Result<Example, ApiError> {
    let run = request
        .rally_run_id
        .as_deref()
        .ok_or_else(|| storage::bad_request("Select CPU rally analysis"))?;
    let frame = request
        .frame_index
        .ok_or_else(|| storage::bad_request("Missing frame index"))?;
    let track = request
        .track_id
        .ok_or_else(|| storage::bad_request("Select the touching player"))?;
    let label = request
        .label
        .ok_or_else(|| storage::bad_request("Missing touch label"))?;
    let overlay =
        native_preview_overlay::training_frame(state, &request.project_id, run, frame).await?;
    if overlay["video"] != video {
        return Err(storage::bad_request("Training source mismatch"));
    }
    let frames = overlay["frames"]
        .as_array()
        .ok_or_else(|| storage::bad_request("No training frames"))?;
    let row = frames
        .iter()
        .find(|item| item["frame"].as_u64() == Some(frame))
        .ok_or_else(|| storage::bad_request("Training frame not in source"))?;
    if row["pose_points_interpolated"] == true {
        return Err(storage::bad_request(
            "Поза интерполирована. Выберите кадр с измерением RTMPose (шаг 0,1 с).",
        ));
    }
    let main = observed_movement(row, track).ok_or_else(|| {
        storage::bad_request("Нет однозначного соответствия Track ID и точек RTMPose на этом кадре")
    })?;
    let ball = main.ball.ok_or_else(|| {
        storage::bad_request(
            "Только настоящее наблюдение мяча VBallNet можно сохранить как касание",
        )
    })?;
    let width = overlay["width"]
        .as_u64()
        .ok_or_else(|| storage::bad_request("No source width"))? as u32;
    let height = overlay["height"]
        .as_u64()
        .ok_or_else(|| storage::bad_request("No source height"))? as u32;
    let evidence = PoseEvidence {
        keypoints: main.keypoints,
        ball_x_norm: ball[0],
        ball_y_norm: ball[1],
        video_width: width as f64,
        video_height: height as f64,
        player_bbox_norm: main.bbox,
    };
    if !volley_core::pose_classification::contact_distance(&evidence, 0.35)
        .is_some_and(|distance| distance <= 0.35)
    {
        return Err(storage::bad_request(
            "Нет пересечения мяча и точек тела. Перейдите на момент контакта.",
        ));
    }
    let height_px = ((main.bbox[3] - main.bbox[1]) * height as f64).max(1.0);
    let joint = (0..17)
        .filter(|&index| main.keypoints[index][2] >= 0.35)
        .min_by(|&a, &b| {
            let distance = |i: usize| {
                ((main.keypoints[i][0] - ball[0]) * width as f64)
                    .hypot((main.keypoints[i][1] - ball[1]) * height as f64)
                    / height_px
            };
            distance(a).total_cmp(&distance(b))
        })
        .ok_or_else(|| storage::bad_request("No reliable contact joint"))?;
    // Store up to 13 measured frames (±0.2 s). No invented pose is training data.
    let mut motion = frames
        .iter()
        .filter_map(|item| {
            let offset = item["frame"].as_i64()? - frame as i64;
            let mut sample = observed_movement(item, track)?;
            sample.offset = i16::try_from(offset).ok()?;
            (sample.offset.abs() <= 6).then_some(sample)
        })
        .collect::<Vec<_>>();
    motion.sort_by_key(|m| m.offset);
    let value = Example::Touch {
        id: Uuid::new_v4().to_string(),
        project_id: request.project_id.clone(),
        video: video.into(),
        rally_run_id: run.into(),
        replay_sha256: overlay["replay_sha256"].as_str().unwrap_or_default().into(),
        frame_index: frame,
        track_id: track,
        label,
        keypoints: main.keypoints,
        bbox: main.bbox,
        ball,
        width,
        height,
        contact_joint_index: joint as u8,
        motion,
    };
    if !valid_example(&value) {
        return Err(storage::bad_request("Invalid source touch evidence"));
    }
    Ok(value)
}
/// 16x9 low-frequency RGB camera signature; no original frame pixels are retained.
pub(crate) fn camera_signature(rgb: &[u8]) -> Option<Vec<u8>> {
    if rgb.len() != 640 * 360 * 3 {
        return None;
    }
    let mut output = Vec::with_capacity(432);
    for y in 0..9 {
        for x in 0..16 {
            let mut totals = [0u64; 3];
            for yy in y * 40..(y + 1) * 40 {
                for xx in x * 40..(x + 1) * 40 {
                    let k = (yy * 640 + xx) * 3;
                    for channel in 0..3 {
                        totals[channel] += u64::from(rgb[k + channel]);
                    }
                }
            }
            output.extend(totals.map(|sum| (sum / 1600) as u8));
        }
    }
    Some(output)
}
async fn capture_court(
    state: Arc<AppState>,
    request: &EditRequest,
    video: &str,
) -> Result<Example, ApiError> {
    let corners = request
        .corners
        .ok_or_else(|| storage::bad_request("Missing court corners"))?;
    if !video_edits::court_is_valid(&corners)
        || request.net.is_some_and(|n| {
            n.iter()
                .flatten()
                .any(|v| !v.is_finite() || !(-0.5..=1.5).contains(v))
        })
    {
        return Err(storage::bad_request("Invalid court geometry"));
    }
    // Geometry must be explicitly saved in the video editor before it can train.
    let Json(saved) = video_edits::get(
        State(state.clone()),
        Query(HashMap::from([
            ("project_id".into(), request.project_id.clone()),
            ("video".into(), video.into()),
        ])),
    )
    .await?;
    if saved["court"] != json!(corners) || saved["net"] != json!(request.net) {
        return Err(storage::bad_request(
            "Save the manually reviewed court before teaching",
        ));
    }
    let frame = request
        .frame_index
        .ok_or_else(|| storage::bad_request("Missing camera frame"))?;
    let filename = video_edits::calibration_source(&state, &request.project_id, video)?;
    let dimensions = volley_media::probe(&filename).map_err(storage::bad_request)?;
    let mut process = tokio::process::Command::new("ffmpeg");
    process
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            &format!("{:.3}", frame as f64 / 30.0),
        ])
        .arg("-i")
        .arg(&filename)
        .args([
            "-frames:v",
            "1",
            "-vf",
            "scale=640:360:flags=bicubic,format=rgb24",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .stdout(Stdio::piped())
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(20), process.output())
        .await
        .map_err(|_| storage::bad_request("Training camera sample timeout"))?
        .map_err(|e| storage::bad_request(format!("FFmpeg camera sample: {e}")))?;
    let signature = camera_signature(&output.stdout)
        .ok_or_else(|| storage::bad_request("Could not read camera training frame"))?;
    let value = Example::Court {
        id: Uuid::new_v4().to_string(),
        project_id: request.project_id.clone(),
        video: video.into(),
        frame_index: frame,
        corners,
        net: request.net,
        camera_signature: signature,
        aspect_ratio: dimensions.width as f64 / dimensions.height as f64,
    };
    if !valid_example(&value) {
        return Err(storage::bad_request("Invalid court training sample"));
    }
    Ok(value)
}
pub async fn list(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HashMap<String, String>>,
) -> ApiResult<Value> {
    if let Some(project) = query.get("project_id") {
        projects::existing_project(&state, project)?;
    }
    let library = locked(|| read(&state))?;
    let items = library
        .examples
        .iter()
        .rev()
        .map(|e| match e {
            Example::Touch {
                id,
                project_id,
                video,
                frame_index,
                track_id,
                label,
                ..
            } => json!({
                "id":id,"type":"touch","project_id":project_id,"video":video,
                "frame_index":frame_index,"track_id":track_id,"label":label
            }),
            Example::Court {
                id,
                project_id,
                video,
                frame_index,
                ..
            } => json!({
                "id":id,"type":"court","project_id":project_id,"video":video,
                "frame_index":frame_index,"label":"court_and_net"
            }),
        })
        .collect::<Vec<_>>();
    Ok(Json(
        json!({"schema":SCHEMA,"revision":library.revision,"examples":items}),
    ))
}
pub async fn edit(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<EditRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    if !state.writes_enabled {
        return Err(storage::conflict("Training library is read-only"));
    }
    projects::existing_project(&state, &req.project_id)?;
    let example = match req.action.as_str() {
        "touch" | "court" => {
            let video = req
                .video
                .as_deref()
                .ok_or_else(|| storage::bad_request("Missing training video"))?;
            source(&state, &req.project_id, video)?;
            Some(if req.action == "touch" {
                capture_touch(state.clone(), &req, video).await?
            } else {
                capture_court(state.clone(), &req, video).await?
            })
        }
        "delete" => None,
        _ => return Err(storage::bad_request("Invalid training action")),
    };
    let revision = locked(|| {
        let mut library = read(&state)?;
        if let Some(example) = example {
            // One source frame/type is one label: correcting it replaces the old example.
            library.examples.retain(|row| match (&example, row) {
                (
                    Example::Touch {
                        project_id: p,
                        video: v,
                        frame_index: f,
                        track_id: t,
                        ..
                    },
                    Example::Touch {
                        project_id: q,
                        video: w,
                        frame_index: g,
                        track_id: u,
                        ..
                    },
                ) => p != q || v != w || f != g || t != u,
                (
                    Example::Court {
                        project_id: p,
                        video: v,
                        ..
                    },
                    Example::Court {
                        project_id: q,
                        video: w,
                        ..
                    },
                ) => p != q || v != w,
                _ => true,
            });
            if library.examples.len() >= MAX_EXAMPLES {
                return Err(storage::bad_request(
                    "Training library is full; delete incorrect or obsolete examples",
                ));
            }
            library.examples.push(example);
        } else {
            let id = req
                .id
                .as_deref()
                .ok_or_else(|| storage::bad_request("Missing training example id"))?;
            if Uuid::parse_str(id).is_err() {
                return Err(storage::bad_request("Invalid training example id"));
            }
            let count = library.examples.len();
            library.examples.retain(|e| e.id() != id);
            if count == library.examples.len() {
                return Err(storage::bad_request("Training example not found"));
            }
        }
        library.revision += 1;
        write(&state, &library)?;
        Ok(library.revision)
    })?;
    Ok(Json(json!({"ok":true,"revision":revision})))
}
pub(crate) fn touch_examples(state: &AppState) -> Result<Vec<Example>, ApiError> {
    Ok(locked(|| read(state))?
        .examples
        .into_iter()
        .filter(|e| matches!(e, Example::Touch { .. }))
        .collect())
}
/// Compare shoulder-relative wrist travel before/after impact. This is temporal
/// motion evidence, not just a screenshot embedding.
fn motion_difference(
    current: &[MotionSample],
    current_size: (f64, f64),
    saved: &[MotionSample],
    saved_size: (f64, f64),
) -> Option<f64> {
    fn displacement(
        samples: &[MotionSample],
        size: (f64, f64),
        shoulder: usize,
        wrist: usize,
    ) -> Option<[f64; 2]> {
        let before = samples
            .iter()
            .filter(|s| s.offset < 0)
            .max_by_key(|s| s.offset)?;
        let after = samples
            .iter()
            .filter(|s| s.offset > 0)
            .min_by_key(|s| s.offset)?;
        let relative = |s: &MotionSample| -> Option<[f64; 2]> {
            let a = s.keypoints[shoulder];
            let b = s.keypoints[wrist];
            if a[2] < 0.35 || b[2] < 0.35 {
                return None;
            }
            let h = (s.bbox[3] - s.bbox[1]) * size.1;
            if h < 26.0 {
                return None;
            }
            Some([(b[0] - a[0]) * size.0 / h, (b[1] - a[1]) * size.1 / h])
        };
        let a = relative(before)?;
        let b = relative(after)?;
        Some([b[0] - a[0], b[1] - a[1]])
    }
    let mut sum = 0.0;
    let mut count = 0;
    for (shoulder, wrist) in [(5, 9), (6, 10)] {
        if let (Some(a), Some(b)) = (
            displacement(current, current_size, shoulder, wrist),
            displacement(saved, saved_size, shoulder, wrist),
        ) {
            sum += (a[0] - b[0]).hypot(a[1] - b[1]).min(2.0);
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f64)
}
/// Explainable nearest-pose matching. Only influences classification of already
/// independently detected body/ball contacts, never creates a touch.
pub(crate) fn predict(
    e: &PoseEvidence,
    geometry: &ActionPrediction,
    examples: &[Example],
    movement: &[MotionSample],
) -> Option<ActionPrediction> {
    fn diff(a: &PoseEvidence, b: &PoseEvidence) -> Option<f64> {
        let indices = [5usize, 6, 7, 8, 9, 10, 11, 12];
        let normal = |e: &PoseEvidence, index: usize| -> Option<[f64; 2]> {
            let p = e.keypoints[index];
            let height = (e.player_bbox_norm[3] - e.player_bbox_norm[1]) * e.video_height;
            if p[2] < 0.35 || height < 26.0 {
                return None;
            }
            Some([
                (p[0] - e.player_bbox_norm[0]) * e.video_width / height,
                (p[1] - e.player_bbox_norm[1]) * e.video_height / height,
            ])
        };
        let mut sum = 0.0;
        let mut weight = 0.0;
        for index in indices {
            if let (Some(x), Some(y)) = (normal(a, index), normal(b, index)) {
                let w = if index == 9 || index == 10 { 2.5 } else { 1.0 };
                sum += w * (x[0] - y[0]).hypot(x[1] - y[1]).min(2.0);
                weight += w;
            }
        }
        if weight < 6.0 {
            return None;
        }
        let ballpos = |e: &PoseEvidence| {
            let height =
                ((e.player_bbox_norm[3] - e.player_bbox_norm[1]) * e.video_height).max(1.0);
            [
                (e.ball_x_norm - e.player_bbox_norm[0]) * e.video_width / height,
                (e.ball_y_norm - e.player_bbox_norm[1]) * e.video_height / height,
            ]
        };
        let ball1 = ballpos(a);
        let ball2 = ballpos(b);
        Some(
            (sum + 2.0 * (ball1[0] - ball2[0]).hypot(ball1[1] - ball2[1]).min(2.0))
                / (weight + 2.0),
        )
    }
    if !geometry.valid() {
        return None;
    }
    let mut candidates = examples
        .iter()
        .filter_map(|sample| {
            let Example::Touch {
                label,
                keypoints,
                bbox,
                ball,
                width,
                height,
                motion,
                ..
            } = sample
            else {
                return None;
            };
            let saved = PoseEvidence {
                keypoints: *keypoints,
                player_bbox_norm: *bbox,
                ball_x_norm: ball[0],
                ball_y_norm: ball[1],
                video_width: *width as f64,
                video_height: *height as f64,
            };
            let position = diff(e, &saved)?;
            let temporal = motion_difference(
                movement,
                (e.video_width, e.video_height),
                motion,
                (*width as f64, *height as f64),
            );
            let distance = temporal.map_or(position, |motion_distance| {
                0.65 * position + 0.35 * motion_distance
            });
            Some((*label, (-4.5 * distance).exp()))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let (label, score) = *candidates.first()?;
    let competing = candidates
        .iter()
        .filter(|(kind, _)| *kind != label)
        .map(|(_, s)| *s)
        .fold(0.0_f64, f64::max);
    let matching = candidates
        .iter()
        .filter(|(kind, s)| *kind == label && *s >= 0.82)
        .count();
    if score < if matching >= 2 { 0.86 } else { 0.95 } || score - competing < 0.07 {
        return None;
    }
    Some(ActionPrediction {
        kind: label,
        evidence_score: score,
        method: PredictionMethod::LearnedPoseV1,
        depth_ratio: geometry.depth_ratio,
        posture_ratio: geometry.posture_ratio,
    })
}
/// Suggest geometry only for near-identical camera views; no camera extrapolation.
pub(crate) fn match_court(state: &AppState, image: &[u8], aspect: f64) -> Option<Value> {
    let signature = camera_signature(image)?;
    let examples = touch_safe_court_examples(state)?;
    let mut best: Option<(f64, &Example)> = None;
    for example in &examples {
        let Example::Court {
            camera_signature,
            aspect_ratio,
            ..
        } = example
        else {
            continue;
        };
        if (aspect_ratio / aspect - 1.0).abs() > 0.015 {
            continue;
        }
        let mean = signature
            .iter()
            .zip(camera_signature)
            .map(|(a, b)| a.abs_diff(*b) as f64)
            .sum::<f64>()
            / (signature.len() as f64 * 255.0);
        let score = 1.0 - mean;
        if score >= 0.955 && best.as_ref().is_none_or(|(b, _)| score > *b) {
            best = Some((score, example));
        }
    }
    let (
        score,
        Example::Court {
            id, corners, net, ..
        },
    ) = best?
    else {
        return None;
    };
    Some(
        json!({"detected":true,"court":corners,"net":net,"confidence":score,
        "learned":true,"example_id":id,"inferred":false,"weak_consensus":false}),
    )
}
fn touch_safe_court_examples(state: &AppState) -> Option<Vec<Example>> {
    Some(
        locked(|| read(state))
            .ok()?
            .examples
            .into_iter()
            .filter(|e| matches!(e, Example::Court { .. }))
            .collect(),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn camera_signature_is_bounded() {
        assert!(camera_signature(&[]).is_none());
        let pixels = vec![100u8; 640 * 360 * 3];
        assert_eq!(camera_signature(&pixels), Some(vec![100u8; 432]));
    }
    #[test]
    fn confirmed_pose_and_wrist_motion_match_one_label_but_not_competitor() {
        let mut joints = [[0.5, 0.5, 0.0]; 17];
        joints[5] = [0.44, 0.35, 0.9];
        joints[6] = [0.56, 0.35, 0.9];
        joints[7] = [0.43, 0.38, 0.9];
        joints[8] = [0.58, 0.38, 0.9];
        joints[9] = [0.46, 0.31, 0.9];
        joints[10] = [0.55, 0.39, 0.9];
        joints[11] = [0.45, 0.65, 0.9];
        joints[12] = [0.54, 0.65, 0.9];
        let bbox = [0.30, 0.20, 0.70, 0.90];
        let mut before = joints;
        before[9][1] = 0.42;
        let mut after = joints;
        after[9][1] = 0.27;
        let motion = vec![
            MotionSample {
                offset: -3,
                keypoints: before,
                bbox,
                ball: None,
            },
            MotionSample {
                offset: 0,
                keypoints: joints,
                bbox,
                ball: Some([0.46, 0.31]),
            },
            MotionSample {
                offset: 3,
                keypoints: after,
                bbox,
                ball: None,
            },
        ];
        let sample = Example::Touch {
            id: Uuid::new_v4().to_string(),
            project_id: "p".into(),
            video: "v.mp4".into(),
            rally_run_id: "r".into(),
            replay_sha256: "a".repeat(64),
            frame_index: 30,
            track_id: 3,
            label: ActionKind::Attack,
            keypoints: joints,
            bbox,
            ball: [0.46, 0.31],
            width: 1920,
            height: 1080,
            contact_joint_index: 9,
            motion: motion.clone(),
        };
        assert!(valid_example(&sample));
        let evidence = PoseEvidence {
            keypoints: joints,
            ball_x_norm: 0.46,
            ball_y_norm: 0.31,
            video_width: 1920.0,
            video_height: 1080.0,
            player_bbox_norm: bbox,
        };
        let geometry = ActionPrediction {
            kind: ActionKind::Dig,
            evidence_score: 0.6,
            method: PredictionMethod::ContactGeometryV1,
            depth_ratio: 0.2,
            posture_ratio: 0.5,
        };
        let result = predict(&evidence, &geometry, &[sample.clone()], &motion).unwrap();
        assert_eq!(result.kind, ActionKind::Attack);
        assert_eq!(result.method, PredictionMethod::LearnedPoseV1);
        let mut competitor = sample.clone();
        if let Example::Touch { label, .. } = &mut competitor {
            *label = ActionKind::Dig;
        }
        assert!(predict(&evidence, &geometry, &[competitor, sample], &motion).is_none());
    }
    #[test]
    fn pose_similarity_rejects_unrelated() {
        let e = PoseEvidence {
            keypoints: [[0.5, 0.5, 0.0]; 17],
            ball_x_norm: 0.5,
            ball_y_norm: 0.5,
            video_width: 1920.0,
            video_height: 1080.0,
            player_bbox_norm: [0.3, 0.2, 0.7, 0.9],
        };
        let prediction = ActionPrediction {
            kind: ActionKind::Dig,
            evidence_score: 0.5,
            method: PredictionMethod::PoseKeypointsV1,
            depth_ratio: 0.0,
            posture_ratio: 0.5,
        };
        assert!(predict(&e, &prediction, &[], &[]).is_none());
    }
}
