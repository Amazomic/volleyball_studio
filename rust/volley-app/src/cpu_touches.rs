//! Re-evaluate motion-supported Touch candidates exclusively from immutable
//! native CPU Track IDs and selected-ball proposals, never rerunning ONNX.
//! These events are not yet authoritative game/rally/athlete statistics.
use crate::{
    court_touch,
    cpu_retrack::opaque_id,
    inference_stream, native_preview_pose, preparation, project_guard, projects,
    storage::{self, safe_file, safe_folder, ApiError, ApiResult, BoundedReader},
    training_samples, video_edits, AppState,
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
    collections::{BTreeMap, VecDeque},
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use uuid::Uuid;
#[cfg(test)]
use volley_core::touch_pipeline::evaluate_block;
use volley_core::{
    court_geometry::CourtTransform,
    fusion::Candidate,
    pose_classification,
    touch_pipeline::{
        attach_touch_source_evidence, detect_ball_trajectory_touches, exclude_ball_teleports,
        observed_pose_motion, pose_contact_at_frame, pose_contact_from_nearby_sample,
        reconstruct_track_gaps, PhysicalTouchCandidate, PoseFrame, PosePlayer, TouchAlgorithm,
        TouchBall, TouchFrame, TouchPlayer, TouchSettings,
    },
};
use volley_inference::trajectory_tcn::{TrajectoryTcn, MODEL_FILENAME};

const MAX_INPUT_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 16384;
const CHUNK: usize = 240;
const HALO: usize = 96;
const OUTPUT_FILE: &str = "touch_candidates.jsonl";
const BALL_TRAJECTORY_FILE: &str = "trajectory.jsonl";
const BALL_TRAJECTORY_SCHEMA: &str = "volley-native-stabilized-ball-trajectory/1";

#[cfg(test)]
fn pose_frames(root: &Path, fusion: &str, frames: &[TouchFrame]) -> Result<Vec<PoseFrame>, String> {
    pose_frames_policy(root, fusion, frames, None, &BTreeMap::new())
}

fn pose_frames_policy(
    root: &Path,
    fusion: &str,
    frames: &[TouchFrame],
    interpolation: Option<bool>,
    refined: &BTreeMap<u64, Vec<PosePlayer>>,
) -> Result<Vec<PoseFrame>, String> {
    let Some(first) = frames.first() else {
        return Ok(Vec::new());
    };
    let Some(last) = frames.last() else {
        return Ok(Vec::new());
    };
    let mut values: Vec<Value> = frames
        .iter()
        .map(|frame| {
            json!({
                "frame":frame.frame_index,
                "players":frame.players.iter().map(|player| {
                    let mut row = json!({"id":player.track_id,"box":player.bbox_xyxy_norm});
                    if let Some(pose) = refined.get(&frame.frame_index).and_then(|poses| {
                        poses.iter().find(|pose| pose.track_id == player.track_id)
                    }) {
                        row["pose"] = json!(pose.keypoints);
                        row["pose_source"] = json!(crate::cpu_retrack::POSE_REFINEMENT_POLICY);
                    }
                    row
                }).collect::<Vec<_>>()
            })
        })
        .collect();
    if !native_preview_pose::enrich_continuous_policy(
        root,
        fusion,
        first.frame_index,
        last.frame_index + 1,
        &mut values,
        interpolation,
    )? && refined.is_empty()
    {
        return Err("Для выбранного режима необходим сохранённый анализ позы".into());
    }
    Ok(values
        .into_iter()
        .filter_map(|frame| {
            let frame_index = frame["frame"].as_u64()?;
            let players: Vec<PosePlayer> = frame["players"]
                .as_array()?
                .iter()
                .filter_map(|player| {
                    Some(PosePlayer {
                        track_id: player["id"].as_u64()?,
                        keypoints: serde_json::from_value(player.get("pose")?.clone()).ok()?,
                        interpolated: player["pose_interpolated"] == true,
                    })
                })
                .collect();
            (!players.is_empty()).then_some(PoseFrame {
                frame_index,
                players,
            })
        })
        .collect())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TouchRequest {
    pub(crate) project_id: String,
    pub(crate) video: String,
    pub(crate) replay_id: String,
    #[serde(default)]
    pub(crate) settings: Option<TouchSettings>,
    #[serde(default)]
    pub(crate) pose_classification: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct PrimaryRallyScope {
    pub intervals: Vec<(u64, u64)>,
    pub source_frames: u64,
    pub ball_job_id: String,
    pub sha256: String,
}

pub(crate) fn align_primary_intervals(
    intervals: &[(u64, u64)],
    source_frames: u64,
    replay_frames: u64,
) -> Result<Vec<(u64, u64)>, String> {
    if replay_frames == 0
        || !replay_frames.is_multiple_of(9)
        || source_frames < replay_frames
        || source_frames > replay_frames + 8
        || intervals
            .iter()
            .any(|&(start, end)| start > end || end >= source_frames)
        || intervals.windows(2).any(|pair| pair[0].1 >= pair[1].0)
    {
        return Err("Primary GPU rally intervals do not match the CPU replay timeline".into());
    }
    Ok(intervals
        .iter()
        .filter(|&&(start, _)| start < replay_frames)
        .map(|&(start, end)| (start, end.min(replay_frames - 1)))
        .collect())
}

#[derive(Debug, Deserialize)]
pub struct TouchQuery {
    project_id: String,
    touch_id: String,
}

#[derive(Deserialize)]
struct TrackedCandidate {
    track_id: u64,
    detection: Candidate,
    is_verified_athlete: bool,
    pose: Option<[[f64; 3]; 17]>,
    pose_source: Option<String>,
}
#[derive(Deserialize)]
struct SelectedBall {
    source_model: String,
    detection: Candidate,
}
#[derive(Deserialize)]
pub(crate) struct TrackedFrame {
    frame_index: u64,
    tracked_players: Vec<TrackedCandidate>,
    selected_ball: Option<SelectedBall>,
    track_ids_provisional: bool,
    confirmed_athlete_identities: bool,
    touches_confirmed: bool,
}

pub(crate) fn transform_frame(item: TrackedFrame, expected: u64) -> Result<TouchFrame, String> {
    if item.frame_index != expected
        || !item.track_ids_provisional
        || item.confirmed_athlete_identities
        || item.touches_confirmed
        || item.tracked_players.len() > 128
    {
        return Err("CPU source has incorrect frame index or verified status".into());
    }
    let mut players = Vec::with_capacity(item.tracked_players.len());
    for player in item.tracked_players {
        let sample = &player.detection;
        if player.is_verified_athlete
            || player.track_id == 0
            || sample.frame_index != expected
            || sample.class_name != "player"
            || sample.query_index.is_none()
            || sample.score <= 0.0
        {
            return Err("CPU source contains fabricated athlete identity or invalid track".into());
        }
        sample.validate(expected - (expected % 9), "players")?;
        players.push(TouchPlayer {
            track_id: player.track_id,
            score: sample.score,
            bbox_xyxy_norm: sample.bbox_xyxy_norm,
            touch_probability: sample.touch_probability,
        });
    }
    let selected_ball = item
        .selected_ball
        .map(|ball| -> Result<TouchBall, String> {
            if ball.source_model != "vballnet"
                || ball.detection.frame_index != expected
                || ball.detection.class_name != "ball"
            {
                return Err("Invalid source for selected ball".into());
            }
            ball.detection.validate(expected - (expected % 9), "ball")?;
            Ok(TouchBall {
                source_model: ball.source_model,
                score: ball.detection.score,
                bbox_xyxy_norm: ball.detection.bbox_xyxy_norm,
            })
        })
        .transpose()?;
    Ok(TouchFrame {
        frame_index: expected,
        players,
        selected_ball,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CanonicalBallFrame {
    pub frame: u64,
    pub ball: Option<[f64; 2]>,
    pub interpolated: bool,
    pub rejected: bool,
}

impl CanonicalBallFrame {
    pub(crate) fn validate(&self, expected: u64) -> Result<(), String> {
        if self.frame != expected
            || self.ball.is_some_and(|[x, y]| {
                !x.is_finite()
                    || !y.is_finite()
                    || !(0.0..=1.0).contains(&x)
                    || !(0.0..=1.0).contains(&y)
            })
            || (self.interpolated && self.ball.is_none())
        {
            return Err("Invalid canonical Ball Stabilizer trajectory frame".into());
        }
        Ok(())
    }
}

pub(crate) struct StabilizedReplay {
    pub source_sha256: String,
    pub motion: volley_core::ball_interpolation::StabilizedBallMotion,
    pub settings: volley_core::ball_interpolation::BallStabilizerSettings,
    pub artifact_id: String,
    pub trajectory_sha256: String,
    pub frames: u64,
    pub project_root: PathBuf,
    pub replay_id: String,
}

impl StabilizedReplay {
    pub(crate) fn summary(&self) -> Value {
        json!({
            "schema":"volley-ball-stabilizer/1",
            "version":volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION,
            "source_replay_sha256":self.source_sha256,
            "source_replay_id":self.replay_id,
            "settings":self.settings,
            "artifact_id":self.artifact_id,
            "trajectory_file":BALL_TRAJECTORY_FILE,
            "trajectory_sha256":self.trajectory_sha256,
            "frames":self.frames,
            "confidence_frames":self.motion.confidence_frames,
            "observed_frames":self.motion.observed_frames,
            "rejected_frames":self.motion.rejected_frames.len(),
            "interpolated_frames":self.motion.interpolated.iter().filter(|&&v| v).count()
        })
    }
}

pub(crate) fn verified_ball_trajectory(
    project_root: &Path,
    summary: &Value,
    project_id: &str,
    video: &str,
    replay_id: &str,
    replay_sha256: &str,
    frames: u64,
) -> Result<(PathBuf, String), String> {
    let artifact_id = summary["artifact_id"]
        .as_str()
        .filter(|id| opaque_id(id))
        .ok_or("Missing Ball Stabilizer artifact ID")?;
    let expected_sha = summary["trajectory_sha256"]
        .as_str()
        .filter(|sha| volley_core::contracts::sha256(sha))
        .ok_or("Missing canonical ball trajectory SHA")?;
    let version = summary["version"]
        .as_str()
        .ok_or("Missing Ball Stabilizer version")?;
    if summary["schema"] != "volley-ball-stabilizer/1"
        || !matches!(
            version,
            "ball_stabilizer_v1"
                | "ball_stabilizer_v2"
                | "ball_stabilizer_v3"
                | "ball_stabilizer_v4"
                | "ball_stabilizer_v5"
                | "ball_stabilizer_v6"
                | "ball_stabilizer_v7"
        )
        || summary["source_replay_id"] != replay_id
        || summary["source_replay_sha256"] != replay_sha256
        || summary["trajectory_file"] != BALL_TRAJECTORY_FILE
        || summary["frames"].as_u64() != Some(frames)
    {
        return Err("Ball Stabilizer lineage does not match selected replay".into());
    }
    let root = project_root.join("native_cpu_ball_stabilizer");
    safe_folder(&root)?;
    let folder = root.join(artifact_id);
    safe_folder(&folder)?;
    let manifest: Value = serde_json::from_slice(
        &fs::read(safe_file(&folder, "manifest.json", MAX_MANIFEST_BYTES)?)
            .map_err(|e| format!("Read canonical ball manifest: {e}"))?,
    )
    .map_err(|e| format!("Parse canonical ball manifest: {e}"))?;
    if manifest["schema"] != BALL_TRAJECTORY_SCHEMA
        || manifest["state"] != "complete"
        || manifest["project_id"] != project_id
        || manifest["video"] != video
        || manifest["artifact_id"] != artifact_id
        || manifest["source_replay_id"] != replay_id
        || manifest["source_replay_sha256"] != replay_sha256
        || manifest["frames"].as_u64() != Some(frames)
        || (if version == "ball_stabilizer_v1" {
            !manifest["algorithm_version"].is_null()
        } else {
            manifest["algorithm_version"] != version
        })
        || manifest["settings"] != summary["settings"]
        || manifest["file"] != BALL_TRAJECTORY_FILE
        || manifest["sha256"] != expected_sha
        || manifest["confidence_frames"] != summary["confidence_frames"]
        || manifest["observed_frames"] != summary["observed_frames"]
        || manifest["rejected_frames"] != summary["rejected_frames"]
        || manifest["interpolated_frames"] != summary["interpolated_frames"]
    {
        return Err("Canonical Ball Stabilizer manifest mismatch".into());
    }
    let path = safe_file(&folder, BALL_TRAJECTORY_FILE, MAX_INPUT_BYTES)?;
    Ok((path, expected_sha.to_owned()))
}

fn read_canonical_ball_motion(
    path: &Path,
    sha256: &str,
    frames: u64,
    confidence_frames: usize,
    observed_frames: usize,
) -> Result<volley_core::ball_interpolation::StabilizedBallMotion, String> {
    let mut reader = BoundedReader::open(path)?;
    let mut points = Vec::with_capacity(frames as usize);
    let mut interpolated = Vec::with_capacity(frames as usize);
    let mut rejected_frames = Vec::new();
    for index in 0..frames {
        let line = reader
            .line()?
            .ok_or("Canonical ball trajectory truncated")?;
        let row: CanonicalBallFrame = serde_json::from_slice(&line)
            .map_err(|e| format!("Invalid canonical ball frame: {e}"))?;
        row.validate(index)?;
        points.push(row.ball);
        interpolated.push(row.interpolated);
        if row.rejected {
            rejected_frames.push(index as usize);
        }
    }
    if reader.line()?.is_some() || reader.finish() != sha256 {
        return Err("Canonical Ball Stabilizer SHA or frame count mismatch".into());
    }
    let motion = volley_core::ball_interpolation::StabilizedBallMotion {
        points,
        interpolated,
        rejected_frames,
        confidence_frames,
        observed_frames,
    };
    if motion.observed_frames
        != motion
            .points
            .iter()
            .zip(&motion.interpolated)
            .filter(|(point, inferred)| point.is_some() && !**inferred)
            .count()
    {
        return Err("Canonical Ball Stabilizer observed count mismatch".into());
    }
    Ok(motion)
}

const PRIMARY_TOUCH_BREAKS_FILE: &str = "touch_detector.json";
const PRIMARY_TOUCH_BREAKS_MAX_BYTES: u64 = 64 * 1024 * 1024;

pub(crate) fn write_primary_touch_breaks(
    folder: &Path,
    motion: &volley_core::ball_interpolation::StabilizedBallMotion,
    dimensions: &Value,
    cancel: &AtomicBool,
) -> Result<String, String> {
    use volley_core::motion::{trajectory_direction_breaks, BallObservation, DirectionBreakConfig};

    let dimension = |key: &str| {
        dimensions[key]
            .as_u64()
            .filter(|size| (64..=16384).contains(size))
            .ok_or("Invalid primary Touch Detector video dimensions")
    };
    let (width, height) = (dimension("width")?, dimension("height")?);
    let mut breaks = Vec::new();
    for start in (0..motion.points.len()).step_by(CHUNK) {
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let end = (start + CHUNK).min(motion.points.len());
        let first = start.saturating_sub(HALO);
        let last = (end + HALO).min(motion.points.len());
        let points = motion.points[first..last]
            .iter()
            .enumerate()
            .filter_map(|(offset, point)| {
                let [x, y] = (*point)?;
                Some(BallObservation {
                    frame: (first + offset) as i64,
                    x: x * width as f64,
                    y: y * height as f64,
                })
            })
            .collect::<Vec<_>>();
        breaks.extend(
            trajectory_direction_breaks(&points, DirectionBreakConfig::default())
                .into_iter()
                .filter(|impact| (start as i64..end as i64).contains(&impact.frame)),
        );
    }
    let file = folder.join(PRIMARY_TOUCH_BREAKS_FILE);
    storage::write_json_atomic(
        &file,
        &json!({
            "width":width,"height":height,"frames":motion.points.len(),"breaks":breaks
        }),
    )
    .map_err(|e| format!("Save Touch Detector output: {e:?}"))?;
    let bytes = fs::read(&file).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

fn read_primary_touch_breaks(
    project: &Path,
    replay: &Value,
    width: u32,
    height: u32,
    frames: u64,
) -> Result<Option<Vec<volley_core::motion::TrajectoryBreak>>, String> {
    let Some(job) = replay["player_frame_scope_ball_job_id"]
        .as_str()
        .filter(|id| opaque_id(id))
    else {
        return Ok(None);
    };
    let folder = project.join("native_gpu_runs").join(job);
    let mask = fs::read(safe_file(
        &folder,
        "rally_mask.json",
        PRIMARY_TOUCH_BREAKS_MAX_BYTES,
    )?)
    .map_err(|e| e.to_string())?;
    if replay["player_frame_scope_sha256"] != format!("{:x}", Sha256::digest(&mask)) {
        return Err("Primary rally mask changed before Touch Detector reuse".into());
    }
    let evidence: Value = serde_json::from_slice(&mask).map_err(|e| e.to_string())?;
    let checksum = &evidence["mask"]["touch_detector_sha256"];
    if checksum.is_null() {
        return Ok(None);
    }
    let bytes = fs::read(safe_file(
        &folder,
        PRIMARY_TOUCH_BREAKS_FILE,
        PRIMARY_TOUCH_BREAKS_MAX_BYTES,
    )?)
    .map_err(|e| e.to_string())?;
    if *checksum != format!("{:x}", Sha256::digest(&bytes)) {
        return Err("Primary Touch Detector SHA mismatch".into());
    }
    let mut artifact: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if artifact["width"] != width
        || artifact["height"] != height
        || !artifact["frames"]
            .as_u64()
            .is_some_and(|count| (frames..=frames + 8).contains(&count))
    {
        return Err("Primary Touch Detector dimensions or frame count mismatch".into());
    }
    let impacts: Vec<volley_core::motion::TrajectoryBreak> =
        serde_json::from_value(artifact["breaks"].take()).map_err(|e| e.to_string())?;
    if impacts
        .windows(2)
        .any(|pair| pair[0].frame >= pair[1].frame)
    {
        return Err("Primary Touch Detector frames out of order".into());
    }
    Ok(Some(impacts))
}

pub(crate) fn write_ball_trajectory(
    folder: &Path,
    motion: &volley_core::ball_interpolation::StabilizedBallMotion,
    cancel: &AtomicBool,
) -> Result<String, String> {
    let part = folder.join("trajectory.jsonl.partial");
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&part)
        .map_err(|e| format!("Create canonical ball trajectory: {e}"))?;
    let mut writer = BufWriter::new(output);
    let mut digest = Sha256::new();
    let rejected: std::collections::HashSet<usize> =
        motion.rejected_frames.iter().copied().collect();
    for (index, point) in motion.points.iter().enumerate() {
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let row = CanonicalBallFrame {
            frame: index as u64,
            ball: *point,
            interpolated: motion.interpolated[index],
            rejected: rejected.contains(&index),
        };
        row.validate(index as u64)?;
        let mut line = serde_json::to_vec(&row).map_err(|e| e.to_string())?;
        line.push(b'\n');
        writer
            .write_all(&line)
            .map_err(|e| format!("Write canonical ball frame: {e}"))?;
        digest.update(&line);
    }
    writer
        .flush()
        .map_err(|e| format!("Flush canonical ball trajectory: {e}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| format!("Sync canonical ball trajectory: {e}"))?;
    drop(writer);
    let checksum = format!("{:x}", digest.finalize());
    fs::rename(&part, folder.join(BALL_TRAJECTORY_FILE))
        .map_err(|e| format!("Publish canonical ball trajectory: {e}"))?;
    Ok(checksum)
}

fn persist_stabilized_replay(
    project: &Path,
    source: (&str, &str, &str),
    sha: &str,
    settings: volley_core::ball_interpolation::BallStabilizerSettings,
    motion: &volley_core::ball_interpolation::StabilizedBallMotion,
    cancel: &AtomicBool,
) -> Result<(String, String), String> {
    let (project_id, video, replay_id) = source;
    let root = project.join("native_cpu_ball_stabilizer");
    fs::create_dir_all(&root).map_err(|e| format!("Create Ball Stabilizer root: {e}"))?;
    safe_folder(&root)?;
    let artifact_id = Uuid::new_v4().simple().to_string();
    let folder = root.join(&artifact_id);
    fs::create_dir(&folder).map_err(|e| format!("Create Ball Stabilizer artifact: {e}"))?;
    let result = (|| -> Result<String, String> {
        let checksum = write_ball_trajectory(&folder, motion, cancel)?;
        let manifest = json!({
            "schema":BALL_TRAJECTORY_SCHEMA,
            "algorithm_version":volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION,
            "state":"complete",
            "project_id":project_id,"video":video,"artifact_id":artifact_id,
            "source_replay_id":replay_id,"source_replay_sha256":sha,
            "settings":settings,"frames":motion.points.len(),
            "file":BALL_TRAJECTORY_FILE,"sha256":checksum,
            "confidence_frames":motion.confidence_frames,
            "observed_frames":motion.observed_frames,
            "rejected_frames":motion.rejected_frames.len(),
            "interpolated_frames":motion.interpolated.iter().filter(|&&v|v).count()
        });
        storage::write_json_atomic(&folder.join("manifest.json"), &manifest)
            .map_err(|e| format!("Write canonical Ball Stabilizer manifest: {e:?}"))?;
        Ok(checksum)
    })();
    match result {
        Ok(sha) => Ok((artifact_id, sha)),
        Err(error) => {
            let _ = fs::remove_dir_all(&folder);
            Err(error)
        }
    }
}

fn load_primary_ball_trajectory(
    project: &Path,
    replay: &Value,
    frames: u64,
) -> Result<
    Option<(
        volley_core::ball_interpolation::StabilizedBallMotion,
        volley_core::ball_interpolation::BallStabilizerSettings,
    )>,
    String,
> {
    let Some(job) = replay["player_frame_scope_ball_job_id"].as_str() else {
        return Ok(None);
    };
    if !opaque_id(job) {
        return Err("Invalid primary Ball Stabilizer source ID".into());
    }
    let folder = project.join("native_gpu_runs").join(job);
    let mask = safe_file(&folder, "rally_mask.json", MAX_MANIFEST_BYTES)?;
    let bytes = fs::read(mask).map_err(|e| e.to_string())?;
    if replay["player_frame_scope_sha256"] != format!("{:x}", Sha256::digest(&bytes)) {
        return Err("Primary rally mask changed before Ball Stabilizer reuse".into());
    }
    let evidence: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let source = &evidence["mask"]["ball_stabilizer"];
    if source.is_null() {
        return Ok(None);
    }
    if source["algorithm"] != volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION
        || evidence["ball_job_id"] != job
    {
        return Err("Incompatible primary Ball Stabilizer artifact".into());
    }
    let file_sha = source["trajectory_sha256"]
        .as_str()
        .filter(|sha| volley_core::contracts::sha256(sha))
        .ok_or("Missing primary Ball Stabilizer trajectory checksum")?;
    let source_frames = evidence["mask"]["source_frame_count"]
        .as_u64()
        .filter(|count| (frames..=frames + 8).contains(count))
        .ok_or("Primary Ball Stabilizer frame count mismatch")?;
    let count = |key: &str| {
        source[key]
            .as_u64()
            .filter(|value| *value <= source_frames)
            .map(|value| value as usize)
            .ok_or("Invalid primary Ball Stabilizer frame statistics")
    };
    let settings =
        serde_json::from_value::<volley_core::ball_interpolation::BallStabilizerSettings>(
            source["settings"].clone(),
        )
        .map_err(|e| e.to_string())?;
    settings.validate()?;
    let trajectory = safe_file(&folder, BALL_TRAJECTORY_FILE, MAX_INPUT_BYTES)?;
    let mut motion = read_canonical_ball_motion(
        &trajectory,
        file_sha,
        source_frames,
        count("confidence_frames")?,
        count("observed_frames")?,
    )?;
    motion.points.truncate(frames as usize);
    motion.interpolated.truncate(frames as usize);
    motion
        .rejected_frames
        .retain(|&frame| frame < frames as usize);
    motion.observed_frames = motion
        .points
        .iter()
        .zip(&motion.interpolated)
        .filter(|(point, interpolated)| point.is_some() && !**interpolated)
        .count();
    motion.confidence_frames = motion.observed_frames + motion.rejected_frames.len();
    Ok(Some((motion, settings)))
}

pub(crate) fn stabilize_primary_replay(
    state: &AppState,
    project_id: &str,
    video: &str,
    replay_id: &str,
    settings: volley_core::ball_interpolation::BallStabilizerSettings,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, u64) -> Result<(), String>,
) -> Result<StabilizedReplay, String> {
    settings.validate().map_err(str::to_owned)?;
    if !state.writes_enabled {
        return Err("Ball Stabilizer needs an isolated writable runtime".into());
    }
    storage::project_id(project_id).map_err(|e| format!("{e:?}"))?;
    storage::video_name(video).map_err(|e| format!("{e:?}"))?;
    if !opaque_id(replay_id) {
        return Err("Invalid CPU replay ID".into());
    }
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let project = projects::project_root(state).join(project_id);
    safe_folder(&project)?;
    let replay = project.join("native_cpu_replays").join(replay_id);
    safe_folder(&replay)?;
    let manifest: Value = serde_json::from_slice(
        &fs::read(safe_file(&replay, "manifest.json", MAX_MANIFEST_BYTES)?)
            .map_err(|e| format!("Read CPU replay manifest: {e}"))?,
    )
    .map_err(|e| format!("Parse CPU replay manifest: {e}"))?;
    let frames = manifest["frames"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= 48 * 3600 * 30)
        .ok_or("Invalid replay frame count")?;
    let sha = manifest["sha256"]
        .as_str()
        .filter(|v| volley_core::contracts::sha256(v))
        .ok_or("Invalid replay SHA-256")?
        .to_owned();
    if manifest["schema"] != "volley-native-cpu-tracking/1"
        || manifest["state"] != "provisional_tracklets_rebuilt"
        || manifest["project_id"] != project_id
        || manifest["video"] != video
        || manifest["replay_id"] != replay_id
        || manifest["file"] != "tracked_frames.jsonl"
        || manifest["fps"] != 30
        || manifest["touches_confirmed"] != false
        || manifest["track_ids_provisional"] != true
    {
        return Err("Incompatible CPU replay for Ball Stabilizer".into());
    }
    let path = safe_file(&replay, "tracked_frames.jsonl", MAX_INPUT_BYTES)?;
    let (computed, settings) = match load_primary_ball_trajectory(&project, &manifest, frames)? {
        Some((motion, primary_settings)) => (motion, primary_settings),
        None => (
            stabilize_replay_rows(&path, &sha, frames, settings, cancel, &mut progress)?,
            settings,
        ),
    };
    let (artifact_id, trajectory_sha256) = persist_stabilized_replay(
        &project,
        (project_id, video, replay_id),
        &sha,
        settings,
        &computed,
        cancel,
    )?;
    let artifact_path = project
        .join("native_cpu_ball_stabilizer")
        .join(&artifact_id)
        .join(BALL_TRAJECTORY_FILE);
    let motion = read_canonical_ball_motion(
        &artifact_path,
        &trajectory_sha256,
        frames,
        computed.confidence_frames,
        computed.observed_frames,
    )?;
    Ok(StabilizedReplay {
        source_sha256: sha,
        motion,
        settings,
        artifact_id,
        trajectory_sha256,
        frames,
        project_root: project,
        replay_id: replay_id.to_owned(),
    })
}

fn stabilize_replay_rows(
    path: &Path,
    sha: &str,
    frames: u64,
    settings: volley_core::ball_interpolation::BallStabilizerSettings,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, u64) -> Result<(), String>,
) -> Result<volley_core::ball_interpolation::StabilizedBallMotion, String> {
    let mut reader = BoundedReader::open(path)?;
    let mut samples = Vec::new();
    progress(0, frames)?;
    for index in 0..frames {
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let line = reader
            .line()?
            .ok_or("Truncated CPU replay for Ball Stabilizer")?;
        let row: TrackedFrame = serde_json::from_slice(&line)
            .map_err(|e| format!("Invalid CPU replay frame for stabilizer: {e}"))?;
        let frame = transform_frame(row, index)?;
        if let Some(ball) = frame.selected_ball {
            let [x1, y1, x2, y2] = ball.bbox_xyxy_norm;
            samples.push(volley_core::ball_interpolation::ScoredBallSample {
                frame: index as usize,
                point: [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
                confidence: ball.score as f32,
            });
        }
        if (index + 1) % 4096 == 0 {
            progress(index + 1, frames)?;
        }
    }
    if reader.line()?.is_some() || reader.finish() != sha {
        return Err("Changed or extra CPU replay frames during stabilization".into());
    }
    let motion =
        volley_core::ball_interpolation::stabilize_ball_motion(&samples, frames as usize, settings)
            .map_err(str::to_owned)?;
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    progress(frames, frames)?;
    Ok(motion)
}

/// Rewrite only the derived TouchFrame working copy. Synthetic observations
/// cannot themselves confirm a contact; the emitter checks their provenance.
fn use_stabilized_ball(
    block: &mut [TouchFrame],
    motion: &volley_core::ball_interpolation::StabilizedBallMotion,
) {
    for frame in block {
        let index = frame.frame_index as usize;
        let Some([x, y]) = motion.points.get(index).copied().flatten() else {
            frame.selected_ball = None;
            continue;
        };
        if !motion.interpolated[index] && frame.selected_ball.is_none() {
            continue;
        }
        let (half_w, half_h) = frame.selected_ball.as_ref().map_or((0.005, 0.005), |ball| {
            let [x1, y1, x2, y2] = ball.bbox_xyxy_norm;
            ((x2 - x1) * 0.5, (y2 - y1) * 0.5)
        });
        let observed = !motion.interpolated[index];
        frame.selected_ball = Some(TouchBall {
            source_model: if observed {
                "vballnet"
            } else {
                volley_core::touch_pipeline::INTERPOLATED_BALL_SOURCE
            }
            .into(),
            score: if observed {
                frame.selected_ball.as_ref().unwrap().score
            } else {
                f64::from(0.55f32)
            },
            bbox_xyxy_norm: [
                (x - half_w).max(0.0),
                (y - half_h).max(0.0),
                (x + half_w).min(1.0),
                (y + half_h).min(1.0),
            ],
        });
    }
}

#[cfg(test)]
mod ball_stabilizer_postprocessing_tests {
    use super::*;

    #[test]
    fn primary_trajectory_reuse_checks_both_hashes_and_aligns_the_replay_tail() {
        let project = std::env::temp_dir().join(format!("volley-primary-ball-{}", Uuid::new_v4()));
        let job = Uuid::new_v4().simple().to_string();
        let folder = project.join("native_gpu_runs").join(&job);
        fs::create_dir_all(&folder).unwrap();
        let motion = volley_core::ball_interpolation::StabilizedBallMotion {
            points: vec![Some([0.2, 0.5]), Some([0.3, 0.5]), Some([0.4, 0.5])],
            interpolated: vec![false, true, false],
            rejected_frames: vec![],
            confidence_frames: 2,
            observed_frames: 2,
        };
        let checksum = write_ball_trajectory(&folder, &motion, &AtomicBool::new(false)).unwrap();
        let evidence = serde_json::to_vec(&json!({
            "ball_job_id": job,
            "mask": {
                "source_frame_count": 3,
                "ball_stabilizer": {
                    "algorithm": volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION,
                    "settings": volley_core::ball_interpolation::BallStabilizerSettings::default(),
                    "trajectory_sha256": checksum,
                    "confidence_frames": 2,
                    "observed_frames": 2,
                }
            },
        }))
        .unwrap();
        fs::write(folder.join("rally_mask.json"), &evidence).unwrap();
        let mut replay = json!({
            "player_frame_scope_ball_job_id": job,
            "player_frame_scope_sha256": format!("{:x}", Sha256::digest(&evidence)),
        });
        let (reused, _) = load_primary_ball_trajectory(&project, &replay, 2)
            .unwrap()
            .unwrap();
        assert_eq!(reused.points, motion.points[..2]);
        assert_eq!(reused.interpolated, [false, true]);
        assert_eq!(reused.observed_frames, 1);
        fs::write(folder.join(BALL_TRAJECTORY_FILE), b"invalid").unwrap();
        assert!(load_primary_ball_trajectory(&project, &replay, 2).is_err());
        replay["player_frame_scope_sha256"] = json!("invalid");
        assert_eq!(
            load_primary_ball_trajectory(&project, &replay, 2).unwrap_err(),
            "Primary rally mask changed before Ball Stabilizer reuse"
        );
        fs::remove_dir_all(project).unwrap();
    }

    #[test]
    fn stabilized_cpu_replay_filters_false_ball_interpolates_gap_and_preserves_sha() {
        let root = std::env::temp_dir().join(format!("volley-stabilizer-cpu-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("tracked_frames.jsonl");
        let mut bytes = Vec::new();
        let mut block = Vec::new();
        for index in 0..63_u64 {
            let x = if index == 20 || (42..=49).contains(&index) {
                0.92
            } else {
                0.2 + index as f64 * 0.003
            };
            let score = if index == 6 { 0.49 } else { 0.93 };
            let ball = if (31..=33).contains(&index) {
                None
            } else {
                Some(json!({"source_model":"vballnet","detection":{
                    "frame_index":index,"class_name":"ball","score":score,
                    "bbox_xyxy_norm":[x-0.005,0.4,x+0.005,0.41]
                }}))
            };
            let row = json!({"frame_index":index,"tracked_players":[],
                "selected_ball":ball,"track_ids_provisional":true,
                "confirmed_athlete_identities":false,"touches_confirmed":false});
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
            block.push(TouchFrame {
                frame_index: index,
                players: Vec::new(),
                selected_ball: ball.map(|v| TouchBall {
                    source_model: "vballnet".into(),
                    score: v["detection"]["score"].as_f64().unwrap(),
                    bbox_xyxy_norm: serde_json::from_value(
                        v["detection"]["bbox_xyxy_norm"].clone(),
                    )
                    .unwrap(),
                }),
            });
        }
        fs::write(&path, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let mut reported = Vec::new();
        let stopped = AtomicBool::new(false);
        let motion =
            stabilize_replay_rows(&path, &sha, 63, Default::default(), &stopped, |a, b| {
                reported.push((a, b));
                Ok(())
            })
            .unwrap();
        assert_eq!(reported.first(), Some(&(0, 63)));
        assert_eq!(reported.last(), Some(&(63, 63)));
        assert!(motion.rejected_frames.contains(&20));
        assert!(motion.interpolated[20]);
        for frame in 42..=49 {
            assert!(motion.rejected_frames.contains(&frame));
            assert!(motion.interpolated[frame]);
        }
        for i in 31..=33 {
            assert!(motion.interpolated[i]);
        }
        assert!(motion.points[6].is_some());
        assert!(!motion.interpolated[6]);
        let original = block.clone();
        let mut canonical = motion.clone();
        canonical.points[6] = Some([0.15, 0.25]);
        use_stabilized_ball(&mut block, &canonical);
        let accepted = block[6].selected_ball.as_ref().unwrap();
        let center_x = (accepted.bbox_xyxy_norm[0] + accepted.bbox_xyxy_norm[2]) * 0.5;
        assert!((center_x - 0.15).abs() < 1e-9);
        assert_eq!(accepted.source_model, "vballnet");
        assert_eq!(
            block[20].selected_ball.as_ref().unwrap().source_model,
            "ball_stabilizer_interpolated"
        );
        assert_eq!(
            block[32].selected_ball.as_ref().unwrap().source_model,
            "ball_stabilizer_interpolated"
        );
        assert_eq!(block[6].selected_ball.as_ref().unwrap().score, 0.49);
        assert!(
            (original[20].selected_ball.as_ref().unwrap().bbox_xyxy_norm[0] - 0.915).abs() < 1e-9
        );
        assert!(block[20].selected_ball.as_ref().unwrap().bbox_xyxy_norm[0] < 0.4);
        assert!(block[32].selected_ball.is_some());
        assert!(original[32].selected_ball.is_none());
        assert_eq!(
            original[24].selected_ball.as_ref().unwrap().score,
            block[24].selected_ball.as_ref().unwrap().score
        );
        let still_raw = fs::read(&path).unwrap();
        assert_eq!(
            still_raw, bytes,
            "ball stabilizer must not mutate the replay"
        );
        let mut changed = bytes;
        changed[25] = if changed[25] == b'a' { b'b' } else { b'a' };
        fs::write(&path, changed).unwrap();
        assert!(stabilize_replay_rows(
            &path,
            &sha,
            63,
            Default::default(),
            &stopped,
            |_, _| Ok(())
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn canonical_trajectory_is_immutable_sha_verified_and_source_bound() {
        let root = std::env::temp_dir().join(format!("volley-canonical-ball-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let samples = (0..32usize)
            .filter(|&i| i != 11 && i != 12)
            .map(|i| volley_core::ball_interpolation::ScoredBallSample {
                frame: i,
                point: [0.3 + i as f64 * 0.002, 0.4],
                confidence: 0.95,
            })
            .collect::<Vec<_>>();
        let settings = volley_core::ball_interpolation::BallStabilizerSettings::default();
        let motion =
            volley_core::ball_interpolation::stabilize_ball_motion(&samples, 32, settings).unwrap();
        assert!(motion.interpolated[11]);
        let cancel = AtomicBool::new(false);
        let replay_sha = "a".repeat(64);
        let (id, digest) = persist_stabilized_replay(
            &root,
            ("abc123def456", "match.mp4", &"d".repeat(32)),
            &replay_sha,
            settings,
            &motion,
            &cancel,
        )
        .unwrap();
        let summary = json!({
             "schema":"volley-ball-stabilizer/1","version":volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION,
            "source_replay_id":"d".repeat(32),"source_replay_sha256":replay_sha,
            "artifact_id":id,"trajectory_file":BALL_TRAJECTORY_FILE,
            "trajectory_sha256":digest,"frames":32,"settings":settings,
            "confidence_frames":motion.confidence_frames,"observed_frames":motion.observed_frames,
            "rejected_frames":motion.rejected_frames.len(),
            "interpolated_frames":motion.interpolated.iter().filter(|&&v|v).count()
        });
        let (path, sha) = verified_ball_trajectory(
            &root,
            &summary,
            "abc123def456",
            "match.mp4",
            &"d".repeat(32),
            &replay_sha,
            32,
        )
        .unwrap();
        let restored = read_canonical_ball_motion(
            &path,
            &sha,
            32,
            motion.confidence_frames,
            motion.observed_frames,
        )
        .unwrap();
        assert_eq!(restored.points, motion.points);
        assert_eq!(restored.interpolated, motion.interpolated);
        let manifest_path = root
            .join("native_cpu_ball_stabilizer")
            .join(&id)
            .join("manifest.json");
        let published_manifest: Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        let mut previous_manifest = published_manifest.clone();
        previous_manifest
            .as_object_mut()
            .unwrap()
            .remove("algorithm_version");
        storage::write_json_atomic(&manifest_path, &previous_manifest).unwrap();
        let mut previous_summary = summary.clone();
        previous_summary["version"] = json!("ball_stabilizer_v1");
        assert!(verified_ball_trajectory(
            &root,
            &previous_summary,
            "abc123def456",
            "match.mp4",
            &"d".repeat(32),
            &replay_sha,
            32,
        )
        .is_ok());
        assert!(verified_ball_trajectory(
            &root,
            &summary,
            "abc123def456",
            "match.mp4",
            &"d".repeat(32),
            &replay_sha,
            32,
        )
        .is_err());
        storage::write_json_atomic(&manifest_path, &published_manifest).unwrap();
        let mut v2_manifest = published_manifest.clone();
        v2_manifest["algorithm_version"] = json!("ball_stabilizer_v2");
        storage::write_json_atomic(&manifest_path, &v2_manifest).unwrap();
        let mut v2_summary = summary.clone();
        v2_summary["version"] = json!("ball_stabilizer_v2");
        assert!(verified_ball_trajectory(
            &root,
            &v2_summary,
            "abc123def456",
            "match.mp4",
            &"d".repeat(32),
            &replay_sha,
            32,
        )
        .is_ok());
        assert!(verified_ball_trajectory(
            &root,
            &summary,
            "abc123def456",
            "match.mp4",
            &"d".repeat(32),
            &replay_sha,
            32,
        )
        .is_err());
        storage::write_json_atomic(&manifest_path, &published_manifest).unwrap();
        let mut v3_manifest = published_manifest.clone();
        v3_manifest["algorithm_version"] = json!("ball_stabilizer_v3");
        storage::write_json_atomic(&manifest_path, &v3_manifest).unwrap();
        let mut v3_summary = summary.clone();
        v3_summary["version"] = json!("ball_stabilizer_v3");
        assert!(verified_ball_trajectory(
            &root,
            &v3_summary,
            "abc123def456",
            "match.mp4",
            &"d".repeat(32),
            &replay_sha,
            32,
        )
        .is_ok());
        assert!(verified_ball_trajectory(
            &root,
            &summary,
            "abc123def456",
            "match.mp4",
            &"d".repeat(32),
            &replay_sha,
            32,
        )
        .is_err());
        for version in ["ball_stabilizer_v4", "ball_stabilizer_v5"] {
            let mut older_manifest = published_manifest.clone();
            older_manifest["algorithm_version"] = json!(version);
            storage::write_json_atomic(&manifest_path, &older_manifest).unwrap();
            let mut older_summary = summary.clone();
            older_summary["version"] = json!(version);
            assert!(verified_ball_trajectory(
                &root,
                &older_summary,
                "abc123def456",
                "match.mp4",
                &"d".repeat(32),
                &replay_sha,
                32,
            )
            .is_ok());
        }
        storage::write_json_atomic(&manifest_path, &published_manifest).unwrap();
        assert!(verified_ball_trajectory(
            &root,
            &summary,
            "abc123def456",
            "other.mp4",
            &"d".repeat(32),
            &replay_sha,
            32,
        )
        .is_err());
        let original = fs::read(&path).unwrap();
        let mut tampered = original.clone();
        tampered[2] = if tampered[2] == b'a' { b'b' } else { b'a' };
        fs::write(&path, tampered).unwrap();
        assert!(read_canonical_ball_motion(
            &path,
            &sha,
            32,
            motion.confidence_frames,
            motion.observed_frames,
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }
}

/// Postprocessing-only confidence gate. Never changes raw GPU/replay artifacts.
fn gate_ball_confidence(frames: &mut [TouchFrame], min_score: f64) {
    for frame in frames {
        if frame
            .selected_ball
            .as_ref()
            .is_some_and(|ball| ball.source_model != "vballnet" || ball.score < min_score)
        {
            frame.selected_ball = None;
        }
    }
}

#[cfg(test)]
mod ball_confidence_tests {
    use super::*;

    #[test]
    fn postprocessing_removes_every_vballnet_observation_below_67_percent() {
        let mut frames = [0.08, 0.51, 0.66, 0.67, 0.81]
            .into_iter()
            .enumerate()
            .map(|(frame, score)| TouchFrame {
                frame_index: frame as u64,
                players: Vec::new(),
                selected_ball: Some(TouchBall {
                    source_model: "vballnet".into(),
                    score,
                    bbox_xyxy_norm: [0.3, 0.3, 0.32, 0.32],
                }),
            })
            .collect::<Vec<_>>();
        let raw_copy = frames.clone();
        gate_ball_confidence(&mut frames, 0.67);
        assert_eq!(
            frames
                .iter()
                .map(|f| f.selected_ball.is_some())
                .collect::<Vec<_>>(),
            vec![false, false, false, true, true]
        );
        assert!(raw_copy.iter().all(|f| f.selected_ball.is_some()));
    }
}

fn strong_occluded_launch(event: &PhysicalTouchCandidate) -> bool {
    event.physical_motion_supported
        && event.provisional_track_id.is_some()
        && event.ball_occluded_at_contact
        && event.inferred_body_reach_supported
        && !event.body_contact_supported
        && event.contact_landmark.is_none()
        && event.motion_frame_index == event.frame_index
        && event.probability.is_finite()
        && event.probability >= 0.65
        && event.speed_before_px_frame.is_finite()
        && (0.0..3.0).contains(&event.speed_before_px_frame)
        && event.speed_after_px_frame.is_finite()
        && (14.0..=60.0).contains(&event.speed_after_px_frame)
        && event.velocity_residual_px_frame.is_finite()
        && event.velocity_residual_px_frame >= 14.0
}

fn retain_refined_pose_candidates(events: &mut Vec<PhysicalTouchCandidate>, supported: &[bool]) {
    let mut flags = supported.iter().copied();
    events.retain(|event| flags.next().unwrap_or(false) || strong_occluded_launch(event));
}

#[cfg(test)]
fn retain_pose_confirmed_events(events: &mut Vec<PhysicalTouchCandidate>) {
    events.retain(|event| {
        event.physical_motion_supported
            && event.provisional_track_id.is_some()
            && (event.body_contact_supported && event.contact_landmark.is_some()
                || strong_occluded_launch(event))
    });
}

fn is_launch_departure_echo(
    launch: &PhysicalTouchCandidate,
    departure: &PhysicalTouchCandidate,
    separation: u64,
) -> bool {
    let elapsed = departure.frame_index.saturating_sub(launch.frame_index);
    strong_occluded_launch(launch)
        && departure.frame_index >= launch.frame_index
        && (1..=separation).contains(&elapsed)
        && departure.physical_motion_supported
        && departure.direction_change_deg.is_finite()
        && departure.direction_change_deg < 25.0
        && departure.speed_before_px_frame.is_finite()
        && departure.speed_after_px_frame.is_finite()
        && departure.speed_after_px_frame > departure.speed_before_px_frame + 8.0
        && departure.speed_before_px_frame <= launch.speed_after_px_frame
        && departure.ball_x_px.is_finite()
        && departure.ball_y_px.is_finite()
        && (departure.ball_x_px - launch.ball_x_px).hypot(departure.ball_y_px - launch.ball_y_px)
            <= 16.0 + launch.speed_after_px_frame * elapsed as f64 * 1.45
}

fn source_frame_contains_track(frames: &[TouchFrame], frame_index: u64, track_id: u64) -> bool {
    let Some(first) = frames.first() else {
        return false;
    };
    frame_index
        .checked_sub(first.frame_index)
        .and_then(|offset| usize::try_from(offset).ok())
        .and_then(|offset| frames.get(offset))
        .is_some_and(|frame| {
            frame.frame_index == frame_index
                && frame
                    .players
                    .iter()
                    .any(|player| player.track_id == track_id)
        })
}

#[cfg(test)]
mod source_track_integrity_tests {
    use super::*;

    #[test]
    fn near_hand_overrides_other_players_foot_in_pending_selection() {
        let contact =
            |joint, distance, normalized_distance| volley_core::pose_contact::BodyContact {
                joint_indices: [joint, joint],
                point_px: [320.0 + distance, 144.0],
                distance_px: distance,
                normalized_distance,
            };
        let event = |id, probability, landmark| PhysicalTouchCandidate {
            frame_index: 1086,
            provisional_track_id: Some(id),
            probability,
            ball_x_px: 320.0,
            ball_y_px: 144.0,
            motion_frame_index: 1086,
            direction_change_deg: 90.0,
            speed_before_px_frame: 10.0,
            speed_after_px_frame: 13.0,
            velocity_residual_px_frame: 25.0,
            technique: "unknown_touch",
            action_prediction: None,
            court: None,
            physical_motion_supported: true,
            body_contact_supported: true,
            contact_landmark: Some(landmark),
            contact_pose_sample_frame: None,
            ball_occluded_at_contact: false,
            inferred_body_reach_supported: false,
            athlete_identity_verified: false,
        };
        let foot = event(17, 0.90, contact(15, 1.0, 0.0));
        let hand = event(42, 0.65, contact(9, 9.0, 0.02));
        let mut pending = None;
        assert!(select_pending_touch(&mut pending, foot, 3).is_none());
        assert!(select_pending_touch(&mut pending, hand, 3).is_none());
        assert_eq!(pending.unwrap().provisional_track_id, Some(42));
    }

    fn occluded_launch(frame: u64) -> PhysicalTouchCandidate {
        PhysicalTouchCandidate {
            frame_index: frame,
            provisional_track_id: Some(12),
            probability: 0.82,
            ball_x_px: 637.5,
            ball_y_px: 377.5,
            motion_frame_index: frame,
            direction_change_deg: 0.0,
            speed_before_px_frame: 0.7,
            speed_after_px_frame: 19.2,
            velocity_residual_px_frame: 19.6,
            technique: "unknown_touch",
            action_prediction: None,
            court: None,
            physical_motion_supported: true,
            body_contact_supported: false,
            contact_landmark: None,
            contact_pose_sample_frame: None,
            ball_occluded_at_contact: true,
            inferred_body_reach_supported: true,
            athlete_identity_verified: false,
        }
    }

    #[test]
    fn pose_refinement_keeps_only_pose_support_or_strong_occluded_impact() {
        let launch = occluded_launch(651);
        let mut weak = occluded_launch(652);
        weak.ball_occluded_at_contact = false;
        let mut supported = occluded_launch(660);
        supported.ball_occluded_at_contact = false;
        supported.inferred_body_reach_supported = false;
        supported.body_contact_supported = true;
        supported.contact_landmark = Some(volley_core::pose_contact::BodyContact {
            joint_indices: [9, 9],
            point_px: [637.5, 322.5],
            distance_px: 5.0,
            normalized_distance: 0.04,
        });
        let mut events = vec![launch, weak, supported];
        retain_refined_pose_candidates(&mut events, &[false, false, true]);
        assert_eq!(
            events
                .iter()
                .map(|event| event.frame_index)
                .collect::<Vec<_>>(),
            vec![651, 660]
        );
    }

    #[test]
    fn inferred_impact_with_real_stable_launch_survives_pose_requirement() {
        let launch = occluded_launch(651);
        let mut events = vec![launch.clone()];
        retain_pose_confirmed_events(&mut events);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].frame_index, 651);
        assert!(!events[0].body_contact_supported);
        for change in 0..7 {
            let mut weaker = launch.clone();
            match change {
                0 => weaker.ball_occluded_at_contact = false,
                1 => weaker.inferred_body_reach_supported = false,
                2 => weaker.speed_before_px_frame = 4.0,
                3 => weaker.speed_after_px_frame = 8.0,
                4 => weaker.velocity_residual_px_frame = 5.0,
                5 => weaker.provisional_track_id = None,
                _ => weaker.physical_motion_supported = false,
            }
            let mut results = vec![weaker];
            retain_pose_confirmed_events(&mut results);
            assert!(results.is_empty(), "invalid launch variant {change}");
        }
    }

    #[test]
    fn next_frame_pose_departure_does_not_replace_occluded_launch_time() {
        let launch = occluded_launch(651);
        let mut departure = occluded_launch(654);
        departure.ball_y_px = 322.5;
        departure.ball_occluded_at_contact = false;
        departure.inferred_body_reach_supported = false;
        departure.body_contact_supported = true;
        departure.contact_landmark = Some(volley_core::pose_contact::BodyContact {
            joint_indices: [9, 9],
            point_px: [637.5, 322.5],
            distance_px: 5.0,
            normalized_distance: 0.04,
        });
        departure.direction_change_deg = 7.8;
        departure.speed_before_px_frame = 8.5;
        departure.speed_after_px_frame = 31.1;
        departure.velocity_residual_px_frame = 31.3;
        departure.probability = 0.77;
        let mut pending = None;
        assert!(select_pending_touch(&mut pending, launch, 3).is_none());
        assert!(select_pending_touch(&mut pending, departure, 3).is_none());
        assert_eq!(pending.as_ref().unwrap().frame_index, 651);
        let independent = occluded_launch(660);
        assert_eq!(
            select_pending_touch(&mut pending, independent, 3)
                .unwrap()
                .frame_index,
            651
        );
    }

    #[test]
    fn reconstructed_player_does_not_authorize_missing_source_track() {
        let mut original = (100..103)
            .map(|frame_index| TouchFrame {
                frame_index,
                players: Vec::new(),
                selected_ball: None,
            })
            .collect::<Vec<_>>();
        let player = TouchPlayer {
            track_id: 17,
            score: 0.9,
            bbox_xyxy_norm: [0.30, 0.20, 0.40, 0.70],
            touch_probability: None,
        };
        original[0].players.push(player.clone());
        original[2].players.push(player);
        let interpolated = reconstruct_track_gaps(&original);
        assert!(interpolated[1].players.iter().any(|p| p.track_id == 17));
        assert!(source_frame_contains_track(&original, 100, 17));
        assert!(source_frame_contains_track(&original, 102, 17));
        assert!(!source_frame_contains_track(&original, 101, 17));
        assert!(!source_frame_contains_track(&original, 100, 18));
        assert!(!source_frame_contains_track(&original, 99, 17));
        assert!(!source_frame_contains_track(&original, 103, 17));
    }
}

fn emit_record(
    writer: &mut BufWriter<std::fs::File>,
    hash: &mut Sha256,
    event: &PhysicalTouchCandidate,
) -> Result<(), String> {
    let mut bytes =
        serde_json::to_vec(event).map_err(|e| format!("Serialize motion-backed touch: {e}"))?;
    bytes.push(b'\n');
    writer
        .write_all(&bytes)
        .map_err(|e| format!("Write CPU touch: {e}"))?;
    hash.update(&bytes);
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct InitialServeProposal {
    rally_start_frame: u64,
    trajectory_start_frame: Option<u64>,
    provisional_track_id: Option<u64>,
    kind: &'static str,
    identity_verified: bool,
    source: &'static str,
}

pub(crate) fn initial_serve_proposal(
    frames: &[TouchFrame],
    rally_start: u64,
) -> InitialServeProposal {
    let first_ball = frames.iter().find(|frame| {
        frame.frame_index >= rally_start
            && frame.frame_index < rally_start.saturating_add(30)
            && frame.selected_ball.is_some()
    });
    let first_position = first_ball
        .and_then(|frame| frame.selected_ball.as_ref())
        .map(|ball| {
            let [x1, y1, x2, y2] = ball.bbox_xyxy_norm;
            [(x1 + x2) * 0.5, (y1 + y2) * 0.5]
        });
    let candidate = first_position.and_then(|position| {
        let mut distances = std::collections::BTreeMap::<u64, f64>::new();
        for frame in frames.iter().filter(|frame| {
            frame.frame_index >= rally_start && frame.frame_index < rally_start.saturating_add(90)
        }) {
            for player in &frame.players {
                let [x1, y1, x2, y2] = player.bbox_xyxy_norm;
                let distance = (position[0] - position[0].clamp(x1, x2))
                    .hypot(position[1] - position[1].clamp(y1, y2));
                distances
                    .entry(player.track_id)
                    .and_modify(|best| *best = best.min(distance))
                    .or_insert(distance);
            }
        }
        let mut ranked = distances.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (id, distance) = *ranked.first()?;
        (distance <= 0.035
            && ranked
                .get(1)
                .is_none_or(|second| second.1 > distance + 0.035))
        .then_some(id)
    });
    InitialServeProposal {
        rally_start_frame: rally_start,
        trajectory_start_frame: first_ball.map(|frame| frame.frame_index),
        provisional_track_id: candidate,
        kind: "serve",
        identity_verified: false,
        source: "rally_trajectory_start_v1",
    }
}
fn select_pending_touch(
    pending: &mut Option<PhysicalTouchCandidate>,
    event: PhysicalTouchCandidate,
    separation: u64,
) -> Option<PhysicalTouchCandidate> {
    if let Some(last) = pending {
        if event.is_motion_echo_of(last) {
            return None;
        }
        if last.is_motion_echo_of(&event) {
            *pending = Some(event);
            return None;
        }
        if is_launch_departure_echo(last, &event, separation) {
            return None;
        }
        if event.frame_index.saturating_sub(last.frame_index) <= separation {
            let better = match (event.contact_landmark, last.contact_landmark) {
                (Some(next), Some(previous))
                    if event.body_contact_supported && last.body_contact_supported =>
                {
                    next.priority_distance() < previous.priority_distance()
                        || (next.priority_distance() == previous.priority_distance()
                            && event.probability > last.probability)
                }
                _ => {
                    (event.body_contact_supported, event.probability)
                        > (last.body_contact_supported, last.probability)
                }
            };
            if better {
                *pending = Some(event);
            }
            return None;
        }
    }
    pending.replace(event)
}

/// RTMPose is read from the same saved GPU fusion as the selected CPU replay.
/// A single-frame pose overlap cannot promote a candidate to a real Touch:
/// require same-Track-ID body intersection before/at/after a motion-gated hit.
#[cfg(test)]
fn refine_pose_events(
    events: &mut Vec<PhysicalTouchCandidate>,
    frames: &[TouchFrame],
    video_width: u32,
    video_height: u32,
    project_root: &Path,
    fusion_id: &str,
    settings: TouchSettings,
) -> Result<(), String> {
    refine_pose_events_policy(
        events,
        frames,
        (video_width, video_height),
        project_root,
        fusion_id,
        settings,
        None,
        &BTreeMap::new(),
    )
}

#[expect(clippy::too_many_arguments)]
fn refine_pose_events_policy(
    events: &mut Vec<PhysicalTouchCandidate>,
    frames: &[TouchFrame],
    dimensions: (u32, u32),
    project_root: &Path,
    fusion_id: &str,
    settings: TouchSettings,
    interpolation: Option<bool>,
    refined: &BTreeMap<u64, Vec<PosePlayer>>,
) -> Result<(), String> {
    let (video_width, video_height) = dimensions;
    if events.is_empty() || frames.is_empty() {
        return Ok(());
    }
    if refined.is_empty() && native_preview_pose::source_info(project_root, fusion_id)?.is_none() {
        if settings.require_pose {
            return Err("Для выбранного режима необходим сохранённый анализ позы".into());
        }
        return Ok(());
    }
    let poses = pose_frames_policy(project_root, fusion_id, frames, interpolation, refined)?;
    let first = frames[0].frame_index;
    let mut supported = vec![false; events.len()];
    for (index, event) in events.iter_mut().enumerate() {
        if !event.physical_motion_supported {
            continue;
        }
        let original_track_id = event.provisional_track_id;
        let Some(frame) = event
            .frame_index
            .checked_sub(first)
            .and_then(|offset| frames.get(usize::try_from(offset).ok()?))
            .filter(|frame| frame.frame_index == event.frame_index)
        else {
            continue;
        };
        let Some(ball) = frame
            .selected_ball
            .as_ref()
            .filter(|ball| ball.source_model == "vballnet")
        else {
            continue;
        };
        let w = video_width as f64;
        let h = video_height as f64;
        let [x1, y1, x2, y2] = ball.bbox_xyxy_norm;
        let center = [(x1 + x2) * 0.5 * w, (y1 + y2) * 0.5 * h];
        let radius = ((x2 - x1) * w + (y2 - y1) * h) * 0.25;
        // A stored event may not borrow a different ball's location.
        if (event.ball_x_px - center[0]).hypot(event.ball_y_px - center[1]) > radius.max(2.0) {
            continue;
        }
        let contact = frame
            .players
            .iter()
            .filter_map(|player| {
                let id = player.track_id;
                pose_contact_at_frame(
                    frames,
                    &poses,
                    frame,
                    id,
                    video_width,
                    video_height,
                    settings,
                )
                .map(|(evidence, point)| (id, evidence, point, None))
                .or_else(|| {
                    if event.ball_occluded_at_contact
                        || event.direction_change_deg < 75.0
                        || event.velocity_residual_px_frame < 18.0
                        || event.motion_frame_index.abs_diff(event.frame_index) > 3
                    {
                        return None;
                    }
                    pose_contact_from_nearby_sample(
                        frames,
                        &poses,
                        frame,
                        id,
                        video_width,
                        video_height,
                        settings,
                    )
                    .map(|(evidence, point, sample_frame)| {
                        (id, evidence, point, Some(sample_frame))
                    })
                })
            })
            .min_by(|a, b| a.2.priority_distance().total_cmp(&b.2.priority_distance()));
        let Some((selected_track, evidence, point, sample_frame)) = contact else {
            continue;
        };
        supported[index] = true;
        event.provisional_track_id = Some(selected_track);
        event.body_contact_supported = true;
        event.contact_landmark = Some(point);
        event.contact_pose_sample_frame = sample_frame;
        event.action_prediction = (Some(selected_track) == original_track_id)
            .then_some(event.action_prediction.as_ref())
            .flatten()
            .and_then(|geometry| {
                let (before, after) =
                    observed_pose_motion(&poses, event.frame_index, selected_track)?;
                pose_classification::refine_with_motion(&evidence, geometry, &before, &after)
            });
    }
    if settings.require_pose {
        retain_refined_pose_candidates(events, &supported);
    }
    Ok(())
}

/// Match manually named motions only after independent same-Track-ID ball/body
/// evidence. Training cannot create touches or invent athlete identities.
#[expect(clippy::too_many_arguments)]
fn apply_training(
    events: &mut [PhysicalTouchCandidate],
    frames: &[TouchFrame],
    pose_source: Option<(&Path, &str, bool)>,
    width: u32,
    height: u32,
    settings: TouchSettings,
    training: &[training_samples::Example],
    refined: &BTreeMap<u64, Vec<PosePlayer>>,
) -> Result<(), String> {
    if events.is_empty() || training.is_empty() {
        return Ok(());
    }
    let Some((root, fusion, interpolation)) = pose_source else {
        return Ok(());
    };
    let poses = pose_frames_policy(root, fusion, frames, Some(interpolation), refined)?;
    let Some(first) = frames.first().map(|f| f.frame_index) else {
        return Ok(());
    };
    for event in events {
        if !event.physical_motion_supported || !event.body_contact_supported {
            continue;
        }
        let Some(track_id) = event.provisional_track_id else {
            continue;
        };
        let Some(frame) = event
            .frame_index
            .checked_sub(first)
            .and_then(|n| frames.get(n as usize))
            .filter(|f| f.frame_index == event.frame_index)
        else {
            continue;
        };
        let Some((evidence, _)) =
            pose_contact_at_frame(frames, &poses, frame, track_id, width, height, settings)
        else {
            continue;
        };
        let Some(geometry) = event.action_prediction.as_ref() else {
            continue;
        };
        let movement = poses
            .iter()
            .filter_map(|sample| {
                if sample.frame_index.abs_diff(event.frame_index) > 6 {
                    return None;
                }
                let pose = sample.players.iter().find(|p| p.track_id == track_id)?;
                let current = sample
                    .frame_index
                    .checked_sub(first)
                    .and_then(|index| frames.get(index as usize))?;
                let player = current.players.iter().find(|p| p.track_id == track_id)?;
                let ball = current
                    .selected_ball
                    .as_ref()
                    .filter(|b| b.source_model == "vballnet")
                    .map(|b| {
                        let bb = b.bbox_xyxy_norm;
                        [(bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5]
                    });
                Some(training_samples::MotionSample {
                    offset: (sample.frame_index as i64 - event.frame_index as i64) as i16,
                    keypoints: pose.keypoints,
                    bbox: player.bbox_xyxy_norm,
                    ball,
                })
            })
            .collect::<Vec<_>>();
        if let Some(prediction) =
            training_samples::predict(&evidence, geometry, training, &movement)
        {
            event.action_prediction = Some(prediction);
        }
    }
    Ok(())
}

/// Bounded 240-frame chunk, 40-frame halos for 5-frame smoothing, motion
/// window and time-cluster NMS. A single pending event deduplicates across
/// chunk seams. Source is hashed in the same pass. Whole matches never
/// require their frame records or raw player boxes to be resident in memory.
#[cfg(test)]
pub(crate) fn detect_stream(
    source: &Path,
    output: &Path,
    expected_sha: &str,
    count: u64,
    width: u32,
    height: u32,
    settings: TouchSettings,
) -> Result<(String, u64, u64), String> {
    detect_stream_with_pose(
        None,
        source,
        output,
        expected_sha,
        count,
        width,
        height,
        settings,
        true,
        None,
        None,
        None,
        &[],
        None,
        None,
        None,
        &[],
        None,
        None,
        None,
    )
}

#[cfg(test)]
fn touch_confidence_passes(probability: f64, threshold: f64) -> bool {
    probability.is_finite() && probability >= threshold
}

#[expect(clippy::too_many_arguments)]
fn detect_stream_with_pose(
    _tcn: Option<&mut TrajectoryTcn>,
    source: &Path,
    output: &Path,
    expected_sha: &str,
    count: u64,
    width: u32,
    height: u32,
    settings: TouchSettings,
    interpolate_players: bool,
    pose_source: Option<(&Path, &str, bool)>,
    court: Option<&CourtTransform>,
    net: Option<[[f64; 2]; 2]>,
    training: &[training_samples::Example],
    ball_mask: Option<&crate::cpu_retrack::ball_track::BallTrackMask>,
    stabilized: Option<&StabilizedReplay>,
    primary_breaks: Option<&[volley_core::motion::TrajectoryBreak]>,
    rally_starts: &[u64],
    preliminary_intervals: Option<&[(u64, u64)]>,
    mut on_progress: Option<&mut dyn FnMut(u64, u64) -> Result<(), String>>,
    cancel: Option<&AtomicBool>,
) -> Result<(String, u64, u64), String> {
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        return Err("cancelled".into());
    }
    settings.validate()?;
    if preliminary_intervals.is_some_and(|intervals| {
        intervals
            .iter()
            .any(|(start, end)| start > end || *end >= count)
            || intervals.windows(2).any(|pair| pair[0].1 >= pair[1].0)
    }) {
        return Err("Invalid preliminary rally intervals".into());
    }
    if count == 0 || count > 48 * 3600 * 30 {
        return Err("Touch source frame count exceeds limit".into());
    }
    let full_range = [(0, count - 1)];
    let windows = preliminary_intervals.unwrap_or(&full_range);
    let scoped_frames: u64 = windows.iter().map(|(start, end)| end - start + 1).sum();
    let mut reader = BoundedReader::open(source)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| format!("Create isolated touch output: {e}"))?;
    let mut writer = BufWriter::new(file);
    let mut output_hash = Sha256::new();
    let mut buffer: VecDeque<TouchFrame> = VecDeque::new();
    let mut refined = BTreeMap::<u64, Vec<PosePlayer>>::new();
    let mut next_emit = windows.first().map(|(start, _)| *start).unwrap_or(count);
    let mut window_index = 0;
    let mut completed_frames = 0;
    let mut emitted: u64 = 0;
    let mut motion_candidates: u64 = 0;
    let mut pending: Option<PhysicalTouchCandidate> = None;
    let mut initial_serves = Vec::<InitialServeProposal>::new();
    let separation = (settings.min_interval_sec * 30.0).round().max(1.0) as u64;
    let mut process_block = |start: u64,
                             end: u64,
                             input: &VecDeque<TouchFrame>,
                             refined: &BTreeMap<u64, Vec<PosePlayer>>,
                             window: (u64, u64)|
     -> Result<(), String> {
        let region_start = start.saturating_sub(HALO as u64).max(window.0);
        let region_end = (end + HALO as u64).min(window.1 + 1);
        let first = input
            .front()
            .ok_or("Touch frame buffer is empty")?
            .frame_index;
        if region_start < first
            || region_end
                > input
                    .back()
                    .ok_or("Touch frame buffer truncated")?
                    .frame_index
                    + 1
        {
            return Err("Touch segment missing lookbehind or lookahead".into());
        }
        let offset = (region_start - first) as usize;
        let n = (region_end - region_start) as usize;
        let mut block = input
            .iter()
            .skip(offset)
            .take(n)
            .cloned()
            .collect::<Vec<_>>();
        // Consume the completed postprocessing Ball Stabilizer pass, if present.
        // Raw GPU/replay evidence remains immutable and SHA-verified.
        if let Some(stabilized) = stabilized {
            use_stabilized_ball(&mut block, &stabilized.motion);
        } else {
            gate_ball_confidence(&mut block, settings.min_ball_score);
        }
        let clean = if stabilized.is_some() {
            block
        } else if let Some(mask) = ball_mask {
            for frame in &mut block {
                if !mask.accepted[frame.frame_index as usize] {
                    frame.selected_ball = None;
                }
            }
            block
        } else {
            exclude_ball_teleports(&block, width, height)
        };
        // Reconstruct bounded positions of existing tracked skeletons only.
        // The immutable replay and original GPU player detections remain unchanged.
        let continuous = if interpolate_players {
            reconstruct_track_gaps(&clean)
        } else {
            clean
        };
        if start == window.0 {
            initial_serves.push(initial_serve_proposal(&continuous, window.0));
        }
        let filtered = if let Some(calibration) = court {
            court_touch::filter_on_court(&continuous, calibration)
        } else {
            continuous
        };
        for range in volley_core::touch_pipeline::rally_frame_ranges(&filtered, rally_starts)? {
            let evidence_frames: &[TouchFrame] = &filtered[range];
            let start = start.max(evidence_frames[0].frame_index);
            let end = end.min(evidence_frames.last().unwrap().frame_index + 1);
            if start >= end {
                continue;
            }
            let mut events = detect_ball_trajectory_touches(
                evidence_frames,
                width,
                height,
                start,
                end,
                settings,
                court.map(|court| (court, window.1)),
                primary_breaks.map(|events| {
                    let first = events.partition_point(|event| event.frame < start as i64);
                    let last = events.partition_point(|event| event.frame < end as i64);
                    &events[first..last]
                }),
            )?;
            attach_touch_source_evidence(
                &mut events,
                evidence_frames,
                width,
                height,
                settings.max_reach_ratio,
            );
            if let Some((root, fusion, interpolation)) = pose_source {
                let mut classification_settings = settings;
                classification_settings.require_pose = false;
                refine_pose_events_policy(
                    &mut events,
                    evidence_frames,
                    (width, height),
                    root,
                    fusion,
                    classification_settings,
                    Some(interpolation),
                    refined,
                )?;
            }
            apply_training(
                &mut events,
                evidence_frames,
                pose_source,
                width,
                height,
                settings,
                training,
                refined,
            )?;
            if let Some(intervals) = preliminary_intervals {
                events.retain(|event| {
                    intervals.iter().any(|(first, last)| {
                        event.frame_index >= *first && event.frame_index <= *last
                    })
                });
            }
            if let Some(calibration) = court {
                court_touch::annotate(&mut events, evidence_frames, calibration);
            }
            if net.is_some() {
                for event in &mut events {
                    if event.action_prediction.as_ref().is_some_and(|p| {
                        p.kind == volley_core::action_prediction::ActionKind::Block
                    }) && !volley_core::court_geometry::observed_net_contact(
                        net,
                        [
                            event.ball_x_px / width as f64,
                            event.ball_y_px / height as f64,
                        ],
                    ) {
                        event.action_prediction = None;
                    }
                }
            }

            for mut event in events {
                if let Some(track_id) = event.provisional_track_id {
                    let (first_slice, second_slice) = input.as_slices();
                    if !source_frame_contains_track(first_slice, event.frame_index, track_id)
                        && !source_frame_contains_track(second_slice, event.frame_index, track_id)
                    {
                        event.provisional_track_id = None;
                        event.body_contact_supported = false;
                        event.contact_landmark = None;
                        event.contact_pose_sample_frame = None;
                        event.action_prediction = None;
                        event.court = None;
                    }
                }
                motion_candidates += 1;

                if let Some(last) = select_pending_touch(&mut pending, event, separation) {
                    emit_record(&mut writer, &mut output_hash, &last)?;
                    emitted += 1;
                }
            }
        }
        Ok(())
    };
    for index in 0..count {
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        let bytes = reader.line()?.ok_or("CPU tracked frame JSONL truncated")?;
        let Some(&(window_start, window_end)) = windows.get(window_index) else {
            continue;
        };
        if index < window_start {
            continue;
        }
        let raw: TrackedFrame = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Invalid CPU tracked frame: {e}"))?;
        let measured = raw
            .tracked_players
            .iter()
            .filter_map(|player| {
                matches!(
                    player.pose_source.as_deref(),
                    Some(
                        crate::cpu_retrack::POSE_REFINEMENT_POLICY
                            | crate::cpu_retrack::LEGACY_POSE_REFINEMENT_POLICY
                            | crate::cpu_retrack::PREVIOUS_POSE_REFINEMENT_POLICY
                    )
                )
                .then_some(PosePlayer {
                    track_id: player.track_id,
                    keypoints: player.pose?,
                    interpolated: false,
                })
            })
            .collect::<Vec<_>>();
        if !measured.is_empty() {
            refined.insert(index, measured);
        }
        buffer.push_back(transform_frame(raw, index)?);
        while next_emit + (CHUNK + HALO) as u64 <= index + 1 {
            let end = next_emit + CHUNK as u64;
            process_block(
                next_emit,
                end,
                &buffer,
                &refined,
                (window_start, window_end),
            )?;
            completed_frames += end - next_emit;
            next_emit = end;
            if let Some(report) = on_progress.as_deref_mut() {
                report(completed_frames, scoped_frames)?;
            }
            while buffer
                .front()
                .is_some_and(|row| row.frame_index < next_emit.saturating_sub(HALO as u64))
            {
                buffer.pop_front();
            }
            refined.retain(|frame, _| *frame >= next_emit.saturating_sub(HALO as u64));
        }
        if index == window_end {
            while next_emit <= window_end {
                let end = (next_emit + CHUNK as u64).min(window_end + 1);
                process_block(
                    next_emit,
                    end,
                    &buffer,
                    &refined,
                    (window_start, window_end),
                )?;
                completed_frames += end - next_emit;
                next_emit = end;
                if let Some(report) = on_progress.as_deref_mut() {
                    report(completed_frames, scoped_frames)?;
                }
                while buffer
                    .front()
                    .is_some_and(|row| row.frame_index < next_emit.saturating_sub(HALO as u64))
                {
                    buffer.pop_front();
                }
            }
            buffer.clear();
            refined.clear();
            window_index += 1;
            next_emit = windows
                .get(window_index)
                .map(|(start, _)| *start)
                .unwrap_or(count);
        }
    }
    if reader.line()?.is_some() {
        return Err("CPU replay has extra undeclared frames".into());
    }
    if reader.finish() != expected_sha {
        return Err("CPU replay SHA-256 changed while reading Touch input".into());
    }
    if let Some(last) = pending {
        emit_record(&mut writer, &mut output_hash, &last)?;
        emitted += 1;
    }
    writer
        .flush()
        .map_err(|e| format!("Flush physical touch candidates: {e}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| format!("Sync physical touch candidates: {e}"))?;
    if preliminary_intervals.is_some() {
        let path = output.with_file_name("initial_serve_proposals.partial.json");
        fs::write(
            &path,
            serde_json::to_vec(&initial_serves).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("Save initial serve proposals: {e}"))?;
    }
    Ok((
        format!("{:x}", output_hash.finalize()),
        emitted,
        motion_candidates,
    ))
}

fn published_touch_settings(requested: Option<TouchSettings>) -> Result<TouchSettings, String> {
    let mut settings = requested.unwrap_or_default().validate()?;
    settings.algorithm = TouchAlgorithm::BallTrajectory;
    settings.require_pose = false;
    Ok(settings)
}

pub(crate) fn process(state: &AppState, req: &TouchRequest) -> Result<Value, ApiError> {
    process_with_ball_scope_and_rally_settings(
        state,
        req,
        &[],
        volley_core::rally_reconstruction::RallySettings::default(),
    )
}

pub(crate) fn process_discovery(state: &AppState, req: &TouchRequest) -> Result<Value, ApiError> {
    process_discovery_with_rally_settings(
        state,
        req,
        volley_core::rally_reconstruction::RallySettings::default(),
    )
}

pub(crate) fn process_discovery_with_rally_settings(
    state: &AppState,
    req: &TouchRequest,
    rally_settings: volley_core::rally_reconstruction::RallySettings,
) -> Result<Value, ApiError> {
    process_with_ball_scope_and_rally_settings(state, req, &[], rally_settings)
}

pub(crate) fn process_with_ball_scope_and_rally_settings(
    state: &AppState,
    req: &TouchRequest,
    rally_starts: &[u64],
    rally_settings: volley_core::rally_reconstruction::RallySettings,
) -> Result<Value, ApiError> {
    process_with_ball_scope_and_rally_settings_report(
        state,
        req,
        rally_starts,
        rally_settings,
        None,
        None,
        None,
        None,
    )
}

pub(crate) fn process_primary_with_progress_cancel(
    state: &AppState,
    req: &TouchRequest,
    rally_settings: volley_core::rally_reconstruction::RallySettings,
    scope: &PrimaryRallyScope,
    stabilized: &StabilizedReplay,
    report: &mut dyn FnMut(u64, u64) -> Result<(), String>,
    cancel: &AtomicBool,
) -> Result<Value, ApiError> {
    process_with_ball_scope_and_rally_settings_report(
        state,
        req,
        &[],
        rally_settings,
        Some(scope),
        Some(stabilized),
        Some(report),
        Some(cancel),
    )
}

#[expect(clippy::too_many_arguments)]
fn process_with_ball_scope_and_rally_settings_report(
    state: &AppState,
    req: &TouchRequest,
    rally_starts: &[u64],
    rally_settings: volley_core::rally_reconstruction::RallySettings,
    primary_scope: Option<&PrimaryRallyScope>,
    stabilized: Option<&StabilizedReplay>,
    mut report: Option<&mut dyn FnMut(u64, u64) -> Result<(), String>>,
    cancel: Option<&AtomicBool>,
) -> Result<Value, ApiError> {
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        return Err(storage::conflict("cancelled"));
    }
    if !state.writes_enabled {
        return Err(storage::conflict(
            "Touch recalculation requires an isolated working directory",
        ));
    }
    storage::project_id(&req.project_id)?;
    storage::video_name(&req.video)?;
    if !opaque_id(&req.replay_id) {
        return Err(storage::bad_request("Invalid CPU track replay ID"));
    }
    let settings = published_touch_settings(req.settings).map_err(storage::bad_request)?;
    let rally_settings = rally_settings.validate().map_err(storage::bad_request)?;
    // Fail before creating run artifacts if no trained checkpoint exists.
    // Model bytes never replace or mutate saved VBallNet inference evidence.
    let mut tcn = if settings.algorithm == TouchAlgorithm::TrajectoryTcn {
        Some(
            TrajectoryTcn::load(&state.repository_root.join("models").join(MODEL_FILENAME))
                .map_err(storage::bad_request)?,
        )
    } else {
        None
    };
    let tcn_sha = tcn.as_ref().map(|network| network.sha256().to_owned());
    let _guard = project_guard::write_guard(state, &req.project_id)?;
    let project = projects::existing_project(state, &req.project_id)?;
    if !project["videos"]
        .as_array()
        .is_some_and(|v| v.iter().any(|name| name.as_str() == Some(&req.video)))
    {
        return Err(storage::bad_request("Video not present in project"));
    }
    let plan = preparation::plan_for_run(
        state,
        &req.project_id,
        &req.video,
        "native_cpu_replays",
        &req.replay_id,
    )?;
    let prepared = plan
        .ready_path()
        .ok_or_else(|| storage::bad_request("Prepared video unavailable"))?;
    let (size, mtime) = inference_stream::source_stamp(prepared).map_err(storage::bad_request)?;
    let dimensions = volley_media::probe(prepared).map_err(storage::bad_request)?;
    if !(64..=16384).contains(&dimensions.width)
        || !(64..=16384).contains(&dimensions.height)
        || plan.identity.fps != 30
    {
        return Err(storage::bad_request(
            "CPU Touch requires 30-fps prepared media dimensions",
        ));
    }
    let court_calibration = video_edits::court_snapshot(state, &req.project_id, &req.video)?;
    let training_examples = training_samples::touch_examples(state)?;
    let project_dir = projects::project_root(state).join(&req.project_id);
    safe_folder(&project_dir).map_err(storage::bad_request)?;
    let replay_root = project_dir.join("native_cpu_replays");
    safe_folder(&replay_root).map_err(storage::bad_request)?;
    let replay_dir = replay_root.join(&req.replay_id);
    safe_folder(&replay_dir).map_err(storage::bad_request)?;
    let manifest_file = safe_file(&replay_dir, "manifest.json", MAX_MANIFEST_BYTES)
        .map_err(storage::bad_request)?;
    let manifest: Value = serde_json::from_slice(
        &fs::read(manifest_file)
            .map_err(|e| storage::internal(format!("Read CPU tracking manifest: {e}")))?,
    )
    .map_err(|e| storage::bad_request(format!("Invalid tracking manifest: {e}")))?;
    volley_media::preparation::verify_source_provenance(&plan.source, &manifest["original_source"])
        .map_err(storage::conflict)?;
    if manifest["schema"] != "volley-native-cpu-tracking/1"
        || manifest["state"] != "provisional_tracklets_rebuilt"
        || manifest["project_id"].as_str() != Some(&req.project_id)
        || manifest["replay_id"].as_str() != Some(&req.replay_id)
        || manifest["video"].as_str() != Some(&req.video)
        || manifest["fps"] != 30
        || manifest["file"] != "tracked_frames.jsonl"
        || manifest["track_ids_provisional"] != true
        || manifest["athlete_identities_confirmed"] != false
        || manifest["touches_confirmed"] != false
        || manifest["cpu_processing_complete"] != false
        || manifest["source_identity"] != json!({"bytes":size,"mtime_ns":mtime})
    {
        return Err(storage::bad_request(
            "CPU tracked frames are incompatible with source video",
        ));
    }
    let frames = manifest["frames"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= 48 * 3600 * 30)
        .ok_or_else(|| storage::bad_request("Invalid CPU tracked frame count"))?;
    let primary_intervals = primary_scope
        .map(|scope| align_primary_intervals(&scope.intervals, scope.source_frames, frames))
        .transpose()
        .map_err(storage::bad_request)?;
    let mut primary_starts = vec![0];
    if let Some(intervals) = &primary_intervals {
        primary_starts.extend(intervals.iter().map(|(start, _)| *start).filter(|v| *v > 0));
    }
    let rally_starts = if primary_scope.is_some() {
        primary_starts.as_slice()
    } else {
        rally_starts
    };
    if let Some(update) = report.as_deref_mut() {
        let scoped_frames = primary_intervals
            .as_deref()
            .map(|intervals| intervals.iter().map(|(start, end)| end - start + 1).sum())
            .unwrap_or(frames);
        update(0, scoped_frames).map_err(storage::internal)?;
    }
    let sha = manifest["sha256"]
        .as_str()
        .filter(|s| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        })
        .ok_or_else(|| storage::bad_request("Tracked-frame checksum missing"))?
        .to_owned();
    let source = safe_file(&replay_dir, "tracked_frames.jsonl", MAX_INPUT_BYTES)
        .map_err(storage::bad_request)?;
    if let Some(stabilized) = stabilized {
        if stabilized.source_sha256 != sha
            || stabilized.motion.points.len() as u64 != frames
            || stabilized.replay_id != req.replay_id
            || stabilized.project_root != project_dir
        {
            return Err(storage::conflict(
                "Ball Stabilizer does not match this CPU replay",
            ));
        }
    }
    let fusion_id = manifest["source_fusion_id"]
        .as_str()
        .filter(|id| opaque_id(id))
        .ok_or_else(|| storage::bad_request("Touch replay has no valid GPU fusion lineage"))?
        .to_owned();
    let output_root = project_dir.join("native_cpu_touch_runs");
    match fs::symlink_metadata(&output_root) {
        Ok(meta) if !meta.is_dir() || meta.file_type().is_symlink() => {
            return Err(storage::bad_request("Unsafe native touch run root"))
        }
        Ok(_) => (),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => fs::create_dir(&output_root)
            .map_err(|e| storage::internal(format!("Create native Touch runs folder: {e}")))?,
        Err(e) => {
            return Err(storage::internal(format!(
                "Check native Touch runs root: {e}"
            )))
        }
    }
    let touch_id = Uuid::new_v4().simple().to_string();
    let folder = output_root.join(&touch_id);
    fs::create_dir(&folder)
        .map_err(|e| storage::internal(format!("Create Touch run folder: {e}")))?;
    let result = (|| -> Result<Value, String> {
        let part = folder.join("touch_candidates.jsonl.partial");
        let pose_requested = req.pose_classification
            || settings.require_pose
            || settings.algorithm == TouchAlgorithm::PoseMotion
            || native_preview_pose::source_info(&project_dir, &fusion_id)?.is_some();
        let training_sha = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&training_examples).map_err(|e| e.to_string())?)
        );
        let pose_info = if pose_requested {
            native_preview_pose::source_info(&project_dir, &fusion_id)?
        } else {
            None
        };
        let ball_mask = if stabilized.is_none() {
            Some(crate::cpu_retrack::ball_track::read_with_cancel(
                &source,
                &sha,
                frames,
                (dimensions.width, dimensions.height),
                settings.min_ball_score,
                if primary_scope.is_some() {
                    &[]
                } else {
                    rally_starts
                },
                cancel,
            )?)
        } else {
            None
        };
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        let preliminary = if let Some(intervals) = &primary_intervals {
            intervals.clone()
        } else {
            let continuous = crate::cpu_retrack::ball_track::continuous_ball_intervals_with_cancel(
                &source,
                &sha,
                frames,
                settings.min_ball_score,
                crate::cpu_retrack::ball_track::CONTINUOUS_MIN_SECONDS,
                cancel,
            )?;
            crate::cpu_retrack::ball_track::filter_short_rallies(
                crate::cpu_retrack::ball_track::merge_continuous_intervals(
                    &continuous,
                    crate::cpu_retrack::ball_track::CONTINUOUS_MERGE_SECONDS,
                )?,
            )
        };
        let preliminary_scope_applied = true;
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        let mut checked_progress = |done, total| {
            if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
                return Err("cancelled".into());
            }
            if let Some(report) = report.as_deref_mut() {
                report(done, total)?;
            }
            Ok(())
        };
        let persisted = if let Some(stabilized) = stabilized {
            let (path, file_sha) = verified_ball_trajectory(
                &project_dir,
                &stabilized.summary(),
                &req.project_id,
                &req.video,
                &req.replay_id,
                &sha,
                frames,
            )?;
            let motion = read_canonical_ball_motion(
                &path,
                &file_sha,
                frames,
                stabilized.motion.confidence_frames,
                stabilized.motion.observed_frames,
            )?;
            Some(StabilizedReplay {
                motion,
                source_sha256: sha.clone(),
                settings: stabilized.settings,
                artifact_id: stabilized.artifact_id.clone(),
                trajectory_sha256: file_sha,
                frames,
                project_root: project_dir.clone(),
                replay_id: req.replay_id.clone(),
            })
        } else {
            None
        };
        let primary_breaks = if persisted.is_some() {
            read_primary_touch_breaks(
                &project_dir,
                &manifest,
                dimensions.width,
                dimensions.height,
                frames,
            )?
        } else {
            None
        };
        let (digest, events, raw) = detect_stream_with_pose(
            tcn.as_mut(),
            &source,
            &part,
            &sha,
            frames,
            dimensions.width,
            dimensions.height,
            settings,
            manifest["interpolate_players"].as_bool().unwrap_or(true),
            pose_requested.then_some((
                project_dir.as_path(),
                fusion_id.as_str(),
                manifest["interpolate_pose"].as_bool().unwrap_or(true),
            )),
            court_calibration
                .as_ref()
                .and_then(|value| value.transform.as_ref()),
            court_calibration.as_ref().and_then(|value| value.net),
            &training_examples,
            ball_mask.as_deref(),
            persisted.as_ref(),
            primary_breaks.as_deref(),
            rally_starts,
            preliminary_scope_applied.then_some(preliminary.as_slice()),
            Some(&mut checked_progress),
            cancel,
        )?;
        if inference_stream::source_stamp(prepared)? != (size, mtime) {
            return Err("Prepared source changed during CPU Touch replay".into());
        }
        if pose_requested
            && native_preview_pose::source_info(&project_dir, &fusion_id)? != pose_info
        {
            return Err("Pose evidence changed during CPU Touch replay".into());
        }
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        fs::rename(part, folder.join(OUTPUT_FILE))
            .map_err(|e| format!("Publish motion-gated Touch events: {e}"))?;
        let mut record = json!({
            "schema":"volley-native-cpu-touch-proposals/1",
            "state":"motion_gated_touch_proposals",
            "project_id":req.project_id,
            "video":req.video,
            "touch_id":touch_id,
            "source_replay_id":req.replay_id,
            "source_replay_sha256":sha,
            "source_identity":{"bytes":size,"mtime_ns":mtime},
            "original_source":manifest["original_source"],
            "analysis_range":manifest["analysis_range"],
            "width":dimensions.width,"height":dimensions.height,"fps":30,
            "frames":frames,
            "settings":settings,
            "ball_tracking_policy":if primary_scope.is_some() || rally_starts.is_empty() {
                crate::cpu_retrack::ball_track::LOCAL_POLICY
            } else {
                crate::cpu_retrack::ball_track::POLICY
            },
            "leading_ball_rally_starts":if primary_scope.is_some() {json!([])} else {json!(rally_starts)},
            "leading_ball_min_score":settings.min_ball_score,
            "interpolate_players":manifest["interpolate_players"].as_bool().unwrap_or(true),
            "court_revision":court_calibration.as_ref().map(|c| &c.revision),
            "court_context_enabled":court_calibration.is_some(),
            "motion_supported_proposals_before_dedup":raw,
            "touch_candidates":events,
            "method":match settings.algorithm {
                TouchAlgorithm::PoseMotion => "vballnet_trajectory_body_contact_v12",
                TouchAlgorithm::Legacy => "vballnet_gravity_player_geometry_pose_contact_v11",
                TouchAlgorithm::TrajectoryTcn => "vballnet_volley3d_tcn_v1",
                TouchAlgorithm::BallTrajectory => "vballnet_touch_detector_v4",
            },
            "tcn_model_sha256":tcn_sha,
            "trajectory_detector_version":volley_core::touch_pipeline::TRAJECTORY_DETECTOR_VERSION,
            "pose_classification_requested":pose_requested,
            "human_training_example_count":training_examples.len(),
            "human_training_touch_sha256":training_sha,
            "pose_source_fusion_id": if pose_requested {json!(fusion_id)} else {Value::Null},
            "pose_source":pose_info.as_ref().map(|(id, sha, stride)| json!({"pose_run_id":id,"sha256":sha,"stride":stride})),
            "body_trajectory_intersection_required":settings.require_pose,
            "ball_source_switch_guard":true,
            "file":OUTPUT_FILE,
            "sha256":digest,
            "contacts_authoritatively_confirmed":false,
            "athlete_identities_verified":false,
            "action_technique_classification_complete":false,
            "automatic_action_predictions_enabled":true,
            "automatic_action_prediction_method":if pose_requested {
                volley_core::pose_classification::ACTION_PREDICTION_POLICY_VERSION
            } else { "unknown_touch_no_pose" },
            "cpu_processing_complete":false,
        });
        record["primary_touch_detector_reused"] = json!(primary_breaks.is_some());
        if let Some(stabilized) = stabilized {
            record["ball_stabilizer"] = stabilized.summary();
        }
        let serve_proposals_file = folder.join("initial_serve_proposals.partial.json");
        let serve_proposals: Value = serde_json::from_slice(
            &fs::read(&serve_proposals_file)
                .map_err(|e| format!("Read initial serve proposals: {e}"))?,
        )
        .map_err(|e| format!("Invalid initial serve proposals: {e}"))?;
        fs::remove_file(&serve_proposals_file)
            .map_err(|e| format!("Remove temporary serve proposals: {e}"))?;
        record["initial_serve_proposal_count"] =
            json!(serve_proposals.as_array().map_or(0, Vec::len));
        let preliminary_data = json!({
            "initial_serve_proposals":serve_proposals,
            "schema":"volley-native-preliminary-continuous-rallies/1",
            "source_replay_id":req.replay_id,
            "source_replay_sha256":sha,
            "frames":frames,
            "settings":rally_settings,
            "min_ball_score":settings.min_ball_score,
            "min_trajectory_seconds":crate::cpu_retrack::ball_track::CONTINUOUS_MIN_SECONDS,
            "merge_gap_seconds":crate::cpu_retrack::ball_track::CONTINUOUS_MERGE_SECONDS,
            "rally_intervals":preliminary,
            "primary_ball_job_id":primary_scope.map(|scope| scope.ball_job_id.as_str()),
            "primary_rally_mask_sha256":primary_scope.map(|scope| scope.sha256.as_str()),
            "primary_source_frames":primary_scope.map(|scope| scope.source_frames),
        });
        storage::write_json_atomic(&folder.join("preliminary_rallies.json"), &preliminary_data)
            .map_err(|e| format!("Save preliminary ball-only gameplay: {e:?}"))?;
        let preliminary_bytes = fs::read(folder.join("preliminary_rallies.json"))
            .map_err(|e| format!("Read preliminary gameplay checksum: {e}"))?;
        record["preliminary_rally_scope_version"] = json!(if primary_scope.is_some() {
            "primary_gpu_rally_mask_v1"
        } else {
            "coherent_vballnet_motion_v2"
        });
        record["primary_ball_job_id"] =
            json!(primary_scope.map(|scope| scope.ball_job_id.as_str()));
        record["primary_rally_mask_sha256"] =
            json!(primary_scope.map(|scope| scope.sha256.as_str()));
        record["preliminary_rally_scope_applied"] = json!(preliminary_scope_applied);
        record["preliminary_rally_settings"] = json!(rally_settings);
        record["preliminary_rally_file"] = json!("preliminary_rallies.json");
        record["preliminary_rally_sha256"] =
            json!(format!("{:x}", Sha256::digest(&preliminary_bytes)));
        record["preliminary_rally_interval_count"] = json!(preliminary_data["rally_intervals"]
            .as_array()
            .unwrap()
            .len());
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        storage::write_json_atomic(&folder.join("manifest.json"), &record)
            .map_err(|e| format!("Save native Touch manifest: {e:?}"))?;
        Ok(record)
    })();
    match result {
        Ok(value) => Ok(value),
        Err(error) => {
            let _ = fs::remove_dir_all(&folder);
            Err(storage::bad_request(error))
        }
    }
}
pub async fn run(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<TouchRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let result = tokio::task::spawn_blocking(move || process(&state, &request))
        .await
        .map_err(|e| storage::internal(format!("CPU Touch task failed: {e}")))??;
    Ok(Json(result))
}
pub async fn status(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TouchQuery>,
) -> ApiResult<Value> {
    storage::project_id(&query.project_id)?;
    if !opaque_id(&query.touch_id) {
        return Err(storage::bad_request("Invalid Touch ID"));
    }
    projects::existing_project(&state, &query.project_id)?;
    let project = projects::project_root(&state).join(&query.project_id);
    safe_folder(&project).map_err(storage::bad_request)?;
    let root = project.join("native_cpu_touch_runs");
    safe_folder(&root).map_err(storage::bad_request)?;
    let folder = root.join(&query.touch_id);
    safe_folder(&folder).map_err(storage::bad_request)?;
    let data = fs::read(
        safe_file(&folder, "manifest.json", MAX_MANIFEST_BYTES).map_err(storage::bad_request)?,
    )
    .map_err(|e| storage::internal(format!("Read native Touch metadata: {e}")))?;
    let value: Value = serde_json::from_slice(&data)
        .map_err(|e| storage::bad_request(format!("Corrupt native Touch metadata: {e}")))?;
    if value["schema"] != "volley-native-cpu-touch-proposals/1"
        || value["touch_id"].as_str() != Some(&query.touch_id)
        || value["project_id"].as_str() != Some(&query.project_id)
    {
        return Err(storage::bad_request(
            "Touch manifest belongs to different project",
        ));
    }
    Ok(Json(value))
}

#[cfg(test)]
mod tests {
    #[test]
    fn initial_serve_can_link_a_track_visible_after_rally_start() {
        let mut frames = (100..200)
            .map(|frame_index| TouchFrame {
                frame_index,
                players: Vec::new(),
                selected_ball: (frame_index < 110).then_some(TouchBall {
                    source_model: "vballnet".into(),
                    score: 0.9,
                    bbox_xyxy_norm: [0.49, 0.49, 0.51, 0.51],
                }),
            })
            .collect::<Vec<_>>();
        frames[60].players.push(TouchPlayer {
            track_id: 42,
            score: 0.9,
            bbox_xyxy_norm: [0.44, 0.3, 0.56, 0.75],
            touch_probability: None,
        });
        let proposal = initial_serve_proposal(&frames, 100);
        assert_eq!(proposal.rally_start_frame, 100);
        assert_eq!(proposal.trajectory_start_frame, Some(100));
        assert_eq!(proposal.provisional_track_id, Some(42));
        assert!(!proposal.identity_verified);
    }

    #[test]
    fn initial_serve_does_not_guess_between_ambiguous_players() {
        let frames = vec![TouchFrame {
            frame_index: 200,
            selected_ball: Some(TouchBall {
                source_model: "vballnet".into(),
                score: 0.9,
                bbox_xyxy_norm: [0.49, 0.49, 0.51, 0.51],
            }),
            players: [7, 8]
                .map(|track_id| TouchPlayer {
                    track_id,
                    score: 0.9,
                    bbox_xyxy_norm: [0.44, 0.3, 0.56, 0.75],
                    touch_probability: None,
                })
                .to_vec(),
        }];
        let proposal = initial_serve_proposal(&frames, 200);
        assert_eq!(proposal.provisional_track_id, None);
    }
    #[test]
    fn minimum_contact_confidence_applies_to_all_pose_motion_candidates() {
        let scores = [0.29, 0.48, 0.64, 0.83];
        let enabled = |threshold| {
            scores
                .iter()
                .copied()
                .filter(|score| touch_confidence_passes(*score, threshold))
                .collect::<Vec<_>>()
        };
        assert_eq!(enabled(0.48), vec![0.48, 0.64, 0.83]);
        assert_eq!(enabled(0.80), vec![0.83]);
        assert_eq!(enabled(0.99), Vec::<f64>::new());
        assert!(!touch_confidence_passes(f64::NAN, 0.48));
    }

    use super::*;
    use std::path::PathBuf;

    #[test]
    fn cancelled_touch_scan_does_not_open_source_or_publish_files() {
        let cancelled = AtomicBool::new(true);
        let missing = std::env::temp_dir().join(format!("no-touch-source-{}", Uuid::new_v4()));
        let output = missing.with_extension("jsonl");
        let result = detect_stream_with_pose(
            None,
            &missing,
            &output,
            "no-sha",
            9,
            640,
            360,
            TouchSettings::default(),
            true,
            None,
            None,
            None,
            &[],
            None,
            None,
            None,
            &[],
            None,
            None,
            Some(&cancelled),
        );
        assert_eq!(result.unwrap_err(), "cancelled");
        assert!(!output.exists());
    }

    fn recorded_rebound_and_departure(frame: u64) -> [PhysicalTouchCandidate; 2] {
        let impact = PhysicalTouchCandidate {
            frame_index: frame,
            provisional_track_id: Some(6),
            probability: 0.73045,
            ball_x_px: 496.25,
            ball_y_px: 350.0,
            motion_frame_index: frame,
            direction_change_deg: 164.05,
            speed_before_px_frame: 15.0,
            speed_after_px_frame: 18.20,
            velocity_residual_px_frame: 32.88,
            technique: "unknown_touch",
            action_prediction: None,
            court: None,
            physical_motion_supported: true,
            body_contact_supported: true,
            contact_landmark: None,
            contact_pose_sample_frame: None,
            ball_occluded_at_contact: true,
            inferred_body_reach_supported: false,
            athlete_identity_verified: false,
        };
        let departure = PhysicalTouchCandidate {
            frame_index: frame + 4,
            provisional_track_id: Some(4),
            probability: 0.62499,
            ball_x_px: 515.0,
            ball_y_px: 282.5,
            motion_frame_index: frame + 4,
            direction_change_deg: 6.71,
            speed_before_px_frame: 7.40,
            speed_after_px_frame: 16.93,
            velocity_residual_px_frame: 14.36,
            body_contact_supported: false,
            ball_occluded_at_contact: false,
            ..impact.clone()
        };
        [impact, departure]
    }

    #[test]
    fn published_settings_do_not_require_identity_or_pose_for_detection() {
        let default = published_touch_settings(None).unwrap();
        assert!(!default.require_pose);
        assert_eq!(default.algorithm, TouchAlgorithm::BallTrajectory);
        let optional = TouchSettings {
            require_pose: true,
            algorithm: TouchAlgorithm::PoseMotion,
            ..TouchSettings::default()
        };
        let normalized = published_touch_settings(Some(optional)).unwrap();
        assert!(!normalized.require_pose);
        assert_eq!(normalized.algorithm, TouchAlgorithm::BallTrajectory);
    }

    #[test]
    fn only_pose_confirmed_motion_events_are_saved_as_touches() {
        let [mut candidate, geometry_only] = recorded_rebound_and_departure(235);
        let no_landmark = candidate.clone();
        candidate.contact_landmark = Some(volley_core::pose_contact::BodyContact {
            joint_indices: [9, 9],
            point_px: [496.25, 350.0],
            distance_px: 0.0,
            normalized_distance: 0.0,
        });
        let mut touches = vec![geometry_only, no_landmark, candidate];
        retain_pose_confirmed_events(&mut touches);
        assert_eq!(touches.len(), 1);
        assert_eq!(touches[0].technique, "unknown_touch");
        assert!(touches[0].action_prediction.is_none());
        assert!(touches[0].contact_landmark.is_some());
    }

    #[test]
    fn stream_selector_discards_departure_echo_from_different_track() {
        let [impact, departure] = recorded_rebound_and_departure(235);
        let mut pending = Some(impact);
        assert!(select_pending_touch(&mut pending, departure, 3).is_none());
        assert_eq!(pending.unwrap().frame_index, 235);
    }

    #[test]
    fn stream_selector_discards_departure_echo_across_block_boundary() {
        let [impact, departure] = recorded_rebound_and_departure(CHUNK as u64 - 1);
        let mut pending = Some(impact);
        assert!(select_pending_touch(&mut pending, departure, 3).is_none());
        assert_eq!(pending.unwrap().frame_index, CHUNK as u64 - 1);
    }

    #[test]
    fn stream_selector_replaces_approach_echo_with_later_rebound() {
        let [mut impact, mut approach] = recorded_rebound_and_departure(235);
        impact.frame_index = 243;
        impact.motion_frame_index = 243;
        approach.frame_index = 239;
        approach.motion_frame_index = 239;
        let mut pending = Some(approach);
        assert!(select_pending_touch(&mut pending, impact, 3).is_none());
        assert_eq!(pending.unwrap().frame_index, 243);
    }

    #[test]
    fn stream_selector_keeps_independent_body_contact_after_rebound() {
        let [impact, mut second] = recorded_rebound_and_departure(235);
        second.body_contact_supported = true;
        let mut pending = Some(impact);
        let emitted = select_pending_touch(&mut pending, second, 3).unwrap();
        assert_eq!(
            (emitted.frame_index, pending.unwrap().frame_index),
            (235, 239)
        );
    }

    #[test]
    fn stream_selector_keeps_separate_turn_after_rebound() {
        let [impact, mut second] = recorded_rebound_and_departure(235);
        second.direction_change_deg = 80.0;
        let mut pending = Some(impact);
        let emitted = select_pending_touch(&mut pending, second, 3).unwrap();
        assert_eq!(
            (emitted.frame_index, pending.unwrap().frame_index),
            (235, 239)
        );
    }

    /// Optional read-only quality gate for the 25-second new2 pilot replay.
    #[test]
    #[ignore = "requires VOLLEY_NEW2_TRACKED_FRAMES pointing at a local 756-frame replay"]
    fn new2_physics_priority_does_not_duplicate_7s_or_17s_touches() {
        let input = PathBuf::from(
            std::env::var("VOLLEY_NEW2_TRACKED_FRAMES").expect("pilot replay path required"),
        );
        let manifest: Value = serde_json::from_slice(
            &fs::read(input.parent().unwrap().join("manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["frames"], 756);
        let sha = manifest["sha256"].as_str().unwrap();
        {
            let output =
                std::env::temp_dir().join(format!("volley-new2-physics-{}.jsonl", Uuid::new_v4()));
            let settings = TouchSettings::default();
            let run = detect_stream(&input, &output, sha, 756, 1280, 720, settings).unwrap();
            let events: Vec<Value> = fs::read_to_string(&output)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            fs::remove_file(&output).unwrap();
            assert_eq!(run.1 as usize, events.len());
            for (start, end) in [(210, 240), (520, 570), (390, 420)] {
                let near: Vec<_> = events
                    .iter()
                    .filter(|row| (start..end).contains(&row["frame_index"].as_u64().unwrap()))
                    .collect();
                assert_eq!(
                    near.len(),
                    1,
                    "pilot: expected one real contact in frames {start}..{end}; actual={near:?}"
                );
            }
        }
    }

    #[test]
    fn unsupported_ball_is_not_touch_evidence() {
        let (bytes, _) = sample_frames(true, 320);
        let first = bytes.split(|byte| *byte == b'\n').next().unwrap();
        let mut raw: Value = serde_json::from_slice(first).unwrap();
        raw["selected_ball"]["source_model"] = json!("unsupported");
        let frame: TrackedFrame = serde_json::from_value(raw).unwrap();
        assert!(transform_frame(frame, 0).is_err());
    }
    #[test]
    fn saved_pose_refines_motion_touch_without_rerunning_gpu() {
        use volley_core::action_prediction::{ActionKind, ActionPrediction, PredictionMethod};
        let root = std::env::temp_dir().join(format!("touch-pose-{}", Uuid::new_v4()));
        let fusion = "a".repeat(32);
        let pose_id = "b".repeat(32);
        let fusion_dir = root.join("native_gpu_fusions").join(&fusion);
        let pose_dir = root.join("native_gpu_pose_runs").join(&pose_id);
        fs::create_dir_all(&fusion_dir).unwrap();
        fs::create_dir_all(&pose_dir).unwrap();
        let source_sha = "e".repeat(64);
        let identity = json!({"bytes":1234,"mtime_ns":4321});
        fs::write(
            fusion_dir.join("manifest.json"),
            serde_json::to_vec(&json!({
                "schema":"volley-native-gpu-fusion/1","fusion_id":fusion,
                "project_id":"abc123def456","video":"video.mp4","sha256":source_sha,
                "source_identity":identity,
            }))
            .unwrap(),
        )
        .unwrap();
        let mut points = [[0.5, 0.5, 0.0]; 17];
        points[5] = [0.43, 0.44, 0.95];
        points[6] = [0.57, 0.44, 0.95];
        points[7] = [0.41, 0.31, 0.96];
        points[8] = [0.59, 0.31, 0.96];
        points[9] = [0.49, 0.22, 0.96];
        points[10] = [0.52, 0.22, 0.96];
        let row = [24, 30, 36, 234, 240, 246]
            .into_iter()
            .map(|frame| {
                let mut sample_points = points;
                if frame >= 234 {
                    sample_points[9] = [0.47 + (frame - 234) as f64 / 150.0, 0.375, 0.95];
                } else if frame == 36 {
                    sample_points[9][1] = 0.35;
                }
                let bbox = if frame >= 234 {
                    [0.27, 0.15, 0.47, 0.65]
                } else {
                    [0.35, 0.17, 0.65, 0.90]
                };
                format!(
                    "{}\n",
                    json!({"frame":frame,
                    "players":[{"box":bbox,"keypoints":sample_points}]})
                )
            })
            .collect::<String>();
        fs::write(pose_dir.join("poses.jsonl"), &row).unwrap();
        let digest = format!("{:x}", Sha256::digest(row.as_bytes()));
        fs::write(
            pose_dir.join("manifest.json"),
            serde_json::to_vec(&json!({
                "schema":"volley-native-gpu-pose/1","state":"pose_evidence_complete",
                "pose_run_id":pose_id,"fusion_id":fusion,"project_id":"abc123def456",
                "video":"video.mp4","model_sha256":volley_inference::pose::MODEL_SHA256,
                "source_fusion_sha256":source_sha,"source_identity":identity,
                "file":"poses.jsonl","sha256":digest,"stride":6
            }))
            .unwrap(),
        )
        .unwrap();
        let tracked = TouchFrame {
            frame_index: 31,
            players: vec![TouchPlayer {
                track_id: 8,
                score: 0.90,
                touch_probability: Some(0.94),
                bbox_xyxy_norm: [0.35, 0.17, 0.65, 0.90],
            }],
            selected_ball: None,
        };
        let block: Vec<TouchFrame> = (24..=36)
            .map(|frame_index| {
                let mut frame = tracked.clone();
                frame.frame_index = frame_index;
                let ball_x = if frame_index <= 31 {
                    0.43 + (frame_index - 24) as f64 * 0.01
                } else {
                    0.50 - (frame_index - 31) as f64 * 0.01
                };
                frame.selected_ball = Some(TouchBall {
                    source_model: "vballnet".into(),
                    score: 0.97,
                    bbox_xyxy_norm: [ball_x - 0.01, 0.19, ball_x + 0.01, 0.21],
                });
                frame
            })
            .collect();
        let samples = pose_frames(&root, &fusion, &block).unwrap();
        assert_eq!(
            samples
                .iter()
                .map(|frame| frame.frame_index)
                .collect::<Vec<_>>(),
            (24..=36).collect::<Vec<_>>()
        );
        for sample in &samples {
            assert_eq!(sample.players[0].track_id, 8);
            assert_eq!(
                sample.players[0].interpolated,
                ![24, 30, 36].contains(&sample.frame_index),
            );
        }
        assert!(samples[7].players[0].keypoints[9][1] > 0.22);
        assert!(samples[7].players[0].keypoints[9][1] < 0.26);
        let mut events = vec![PhysicalTouchCandidate {
            frame_index: 31,
            provisional_track_id: Some(8),
            probability: 0.94,
            ball_x_px: 320.0,
            ball_y_px: 72.0,
            motion_frame_index: 31,
            direction_change_deg: 80.0,
            speed_before_px_frame: 7.0,
            speed_after_px_frame: 13.0,
            velocity_residual_px_frame: 12.0,
            technique: "unknown_touch",
            court: None,
            action_prediction: Some(ActionPrediction {
                kind: ActionKind::Attack,
                evidence_score: 0.87,
                method: PredictionMethod::ContactGeometryV1,
                depth_ratio: -0.1,
                posture_ratio: 0.4,
            }),
            physical_motion_supported: true,
            body_contact_supported: false,
            contact_landmark: None,
            contact_pose_sample_frame: None,
            ball_occluded_at_contact: false,
            inferred_body_reach_supported: false,
            athlete_identity_verified: false,
        }];
        let unconfirmed_motion = events[0].clone();
        refine_pose_events(
            &mut events,
            &block,
            640,
            360,
            &root,
            &fusion,
            TouchSettings::default(),
        )
        .unwrap();
        let changed = events[0].action_prediction.as_ref().unwrap();
        assert_eq!(changed.kind, ActionKind::OverheadPass);
        assert_eq!(changed.method, PredictionMethod::PoseKeypointsMotionV1);
        let mut pose_optional = vec![unconfirmed_motion];
        pose_optional[0].ball_x_px = 620.0;
        refine_pose_events(
            &mut pose_optional,
            &block,
            640,
            360,
            &root,
            &fusion,
            TouchSettings {
                algorithm: TouchAlgorithm::PoseMotion,
                require_pose: false,
                ..TouchSettings::default()
            },
        )
        .unwrap();
        assert_eq!(
            pose_optional.len(),
            1,
            "motion evidence must survive missing pose"
        );
        assert!(!pose_optional[0].body_contact_supported);
        let mut distant = events.clone();
        distant[0].ball_x_px = 620.0;
        refine_pose_events(
            &mut distant,
            &block,
            640,
            360,
            &root,
            &fusion,
            TouchSettings {
                require_pose: true,
                ..TouchSettings::default()
            },
        )
        .unwrap();
        assert!(distant.is_empty());
        let mut close = events.clone();
        refine_pose_events(
            &mut close,
            &block,
            640,
            360,
            &root,
            &fusion,
            TouchSettings {
                require_pose: true,
                ..TouchSettings::default()
            },
        )
        .unwrap();
        assert_eq!(close.len(), 1);
        // A different fusion cannot borrow the saved pose from this one.
        refine_pose_events(
            &mut events,
            &[tracked],
            640,
            360,
            &root,
            &"c".repeat(32),
            TouchSettings::default(),
        )
        .unwrap();
        assert_eq!(
            events[0].action_prediction.as_ref().unwrap().method,
            PredictionMethod::PoseKeypointsMotionV1
        );
        let (source_bytes, source_sha) = sample_frames(true, 240);
        let source = root.join("tracked.jsonl");
        fs::write(&source, source_bytes).unwrap();
        let (_, count, _) = detect_stream_with_pose(
            None,
            &source,
            &root.join("pose-contacts.jsonl"),
            &source_sha,
            630,
            640,
            360,
            TouchSettings {
                algorithm: TouchAlgorithm::PoseMotion,
                ..TouchSettings::default()
            },
            true,
            Some((&root, &fusion, true)),
            None,
            None,
            &[],
            None,
            None,
            None,
            &[],
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(count, 1);
        for algorithm in [TouchAlgorithm::Legacy, TouchAlgorithm::PoseMotion] {
            let output = root.join(format!("rally-cut-{algorithm:?}.jsonl"));
            detect_stream_with_pose(
                None,
                &source,
                &output,
                &source_sha,
                630,
                640,
                360,
                TouchSettings {
                    algorithm,
                    ..TouchSettings::default()
                },
                true,
                Some((&root, &fusion, true)),
                None,
                None,
                &[],
                None,
                None,
                None,
                &[240],
                None,
                None,
                None,
            )
            .unwrap();
            let near: Vec<Value> = fs::read_to_string(output)
                .unwrap()
                .lines()
                .map(|s| serde_json::from_str::<Value>(s).unwrap())
                .filter(|v| v["frame_index"].as_u64().unwrap().abs_diff(240) <= 5)
                .collect();
            assert!(
                near.is_empty(),
                "rally boundary cannot become {algorithm:?} touch: {near:?}"
            );
        }
        let _ = fs::remove_dir_all(root);
    }

    fn sample_frames(curve: bool, impact: u64) -> (Vec<u8>, String) {
        let mut bytes = Vec::new();
        for i in 0..630_u64 {
            let x = if i < impact - 40 {
                0.20
            } else if curve && i >= impact {
                0.50 - (i - impact).min(40) as f64 * 0.0075
            } else {
                0.20 + (i - (impact - 40)).min(80) as f64 * 0.0075
            };
            let y = 0.36;
            let candidate = Candidate {
                frame_index: i,
                class_name: "player".into(),
                score: 0.95,
                bbox_xyxy_norm: [0.27, 0.15, 0.47, 0.65],
                query_index: Some(0),
                touch_probability: Some(if (impact - 3..=impact + 3).contains(&i) {
                    0.97
                } else {
                    0.01
                }),
                head_point_norm: None,
            };
            let ball = Candidate {
                frame_index: i,
                class_name: "ball".into(),
                score: 0.92,
                bbox_xyxy_norm: [x, y, x + 0.02, y + 0.03],
                query_index: None,
                touch_probability: None,
                head_point_norm: None,
            };
            let row = json!({
                "frame_index":i,
                "tracked_players":[{"track_id":1,"detection":candidate,"is_verified_athlete":false}],
                "selected_ball":{"source_model":"vballnet","detection":ball},
                "track_ids_provisional":true,"confirmed_athlete_identities":false,
                "touches_confirmed":false,
            });
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        let sha = format!("{:x}", Sha256::digest(&bytes));
        (bytes, sha)
    }
    #[test]
    fn scoped_touch_replay_only_processes_primary_rally_frames() {
        let root = std::env::temp_dir().join(format!("touch-primary-scope-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source.jsonl");
        let (bytes, sha) = sample_frames(true, 320);
        fs::write(&source, bytes).unwrap();
        let mut progress = Vec::new();
        let (_, emitted, _) = detect_stream_with_pose(
            None,
            &source,
            &root.join("inside.jsonl"),
            &sha,
            630,
            640,
            360,
            TouchSettings::default(),
            true,
            None,
            None,
            None,
            &[],
            None,
            None,
            None,
            &[],
            Some(&[(10, 90), (285, 365)]),
            Some(&mut |done, total| {
                progress.push((done, total));
                Ok(())
            }),
            None,
        )
        .unwrap();
        assert_eq!(progress, vec![(81, 162), (162, 162)]);
        assert!(emitted > 0);
        for row in fs::read_to_string(root.join("inside.jsonl"))
            .unwrap()
            .lines()
        {
            let event: Value = serde_json::from_str(row).unwrap();
            let index = event["frame_index"].as_u64().unwrap();
            assert!((10..=90).contains(&index) || (285..=365).contains(&index));
        }
        let mut outside_progress = Vec::new();
        let (_, outside, _) = detect_stream_with_pose(
            None,
            &source,
            &root.join("outside.jsonl"),
            &sha,
            630,
            640,
            360,
            TouchSettings::default(),
            true,
            None,
            None,
            None,
            &[],
            None,
            None,
            None,
            &[],
            Some(&[(0, 80), (450, 530)]),
            Some(&mut |done, total| {
                outside_progress.push((done, total));
                Ok(())
            }),
            None,
        )
        .unwrap();
        assert_eq!(outside_progress, vec![(81, 162), (162, 162)]);
        assert_eq!(outside, 0);
        let mut empty_progress = Vec::new();
        let (_, empty, _) = detect_stream_with_pose(
            None,
            &source,
            &root.join("empty.jsonl"),
            &sha,
            630,
            640,
            360,
            TouchSettings::default(),
            true,
            None,
            None,
            None,
            &[],
            None,
            None,
            None,
            &[],
            Some(&[]),
            Some(&mut |done, total| {
                empty_progress.push((done, total));
                Ok(())
            }),
            None,
        )
        .unwrap();
        assert_eq!(empty, 0);
        assert!(empty_progress.is_empty());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn chunked_replay_verifies_sha_and_does_not_invent_straight_flight_touch() {
        let root = std::env::temp_dir().join(format!("touch-replay-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source.jsonl");
        let (bytes, sha) = sample_frames(true, 320);
        fs::write(&source, bytes).unwrap();
        let (result_sha, events, raw) = detect_stream(
            &source,
            &root.join("out.jsonl"),
            &sha,
            630,
            640,
            360,
            TouchSettings::default(),
        )
        .unwrap();
        assert!(events <= raw);
        assert!(events >= 1, "Expected synthetic reversal candidate");
        let written = fs::read(root.join("out.jsonl")).unwrap();
        assert_eq!(result_sha, format!("{:x}", Sha256::digest(&written)));
        let (straight, straight_hash) = sample_frames(false, 320);
        fs::write(&source, straight).unwrap();
        let (_digest, straight_events, _) = detect_stream(
            &source,
            &root.join("straight.jsonl"),
            &straight_hash,
            630,
            640,
            360,
            TouchSettings::default(),
        )
        .unwrap();
        assert_eq!(straight_events, 0);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn touch_crosses_chunk_boundaries_once_and_cpu_threshold_replays() {
        let root = std::env::temp_dir().join(format!("touch-seams-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        // The seam check must use a physically reachable trajectory reversal,
        // not a confidence-only peak rejected by motion-first detection.
        let reference = TouchSettings {
            min_gap: 10,
            max_reach_ratio: 0.70,
            min_interval_sec: 0.35,
            ..TouchSettings::default()
        };
        for impact in [240_u64, 479_u64] {
            let (bytes, sha) = sample_frames(true, impact);
            let source = root.join(format!("source-{impact}.jsonl"));
            fs::write(&source, bytes).unwrap();
            let path = root.join(format!("default-{impact}.jsonl"));
            let (_, events, _) =
                detect_stream(&source, &path, &sha, 630, 640, 360, reference).unwrap();
            assert_eq!(
                events, 1,
                "Touch should appear exactly once near chunk border {impact}"
            );
            let rows = fs::read_to_string(path).unwrap();
            let parsed = rows
                .lines()
                .map(|s| serde_json::from_str::<Value>(s).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(parsed.len(), 1);
            assert!(parsed[0]["frame_index"].as_u64().unwrap().abs_diff(impact) <= 4);
            let strict = TouchSettings {
                threshold: 0.99,
                ..reference
            };
            let (_, strict_events, _) = detect_stream(
                &source,
                &root.join(format!("strict-{impact}.jsonl")),
                &sha,
                630,
                640,
                360,
                strict,
            )
            .unwrap();
            assert_eq!(
                strict_events, 0,
                "A CPU threshold rerun must not reuse outdated contact decisions"
            );
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_checksum_or_truncated_rows_rejected() {
        let root = std::env::temp_dir().join(format!("touch-bad-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let (bytes, sha) = sample_frames(true, 320);
        let source = root.join("raw.jsonl");
        fs::write(&source, &bytes).unwrap();
        let err = detect_stream(
            &source,
            &root.join("bad.jsonl"),
            &"0".repeat(64),
            630,
            640,
            360,
            TouchSettings::default(),
        )
        .unwrap_err();
        assert!(err.contains("SHA-256"));
        fs::write(&source, &bytes[..bytes.len() / 2]).unwrap();
        assert!(detect_stream(
            &source,
            &root.join("truncated.jsonl"),
            &sha,
            630,
            640,
            360,
            TouchSettings::default()
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod local_tuning_diagnostic {
    use super::*;

    #[test]
    #[ignore = "requires VOLLEY_TOUCH_ECHO_FIXTURE with isolated replay and pose evidence"]
    fn recorded_rebound_at_7_8s_has_no_departure_echo() {
        let root = std::path::PathBuf::from(std::env::var("VOLLEY_TOUCH_ECHO_FIXTURE").unwrap());
        let source = root.join("tracked_frames.jsonl");
        let manifest: Value =
            serde_json::from_slice(&fs::read(root.join("touch_manifest.json")).unwrap()).unwrap();
        let settings: TouchSettings = serde_json::from_value(manifest["settings"].clone()).unwrap();
        let fusion = manifest["pose_source_fusion_id"].as_str().unwrap();
        let output = root.join(format!("touches-{}.jsonl", Uuid::new_v4()));
        detect_stream_with_pose(
            None,
            &source,
            &output,
            manifest["source_replay_sha256"].as_str().unwrap(),
            manifest["frames"].as_u64().unwrap(),
            1280,
            720,
            settings,
            true,
            Some((&root, fusion, true)),
            None,
            None,
            &[],
            None,
            None,
            None,
            &[],
            None,
            None,
            None,
        )
        .unwrap();
        let events: Vec<Value> = fs::read_to_string(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .filter(|event: &Value| (232..=242).contains(&event["frame_index"].as_u64().unwrap()))
            .collect();
        assert_eq!(
            events.len(),
            1,
            "one recorded rebound must produce one event: {events:?}"
        );
        assert_eq!(events[0]["frame_index"], 235);
    }

    #[test]
    #[ignore = "Use VOLLEY_TOUCH_REPLAY for isolated analysis; never writes production"]
    fn inspect_real_motion_candidates() {
        let path = std::env::var("VOLLEY_TOUCH_REPLAY").unwrap();
        let file = std::fs::read_to_string(path).unwrap();
        let frames = file
            .lines()
            .enumerate()
            .map(|(i, line)| {
                transform_frame(
                    serde_json::from_str::<TrackedFrame>(line).unwrap(),
                    i as u64,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let clean = reconstruct_track_gaps(&exclude_ball_teleports(&frames, 1280, 720));
        for (gap, interval, reach) in [(5, 0.2, 0.75), (5, 0.2, 0.90), (5, 0.2, 1.00)] {
            let settings = TouchSettings {
                min_gap: gap,
                min_interval_sec: interval,
                threshold: 0.48,
                max_reach_ratio: reach,
                ..TouchSettings::default()
            };
            let events =
                evaluate_block(&clean, 1280, 720, 0, clean.len() as u64, settings).unwrap();
            println!(
                "TUNING gap={gap} interval={interval} reach={reach}, count={} => {:?}",
                events.len(),
                events
                    .iter()
                    .map(|e| (
                        e.frame_index,
                        e.provisional_track_id,
                        (e.probability * 100.0).round() as i64,
                    ))
                    .collect::<Vec<_>>()
            );
        }
    }
}
