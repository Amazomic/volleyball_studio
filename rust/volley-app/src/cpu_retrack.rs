//! CPU-only replay of verified native RF-DETR/VBallNet fusion frames.
//! Reconstructs provisional player Track IDs without rerunning ONNX.
//! A tracklet ID is NOT an authenticated real athlete identity.
use crate::{
    inference_stream, preparation, project_guard, projects,
    storage::{
        self, safe_file, safe_folder, ApiError, ApiResult, BoundedReader,
        MAX_JSONL_ROW_BYTES as MAX_ROW_BYTES,
    },
    AppState,
};
use axum::{
    extract::{Query, State},
    http::HeaderMap,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use uuid::Uuid;
use volley_core::{
    fusion::Candidate,
    tracker::{bytetrack::ByteTracker, TrackConfirmation, TrackingOptions},
};

const MAX_MANIFEST_BYTES: u64 = 16 * 1024;
const MAX_INPUT_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const OUTPUT_FILE: &str = "tracked_frames.jsonl";
pub(crate) const TRACK_ASSOCIATION_VERSION: &str = "bytetrack_rally_pose_gap_v23";
pub(crate) mod ball_track;
mod gap_interpolation;
mod occlusion;
mod pose_refinement;
mod pose_source;
mod track_stitch;
pub(crate) use pose_refinement::{
    Refinement as PoseRefinement, LEGACY_POLICY as LEGACY_POSE_REFINEMENT_POLICY,
    POLICY as POSE_REFINEMENT_POLICY, PREVIOUS_POLICY as PREVIOUS_POSE_REFINEMENT_POLICY,
    PREVIOUS_TRACKED_CROP_POLICY as PREVIOUS_TRACKED_CROP_REFINEMENT_POLICY,
    PRIOR_CROP_POLICY as PRIOR_CROP_REFINEMENT_POLICY,
    REJECTED_LEADING_POLICY as HISTORICAL_LEADING_CROP_POLICY,
};

pub(crate) use volley_core::contracts::job_id as opaque_id;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayRequest {
    pub(crate) project_id: String,
    pub(crate) video: String,
    pub(crate) fusion_id: String,
    #[serde(default)]
    pub(crate) settings: Option<TrackingOptions>,
    #[serde(default = "default_interpolation_enabled")]
    pub(crate) interpolate_players: bool,
    #[serde(default = "default_interpolation_enabled")]
    pub(crate) interpolate_pose: bool,
}
const fn default_interpolation_enabled() -> bool {
    true
}
#[derive(Deserialize)]
pub struct ReplayQuery {
    project_id: String,
    replay_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FusedFrame {
    frame_index: u64,
    player_candidates: Vec<Candidate>,
    ball_candidates: Vec<Value>,
    selected_ball: Option<Value>,
    player_track_ids_confirmed: bool,
    touches_confirmed: bool,
}

#[derive(Clone, Debug)]
struct PlayerFrameScope {
    selected: Vec<(u64, u64)>,
    rallies: Vec<(u64, u64)>,
    ball_job_id: String,
    mask_sha256: String,
}

impl PlayerFrameScope {
    fn interval(&self, frame: u64) -> Option<(u64, u64)> {
        let index = self.selected.partition_point(|&(start, _)| start <= frame);
        self.selected
            .get(index.checked_sub(1)?)
            .copied()
            .filter(|&(_, end)| frame <= end)
    }

    fn contains(&self, frame: u64) -> bool {
        self.interval(frame).is_some()
    }

    fn same_interval(&self, first: u64, last: u64) -> bool {
        self.interval(first)
            .is_some_and(|(start, end)| first >= start && last <= end)
    }

    fn same_confirmed_rally(&self, first: u64, last: u64) -> bool {
        let index = self.rallies.partition_point(|&(start, _)| start <= first);
        index > 0 && last <= self.rallies[index - 1].1
    }

    fn frame_count(&self) -> u64 {
        self.selected
            .iter()
            .map(|&(first, last)| last - first + 1)
            .sum()
    }
}

fn load_player_frame_scope(
    project_root: &Path,
    fusion: &Value,
    replay_frames: u64,
) -> Result<Option<PlayerFrameScope>, String> {
    let Some(ball_job_id) = fusion["sources"]["vballnet"]["job_id"].as_str() else {
        return Ok(None);
    };
    if !opaque_id(ball_job_id) {
        return Err("Invalid ball GPU source in fusion".into());
    }
    let root = project_root.join("native_gpu_runs").join(ball_job_id);
    let file = root.join("rally_mask.json");
    if !file.exists() {
        return Ok(None);
    }
    let file = safe_file(&root, "rally_mask.json", 64 * 1024 * 1024)?;
    let bytes = fs::read(&file).map_err(|error| format!("Read player rally scope: {error}"))?;
    let evidence: Value =
        serde_json::from_slice(&bytes).map_err(|_| "Invalid saved primary rally scope")?;
    let mask = &evidence["mask"];
    let source_frames = mask["source_frame_count"]
        .as_u64()
        .ok_or("Missing primary rally frame count")?;
    let selected: Vec<(u64, u64)> = serde_json::from_value(mask["selected"].clone())
        .map_err(|_| "Invalid selected player rally intervals")?;
    let total_selected: u64 = selected
        .iter()
        .try_fold(0_u64, |count, &(first, last)| {
            count.checked_add(last.checked_sub(first)?.checked_add(1)?)
        })
        .ok_or("Invalid selected player rally interval length")?;
    if selected.is_empty()
        || evidence["schema"] != "volley-primary-rally-mask-evidence/1"
        || mask["schema"] != "volley-primary-rally-mask/1"
        || mask["movement_filter_version"] != "stationary_ball_min_rally_5s_v3"
        || evidence["ball_job_id"] != ball_job_id
        || evidence["original_source"] != fusion["original_source"]
        || evidence["analysis_range"] != fusion["analysis_range"]
        || evidence["selected_frames"].as_u64() != Some(total_selected)
        || mask["selected_frames"].as_u64() != Some(total_selected)
    {
        return Err("Primary player frame scope does not match GPU fusion".into());
    }
    let selected =
        crate::cpu_touches::align_primary_intervals(&selected, source_frames, replay_frames)?;
    let rallies = match mask.get("merged") {
        Some(value) => {
            let intervals = serde_json::from_value::<Vec<(u64, u64)>>(value.clone())
                .map_err(|_| "Invalid confirmed rally intervals for player tracking")?;
            crate::cpu_touches::align_primary_intervals(&intervals, source_frames, replay_frames)?
        }
        None => Vec::new(),
    };
    Ok(Some(PlayerFrameScope {
        selected,
        rallies,
        ball_job_id: ball_job_id.to_owned(),
        mask_sha256: format!("{:x}", Sha256::digest(&bytes)),
    }))
}

#[derive(Default)]
struct ReplayTotals {
    frames: u64,
    detections: u64,
    raw_player_detections: u64,
    rejected: u64,
    ambiguous: u64,
    frames_with_ball: u64,
    tentative_tracks: u64,
    unconfirmed_tracks: u64,
    short_track_detections: u64,
    static_tracks: u64,
    static_track_detections: u64,
    duplicate_track_ids_merged: u64,
    identity_turnover_splits: u64,
    identity_handoff_splits: u64,
    pose_detections: u64,
    unassigned_rally_boxes: u64,
    combined_pose_crops: u64,
    interpolated_player_boxes: u64,
    processed_player_frames: u64,
    bytetrack_low_confidence_matches: u64,
    bytetrack_lost_low_confidence_matches: u64,
}

/// Replay once with provisional IDs, then filter complete-video track evidence
/// before publishing any derived Track IDs. A single streaming pass cannot
/// retract the first frame of an ID later shown to be a two-frame false positive.
#[cfg(test)]
fn replay_frames(
    source: &Path,
    output: &Path,
    count: u64,
    config: TrackingOptions,
    expected_sha: &str,
) -> Result<(String, ReplayTotals, u64), String> {
    if !(1..=48 * 3600 * 30).contains(&count) {
        return Err("Invalid fused CPU frame count".into());
    }
    replay_frames_with_pose(source, output, count, config, expected_sha, None)
}

#[cfg(test)]
fn replay_frames_with_pose(
    source: &Path,
    output: &Path,
    count: u64,
    config: TrackingOptions,
    expected_sha: &str,
    pose: Option<&pose_source::Snapshot>,
) -> Result<(String, ReplayTotals, u64), String> {
    replay_frames_with_pose_options(
        source,
        output,
        count,
        config,
        expected_sha,
        pose,
        ReplayOptions {
            phase: ReplayPhase::Normal,
            interpolate_players: true,
            scope: None,
            cancel: None,
        },
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReplayPhase {
    Normal,
    StablePlan,
    #[cfg(test)]
    Refined,
}

struct ReplayOptions<'a> {
    phase: ReplayPhase,
    interpolate_players: bool,
    scope: Option<&'a PlayerFrameScope>,
    cancel: Option<&'a AtomicBool>,
}

fn replay_frames_with_pose_options(
    source: &Path,
    output: &Path,
    count: u64,
    config: TrackingOptions,
    expected_sha: &str,
    pose: Option<&pose_source::Snapshot>,
    options: ReplayOptions<'_>,
) -> Result<(String, ReplayTotals, u64), String> {
    if !(1..=48 * 3600 * 30).contains(&count) {
        return Err("Invalid fused CPU frame count".into());
    }
    let config = config.validate()?;
    let tentative = output.with_extension("unconfirmed");
    let result = replay_frames_verified(
        source,
        (&tentative, output),
        count,
        config,
        expected_sha,
        pose,
        options,
    );
    // This intermediate output is never exposed as a CPU run, including errors.
    let _ = fs::remove_file(tentative);
    result
}

/// The CPU replay contains confirmed *observations*, plus bounded
/// inferred positions marked explicitly as such. The raw GPU proposals
/// remain byte-for-byte intact. Missing IDs may be filled only between two
/// real observations of that same confirmed Track ID, at most 3 seconds apart.
fn fill_confirmed_track_gaps(
    source: &Path,
    expected_sha: &str,
    frame_count: u64,
    scope: Option<&PlayerFrameScope>,
    cancel: Option<&AtomicBool>,
) -> Result<(String, u64), String> {
    use std::collections::{BTreeMap, VecDeque};

    let tmp = source.with_extension("continuity.partial");
    let mut input = BoundedReader::open(source)?;
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&tmp)
        .map_err(|e| format!("Create bounded tracking interpolation: {e}"))?;
    let mut output = BufWriter::new(file);
    let mut digest = Sha256::new();
    let mut queued: VecDeque<Value> = VecDeque::with_capacity(91);
    let mut anchors: BTreeMap<u64, (u64, Candidate)> = BTreeMap::new();
    let mut previous_roster = None;
    let mut inferred = 0_u64;
    let write_one =
        |record: Value, writer: &mut BufWriter<File>, hash: &mut Sha256| -> Result<(), String> {
            let mut bytes = serde_json::to_vec(&record).map_err(|e| e.to_string())?;
            bytes.push(b'\n');
            if bytes.len() > MAX_ROW_BYTES {
                return Err("Interpolated tracking frame exceeds 512 KiB".into());
            }
            writer.write_all(&bytes).map_err(|e| e.to_string())?;
            hash.update(&bytes);
            Ok(())
        };
    let iou = |a: [f64; 4], b: [f64; 4]| {
        let dx = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
        let dy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
        let s = |b: [f64; 4]| (b[2] - b[0]) * (b[3] - b[1]);
        dx * dy / (s(a) + s(b) - dx * dy).max(1e-9)
    };
    for current in 0..frame_count {
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        let bytes = input.line()?.ok_or("Confirmed tracking is truncated")?;
        let row: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if row["frame_index"].as_u64() != Some(current) {
            return Err("Confirmed tracking frames out of order".into());
        }
        let roster = row["rally_roster_window"].as_u64();
        if roster != previous_roster {
            anchors.clear();
            previous_roster = roster;
        }
        let observations: Vec<_> = row["tracked_players"]
            .as_array()
            .ok_or("Missing confirmed tracks")?
            .iter()
            .map(|entry| {
                Ok((
                    entry["track_id"]
                        .as_u64()
                        .ok_or("Invalid confirmed Track ID")?,
                    serde_json::from_value::<Candidate>(entry["detection"].clone())
                        .map_err(|e| format!("Confirmed Track candidate: {e}"))?,
                    entry["occlusion_bridge"]["left"]
                        .as_u64()
                        .zip(entry["occlusion_bridge"]["occluder"].as_u64()),
                ))
            })
            .collect::<Result<_, String>>()?;
        queued.push_back(row);
        anchors.retain(|_, (last, _)| current.saturating_sub(*last) <= 90);
        for (track_id, observation, bridge) in observations {
            if let Some((left, prior)) = anchors.get(&track_id) {
                let duration = current - left;
                let occluder = bridge.filter(|(start, _)| start == left).map(|(_, id)| id);
                if (2..=90).contains(&duration)
                    && prior.score.min(observation.score) >= 0.35
                    && scope.is_none_or(|mask| mask.same_interval(*left, current))
                {
                    let a = prior.bbox_xyxy_norm;
                    let b = observation.bbox_xyxy_norm;
                    let aw = a[2] - a[0];
                    let bw = b[2] - b[0];
                    let ah = a[3] - a[1];
                    let bh = b[3] - b[1];
                    let center = |r: [f64; 4]| [(r[0] + r[2]) * 0.5, (r[1] + r[3]) * 0.5];
                    let ca = center(a);
                    let cb = center(b);
                    let drift = (ca[0] - cb[0]).hypot(ca[1] - cb[1]);
                    if aw.min(bw) >= 0.005
                        && ah.min(bh) >= 0.02
                        && aw.max(bw) / aw.min(bw) <= if duration <= 30 { 3.0 } else { 1.8 }
                        && ah.max(bh) / ah.min(bh) <= 1.8
                        && drift <= (0.035 + (duration - 1) as f64 * 0.003).min(0.15)
                        // The missing ID may have been replaced by another
                        // source-observed ID during the gap. Do not paint a
                        // second synthetic body across an already tracked
                        // player (reproduced at video 1:19:35.7).
                        && !gap_interpolation::alternate_track_bridges_gap(
                            &queued, track_id, *left, prior, current, &observation,
                        )
                    {
                        for entry in queued.iter_mut().filter(|v| {
                            let f = v["frame_index"].as_u64().unwrap_or(u64::MAX);
                            f > *left && f < current
                        }) {
                            let at = entry["frame_index"].as_u64().unwrap();
                            let t = (at - left) as f64 / duration as f64;
                            let interpolated_box =
                                std::array::from_fn(|i| a[i] * (1.0 - t) + b[i] * t);
                            let mut estimate = prior.clone();
                            estimate.frame_index = at;
                            estimate.bbox_xyxy_norm = interpolated_box;
                            estimate.score = prior.score.min(observation.score) * 0.80;
                            estimate.query_index = Some(0);
                            estimate.touch_probability = None;
                            estimate.head_point_norm = prior
                                .head_point_norm
                                .zip(observation.head_point_norm)
                                .map(|(a, b)| std::array::from_fn(|i| a[i] * (1.0 - t) + b[i] * t));
                            estimate.validate(at - at % 9, "players")?;
                            let rows = entry["tracked_players"]
                                .as_array_mut()
                                .ok_or("Invalid queued tracking frame")?;
                            if rows.len() >= 128
                                || rows.iter().any(|v| {
                                    v["track_id"].as_u64() == Some(track_id)
                                        || serde_json::from_value::<Candidate>(
                                            v["detection"].clone(),
                                        )
                                        .is_ok_and(|present| {
                                            iou(present.bbox_xyxy_norm, interpolated_box) > 0.65
                                                || volley_core::tracker::same_head_body_duplicate(
                                                    &present, &estimate,
                                                )
                                                || volley_core::tracker::same_partial_body_duplicate(
                                                    &present, &estimate,
                                                )
                                        })
                                })
                            {
                                continue;
                            }
                            let mut estimate_row = json!({
                                "track_id":track_id,
                                "detection":estimate,
                                "measurement_source":"interpolated_track_bbox",
                                "is_verified_athlete":false
                            });
                            if let Some(occluder) = occluder {
                                estimate_row["occluding_track_id"] = json!(occluder);
                            }
                            rows.push(estimate_row);
                            if entry["interpolated_track_ids"].is_null() {
                                entry["interpolated_track_ids"] = json!([]);
                            }
                            entry["interpolated_track_ids"]
                                .as_array_mut()
                                .unwrap()
                                .push(json!(track_id));
                            inferred += 1;
                        }
                    }
                }
            }
            anchors.insert(track_id, (current, observation));
        }
        while queued.len() > 90 {
            write_one(queued.pop_front().unwrap(), &mut output, &mut digest)?;
        }
    }
    for record in queued {
        write_one(record, &mut output, &mut digest)?;
    }
    if input.line()?.is_some() || input.finish() != expected_sha {
        return Err("Confirmed tracking changed during interpolation".into());
    }
    output.flush().map_err(|e| e.to_string())?;
    output.get_ref().sync_all().map_err(|e| e.to_string())?;
    drop(output);
    // Atomic on the repository's existing filesystem.
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        return Err("cancelled".into());
    }
    fs::rename(&tmp, source).map_err(|e| format!("Commit confirmed tracking: {e}"))?;
    Ok((format!("{:x}", digest.finalize()), inferred))
}

fn replay_frames_verified(
    source: &Path,
    outputs: (&Path, &Path),
    count: u64,
    config: TrackingOptions,
    expected_sha: &str,
    pose: Option<&pose_source::Snapshot>,
    options: ReplayOptions<'_>,
) -> Result<(String, ReplayTotals, u64), String> {
    let ReplayOptions {
        interpolate_players,
        scope,
        cancel,
        phase,
    } = options;
    let minimum = if phase == ReplayPhase::StablePlan {
        1
    } else {
        config.min_track_observations
    };
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        return Err("cancelled".into());
    }
    let (tentative, output) = outputs;
    let mut reader = BoundedReader::open(source)?;
    let mut pose_rows = pose.map(pose_source::PoseRows::open).transpose()?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(tentative)
        .map_err(|e| format!("Create unconfirmed CPU tracking snapshot: {e}"))?;
    let mut writer = BufWriter::new(file);
    let mut tentative_sha = Sha256::new();
    let mut tracker = ByteTracker::new(config)?;
    let mut confirmations = TrackConfirmation::default();
    let mut totals = ReplayTotals::default();
    for expected_frame in 0..count {
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        let line = reader.line()?.ok_or("Fused frames truncated")?;
        let frame: FusedFrame =
            serde_json::from_slice(&line).map_err(|e| format!("Invalid fused frame JSON: {e}"))?;
        if frame.frame_index != expected_frame
            || frame.player_track_ids_confirmed
            || frame.touches_confirmed
            || frame.ball_candidates.len() > 2
        {
            return Err("Fusion frame index or raw status changed".into());
        }
        let player_frame = scope.is_none_or(|mask| mask.contains(expected_frame));
        if !player_frame && !frame.player_candidates.is_empty() {
            return Err("Player detections exist outside the GPU rally selection".into());
        }
        totals.raw_player_detections += frame.player_candidates.len() as u64;
        let (proposals, joints, pose_count) = if let Some(rows) = pose_rows.as_mut() {
            if player_frame {
                rows.frame(expected_frame, &frame.player_candidates)?
            } else {
                rows.skip_frame(expected_frame)?;
                (Vec::new(), Vec::new(), 0)
            }
        } else {
            let mut boxes = frame.player_candidates.clone();
            if phase == ReplayPhase::StablePlan {
                for candidate in &mut boxes {
                    candidate.head_point_norm = None;
                }
            }
            (boxes, Vec::new(), 0)
        };
        let tracking = if player_frame {
            let tracking = if phase == ReplayPhase::StablePlan {
                tracker.process_crop_frame(expected_frame, &proposals)?
            } else {
                tracker.process_frame_with_poses(expected_frame, &proposals, &joints)?
            };
            confirmations.observe(expected_frame, &tracking.tracks)?;
            totals.processed_player_frames += 1;
            totals.rejected += tracking.rejected_boxes as u64;
            totals.ambiguous += tracking.ambiguous_detections as u64;
            Some(tracking)
        } else {
            None
        };
        totals.pose_detections += pose_count;
        let mut tracked_players = serde_json::to_value(
            tracking
                .as_ref()
                .map(|row| row.tracks.as_slice())
                .unwrap_or(&[]),
        )
        .map_err(|e| format!("Serialize tracked pose observations: {e}"))?;
        if let Some(rows) = tracked_players.as_array_mut() {
            for row in rows.iter_mut() {
                let index = row["detection"]["query_index"].as_u64().unwrap_or(u64::MAX) as usize;
                let measured = joints.get(index).is_some_and(Option::is_some);
                row["measurement_source"] = json!(if measured {
                    "rtmpose_rgb"
                } else {
                    "detector_bbox"
                });
                if let Some(Some(pose)) = joints.get(index) {
                    row["pose"] = json!(pose.joints);
                    row["pose_interpolated"] = json!(false);
                    #[cfg(test)]
                    if phase == ReplayPhase::Refined {
                        row["pose_source"] = json!(POSE_REFINEMENT_POLICY);
                    }
                }
            }
            if scope.is_some() && phase == ReplayPhase::Normal {
                totals.combined_pose_crops +=
                    pose_refinement::stabilize_crops(rows, &frame.player_candidates);
            }
        }
        totals.frames += 1;
        let selected_ball = frame
            .selected_ball
            .filter(|ball| ball["source_model"] == "vballnet");
        if selected_ball.is_some() {
            totals.frames_with_ball += 1;
        }
        let mut bytes = serde_json::to_vec(&json!({
            "frame_index":expected_frame,
            "tracked_players":tracked_players,
            "raw_player_candidates":frame.player_candidates,
            "ball_candidates":frame.ball_candidates,
            "selected_ball":selected_ball,
            "ambiguous_detections":tracking.as_ref().map_or(0, |row| row.ambiguous_detections),
            "new_track_diagnostics":tracking.as_ref().map(|row| row.new_track_diagnostics.as_slice()).unwrap_or(&[]),
            "track_ids_provisional":true,
            "confirmed_athlete_identities":false,
            "touches_confirmed":false,
        }))
        .map_err(|e| format!("Serialize CPU tracking frame: {e}"))?;
        bytes.push(b'\n');
        if bytes.len() > MAX_ROW_BYTES {
            return Err("Tracked CPU frame exceeds 512 KiB".into());
        }
        writer
            .write_all(&bytes)
            .map_err(|e| format!("Write unconfirmed tracked frame: {e}"))?;
        tentative_sha.update(&bytes);
    }
    if reader.line()?.is_some() {
        return Err("Extra undeclared GPU fusion frames".into());
    }
    if let Some(rows) = pose_rows {
        rows.finish(count)?;
    }
    if reader.finish() != expected_sha {
        return Err("GPU fusion checksum mismatch during CPU replay".into());
    }
    writer
        .flush()
        .map_err(|e| format!("Flush tentative tracking: {e}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| format!("Sync tentative tracking: {e}"))?;
    drop(writer);
    let expected_tentative_sha = format!("{:x}", tentative_sha.finalize());
    let summary = confirmations.summary(minimum);
    totals.detections = summary.confirmed_detections;
    totals.unconfirmed_tracks = summary.discarded_tracks;
    totals.short_track_detections = summary.discarded_detections;
    totals.tentative_tracks = tracker.next_id() - 1;
    totals.bytetrack_low_confidence_matches = tracker.low_confidence_matches();
    totals.bytetrack_lost_low_confidence_matches = tracker.lost_low_confidence_matches();
    debug_assert_eq!(
        totals.tentative_tracks,
        summary.confirmed_tracks + summary.discarded_tracks
    );

    // Round two copies only confirmed tracks into the immutable CPU snapshot.
    // Ball evidence and the original raw candidate proposals stay untouched,
    // but Touch/Rally stages will see only actual published Track IDs.
    let mut tentative_reader = BoundedReader::open(tentative)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| format!("Create filtered CPU tracking output: {e}"))?;
    let mut writer = BufWriter::new(file);
    let mut sha = Sha256::new();
    let mut retained = 0_u64;
    let mut discarded = 0_u64;
    for expected_frame in 0..count {
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        let line = tentative_reader
            .line()?
            .ok_or("Unconfirmed tracking truncated")?;
        let mut frame: Value =
            serde_json::from_slice(&line).map_err(|e| format!("Invalid tentative frame: {e}"))?;
        if frame["frame_index"].as_u64() != Some(expected_frame) {
            return Err("Unconfirmed CPU frame index changed".into());
        }
        let players = frame["tracked_players"]
            .as_array_mut()
            .ok_or("Unconfirmed CPU tracks array missing")?;
        let before = players.len();
        players.retain(|player| {
            player["track_id"]
                .as_u64()
                .is_some_and(|id| confirmations.confirmed(id, minimum))
        });
        retained += players.len() as u64;
        discarded += (before - players.len()) as u64;
        let mut bytes = serde_json::to_vec(&frame)
            .map_err(|e| format!("Serialize filtered CPU tracking frame: {e}"))?;
        bytes.push(b'\n');
        if bytes.len() > MAX_ROW_BYTES {
            return Err("Filtered CPU frame exceeds 512 KiB".into());
        }
        writer
            .write_all(&bytes)
            .map_err(|e| format!("Write filtered CPU frame: {e}"))?;
        sha.update(&bytes);
    }
    if tentative_reader.line()?.is_some() {
        return Err("Unconfirmed tracking contains extra rows".into());
    }
    if tentative_reader.finish() != expected_tentative_sha
        || retained != summary.confirmed_detections
        || discarded != summary.discarded_detections
    {
        return Err("Tentative CPU tracking changed between verification and filtering".into());
    }
    writer
        .flush()
        .map_err(|e| format!("Flush confirmed CPU frames: {e}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| format!("Sync confirmed CPU frames: {e}"))?;
    drop(writer);
    let observed_sha = format!("{:x}", sha.finalize());
    let (stitched_sha, merged, removed_duplicate_boxes) =
        track_stitch::stitch(output, &observed_sha, count, scope, interpolate_players)?;
    totals.duplicate_track_ids_merged = merged;
    totals.detections = totals
        .detections
        .checked_sub(removed_duplicate_boxes)
        .ok_or("More duplicate detections removed than confirmed")?;
    totals.identity_turnover_splits = 0;
    totals.identity_handoff_splits = 0;
    let stitched_sha = if phase == ReplayPhase::StablePlan {
        let pruned = track_stitch::prune_unreliable(
            output,
            &stitched_sha,
            count,
            config.min_track_observations,
            cancel,
        )?;
        totals.unconfirmed_tracks += pruned.short_tracks;
        totals.short_track_detections += pruned.short_detections;
        totals.static_tracks = pruned.static_tracks;
        totals.static_track_detections = pruned.static_detections;
        totals.detections = totals
            .detections
            .checked_sub(pruned.short_detections + pruned.static_detections)
            .ok_or("Too many pruned detections")?;
        pruned.sha
    } else {
        stitched_sha
    };
    let mut roster_track_count = None;
    let stitched_sha = if phase == ReplayPhase::StablePlan {
        if let Some(scope) = scope.filter(|s| !s.rallies.is_empty()) {
            let roster =
                track_stitch::reconcile_rally_rosters(output, &stitched_sha, count, scope, cancel)?;
            roster_track_count = Some(roster.tracks);
            totals.unassigned_rally_boxes = roster.unmatched_boxes;
            totals.combined_pose_crops += roster.merged_boxes;
            totals.detections = totals
                .detections
                .checked_sub(roster.merged_boxes + roster.unmatched_boxes)
                .ok_or("Roster removed too many boxes")?;
            roster.sha
        } else {
            stitched_sha
        }
    } else {
        stitched_sha
    };
    let stitched_sha = if phase == ReplayPhase::StablePlan {
        pose_refinement::restore_partial_crops(output, &stitched_sha, count, scope, cancel)?
    } else {
        stitched_sha
    };
    let (continuous_sha, inferred_positions) = if interpolate_players {
        fill_confirmed_track_gaps(output, &stitched_sha, count, scope, cancel)?
    } else {
        (stitched_sha, 0)
    };
    let (continuous_sha, occluded_positions) =
        if interpolate_players && phase == ReplayPhase::StablePlan {
            occlusion::apply(output, &continuous_sha, count, cancel)?
        } else {
            (continuous_sha, 0)
        };
    totals.interpolated_player_boxes = inferred_positions + occluded_positions;
    totals.combined_pose_crops += removed_duplicate_boxes;
    let track_count = summary
        .confirmed_tracks
        .checked_sub(
            merged
                + if phase == ReplayPhase::StablePlan {
                    totals.unconfirmed_tracks + totals.static_tracks
                } else {
                    0
                },
        )
        .ok_or("More Track IDs stitched or discarded than confirmed")?;
    Ok((
        continuous_sha,
        totals,
        roster_track_count.unwrap_or(track_count),
    ))
}

fn confirmation_threshold(requested: u64, sample_period: Option<u64>) -> u64 {
    // The dense default requires 12 observed frames. At 3 Hz, requiring
    // six independent observations already represents about 1.7 seconds.
    if sample_period == Some(10) && requested == TrackingOptions::default().min_track_observations {
        6
    } else {
        requested
    }
}

pub(crate) fn has_refined_pose(
    project: &Path,
    fusion_id: &str,
    fusion_sha: &str,
    mask_sha: Option<&str>,
) -> Result<bool, String> {
    pose_source::locate_refined(project, fusion_id, fusion_sha, mask_sha, None).map(|s| s.is_some())
}

pub(crate) fn process(state: &AppState, req: &ReplayRequest) -> Result<Value, ApiError> {
    process_inner(state, req, None, None, |_| Ok(()))
}

pub(crate) enum TrackingProgress {
    BoxesReady(u64),
    PoseCached(u64),
    PoseFrames { processed: u64, total: u64 },
    FinalPass,
}

pub(crate) fn process_with_cancel(
    state: &AppState,
    req: &ReplayRequest,
    cancel: &AtomicBool,
    refinement: Option<PoseRefinement>,
    on_pose: impl FnMut(TrackingProgress) -> Result<(), String>,
) -> Result<Value, ApiError> {
    process_inner(state, req, Some(cancel), refinement, on_pose)
}

fn process_inner(
    state: &AppState,
    req: &ReplayRequest,
    cancel: Option<&AtomicBool>,
    refinement: Option<PoseRefinement>,
    mut on_pose: impl FnMut(TrackingProgress) -> Result<(), String>,
) -> Result<Value, ApiError> {
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        return Err(storage::conflict("cancelled"));
    }
    if !state.writes_enabled {
        return Err(storage::conflict(
            "CPU replay requires isolated VOLLEY_RUST_DATA_ROOT",
        ));
    }
    storage::project_id(&req.project_id)?;
    storage::video_name(&req.video)?;
    if !opaque_id(&req.fusion_id) {
        return Err(storage::bad_request("Invalid GPU fusion ID"));
    }
    let mut settings = req
        .settings
        .unwrap_or_default()
        .validate()
        .map_err(storage::bad_request)?;
    let _guard = project_guard::write_guard(state, &req.project_id)?;
    let project = projects::existing_project(state, &req.project_id)?;
    if !project["videos"]
        .as_array()
        .is_some_and(|a| a.iter().any(|v| v.as_str() == Some(&req.video)))
    {
        return Err(storage::bad_request(
            "CPU replay video does not belong to project",
        ));
    }
    let plan = preparation::plan_for_run(
        state,
        &req.project_id,
        &req.video,
        "native_gpu_fusions",
        &req.fusion_id,
    )?;
    let prepared = plan
        .ready_path()
        .ok_or_else(|| storage::bad_request("Prepared video is unavailable"))?;
    let (size, mtime) = inference_stream::source_stamp(prepared).map_err(storage::bad_request)?;
    let project_root = projects::project_root(state).join(&req.project_id);
    safe_folder(&project_root).map_err(storage::bad_request)?;
    let fusions = project_root.join("native_gpu_fusions");
    safe_folder(&fusions).map_err(storage::bad_request)?;
    let source_folder = fusions.join(&req.fusion_id);
    safe_folder(&source_folder).map_err(storage::bad_request)?;
    let manifest: Value = serde_json::from_slice(
        &fs::read(
            safe_file(&source_folder, "manifest.json", MAX_MANIFEST_BYTES)
                .map_err(storage::bad_request)?,
        )
        .map_err(|e| storage::internal(format!("Read fusion manifest: {e}")))?,
    )
    .map_err(|e| storage::bad_request(format!("Invalid fusion manifest: {e}")))?;
    if manifest["schema"] != "volley-native-gpu-fusion/1"
        || manifest["state"] != "raw_candidates_fused"
        || manifest["project_id"].as_str() != Some(&req.project_id)
        || manifest["fusion_id"].as_str() != Some(&req.fusion_id)
        || manifest["video"].as_str() != Some(&req.video)
        || manifest["fps"] != 30
        || manifest["file"] != "frames.jsonl"
        || manifest["cpu_processing_complete"] != false
        || manifest["player_track_ids_confirmed"] != false
        || manifest["touches_confirmed"] != false
        || manifest["source_identity"] != json!({"bytes":size,"mtime_ns":mtime})
    {
        return Err(storage::bad_request(
            "GPU fusion does not match project or prepared video",
        ));
    }
    let frames = manifest["emitted_frames"]
        .as_u64()
        .filter(|v| *v > 0 && *v <= 48 * 3600 * 30)
        .ok_or_else(|| storage::bad_request("Invalid GPU fusion frame count"))?;
    volley_media::preparation::verify_source_provenance(&plan.source, &manifest["original_source"])
        .map_err(storage::conflict)?;
    if manifest["windows"].as_u64() != Some(frames / 9) || frames % 9 != 0 {
        return Err(storage::bad_request(
            "Fusion window count does not match emitted frames",
        ));
    }
    // Sparse inference reduces real observations by 10x. Preserve custom
    // thresholds, but calibrate the default for confirmation based on the
    // sampling metadata. Interpolated frames never count as measurements.
    let requested_min = settings.min_track_observations;
    settings.min_track_observations = confirmation_threshold(
        requested_min,
        manifest["player_sampling_period_frames"].as_u64(),
    );
    let expected = manifest["sha256"]
        .as_str()
        .filter(|v| {
            v.len() == 64
                && v.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        })
        .ok_or_else(|| storage::bad_request("Fusion SHA-256 missing or invalid"))?;
    let source =
        safe_file(&source_folder, "frames.jsonl", MAX_INPUT_BYTES).map_err(storage::bad_request)?;
    let player_scope =
        load_player_frame_scope(&project_root, &manifest, frames).map_err(storage::bad_request)?;
    // Pose evidence was measured on actual GPU RGB frames and bound to the
    // selected fusion SHA. Legacy fusions without RTMPose remain readable.
    let saved_refined = pose_source::locate_refined(
        &project_root,
        &req.fusion_id,
        expected,
        player_scope
            .as_ref()
            .map(|scope| scope.mask_sha256.as_str()),
        Some((
            &settings,
            refinement.as_ref().map(|r| r.stride),
            req.interpolate_players,
        )),
    )
    .map_err(storage::bad_request)?;
    let root = project_root.join("native_cpu_replays");
    match fs::symlink_metadata(&root) {
        Ok(meta) if meta.file_type().is_symlink() || !meta.is_dir() => {
            return Err(storage::bad_request("Unsafe CPU replay directory"))
        }
        Ok(_) => (),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => fs::create_dir(&root)
            .map_err(|e| storage::internal(format!("Create CPU replay root: {e}")))?,
        Err(e) => return Err(storage::internal(format!("Check CPU replay root: {e}"))),
    }
    let replay_id = Uuid::new_v4().simple().to_string();
    let directory = root.join(&replay_id);
    fs::create_dir(&directory).map_err(|e| storage::internal(format!("Create CPU replay: {e}")))?;
    let result = (|| -> Result<Value, String> {
        let part = directory.join("tracked_frames.jsonl.partial");
        let refine_pose = refinement.is_some() && player_scope.is_some() && saved_refined.is_none();
        let (mut digest, mut totals, track_count) = replay_frames_with_pose_options(
            &source,
            &part,
            frames,
            settings,
            expected,
            None,
            ReplayOptions {
                phase: if player_scope.is_some() {
                    ReplayPhase::StablePlan
                } else {
                    ReplayPhase::Normal
                },
                interpolate_players: req.interpolate_players,
                scope: player_scope.as_ref(),
                cancel,
            },
        )?;
        on_pose(TrackingProgress::BoxesReady(totals.tentative_tracks))?;
        if !refine_pose {
            on_pose(TrackingProgress::FinalPass)?;
        }
        let mut refined_poses = 0;
        let mut refined_pose_sha = saved_refined.as_ref().map(|s| s.sha.clone());
        let mut refined_pose_stride = saved_refined.as_ref().map(|s| s.stride);
        let mut refined_pose_owner = saved_refined.as_ref().map(|s| s.id.clone());
        let mut refine_input_sha = None;
        if let Some(snapshot) = &saved_refined {
            let manifest: Value = serde_json::from_slice(
                &fs::read(safe_file(
                    &snapshot.folder,
                    "manifest.json",
                    MAX_MANIFEST_BYTES,
                )?)
                .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            if manifest["pose_refinement_input_sha256"] != digest {
                return Err("RTMPose crop plan changed; run the primary GPU phase again".into());
            }
            refine_input_sha = Some(digest.clone());
            let (attached_sha, attached_poses, inferred_frames) =
                pose_refinement::attach(&part, &digest, frames, snapshot, cancel)?;
            digest = attached_sha;
            refined_poses = attached_poses;
            on_pose(TrackingProgress::PoseCached(inferred_frames))?;
            totals.pose_detections = refined_poses;
        }
        if refine_pose {
            let decoded_frames = manifest["decoded_frames"]
                .as_u64()
                .filter(|n| (frames..=frames + 8).contains(n))
                .ok_or("RTMPose refinement requires the original decoded frame count")?;
            let range = serde_json::from_value(manifest["analysis_range"].clone())
                .map_err(|e| format!("Invalid RTMPose refinement range: {e}"))?;
            refine_input_sha = Some(digest.clone());
            let refinement = refinement.ok_or("RTMPose refinement settings unavailable")?;
            let stride = refinement.stride;
            let (pose_sha, _accepted) = pose_refinement::refine(
                &state.repository_root,
                pose_refinement::Source {
                    original: &plan.source,
                    replay: &part,
                    replay_sha: &digest,
                    frames,
                    decoded_frames,
                    range,
                    scope: player_scope.as_ref().ok_or("Missing RTMPose rally scope")?,
                },
                refinement,
                cancel.ok_or("RTMPose refinement cancellation unavailable")?,
                |processed, total| on_pose(TrackingProgress::PoseFrames { processed, total }),
            )?;
            refined_pose_sha = Some(pose_sha.clone());
            refined_pose_stride = Some(stride);
            refined_pose_owner = Some(replay_id.clone());
            let snapshot = pose_source::Snapshot {
                id: replay_id.clone(),
                sha: pose_sha,
                stride,
                folder: directory.clone(),
            };
            on_pose(TrackingProgress::FinalPass)?;
            let (attached_sha, attached_poses, _) =
                pose_refinement::attach(&part, &digest, frames, &snapshot, cancel)?;
            digest = attached_sha;
            refined_poses = attached_poses;
            totals.pose_detections = refined_poses;
            volley_media::preparation::verify_source_provenance(
                &plan.source,
                &manifest["original_source"],
            )?;
        }
        let mut interpolated_poses = 0;
        if req.interpolate_pose && refined_pose_sha.is_some() {
            let (interpolated_sha, count) =
                pose_refinement::interpolate_gaps(&part, &digest, frames, cancel)?;
            digest = interpolated_sha;
            interpolated_poses = count;
        }
        if inference_stream::source_stamp(prepared)? != (size, mtime) {
            return Err("Prepared video changed during CPU replay".into());
        }
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        fs::rename(part, directory.join(OUTPUT_FILE))
            .map_err(|e| format!("Publish CPU replay output: {e}"))?;
        let mut record = json!({
            "schema":"volley-native-cpu-tracking/1",
            "state":"provisional_tracklets_rebuilt",
            "interpolated_positions_version":if req.interpolate_players {"bounded_same_track_3s_v2"} else {"disabled_v1"},
            "pose_storage_version":"measured_and_bounded_interpolated_coco17_v2",
            "interpolate_players":req.interpolate_players,
            "interpolate_pose":req.interpolate_pose,
            "project_id":req.project_id,
            "video":req.video,
            "replay_id":replay_id,
            "source_fusion_id":req.fusion_id,
            "source_fusion_sha256":expected,
             "track_association_version":TRACK_ASSOCIATION_VERSION,
            "source_pose_run_id":null,
            "source_pose_sha256":null,
            "pose_measured_track_proposals":totals.pose_detections,
            "ball_selection_policy":"vballnet_only_v1",
            "source_identity":{"bytes":size,"mtime_ns":mtime},
            "original_source":manifest["original_source"],
            "analysis_range":manifest["analysis_range"],
            "fps":30,
            "settings":settings,
            "frames":totals.frames,
            "tracked_player_detections":totals.detections,
            "raw_player_detections":totals.raw_player_detections,
            "discarded_player_detections":totals.rejected + totals.short_track_detections + totals.static_track_detections,
            "discarded_short_track_detections":totals.short_track_detections,
            "discarded_short_track_ids":totals.unconfirmed_tracks,
            "tentative_track_count":totals.tentative_tracks,
            "duplicate_track_ids_merged":totals.duplicate_track_ids_merged,
            "identity_turnover_version":"disabled_bytetrack_v1",
            "identity_turnover_splits":totals.identity_turnover_splits,
            "identity_handoff_splits":totals.identity_handoff_splits,
            "ambiguous_detections":totals.ambiguous,
            "frames_with_ball_proposals":totals.frames_with_ball,
            "provisional_track_count":track_count,
            "file":OUTPUT_FILE,
            "sha256":digest,
            "track_ids_provisional":true,
            "athlete_identities_confirmed":false,
            "touches_confirmed":false,
            "cpu_processing_complete":false,
        });
        record["interpolated_pose_count"] = json!(interpolated_poses);
        record["discarded_static_track_ids"] = json!(totals.static_tracks);
        record["discarded_static_track_detections"] = json!(totals.static_track_detections);
        record["unassigned_rally_boxes"] = json!(totals.unassigned_rally_boxes);
        let has_refined = refine_pose || saved_refined.is_some();
        record["pose_refinement_policy"] = json!(has_refined.then_some(POSE_REFINEMENT_POLICY));
        record["refined_pose_count"] = json!(refined_poses);
        record["refined_pose_sha256"] = json!(refined_pose_sha);
        record["refined_pose_stride"] = json!(refined_pose_stride);
        record["refined_pose_source_replay_id"] = json!(refined_pose_owner);
        record["refined_pose_model_sha256"] =
            json!(has_refined.then_some(volley_inference::pose::MODEL_SHA256));
        record["pose_refinement_input_sha256"] = json!(refine_input_sha);
        record["player_frame_scope_version"] = json!(if player_scope.is_some() {
            "primary_gpu_selected_intervals_v1"
        } else {
            "legacy_unscoped_v1"
        });
        record["player_frame_scope_ball_job_id"] = json!(player_scope
            .as_ref()
            .map(|scope| scope.ball_job_id.as_str()));
        record["player_frame_scope_sha256"] = json!(player_scope
            .as_ref()
            .map(|scope| scope.mask_sha256.as_str()));
        record["player_frame_scope_frames"] =
            json!(player_scope.as_ref().map(PlayerFrameScope::frame_count));
        record["player_frames_processed"] = json!(totals.processed_player_frames);
        record["track_detector_low_score"] =
            json!(volley_core::tracker::bytetrack::LOW_DETECTION_SCORE);
        record["track_detector_high_score"] =
            json!(volley_core::tracker::bytetrack::HIGH_DETECTION_SCORE);
        record["legacy_tracker_enabled"] = json!(false);
        record["bytetrack_low_confidence_matches"] = json!(totals.bytetrack_low_confidence_matches);
        record["bytetrack_lost_low_confidence_matches"] =
            json!(totals.bytetrack_lost_low_confidence_matches);
        record["pose_proposal_role"] = json!(if has_refined {
            "measured_pose_attached_to_stable_track_id"
        } else {
            "detector_boxes_only"
        });
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        storage::write_json_atomic(&directory.join("manifest.json"), &record)
            .map_err(|e| format!("Persist CPU tracking manifest: {e:?}"))?;
        crate::native_player_tracks::prepare_replay_index(
            &directory.join(OUTPUT_FILE),
            &digest,
            totals.frames,
        )?;
        Ok(record)
    })();
    match result {
        Ok(v) => Ok(v),
        Err(error) => {
            let _ = fs::remove_dir_all(&directory);
            Err(storage::bad_request(error))
        }
    }
}

pub async fn replay(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ReplayRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let result = tokio::task::spawn_blocking(move || process(&state, &request))
        .await
        .map_err(|e| storage::internal(format!("Native CPU tracking worker: {e}")))??;
    Ok(Json(result))
}
pub async fn status(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ReplayQuery>,
) -> ApiResult<Value> {
    storage::project_id(&query.project_id)?;
    if !opaque_id(&query.replay_id) {
        return Err(storage::bad_request("Invalid CPU replay ID"));
    }
    projects::existing_project(&state, &query.project_id)?;
    let project = projects::project_root(&state).join(&query.project_id);
    safe_folder(&project).map_err(storage::bad_request)?;
    let root = project.join("native_cpu_replays");
    safe_folder(&root).map_err(storage::bad_request)?;
    let folder = root.join(&query.replay_id);
    safe_folder(&folder).map_err(storage::bad_request)?;
    let value: Value = serde_json::from_slice(
        &fs::read(
            safe_file(&folder, "manifest.json", MAX_MANIFEST_BYTES)
                .map_err(storage::bad_request)?,
        )
        .map_err(|e| storage::internal(format!("Read CPU manifest: {e}")))?,
    )
    .map_err(|e| storage::bad_request(format!("Malformed CPU replay manifest: {e}")))?;
    if value["schema"] != "volley-native-cpu-tracking/1"
        || value["replay_id"].as_str() != Some(&query.replay_id)
        || value["project_id"].as_str() != Some(&query.project_id)
    {
        return Err(storage::bad_request(
            "CPU replay metadata belongs to a different project",
        ));
    }
    Ok(Json(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_crop_plan_uses_detector_observations_and_interpolates_before_pose() {
        let root = std::env::temp_dir().join(format!("volley-preliminary-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("fusion.jsonl");
        let output = root.join("preliminary.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..18_u64 {
            let players = if frame % 3 == 0 {
                vec![example(frame, 0.2, 0)]
            } else {
                Vec::new()
            };
            serde_json::to_writer(
                &mut bytes,
                &json!({
                    "frame_index":frame,"player_candidates":players,
                    "ball_candidates":[],"selected_ball":null,
                    "player_track_ids_confirmed":false,"touches_confirmed":false
                }),
            )
            .unwrap();
            bytes.push(b'\n');
        }
        fs::write(&source, &bytes).unwrap();
        let (_, totals, _) = replay_frames_with_pose_options(
            &source,
            &output,
            18,
            TrackingOptions {
                min_track_observations: 6,
                ..TrackingOptions::default()
            },
            &format!("{:x}", Sha256::digest(&bytes)),
            None,
            ReplayOptions {
                interpolate_players: true,
                scope: None,
                cancel: None,
                phase: ReplayPhase::StablePlan,
            },
        )
        .unwrap();
        assert_eq!(totals.unconfirmed_tracks, 0);
        assert!(totals.interpolated_player_boxes > 0);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn rally_scope_only_tracks_selected_frames_and_preserves_full_ball() {
        let root = std::env::temp_dir().join(format!("rally-cpu-scope-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("fusion.jsonl");
        let output = root.join("tracked.jsonl");
        let scope = PlayerFrameScope {
            rallies: Vec::new(),
            selected: vec![(0, 20), (50, 70)],
            ball_job_id: "0123456789abcdef0123456789abcdef".into(),
            mask_sha256: "a".repeat(64),
        };
        let mut source_bytes = Vec::new();
        for frame in 0..90_u64 {
            let players = if [0, 4, 8, 12, 16, 20, 50, 54, 58, 62, 66, 70].contains(&frame) {
                vec![example(frame, 0.2, 0)]
            } else {
                Vec::new()
            };
            let mut bytes = serde_json::to_vec(&json!({
                "frame_index":frame,
                "player_candidates":players,
                "ball_candidates":[],
                "selected_ball":{"source_model":"vballnet","frame_index":frame},
                "player_track_ids_confirmed":false,
                "touches_confirmed":false,
            }))
            .unwrap();
            bytes.push(b'\n');
            source_bytes.extend(bytes);
        }
        fs::write(&source, &source_bytes).unwrap();
        let hash = format!("{:x}", Sha256::digest(&source_bytes));
        let (digest, totals, _) = replay_frames_with_pose_options(
            &source,
            &output,
            90,
            TrackingOptions {
                min_track_observations: 6,
                ..TrackingOptions::default()
            },
            &hash,
            None,
            ReplayOptions {
                phase: ReplayPhase::Normal,
                interpolate_players: true,
                scope: Some(&scope),
                cancel: None,
            },
        )
        .unwrap();
        let result = fs::read(&output).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&result)), digest);
        let rows: Vec<Value> = result
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 90);
        assert_eq!(totals.frames, 90);
        assert_eq!(totals.processed_player_frames, 42);
        assert!(rows.iter().all(|row| !row["selected_ball"].is_null()));
        for row in &rows[21..50] {
            assert!(row["tracked_players"].as_array().unwrap().is_empty());
        }
        assert!(!rows[10]["tracked_players"].as_array().unwrap().is_empty());
        assert!(!rows[15]["tracked_players"].as_array().unwrap().is_empty());

        let bad_line = serde_json::to_vec(&json!({
            "frame_index":35,
            "player_candidates":[example(35, 0.2, 0)],
            "ball_candidates":[],
            "selected_ball":{"source_model":"vballnet","frame_index":35},
            "player_track_ids_confirmed":false,
            "touches_confirmed":false,
        }))
        .unwrap();
        let mut lines = source_bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(Vec::from)
            .collect::<Vec<_>>();
        lines[35] = bad_line;
        let corrupted: Vec<u8> = lines
            .into_iter()
            .flat_map(|mut row| {
                row.push(b'\n');
                row
            })
            .collect();
        fs::write(&source, &corrupted).unwrap();
        let bad_hash = format!("{:x}", Sha256::digest(&corrupted));
        let rejected = replay_frames_with_pose_options(
            &source,
            &root.join("invalid.jsonl"),
            90,
            TrackingOptions::default(),
            &bad_hash,
            None,
            ReplayOptions {
                phase: ReplayPhase::Normal,
                interpolate_players: false,
                scope: Some(&scope),
                cancel: None,
            },
        );
        assert!(
            matches!(rejected, Err(error) if error.contains("outside the GPU rally selection"))
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn player_scope_is_sha_bound_to_primary_gpu_mask_and_clips_tail() {
        let root = std::env::temp_dir().join(format!("rally-scope-source-{}", Uuid::new_v4()));
        let ball = "0123456789abcdef0123456789abcdef";
        let dir = root.join("native_gpu_runs").join(ball);
        fs::create_dir_all(&dir).unwrap();
        let fusion = json!({
            "sources":{"vballnet":{"job_id":ball}},
            "original_source":{"sha256":"source"},
            "analysis_range":null
        });
        let mut evidence = json!({
            "schema":"volley-primary-rally-mask-evidence/1",
            "ball_job_id":ball,
            "original_source":{"sha256":"source"},
            "analysis_range":null,
            "selected_frames":42,
            "mask":{
                "schema":"volley-primary-rally-mask/1",
                "source_frame_count":98,
                "movement_filter_version":"stationary_ball_min_rally_5s_v3",
                "selected_frames":42,
                "selected":[[0,20],[77,97]]
            }
        });
        storage::write_json_atomic(&dir.join("rally_mask.json"), &evidence).unwrap();
        let mask = load_player_frame_scope(&root, &fusion, 90)
            .unwrap()
            .unwrap();
        assert_eq!(mask.selected, vec![(0, 20), (77, 89)]);
        assert!(!mask.contains(50));
        assert!(mask.same_interval(77, 89));
        assert!(!mask.same_interval(20, 77));
        evidence["ball_job_id"] = json!("unrelated");
        storage::write_json_atomic(&dir.join("rally_mask.json"), &evidence).unwrap();
        assert!(load_player_frame_scope(&root, &fusion, 90).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cancelled_tracking_does_not_open_or_publish_files() {
        let cancelled = AtomicBool::new(true);
        let missing = std::env::temp_dir().join(format!("no-cpu-source-{}", Uuid::new_v4()));
        let output = missing.with_extension("jsonl");
        let result = replay_frames_with_pose_options(
            &missing,
            &output,
            9,
            TrackingOptions::default(),
            "no-sha",
            None,
            ReplayOptions {
                phase: ReplayPhase::Normal,
                interpolate_players: false,
                scope: None,
                cancel: Some(&cancelled),
            },
        );
        assert!(matches!(result, Err(ref error) if error == "cancelled"));
        assert!(!output.exists());
    }

    #[test]
    fn sparse_confirmation_uses_real_samples_without_loosening_custom_limits() {
        assert_eq!(confirmation_threshold(12, None), 12);
        assert_eq!(confirmation_threshold(12, Some(1)), 12);
        assert_eq!(confirmation_threshold(12, Some(10)), 6);
        assert_eq!(confirmation_threshold(20, Some(10)), 20);
    }
    fn example(frame: u64, x: f64, q: usize) -> Candidate {
        Candidate {
            frame_index: frame,
            class_name: "player".into(),
            score: 0.9,
            bbox_xyxy_norm: [x, 0.1, x + 0.1, 0.7],
            query_index: Some(q),
            touch_probability: Some(0.8),
            head_point_norm: None,
        }
    }
    #[test]
    fn cpu_replays_hash_bound_frames_and_provisional_ids() {
        let root = std::env::temp_dir().join(format!("volley-replay-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let file = root.join("source.jsonl");
        let mut bytes = Vec::new();
        for i in 0..18 {
            let item = json!({
                "frame_index":i,
                "player_candidates":if i<12 {
                    vec![example(i,0.2+i as f64*0.002,i as usize % 2)]
                }else{vec![]},
                "ball_candidates":if i==0 { vec![json!({"source_model":"unsupported","detection":{"frame_index":0,"class_name":"ball","score":0.9,"bbox_xyxy_norm":[0.2,0.2,0.3,0.3]}})] } else { Vec::<Value>::new() },
                "selected_ball":if i==0 { json!({"source_model":"unsupported","detection":{"frame_index":0,"class_name":"ball","score":0.9,"bbox_xyxy_norm":[0.2,0.2,0.3,0.3]}}) } else { Value::Null },
                "player_track_ids_confirmed":false,
                "touches_confirmed":false,
            });
            bytes.extend(serde_json::to_vec(&item).unwrap());
            bytes.push(b'\n');
        }
        fs::write(&file, &bytes).unwrap();
        let original_sha = format!("{:x}", Sha256::digest(&bytes));
        let output = root.join("out.jsonl");
        let (sha, counts, ids) = replay_frames(
            &file,
            &output,
            18,
            TrackingOptions::default(),
            &original_sha,
        )
        .unwrap();
        assert_eq!(ids, 1);
        assert_eq!(counts.detections, 12);
        assert_eq!(counts.frames, 18);
        assert_eq!(counts.tentative_tracks, 1);
        assert_eq!(counts.unconfirmed_tracks, 0);
        assert_eq!(
            sha,
            format!("{:x}", Sha256::digest(fs::read(&output).unwrap()))
        );
        let frames = fs::read_to_string(output)
            .unwrap()
            .lines()
            .map(|v| serde_json::from_str::<Value>(v).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(frames[0]["tracked_players"][0]["track_id"], 1);
        assert!(frames[0]["selected_ball"].is_null());
        assert_eq!(
            frames[0]["ball_candidates"][0]["source_model"],
            "unsupported"
        );
        assert_eq!(frames[1]["tracked_players"][0]["track_id"], 1);
        assert_eq!(
            frames[1]["tracked_players"][0]["detection"]["query_index"],
            1
        );
        assert_eq!(frames[1]["touches_confirmed"], false);
        assert_eq!(frames[11]["tracked_players"][0]["track_id"], 1);
        assert_eq!(frames[12]["tracked_players"].as_array().unwrap().len(), 0);
        let mut corrupted = bytes.clone();
        corrupted[10] = b' ';
        fs::write(&file, corrupted).unwrap();
        assert!(replay_frames(
            &file,
            &root.join("bad.jsonl"),
            18,
            TrackingOptions::default(),
            &original_sha
        )
        .is_err());
        assert!(!root.join("bad.jsonl").exists());
        assert!(!root.join("bad.unconfirmed").exists());
        fs::remove_dir_all(&root).unwrap();
    }
    #[test]
    fn short_gaps_survive_temporary_bbox_width_changes_at_10_seconds() {
        let root = std::env::temp_dir().join(format!("volley-width-gap-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("tracked.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..21_u64 {
            let bbox = match frame {
                0 => Some([0.394, 0.470, 0.414, 0.619]),
                10 => Some([0.367, 0.477, 0.420, 0.609]),
                20 => Some([0.374, 0.471, 0.392, 0.599]),
                _ => None,
            };
            let tracks = bbox
                .into_iter()
                .map(|bbox| {
                    let mut detection = example(frame, 0.4, 0);
                    detection.bbox_xyxy_norm = bbox;
                    detection.score = 0.6;
                    json!({"track_id":4,"detection":detection,"is_verified_athlete":false})
                })
                .collect::<Vec<_>>();
            bytes.extend(
                serde_json::to_vec(&json!({"frame_index":frame,"tracked_players":tracks})).unwrap(),
            );
            bytes.push(b'\n');
        }
        fs::write(&source, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let (_, inferred) = fill_confirmed_track_gaps(&source, &sha, 21, None, None).unwrap();
        assert_eq!(inferred, 18);
        let rows = fs::read_to_string(&source).unwrap();
        for line in rows.lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            assert_eq!(row["tracked_players"].as_array().unwrap().len(), 1);
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn real_player_detection_blocks_duplicate_interpolated_id() {
        let root = std::env::temp_dir().join(format!("volley-duplicate-gap-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("tracked.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..6_u64 {
            let mut tracked = Vec::new();
            if frame == 0 || frame == 5 {
                let mut full = example(frame, 0.50, 0);
                full.bbox_xyxy_norm = [0.50, 0.40, 0.54, 0.55];
                full.head_point_norm = Some([0.52, 0.41]);
                full.score = 0.80;
                tracked.push(json!({"track_id":1,"detection":full,"is_verified_athlete":false}));
            }
            if (1..=4).contains(&frame) {
                let mut crop = example(frame, 0.502, 1);
                crop.bbox_xyxy_norm = [0.502, 0.40, 0.538, 0.48];
                crop.head_point_norm = Some([0.521, 0.41]);
                crop.score = 0.70;
                tracked.push(json!({"track_id":2,"detection":crop,"is_verified_athlete":false}));
            }
            bytes.extend(
                serde_json::to_vec(&json!({
                    "frame_index":frame,"tracked_players":tracked,
                    "track_ids_provisional":true,"confirmed_athlete_identities":false,
                    "touches_confirmed":false
                }))
                .unwrap(),
            );
            bytes.push(b'\n');
        }
        fs::write(&source, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let (written, inferred) = fill_confirmed_track_gaps(&source, &sha, 6, None, None).unwrap();
        assert_eq!(inferred, 0);
        assert_eq!(
            written,
            format!("{:x}", Sha256::digest(fs::read(&source).unwrap()))
        );
        for (frame, line) in fs::read_to_string(&source).unwrap().lines().enumerate() {
            let row: Value = serde_json::from_str(line).unwrap();
            let tracks = row["tracked_players"].as_array().unwrap();
            assert_eq!(tracks.len(), 1, "duplicate on frame {frame}");
            assert_eq!(
                tracks[0]["track_id"],
                if frame == 0 || frame == 5 { 1 } else { 2 }
            );
        }
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn video_1h19m35s_does_not_interpolate_ghost_id_795_over_real_id_792() {
        // Observed, not interpolated player/pose positions from the
        // saved GPU/CPU evidence in mm, frames 143220..143290.
        // The neural detector sees one moving athlete, but the old tracker
        // switched its ID from 795 to 792 and back. The former interpolation
        // drew ID 795 over ID 792 for ~2 seconds, creating six boxes for five people.
        let anchors: &[(u64, u64, [f64; 4], [f64; 2])] = &[
            (
                0,
                795,
                [0.419879, 0.673657, 0.486111, 0.870334],
                [0.441386, 0.694498],
            ),
            (
                10,
                792,
                [0.396340, 0.687423, 0.463623, 0.944206],
                [0.428407, 0.714420],
            ),
            (
                20,
                792,
                [0.339293, 0.693956, 0.444178, 0.991796],
                [0.371909, 0.720245],
            ),
            (
                30,
                792,
                [0.300434, 0.691609, 0.374569, 1.0],
                [0.324086, 0.723238],
            ),
            (
                40,
                792,
                [0.275320, 0.688571, 0.334482, 0.967441],
                [0.298120, 0.715828],
            ),
            (
                50,
                792,
                [0.266388, 0.669926, 0.325408, 0.971117],
                [0.298848, 0.700619],
            ),
            (
                60,
                792,
                [0.273589, 0.663533, 0.340887, 0.969568],
                [0.315762, 0.693472],
            ),
            (
                70,
                795,
                [0.310650, 0.656498, 0.351806, 0.875109],
                [0.342388, 0.679225],
            ),
        ];
        let root = std::env::temp_dir().join(format!("volley-ghost-gap-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let file = root.join("confirmed.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..=70_u64 {
            let tracked = anchors
                .iter()
                .filter(|(f, _, _, _)| *f == frame)
                .map(|&(_, track_id, bbox_xyxy_norm, head)| {
                    json!({"track_id":track_id,"is_verified_athlete":false,
                    "detection":{
                        "frame_index":frame,"class_name":"player",
                        "score":0.70,"bbox_xyxy_norm":bbox_xyxy_norm,
                        "query_index":0,"head_point_norm":head
                    }})
                })
                .collect::<Vec<_>>();
            bytes.extend(
                serde_json::to_vec(&json!({
                    "frame_index":frame,"tracked_players":tracked
                }))
                .unwrap(),
            );
            bytes.push(b'\n');
        }
        fs::write(&file, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let (after, inferred) = fill_confirmed_track_gaps(&file, &sha, 71, None, None).unwrap();
        assert!(
            inferred > 0,
            "legitimate gaps in ID 792 remain interpolated"
        );
        assert_eq!(
            after,
            format!("{:x}", Sha256::digest(fs::read(&file).unwrap()))
        );
        for (index, line) in fs::read_to_string(&file).unwrap().lines().enumerate() {
            let row: Value = serde_json::from_str(line).unwrap();
            let ids = row["tracked_players"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["track_id"].as_u64().unwrap())
                .collect::<Vec<_>>();
            if index > 0 && index < 70 {
                assert!(!ids.contains(&795), "ghost ID 795 at frame offset {index}");
            }
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cpu_discards_short_ids_from_every_frame_but_preserves_raw_candidates() {
        let root = std::env::temp_dir().join(format!("volley-short-id-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("fusion.jsonl");
        let output = root.join("tracked.jsonl");
        let mut raw = Vec::new();
        for frame in 0..30_u64 {
            let mut players = Vec::new();
            if frame < 25 && !(12..=14).contains(&frame) {
                players.push(example(
                    frame,
                    0.2 + frame as f64 * 0.001,
                    frame as usize % 2,
                ));
            }
            if (5..=7).contains(&frame) {
                players.push(example(frame, 0.72, 2));
            }
            let item = json!({
                "frame_index":frame,
                "player_candidates":players,
                "ball_candidates":[],
                "selected_ball":null,
                "player_track_ids_confirmed":false,
                "touches_confirmed":false,
            });
            raw.extend(serde_json::to_vec(&item).unwrap());
            raw.push(b'\n');
        }
        fs::write(&source, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let (digest, totals, ids) =
            replay_frames(&source, &output, 30, TrackingOptions::default(), &sha).unwrap();
        assert_eq!(totals.raw_player_detections, 25);
        assert_eq!(totals.tentative_tracks, 2);
        assert_eq!(ids, 1);
        assert_eq!(totals.unconfirmed_tracks, 1);
        assert_eq!(totals.short_track_detections, 3);
        assert_eq!(totals.detections, 22);
        assert_eq!(
            digest,
            format!("{:x}", Sha256::digest(fs::read(&output).unwrap()))
        );
        let frames: Vec<Value> = fs::read_to_string(&output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        for frame in &frames {
            let tracks = frame["tracked_players"].as_array().unwrap();
            assert!(tracks.iter().all(|track| track["track_id"] == 1));
            assert!(tracks.len() <= 1);
        }
        assert_eq!(frames[5]["tracked_players"].as_array().unwrap().len(), 1);
        assert_eq!(
            frames[5]["raw_player_candidates"].as_array().unwrap().len(),
            2
        );
        assert_eq!(frames[13]["tracked_players"][0]["track_id"], 1);
        assert_eq!(frames[13]["interpolated_track_ids"][0], 1);
        assert!(frames[13]["raw_player_candidates"]
            .as_array()
            .unwrap()
            .is_empty());
        assert!(
            frames[13]["tracked_players"][0]["detection"]["score"]
                .as_f64()
                .unwrap()
                < frames[15]["tracked_players"][0]["detection"]["score"]
                    .as_f64()
                    .unwrap()
        );
        assert_eq!(
            frames[15]["tracked_players"][0]["track_id"], 1,
            "3-frame occlusion must not fragment the persistent athlete"
        );
        assert_eq!(fs::read(&source).unwrap(), raw);
        assert!(!root.join("tracked.unconfirmed").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn confirmation_threshold_12_is_inclusive_and_cannot_fabricate_ids() {
        let root = std::env::temp_dir().join(format!("volley-threshold-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("fusion.jsonl");
        let mut raw = Vec::new();
        for frame in 0..18 {
            raw.extend(
                serde_json::to_vec(&json!({
                    "frame_index":frame,
                    "player_candidates":if frame<11 {vec![example(frame,0.2,0)]}else{vec![]},
                    "ball_candidates":[],"selected_ball":null,
                    "player_track_ids_confirmed":false,"touches_confirmed":false
                }))
                .unwrap(),
            );
            raw.push(b'\n');
        }
        fs::write(&source, &raw).unwrap();
        let hash = format!("{:x}", Sha256::digest(&raw));
        let (_, counts, ids) = replay_frames(
            &source,
            &root.join("hidden.jsonl"),
            18,
            TrackingOptions::default(),
            &hash,
        )
        .unwrap();
        assert_eq!(ids, 0);
        assert_eq!(counts.unconfirmed_tracks, 1);
        assert_eq!(counts.short_track_detections, 11);
        let (_, counts, ids) = replay_frames(
            &source,
            &root.join("visible.jsonl"),
            18,
            TrackingOptions {
                min_track_observations: 11,
                ..TrackingOptions::default()
            },
            &hash,
        )
        .unwrap();
        assert_eq!(ids, 1);
        assert_eq!(counts.detections, 11);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cpu_recalc_versions_from_same_fusion_without_gpu() {
        use std::process::Command;
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let root = std::env::temp_dir().join(format!("volley-recalc-{}", Uuid::new_v4()));
        let repo = root.join("original");
        let data = root.join("sandbox");
        let id = "1234abcdef01";
        let video = "fixture.mp4";
        let fusion_id = "0123456789abcdef0123456789abcdef";
        fs::create_dir_all(repo.join("project")).unwrap();
        let project_dir = data.join("projects").join(id);
        let fusion_dir = project_dir.join("native_gpu_fusions").join(fusion_id);
        fs::create_dir_all(&fusion_dir).unwrap();
        let source = repo.join("project").join(video);
        let command = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30:duration=0.6",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-y",
            ])
            .arg(&source)
            .status()
            .unwrap();
        assert!(command.success());
        storage::write_json_atomic(
            &data.join("projects").join("projects.json"),
            &json!({"schema":1,"projects":[{"id":id,"videos":[video],"name":"Test"}]}),
        )
        .unwrap();
        let state = AppState {
            repository_root: repo.clone(),
            data_root: data.clone(),
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
        };
        let plan = preparation::plan_for(&state, id, video).unwrap();
        assert!(plan.ready_path().is_some());
        let (size, mtime) = inference_stream::source_stamp(&source).unwrap();
        let mut raw = Vec::new();
        for frame in 0..18_u64 {
            let proposal = Candidate {
                frame_index: frame,
                class_name: "player".into(),
                score: 0.8,
                bbox_xyxy_norm: [0.10, 0.20, 0.30, 0.70],
                query_index: Some((frame % 2) as usize),
                touch_probability: Some(0.7),
                head_point_norm: None,
            };
            raw.extend(
                serde_json::to_vec(&json!({
                    "frame_index":frame,
                    "player_candidates":[proposal],
                    "ball_candidates":[],
                    "selected_ball":null,
                    "player_track_ids_confirmed":false,
                    "touches_confirmed":false,
                }))
                .unwrap(),
            );
            raw.push(b'\n');
        }
        fs::write(fusion_dir.join("frames.jsonl"), &raw).unwrap();
        let rawsha = format!("{:x}", Sha256::digest(&raw));
        storage::write_json_atomic(
            &fusion_dir.join("manifest.json"),
            &json!({
                "schema":"volley-native-gpu-fusion/1",
                "state":"raw_candidates_fused",
                "project_id":id,"video":video,"fusion_id":fusion_id,
                "fps":30,"windows":2,"emitted_frames":18,
                "file":"frames.jsonl","sha256":rawsha,
                "source_identity":{"bytes":size,"mtime_ns":mtime},
                "cpu_processing_complete":false,
                "player_track_ids_confirmed":false,
                "touches_confirmed":false,
            }),
        )
        .unwrap();
        let request = ReplayRequest {
            project_id: id.into(),
            video: video.into(),
            fusion_id: fusion_id.into(),
            settings: None,
            interpolate_players: true,
            interpolate_pose: true,
        };
        let first = process(&state, &request).unwrap();
        assert_eq!(first["provisional_track_count"], 1);
        assert_eq!(first["tracked_player_detections"], 18);
        assert_eq!(first["raw_player_detections"], 18);
        assert_eq!(first["tentative_track_count"], 1);
        assert_eq!(first["discarded_short_track_ids"], 0);
        let stricter = ReplayRequest {
            settings: Some(TrackingOptions {
                min_player_score: 0.95,
                ..TrackingOptions::default()
            }),
            ..request
        };
        let second = process(&state, &stricter).unwrap();
        assert_eq!(second["provisional_track_count"], 0);
        assert_eq!(second["discarded_player_detections"], 18);
        assert_eq!(second["raw_player_detections"], 18);
        assert_eq!(second["tentative_track_count"], 0);
        assert_ne!(first["replay_id"], second["replay_id"]);
        assert_eq!(fs::read(fusion_dir.join("frames.jsonl")).unwrap(), raw);
        let first_id = first["replay_id"].as_str().unwrap();
        let first_manifest = data
            .join("projects")
            .join(id)
            .join("native_cpu_replays")
            .join(first_id)
            .join("manifest.json");
        assert_eq!(storage::read_json(&first_manifest).unwrap().unwrap(), first);
        assert!(!repo.join("results").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pose_first_replay_stabilizes_ids_without_detector_boxes_and_keeps_provenance() {
        let root = std::env::temp_dir().join(format!("volley-cpu-pose-{}", Uuid::new_v4()));
        let pose_dir = root.join("poses");
        fs::create_dir_all(&pose_dir).unwrap();
        let fusion = root.join("fusion.jsonl");
        let mut fusion_bytes = Vec::new();
        for frame in 0..90_u64 {
            let proposals = if frame % 10 == 0 {
                vec![
                    Candidate {
                        frame_index: frame,
                        class_name: "player".into(),
                        score: 0.84,
                        bbox_xyxy_norm: [0.18, 0.15, 0.35, 0.78],
                        query_index: Some(0),
                        touch_probability: None,
                        head_point_norm: Some([0.24, 0.2]),
                    },
                    Candidate {
                        frame_index: frame,
                        class_name: "player".into(),
                        score: 0.84,
                        bbox_xyxy_norm: [0.60, 0.15, 0.78, 0.78],
                        query_index: Some(1),
                        touch_probability: None,
                        head_point_norm: Some([0.66, 0.2]),
                    },
                ]
            } else {
                vec![]
            };
            let mut line = serde_json::to_vec(&json!({
                "frame_index":frame,"player_candidates":proposals,
                "ball_candidates":[],"selected_ball":null,
                "player_track_ids_confirmed":false,"touches_confirmed":false
            }))
            .unwrap();
            line.push(b'\n');
            fusion_bytes.extend(line);
        }
        fs::write(&fusion, &fusion_bytes).unwrap();
        let mut pose_bytes = Vec::new();
        for frame in (0..90_u64).step_by(3) {
            let person = |x: f64| {
                let keypoints = (0..17)
                    .map(|i| {
                        json!([
                            x + (i % 3) as f64 * 0.014,
                            0.20 + (i / 3) as f64 * 0.055,
                            0.96
                        ])
                    })
                    .collect::<Vec<_>>();
                json!({"box":[x-0.03,0.16,x+0.11,0.81],
                    "crop_origin":if frame%10==0 {"observed"}else{"interpolated_box"},
                    "keypoints":keypoints})
            };
            let mut line = serde_json::to_vec(&json!({
                "frame":frame,"players":[person(0.21+frame as f64*0.0003),person(0.63)]
            }))
            .unwrap();
            line.push(b'\n');
            pose_bytes.extend(line);
        }
        fs::write(pose_dir.join("poses.jsonl"), &pose_bytes).unwrap();
        let snapshot = pose_source::Snapshot {
            id: "abc".into(),
            folder: pose_dir.clone(),
            stride: 3,
            sha: format!("{:x}", Sha256::digest(&pose_bytes)),
        };
        let output = root.join("tracked.jsonl");
        let fusion_sha = format!("{:x}", Sha256::digest(&fusion_bytes));
        let (sha, counts, ids) = replay_frames_with_pose_options(
            &fusion,
            &output,
            90,
            TrackingOptions {
                min_track_observations: 6,
                ..TrackingOptions::default()
            },
            &fusion_sha,
            Some(&snapshot),
            ReplayOptions {
                phase: ReplayPhase::Refined,
                interpolate_players: true,
                scope: None,
                cancel: None,
            },
        )
        .unwrap();
        assert_eq!(
            ids, 2,
            "each measured body must keep one provisional Track ID"
        );
        assert!(counts.pose_detections >= 59);
        let bytes = fs::read(&output).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), sha);
        let rows = String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        let at_three = &rows[3];
        assert_eq!(
            at_three["raw_player_candidates"].as_array().unwrap().len(),
            0
        );
        assert_eq!(at_three["tracked_players"].as_array().unwrap().len(), 2);
        assert!(at_three["tracked_players"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["pose"]
                .as_array()
                .is_some_and(|points| points.len() == 17)
                && p["pose_interpolated"] == false));
        assert!(at_three["tracked_players"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["measurement_source"] == "rtmpose_rgb"
                && p["pose_source"] == POSE_REFINEMENT_POLICY));
        let at_ten = &rows[10];
        assert_eq!(at_ten["raw_player_candidates"].as_array().unwrap().len(), 2);
        assert_eq!(
            at_ten["interpolated_track_ids"].as_array().unwrap().len(),
            2,
            "between real pose samples only the existing IDs may be visually interpolated"
        );
        assert!(
            at_ten["tracked_players"]
                .as_array()
                .unwrap()
                .iter()
                .all(|p| p["measurement_source"] == "interpolated_track_bbox"),
            "interpolated Track boxes must never claim a measured RTMPose observation"
        );
        let ids_at_three = at_three["tracked_players"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["track_id"].as_u64().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        let ids_at_seventy_five = rows[75]["tracked_players"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["track_id"].as_u64().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ids_at_three, ids_at_seventy_five);
        let without_gaps = root.join("tracked-no-interpolation.jsonl");
        let (disabled_sha, _, disabled_tracks) = replay_frames_with_pose_options(
            &fusion,
            &without_gaps,
            90,
            TrackingOptions {
                min_track_observations: 6,
                ..TrackingOptions::default()
            },
            &fusion_sha,
            Some(&snapshot),
            ReplayOptions {
                phase: ReplayPhase::Normal,
                interpolate_players: false,
                scope: None,
                cancel: None,
            },
        )
        .unwrap();
        let disabled_bytes = fs::read(&without_gaps).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&disabled_bytes)),
            disabled_sha
        );
        assert_eq!(disabled_tracks, 2);
        let off_rows = String::from_utf8(disabled_bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert!(off_rows[10]["tracked_players"]
            .as_array()
            .unwrap()
            .is_empty());
        assert!(off_rows[10].get("interpolated_track_ids").is_none());
        assert_eq!(off_rows[3]["tracked_players"].as_array().unwrap().len(), 2);
        assert!(fs::read(&fusion).unwrap() == fusion_bytes);
        assert!(fs::read(pose_dir.join("poses.jsonl")).unwrap() == pose_bytes);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reject_unsafe_count_and_invalid_id() {
        assert!(!opaque_id("../foo"));
        assert!(opaque_id("0123456789abcdef0123456789abcdef"));
        let folder = std::env::temp_dir().join(format!("cpu-safe-{}", Uuid::new_v4()));
        fs::create_dir_all(&folder).unwrap();
        assert!(replay_frames(
            &folder.join("missing"),
            &folder.join("out"),
            0,
            TrackingOptions::default(),
            "sha"
        )
        .is_err());
        fs::remove_dir_all(folder).unwrap();
    }
}
