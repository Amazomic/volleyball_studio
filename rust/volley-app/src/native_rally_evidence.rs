use crate::{
    cpu_retrack::opaque_id,
    inference_stream, preparation, projects,
    storage::{self, safe_file, safe_folder, ApiError, BoundedReader},
    AppState,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};
use volley_core::action_prediction::{ActionKind, ActionPrediction, PredictionMethod};
use volley_core::event_evidence::{
    bounded_clips, interpret_motion, verify_rally_interval, Interval, MotionInput,
};
use volley_core::rally_selector::{
    self, Choice as SharedChoice, GoldFilter, SourceFormat, SourceRally,
};

mod camera;
pub(crate) use camera::verified_camera_tracks;
#[cfg(test)]
mod tests;

const MAX_MANIFEST: u64 = 16 * 1024;
const MAX_EVIDENCE: u64 = 64 * 1024 * 1024 * 1024;
const MAX_PREVIEW: usize = 80;
const MAX_EVENTS_PER_CARD: usize = 48;
const MAX_RALLIES: usize = 200_000;
const DUPLICATE_TRACK_FRAMES: u64 = 4;

fn adjacent_same_track(last: Option<(u64, u64)>, track_id: u64, frame: u64) -> bool {
    last.is_some_and(|(id, previous)| {
        id == track_id && frame.saturating_sub(previous) <= DUPLICATE_TRACK_FRAMES
    })
}

pub(crate) fn retain_rally_touch(
    last: &mut BTreeMap<u64, u64>,
    frame: u64,
    track: u64,
    pose_verified: bool,
) -> bool {
    if !pose_verified && adjacent_same_track(last.get(&track).map(|&at| (track, at)), track, frame)
    {
        return false;
    }
    last.insert(track, frame);
    true
}

type ContactContext = (Option<u64>, bool, Option<ActionKind>);

pub(crate) struct FinalTouch {
    pub(crate) attributed: bool,
    pub(crate) pose_supported: bool,
    pub(crate) kind: Option<ActionKind>,
}

pub(crate) fn finalize_touch(
    event: &NativeMotion,
    in_rally: bool,
    context: Option<&ContactContext>,
    last: &mut BTreeMap<u64, u64>,
) -> Result<FinalTouch, String> {
    if context.is_some_and(|(id, _, _)| *id != event.provisional_track_id) {
        return Err("Rally context references a foreign-frame Track ID".into());
    }
    let pose_supported = event.body_contact_supported && event.contact_landmark.is_some();
    let attributed = in_rally
        && !context.is_some_and(|(_, ignored, _)| *ignored)
        && event
            .provisional_track_id
            .is_some_and(|id| retain_rally_touch(last, event.frame_index, id, pose_supported));
    let kind = if attributed && pose_supported {
        let predicted = event.predicted_kind();
        match context {
            Some((_, _, kind)) => predicted
                .and(*kind)
                .filter(|kind| *kind != ActionKind::LowTouch),
            None => predicted,
        }
    } else {
        None
    };
    Ok(FinalTouch {
        attributed,
        pose_supported: attributed && pose_supported,
        kind,
    })
}

#[derive(Debug, Deserialize)]
struct CourtTimelineContext {
    events: Vec<CourtTimelineEvent>,
}
#[derive(Debug, Deserialize)]
struct CourtTimelineEvent {
    frame_index: u64,
    provisional_track_id: Option<u64>,
    estimated_kind: Option<ActionKind>,
    ignored_serve_toss: bool,
}
// Physical contacts with a marked surface are provisional and never have
// an athlete Track ID. Keep them isolated from the player Touch stream.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeSurfaceEvent {
    frame_index: u64,
    surface: String,
    outcome: String,
    evidence: String,
    provisional: bool,
}
impl NativeSurfaceEvent {
    fn valid(&self, start: u64, end: u64) -> bool {
        self.provisional
            && (start..=end).contains(&self.frame_index)
            && matches!(
                (
                    self.surface.as_str(),
                    self.outcome.as_str(),
                    self.evidence.as_str()
                ),
                (
                    "floor",
                    "near_half" | "far_half" | "out" | "line_uncertain",
                    "nine_observed_ball_frames_vertical_reversal"
                ) | (
                    "net",
                    "possible_deflection",
                    "trajectory_deflection_inside_projected_net"
                )
            )
    }
}
#[derive(Debug, Deserialize)]
struct NativeRally {
    candidate_id: String,
    detected_start_frame: u64,
    detected_end_frame: u64,
    ball_observations: u64,
    ball_coverage: f64,
    motion_supported_touch_proposals: u64,
    #[serde(default)]
    method: Option<String>,
    first_touch_frame: u64,
    last_touch_frame: u64,
    max_touch_probability: f64,
    classification: String,
    authoritative_rally: bool,
    athlete_identities_verified: bool,
    serve_verified: bool,
    #[serde(default)]
    court_contact_context: Option<CourtTimelineContext>,
    #[serde(default)]
    surface_event_candidates: Vec<NativeSurfaceEvent>,
    reviewed: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeMotion {
    frame_index: u64,
    provisional_track_id: Option<u64>,
    probability: f64,
    ball_x_px: f64,
    ball_y_px: f64,
    motion_frame_index: u64,
    direction_change_deg: f64,
    speed_before_px_frame: f64,
    speed_after_px_frame: f64,
    velocity_residual_px_frame: f64,
    technique: String,
    #[serde(default)]
    action_prediction: Option<ActionPrediction>,
    #[serde(default)]
    court: Option<volley_core::court_geometry::CourtContactEvidence>,
    physical_motion_supported: bool,
    #[serde(default)]
    pub(crate) body_contact_supported: bool,
    #[serde(default)]
    pub(crate) contact_landmark: Option<volley_core::pose_contact::BodyContact>,
    #[serde(default)]
    contact_pose_sample_frame: Option<u64>,
    #[serde(default)]
    ball_occluded_at_contact: bool,
    #[serde(default)]
    inferred_body_reach_supported: bool,
    athlete_identity_verified: bool,
}
impl NativeMotion {
    pub(crate) fn event_key(&self) -> (u64, Option<u64>) {
        (self.frame_index, self.provisional_track_id)
    }
    pub(crate) fn is_physical(&self) -> bool {
        self.physical_motion_supported && !self.athlete_identity_verified
    }
    pub(crate) fn probability(&self) -> f64 {
        self.probability
    }
    pub(crate) fn ball_occluded_at_contact(&self) -> bool {
        self.ball_occluded_at_contact
    }
    pub(crate) fn predicted_kind(&self) -> Option<ActionKind> {
        self.action_prediction
            .as_ref()
            .filter(|prediction| {
                matches!(
                    prediction.method,
                    PredictionMethod::PoseKeypointsMotionV1
                        | PredictionMethod::LearnedPoseV1
                        | PredictionMethod::CourtPoseMotionV1
                )
            })
            .map(|prediction| prediction.kind)
            .filter(|kind| *kind != ActionKind::LowTouch)
    }
    pub(crate) fn valid(&self, frames: u64, width: u32, height: u32) -> bool {
        self.frame_index < frames
            && self.motion_frame_index < frames
            && self.motion_frame_index.abs_diff(self.frame_index)
                <= volley_core::touch_pipeline::MAX_OBSERVED_TOUCH_OFFSET_FRAMES
            && self.physical_motion_supported
            && (!self.body_contact_supported || self.provisional_track_id.is_some())
            && self.contact_pose_sample_frame.is_none_or(|at| {
                self.body_contact_supported
                    && self.contact_landmark.is_some()
                    && at < frames
                    && at.abs_diff(self.frame_index) <= 3
            })
            && self
                .contact_landmark
                .as_ref()
                .is_none_or(|part| part.valid(width, height))
            && (!self.ball_occluded_at_contact
                || self.body_contact_supported
                || self.inferred_body_reach_supported)
            && (!self.inferred_body_reach_supported
                || self.ball_occluded_at_contact && !self.body_contact_supported)
            && !self.athlete_identity_verified
            && self.technique == "unknown_touch"
            && self
                .action_prediction
                .as_ref()
                .is_none_or(|prediction| self.provisional_track_id.is_some() && prediction.valid())
            && self
                .court
                .is_none_or(|court| self.provisional_track_id.is_some() && court.valid())
            && self
                .provisional_track_id
                .is_none_or(|id| id > 0 && id <= i64::MAX as u64)
            && self.probability.is_finite()
            && (0.0..=1.0).contains(&self.probability)
            && self.ball_x_px.is_finite()
            && (0.0..=width as f64).contains(&self.ball_x_px)
            && self.ball_y_px.is_finite()
            && (0.0..=height as f64).contains(&self.ball_y_px)
    }
}
fn expected_id(replay_id: &str, touch_id: &str, start: u64, end: u64) -> String {
    let material = format!("{replay_id}:{touch_id}:{start}:{end}");
    format!("{:x}", Sha256::digest(material.as_bytes()))[..24].to_string()
}
fn checksum(value: &Value) -> Result<&str, String> {
    value
        .as_str()
        .filter(|s| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        })
        .ok_or("Missing SHA-256 input checksum".into())
}
pub(crate) fn read_manifest(dir: &Path, kind: &str) -> Result<Value, String> {
    safe_folder(dir)?;
    serde_json::from_slice(
        &fs::read(safe_file(dir, "manifest.json", MAX_MANIFEST)?)
            .map_err(|e| format!("Read {kind} manifest: {e}"))?,
    )
    .map_err(|e| format!("Malformed {kind} manifest: {e}"))
}
pub(crate) fn validate_ids(list: &[String]) -> Result<(), String> {
    if list.is_empty()
        || list.len() > MAX_PREVIEW
        || list.iter().any(|id| {
            id.len() != 24
                || !id
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        })
        || list.iter().collect::<HashSet<_>>().len() != list.len()
    {
        return Err("Select between 1 and 80 distinct saved rally IDs".into());
    }
    Ok(())
}
#[derive(Debug, Clone)]
pub(crate) struct ValidatedRun {
    pub(crate) project_id: String,
    pub(crate) video: String,
    pub(crate) run_id: String,
    pub(crate) project_source: PathBuf,
    pub(crate) rally_file: PathBuf,
    pub(crate) touch_file: PathBuf,
    pub(crate) replay_id: String,
    pub(crate) replay_sha: String,
    pub(crate) touch_id: String,
    pub(crate) rally_sha: String,
    pub(crate) touch_sha: String,
    pub(crate) frames: u64,
    pub(crate) source_duration_sec: f64,
    pub(crate) analysis_range: Option<volley_media::raw_decoder::AnalysisRange>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) context: u64,
    pub(crate) expected_rallies: u64,
    pub(crate) expected_touches: u64,
    /// Ball confidence chosen for this immutable CPU version.
    pub(crate) min_ball_score: f64,
}

pub(crate) fn inspect_project(
    state: &AppState,
    project_id: &str,
    rally_run_id: &str,
) -> Result<ValidatedRun, ApiError> {
    storage::project_id(project_id)?;
    if !opaque_id(rally_run_id) {
        return Err(storage::bad_request("Invalid native rally run ID"));
    }
    let project = projects::existing_project(state, project_id)?;
    let project_dir = projects::project_root(state).join(project_id);
    safe_folder(&project_dir).map_err(storage::bad_request)?;
    let run_root = project_dir.join("native_cpu_rallies");
    safe_folder(&run_root).map_err(storage::bad_request)?;
    let run_dir = run_root.join(rally_run_id);
    let rally = read_manifest(&run_dir, "rally").map_err(storage::bad_request)?;
    if rally["schema"] != "volley-native-cpu-rally-proposals/1"
        || rally["state"] != "provisional_rally_segments"
        || rally["project_id"].as_str() != Some(project_id)
        || rally["rally_run_id"].as_str() != Some(rally_run_id)
        || rally["file"] != "rally_candidates.jsonl"
        || rally["fps"] != 30
        || rally["rallies_authoritatively_confirmed"] != false
        || rally["athlete_identities_verified"] != false
        || rally["action_classification_complete"] != false
        || rally["montage_ready"] != false
        || rally["cpu_processing_complete"] != false
    {
        return Err(storage::bad_request(
            "Cannot preview an untrusted rally snapshot",
        ));
    }
    let video = rally["video"]
        .as_str()
        .ok_or_else(|| storage::bad_request("Missing saved rally video"))?
        .to_owned();
    storage::video_name(&video)?;
    if !project["videos"]
        .as_array()
        .is_some_and(|a| a.iter().any(|entry| entry.as_str() == Some(&video)))
    {
        return Err(storage::bad_request(
            "Saved rally video does not belong to project",
        ));
    }
    let replay_id = rally["source_replay_id"]
        .as_str()
        .filter(|id| opaque_id(id))
        .ok_or_else(|| storage::bad_request("Invalid source Track ID snapshot"))?
        .to_owned();
    let touch_id = rally["source_touch_id"]
        .as_str()
        .filter(|id| opaque_id(id))
        .ok_or_else(|| storage::bad_request("Invalid source Touch snapshot"))?
        .to_owned();
    let touch_root = project_dir.join("native_cpu_touch_runs");
    safe_folder(&touch_root).map_err(storage::bad_request)?;
    let touch_dir = touch_root.join(&touch_id);
    let touch = read_manifest(&touch_dir, "Touch").map_err(storage::bad_request)?;
    let plan = preparation::plan_for_run(
        state,
        project_id,
        &video,
        "native_cpu_rallies",
        rally_run_id,
    )?;
    let prepared = plan
        .ready_path()
        .ok_or_else(|| storage::bad_request("Prepared source is not ready"))?;
    let (bytes, mtime) = inference_stream::source_stamp(prepared).map_err(storage::bad_request)?;
    let identity = json!({"bytes":bytes,"mtime_ns":mtime});
    let dimensions = volley_media::probe(prepared).map_err(storage::bad_request)?;
    let frames = rally["frames"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= 48 * 3600 * 30)
        .ok_or_else(|| storage::bad_request("Invalid saved rally video length"))?;
    let count = rally["rally_candidates"]
        .as_u64()
        .filter(|n| *n <= MAX_RALLIES as u64)
        .ok_or_else(|| storage::bad_request("Invalid saved rally candidate count"))?;
    let touches = rally["touch_candidates_ingested"]
        .as_u64()
        .filter(|n| *n <= frames)
        .ok_or_else(|| storage::bad_request("Invalid saved Touch candidate count"))?;
    if plan.identity.fps != 30
        || rally["width"].as_u64() != Some(dimensions.width as u64)
        || rally["height"].as_u64() != Some(dimensions.height as u64)
        || rally["source_identity"] != identity
        || !matches!(
            touch["schema"].as_str(),
            Some("volley-native-cpu-touch-proposals/1")
        )
        || touch["state"] != "motion_gated_touch_proposals"
        || touch["touch_id"].as_str() != Some(&touch_id)
        || touch["project_id"].as_str() != Some(project_id)
        || touch["video"].as_str() != Some(&video)
        || touch["source_replay_id"].as_str() != Some(&replay_id)
        || touch["source_identity"] != identity
        || touch["frames"].as_u64() != Some(frames)
        || touch["width"].as_u64() != Some(dimensions.width as u64)
        || touch["height"].as_u64() != Some(dimensions.height as u64)
        || touch["file"] != "touch_candidates.jsonl"
        || touch["touch_candidates"].as_u64() != Some(touches)
        || touch["contacts_authoritatively_confirmed"] != false
        || touch["athlete_identities_verified"] != false
        || touch["cpu_processing_complete"] != false
        || rally["source_touch_sha256"] != touch["sha256"]
        || rally["source_replay_sha256"] != touch["source_replay_sha256"]
    {
        return Err(storage::bad_request(
            "Rally/Touch source version or video identity mismatch",
        ));
    }
    let rally_sha = checksum(&rally["sha256"])
        .map_err(storage::bad_request)?
        .to_owned();
    let touch_sha = checksum(&touch["sha256"])
        .map_err(storage::bad_request)?
        .to_owned();
    if rally["source_touch_sha256"].as_str() != Some(touch_sha.as_str()) {
        return Err(storage::bad_request(
            "Touch SHA does not match recorded source",
        ));
    }
    let context = rally["settings"]["suggested_context_frames"]
        .as_u64()
        .filter(|n| *n <= 180)
        .ok_or_else(|| storage::bad_request("Invalid saved context padding"))?;
    volley_media::preparation::verify_source_provenance(&plan.source, &rally["original_source"])
        .map_err(storage::conflict)?;
    if rally["analysis_range"] != touch["analysis_range"] {
        return Err(storage::conflict(
            "Rally and Touch analysis intervals differ",
        ));
    }
    if rally["original_source"] != touch["original_source"] {
        return Err(storage::conflict(
            "Rally and Touch original content differs",
        ));
    }
    let source_info = volley_media::probe(&plan.source).map_err(storage::bad_request)?;
    let analysis_range: Option<volley_media::raw_decoder::AnalysisRange> =
        serde_json::from_value(rally["analysis_range"].clone())
            .map_err(|e| storage::bad_request(format!("Invalid saved analysis range: {e}")))?;
    if let Some(range) = analysis_range {
        range
            .validate(source_info.duration_sec)
            .map_err(storage::bad_request)?;
    }
    let analyzed_duration =
        analysis_range.map_or(source_info.duration_sec, |r| r.end_sec - r.start_sec);
    if frames as f64 / 30.0 > analyzed_duration + 0.067 {
        return Err(storage::bad_request(
            "Rally source frames exceed original video's timeline",
        ));
    }
    Ok(ValidatedRun {
        project_id: project_id.into(),
        video,
        run_id: rally_run_id.into(),
        project_source: plan.source,
        rally_file: safe_file(&run_dir, "rally_candidates.jsonl", MAX_EVIDENCE)
            .map_err(storage::bad_request)?,
        touch_file: safe_file(&touch_dir, "touch_candidates.jsonl", MAX_EVIDENCE)
            .map_err(storage::bad_request)?,
        replay_sha: checksum(&rally["source_replay_sha256"])
            .map_err(storage::bad_request)?
            .to_owned(),
        replay_id,
        touch_id,
        rally_sha,
        touch_sha,
        frames,
        source_duration_sec: source_info.duration_sec,
        analysis_range,
        width: dimensions.width,
        height: dimensions.height,
        context,
        expected_rallies: count,
        expected_touches: touches,
        min_ball_score: rally["settings"]["min_ball_score"]
            .as_f64()
            .filter(|value| value.is_finite() && (0.01..=1.0).contains(value))
            .unwrap_or(0.35)
            .max(
                touch["settings"]["min_ball_score"]
                    .as_f64()
                    .filter(|value| value.is_finite() && (0.01..=1.0).contains(value))
                    .unwrap_or(0.0),
            ),
    })
}
#[derive(Clone)]
pub(crate) enum Choice {
    Page {
        offset: usize,
        limit: usize,
        filter: String,
        sort_by: String,
        marked: BTreeSet<String>,
    },
    Manual {
        ids: Vec<String>,
    },
    Timeline,
}
pub(crate) struct PreviewContent {
    pub(crate) value: Value,
    clips: Vec<volley_media::montage::SelectedClip>,
}
/// Hash-verify all source evidence; expose only 1..80 selected candidates.
pub(crate) fn generate(run: &ValidatedRun, choice: Choice) -> Result<PreviewContent, String> {
    let (offset, limit, manual, filter, sort_by, marked, timeline) = match choice {
        Choice::Page {
            offset,
            limit,
            filter,
            sort_by,
            marked,
        } => {
            if limit == 0
                || limit > MAX_PREVIEW
                || offset > MAX_RALLIES
                || !matches!(filter.as_str(), "all" | "marked" | "unmarked")
                || !matches!(sort_by.as_str(), "time" | "duration_desc" | "touches_desc")
            {
                return Err("Invalid preview page, gold filter or sort order".into());
            }
            (offset, limit, None, filter, sort_by, marked, false)
        }
        Choice::Manual { ids } => {
            validate_ids(&ids)?;
            (
                0,
                ids.len(),
                Some(ids),
                "all".into(),
                "time".into(),
                BTreeSet::new(),
                false,
            )
        }
        Choice::Timeline => (
            0,
            1,
            None,
            "all".into(),
            "time".into(),
            BTreeSet::new(),
            true,
        ),
    };
    let mut timeline_rallies = Vec::<Value>::new();
    let mut surface_timeline = Vec::<Value>::new();
    let mut timeline_contexts = HashMap::<u64, (Option<u64>, bool, Option<ActionKind>)>::new();
    let mut reader = BoundedReader::open(&run.rally_file)?;
    let mut records = Vec::<Value>::with_capacity(limit);
    let mut intervals = Vec::<Interval>::new();
    let mut all_ids = HashMap::<String, usize>::new();
    let mut order = Vec::<String>::new();
    let mut filtered_total = 0usize;
    let mut page_indices = Vec::<usize>::new();
    // Small verified index for global sort before applying offset/limit.
    let sorting = manual.is_none() && !timeline && sort_by != "time";
    let mut ranks = Vec::<(usize, u64, u64)>::new();
    while let Some(bytes) = reader.line()? {
        if intervals.len() >= MAX_RALLIES {
            return Err("Too many rally candidates".into());
        }
        let item: NativeRally =
            serde_json::from_slice(&bytes).map_err(|e| format!("Invalid saved rally row: {e}"))?;
        let candidate = if timeline {
            Value::Null
        } else {
            serde_json::from_slice::<Value>(&bytes)
                .map_err(|e| format!("Invalid candidate JSON: {e}"))?
        };
        let start = item.detected_start_frame;
        let end = item.detected_end_frame;
        if item.candidate_id != expected_id(&run.replay_id, &run.touch_id, start, end)
            || item.classification != "provisional_rally"
            || item.authoritative_rally
            || item.athlete_identities_verified
            || item.serve_verified
            || item.reviewed
            || start > end
            || end >= run.frames
            || intervals.last().is_some_and(|prior| prior.end >= start)
            || !item.ball_coverage.is_finite()
            || !(0.0..=1.0).contains(&item.ball_coverage)
            || !item.max_touch_probability.is_finite()
            || !(0.0..=1.0).contains(&item.max_touch_probability)
            || item.ball_observations > end - start + 1
            || (item.motion_supported_touch_proposals == 0
                && !matches!(
                    item.method.as_deref(),
                    Some("coherent_vballnet_motion_v2" | "primary_gpu_rally_mask_v1")
                ))
            || item.first_touch_frame < start
            || item.last_touch_frame > end
            || item.first_touch_frame > item.last_touch_frame
            || item.surface_event_candidates.len() > 256
            || item
                .surface_event_candidates
                .iter()
                .any(|event| !event.valid(start, end))
            || item
                .surface_event_candidates
                .windows(2)
                .any(|pair| pair[0].frame_index >= pair[1].frame_index)
            || all_ids
                .insert(item.candidate_id.clone(), intervals.len())
                .is_some()
        {
            return Err("Untrusted, duplicate or overlapping rally candidate".into());
        }
        let index = intervals.len();
        intervals.push(Interval { start, end });
        if timeline {
            for event in &item.surface_event_candidates {
                if surface_timeline.len() >= 30_000 {
                    return Err("Too many provisional surface events in timeline".into());
                }
                surface_timeline.push(json!({
                    "frame_index":event.frame_index,
                    "surface":event.surface,
                    "outcome":event.outcome,
                    "evidence":event.evidence,
                    "provisional":true,
                }));
            }
            if let Some(ref context) = item.court_contact_context {
                for touch in &context.events {
                    if !(start..=end).contains(&touch.frame_index)
                        || timeline_contexts
                            .insert(
                                touch.frame_index,
                                (
                                    touch.provisional_track_id,
                                    touch.ignored_serve_toss,
                                    touch.estimated_kind,
                                ),
                            )
                            .is_some()
                    {
                        return Err("Invalid or duplicate saved court rally contact context".into());
                    }
                }
            }
            timeline_rallies.push(json!({
                "candidate_id":item.candidate_id.clone(),
                "start_frame":start,
                "end_frame":end,
            }));
        }
        let chosen = if timeline {
            false
        } else if let Some(ids) = manual.as_ref() {
            ids.iter().any(|id| id == &item.candidate_id)
        } else {
            let eligible = match filter.as_str() {
                "marked" => marked.contains(&item.candidate_id),
                "unmarked" => !marked.contains(&item.candidate_id),
                _ => true,
            };
            if eligible {
                let position = filtered_total;
                filtered_total += 1;
                if sorting {
                    ranks.push((
                        index,
                        end - start + 1,
                        item.motion_supported_touch_proposals,
                    ));
                    false
                } else if position >= offset && position < offset.saturating_add(limit) {
                    page_indices.push(index);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        if chosen {
            records.push(candidate);
            order.push(item.candidate_id);
        }
    }
    if intervals.len() as u64 != run.expected_rallies || reader.finish() != run.rally_sha {
        return Err("Rally JSONL count or SHA-256 mismatch".into());
    }
    if sorting {
        // Rank the whole SHA-verified snapshot before paging, never just the
        // currently displayed cards. Ties retain original chronological order.
        match sort_by.as_str() {
            "duration_desc" => {
                ranks.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)))
            }
            "touches_desc" => {
                ranks.sort_unstable_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)))
            }
            _ => unreachable!("sort_by was validated"),
        }
        page_indices.extend(
            ranks
                .iter()
                .skip(offset)
                .take(limit)
                .map(|&(index, _, _)| index),
        );
        // Re-read only the requested card bodies, retaining page order.
        // The second source read is independently checksum-verified.
        let positions = page_indices
            .iter()
            .enumerate()
            .map(|(position, &index)| (index, position))
            .collect::<HashMap<_, _>>();
        let mut selected = vec![None::<Value>; page_indices.len()];
        let mut reread = BoundedReader::open(&run.rally_file)?;
        let mut index = 0usize;
        while let Some(bytes) = reread.line()? {
            if let Some(&position) = positions.get(&index) {
                selected[position] = Some(
                    serde_json::from_slice(&bytes)
                        .map_err(|e| format!("Invalid ranked rally row: {e}"))?,
                );
            }
            index += 1;
        }
        if index != intervals.len() || reread.finish() != run.rally_sha {
            return Err("Rally source changed during sorted preview".into());
        }
        for (position, record) in selected.into_iter().enumerate() {
            let record = record.ok_or("Missing ranked rally source row")?;
            let interval = intervals[page_indices[position]];
            let id = expected_id(&run.replay_id, &run.touch_id, interval.start, interval.end);
            if record["candidate_id"].as_str() != Some(id.as_str()) {
                return Err("Ranked rally ID changed during preview".into());
            }
            records.push(record);
            order.push(id);
        }
    }
    if let Some(ids) = manual.as_ref() {
        if records.len() != ids.len() {
            return Err("A selected rally ID is stale or absent".into());
        }
    }
    if timeline {
        // The complete source-scoped index is SHA-verified in one bounded pass,
        // independent of pagination or visual filters.
        let mut contacts = BoundedReader::open(&run.touch_file)?;
        let mut touch_frames = Vec::<u64>::new();
        let mut raw_touch_count = 0u64;
        let mut last_attributed = vec![BTreeMap::<u64, u64>::new(); intervals.len()];

        let mut estimated_actions = Vec::<Value>::new();
        // Complete postprocessing result, including physical Touch candidates
        // whose action type could not be classified. Never mix with GPU boxes.
        let mut touch_annotations = Vec::<Value>::new();
        let mut prior = None;
        while let Some(bytes) = contacts.line()? {
            let event: NativeMotion = serde_json::from_slice(&bytes)
                .map_err(|e| format!("Invalid native Touch evidence row: {e}"))?;
            if !event.valid(run.frames, run.width, run.height)
                || prior.is_some_and(|last| event.frame_index <= last)
            {
                return Err("Untrusted, nonphysical or unsorted native Touch row".into());
            }
            prior = Some(event.frame_index);
            raw_touch_count += 1;
            let Some(_) = event.provisional_track_id else {
                continue;
            };
            let contextual = timeline_contexts.get(&event.frame_index);
            let rally_index =
                intervals.partition_point(|interval| interval.end < event.frame_index);
            let in_rally = intervals
                .get(rally_index)
                .is_some_and(|interval| interval.start <= event.frame_index);
            let mut outside = BTreeMap::new();
            let final_touch = finalize_touch(
                &event,
                in_rally,
                contextual,
                if in_rally {
                    &mut last_attributed[rally_index]
                } else {
                    &mut outside
                },
            )?;
            if contextual.is_some_and(|(_, ignored, _)| *ignored) {
                continue;
            }
            touch_frames.push(event.frame_index);
            let touch_attributed = final_touch.attributed;
            let verified_contact = final_touch.pose_supported;
            let predicted_kind = final_touch.kind;
            touch_annotations.push(json!({
                "frame_index":event.frame_index,
                "kind":predicted_kind.map(ActionKind::as_str).unwrap_or("unknown_touch"),
                "ball_position_px":[event.ball_x_px,event.ball_y_px],
                "provisional_track_id":event.provisional_track_id,
                "contact_landmark":event.contact_landmark,
                "contact_pose_sample_frame":event.contact_pose_sample_frame,
                "contact_verified":verified_contact,
                "touch_attributed":touch_attributed,
                "assignment_method":if touch_attributed {if verified_contact {"pose"} else {"nearest_player"}} else {"candidate"},
            }));
            if let Some(kind) = predicted_kind {
                estimated_actions.push(json!({
                    "frame_index":event.frame_index,
                    "kind":kind.as_str(),
                    "ball_position_px":[event.ball_x_px,event.ball_y_px]
                }));
            }
        }
        if raw_touch_count != run.expected_touches || contacts.finish() != run.touch_sha {
            return Err("Touch count or SHA-256 mismatch during rally timeline".into());
        }
        return Ok(PreviewContent {
            value: json!({
                "schema": "volley-native-rally-timeline/1",
                "fps":30,
                "project_id": run.project_id,
                "rally_run_id": run.run_id,
                "video": run.video,
                "total_frames": run.frames,
                "video_duration_sec": run.source_duration_sec,
                "analysis_offset_sec": run.analysis_range.map_or(0.0, |r| r.start_sec),
                "analysis_range":run.analysis_range,
                "rallies": timeline_rallies,
                "touch_frames": touch_frames,
                "raw_touch_candidates": raw_touch_count,
                "estimated_actions": estimated_actions,
                "touch_annotations": touch_annotations,
                "surface_contact_candidates": surface_timeline,
                "contextual_contacts":timeline_contexts,
                "authoritative_game_events": false,
                "source_rallies_sha256": run.rally_sha,
                "source_touches_sha256": run.touch_sha,
                "source_replay_sha256": run.replay_sha,
            }),
            clips: Vec::new(),
        });
    }
    // The one typed selector validates immutable, source-scoped IDs before
    // either the preview page or a manual montage receives those intervals.
    let mut original_ids = vec![""; intervals.len()];
    for (id, idx) in &all_ids {
        original_ids[*idx] = id;
    }
    let source_refs = intervals
        .iter()
        .enumerate()
        .map(|(i, interval)| SourceRally {
            source: SourceFormat::NativeCpu,
            stable_id: original_ids[i],
            source_scope: &run.video,
            interval: *interval,
            marked: marked.contains(original_ids[i]),
        })
        .collect::<Vec<_>>();
    let core_indices = match manual.as_ref() {
        Some(ids) => rally_selector::select(&source_refs, SharedChoice::Manual(ids))?,
        None if sorting => {
            if page_indices.is_empty() {
                Vec::new()
            } else {
                let ids = page_indices
                    .iter()
                    .map(|&index| original_ids[index].to_owned())
                    .collect::<Vec<_>>();
                rally_selector::select(&source_refs, SharedChoice::Manual(&ids))?
            }
        }
        None => rally_selector::select(
            &source_refs,
            SharedChoice::Page {
                offset,
                limit,
                gold: GoldFilter::parse(&filter)?,
            },
        )?,
    };
    if manual.is_none() && core_indices != page_indices {
        return Err("Source-scoped preview selector disagrees with saved page".into());
    }
    let bounds = bounded_clips(&intervals, run.frames, run.context)?;
    let mut by_id = HashMap::<String, Value>::new();
    for (record, id) in records.into_iter().zip(order) {
        by_id.insert(id, record);
    }
    let selected_indices = core_indices;
    let mut cards = Vec::<Value>::with_capacity(selected_indices.len());
    let mut clips = Vec::with_capacity(selected_indices.len());
    for &i in &selected_indices {
        let interval = intervals[i];
        let id = expected_id(&run.replay_id, &run.touch_id, interval.start, interval.end);
        let mut record = by_id
            .remove(&id)
            .ok_or("Missing validated rally candidate")?;
        let clip = &bounds[i];
        let raw_clip = volley_media::montage::SelectedClip {
            video: run.video.clone(),
            start_frame: clip.start_frame,
            end_frame: clip.end_frame,
            analysis_offset_sec: Some(run.analysis_range.map_or(0.0, |r| r.start_sec)),
        };
        let scoped = verify_rally_interval("native_cpu", &id, interval, clip.clone(), run.frames)?;
        record["source_interval_contract"] = json!(scoped);
        record["manual_clip"] = json!({
            "video":run.video,"start_frame":clip.start_frame,
            "end_frame":clip.end_frame,
            "analysis_offset_sec":run.analysis_range.map_or(0.0, |r| r.start_sec)
        });
        record["clip_boundaries_provisional"] = json!(true);
        record["manual_selection_only"] = json!(true);
        record["motion_events"] = json!([]);
        record["motion_event_count"] = json!(0);
        record["raw_motion_event_count"] = json!(0);
        record["ignored_serve_toss_count"] = json!(0);
        // All events in the candidate, not merely the bounded 14 shown in the UI.
        // These are automatic hypotheses, never confirmed action statistics.
        record["estimated_action_counts"] = json!({});
        record["unclassified_action_count"] = json!(0);
        record["motion_event_preview_truncated"] = json!(false);
        cards.push(record);
        clips.push(raw_clip);
    }
    // Verify the complete Touch source as it is streamed, attaching only
    // physical evidence from intervals on the requested preview page.
    let mut contacts = BoundedReader::open(&run.touch_file)?;
    let mut prior = None;
    let mut count = 0u64;
    let mut last_attributed = vec![BTreeMap::<u64, u64>::new(); cards.len()];

    while let Some(bytes) = contacts.line()? {
        let event: NativeMotion = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Invalid native Touch evidence row: {e}"))?;
        if !event.valid(run.frames, run.width, run.height)
            || prior.is_some_and(|last| event.frame_index <= last)
        {
            return Err("Untrusted, nonphysical or unsorted native Touch row".into());
        }
        prior = Some(event.frame_index);
        count += 1;
        if let Some((idx, _)) = selected_indices.iter().enumerate().find(|(_, index)| {
            let interval = intervals[**index];
            event.frame_index >= interval.start && event.frame_index <= interval.end
        }) {
            let annotation = interpret_motion(MotionInput {
                probability: event.probability,
                direction_change_deg: event.direction_change_deg,
                speed_before_px_frame: event.speed_before_px_frame,
                speed_after_px_frame: event.speed_after_px_frame,
                velocity_residual_px_frame: event.velocity_residual_px_frame,
                provisional_track_id: event.provisional_track_id,
            })?;
            let row = &mut cards[idx];
            let raw = row["raw_motion_event_count"].as_u64().unwrap_or(0);
            row["raw_motion_event_count"] = json!(raw + 1);
            let Some(track_id) = event.provisional_track_id else {
                continue;
            };
            let context = row["court_contact_context"]["events"]
                .as_array()
                .and_then(|entries| {
                    entries
                        .iter()
                        .find(|entry| entry["frame_index"] == event.frame_index)
                })
                .cloned();
            if context.as_ref().is_some_and(|entry| {
                entry["provisional_track_id"].as_u64() != event.provisional_track_id
            }) {
                return Err("Rally context references a foreign-frame Track ID".into());
            }
            if context
                .as_ref()
                .is_some_and(|entry| entry["ignored_serve_toss"] == true)
            {
                let suppressed = row["ignored_serve_toss_count"].as_u64().unwrap_or(0);
                row["ignored_serve_toss_count"] = json!(suppressed + 1);
                continue;
            }
            let pose_verified = event.body_contact_supported && event.contact_landmark.is_some();
            if !retain_rally_touch(
                &mut last_attributed[idx],
                event.frame_index,
                track_id,
                pose_verified,
            ) {
                continue;
            }
            let prev = row["motion_event_count"].as_u64().unwrap_or(0);
            row["motion_event_count"] = json!(prev + 1);
            let resolved_kind = if pose_verified {
                let predicted = event.predicted_kind();
                if let Some(ref resolved) = context {
                    predicted
                        .and_then(|_| resolved["estimated_kind"].as_str())
                        .filter(|kind| *kind != "low_touch")
                } else {
                    predicted.map(ActionKind::as_str)
                }
            } else {
                None
            };
            if let Some(kind) = resolved_kind {
                let counts = row["estimated_action_counts"]
                    .as_object_mut()
                    .ok_or("Invalid action estimate counts")?;
                let count = counts.get(kind).and_then(Value::as_u64).unwrap_or(0);
                counts.insert(kind.to_owned(), json!(count + 1));
            } else {
                let count = row["unclassified_action_count"].as_u64().unwrap_or(0);
                row["unclassified_action_count"] = json!(count + 1);
            }
            if row["motion_events"]
                .as_array()
                .is_some_and(|a| a.len() < MAX_EVENTS_PER_CARD)
            {
                let mut detail = serde_json::to_value(annotation)
                    .map_err(|e| format!("Serialize motion evidence: {e}"))?;
                detail["frame_index"] = json!(event.frame_index);
                detail["ball_position_px"] = json!([event.ball_x_px, event.ball_y_px]);
                detail["action_prediction"] = if pose_verified {
                    json!(event.action_prediction)
                } else {
                    Value::Null
                };
                detail["assignment_method"] = json!(if pose_verified {
                    "pose"
                } else {
                    "nearest_player"
                });
                if pose_verified {
                    if let Some(ref rally_context) = context {
                        detail["rally_context"] = rally_context.clone();
                    }
                }
                if event.court.is_some() {
                    detail["court"] = json!(event.court);
                }
                row["motion_events"]
                    .as_array_mut()
                    .ok_or("Invalid motion event card")?
                    .push(detail);
            } else {
                row["motion_event_preview_truncated"] = json!(true);
            }
        }
    }
    if count != run.expected_touches || contacts.finish() != run.touch_sha {
        return Err("Touch count or SHA-256 mismatch during rally preview".into());
    }
    Ok(PreviewContent {
        value: json!({
            "schema":"volley-native-rally-preview/1",
                "fps":30,
            "project_id":run.project_id,
            "video":run.video,
            "rally_run_id":run.run_id,
            "analysis_offset_sec":run.analysis_range.map_or(0.0, |r| r.start_sec),
            "analysis_range":run.analysis_range,
            "source_replay_id":run.replay_id,
            "source_touch_id":run.touch_id,
            "total_candidates":intervals.len(),
            "filtered_candidates":if manual.is_some(){intervals.len()}else{filtered_total},
            "gold_filter":filter,
            "sort_by":sort_by,
            "offset":offset,
            "limit":limit,
            "items":cards,
            "source_rallies_sha256":run.rally_sha,
            "source_touches_sha256":run.touch_sha,
            "mode":"read_only_provisional_preview",
            "requires_explicit_manual_selection":true,
            "automatic_montage_eligible":false,
            "authoritative_game_events":false,
            "athlete_identities_verified":false,
            "selection_changed_sources":false
        }),
        clips,
    })
}
pub(crate) fn verify_run_or_candidate(
    state: &AppState,
    project_id: &str,
    rally_run_id: &str,
    candidate_id: Option<&str>,
) -> Result<(), ApiError> {
    let run = inspect_project(state, project_id, rally_run_id)?;
    let selection = match candidate_id {
        Some(id) => Choice::Manual {
            ids: vec![id.to_owned()],
        },
        None => Choice::Page {
            offset: 0,
            limit: 1,
            filter: "all".into(),
            sort_by: "time".into(),
            marked: BTreeSet::new(),
        },
    };
    generate(&run, selection).map_err(storage::bad_request)?;
    Ok(())
}

/// Active PLAY intervals, excluding context padding, from the exact
/// SHA-verified Rally + Touch version. CPU tracking remains full-video evidence
/// because rally discovery depends on it; downstream IDs are rally-scoped.
#[derive(Clone, Debug)]
pub(crate) struct RallyWindows {
    pub(crate) intervals: Vec<(u64, u64)>,
    pub(crate) contexts: BTreeMap<u64, ContactContext>,
}
impl RallyWindows {
    pub(crate) fn contains(&self, frame: u64) -> bool {
        let end = self.intervals.partition_point(|(start, _)| *start <= frame);
        end > 0 && frame <= self.intervals[end - 1].1
    }
}
pub(crate) fn verified_track_windows(run: &ValidatedRun) -> Result<RallyWindows, String> {
    let timeline = generate(run, Choice::Timeline)?.value;
    let rows = timeline["rallies"]
        .as_array()
        .ok_or("Missing verified rally intervals")?;
    let mut intervals = Vec::with_capacity(rows.len());
    for row in rows {
        let start = row["start_frame"].as_u64().ok_or("Invalid rally start")?;
        let end = row["end_frame"].as_u64().ok_or("Invalid rally end")?;
        if start > end
            || end >= run.frames
            || intervals
                .last()
                .is_some_and(|(_, previous_end)| *previous_end >= start)
        {
            return Err("Untrusted overlapping native rally intervals".into());
        }
        intervals.push((start, end));
    }
    let contexts = serde_json::from_value(timeline["contextual_contacts"].clone())
        .map_err(|e| format!("Invalid final contact context: {e}"))?;
    Ok(RallyWindows {
        intervals,
        contexts,
    })
}

/// Full native source validation is delegated to generate(), then a second
/// bounded SHA-verified pass yields every stable, non-overlapping original
/// detected rally interval for matching reviewed attack frames.
pub(crate) fn verified_rally_intervals(
    run: &ValidatedRun,
) -> Result<Vec<(String, u64, u64)>, String> {
    generate(
        run,
        Choice::Page {
            offset: 0,
            limit: 1,
            filter: "all".into(),
            sort_by: "time".into(),
            marked: BTreeSet::new(),
        },
    )?;
    let mut source = BoundedReader::open(&run.rally_file)?;
    let mut found = Vec::new();
    while let Some(line) = source.line()? {
        if found.len() >= MAX_RALLIES {
            return Err("Too many validated rally rows".into());
        }
        let rally: NativeRally = serde_json::from_slice(&line)
            .map_err(|e| format!("Malformed verified rally interval: {e}"))?;
        found.push((
            rally.candidate_id,
            rally.detected_start_frame,
            rally.detected_end_frame,
        ));
    }
    if found.len() as u64 != run.expected_rallies || source.finish() != run.rally_sha {
        return Err("Native rally source changed during highlight review".into());
    }
    Ok(found)
}

/// Verified highlight candidate: candidate id, start/end frames,
/// motion-supported touch proposals and ball coverage. Ranking only.
pub(crate) type QuickRallyCandidate = (String, u64, u64, u64, f64);

/// All selected candidates are checked against the canonical, original-video
/// manual selector (including context, source bounds and full SHA evidence).
/// For independent Quick Reel only: motion evidence is a HIGHLIGHT RANKING,
/// not an authoritative rally, player or game-action classification.
/// The canonical generator authenticates BOTH SHA-bound rally and Touch files.
pub(crate) fn verified_quick_rally_candidates(
    run: &ValidatedRun,
) -> Result<Vec<QuickRallyCandidate>, String> {
    verified_rally_intervals(run)?;
    let mut reader = BoundedReader::open(&run.rally_file)?;
    let mut rows = Vec::new();
    while let Some(line) = reader.line()? {
        if rows.len() >= MAX_RALLIES {
            return Err("Quick Reel rally source too large".into());
        }
        let item: NativeRally = serde_json::from_slice(&line)
            .map_err(|e| format!("Invalid verified quick reel candidate: {e}"))?;
        rows.push((
            item.candidate_id,
            item.detected_start_frame,
            item.detected_end_frame,
            item.motion_supported_touch_proposals,
            item.ball_coverage,
        ));
    }
    if rows.len() as u64 != run.expected_rallies || reader.finish() != run.rally_sha {
        return Err("Quick Reel rally source SHA or count changed".into());
    }
    Ok(rows)
}

pub(crate) fn verified_manual_clips(
    run: &ValidatedRun,
    ids: Vec<String>,
) -> Result<Vec<volley_media::montage::SelectedClip>, String> {
    manual_plan(run, ids)
}
fn manual_plan(
    run: &ValidatedRun,
    ids: Vec<String>,
) -> Result<Vec<volley_media::montage::SelectedClip>, String> {
    let result = generate(run, Choice::Manual { ids })?;
    let mut seconds = 0.0;
    for clip in &result.clips {
        let actual = volley_media::montage::OriginalClip::from_analysis(
            &run.project_source,
            clip.start_frame,
            clip.end_frame,
            0.0,
        )?;
        seconds += actual.duration_sec();
        if seconds > 600.0 {
            return Err("Manually selected episodes exceed the 600-second montage limit".into());
        }
    }
    if seconds < 1.0 {
        return Err("Manual montage must last at least one second".into());
    }
    Ok(result.clips)
}
