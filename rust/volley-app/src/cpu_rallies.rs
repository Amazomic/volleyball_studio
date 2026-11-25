//! Native CPU provisional rally reconstruction over immutable, SHA-256
//! verified player-track and motion-supported Touch replay snapshots.
//! This does NOT certify rallies, serve/attack techniques, or player identity.
mod court_history;
pub(crate) mod rally_metadata;
use crate::{
    cpu_retrack::opaque_id,
    inference_stream, preparation, project_guard, projects,
    storage::{self, safe_file, safe_folder, ApiError, ApiResult, BoundedReader},
    AppState,
};
use axum::{
    extract::{Query, State},
    http::HeaderMap,
    Json,
};
use court_history::CourtHistory;
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
    action_prediction::ActionPrediction,
    fusion::Candidate,
    rally_contact_context::{BallSample as CourtBallSample, ContactEvidence},
    rally_reconstruction::{
        ground_bounce_candidate, net_deflection_candidate, stable_track_count, BallFrame,
        ContactFrame, RallyProposal, RallySegmenter, RallySettings, RallyTotals,
        SurfaceEventProposal,
    },
};

const MAX_MANIFEST: u64 = 16 * 1024;
const MAX_EVIDENCE: u64 = 64 * 1024 * 1024 * 1024;
const OUTPUT_FILE: &str = "rally_candidates.jsonl";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RallyRequest {
    pub(crate) project_id: String,
    pub(crate) video: String,
    pub(crate) replay_id: String,
    pub(crate) touch_id: String,
    #[serde(default)]
    pub(crate) settings: Option<RallySettings>,
}

#[derive(Deserialize)]
pub struct RallyQuery {
    project_id: String,
    rally_run_id: String,
}

#[derive(Deserialize)]
struct SelectedBall {
    source_model: String,
    detection: Candidate,
}
#[derive(Deserialize)]
struct ReplayTrack {
    track_id: u64,
    is_verified_athlete: bool,
    detection: Candidate,
}
#[derive(Deserialize)]
struct ReplayFrame {
    frame_index: u64,
    #[serde(default)]
    tracked_players: Vec<ReplayTrack>,
    #[serde(default)]
    interpolated_track_ids: Vec<u64>,
    selected_ball: Option<SelectedBall>,
    track_ids_provisional: bool,
    confirmed_athlete_identities: bool,
    touches_confirmed: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MotionContact {
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
    body_contact_supported: bool,
    #[serde(default)]
    contact_landmark: Option<volley_core::pose_contact::BodyContact>,
    #[serde(default)]
    contact_pose_sample_frame: Option<u64>,
    #[serde(default)]
    ball_occluded_at_contact: bool,
    #[serde(default)]
    inferred_body_reach_supported: bool,
    athlete_identity_verified: bool,
}
impl MotionContact {
    fn verify(&self, frames: u64, width: u32, height: u32) -> Result<(), String> {
        if self.frame_index >= frames
            || !self.physical_motion_supported
            || self.athlete_identity_verified
            || (self.body_contact_supported && self.provisional_track_id.is_none())
            || self
                .contact_landmark
                .as_ref()
                .is_some_and(|part| !part.valid(width, height))
            || self.contact_pose_sample_frame.is_some_and(|at| {
                !self.body_contact_supported
                    || self.contact_landmark.is_none()
                    || at >= frames
                    || at.abs_diff(self.frame_index) > 3
            })
            || (self.ball_occluded_at_contact
                && !self.body_contact_supported
                && !self.inferred_body_reach_supported
                && self.provisional_track_id.is_some())
            || (self.inferred_body_reach_supported
                && (!self.ball_occluded_at_contact || self.body_contact_supported))
            || self.technique != "unknown_touch"
            || self.action_prediction.as_ref().is_some_and(|prediction| {
                self.provisional_track_id.is_none() || !prediction.valid()
            })
            || self
                .court
                .is_some_and(|ctx| self.provisional_track_id.is_none() || !ctx.valid())
            || self
                .provisional_track_id
                .is_some_and(|id| id == 0 || id > i64::MAX as u64)
            || self.motion_frame_index >= frames
            || self.motion_frame_index.abs_diff(self.frame_index)
                > volley_core::touch_pipeline::MAX_OBSERVED_TOUCH_OFFSET_FRAMES
            || !self.probability.is_finite()
            || !(0.0..=1.0).contains(&self.probability)
            || ![
                self.ball_x_px,
                self.ball_y_px,
                self.direction_change_deg,
                self.speed_before_px_frame,
                self.speed_after_px_frame,
                self.velocity_residual_px_frame,
            ]
            .iter()
            .all(|n| n.is_finite() && *n >= 0.0)
            || self.ball_x_px > width as f64
            || self.ball_y_px > height as f64
            || self.direction_change_deg > 180.0
            || self.velocity_residual_px_frame == 0.0
        {
            return Err("Invalid, out-of-order or nonphysical Touch evidence".into());
        }
        Ok(())
    }
}

fn source_manifest(folder: &Path, kind: &str) -> Result<Value, String> {
    safe_folder(folder)?;
    let bytes = fs::read(safe_file(folder, "manifest.json", MAX_MANIFEST)?)
        .map_err(|e| format!("Read {kind} manifest: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("Malformed {kind} manifest: {e}"))
}
fn sha(value: &Value) -> Result<String, String> {
    value
        .as_str()
        .filter(|s| {
            s.len() == 64
                && s.bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        })
        .map(str::to_owned)
        .ok_or("Missing or invalid SHA-256".into())
}

fn next_contact(reader: &mut BoundedReader) -> Result<Option<MotionContact>, String> {
    reader
        .line()?
        .map(|bytes| {
            serde_json::from_slice(&bytes).map_err(|e| format!("Corrupt Touch JSONL: {e}"))
        })
        .transpose()
}

#[derive(Default)]
struct OutputCounters {
    proposals: u64,
    ingested_touch_candidates: u64,
}
#[expect(clippy::too_many_arguments)]
fn write_proposal(
    writer: &mut BufWriter<File>,
    out_hash: &mut Sha256,
    proposal: RallyProposal,
    history: &CourtHistory,
    surface_events: &std::collections::VecDeque<SurfaceEventProposal>,
    source_ids: (&str, &str),
    counters: &mut OutputCounters,
    track_counts: &[u8],
    metadata: Option<Value>,
) -> Result<(), String> {
    let (replay_id, touch_id) = source_ids;
    let mut value =
        serde_json::to_value(&proposal).map_err(|e| format!("Serialize rally candidate: {e}"))?;
    // Physical surface candidates have no athlete Track ID and must remain
    // separate from the player Touch statistics/review contract.
    value["stable_track_count"] = if proposal.method == "primary_gpu_rally_mask_v1" {
        json!(proposal.provisional_track_ids.len())
    } else {
        json!(stable_track_count(
            track_counts
                .get(proposal.detected_start_frame as usize..=proposal.detected_end_frame as usize)
                .ok_or("Rally track-count interval outside source")?
        ))
    };
    value["surface_event_candidates"] = json!(surface_events
        .iter()
        .filter(
            |event| (proposal.detected_start_frame..=proposal.detected_end_frame)
                .contains(&event.frame_index)
        )
        .collect::<Vec<_>>());
    if let Some(context) = history.for_rally(&proposal) {
        value["estimated_gameplay_contacts"] =
            json!(context.events.len() as u64 - context.ignored_serve_toss_count as u64);
        value["court_contact_context"] = json!(context);
    }
    let key = format!(
        "{replay_id}:{touch_id}:{}:{}",
        proposal.detected_start_frame, proposal.detected_end_frame
    );
    let unique = format!("{:x}", Sha256::digest(key.as_bytes()));
    value["candidate_id"] = json!(&unique[..24]);
    value["reviewed"] = json!(false);
    if let Some(mut metadata) = metadata {
        metadata["rally_id"] = value["candidate_id"].clone();
        value["metadata"] = metadata;
    }
    if proposal.method == "primary_gpu_rally_mask_v1" {
        value["ball_coverage_not_recomputed"] = json!(true);
    }
    let mut data =
        serde_json::to_vec(&value).map_err(|e| format!("Encode rally candidate: {e}"))?;
    data.push(b'\n');
    writer
        .write_all(&data)
        .map_err(|e| format!("Write rally candidate: {e}"))?;
    out_hash.update(&data);
    counters.proposals += 1;
    Ok(())
}

#[cfg(test)]
fn reject_teleport_frames(
    source: &Path,
    sha: &str,
    frames: u64,
    width: u32,
    height: u32,
    min_score: f64,
) -> Result<Vec<bool>, String> {
    Ok(
        crate::cpu_retrack::ball_track::read(source, sha, frames, width, height, min_score, &[0])?
            .rejected
            .clone(),
    )
}
// The rebuild contract deliberately enumerates the full verified source
// identity (files, SHAs, run ids, geometry). Grouping into parameter structs
// belongs to the stage E decomposition, not to a lint-driven rewrite of
// tested CPU pipeline code.
#[cfg(test)]
#[expect(clippy::too_many_arguments)]
fn rebuild_from_snapshots(
    frames_file: &Path,
    touches_file: &Path,
    replay_sha: &str,
    touch_sha: &str,
    frames: u64,
    expected_touches: u64,
    width: u32,
    height: u32,
    settings: RallySettings,
    output: &Path,
    replay_id: &str,
    touch_id: &str,
    court: Option<&volley_core::court_geometry::CourtTransform>,
    ball_scope: Option<(&[u64], f64)>,
) -> Result<(String, RallyTotals, OutputCounters), String> {
    rebuild_from_snapshots_with_scope(
        frames_file,
        touches_file,
        replay_sha,
        touch_sha,
        frames,
        expected_touches,
        width,
        height,
        settings,
        output,
        replay_id,
        touch_id,
        court,
        ball_scope,
        None,
        None,
    )
}

#[expect(clippy::too_many_arguments)]
fn rebuild_from_snapshots_with_scope(
    frames_file: &Path,
    touches_file: &Path,
    replay_sha: &str,
    touch_sha: &str,
    frames: u64,
    expected_touches: u64,
    width: u32,
    height: u32,
    settings: RallySettings,
    output: &Path,
    replay_id: &str,
    touch_id: &str,
    court: Option<&volley_core::court_geometry::CourtTransform>,
    ball_scope: Option<(&[u64], f64)>,
    preliminary_intervals: Option<&[(u64, u64)]>,
    cancel: Option<&AtomicBool>,
) -> Result<(String, RallyTotals, OutputCounters), String> {
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        return Err("cancelled".into());
    }
    settings.validate()?;
    if frames == 0 || frames > 48 * 3600 * 30 || expected_touches > frames {
        return Err("Native rally source frame/contact count exceeds limits".into());
    }
    let (rally_starts, tracking_score) = ball_scope.unwrap_or((&[], settings.min_ball_score));
    let ball_mask = crate::cpu_retrack::ball_track::read_with_cancel(
        frames_file,
        replay_sha,
        frames,
        (width, height),
        tracking_score,
        rally_starts,
        cancel,
    )?;
    let accepted_ball_frames = &ball_mask.accepted;
    let mut inputs = BoundedReader::open(frames_file)?;
    let mut touches = BoundedReader::open(touches_file)?;
    let output_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)
        .map_err(|e| format!("Create isolated rally output: {e}"))?;
    let mut writer = BufWriter::new(output_file);
    let mut digest = Sha256::new();
    let mut totals = OutputCounters::default();
    let mut segmenter = RallySegmenter::new(settings, width, height)?;
    let mut track_counts = Vec::with_capacity(frames as usize);
    let mut continuous =
        preliminary_intervals.map(|intervals| intervals.iter().copied().peekable());
    let mut continuity_ball_frames = 0u64;
    let mut continuity_touch_count = 0u64;
    let mut continuity_first_touch = None::<u64>;
    let mut continuity_last_touch = None::<u64>;
    let mut continuity_max_probability = 0.0f64;
    let mut continuity_track_ids = std::collections::BTreeSet::<u64>::new();
    let mut continuity_unknown = 0u64;
    let mut history = CourtHistory::default();
    let mut next = next_contact(&mut touches)?;
    let mut previous_contact: Option<u64> = None;
    let mut ground_window = std::collections::VecDeque::new();
    let mut surface_events = std::collections::VecDeque::<SurfaceEventProposal>::new();
    let net_surface = court.and_then(|c| c.zones()).and_then(|z| z.net_surface);
    for frame_index in 0..frames {
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        let bytes = inputs.line()?.ok_or("Tracked frame JSONL is truncated")?;
        let frame: ReplayFrame = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Malformed tracked source frame: {e}"))?;
        if frame.frame_index != frame_index
            || !frame.track_ids_provisional
            || frame.confirmed_athlete_identities
            || frame.touches_confirmed
        {
            return Err("Tracked frame ordering or identity status is incompatible".into());
        }
        let in_preliminary_rally = preliminary_intervals.is_none_or(|intervals| {
            let index = intervals.partition_point(|(_, end)| *end < frame_index);
            intervals
                .get(index)
                .is_some_and(|(start, end)| *start <= frame_index && frame_index <= *end)
        });
        let ball = match frame.selected_ball {
            Some(ball) => {
                if ball.source_model != "vballnet"
                    || ball.detection.frame_index != frame_index
                    || ball.detection.class_name != "ball"
                {
                    return Err("Invalid selected-ball proposal from CPU replay".into());
                }
                ball.detection
                    .validate(frame_index - frame_index % 9, "ball")?;
                let [left, top, right, bottom] = ball.detection.bbox_xyxy_norm;
                if ball.detection.score >= settings.min_ball_score
                    && accepted_ball_frames[frame_index as usize]
                    && in_preliminary_rally
                {
                    history.push_ball(CourtBallSample {
                        frame: frame_index,
                        y_px: (top + bottom) * 0.5 * height as f64,
                        radius_px: ((right - left) * width as f64 + (bottom - top) * height as f64)
                            * 0.25,
                    });
                }
                Some(BallFrame {
                    x_px: (left + right) * 0.5 * width as f64,
                    y_px: (top + bottom) * 0.5 * height as f64,
                    score: ball.detection.score,
                })
            }
            None => None,
        };
        if frame.tracked_players.len() > 128 {
            return Err("Too many CPU player observations per rally frame".into());
        }
        let mut verified_frame_ids = std::collections::BTreeSet::<u64>::new();
        // Distinguish immutable raw ball evidence from confidence-qualified
        // ball observations used by touch/court/rally reconstruction.
        let reliable_ball = ball.filter(|b| {
            in_preliminary_rally
                && b.score >= settings.min_ball_score
                && accepted_ball_frames[frame_index as usize]
        });
        let near_player = reliable_ball.is_some_and(|b| {
            frame.tracked_players.iter().any(|tracked| {
                let [x1, y1, x2, y2] = tracked.detection.bbox_xyxy_norm;
                let (x1, y1, x2, y2) = (
                    x1 * width as f64,
                    y1 * height as f64,
                    x2 * width as f64,
                    y2 * height as f64,
                );
                let margin = (y2 - y1) * 0.20;
                b.x_px >= x1 - margin
                    && b.x_px <= x2 + margin
                    && b.y_px >= y1 - margin
                    && b.y_px <= y2 + margin
            })
        });
        for tracked in &frame.tracked_players {
            if tracked.track_id == 0
                || tracked.track_id > i64::MAX as u64
                || tracked.is_verified_athlete
                || tracked.detection.frame_index != frame_index
                || tracked.detection.class_name != "player"
                || !verified_frame_ids.insert(tracked.track_id)
            {
                return Err("Rally source has forged or repeated Track IDs".into());
            }
            tracked
                .detection
                .validate(frame_index - frame_index % 9, "players")?;
        }
        let inferred: std::collections::BTreeSet<_> =
            frame.interpolated_track_ids.iter().copied().collect();
        if inferred.len() != frame.interpolated_track_ids.len()
            || !inferred.is_subset(&verified_frame_ids)
        {
            return Err("Invalid interpolated Track ID metadata".into());
        }
        track_counts.push((verified_frame_ids.len() - inferred.len()) as u8);
        let mut events = Vec::new();
        while next.as_ref().is_some_and(|t| t.frame_index == frame_index) {
            if !in_preliminary_rally {
                return Err("Detailed Touch is outside preliminary gameplay intervals".into());
            }
            let item = next.take().expect("checked");
            item.verify(frames, width, height)?;
            if let Some(track_id) = item.provisional_track_id {
                if !verified_frame_ids.contains(&track_id) {
                    return Err("Touch references a Track ID absent from its source frame".into());
                }
            }
            // A Touch candidate cannot claim an observed ball when the
            // original selected-ball snapshot has no ball in this frame.
            if item.ball_occluded_at_contact {
                // Never claim an observed ball at an occlusion: only a
                // pose-confirmed, dual-sided VBallNet flight fit is permitted.
                if reliable_ball.is_some()
                    || (!item.body_contact_supported && !item.inferred_body_reach_supported)
                {
                    return Err("Invalid occluded-ball touch evidence".into());
                }
            } else if reliable_ball.is_none() {
                return Err(
                    "Touch candidate has no confidence-qualified VBallNet ball in its source frame"
                        .into(),
                );
            }
            if previous_contact.is_some_and(|prev| item.frame_index <= prev) {
                return Err("Motion contacts must be strictly ordered and unique".into());
            }
            previous_contact = Some(item.frame_index);
            totals.ingested_touch_candidates += 1;
            history.push_touch(ContactEvidence {
                frame: item.frame_index,
                track_id: item.provisional_track_id,
                side: item.court.map(|court| court.side),
                in_serve_zone: item.court.is_some_and(|court| court.in_serve_zone),
                kind: item
                    .action_prediction
                    .as_ref()
                    .map(|prediction| prediction.kind),
                ball_y_px: item.ball_y_px,
                speed_before_px_frame: item.speed_before_px_frame,
                speed_after_px_frame: item.speed_after_px_frame,
                velocity_residual_px_frame: item.velocity_residual_px_frame,
            });
            if continuous.is_some() {
                continuity_touch_count += 1;
                continuity_first_touch.get_or_insert(item.frame_index);
                continuity_last_touch = Some(item.frame_index);
                continuity_max_probability = continuity_max_probability.max(item.probability);
                if let Some(id) = item.provisional_track_id {
                    continuity_track_ids.insert(id);
                } else {
                    continuity_unknown += 1;
                }
            }
            events.push(ContactFrame {
                probability: item.probability,
                provisional_track_id: item.provisional_track_id,
            });
            next = next_contact(&mut touches)?;
        }
        if next.as_ref().is_some_and(|t| t.frame_index < frame_index) {
            return Err("Out-of-order native Touch frame index".into());
        }
        if continuous.is_none() {
            if let Some(proposal) = segmenter.push(frame_index, reliable_ball, &events)? {
                write_proposal(
                    &mut writer,
                    &mut digest,
                    proposal,
                    &history,
                    &surface_events,
                    (replay_id, touch_id),
                    &mut totals,
                    &track_counts,
                    None,
                )?;
            }
        } else if in_preliminary_rally && reliable_ball.is_some() {
            continuity_ball_frames += 1;
        }
        // A bounce outside the sidelines is still a candidate for ending a rally.
        // The observed ball centre is projected onto the floor ONLY after
        // the independent nine-frame reversal test succeeds.
        let has_calibrated_court = court.is_some() && reliable_ball.is_some();
        ground_window.push_back((
            frame_index,
            reliable_ball,
            near_player,
            !events.is_empty(),
            has_calibrated_court,
        ));
        if ground_window.len() == 9 {
            let center = &ground_window[4];
            let samples: Option<Vec<BallFrame>> =
                ground_window.iter().map(|entry| entry.1).collect();
            if let Some(samples) = samples {
                let points: [BallFrame; 9] =
                    samples.try_into().map_err(|_| "Invalid ground window")?;
                let near_contact = ground_window.iter().any(|entry| entry.3);
                let near_player = ground_window.iter().skip(3).take(3).any(|entry| entry.2);
                let scale = 1280.0 / width.max(height) as f64;
                let scaled = points.map(|point| BallFrame {
                    x_px: point.x_px * scale,
                    y_px: point.y_px * scale,
                    score: point.score,
                });
                let is_floor =
                    ground_bounce_candidate(&scaled, near_player || near_contact, center.4);
                if is_floor {
                    if let Some(outcome) = court.and_then(|c| {
                        c.tentative_landing([
                            points[4].x_px / width as f64,
                            points[4].y_px / height as f64,
                        ])
                    }) {
                        surface_events.push_back(SurfaceEventProposal {
                            frame_index: center.0,
                            surface: "floor",
                            outcome,
                            evidence: "nine_observed_ball_frames_vertical_reversal",
                            position: None,
                            provisional: true,
                        });
                    }
                    if let Some(proposal) = continuous
                        .is_none()
                        .then(|| segmenter.finish_at_possible_ground_bounce(center.0))
                        .flatten()
                    {
                        write_proposal(
                            &mut writer,
                            &mut digest,
                            proposal,
                            &history,
                            &surface_events,
                            (replay_id, touch_id),
                            &mut totals,
                            &track_counts,
                            None,
                        )?;
                    }
                } else if let Some(surface) = net_surface.as_ref() {
                    if net_deflection_candidate(
                        &points,
                        surface,
                        width as f64,
                        height as f64,
                        near_player || near_contact,
                    ) && !surface_events.back().is_some_and(|event| {
                        event.surface == "net" && center.0.saturating_sub(event.frame_index) < 10
                    }) {
                        surface_events.push_back(SurfaceEventProposal {
                            frame_index: center.0,
                            surface: "net",
                            outcome: "possible_deflection",
                            evidence: "trajectory_deflection_inside_projected_net",
                            position: None,
                            provisional: true,
                        });
                    }
                }
            }
            ground_window.pop_front();
        }
        while surface_events
            .front()
            .is_some_and(|event| frame_index.saturating_sub(event.frame_index) > 1000)
        {
            surface_events.pop_front();
        }
        if let Some(intervals) = continuous.as_mut() {
            if let Some(&(start, end)) = intervals.peek() {
                if frame_index == end {
                    let count = end - start + 1;
                    let proposal = RallyProposal {
                        detected_start_frame: start,
                        detected_end_frame: end,
                        duration_sec: (count as f64 / 30.0 * 10000.0).round() / 10000.0,
                        ball_observations: continuity_ball_frames,
                        ball_coverage: (continuity_ball_frames as f64 / count as f64 * 10000.0)
                            .round()
                            / 10000.0,
                        motion_supported_touch_proposals: continuity_touch_count,
                        first_touch_frame: continuity_first_touch.unwrap_or(start),
                        last_touch_frame: continuity_last_touch.unwrap_or(end),
                        max_touch_probability: continuity_max_probability,
                        provisional_track_ids: continuity_track_ids.iter().copied().collect(),
                        unidentified_contacts: continuity_unknown,
                        classification: "provisional_rally",
                        method: "coherent_vballnet_motion_v2",
                        authoritative_rally: false,
                        athlete_identities_verified: false,
                        serve_verified: false,
                        suggested_context_before_frames: settings.suggested_context_frames,
                        suggested_context_after_frames: settings.suggested_context_frames,
                        end_reason: "continuous_ball_trajectory",
                    };
                    write_proposal(
                        &mut writer,
                        &mut digest,
                        proposal,
                        &history,
                        &surface_events,
                        (replay_id, touch_id),
                        &mut totals,
                        &track_counts,
                        None,
                    )?;
                    intervals.next();
                    continuity_ball_frames = 0;
                    continuity_touch_count = 0;
                    continuity_first_touch = None;
                    continuity_last_touch = None;
                    continuity_max_probability = 0.0;
                    continuity_track_ids.clear();
                    continuity_unknown = 0;
                }
            }
        }
        history.trim(frame_index);
    }
    if next.is_some() || inputs.line()?.is_some() || touches.line()?.is_some() {
        return Err("Native rally inputs contain undeclared frames or contacts".into());
    }
    if totals.ingested_touch_candidates != expected_touches {
        return Err("Touch candidate count differs from versioned manifest".into());
    }
    if inputs.finish() != replay_sha || touches.finish() != touch_sha {
        return Err("Native rally source SHA-256 mismatch".into());
    }
    if continuous.is_none() {
        if let Some(proposal) = segmenter.finish() {
            write_proposal(
                &mut writer,
                &mut digest,
                proposal,
                &history,
                &surface_events,
                (replay_id, touch_id),
                &mut totals,
                &track_counts,
                None,
            )?;
        }
    }
    let mut diagnostics = segmenter.totals().clone();
    if continuous.is_some() {
        diagnostics.accepted_proposals = totals.proposals;
    }
    writer
        .flush()
        .map_err(|e| format!("Flush rally snapshot: {e}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| format!("Sync rally snapshot: {e}"))?;
    Ok((format!("{:x}", digest.finalize()), diagnostics, totals))
}

fn parse_ground_contacts(
    data: &Value,
    manifest: &Value,
    intervals: &[(u64, u64)],
) -> Result<std::collections::VecDeque<SurfaceEventProposal>, String> {
    let mut events = std::collections::VecDeque::new();
    if data["ground_contacts"].is_null()
        && manifest["ground_contact_count"].is_null()
        && manifest["ground_detector_version"].is_null()
    {
        return Ok(events);
    }
    let rows = data["ground_contacts"]
        .as_array()
        .ok_or("Invalid ground contacts")?;
    if rows.len() > 30_000
        || manifest["ground_contact_count"].as_u64() != Some(rows.len() as u64)
        || manifest["ground_detector_version"] != volley_core::motion::GROUND_DETECTOR_VERSION
    {
        return Err("Invalid ground contact count or version".into());
    }
    let mut counts = vec![0usize; intervals.len()];
    for row in rows {
        let frame = row["frame"].as_u64().ok_or("Invalid ground frame")?;
        let position: [f64; 2] = serde_json::from_value(row["position"].clone())
            .map_err(|_| "Invalid ground position")?;
        let outcome = match row["landing"].as_str() {
            Some("near_half") => "near_half",
            Some("far_half") => "far_half",
            Some("line_uncertain") => "line_uncertain",
            Some("out") => "out",
            _ => return Err("Invalid ground landing".into()),
        };
        let i = intervals.partition_point(|&(_, end)| end < frame);
        if !intervals
            .get(i)
            .is_some_and(|&(start, end)| (start..=end).contains(&frame))
            || position
                .iter()
                .any(|v| !v.is_finite() || !(-0.5..=1.5).contains(v))
            || events
                .back()
                .is_some_and(|event: &SurfaceEventProposal| event.frame_index >= frame)
        {
            return Err("Untrusted ground contact".into());
        }
        counts[i] += 1;
        if counts[i] > 256 {
            return Err("Too many ground contacts in a rally".into());
        }
        events.push_back(SurfaceEventProposal {
            frame_index: frame,
            surface: "floor",
            outcome,
            evidence: "trajectory_ground_contact_v3",
            position: Some(position),
            provisional: true,
        });
    }
    Ok(events)
}

#[expect(clippy::too_many_arguments)]
fn publish_primary_from_touches(
    touch_file: &Path,
    touch_sha: &str,
    expected_touches: u64,
    frames: u64,
    width: u32,
    height: u32,
    intervals: &[(u64, u64)],
    settings: RallySettings,
    output: &Path,
    replay_id: &str,
    touch_id: &str,
    rosters: &[Vec<u64>],
    side_source: Option<(&Path, &str, &crate::video_edits::CourtSnapshot)>,
    offset_sec: f64,
    surface_events: &std::collections::VecDeque<SurfaceEventProposal>,
    initial_serves: &std::collections::BTreeMap<u64, (u64, u64)>,
    cancel: Option<&AtomicBool>,
) -> Result<(String, RallyTotals, OutputCounters), String> {
    if rosters.len() != intervals.len()
        || rosters.iter().any(|ids| {
            ids.len() > 128 || ids.windows(2).any(|pair| pair[0] >= pair[1]) || ids.contains(&0)
        })
    {
        return Err("Invalid primary rally rosters".into());
    }
    if intervals
        .iter()
        .any(|&(start, end)| start > end || end >= frames)
        || intervals.windows(2).any(|pair| pair[0].1 >= pair[1].0)
    {
        return Err("Invalid primary GPU rally boundaries".into());
    }
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        return Err("cancelled".into());
    }
    let mut source = BoundedReader::open(touch_file)?;
    let output_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| format!("Create primary rally snapshot: {e}"))?;
    let mut writer = BufWriter::new(output_file);
    let mut digest = Sha256::new();
    let mut counters = OutputCounters::default();
    let history = CourtHistory::default();
    let mut next = next_contact(&mut source)?;
    let mut previous = None;
    for (&(start, end), roster) in intervals.iter().zip(rosters) {
        let mut first = None;
        let mut last = None;
        let mut maximum: f64 = 0.0;
        let mut unidentified = 0;
        let mut observed = Vec::new();
        let mut count = 0;
        while next.as_ref().is_some_and(|item| item.frame_index <= end) {
            if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
                return Err("cancelled".into());
            }
            let item = next.take().expect("checked");
            if item.frame_index < start || previous.is_some_and(|p| item.frame_index <= p) {
                return Err("Touch is outside primary GPU rally intervals or out of order".into());
            }
            item.verify(frames, width, height)?;
            previous = Some(item.frame_index);
            if surface_events
                .binary_search_by_key(&item.frame_index, |event| event.frame_index)
                .is_ok()
            {
                return Err("Ground contact was also published as a player Touch".into());
            }
            first.get_or_insert(item.frame_index);
            last = Some(item.frame_index);
            maximum = maximum.max(item.probability);
            observed.push((
                item.frame_index,
                item.provisional_track_id,
                item.action_prediction
                    .as_ref()
                    .map(|p| p.kind.as_str())
                    .unwrap_or("unknown_touch"),
            ));
            count += 1;
            counters.ingested_touch_candidates += 1;
            if let Some(track_id) = item.provisional_track_id {
                if roster.binary_search(&track_id).is_err() {
                    return Err("Touch Track ID absent from stable rally roster".into());
                }
            } else {
                unidentified += 1;
            }
            next = next_contact(&mut source)?;
        }
        let proposal = RallyProposal {
            detected_start_frame: start,
            detected_end_frame: end,
            duration_sec: ((end - start + 1) as f64 / 30.0 * 10000.0).round() / 10000.0,
            ball_observations: 0,
            ball_coverage: 0.0,
            motion_supported_touch_proposals: count,
            first_touch_frame: first.unwrap_or(start),
            last_touch_frame: last.unwrap_or(end),
            max_touch_probability: maximum,
            provisional_track_ids: roster.clone(),
            unidentified_contacts: unidentified,
            classification: "provisional_rally",
            method: "primary_gpu_rally_mask_v1",
            authoritative_rally: false,
            athlete_identities_verified: false,
            serve_verified: false,
            suggested_context_before_frames: settings.suggested_context_frames,
            suggested_context_after_frames: settings.suggested_context_frames,
            end_reason: "primary_gpu_ball_trajectory",
        };
        let sides = if let Some((path, sha, snapshot)) = side_source {
            if let Some(court) = snapshot.transform.as_ref() {
                crate::native_preview_overlay::team_sides::rally_sides(
                    (path, sha, frames),
                    (start, end),
                    court,
                    &snapshot.revision,
                )?
                .as_ref()
                .clone()
            } else {
                Default::default()
            }
        } else {
            Default::default()
        };
        let first_floor = surface_events
            .iter()
            .find(|event| event.surface == "floor" && (start..=end).contains(&event.frame_index));
        let mut metadata = rally_metadata::build(
            start,
            end,
            &sides,
            roster,
            &observed,
            first_floor.map(|event| {
                (
                    event.frame_index,
                    volley_core::rally_score::scoring_landing(
                        event.outcome,
                        event.position,
                        side_source.and_then(|(_, _, snapshot)| snapshot.transform.as_ref()),
                    ),
                )
            }),
            offset_sec,
        );
        rally_metadata::include_initial_serve(
            &mut metadata,
            roster,
            (start, end),
            first_floor.map(|event| event.frame_index),
            offset_sec,
            initial_serves.get(&start).copied(),
        );
        write_proposal(
            &mut writer,
            &mut digest,
            proposal,
            &history,
            surface_events,
            (replay_id, touch_id),
            &mut counters,
            &[],
            Some(metadata),
        )?;
    }
    if next.is_some()
        || source.line()?.is_some()
        || source.finish() != touch_sha
        || counters.ingested_touch_candidates != expected_touches
    {
        return Err("Primary rally Touch source count, scope or SHA differs".into());
    }
    writer
        .flush()
        .map_err(|e| format!("Flush primary rally snapshot: {e}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| format!("Sync primary rally snapshot: {e}"))?;
    let stats = RallyTotals {
        accepted_proposals: counters.proposals,
        ..RallyTotals::default()
    };
    Ok((format!("{:x}", digest.finalize()), stats, counters))
}

fn record_preliminary_scope(manifest: &mut Value, contacts: &Value, applied: bool) {
    if contacts["preliminary_rally_scope_version"] == "coherent_vballnet_motion_v2" {
        for key in [
            "preliminary_rally_scope_version",
            "preliminary_rally_file",
            "preliminary_rally_sha256",
            "preliminary_rally_interval_count",
        ] {
            manifest[key] = contacts[key].clone();
        }
        manifest["preliminary_rally_scope_applied"] = json!(applied);
    } else if contacts["preliminary_rally_scope_version"] == "primary_gpu_rally_mask_v1" {
        for key in [
            "preliminary_rally_scope_version",
            "preliminary_rally_file",
            "preliminary_rally_sha256",
            "preliminary_rally_interval_count",
            "primary_ball_job_id",
            "primary_rally_mask_sha256",
        ] {
            manifest[key] = contacts[key].clone();
        }
        manifest["preliminary_rally_scope_applied"] = json!(applied);
    }
}

pub(crate) fn process(state: &AppState, request: &RallyRequest) -> Result<Value, ApiError> {
    process_inner(state, request, None, false)
}

pub(crate) fn publish_primary_with_cancel(
    state: &AppState,
    request: &RallyRequest,
    cancel: &AtomicBool,
) -> Result<Value, ApiError> {
    process_inner(state, request, Some(cancel), true)
}

fn process_inner(
    state: &AppState,
    request: &RallyRequest,
    cancel: Option<&AtomicBool>,
    primary_only: bool,
) -> Result<Value, ApiError> {
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        return Err(storage::conflict("cancelled"));
    }
    if !state.writes_enabled {
        return Err(storage::conflict(
            "Rally analysis requires an isolated working directory",
        ));
    }
    storage::project_id(&request.project_id)?;
    storage::video_name(&request.video)?;
    if !opaque_id(&request.replay_id) || !opaque_id(&request.touch_id) {
        return Err(storage::bad_request(
            "Invalid CPU Track ID or Touch snapshot ID",
        ));
    }
    let settings = request
        .settings
        .unwrap_or_default()
        .validate()
        .map_err(storage::bad_request)?;
    let _lock = project_guard::write_guard(state, &request.project_id)?;
    let project = projects::existing_project(state, &request.project_id)?;
    if !project["videos"]
        .as_array()
        .is_some_and(|v| v.iter().any(|name| name.as_str() == Some(&request.video)))
    {
        return Err(storage::bad_request("Video does not belong to project"));
    }
    let plan = preparation::plan_for_run(
        state,
        &request.project_id,
        &request.video,
        "native_cpu_replays",
        &request.replay_id,
    )?;
    let prepared = plan
        .ready_path()
        .ok_or_else(|| storage::bad_request("Prepared video missing from selected project"))?;
    if plan.identity.fps != 30 {
        return Err(storage::bad_request("Native rally source must be 30 FPS"));
    }
    let (size, mtime) = inference_stream::source_stamp(prepared).map_err(storage::bad_request)?;
    let dimensions = volley_media::probe(prepared).map_err(storage::bad_request)?;
    let project_dir = projects::project_root(state).join(&request.project_id);
    safe_folder(&project_dir).map_err(storage::bad_request)?;
    let track_root = project_dir.join("native_cpu_replays");
    let touch_root = project_dir.join("native_cpu_touch_runs");
    safe_folder(&track_root).map_err(storage::bad_request)?;
    safe_folder(&touch_root).map_err(storage::bad_request)?;
    let track_dir = track_root.join(&request.replay_id);
    let touch_dir = touch_root.join(&request.touch_id);
    let tracked = source_manifest(&track_dir, "CPU Track ID").map_err(storage::bad_request)?;
    let contacts = source_manifest(&touch_dir, "CPU Touch").map_err(storage::bad_request)?;
    volley_media::preparation::verify_source_provenance(&plan.source, &tracked["original_source"])
        .map_err(storage::conflict)?;
    if tracked["original_source"] != contacts["original_source"] {
        return Err(storage::conflict(
            "Track and Touch original content differs",
        ));
    }
    let court = crate::video_edits::court_snapshot(state, &request.project_id, &request.video)?;
    let current_court_revision = court.as_ref().map(|snapshot| snapshot.revision.clone());
    if contacts["court_revision"] != json!(current_court_revision) {
        return Err(storage::bad_request(
            "Touch results were computed from an outdated court markup",
        ));
    }
    let identity = json!({"bytes":size,"mtime_ns":mtime});
    if tracked["schema"] != "volley-native-cpu-tracking/1"
        || tracked["state"] != "provisional_tracklets_rebuilt"
        || tracked["project_id"].as_str() != Some(&request.project_id)
        || tracked["video"].as_str() != Some(&request.video)
        || tracked["replay_id"].as_str() != Some(&request.replay_id)
        || tracked["source_identity"] != identity
        || tracked["track_ids_provisional"] != true
        || tracked["athlete_identities_confirmed"] != false
        || tracked["touches_confirmed"] != false
        || tracked["cpu_processing_complete"] != false
        || tracked["fps"] != 30
        || tracked["file"] != "tracked_frames.jsonl"
        || contacts["schema"] != "volley-native-cpu-touch-proposals/1"
        || contacts["state"] != "motion_gated_touch_proposals"
        || contacts["project_id"].as_str() != Some(&request.project_id)
        || contacts["video"].as_str() != Some(&request.video)
        || contacts["touch_id"].as_str() != Some(&request.touch_id)
        || contacts["source_replay_id"].as_str() != Some(&request.replay_id)
        || contacts["source_identity"] != identity
        || contacts["width"].as_u64() != Some(dimensions.width as u64)
        || contacts["height"].as_u64() != Some(dimensions.height as u64)
        || contacts["fps"] != 30
        || contacts["file"] != "touch_candidates.jsonl"
        || contacts["contacts_authoritatively_confirmed"] != false
        || contacts["athlete_identities_verified"] != false
        || contacts["cpu_processing_complete"] != false
    {
        return Err(storage::bad_request(
            "CPU tracking and Touch source manifests are incompatible",
        ));
    }
    let frames = tracked["frames"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= 48 * 3600 * 30 && contacts["frames"].as_u64() == Some(*n))
        .ok_or_else(|| storage::bad_request("Native rally frame counts do not match"))?;
    let expected_touches = contacts["touch_candidates"]
        .as_u64()
        .filter(|n| *n <= frames)
        .ok_or_else(|| storage::bad_request("Unbounded or missing motion Touch count"))?;
    let primary = contacts["preliminary_rally_scope_version"] == "primary_gpu_rally_mask_v1";
    if primary_only && !primary {
        return Err(storage::bad_request(
            "Primary VBallNet rally evidence is required",
        ));
    }
    if !primary && contacts["preliminary_rally_scope_version"] != "coherent_vballnet_motion_v2"
        || contacts["preliminary_rally_settings"] != json!(settings)
        || contacts["preliminary_rally_file"] != "preliminary_rallies.json"
        || contacts["preliminary_rally_scope_applied"] != true
    {
        return Err(storage::bad_request(
            "Recalculate CPU touches with the continuous-ball rally detector",
        ));
    }
    let preliminary_file = safe_file(&touch_dir, "preliminary_rallies.json", 64 * 1024 * 1024)
        .map_err(storage::bad_request)?;
    let bytes = fs::read(preliminary_file)
        .map_err(|e| storage::bad_request(format!("Read continuous rally scope: {e}")))?;
    let expected_sha = sha(&contacts["preliminary_rally_sha256"]).map_err(storage::bad_request)?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected_sha {
        return Err(storage::bad_request(
            "Continuous rally scope checksum changed",
        ));
    }
    let data: Value = serde_json::from_slice(&bytes)
        .map_err(|_| storage::bad_request("Malformed continuous rally scope"))?;
    let replay_sha = sha(&tracked["sha256"]).map_err(storage::bad_request)?;
    let min_ball_score = contacts["settings"]["min_ball_score"]
        .as_f64()
        .ok_or_else(|| storage::bad_request("Missing continuous-ball confidence threshold"))?;
    if data["schema"] != "volley-native-preliminary-continuous-rallies/1"
        || data["source_replay_id"] != request.replay_id
        || data["source_replay_sha256"] != replay_sha
        || data["frames"] != frames
        || data["settings"] != json!(settings)
        || data["min_ball_score"] != min_ball_score
        || data["min_trajectory_seconds"] != crate::cpu_retrack::ball_track::CONTINUOUS_MIN_SECONDS
        || data["merge_gap_seconds"] != crate::cpu_retrack::ball_track::CONTINUOUS_MERGE_SECONDS
    {
        return Err(storage::bad_request(
            "Continuous rally scope provenance changed",
        ));
    }
    let intervals: Vec<(u64, u64)> = serde_json::from_value(data["rally_intervals"].clone())
        .map_err(|_| storage::bad_request("Invalid continuous rally intervals"))?;
    if intervals.len() as u64
        != contacts["preliminary_rally_interval_count"]
            .as_u64()
            .unwrap_or(u64::MAX)
        || intervals
            .iter()
            .any(|&(start, end)| start > end || end >= frames)
        || intervals.windows(2).any(|pair| pair[0].1 >= pair[1].0)
    {
        return Err(storage::bad_request("Invalid saved rally intervals"));
    }
    if primary {
        let ball_id = contacts["primary_ball_job_id"]
            .as_str()
            .filter(|id| opaque_id(id))
            .ok_or_else(|| storage::bad_request("Primary GPU ball job ID missing"))?;
        let primary_sha =
            sha(&contacts["primary_rally_mask_sha256"]).map_err(storage::bad_request)?;
        let fusion_id = tracked["source_fusion_id"]
            .as_str()
            .filter(|id| opaque_id(id))
            .ok_or_else(|| storage::bad_request("Missing source GPU fusion"))?;
        let fusion_dir = project_dir.join("native_gpu_fusions").join(fusion_id);
        let fusion = source_manifest(&fusion_dir, "GPU fusion").map_err(storage::bad_request)?;
        if fusion["sources"]["vballnet"]["job_id"] != ball_id {
            return Err(storage::bad_request(
                "Primary rally source is not the replay ball job",
            ));
        }
        let primary_dir = project_dir.join("native_gpu_runs").join(ball_id);
        let primary_file = safe_file(&primary_dir, "rally_mask.json", 64 * 1024 * 1024)
            .map_err(storage::bad_request)?;
        let primary_bytes = fs::read(primary_file)
            .map_err(|e| storage::bad_request(format!("Read primary GPU rallies: {e}")))?;
        if format!("{:x}", Sha256::digest(&primary_bytes)) != primary_sha {
            return Err(storage::bad_request("Primary GPU rally checksum changed"));
        }
        let evidence: Value = serde_json::from_slice(&primary_bytes)
            .map_err(|_| storage::bad_request("Malformed primary GPU rally evidence"))?;
        let raw: Vec<(u64, u64)> = serde_json::from_value(
            evidence["mask"]
                .get("rally_intervals")
                .unwrap_or(&evidence["mask"]["merged"])
                .clone(),
        )
        .map_err(|_| storage::bad_request("Invalid primary GPU rally bounds"))?;
        let source_frames = evidence["mask"]["source_frame_count"]
            .as_u64()
            .ok_or_else(|| storage::bad_request("Primary GPU frame count missing"))?;
        let expected = crate::cpu_touches::align_primary_intervals(&raw, source_frames, frames)
            .map_err(storage::bad_request)?;
        if data["primary_ball_job_id"] != ball_id
            || data["primary_rally_mask_sha256"] != primary_sha
            || data["primary_source_frames"] != source_frames
            || evidence["schema"] != "volley-primary-rally-mask-evidence/1"
            || evidence["ball_job_id"] != ball_id
            || evidence["original_source"] != tracked["original_source"]
            || evidence["analysis_range"] != tracked["analysis_range"]
            || evidence["mask"]["schema"] != "volley-primary-rally-mask/1"
            || expected != intervals
        {
            return Err(storage::bad_request(
                "Primary GPU rally source does not match saved replay",
            ));
        }
    } else {
        let replay_source = safe_file(&track_dir, "tracked_frames.jsonl", MAX_EVIDENCE)
            .map_err(storage::bad_request)?;
        let computed = crate::cpu_retrack::ball_track::continuous_ball_intervals_with_cancel(
            &replay_source,
            &replay_sha,
            frames,
            min_ball_score,
            crate::cpu_retrack::ball_track::CONTINUOUS_MIN_SECONDS,
            cancel,
        )
        .and_then(|segments| {
            crate::cpu_retrack::ball_track::merge_continuous_intervals(
                &segments,
                crate::cpu_retrack::ball_track::CONTINUOUS_MERGE_SECONDS,
            )
        })
        .map(crate::cpu_retrack::ball_track::filter_short_rallies)
        .map_err(storage::bad_request)?;
        if intervals != computed {
            return Err(storage::bad_request(
                "Continuous rally intervals differ from saved trajectory",
            ));
        }
    }
    let ground_events =
        parse_ground_contacts(&data, &contacts, &intervals).map_err(storage::bad_request)?;
    let initial_serves = if primary_only {
        rally_metadata::parse_initial_serves(&data, &contacts, &intervals)
            .map_err(storage::bad_request)?
    } else {
        Default::default()
    };
    let preliminary_intervals = Some(intervals);
    let replay_sha = sha(&tracked["sha256"]).map_err(storage::bad_request)?;
    let touch_sha = sha(&contacts["sha256"]).map_err(storage::bad_request)?;
    if contacts["source_replay_sha256"].as_str() != Some(replay_sha.as_str()) {
        return Err(storage::bad_request(
            "Touch proposals were built from another CPU track version",
        ));
    }
    let tracked_file = safe_file(&track_dir, "tracked_frames.jsonl", MAX_EVIDENCE)
        .map_err(storage::bad_request)?;
    let touch_file = safe_file(&touch_dir, "touch_candidates.jsonl", MAX_EVIDENCE)
        .map_err(storage::bad_request)?;
    let output_root = project_dir.join("native_cpu_rallies");
    match fs::symlink_metadata(&output_root) {
        Ok(m) if m.file_type().is_symlink() || !m.is_dir() => {
            return Err(storage::bad_request("Unsafe native rally output directory"))
        }
        Ok(_) => (),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => fs::create_dir(&output_root)
            .map_err(|e| storage::internal(format!("Create native rally storage: {e}")))?,
        Err(e) => {
            return Err(storage::internal(format!(
                "Check native rally storage: {e}"
            )))
        }
    }
    let run_id = Uuid::new_v4().simple().to_string();
    let folder = output_root.join(&run_id);
    fs::create_dir(&folder)
        .map_err(|e| storage::internal(format!("Create isolated rally snapshot: {e}")))?;
    let result = (|| -> Result<Value, String> {
        let part = folder.join("rally_candidates.jsonl.partial");
        let ball_scope = crate::cpu_retrack::ball_track::saved_scope(&contacts)?;
        let (rally_starts, tracking_score) = ball_scope
            .as_ref()
            .map(|(starts, score)| (starts.as_slice(), *score))
            .unwrap_or((&[], settings.min_ball_score));
        let (output_sha, stats, output) = if primary_only {
            publish_primary_from_touches(
                &touch_file,
                &touch_sha,
                expected_touches,
                frames,
                dimensions.width,
                dimensions.height,
                preliminary_intervals
                    .as_deref()
                    .ok_or("No primary intervals")?,
                settings,
                &part,
                &request.replay_id,
                &request.touch_id,
                &crate::native_player_tracks::replay_rally_track_ids(
                    &tracked_file,
                    &replay_sha,
                    frames,
                    preliminary_intervals
                        .as_deref()
                        .ok_or("No primary intervals")?,
                )?,
                court
                    .as_ref()
                    .map(|snapshot| (tracked_file.as_path(), replay_sha.as_str(), snapshot)),
                tracked["analysis_range"]["start_sec"]
                    .as_f64()
                    .unwrap_or(0.0),
                &ground_events,
                &initial_serves,
                cancel,
            )?
        } else {
            rebuild_from_snapshots_with_scope(
                &tracked_file,
                &touch_file,
                &replay_sha,
                &touch_sha,
                frames,
                expected_touches,
                dimensions.width,
                dimensions.height,
                settings,
                &part,
                &request.replay_id,
                &request.touch_id,
                court
                    .as_ref()
                    .and_then(|snapshot| snapshot.transform.as_ref()),
                Some((rally_starts, tracking_score)),
                preliminary_intervals.as_deref(),
                cancel,
            )?
        };
        if inference_stream::source_stamp(prepared)? != (size, mtime) {
            return Err("Prepared video changed during native rally processing".into());
        }
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        fs::rename(part, folder.join(OUTPUT_FILE))
            .map_err(|e| format!("Publish rally candidates: {e}"))?;
        let mut manifest = json!({
            "schema":"volley-native-cpu-rally-proposals/1",
            "state":"provisional_rally_segments",
            "project_id":request.project_id,
            "video":request.video,
            "rally_run_id":run_id,
            "source_replay_id":request.replay_id,
            "source_replay_sha256":replay_sha,
            "source_touch_id":request.touch_id,
            "source_touch_sha256":touch_sha,
            "source_identity":identity,
            "original_source":tracked["original_source"],
            "analysis_range":tracked["analysis_range"],
            "width":dimensions.width,"height":dimensions.height,"fps":30,
            "frames":frames,
            "settings":settings,
            "ball_tracking_policy":contacts["ball_tracking_policy"],
            "leading_ball_rally_starts":rally_starts,
            "leading_ball_min_score":contacts["leading_ball_min_score"],
            "court_revision":contacts["court_revision"],
            "court_context_enabled":contacts["court_context_enabled"],
            "court_contact_context_version":if primary_only {
                "not_computed_primary_gpu_v1"
            } else {
                "python_serve_toss_possession_v2"
            },
            "ground_boundary_version":if primary_only {
                contacts["ground_detector_version"].as_str().unwrap_or("not_computed_primary_gpu_v1")
            } else {
                "trajectory_court_surface_v3"
            },
            "materialization_version": if primary_only {
                "primary_gpu_intervals_direct_v2"
            } else {
                "cpu_rally_frame_reconstruction_v1"
            },
            "touch_candidates_ingested":output.ingested_touch_candidates,
            "stable_track_count_policy":if primary_only {"stable_rally_roster_ids_v2"} else {"observed_nonzero_frame_mode_v1"},
            "physical_touch_player_gate":"same_frame_verified_cpu_track",
            "physical_touch_ball_gate":"same_frame_selected_ball",
            "unknown_player_contacts_allowed":true,
            "rally_candidates":output.proposals,
            "diagnostics":stats,
            "file":OUTPUT_FILE,
            "sha256":output_sha,
            "rallies_authoritatively_confirmed":false,
            "athlete_identities_verified":false,
            "action_classification_complete":false,
            "montage_ready":false,
            "cpu_processing_complete":false,
        });
        manifest["rally_score_version"] = json!(volley_core::rally_score::RALLY_SCORE_VERSION);
        record_preliminary_scope(&mut manifest, &contacts, preliminary_intervals.is_some());
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        storage::write_json_atomic(&folder.join("manifest.json"), &manifest)
            .map_err(|e| format!("Save native rally manifest: {e:?}"))?;
        if !primary_only {
            crate::native_player_tracks::prepare_rally_index(
                state,
                &request.project_id,
                &request.video,
                &request.replay_id,
                &run_id,
            )?;
        }
        Ok(manifest)
    })();
    match result {
        Ok(v) => Ok(v),
        Err(e) => {
            let _ = fs::remove_dir_all(&folder);
            Err(storage::bad_request(e))
        }
    }
}

pub async fn run(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<RallyRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let result = tokio::task::spawn_blocking(move || process(&state, &request))
        .await
        .map_err(|e| storage::internal(format!("CPU rally task: {e}")))??;
    Ok(Json(result))
}
pub async fn status(
    State(state): State<Arc<AppState>>,
    Query(query): Query<RallyQuery>,
) -> ApiResult<Value> {
    storage::project_id(&query.project_id)?;
    if !opaque_id(&query.rally_run_id) {
        return Err(storage::bad_request("Invalid native rally run ID"));
    }
    projects::existing_project(&state, &query.project_id)?;
    let project_dir = projects::project_root(&state).join(&query.project_id);
    safe_folder(&project_dir).map_err(storage::bad_request)?;
    let root = project_dir.join("native_cpu_rallies");
    safe_folder(&root).map_err(storage::bad_request)?;
    let dir = root.join(&query.rally_run_id);
    let manifest = source_manifest(&dir, "native rally").map_err(storage::bad_request)?;
    if manifest["schema"] != "volley-native-cpu-rally-proposals/1"
        || manifest["project_id"].as_str() != Some(&query.project_id)
        || manifest["rally_run_id"].as_str() != Some(&query.rally_run_id)
    {
        return Err(storage::bad_request(
            "Rally manifest belongs to another project",
        ));
    }
    Ok(Json(manifest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_gpu_rallies_are_published_without_replaying_video_frames() {
        let root = std::env::temp_dir().join(format!("primary-rallies-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let touches = root.join("touches.jsonl");
        let output = root.join("rallies.jsonl");
        let contact = json!({
            "frame_index":28,
            "provisional_track_id":7,
            "probability":0.85,
            "ball_x_px":310.0,
            "ball_y_px":140.0,
            "motion_frame_index":28,
            "direction_change_deg":42.0,
            "speed_before_px_frame":8.0,
            "speed_after_px_frame":7.0,
            "velocity_residual_px_frame":3.0,
            "technique":"unknown_touch",
            "physical_motion_supported":true,
            "athlete_identity_verified":false
        });
        let mut bytes = serde_json::to_vec(&contact).unwrap();
        bytes.push(b'\n');
        fs::write(&touches, &bytes).unwrap();
        let touch_sha = format!("{:x}", Sha256::digest(&bytes));
        let rosters = [vec![1, 2, 3, 7], vec![10, 11, 12, 13, 14, 15]];
        let surface = parse_ground_contacts(
            &json!({"ground_contacts":[
                {"frame":54,"position":[0.3,0.6],"landing":"line_uncertain"},
                {"frame":72,"position":[0.2,0.7],"landing":"out"}
            ]}),
            &json!({"ground_contact_count":2,"ground_detector_version":volley_core::motion::GROUND_DETECTOR_VERSION}),
            &[(12,33),(50,78)],
        ).unwrap();
        let (sha, stats, output_counts) = publish_primary_from_touches(
            &touches,
            &touch_sha,
            1,
            90,
            640,
            360,
            &[(12, 33), (50, 78)],
            RallySettings::default(),
            &output,
            "replay-primary",
            "touch-primary",
            &rosters,
            None,
            0.0,
            &surface,
            &std::collections::BTreeMap::from([(12, (15, 1)), (50, (52, 10))]),
            None,
        )
        .unwrap();
        let saved = fs::read(&output).unwrap();
        assert_eq!(sha, format!("{:x}", Sha256::digest(&saved)));
        assert_eq!(stats.accepted_proposals, 2);
        assert_eq!(output_counts.ingested_touch_candidates, 1);
        let rows: Vec<Value> = saved
            .split(|byte| *byte == b'\n')
            .filter(|bytes| !bytes.is_empty())
            .map(|bytes| serde_json::from_slice(bytes).unwrap())
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["detected_start_frame"], 12);
        assert_eq!(rows[0]["detected_end_frame"], 33);
        assert_eq!(rows[0]["provisional_track_ids"], json!([1, 2, 3, 7]));
        assert_eq!(
            rows[1]["provisional_track_ids"],
            json!([10, 11, 12, 13, 14, 15])
        );
        assert_eq!(rows[0]["stable_track_count"], 4);
        assert_eq!(rows[1]["stable_track_count"], 6);
        assert_eq!(rows[0]["motion_supported_touch_proposals"], 1);
        assert_eq!(
            rows[0]["metadata"]["touches"][0],
            json!({"track_id":1,"kind":"serve","time_sec":0.5})
        );
        assert_eq!(
            rows[1]["metadata"]["touches"][0],
            json!({"track_id":10,"kind":"serve","time_sec":52.0/30.0})
        );
        assert_eq!(rows[1]["detected_start_frame"], 50);
        assert_eq!(rows[1]["detected_end_frame"], 78);
        assert_eq!(rows[1]["motion_supported_touch_proposals"], 0);
        assert_eq!(rows[1]["method"], "primary_gpu_rally_mask_v1");
        assert_eq!(rows[1]["ball_coverage_not_recomputed"], true);
        assert_eq!(
            rows[1]["surface_event_candidates"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            rows[1]["surface_event_candidates"][0]["outcome"],
            "line_uncertain"
        );
        assert_eq!(rows[1]["surface_event_candidates"][1]["outcome"], "out");
        assert_eq!(
            rows[1]["surface_event_candidates"][0]["position"],
            json!([0.3, 0.6])
        );
        assert!(publish_primary_from_touches(
            &touches,
            &touch_sha,
            1,
            90,
            640,
            360,
            &[(50, 78)],
            RallySettings::default(),
            &root.join("outside.jsonl"),
            "replay-primary",
            "touch-primary",
            &rosters[1..],
            None,
            0.0,
            &std::collections::VecDeque::new(),
            &std::collections::BTreeMap::new(),
            None,
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cancelled_rally_scan_does_not_publish_files() {
        let cancelled = AtomicBool::new(true);
        let missing = std::env::temp_dir().join(format!("no-rally-source-{}", Uuid::new_v4()));
        let output = missing.with_extension("jsonl");
        let result = rebuild_from_snapshots_with_scope(
            &missing,
            &missing,
            "no-sha",
            "no-sha",
            9,
            0,
            640,
            360,
            RallySettings::default(),
            &output,
            "replay",
            "touch",
            None,
            None,
            None,
            Some(&cancelled),
        );
        assert!(matches!(result, Err(ref error) if error == "cancelled"));
        assert!(!output.exists());
    }

    #[test]
    fn preliminary_fallback_keeps_versioned_source_for_cache_reuse() {
        let contacts = json!({
            "preliminary_rally_scope_version":"coherent_vballnet_motion_v2",
            "preliminary_rally_file":"preliminary_rallies.json",
            "preliminary_rally_sha256":"a".repeat(64),

            "preliminary_rally_interval_count":0
        });
        let mut fallback = json!({});
        record_preliminary_scope(&mut fallback, &contacts, false);
        assert_eq!(
            fallback["preliminary_rally_scope_version"],
            "coherent_vballnet_motion_v2"
        );
        assert_eq!(fallback["preliminary_rally_scope_applied"], false);
        assert!(fallback["preliminary_trajectory_impact_count"].is_null());
        assert_eq!(
            fallback["preliminary_rally_sha256"],
            contacts["preliminary_rally_sha256"]
        );
        let mut active = json!({});
        record_preliminary_scope(&mut active, &contacts, true);
        assert_eq!(active["preliminary_rally_scope_applied"], true);
        let mut primary = json!({});
        let source = json!({
            "preliminary_rally_scope_version":"primary_gpu_rally_mask_v1",
            "preliminary_rally_file":"preliminary_rallies.json",
            "preliminary_rally_sha256":"b".repeat(64),
            "preliminary_rally_interval_count":2,
            "primary_ball_job_id":"ball-123",
            "primary_rally_mask_sha256":"a".repeat(64)
        });
        record_preliminary_scope(&mut primary, &source, true);
        assert_eq!(
            primary["preliminary_rally_scope_version"],
            "primary_gpu_rally_mask_v1"
        );
        assert_eq!(primary["primary_ball_job_id"], "ball-123");
        assert_eq!(primary["primary_rally_mask_sha256"], "a".repeat(64));
        assert_eq!(primary["preliminary_rally_scope_applied"], true);
        let mut legacy = json!({});
        record_preliminary_scope(&mut legacy, &json!({}), false);
        assert!(legacy["preliminary_rally_scope_version"].is_null());
    }

    fn ball_threshold_observations(
        tracking_score: f64,
        rally_score: f64,
        weak_initial: bool,
    ) -> u64 {
        let root = std::env::temp_dir().join(format!("ball-threshold-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let source = root.join("tracked.jsonl");
        let touches = root.join("touches.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..50 {
            let x = if (5..20).contains(&frame) {
                0.85 + (frame - 5) as f64 * 0.002
            } else {
                0.3 + frame as f64 * 0.002
            };
            bytes.extend(serde_json::to_vec(&json!({"frame_index":frame,
                "track_ids_provisional":true,"confirmed_athlete_identities":false,"touches_confirmed":false,
                "selected_ball":{"source_model":"vballnet","detection":{
                    "frame_index":frame,"class_name":"ball","score":if weak_initial && frame < 5 || !weak_initial && (5..20).contains(&frame) {0.4} else {0.9},
                    "bbox_xyxy_norm":[x,0.4,x+0.005,0.405]}}})).unwrap());
            bytes.push(b'\n');
        }
        fs::write(&source, &bytes).unwrap();
        fs::write(&touches, []).unwrap();
        let (_, totals, _) = rebuild_from_snapshots(
            &source,
            &touches,
            &format!("{:x}", Sha256::digest(&bytes)),
            &format!("{:x}", Sha256::digest([])),
            50,
            0,
            1280,
            720,
            RallySettings {
                min_ball_score: rally_score,
                ..Default::default()
            },
            &root.join("rallies.jsonl"),
            &"a".repeat(32),
            &"b".repeat(32),
            None,
            Some((&[0], tracking_score)),
        )
        .unwrap();
        fs::remove_dir_all(root).unwrap();
        totals.ball_observations
    }

    #[test]
    fn stricter_rally_confidence_does_not_choose_a_different_leading_ball() {
        assert_eq!(ball_threshold_observations(0.35, 0.67, true), 30);
    }

    #[test]
    fn lower_rally_confidence_cannot_restore_unaccepted_ball_candidates() {
        assert_eq!(ball_threshold_observations(0.67, 0.35, false), 35);
    }

    #[test]
    fn confidence_does_not_override_short_false_ball_island() {
        let dir = std::env::temp_dir().join(format!("ball-repair-{}", Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        let path = dir.join("frames.jsonl");
        let mut bytes = Vec::new();
        for i in 0..30_u64 {
            let x = if (10..=17).contains(&i) {
                0.93
            } else {
                0.34 + i as f64 * 0.001
            };
            let row = json!({
                "frame_index":i,
                "selected_ball":{
                    "source_model":"vballnet",
                    "detection":{
                        "score":0.56,
                        "bbox_xyxy_norm":[x,0.35,x+0.02,0.37]
                    }
                }
            });
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        fs::write(&path, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let masked = reject_teleport_frames(&path, &sha, 30, 1280, 720, 0.35).unwrap();
        assert!(masked[10..=17].iter().all(|v| *v));
        assert!(masked[..10].iter().all(|v| !*v));
        assert!(masked[18..].iter().all(|v| !*v));
        // Do not turn low confidence into a hard 67% threshold.
        let under_67 = reject_teleport_frames(&path, &sha, 30, 1280, 720, 0.67).unwrap();
        assert!(under_67.iter().all(|v| !*v));
        assert!(
            fs::read(&path).unwrap() == bytes,
            "Immutable replay remains unchanged"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn contact_from_real_ball_frame_may_estimate_impact_inside_short_gap() {
        let mut touch = MotionContact {
            frame_index: 476,
            provisional_track_id: Some(6),
            probability: 0.81,
            ball_x_px: 820.0,
            ball_y_px: 407.5,
            motion_frame_index: 482,
            direction_change_deg: 178.0,
            speed_before_px_frame: 17.2,
            speed_after_px_frame: 30.3,
            velocity_residual_px_frame: 47.5,
            technique: "unknown_touch".into(),
            action_prediction: None,
            court: None,
            physical_motion_supported: true,
            body_contact_supported: false,
            contact_landmark: None,
            contact_pose_sample_frame: None,
            ball_occluded_at_contact: false,
            inferred_body_reach_supported: false,
            athlete_identity_verified: false,
        };
        assert!(touch.verify(756, 1280, 720).is_ok());
        // An arbitrarily distant guessed impact must still be rejected.
        touch.motion_frame_index = 486;
        assert!(touch.verify(756, 1280, 720).is_err());
        touch.motion_frame_index = 482;
        touch.provisional_track_id = Some(0);
        assert!(touch.verify(756, 1280, 720).is_err());
        touch.provisional_track_id = None;
        assert!(touch.verify(756, 1280, 720).is_ok());
        touch.ball_occluded_at_contact = true;
        assert!(touch.verify(756, 1280, 720).is_ok());
    }

    #[test]
    #[ignore = "requires copied diagnostic inputs"]
    fn copied_replay_ground_boundary_diagnostic() {
        let dir = Path::new(&std::env::var("VOLLEY_COPIED_DIAGNOSTIC").unwrap()).to_path_buf();
        let tracks = dir.join("tracks.jsonl");
        let touches = dir.join("touches.jsonl");
        let court_points: [[f64; 2]; 4] =
            serde_json::from_slice(&fs::read(dir.join("court.json")).unwrap()).unwrap();
        let court = volley_core::court_geometry::CourtTransform::new(court_points).unwrap();
        let output = dir.join("diagnostic-output.jsonl");
        let track_sha = format!("{:x}", Sha256::digest(fs::read(&tracks).unwrap()));
        let touch_sha = format!("{:x}", Sha256::digest(fs::read(&touches).unwrap()));
        let frames = fs::read_to_string(&tracks).unwrap().lines().count() as u64;
        let count = fs::read_to_string(&touches).unwrap().lines().count() as u64;
        rebuild_from_snapshots(
            &tracks,
            &touches,
            &track_sha,
            &touch_sha,
            frames,
            count,
            1280,
            720,
            RallySettings::default(),
            &output,
            "0123456789abcdef0123456789abcdef",
            "abcdef0123456789abcdef0123456789",
            Some(&court),
            None,
        )
        .unwrap();
        eprintln!("{}", fs::read_to_string(output).unwrap());
    }

    // Read-only integration test against an explicitly supplied real,
    // SHA-bound CPU replay. Runs new CPU Touch + Rally snapshots entirely
    // inside a unique temporary directory; never reuses GPU analysis or
    // changes the original Quick Reel's failed status / artifacts.
    #[test]
    #[ignore = "requires VOLLEY_REAL_TRACK_* source environment"]
    fn real_saved_cpu_touch_to_rallies_without_gpu() {
        let source = std::path::PathBuf::from(
            std::env::var("VOLLEY_REAL_TRACKED_FILE").expect("real CPU replay path required"),
        );
        let sha = std::env::var("VOLLEY_REAL_TRACK_SHA").expect("real CPU replay SHA-256 required");
        let parse = |name: &str| -> u64 {
            std::env::var(name)
                .unwrap_or_else(|_| panic!("Missing {name}"))
                .parse::<u64>()
                .expect("numeric source metadata required")
        };
        let frames = parse("VOLLEY_REAL_TRACK_FRAMES");
        let width = u32::try_from(parse("VOLLEY_REAL_TRACK_WIDTH")).unwrap();
        let height = u32::try_from(parse("VOLLEY_REAL_TRACK_HEIGHT")).unwrap();
        let dir = std::env::temp_dir().join(format!("volley-real-cpu-verify-{}", Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        let touches = dir.join("touch_candidates.jsonl");
        let rallies = dir.join("rally_candidates.jsonl");
        let result = (|| -> Result<(u64, u64), String> {
            let (touch_sha, touch_count, raw_peaks) = crate::cpu_touches::detect_stream(
                &source,
                &touches,
                &sha,
                frames,
                width,
                height,
                volley_core::touch_pipeline::TouchSettings::default(),
            )?;
            let (rally_sha, totals, ingested) = rebuild_from_snapshots(
                &source,
                &touches,
                &sha,
                &touch_sha,
                frames,
                touch_count,
                width,
                height,
                RallySettings::default(),
                &rallies,
                "0123456789abcdef0123456789abcdef",
                "abcdef0123456789abcdef0123456789",
                None,
                None,
            )?;
            let actual = format!(
                "{:x}",
                Sha256::digest(fs::read(&rallies).map_err(|e| e.to_string())?)
            );
            if actual != rally_sha || ingested.ingested_touch_candidates != touch_count {
                return Err("Real CPU output hash or consumed touch count differs".into());
            }
            eprintln!(
                "REAL_CPU_OK frames={frames} touch_candidates={touch_count} raw_peaks={raw_peaks} rally_candidates={}",
                totals.accepted_proposals
            );
            Ok((touch_count, ingested.proposals))
        })();
        fs::remove_dir_all(&dir).unwrap();
        let (touches, rallies) = result.expect("fresh CPU Touch -> Rally must succeed");
        assert!(
            touches > 0,
            "Real source should produce physical Touch proposals"
        );
        assert!(
            rallies > 0,
            "Real source should produce provisional rallies"
        );
    }
    #[test]
    fn source_scoped_cpu_touch_then_rallies_never_publish_foreign_track_references() {
        // Deterministic full CPU stage contract: source with a one-frame
        // player occlusion at a reversal is processed by the same Touch
        // writer used by Quick Reel and then fed into the strict Rally reader.
        let root = std::env::temp_dir().join(format!("volley-touch-rally-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let replay = root.join("tracked_frames.jsonl");
        let touches = root.join("touch_candidates.jsonl");
        let rallies = root.join("rally_candidates.jsonl");
        let mut tracked = Vec::new();
        let frames = 450_u64;
        for frame in 0..frames {
            let t = frame % 200;
            // Strong, physically observable VBallNet reversal even without
            // The player box stays within ball reach.
            let x = (0.60 - (100_i64 - t as i64).unsigned_abs() as f64 * 0.008).max(0.10);
            let touch = (96..=104).contains(&frame) || (296..=304).contains(&frame);
            let players = if frame == 100 {
                vec![]
            } else {
                vec![json!({
                    "track_id":3,"is_verified_athlete":false,
                    "detection":{
                        "frame_index":frame,"class_name":"player","score":0.95,
                        "bbox_xyxy_norm":[0.30,0.15,0.65,0.75],
                        "query_index":0,
                        "touch_probability": if touch {0.99} else {0.01}
                    }
                })]
            };
            let row = json!({
                "frame_index":frame,"tracked_players":players,
                "selected_ball":{"source_model":"vballnet","detection":{
                    "frame_index":frame,"class_name":"ball","score":0.92,
                    "bbox_xyxy_norm":[x,0.36,x+0.02,0.39]
                }},
                "track_ids_provisional":true,"confirmed_athlete_identities":false,
                "touches_confirmed":false
            });
            tracked.extend(serde_json::to_vec(&row).unwrap());
            tracked.push(b'\n');
        }
        fs::write(&replay, &tracked).unwrap();
        let replay_sha = format!("{:x}", Sha256::digest(&tracked));
        let result = (|| -> Result<(u64, u64), String> {
            let (touch_sha, contacts, _) = crate::cpu_touches::detect_stream(
                &replay,
                &touches,
                &replay_sha,
                frames,
                640,
                360,
                volley_core::touch_pipeline::TouchSettings::default(),
            )?;
            let (_, totals, ingested) = rebuild_from_snapshots(
                &replay,
                &touches,
                &replay_sha,
                &touch_sha,
                frames,
                contacts,
                640,
                360,
                RallySettings::default(),
                &rallies,
                "0123456789abcdef0123456789abcdef",
                "abcdef0123456789abcdef0123456789",
                None,
                None,
            )?;
            if ingested.ingested_touch_candidates != contacts {
                return Err("CPU rally reconstruction dropped a valid Touch".into());
            }
            Ok((contacts, totals.accepted_proposals))
        })();
        fs::remove_dir_all(&root).unwrap();
        let (contacts, rallies) = result.expect("CPU Touch and Rally stages must agree");
        assert!(
            contacts > 0,
            "Synthetic ball reversal must yield physical Touch"
        );
        // A zero-proposal rally is valid for a synthetic segment that fails
        // other quality gates; the old bug was a fatal foreign-ID rejection.
        assert!(rallies <= contacts);
    }

    fn fixture(
        frames: u64,
        cut: Option<(u64, u64)>,
        touches: &[u64],
    ) -> (Vec<u8>, Vec<u8>, String, String) {
        let mut tracked = Vec::new();
        for frame in 0..frames {
            let ball = if cut.is_some_and(|(begin, end)| frame >= begin && frame <= end) {
                Value::Null
            } else {
                let t = frame % 200;
                let x = 0.10 + (t.min(200 - t) as f64 * 0.005);
                json!({
                    "source_model":"vballnet",
                    "detection":{
                        "frame_index":frame,"class_name":"ball","score":0.92,
                        "bbox_xyxy_norm":[x,0.36,x+0.02,0.39]
                    }
                })
            };
            let observed_id = touches
                .iter()
                .position(|&touch_frame| touch_frame == frame)
                .map(|index| if index % 2 == 0 { 3 } else { 4 })
                .unwrap_or(3);
            let row = json!({
                "frame_index":frame,
                "tracked_players":[{"track_id":observed_id,"is_verified_athlete":false,
                    "detection":{"frame_index":frame,"class_name":"player","score":0.95,
                    "bbox_xyxy_norm":[0.2,0.2,0.4,0.8],"query_index":0}}],
                "selected_ball":ball,
                "track_ids_provisional":true,
                "confirmed_athlete_identities":false,
                "touches_confirmed":false,
            });
            tracked.extend(serde_json::to_vec(&row).unwrap());
            tracked.push(b'\n');
        }
        let mut contacts = Vec::new();
        for (index, &frame) in touches.iter().enumerate() {
            // The fixture alternates players so legal multi-Touch rallies remain valid.
            let track_id = if index % 2 == 0 { 3 } else { 4 };
            let event = json!({
                "frame_index":frame,
                "provisional_track_id":track_id,
                "probability":0.94,
                "ball_x_px":150.0,"ball_y_px":120.0,
                "motion_frame_index":frame,
                "direction_change_deg":90.0,
                "speed_before_px_frame":4.0,
                "speed_after_px_frame":5.0,
                "velocity_residual_px_frame":10.0,
                "technique":"unknown_touch",
                "physical_motion_supported":true,
                "athlete_identity_verified":false,
            });
            contacts.extend(serde_json::to_vec(&event).unwrap());
            contacts.push(b'\n');
        }
        let track_sha = format!("{:x}", Sha256::digest(&tracked));
        let touch_sha = format!("{:x}", Sha256::digest(&contacts));
        (tracked, contacts, track_sha, touch_sha)
    }
    fn verify(
        tracks: &[u8],
        touches: &[u8],
        track_sha: &str,
        touch_sha: &str,
        frames: u64,
        contact_count: u64,
        settings: RallySettings,
    ) -> Result<(Value, Vec<Value>), String> {
        let root = std::env::temp_dir().join(format!("native-rally-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let frames_file = root.join("tracked.jsonl");
        let touches_file = root.join("contacts.jsonl");
        let output = root.join("out.jsonl");
        fs::write(&frames_file, tracks).unwrap();
        fs::write(&touches_file, touches).unwrap();
        let result = rebuild_from_snapshots(
            &frames_file,
            &touches_file,
            track_sha,
            touch_sha,
            frames,
            contact_count,
            640,
            360,
            settings,
            &output,
            "0123456789abcdef0123456789abcdef",
            "abcdef0123456789abcdef0123456789",
            None,
            None,
        );
        let response = result.map(|(sha, stats, counts)| {
            let bytes = fs::read(&output).unwrap();
            assert_eq!(sha, format!("{:x}", Sha256::digest(&bytes)));
            let rows = bytes
                .split(|b| *b == b'\n')
                .filter(|row| !row.is_empty())
                .map(|row| serde_json::from_slice::<Value>(row).unwrap())
                .collect::<Vec<_>>();
            (json!({"proposals":counts.proposals,"stats":stats}), rows)
        });
        fs::remove_dir_all(root).unwrap();
        response
    }
    #[test]
    fn continuous_ball_rally_is_kept_without_any_touch_candidates() {
        let (tracks, touches, track_sha, touch_sha) = fixture(180, None, &[]);
        let root = std::env::temp_dir().join(format!("continuous-rally-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let frames_file = root.join("tracked.jsonl");
        let touches_file = root.join("contacts.jsonl");
        let output = root.join("out.jsonl");
        fs::write(&frames_file, &tracks).unwrap();
        fs::write(&touches_file, &touches).unwrap();
        let settings = RallySettings::default();
        let scope = [(20, 109), (150, 179)];
        let result = rebuild_from_snapshots_with_scope(
            &frames_file,
            &touches_file,
            &track_sha,
            &touch_sha,
            180,
            0,
            640,
            360,
            settings,
            &output,
            "0123456789abcdef0123456789abcdef",
            "abcdef0123456789abcdef0123456789",
            None,
            None,
            Some(&scope),
            None,
        )
        .unwrap();
        assert_eq!(result.2.proposals, 2);
        assert_eq!(result.2.ingested_touch_candidates, 0);
        let rows = fs::read_to_string(&output).unwrap();
        let proposals: Vec<Value> = rows
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(proposals.len(), 2);
        for (proposal, expected) in proposals.iter().zip(scope) {
            assert_eq!(proposal["detected_start_frame"], expected.0);
            assert_eq!(proposal["detected_end_frame"], expected.1);
            assert_eq!(proposal["stable_track_count"], 1);
            assert_eq!(proposal["motion_supported_touch_proposals"], 0);
            assert_eq!(proposal["method"], "coherent_vballnet_motion_v2");
            assert_eq!(proposal["authoritative_rally"], false);
        }
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn two_ball_segments_and_two_pairs_of_motion_touches_make_two_proposals() {
        let (frames, touches, fh, th) = fixture(450, Some((160, 289)), &[30, 80, 310, 370]);
        let (counts, rows) = verify(
            &frames,
            &touches,
            &fh,
            &th,
            450,
            4,
            RallySettings::default(),
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(counts["proposals"], 2);
        assert!(rows[0]["detected_end_frame"].as_u64().unwrap() < 160);
        assert_eq!(rows[1]["detected_start_frame"], 290);
        assert_eq!(rows[0]["motion_supported_touch_proposals"], 2);
        assert_eq!(rows[0]["classification"], "provisional_rally");
        assert_eq!(rows[0]["authoritative_rally"], false);
        assert_eq!(rows[0]["provisional_track_ids"][0], 3);
        assert_ne!(rows[0]["candidate_id"], rows[1]["candidate_id"]);
    }
    #[test]
    fn reconstructed_rally_ends_at_possible_ground_bounce_with_court() {
        let (raw, touches, _, touch_sha) = fixture(150, None, &[20, 70]);
        let mut changed = Vec::new();
        for line in raw
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
        {
            let mut row: Value = serde_json::from_slice(line).unwrap();
            let frame = row["frame_index"].as_u64().unwrap();
            if (100..=120).contains(&frame) {
                let y = 0.70 - (frame as f64 - 110.0).abs() * 0.02;
                row["selected_ball"]["detection"]["bbox_xyxy_norm"][1] = json!(y);
                row["selected_ball"]["detection"]["bbox_xyxy_norm"][3] = json!(y + 0.01);
            }
            changed.extend(serde_json::to_vec(&row).unwrap());
            changed.push(b'\n');
        }
        let root = std::env::temp_dir().join(format!("rally-ground-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let tracks = root.join("tracks.jsonl");
        let contacts = root.join("contacts.jsonl");
        let output = root.join("output.jsonl");
        fs::write(&tracks, &changed).unwrap();
        fs::write(&contacts, &touches).unwrap();
        let court = volley_core::court_geometry::CourtTransform::new([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        .unwrap();
        rebuild_from_snapshots(
            &tracks,
            &contacts,
            &format!("{:x}", Sha256::digest(&changed)),
            &touch_sha,
            150,
            2,
            640,
            360,
            RallySettings::default(),
            &output,
            "0123456789abcdef0123456789abcdef",
            "abcdef0123456789abcdef0123456789",
            Some(&court),
            None,
        )
        .unwrap();
        let data = fs::read_to_string(&output).unwrap();
        let event: Value = serde_json::from_str(data.lines().next().unwrap()).unwrap();
        assert_eq!(event["end_reason"], "possible_ground_bounce");
        assert!((108..=111).contains(&event["detected_end_frame"].as_u64().unwrap()));
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn rescoring_requires_no_gpu_and_single_contact_is_not_a_rally() {
        let (frames, touches, fh, th) = fixture(140, None, &[35]);
        let (_summary, rows) = verify(
            &frames,
            &touches,
            &fh,
            &th,
            140,
            1,
            RallySettings::default(),
        )
        .unwrap();
        assert!(rows.is_empty());
        let relaxed = RallySettings {
            min_touch_candidates: 1,
            ..RallySettings::default()
        };
        let (_, rows) = verify(&frames, &touches, &fh, &th, 140, 1, relaxed).unwrap();
        assert_eq!(rows.len(), 1);
    }
    /// Even with recalculated SHA-256, a Touch belongs only to a visibly
    /// tracked player and observed selected ball in the source CPU frame.
    #[test]
    fn contact_snapshot_cannot_credit_absent_track_id_or_missing_ball() {
        let (frames, touches, _, touch_sha) = fixture(140, None, &[35, 90]);
        let rewrite = |field: &str| -> (Vec<u8>, String) {
            let mut altered = Vec::new();
            for raw in frames.split(|b| *b == b'\n').filter(|row| !row.is_empty()) {
                let mut event: Value = serde_json::from_slice(raw).unwrap();
                if event["frame_index"] == 35 {
                    match field {
                        "track" => event["tracked_players"][0]["track_id"] = json!(999_u64),
                        "ball" => event["selected_ball"] = Value::Null,
                        "verified" => {
                            event["tracked_players"][0]["is_verified_athlete"] = json!(true)
                        }
                        _ => unreachable!(),
                    }
                }
                altered.extend(serde_json::to_vec(&event).unwrap());
                altered.push(b'\n');
            }
            let digest = format!("{:x}", Sha256::digest(&altered));
            (altered, digest)
        };
        for field in ["track", "ball", "verified"] {
            let (altered, hash) = rewrite(field);
            let error = verify(
                &altered,
                &touches,
                &hash,
                &touch_sha,
                140,
                2,
                RallySettings::default(),
            )
            .unwrap_err();
            assert!(
                error.contains("Touch") || error.contains("Track"),
                "Unexpected reason for unsafe source {field}: {error}"
            );
        }
    }
    #[test]
    fn refuses_hash_corruption_out_of_order_touch_and_undeclared_rows() {
        let (frames, touches, fh, th) = fixture(140, None, &[35, 90]);
        assert!(verify(
            &frames,
            &touches,
            &fh,
            &"0".repeat(64),
            140,
            2,
            RallySettings::default()
        )
        .is_err());
        assert!(verify(
            &frames,
            &touches,
            &fh,
            &th,
            140,
            3,
            RallySettings::default()
        )
        .is_err());
        let mut extra = touches.clone();
        extra.extend_from_slice(&touches);
        let newsha = format!("{:x}", Sha256::digest(&extra));
        assert!(verify(
            &frames,
            &extra,
            &fh,
            &newsha,
            140,
            4,
            RallySettings::default()
        )
        .is_err());
        let mut bad = frames.clone();
        bad[0] = b' ';
        assert!(verify(&bad, &touches, &fh, &th, 140, 2, RallySettings::default()).is_err());
        let mut truncated = frames.clone();
        truncated.truncate(frames.len() / 2);
        assert!(verify(
            &truncated,
            &touches,
            &fh,
            &th,
            140,
            2,
            RallySettings::default()
        )
        .is_err());
    }
}
