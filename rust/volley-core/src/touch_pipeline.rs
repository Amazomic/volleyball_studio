//! VBallNet motion-first Touch proposals associated with observed player Tracks.
//! Detection remains an unverified hypothesis, not a confirmed game action.
mod ball_continuity;
mod geometry;
mod player_gap;
mod pose_motion;
mod trajectory_tcn;
pub use ball_continuity::{
    exclude_ball_teleports, flag_ball_teleports, flag_local_ball_artifacts,
    repair_accepted_vballnet_trajectory, repair_vballnet_trajectory,
    repair_vballnet_trajectory_with_players, LeadingBallObservation, LeadingBallSettings,
    LeadingBallTracker,
};
pub use player_gap::reconstruct as reconstruct_track_gaps;
pub use pose_motion::{
    evaluate_pose_block, observed_pose_motion, pose_contact_at_frame,
    pose_contact_from_nearby_sample, PoseFrame, PosePlayer,
};
pub use trajectory_tcn::{evaluate_tcn_peaks, trajectory_features};

/// Maximum time between a real observed VBallNet anchor and a predicted
/// impact inside the bounded (at most ten-frame) missing-ball island.
/// Occluded impacts retain observed flight anchors and a same-frame tracked player.
pub const MAX_OBSERVED_TOUCH_OFFSET_FRAMES: u64 = 9;
pub const TRAJECTORY_DETECTOR_VERSION: &str = "piecewise_ballistic_motion_v6";
pub const INTERPOLATED_BALL_SOURCE: &str = "ball_stabilizer_interpolated";

use crate::{
    action_prediction::{self, ActionPrediction, CourtContext, PositionEvidence},
    motion::BallObservation,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TouchPlayer {
    pub track_id: u64,
    pub score: f64,
    pub touch_probability: Option<f64>,
    /// Geometry normalized to video width and height.
    pub bbox_xyxy_norm: [f64; 4],
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TouchBall {
    pub source_model: String,
    pub score: f64,
    pub bbox_xyxy_norm: [f64; 4],
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TouchFrame {
    pub frame_index: u64,
    pub players: Vec<TouchPlayer>,
    pub selected_ball: Option<TouchBall>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TouchSettings {
    pub algorithm: TouchAlgorithm,
    pub threshold: f64,
    /// Minimum raw VBallNet probability accepted as a physical ball observation.
    /// Low-confidence raw evidence stays immutable for future CPU-only replays.
    pub min_ball_score: f64,
    pub smooth: usize,
    pub min_gap: usize,
    pub max_reach_ratio: f64,
    pub min_interval_sec: f64,
    pub require_pose: bool,
    pub pose_min_confidence: f64,
    pub pose_max_distance: f64,
    /// Minimum change in direction when inferring impact across missing ball
    /// observations. Ordinary ballistic acceleration is not a body contact.
    pub occluded_min_turn_deg: f64,
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TouchAlgorithm {
    #[default]
    Legacy,
    PoseMotion,
    TrajectoryTcn,
    BallTrajectory,
}
impl Default for TouchSettings {
    fn default() -> Self {
        Self {
            algorithm: TouchAlgorithm::Legacy,
            threshold: 0.48,
            min_ball_score: 0.35,
            smooth: 5,
            min_gap: 4,
            max_reach_ratio: 0.75,
            min_interval_sec: 0.10,
            require_pose: false,
            // Keep moderately occluded far-court wrists available to the
            // geometry intersection; motion evidence is still mandatory.
            pose_min_confidence: 0.28,
            pose_max_distance: 0.34,
            occluded_min_turn_deg: 5.0,
        }
    }
}
impl TouchSettings {
    pub fn validate(self) -> Result<Self, String> {
        if !self.threshold.is_finite() || !(0.0..=1.0).contains(&self.threshold) {
            return Err("Touch threshold must be in 0..=1".into());
        }
        if !(1..=31).contains(&self.smooth) || !(1..=150).contains(&self.min_gap) {
            return Err("Touch smooth/min_gap exceeds supported limits".into());
        }
        if !self.min_ball_score.is_finite()
            || !(0.01..=1.0).contains(&self.min_ball_score)
            || !self.max_reach_ratio.is_finite()
            || !(0.05..=1.0).contains(&self.max_reach_ratio)
            || !self.min_interval_sec.is_finite()
            || !(0.10..=3.0).contains(&self.min_interval_sec)
            || !self.pose_min_confidence.is_finite()
            || !(0.0..=1.0).contains(&self.pose_min_confidence)
            || !self.pose_max_distance.is_finite()
            || !(0.01..=1.0).contains(&self.pose_max_distance)
            || !self.occluded_min_turn_deg.is_finite()
            || !(5.0..=180.0).contains(&self.occluded_min_turn_deg)
        {
            return Err("Invalid ball reach or contact separation".into());
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PhysicalTouchCandidate {
    pub frame_index: u64,
    pub provisional_track_id: Option<u64>,
    pub probability: f64,
    pub ball_x_px: f64,
    pub ball_y_px: f64,
    pub motion_frame_index: u64,
    pub direction_change_deg: f64,
    pub speed_before_px_frame: f64,
    pub speed_after_px_frame: f64,
    pub velocity_residual_px_frame: f64,
    pub technique: &'static str,
    /// Unverified, source-backed geometry hypothesis; never a reviewed game action.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_prediction: Option<ActionPrediction>,
    /// Player-foot projection on the user-calibrated court, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub court: Option<crate::court_geometry::CourtContactEvidence>,
    pub physical_motion_supported: bool,
    /// A same-Track-ID COCO17 joint or body segment independently corroborates the hit.
    /// False means a motion-supported, tracked but unconfirmed proposal.
    pub body_contact_supported: bool,
    /// Closest time-validated COCO17 joint or limb, in original video pixels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contact_landmark: Option<crate::pose_contact::BodyContact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contact_pose_sample_frame: Option<u64>,
    /// Impact frame had no observed VBallNet ball: position inferred from
    /// independently observed flights on both sides of an occlusion.
    pub ball_occluded_at_contact: bool,
    /// The observed flights converge within a stable tracked player's
    /// extended reach, but no COCO17 body-joint intersection was verified.
    pub inferred_body_reach_supported: bool,
    pub athlete_identity_verified: bool,
}

impl PhysicalTouchCandidate {
    pub fn is_motion_echo_of(&self, impact: &Self) -> bool {
        let window = crate::motion::TOUCH_MOTION_CONFIG.velocity_window as u64;
        self.physical_motion_supported
            && impact.physical_motion_supported
            && !self.body_contact_supported
            && self.direction_change_deg < 25.0
            && impact.direction_change_deg >= 75.0
            && self.velocity_residual_px_frame < impact.velocity_residual_px_frame
            && self.frame_index.abs_diff(impact.frame_index) <= window
            && self.motion_frame_index.abs_diff(impact.motion_frame_index) <= window
    }
}
fn valid_bbox(b: [f64; 4]) -> bool {
    b.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) && b[2] > b[0] && b[3] > b[1]
}
fn pixel_bbox(b: [f64; 4], width: f64, height: f64) -> [f64; 4] {
    [b[0] * width, b[1] * height, b[2] * width, b[3] * height]
}
fn center(b: [f64; 4]) -> [f64; 2] {
    [(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5]
}
/// Python PlayerStore's centered 61-observation, 80th-percentile height.
fn standing_height(
    frames: &[TouchFrame],
    frame_index: usize,
    track_id: u64,
    video_height: f64,
) -> Option<f64> {
    let heights = |frame: &TouchFrame| {
        frame
            .players
            .iter()
            .find(|player| player.track_id == track_id)
            .map(|player| (player.bbox_xyxy_norm[3] - player.bbox_xyxy_norm[1]) * video_height)
    };
    let mut observed: Vec<f64> = frames[..=frame_index]
        .iter()
        .rev()
        .filter_map(heights)
        .take(31)
        .collect();
    observed.extend(
        frames[frame_index + 1..]
            .iter()
            .filter_map(heights)
            .take(30),
    );
    if observed.is_empty() {
        return None;
    }
    observed.sort_by(f64::total_cmp);
    let quantile = (observed.len() - 1) as f64 * 0.8;
    let lower = quantile.floor() as usize;
    let upper = quantile.ceil() as usize;
    let estimate = observed[lower] + (observed[upper] - observed[lower]) * quantile.fract();
    Some(estimate)
}

/// Evaluate at most 1200 frames including lookbehind and lookahead. Caller
/// emits only [emit_start,emit_end); 32+ frame overlap keeps local smoothing,
/// physical motion and time-based dedup decisions away from block boundaries.
/// Peaks are relative to first frame so this never allocates a full-game curve.
pub fn detect_ball_trajectory_touches(
    frames: &[TouchFrame],
    width: u32,
    height: u32,
    emit_start: u64,
    emit_end: u64,
    settings: TouchSettings,
    court_scope: Option<(&crate::court_geometry::CourtTransform, u64)>,
) -> Result<Vec<PhysicalTouchCandidate>, String> {
    settings.validate()?;
    if !(64..=16384).contains(&width)
        || !(64..=16384).contains(&height)
        || frames.is_empty()
        || frames.len() > 1200
    {
        return Err("Invalid trajectory Touch block dimensions".into());
    }
    let first = frames[0].frame_index;
    let limit = first
        .checked_add(frames.len() as u64)
        .ok_or("Touch frame overflow")?;
    if emit_start < first || emit_start > emit_end || emit_end > limit {
        return Err("Invalid trajectory Touch emission window".into());
    }
    let mut points = Vec::with_capacity(frames.len());
    for (index, row) in frames.iter().enumerate() {
        if row.frame_index != first + index as u64 {
            return Err("Noncontiguous trajectory Touch frames".into());
        }
        if let Some(ball) = &row.selected_ball {
            if (ball.source_model != "vballnet" && ball.source_model != INTERPOLATED_BALL_SOURCE)
                || !valid_bbox(ball.bbox_xyxy_norm)
                || !ball.score.is_finite()
                || !(0.0..=1.0).contains(&ball.score)
            {
                return Err("Invalid trajectory Touch ball evidence".into());
            }
            let [x, y] = center(ball.bbox_xyxy_norm);
            points.push(BallObservation {
                frame: row.frame_index as i64,
                x: x * width as f64,
                y: y * height as f64,
            });
        }
    }
    let mut events = Vec::new();
    for impact in crate::motion::trajectory_direction_breaks(
        &points,
        crate::motion::DirectionBreakConfig::default(),
    ) {
        let Ok(frame) = u64::try_from(impact.frame) else {
            continue;
        };
        if frame < emit_start || frame >= emit_end {
            continue;
        }
        let Some(row) = frames.get((frame - first) as usize) else {
            continue;
        };
        if row
            .selected_ball
            .as_ref()
            .is_some_and(|ball| ball.source_model != "vballnet")
        {
            continue;
        }
        if court_scope.is_some_and(|(court, rally_end_frame)| {
            crate::motion::classify_ground_contact(
                &points,
                &impact,
                court,
                width,
                height,
                rally_end_frame,
            )
            .is_some()
        }) {
            continue;
        }
        let strength = (impact.motion.angle / 170.0).clamp(0.0, 1.0);
        let fit_confidence = 1.0 / (1.0 + impact.fit_error / 6.0);
        let probability = (0.50 + 0.45 * strength * fit_confidence).min(0.95);
        if probability < settings.threshold {
            continue;
        }
        events.push(PhysicalTouchCandidate {
            frame_index: frame,
            provisional_track_id: None,
            probability,
            ball_x_px: impact.position[0],
            ball_y_px: impact.position[1],
            motion_frame_index: impact.motion.motion_frame as u64,
            direction_change_deg: impact.motion.angle,
            speed_before_px_frame: impact.motion.speed_before,
            speed_after_px_frame: impact.motion.speed_after,
            velocity_residual_px_frame: impact.motion.residual,
            technique: "unknown_touch",
            action_prediction: None,
            court: None,
            physical_motion_supported: true,
            body_contact_supported: false,
            contact_landmark: None,
            contact_pose_sample_frame: None,
            ball_occluded_at_contact: !impact.contact_observed,
            inferred_body_reach_supported: false,
            athlete_identity_verified: false,
        });
    }
    Ok(events)
}

pub fn attach_touch_source_evidence(
    events: &mut [PhysicalTouchCandidate],
    frames: &[TouchFrame],
    width: u32,
    height: u32,
    reach_ratio: f64,
) {
    let Some(first) = frames.first().map(|frame| frame.frame_index) else {
        return;
    };
    let (w, h) = (width as f64, height as f64);
    for event in events {
        if event.ball_occluded_at_contact {
            continue;
        }
        let Some(row) = event
            .frame_index
            .checked_sub(first)
            .and_then(|offset| frames.get(offset as usize))
        else {
            continue;
        };
        let selected = row
            .players
            .iter()
            .filter_map(|player| {
                let bbox = pixel_bbox(player.bbox_xyxy_norm, w, h);
                let distance = (bbox[0] - event.ball_x_px)
                    .max(event.ball_x_px - bbox[2])
                    .max(0.0)
                    .hypot(
                        (bbox[1] - event.ball_y_px)
                            .max(event.ball_y_px - bbox[3])
                            .max(0.0),
                    );
                let reach = (bbox[3] - bbox[1]) * reach_ratio;
                (reach > 0.0 && distance <= reach).then_some((player, bbox, distance / reach))
            })
            .min_by(|a, b| a.2.total_cmp(&b.2));
        let Some((player, bbox, _)) = selected else {
            continue;
        };
        event.provisional_track_id = Some(player.track_id);
        let radius = row
            .selected_ball
            .as_ref()
            .map(|ball| {
                ((ball.bbox_xyxy_norm[2] - ball.bbox_xyxy_norm[0]) * w
                    + (ball.bbox_xyxy_norm[3] - ball.bbox_xyxy_norm[1]) * h)
                    * 0.25
            })
            .unwrap_or(0.0);
        event.action_prediction = standing_height(
            frames,
            (event.frame_index - first) as usize,
            player.track_id,
            h,
        )
        .and_then(|standing_height_px| {
            action_prediction::predict(PositionEvidence {
                ball_y_px: event.ball_y_px,
                ball_radius_px: radius,
                player_bbox_px: bbox,
                standing_height_px,
                touch_score: event.probability,
                track_score: player.score,
                context: CourtContext::Unavailable,
            })
        });
    }
}

pub fn evaluate_rally_blocks(
    frames: &[TouchFrame],
    width: u32,
    height: u32,
    emit_start: u64,
    emit_end: u64,
    settings: TouchSettings,
    rally_starts: &[u64],
) -> Result<Vec<PhysicalTouchCandidate>, String> {
    settings.validate()?;
    if rally_starts.windows(2).any(|p| p[0] >= p[1]) {
        return Err("Rally boundaries must be ordered".into());
    }
    if frames.is_empty() {
        return evaluate_block(frames, width, height, emit_start, emit_end, settings);
    }
    if frames.len() > 1200
        || emit_start < frames[0].frame_index
        || emit_start > emit_end
        || emit_end > frames.last().unwrap().frame_index.saturating_add(1)
        || frames
            .iter()
            .enumerate()
            .any(|(i, f)| f.frame_index != frames[0].frame_index + i as u64)
    {
        return Err("Invalid rally touch emission window".into());
    }
    let ranges = rally_frame_ranges(frames, rally_starts)?;
    let mut events = Vec::new();
    for range in ranges {
        let segment = &frames[range];
        let start = emit_start.max(segment[0].frame_index);
        let end = emit_end.min(segment.last().unwrap().frame_index + 1);
        if start < end {
            events.extend(evaluate_block(
                segment, width, height, start, end, settings,
            )?);
        }
    }
    Ok(events)
}

pub fn rally_frame_ranges(
    frames: &[TouchFrame],
    rally_starts: &[u64],
) -> Result<Vec<std::ops::Range<usize>>, String> {
    if rally_starts.windows(2).any(|p| p[0] >= p[1]) {
        return Err("Rally boundaries must be ordered".into());
    }
    let mut boundaries = vec![0];
    boundaries.extend(
        rally_starts
            .iter()
            .map(|start| frames.partition_point(|f| f.frame_index < *start))
            .filter(|&i| i > 0 && i < frames.len()),
    );
    boundaries.push(frames.len());
    boundaries.dedup();
    Ok(boundaries
        .windows(2)
        .filter(|p| p[0] < p[1])
        .map(|p| p[0]..p[1])
        .collect())
}

pub fn evaluate_block(
    frames: &[TouchFrame],
    width: u32,
    height: u32,
    emit_start: u64,
    emit_end: u64,
    settings: TouchSettings,
) -> Result<Vec<PhysicalTouchCandidate>, String> {
    settings.validate()?;
    if !(64..=16384).contains(&width) || !(64..=16384).contains(&height) {
        return Err("Invalid prepared-video width or height".into());
    }
    if frames.is_empty() || frames.len() > 1200 {
        return Err("Touch analysis block must contain 1..=1200 frames".into());
    }
    let first = frames[0].frame_index;
    let end = first
        .checked_add(frames.len() as u64)
        .ok_or("Touch frame index overflow")?;
    if emit_start < first || emit_start > emit_end || emit_end > end {
        return Err("Invalid touch emission frame range".into());
    }
    let (w, h) = (width as f64, height as f64);
    let mut balls = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        if frame.frame_index != first + index as u64 || frame.players.len() > 128 {
            return Err("Non-contiguous touch frames or too many player boxes".into());
        }
        let relative = index as u32;
        for player in &frame.players {
            if player.track_id == 0
                || player.track_id > i64::MAX as u64
                || !valid_bbox(player.bbox_xyxy_norm)
                || !player.score.is_finite()
                || !(0.0..=1.0).contains(&player.score)
            {
                return Err("Invalid tracked player evidence".into());
            }
        }
        if let Some(ball) = &frame.selected_ball {
            if (ball.source_model != "vballnet" && ball.source_model != INTERPOLATED_BALL_SOURCE)
                || !valid_bbox(ball.bbox_xyxy_norm)
                || !ball.score.is_finite()
                || !(0.0..=1.0).contains(&ball.score)
            {
                return Err("Invalid selected-ball evidence".into());
            }
            let point = center(ball.bbox_xyxy_norm);
            balls.push(BallObservation {
                frame: relative as i64,
                x: point[0] * w,
                y: point[1] * h,
            });
        }
    }
    if balls.len() < 5 {
        return Ok(Vec::new());
    }
    let mut geometry = geometry::detect(frames, &balls, w, h, settings.max_reach_ratio);
    geometry.extend(geometry::detect_observed_vertical_rebounds(frames, w, h));
    geometry.extend(geometry::detect_gap_rebounds(
        frames,
        w,
        h,
        settings.max_reach_ratio,
    ));
    let mut result = Vec::with_capacity(geometry.len());
    for candidate in geometry {
        let frame = first + candidate.frame as u64;
        if frame < emit_start || frame >= emit_end {
            continue;
        }
        let source = &frames[candidate.frame];
        let player = source
            .players
            .iter()
            .find(|p| p.track_id == candidate.track_id);
        let observed_ball = source
            .selected_ball
            .as_ref()
            .filter(|ball| ball.source_model == "vballnet");
        let ball = observed_ball.or_else(|| {
            candidate.ball_occluded_at_contact.then(|| {
                frames[..candidate.frame]
                    .iter()
                    .rev()
                    .take(10)
                    .filter_map(|row| row.selected_ball.as_ref())
                    .find(|ball| ball.source_model == "vballnet")
            })?
        });
        let (Some(player), Some(ball)) = (player, ball) else {
            continue;
        };
        let bbox = pixel_bbox(player.bbox_xyxy_norm, w, h);
        // Without body-joint corroboration, an off-body ball that only makes
        // a weak residual bounce must not be attributed to a nearby player.
        // Real hand contact is validated independently by the RTMPose path.
        let body_width = bbox[2] - bbox[0];
        let outside_x = (bbox[0] - candidate.ball_x)
            .max(candidate.ball_x - bbox[2])
            .max(0.0);
        if candidate.motion.speed_after < 10.0 && outside_x > 0.30 * body_width {
            continue;
        }
        let radius = ((ball.bbox_xyxy_norm[2] - ball.bbox_xyxy_norm[0]) * w
            + (ball.bbox_xyxy_norm[3] - ball.bbox_xyxy_norm[1]) * h)
            * 0.25;
        let probability = candidate.probability.min(0.88);
        if probability < settings.threshold {
            continue;
        }
        let action_prediction = standing_height(frames, candidate.frame, candidate.track_id, h)
            .and_then(|height_px| {
                action_prediction::predict(PositionEvidence {
                    ball_y_px: candidate.ball_y,
                    ball_radius_px: radius,
                    player_bbox_px: bbox,
                    standing_height_px: height_px,
                    touch_score: probability,
                    track_score: player.score,
                    context: CourtContext::Unavailable,
                })
            });
        result.push(PhysicalTouchCandidate {
            frame_index: frame,
            provisional_track_id: Some(candidate.track_id),
            probability,
            ball_x_px: candidate.ball_x,
            ball_y_px: candidate.ball_y,
            motion_frame_index: first + candidate.motion.motion_frame as u64,
            direction_change_deg: candidate.motion.angle,
            speed_before_px_frame: candidate.motion.speed_before,
            speed_after_px_frame: candidate.motion.speed_after,
            velocity_residual_px_frame: candidate.motion.residual,
            technique: "unknown_touch",
            action_prediction,
            court: None,
            physical_motion_supported: true,
            body_contact_supported: false,
            contact_landmark: None,
            contact_pose_sample_frame: None,
            ball_occluded_at_contact: candidate.ball_occluded_at_contact,
            inferred_body_reach_supported: candidate.ball_occluded_at_contact,
            athlete_identity_verified: false,
        });
    }
    // Several neighboring frames may describe one physical velocity break.
    // Retain the strongest physics-supported candidate within the impact window.
    result.sort_by(|a, b| {
        b.probability
            .total_cmp(&a.probability)
            .then_with(|| a.frame_index.cmp(&b.frame_index))
    });
    let mut unique: Vec<PhysicalTouchCandidate> = Vec::new();
    let min_gap =
        ((settings.min_interval_sec * 30.0).round().max(1.0) as u64).max(settings.min_gap as u64);
    for candidate in result {
        if unique
            .iter()
            .all(|old| old.frame_index.abs_diff(candidate.frame_index) > min_gap)
        {
            unique.push(candidate);
        }
    }
    unique.sort_by_key(|event| event.frame_index);
    Ok(unique)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_touch_moments_do_not_depend_on_player_identity_or_touch_kind() {
        let frames = (620..=668_u64)
            .map(|frame| {
                let ball = if (648..=653).contains(&frame) {
                    None
                } else {
                    let (x, y) = if frame <= 631 {
                        let t = frame as f64 - 631.0;
                        (980.0 - 3.0 * t, 840.0 + 34.0 * t + 0.25 * t * t)
                    } else if frame <= 642 {
                        let t = frame as f64 - 631.0;
                        (980.0 - 3.0 * t, 840.0 - 35.0 * t + t * t)
                    } else if frame <= 647 {
                        let t = frame as f64 - 643.0;
                        (947.0 + t * 0.7, 575.0 + t * 0.2)
                    } else {
                        let t = frame as f64 - 654.0;
                        (955.0 - t * 2.0, 485.0 - 32.0 * t + t * t)
                    };
                    Some(TouchBall {
                        source_model: "vballnet".into(),
                        score: 0.95,
                        bbox_xyxy_norm: [
                            (x - 4.0) / 1920.0,
                            (y - 4.0) / 1080.0,
                            (x + 4.0) / 1920.0,
                            (y + 4.0) / 1080.0,
                        ],
                    })
                };
                TouchFrame {
                    frame_index: frame,
                    players: Vec::new(),
                    selected_ball: ball,
                }
            })
            .collect::<Vec<_>>();
        let baseline = detect_ball_trajectory_touches(
            &frames,
            1920,
            1080,
            620,
            669,
            TouchSettings::default(),
            None,
        )
        .unwrap();
        assert!(baseline
            .iter()
            .any(|event| event.frame_index.abs_diff(631) <= 2));
        assert!(baseline
            .iter()
            .any(|event| event.frame_index.abs_diff(651) <= 1));
        assert!(baseline
            .iter()
            .all(|event| event.provisional_track_id.is_none()));
        assert!(baseline
            .iter()
            .all(|event| event.action_prediction.is_none()));
        let mut attributed_frames = frames.clone();
        for frame in &mut attributed_frames {
            frame.players.push(TouchPlayer {
                track_id: 17,
                score: 0.99,
                touch_probability: None,
                bbox_xyxy_norm: [0.44, 0.25, 0.56, 0.87],
            });
        }
        let mut classified = detect_ball_trajectory_touches(
            &attributed_frames,
            1920,
            1080,
            620,
            669,
            TouchSettings::default(),
            None,
        )
        .unwrap();
        let baseline_moments = baseline
            .iter()
            .map(|event| event.frame_index)
            .collect::<Vec<_>>();
        assert_eq!(
            classified
                .iter()
                .map(|event| event.frame_index)
                .collect::<Vec<_>>(),
            baseline_moments,
        );
        attach_touch_source_evidence(&mut classified, &attributed_frames, 1920, 1080, 0.75);
        assert_eq!(
            classified
                .iter()
                .map(|event| event.frame_index)
                .collect::<Vec<_>>(),
            baseline_moments,
        );
        assert!(classified
            .iter()
            .all(|event| !event.athlete_identity_verified));
        assert!(classified
            .iter()
            .any(|event| event.provisional_track_id == Some(17)));
        assert!(classified
            .iter()
            .any(|event| event.provisional_track_id.is_none()));
    }

    fn recorded_motion_event(frame: u64, angle: f64, residual: f64) -> PhysicalTouchCandidate {
        PhysicalTouchCandidate {
            frame_index: frame,
            provisional_track_id: Some(if frame == 235 { 6 } else { 4 }),
            probability: 0.7,
            ball_x_px: 500.0,
            ball_y_px: 300.0,
            motion_frame_index: frame,
            direction_change_deg: angle,
            speed_before_px_frame: 7.4,
            speed_after_px_frame: 16.9,
            velocity_residual_px_frame: residual,
            technique: "unknown_touch",
            action_prediction: None,
            court: None,
            physical_motion_supported: true,
            body_contact_supported: false,
            contact_landmark: None,
            contact_pose_sample_frame: None,
            ball_occluded_at_contact: false,
            inferred_body_reach_supported: false,
            athlete_identity_verified: false,
        }
    }

    #[test]
    fn interpolated_ball_positions_support_motion_but_never_anchor_a_touch() {
        let frames = (0..24)
            .map(|i| TouchFrame {
                frame_index: i,
                players: vec![TouchPlayer {
                    track_id: 4,
                    score: 0.95,
                    touch_probability: None,
                    bbox_xyxy_norm: [0.4, 0.2, 0.7, 0.9],
                }],
                selected_ball: Some(TouchBall {
                    source_model: "ball_stabilizer_interpolated".into(),
                    score: 0.55,
                    bbox_xyxy_norm: [0.5, 0.4, 0.51, 0.41],
                }),
            })
            .collect::<Vec<_>>();
        assert!(
            evaluate_block(&frames, 1280, 720, 0, 24, TouchSettings::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn departure_acceleration_is_echo_of_nearby_rebound_despite_different_track_ids() {
        let impact = recorded_motion_event(235, 164.05, 32.88);
        let departure = recorded_motion_event(239, 6.71, 14.36);
        assert!(departure.is_motion_echo_of(&impact));
    }

    #[test]
    fn approach_acceleration_is_echo_of_later_rebound() {
        let impact = recorded_motion_event(243, 164.05, 32.88);
        let approach = recorded_motion_event(239, 6.71, 14.36);
        assert!(approach.is_motion_echo_of(&impact));
    }

    #[test]
    fn independently_body_supported_contact_is_not_motion_echo() {
        let impact = recorded_motion_event(235, 164.05, 32.88);
        let mut second = recorded_motion_event(239, 6.71, 14.36);
        second.body_contact_supported = true;
        assert!(!second.is_motion_echo_of(&impact));
    }

    #[test]
    fn separate_direction_break_is_not_motion_echo() {
        let impact = recorded_motion_event(235, 164.05, 32.88);
        let second = recorded_motion_event(239, 80.0, 14.36);
        assert!(!second.is_motion_echo_of(&impact));
    }

    #[test]
    fn small_turn_outside_rebound_fitting_window_is_not_motion_echo() {
        let impact = recorded_motion_event(235, 164.05, 32.88);
        let second = recorded_motion_event(241, 6.71, 14.36);
        assert!(!second.is_motion_echo_of(&impact));
    }

    #[test]
    fn distant_motion_anchor_is_not_suppressed_by_nearby_contact_frame() {
        let impact = recorded_motion_event(235, 164.05, 32.88);
        let mut second = recorded_motion_event(239, 6.71, 14.36);
        second.motion_frame_index = 241;
        assert!(!second.is_motion_echo_of(&impact));
    }

    #[test]
    fn pose_motion_finds_change_with_moving_wrist_and_rejects_missing_motion() {
        let mut frames = scenario(true, true, false);
        for (i, frame) in frames.iter_mut().enumerate() {
            let x = if i <= 32 {
                0.20 + i as f64 * 0.008
            } else {
                0.456 - (i - 32) as f64 * 0.008
            };
            frame.selected_ball.as_mut().unwrap().bbox_xyxy_norm[0] = x;
            frame.selected_ball.as_mut().unwrap().bbox_xyxy_norm[2] = x + 0.02;
        }
        let mut poses = Vec::new();
        for (frame_index, x) in [(28, 0.42), (32, 0.46), (36, 0.50)] {
            let mut keypoints = [[0.0; 3]; 17];
            keypoints[9] = [x, 0.365, 0.95];
            poses.push(PoseFrame {
                frame_index,
                players: vec![PosePlayer {
                    track_id: 1,
                    keypoints,
                    interpolated: false,
                }],
            });
        }
        let contacts = evaluate_pose_block(
            &frames,
            &poses,
            640,
            360,
            0,
            80,
            TouchSettings {
                require_pose: true,
                ..TouchSettings::default()
            },
        )
        .unwrap();
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].provisional_track_id, Some(1));
        assert!(contacts[0].body_contact_supported);
        assert!(contacts[0].contact_landmark.is_some());
        assert!(
            contacts[0].action_prediction.is_none(),
            "Wrist-only evidence confirms contact, not the technique"
        );
        let mut unrelated = poses.clone();
        unrelated.insert(
            1,
            PoseFrame {
                frame_index: 30,
                players: vec![PosePlayer {
                    track_id: 2,
                    keypoints: poses[0].players[0].keypoints,
                    interpolated: false,
                }],
            },
        );
        assert_eq!(
            evaluate_pose_block(
                &frames,
                &unrelated,
                640,
                360,
                0,
                80,
                TouchSettings::default()
            )
            .unwrap()
            .len(),
            1
        );
        let mut actual_far = poses.clone();
        actual_far[1].players[0].keypoints[9][0] = 0.99;
        assert!(evaluate_pose_block(
            &frames,
            &actual_far,
            640,
            360,
            32,
            33,
            TouchSettings::default()
        )
        .unwrap()
        .is_empty());
        let mut wrong_source = frames.clone();
        for frame in &mut wrong_source {
            frame.selected_ball.as_mut().unwrap().source_model = "unsupported".into();
        }
        assert!(evaluate_pose_block(
            &wrong_source,
            &poses,
            640,
            360,
            0,
            80,
            TouchSettings::default()
        )
        .unwrap()
        .is_empty());
        assert!(evaluate_pose_block(
            &frames,
            &poses[..1],
            640,
            360,
            0,
            80,
            TouchSettings::default()
        )
        .unwrap()
        .is_empty());
        let mut other_player = poses.clone();
        for pose in &mut other_player {
            pose.players[0].track_id = 2;
        }
        assert!(evaluate_pose_block(
            &frames,
            &other_player,
            640,
            360,
            0,
            80,
            TouchSettings::default()
        )
        .unwrap()
        .is_empty());
        let mut low_confidence = poses.clone();
        for pose in &mut low_confidence {
            pose.players[0].keypoints[9][2] = 0.1;
        }
        assert!(evaluate_pose_block(
            &frames,
            &low_confidence,
            640,
            360,
            0,
            80,
            TouchSettings::default()
        )
        .unwrap()
        .is_empty());
        assert!(evaluate_pose_block(
            &scenario(false, true, false),
            &poses,
            640,
            360,
            0,
            80,
            TouchSettings::default()
        )
        .unwrap()
        .is_empty());
        let first = evaluate_pose_block(&frames, &poses, 640, 360, 0, 32, TouchSettings::default())
            .unwrap();
        let second =
            evaluate_pose_block(&frames, &poses, 640, 360, 32, 80, TouchSettings::default())
                .unwrap();
        assert_eq!(first.len() + second.len(), 1);
        for pose in &mut poses {
            pose.players[0].keypoints[9][0] = 0.46;
        }
        assert_eq!(
            evaluate_pose_block(&frames, &poses, 640, 360, 0, 80, TouchSettings::default())
                .unwrap()
                .len(),
            1,
            "A still hand can block a ball if it approaches, intersects, and rebounds"
        );
    }

    fn scenario(curve: bool, close: bool, switching: bool) -> Vec<TouchFrame> {
        (0..80u64)
            .map(|i| {
                let x = if curve && i >= 32 {
                    0.20 + 32.0 * 0.005 - (i - 32) as f64 * 0.005
                } else {
                    0.20 + i as f64 * 0.005
                };
                let player = TouchPlayer {
                    track_id: 1,
                    score: 0.94,
                    touch_probability: Some(if (28..=36).contains(&i) { 0.95 } else { 0.01 }),
                    bbox_xyxy_norm: if close {
                        [0.20, 0.15, 0.45, 0.65]
                    } else {
                        [0.70, 0.15, 0.95, 0.65]
                    },
                };
                TouchFrame {
                    frame_index: i,
                    players: vec![player],
                    selected_ball: Some(TouchBall {
                        source_model: if switching && i >= 32 {
                            "unsupported"
                        } else {
                            "vballnet"
                        }
                        .into(),
                        score: 0.95,
                        bbox_xyxy_norm: [x, 0.35, x + 0.02, 0.38],
                    }),
                }
            })
            .collect()
    }
    fn strong_rebound_frames() -> Vec<TouchFrame> {
        let mut frames = scenario(true, true, false);
        for (i, row) in frames.iter_mut().enumerate() {
            let x = if i <= 32 {
                0.20 + i as f64 * 0.008
            } else {
                0.20 + 32.0 * 0.008 - (i - 32) as f64 * 0.008
            };
            let ball = row.selected_ball.as_mut().unwrap();
            ball.bbox_xyxy_norm[0] = x;
            ball.bbox_xyxy_norm[2] = x + 0.02;
        }
        frames
    }

    #[test]
    fn unsupported_logit_without_ball_impact_never_creates_a_touch() {
        let mut frames = scenario(false, true, false);
        for frame in &mut frames {
            frame.players[0].touch_probability = Some(1.0);
        }
        let events = evaluate_block(
            &frames,
            640,
            360,
            0,
            80,
            TouchSettings {
                ..TouchSettings::default()
            },
        )
        .unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn default_hybrid_touch_profile_matches_short_rally_calibration() {
        let settings = TouchSettings::default();
        assert_eq!(settings.algorithm, TouchAlgorithm::Legacy);
        assert_eq!(settings.threshold, 0.48);
        assert_eq!(settings.smooth, 5);
        assert_eq!(settings.min_gap, 4);
        assert_eq!(settings.max_reach_ratio, 0.75);
        assert_eq!(settings.min_interval_sec, 0.10);
        assert!(!settings.require_pose);
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn old_settings_preserve_existing_behavior() {
        let settings: TouchSettings = serde_json::from_str(r#"{"threshold":0.5,"smooth":5,"min_gap":10,"max_reach_ratio":0.35,"min_interval_sec":0.35}"#).unwrap();
        assert!(!settings.require_pose);
        assert_eq!(settings.algorithm, TouchAlgorithm::Legacy);
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn disabling_unsupported_touch_matches_absent_touch_scores() {
        let frames = scenario(true, true, false);
        let settings = TouchSettings {
            ..TouchSettings::default()
        };
        let without_model = evaluate_block(&frames, 640, 360, 0, 80, settings).unwrap();
        let mut no_scores = frames;
        for frame in &mut no_scores {
            frame.players[0].touch_probability = None;
        }
        let expected =
            evaluate_block(&no_scores, 640, 360, 0, 80, TouchSettings::default()).unwrap();
        assert_eq!(
            serde_json::to_value(without_model).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }

    fn stopped_ball_launch_frames(has_player: bool, fast_exit: bool) -> Vec<TouchFrame> {
        (0..50u64)
            .map(|i| {
                let selected_ball = if (33..=38).contains(&i) {
                    None
                } else {
                    let t = i.saturating_sub(39) as f64;
                    let x = 420.0 - if i >= 39 { t * 1.5 } else { 0.0 };
                    let y = 360.0
                        - if i >= 39 {
                            if fast_exit {
                                60.0 + 20.0 * t
                            } else {
                                4.0 + t
                            }
                        } else {
                            0.0
                        };
                    Some(TouchBall {
                        source_model: "vballnet".into(),
                        score: 0.85,
                        bbox_xyxy_norm: [
                            (x - 5.0) / 1280.0,
                            (y - 5.0) / 720.0,
                            (x + 5.0) / 1280.0,
                            (y + 5.0) / 720.0,
                        ],
                    })
                };
                TouchFrame {
                    frame_index: i,
                    players: if has_player {
                        vec![TouchPlayer {
                            track_id: 11,
                            score: 0.90,
                            touch_probability: None,
                            bbox_xyxy_norm: [0.30, 0.30, 0.36, 0.65],
                        }]
                    } else {
                        Vec::new()
                    },
                    selected_ball,
                }
            })
            .collect()
    }

    #[test]
    fn occluded_stationary_ball_launch_is_timed_inside_real_observation_gap() {
        let frames = stopped_ball_launch_frames(true, true);
        let events = evaluate_block(&frames, 1280, 720, 0, 50, TouchSettings::default()).unwrap();
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(events[0].frame_index, 36, "{events:?}");
        assert!(frames[events[0].frame_index as usize]
            .selected_ball
            .is_none());
        assert_eq!(events[0].provisional_track_id, Some(11));
        assert!(events[0].ball_occluded_at_contact);
        assert!(events[0].inferred_body_reach_supported);
        assert!(!events[0].body_contact_supported);
        assert!(events[0].speed_after_px_frame > 15.0);
    }

    #[test]
    fn occluded_stationary_ball_launch_requires_player_and_strong_outgoing_flight() {
        for (has_player, fast_exit) in [(false, true), (true, false)] {
            let frames = stopped_ball_launch_frames(has_player, fast_exit);
            let events =
                evaluate_block(&frames, 1280, 720, 0, 50, TouchSettings::default()).unwrap();
            assert!(events.is_empty(), "{events:?}");
        }
    }

    #[test]
    fn stabilized_speed_and_direction_change_with_sparse_raw_anchors_is_detected() {
        let mut frames = strong_rebound_frames();
        for (index, frame) in frames.iter_mut().enumerate() {
            let x = if index <= 32 {
                0.20 + index as f64 * 0.008
            } else {
                0.20 + 32.0 * 0.008 - (index - 32) as f64 * 0.0095
            };
            let ball = frame.selected_ball.as_mut().unwrap();
            ball.bbox_xyxy_norm[0] = x;
            ball.bbox_xyxy_norm[2] = x + 0.02;
            if index != 32 && index % 3 != 0 {
                ball.source_model = INTERPOLATED_BALL_SOURCE.into();
            }
        }
        let events = evaluate_block(&frames, 640, 360, 0, 80, TouchSettings::default()).unwrap();
        assert_eq!(events.len(), 1, "{events:?}");
        assert!(events[0].frame_index.abs_diff(32) <= 1, "{events:?}");
        assert!(events[0].speed_after_px_frame > events[0].speed_before_px_frame);
        assert!(events[0].direction_change_deg > 120.0);
    }

    #[test]
    fn stabilized_smooth_flight_with_sparse_raw_anchors_does_not_produce_contact() {
        let mut frames = scenario(false, true, false);
        for (index, frame) in frames.iter_mut().enumerate() {
            if index % 3 != 0 {
                frame.selected_ball.as_mut().unwrap().source_model =
                    INTERPOLATED_BALL_SOURCE.into();
            }
        }
        let events = evaluate_block(&frames, 640, 360, 0, 80, TouchSettings::default()).unwrap();
        assert!(events.is_empty(), "{events:?}");
    }

    #[test]
    fn geometry_recovers_a_real_impact_without_any_unsupported_touch_logits() {
        let mut frames = scenario(true, true, false);
        for (i, row) in frames.iter_mut().enumerate() {
            row.players[0].touch_probability = None;
            let x = if i <= 32 {
                0.20 + i as f64 * 0.008
            } else {
                0.20 + 32.0 * 0.008 - (i - 32) as f64 * 0.008
            };
            row.selected_ball.as_mut().unwrap().bbox_xyxy_norm[0] = x;
            row.selected_ball.as_mut().unwrap().bbox_xyxy_norm[2] = x + 0.02;
        }
        let contacts = evaluate_block(&frames, 640, 360, 0, 80, TouchSettings::default()).unwrap();
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].provisional_track_id, Some(1));
        assert!(contacts[0].action_prediction.is_some());
        assert_eq!(contacts[0].technique, "unknown_touch");
    }

    #[test]
    fn geometry_rejects_linear_flight_far_player_and_ball_source_switch() {
        for (curve, close, switch) in [
            (false, true, false),
            (true, false, false),
            (true, true, true),
        ] {
            let mut frames = scenario(curve, close, switch);
            for row in &mut frames {
                row.players[0].touch_probability = None;
            }
            let contacts = evaluate_block(&frames, 640, 360, 0, 80, TouchSettings::default());
            if switch {
                assert!(contacts.is_err());
            } else {
                assert!(contacts.unwrap().is_empty(), "curve={curve} close={close}");
            }
        }
    }

    #[test]
    fn geometry_rejects_one_frame_wrong_ball_localization() {
        let mut frames = scenario(false, true, false);
        for row in &mut frames {
            row.players[0].touch_probability = None;
        }
        // An isolated model mislocalization must not create two fake contacts.
        frames[32].selected_ball.as_mut().unwrap().bbox_xyxy_norm[0] = 0.70;
        frames[32].selected_ball.as_mut().unwrap().bbox_xyxy_norm[2] = 0.72;
        assert!(
            evaluate_block(&frames, 640, 360, 0, 80, TouchSettings::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn unsupported_does_not_call_one_frame_ball_glitch_a_touch() {
        let mut frames = scenario(false, true, false);
        frames[32].selected_ball.as_mut().unwrap().bbox_xyxy_norm[0] = 0.70;
        frames[32].selected_ball.as_mut().unwrap().bbox_xyxy_norm[2] = 0.72;
        assert!(
            evaluate_block(&frames, 640, 360, 0, 80, TouchSettings::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn unsupported_and_geometry_never_double_count_same_impact() {
        let frames = strong_rebound_frames();
        let settings = TouchSettings {
            ..TouchSettings::default()
        };
        let contacts = evaluate_block(&frames, 640, 360, 0, 80, settings).unwrap();
        assert_eq!(contacts.len(), 1);
    }

    #[test]
    fn changing_ball_motion_plus_unsupported_peaks_and_reach_proposes_contact() {
        let rows = evaluate_block(
            &strong_rebound_frames(),
            640,
            360,
            0,
            80,
            TouchSettings {
                ..TouchSettings::default()
            },
        )
        .unwrap();
        assert!(!rows.is_empty());
        assert!(rows
            .iter()
            .all(|p| p.physical_motion_supported && !p.athlete_identity_verified));
        assert!(rows.iter().any(|p| p.provisional_track_id == Some(1)));
        assert!(rows.iter().all(|p| p.technique == "unknown_touch"));
        assert!(rows.iter().any(|p| {
            p.action_prediction.as_ref().is_some_and(|prediction| {
                prediction.kind == action_prediction::ActionKind::Dig
                    && prediction.evidence_score > 0.0
            })
        }));
    }
    #[test]
    fn straight_flight_or_distant_player_does_not_produce_contact() {
        let normal = TouchSettings::default();
        assert!(
            evaluate_block(&scenario(false, true, false), 640, 360, 0, 80, normal)
                .unwrap()
                .is_empty()
        );
        assert!(
            evaluate_block(&scenario(true, false, false), 640, 360, 0, 80, normal)
                .unwrap()
                .is_empty()
        );
    }
    #[test]
    fn model_switch_near_peak_is_not_a_physical_impact() {
        assert!(evaluate_block(
            &scenario(true, true, true),
            640,
            360,
            0,
            80,
            TouchSettings::default()
        )
        .is_err());
    }
    #[test]
    fn no_reach_or_invalid_settings_or_nonmonotonic_frames_fail_closed() {
        let mut frames = scenario(true, true, false);
        frames[4].frame_index = 7;
        assert!(evaluate_block(&frames, 640, 360, 0, 80, TouchSettings::default()).is_err());
        assert!(TouchSettings {
            smooth: 0,
            ..TouchSettings::default()
        }
        .validate()
        .is_err());
        assert!(evaluate_block(
            &scenario(true, true, false),
            640,
            360,
            80,
            81,
            TouchSettings::default()
        )
        .is_err());
    }
    #[test]
    fn nearby_track_can_support_motion_but_not_claim_missing_source_frame() {
        // The strongest smoothed peak is frame 30, but the old hint has no player
        // detection there. The +/-2 frame associator sees player 1 on 29/31.
        // The playerless peak must not become a published Touch.
        let mut frames = scenario(true, true, false);
        for (index, row) in frames.iter_mut().enumerate() {
            for player in &mut row.players {
                player.touch_probability = Some(if (28..=32).contains(&index) {
                    0.95
                } else {
                    0.01
                });
            }
        }
        frames[30].players.clear();
        let events = evaluate_block(&frames, 640, 360, 0, 80, TouchSettings::default()).unwrap();
        assert!(events.iter().all(|event| {
            event.frame_index != 30
                && event.provisional_track_id.is_some_and(|id| {
                    frames[event.frame_index as usize]
                        .players
                        .iter()
                        .any(|player| player.track_id == id)
                })
        }));
        assert!(events
            .iter()
            .all(|event| frames[event.frame_index as usize].selected_ball.is_some()));
    }

    #[test]
    fn absent_exact_frame_ball_cannot_be_published_from_nearby_motion() {
        let mut frames = scenario(true, true, false);
        for (index, row) in frames.iter_mut().enumerate() {
            for player in &mut row.players {
                player.touch_probability = Some(if (28..=32).contains(&index) {
                    0.95
                } else {
                    0.01
                });
            }
        }
        frames[30].selected_ball = None;
        let events = evaluate_block(&frames, 640, 360, 0, 80, TouchSettings::default()).unwrap();
        assert!(events
            .iter()
            .all(|event| frames[event.frame_index as usize].selected_ball.is_some()));
    }

    #[test]
    fn geometry_events_remain_inside_emission_bounds() {
        let mut frames = scenario(true, true, false);
        for (i, row) in frames.iter_mut().enumerate() {
            row.players[0].touch_probability = None;
            let x = if i <= 32 {
                0.20 + i as f64 * 0.008
            } else {
                0.20 + 32.0 * 0.008 - (i - 32) as f64 * 0.008
            };
            row.selected_ball.as_mut().unwrap().bbox_xyxy_norm[0] = x;
            row.selected_ball.as_mut().unwrap().bbox_xyxy_norm[2] = x + 0.02;
        }
        let before = evaluate_block(&frames, 640, 360, 0, 20, TouchSettings::default()).unwrap();
        let after = evaluate_block(&frames, 640, 360, 20, 80, TouchSettings::default()).unwrap();
        assert!(
            before.is_empty(),
            "A lookahead contact must not be emitted early"
        );
        assert_eq!(after.len(), 1);
    }

    #[test]
    fn emission_bounds_do_not_repeat_events() {
        let frames = strong_rebound_frames();
        let a = evaluate_block(&frames, 640, 360, 0, 30, TouchSettings::default()).unwrap();
        let b = evaluate_block(&frames, 640, 360, 30, 80, TouchSettings::default()).unwrap();
        assert!(a.iter().all(|c| c.frame_index < 30));
        assert!(b.iter().all(|c| c.frame_index >= 30));
        assert!(!b.is_empty());
    }
}
