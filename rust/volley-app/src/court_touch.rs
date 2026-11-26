//! Court-informed CPU Touch classification, using only source-frame evidence.
//! Python parity: PlayerStore foot homography and stable per-track side,
//! player_contacts block threshold, preparation/serve-zone gating.

use std::collections::{BTreeMap, HashMap};
use volley_core::{
    action_prediction::{ActionKind, PredictionMethod},
    court_geometry::{CourtSide, CourtTransform, TeamSideVotes},
    touch_pipeline::{PhysicalTouchCandidate, TouchFrame},
};

/// Discard spectators and benches from the Touch association stage.
/// Track IDs remain unchanged in the immutable CPU replay.
#[must_use]
pub(crate) fn filter_on_court(frames: &[TouchFrame], court: &CourtTransform) -> Vec<TouchFrame> {
    frames
        .iter()
        .cloned()
        .map(|mut frame| {
            frame
                .players
                .retain(|player| court.player_evidence(player.bbox_xyxy_norm).is_some());
            frame
        })
        .collect()
}

pub(crate) const TOUCH_SIDE_ATTRIBUTION_VERSION: &str = "block_opponent_side_v3";

pub(crate) fn pre_crossing_tracks(
    events: &[PhysicalTouchCandidate],
    frames: &[TouchFrame],
    court: &CourtTransform,
    launch: u64,
    server: u64,
) -> BTreeMap<u64, Vec<u64>> {
    let mut result = BTreeMap::new();
    let Some(zones) = court.zones() else {
        return result;
    };
    let Some([a, b, _, _]) = zones.net_surface else {
        return result;
    };
    let mut votes = BTreeMap::<u64, TeamSideVotes>::new();
    for frame in frames {
        for player in &frame.players {
            votes
                .entry(player.track_id)
                .or_default()
                .observe(court, player.bbox_xyxy_norm);
        }
    }
    let Some(side) = votes.get(&server).and_then(TeamSideVotes::side) else {
        return result;
    };
    let (own_x, other_x) = match side {
        CourtSide::Near => (-4.5, 4.5),
        CourtSide::Far => (4.5, -4.5),
    };
    let (Some(own), Some(other)) = (
        court.floor_to_image([own_x, 0.0]),
        court.floor_to_image([other_x, 0.0]),
    ) else {
        return result;
    };
    let signed_distance = |[a, b]: [[f64; 2]; 2], p: [f64; 2]| {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        (dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / dx.hypot(dy)
    };
    let lines = [zones.ground_divider, [a, b]];
    let on_own_side = |point| {
        lines.iter().all(|&line| {
            let sign = signed_distance(line, own);
            let opposite = signed_distance(line, other);
            sign.abs() > 0.005
                && opposite * sign < 0.0
                && signed_distance(line, point) * sign.signum() > 0.005
        })
    };
    for event in events {
        if event.frame_index <= launch.saturating_add(2)
            || event.frame_index > launch.saturating_add(90)
            || event.ball_occluded_at_contact
        {
            continue;
        }
        let mut previous = None;
        let mut valid = true;
        for frame in frames
            .iter()
            .filter(|f| (launch..=event.frame_index).contains(&f.frame_index))
        {
            let Some(ball) = frame
                .selected_ball
                .as_ref()
                .filter(|b| b.source_model == "vballnet")
            else {
                continue;
            };
            let [x1, y1, x2, y2] = ball.bbox_xyxy_norm;
            if frame.frame_index.saturating_sub(previous.unwrap_or(launch)) > 4
                || ![[x1, y1], [x1, y2], [x2, y1], [x2, y2]]
                    .into_iter()
                    .all(on_own_side)
            {
                valid = false;
                break;
            }
            previous = Some(frame.frame_index);
        }
        if !valid || previous != Some(event.frame_index) {
            continue;
        }
        if let Some(row) = frames.iter().find(|f| f.frame_index == event.frame_index) {
            let tracks = row
                .players
                .iter()
                .filter(|p| {
                    votes
                        .get(&p.track_id)
                        .and_then(TeamSideVotes::side)
                        .is_none_or(|candidate| candidate == side)
                })
                .map(|p| p.track_id)
                .collect();
            result.insert(event.frame_index, tracks);
        }
    }
    result
}
pub(crate) fn attach_occluded_followups(
    events: &mut [PhysicalTouchCandidate],
    frames: &[TouchFrame],
    court: &CourtTransform,
    dimensions: (u32, u32),
    previous_contact: Option<&PhysicalTouchCandidate>,
) {
    let Some(zones) = court.zones() else { return };
    let Some([a, b, _, _]) = zones.net_surface else {
        return;
    };
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let length = dx.hypot(dy);
    let distance = |p: [f64; 2]| (dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / length;
    let [left, right] = zones.ground_divider;
    let ground = [(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5];
    let sign = distance(ground).signum();
    if length < 0.05 || distance(ground).abs() < 0.02 {
        return;
    }
    let below_tape = |p: [f64; 2]| {
        let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (length * length);
        (-0.02..=1.02).contains(&t) && distance(p) * sign > 0.01
    };
    let mut votes = BTreeMap::<u64, TeamSideVotes>::new();
    for row in frames {
        for player in &row.players {
            votes
                .entry(player.track_id)
                .or_default()
                .observe(court, player.bbox_xyxy_norm);
        }
    }
    let side_of = |id| votes.get(&id).and_then(TeamSideVotes::side);
    let w = f64::from(dimensions.0);
    let h = f64::from(dimensions.1);
    let point = |event: &PhysicalTouchCandidate| [event.ball_x_px / w, event.ball_y_px / h];
    let observed = |event: &PhysicalTouchCandidate| {
        event.physical_motion_supported
            && event.body_contact_supported
            && !event.ball_occluded_at_contact
            && event.provisional_track_id.is_some()
    };
    let mut previous = previous_contact.filter(|e| observed(e)).cloned();
    for event in events {
        if !event.ball_occluded_at_contact {
            previous = observed(event).then(|| event.clone());
            continue;
        }
        let Some(prior) = previous.as_ref() else {
            continue;
        };
        let Some(side) = prior.provisional_track_id.and_then(side_of) else {
            continue;
        };
        let elapsed = event.frame_index.saturating_sub(prior.frame_index);
        if !event.physical_motion_supported
            || event.provisional_track_id.is_some()
            || !(3..=45).contains(&elapsed)
            || !(0.0..=5.0).contains(&event.speed_before_px_frame)
            || event.speed_after_px_frame < 14.0
            || event.speed_after_px_frame < event.speed_before_px_frame * 3.0
            || event.velocity_residual_px_frame < 14.0
            || !below_tape(point(prior))
            || !below_tape(point(event))
        {
            continue;
        }
        let mut last = None;
        let mut after = None;
        let mut valid = true;
        let mut count = 0;
        for row in frames.iter().filter(|row| {
            row.frame_index >= prior.frame_index
                && row.frame_index <= event.frame_index.saturating_add(4)
        }) {
            let Some(ball) = row
                .selected_ball
                .as_ref()
                .filter(|ball| ball.source_model == "vballnet")
            else {
                continue;
            };
            let [x1, y1, x2, y2] = ball.bbox_xyxy_norm;
            if !below_tape([(x1 + x2) * 0.5, (y1 + y2) * 0.5]) {
                valid = false;
                break;
            }
            if row.frame_index > event.frame_index {
                after = Some(row.frame_index);
                break;
            }
            if row
                .frame_index
                .saturating_sub(last.map_or(prior.frame_index, |(i, _)| i))
                > 4
            {
                valid = false;
                break;
            }
            last = Some((row.frame_index, ((x2 - x1) * w + (y2 - y1) * h) * 0.25));
            count += 1;
        }
        let (Some((before, radius)), Some(after)) = (last, after) else {
            continue;
        };
        if !valid
            || count < 4
            || before >= event.frame_index
            || event.frame_index - before > 12
            || after - before > 13
            || radius <= 0.0
        {
            continue;
        }
        let Some(row) = frames.iter().find(|r| r.frame_index == event.frame_index) else {
            continue;
        };
        let mut candidates = row
            .players
            .iter()
            .filter_map(|player| {
                if side_of(player.track_id) != Some(side) {
                    return None;
                }
                let [x1, y1, x2, y2] = player.bbox_xyxy_norm;
                let bbox = [x1 * w, y1 * h, x2 * w, y2 * h];
                let height = bbox[3] - bbox[1];
                if height < 26.0 {
                    return None;
                }
                let x = event.ball_x_px;
                let y = event.ball_y_px;
                if y > bbox[1] + 0.65 * height {
                    return None;
                }
                let gap =
                    (x - x.clamp(bbox[0], bbox[2])).hypot(y - y.clamp(bbox[1], bbox[3])) / height;
                (gap <= 0.20).then_some((gap, player, bbox, height))
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        let Some(&(gap, player, bbox, height)) = candidates.first() else {
            continue;
        };
        if candidates.get(1).is_some_and(|other| other.0 - gap < 0.12) {
            continue;
        }
        event.provisional_track_id = Some(player.track_id);
        event.inferred_body_reach_supported = true;
        event.court = court
            .player_evidence(player.bbox_xyxy_norm)
            .map(|mut evidence| {
                evidence.side = side;
                evidence
            });
        event.action_prediction = volley_core::action_prediction::predict(
            volley_core::action_prediction::PositionEvidence {
                ball_y_px: event.ball_y_px,
                ball_radius_px: radius,
                player_bbox_px: bbox,
                standing_height_px: height,
                touch_score: event.probability,
                track_score: player.score,
                context: volley_core::action_prediction::CourtContext::Unavailable,
            },
        );
        previous = None;
    }
}
fn stable_sides(frames: &[TouchFrame], court: &CourtTransform) -> HashMap<u64, CourtSide> {
    let mut positions: HashMap<u64, Vec<f64>> = HashMap::new();
    for frame in frames {
        for player in &frame.players {
            if let Some(value) = court.player_evidence(player.bbox_xyxy_norm) {
                positions
                    .entry(player.track_id)
                    .or_default()
                    .push(value.x_m);
            }
        }
    }
    positions
        .into_iter()
        .filter_map(|(id, mut samples)| {
            if samples.is_empty() {
                return None;
            }
            samples.sort_by(f64::total_cmp);
            let median = samples[samples.len() / 2];
            Some((
                id,
                if median < 0.0 {
                    CourtSide::Near
                } else {
                    CourtSide::Far
                },
            ))
        })
        .collect()
}

/// Python reach ellipse: locate ball held by the serving player before a
/// strike, without confusing an airborne projected ball with a ground point.
fn ball_held_near_player(frame: &TouchFrame, track_id: u64) -> bool {
    let Some(player) = frame.players.iter().find(|p| p.track_id == track_id) else {
        return false;
    };
    let Some(ball) = frame.selected_ball.as_ref() else {
        return false;
    };
    let [x1, y1, x2, y2] = player.bbox_xyxy_norm;
    let [bx1, by1, bx2, by2] = ball.bbox_xyxy_norm;
    let (height, width) = ((y2 - y1).max(0.001), (x2 - x1).max(0.001));
    let radius = ((bx2 - bx1).abs() + (by2 - by1).abs()) * 0.25;
    let reach_x = (width * 0.70).max(height * 0.42) + radius;
    let reach_top = y1 - (height * 0.45 + radius);
    let reach_bottom = y2 + (height * 0.15 + radius);
    let reach_y = ((reach_bottom - reach_top) * 0.5).max(0.001);
    let center_y = (reach_bottom + reach_top) * 0.5;
    let x = (bx1 + bx2) * 0.5;
    let y = (by1 + by2) * 0.5;
    ((x - (x1 + x2) * 0.5) / reach_x).hypot((y - center_y) / reach_y) <= 0.60
}

/// Python find_rally_start requires >=0.5 s of handling and then a flight.
/// For CPU Touch this is a conservative local witness (not a verified serve).
fn observed_serve_preparation(frames: &[TouchFrame], frame_index: u64, track_id: u64) -> bool {
    // Source detections can blink during the toss. Count observed evidence
    // instead of demanding a continuous 0.5s ball track.
    const PREPARATION_FRAMES: usize = 8;
    const SERVE_TOSS_FRAMES: u64 = 90;
    const RELEASE_FLIGHT_FRAMES: usize = 3;
    const MAX_MISSING_FRAMES: usize = 4;
    let mut flight = 0usize;
    let mut preparation = 0usize;
    let mut missing = 0usize;
    for frame in frames.iter().rev() {
        if frame.frame_index >= frame_index {
            continue;
        }
        if frame_index - frame.frame_index > SERVE_TOSS_FRAMES {
            break;
        }
        if frame.selected_ball.is_none() {
            missing += 1;
            if missing > MAX_MISSING_FRAMES {
                flight = 0;
                preparation = 0;
            }
            continue;
        }
        missing = 0;
        if flight < RELEASE_FLIGHT_FRAMES {
            if ball_held_near_player(frame, track_id) {
                flight = 0;
            } else {
                flight += 1;
            }
            continue;
        }
        if ball_held_near_player(frame, track_id) {
            preparation += 1;
            if preparation >= PREPARATION_FRAMES {
                return true;
            }
        } else {
            preparation = 0;
        }
    }
    false
}

/// A jump-serve impact may occur inside the court projection, after takeoff
/// behind the end line. Preserve observed server origin, not ball-ground projection.
fn serve_origin_seen(
    frames: &[TouchFrame],
    frame_index: u64,
    track_id: u64,
    court: &CourtTransform,
) -> bool {
    frames
        .iter()
        .rev()
        .filter(|frame| frame.frame_index <= frame_index && frame_index - frame.frame_index <= 45)
        .filter_map(|frame| {
            frame
                .players
                .iter()
                .find(|player| player.track_id == track_id)
        })
        .filter_map(|player| court.player_evidence(player.bbox_xyxy_norm))
        .any(|evidence| evidence.in_serve_zone)
}

/// Annotate only same-frame valid Track IDs. Pose evidence is already applied,
/// and court constraints are the last refinement of the unverified technique.
pub(crate) fn annotate(
    events: &mut [PhysicalTouchCandidate],
    frames: &[TouchFrame],
    court: &CourtTransform,
) {
    let Some(first) = frames.first().map(|f| f.frame_index) else {
        return;
    };
    let sides = stable_sides(frames, court);
    for event in events.iter_mut() {
        let Some(track_id) = event.provisional_track_id else {
            continue;
        };
        let Some(frame) = event
            .frame_index
            .checked_sub(first)
            .and_then(|idx| frames.get(idx as usize))
            .filter(|f| f.frame_index == event.frame_index)
        else {
            continue;
        };
        let Some(player) = frame.players.iter().find(|p| p.track_id == track_id) else {
            continue;
        };
        let Some(mut evidence) = court.player_evidence(player.bbox_xyxy_norm) else {
            continue;
        };
        if event.action_prediction.as_ref().is_some_and(|p| {
            p.kind == ActionKind::Block && p.method == PredictionMethod::CourtPoseMotionV1
        }) {
            if let Some(sample) = event.court {
                evidence = sample;
            }
        }
        evidence.side = sides.get(&track_id).copied().unwrap_or(evidence.side);
        event.court = Some(evidence);
        let Some(ref mut prediction) = event.action_prediction else {
            continue;
        };
        // Original Python: serving zone alone is not enough to call a serve.
        // There must be preparation, source-frame reach, and acceleration.
        if (evidence.in_serve_zone || serve_origin_seen(frames, event.frame_index, track_id, court))
            && matches!(
                prediction.method,
                PredictionMethod::PoseKeypointsMotionV1
                    | PredictionMethod::LearnedPoseV1
                    | PredictionMethod::CourtPoseMotionV1
            )
            && event.speed_after_px_frame / event.speed_before_px_frame.max(0.1) >= 1.2
            && observed_serve_preparation(frames, event.frame_index, track_id)
        {
            prediction.kind = ActionKind::Serve;
            prediction.method = PredictionMethod::CourtPoseMotionV1;
        }
    }
}

pub(crate) fn reconcile_block_origin(
    event: &mut PhysicalTouchCandidate,
    previous_side: Option<CourtSide>,
    net_contact: bool,
) {
    let Some(prediction) = event.action_prediction.as_mut() else {
        return;
    };
    let sides = previous_side.zip(event.court.map(|c| c.side));
    let incoming_opponent = sides.is_some_and(|(previous, current)| previous != current);
    if prediction.kind == ActionKind::Block {
        if net_contact && incoming_opponent {
            return;
        }
        if sides.is_some_and(|(previous, current)| previous == current) {
            prediction.kind = ActionKind::OverheadPass;
        } else {
            event.action_prediction = None;
        }
        return;
    }
    if incoming_opponent
        && net_contact
        && event.court.is_some_and(|c| c.near_net)
        && matches!(
            prediction.method,
            PredictionMethod::PoseKeypointsMotionV1 | PredictionMethod::LearnedPoseV1
        )
        && matches!(
            prediction.kind,
            ActionKind::Attack | ActionKind::OverheadPass
        )
        && prediction.depth_ratio < -0.05
    {
        prediction.kind = ActionKind::Block;
        prediction.method = PredictionMethod::CourtPoseMotionV1;
    }
}

pub(crate) fn recognize_jump_block(
    event: &mut PhysicalTouchCandidate,
    frames: &[TouchFrame],
    poses: &[volley_core::touch_pipeline::PoseFrame],
    court: &CourtTransform,
    dimensions: (u32, u32),
    settings: volley_core::touch_pipeline::TouchSettings,
    eligible: Option<&[u64]>,
) -> bool {
    use volley_core::{action_prediction::ActionPrediction, pose_contact::BodyContact};
    if !event.physical_motion_supported
        || event.ball_occluded_at_contact
        || event.direction_change_deg < 75.0
        || event.velocity_residual_px_frame < 18.0
        || event
            .action_prediction
            .as_ref()
            .is_some_and(|p| matches!(p.kind, ActionKind::Serve | ActionKind::Block))
    {
        return false;
    }
    let Some([a, b, _, _]) = court.zones().and_then(|z| z.net_surface) else {
        return false;
    };
    let (w, h) = (f64::from(dimensions.0), f64::from(dimensions.1));
    if w < 64.0 || h < 64.0 || (b[0] - a[0]).abs() < 0.001 {
        return false;
    }
    let Some(first) = frames.first().map(|f| f.frame_index) else {
        return false;
    };
    let row_at = |f: u64| {
        f.checked_sub(first)
            .and_then(|i| frames.get(i as usize))
            .filter(|r| r.frame_index == f)
    };
    let at = event.frame_index;
    let Some(current) = row_at(at) else {
        return false;
    };
    let ball_point = [event.ball_x_px, event.ball_y_px];
    if !court.supports_net_contact([ball_point[0] / w, ball_point[1] / h]) {
        return false;
    }
    let observed_ball = |row: &TouchFrame| {
        let ball = row
            .selected_ball
            .as_ref()
            .filter(|b| b.source_model == "vballnet" && b.score >= settings.min_ball_score)?;
        let [x1, y1, x2, y2] = ball.bbox_xyxy_norm;
        let radius = ((x2 - x1) * w + (y2 - y1) * h) * 0.25;
        (radius.is_finite() && radius > 0.0)
            .then_some(([(x1 + x2) * w * 0.5, (y1 + y2) * h * 0.5], radius))
    };
    let Some((observed, event_radius)) = observed_ball(current) else {
        return false;
    };
    let distance = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).hypot(p[1] - q[1]);
    if distance(observed, ball_point) > event_radius.max(2.0) {
        return false;
    }
    let point = |p: &volley_core::touch_pipeline::PosePlayer, i: usize| {
        let [x, y, c] = p.keypoints[i];
        (!p.interpolated
            && (0.0..=1.0).contains(&x)
            && (0.0..=1.0).contains(&y)
            && (settings.pose_min_confidence.max(0.35)..=1.0).contains(&c))
        .then_some([x * w, y * h])
    };
    let midpoint_y = |p: &volley_core::touch_pipeline::PosePlayer, i, j| {
        Some((point(p, i)?[1] + point(p, j)?[1]) * 0.5)
    };
    let near_tape = |p: [f64; 2], height: f64| {
        let t = (p[0] / w - a[0]) / (b[0] - a[0]);
        let y = (a[1] + t * (b[1] - a[1])) * h;
        (-0.05..=1.05).contains(&t) && (-0.35 * height..=0.06 * height).contains(&(p[1] - y))
    };
    let mut candidates = BTreeMap::<
        u64,
        (
            f64,
            u64,
            BodyContact,
            ActionPrediction,
            volley_core::court_geometry::CourtContactEvidence,
        ),
    >::new();
    for sample in poses.iter().filter(|p| p.frame_index.abs_diff(at) <= 1) {
        let Some(row) = row_at(sample.frame_index) else {
            continue;
        };
        let Some((sample_ball, radius)) = observed_ball(row) else {
            continue;
        };
        if distance(sample_ball, ball_point)
            > event_radius + 2.0 * event.speed_before_px_frame.max(event.speed_after_px_frame)
        {
            continue;
        }
        for pose in &sample.players {
            let id = pose.track_id;
            if eligible.is_some_and(|ids| !ids.contains(&id))
                || (event.provisional_track_id != Some(id)
                    && event.contact_landmark.is_some_and(|p| {
                        p.joint_indices.iter().all(|i| (7..=10).contains(i))
                            && p.normalized_distance <= 0.04
                    }))
            {
                continue;
            }
            let Some(current_player) = current.players.iter().find(|p| p.track_id == id) else {
                continue;
            };
            let Some(player) = row.players.iter().find(|p| p.track_id == id) else {
                continue;
            };
            let Some(context) = court
                .player_evidence(player.bbox_xyxy_norm)
                .filter(|p| p.near_net)
            else {
                continue;
            };
            if !court
                .player_evidence(current_player.bbox_xyxy_norm)
                .is_some_and(|p| p.x_m.abs() <= 2.1 && p.side == context.side)
            {
                continue;
            }
            let bbox = player.bbox_xyxy_norm;
            let height = (bbox[3] - bbox[1]) * h;
            if height < 26.0 || radius > 0.2 * height || event_radius > 0.2 * height {
                continue;
            }
            let arm = |s, e, r| {
                let (s, e, r) = (point(pose, s)?, point(pose, e)?, point(pose, r)?);
                let u = [e[0] - s[0], e[1] - s[1]];
                let v = [r[0] - e[0], r[1] - e[1]];
                let (l1, l2) = (u[0].hypot(u[1]), v[0].hypot(v[1]));
                (l1 >= 0.04 * height
                    && l2 >= 0.04 * height
                    && e[1] < s[1]
                    && r[1] < e[1]
                    && s[1] - r[1] >= 0.20 * height
                    && (u[0] * v[0] + u[1] * v[1]) / (l1 * l2) >= 0.80
                    && near_tape(r, height))
                .then_some(r)
            };
            let (Some(left), Some(right)) = (arm(5, 7, 9), arm(6, 8, 10)) else {
                continue;
            };
            if !(0.07 * height..=0.35 * height).contains(&distance(left, right)) {
                continue;
            }
            let (Some(hips), Some(feet)) = (midpoint_y(pose, 11, 12), midpoint_y(pose, 15, 16))
            else {
                continue;
            };
            let (Some(shoulder_l), Some(shoulder_r)) = (point(pose, 5), point(pose, 6)) else {
                continue;
            };
            if hips - (shoulder_l[1] + shoulder_r[1]) * 0.5 < 0.14 * height
                || feet - hips < 0.20 * height
            {
                continue;
            }
            let jumped = poses
                .iter()
                .filter(|p| p.frame_index <= at.saturating_sub(8) && at - p.frame_index <= 24)
                .any(|p| {
                    let Some(prior) = p
                        .players
                        .iter()
                        .find(|q| q.track_id == id && !q.interpolated)
                    else {
                        return false;
                    };
                    let Some(prior_player) = row_at(p.frame_index)
                        .and_then(|r| r.players.iter().find(|q| q.track_id == id))
                    else {
                        return false;
                    };
                    court
                        .player_evidence(prior_player.bbox_xyxy_norm)
                        .is_some_and(|c| c.side == context.side && c.x_m.abs() <= 2.8)
                        && midpoint_y(prior, 11, 12).is_some_and(|y| y - hips >= 0.07 * height)
                        && midpoint_y(prior, 15, 16).is_some_and(|y| y - feet >= 0.05 * height)
                });
            if !jumped {
                continue;
            }
            let (joint, wrist) = if distance(left, sample_ball) <= distance(right, sample_ball) {
                (9, left)
            } else {
                (10, right)
            };
            let sample_distance = distance(wrist, sample_ball);
            let event_distance = distance(wrist, ball_point);
            if sample_distance > radius + 0.06 * height
                || event_distance > event_radius + 0.14 * height
            {
                continue;
            }
            let contact = BodyContact {
                joint_indices: [joint, joint],
                point_px: wrist,
                distance_px: event_distance,
                normalized_distance: ((event_distance - event_radius).max(0.0) / height).min(1.0),
            };
            let prediction = ActionPrediction {
                kind: ActionKind::Block,
                method: PredictionMethod::CourtPoseMotionV1,
                evidence_score: (event.probability
                    * pose.keypoints[9][2].min(pose.keypoints[10][2])
                    * (1.0 - contact.normalized_distance))
                    .clamp(0.0, 1.0),
                depth_ratio: ((ball_point[1] - event_radius - bbox[1] * h) / height)
                    .clamp(-2.0, 3.0),
                posture_ratio: ((bbox[2] - bbox[0]) * w / height).clamp(0.0, 4.0),
            };
            let rank = (sample_distance - radius).max(0.0) / height
                + 0.01 * sample.frame_index.abs_diff(at) as f64;
            if prediction.valid()
                && contact.valid(dimensions.0, dimensions.1)
                && candidates.get(&id).is_none_or(|old| rank < old.0)
            {
                candidates.insert(id, (rank, sample.frame_index, contact, prediction, context));
            }
        }
    }
    if candidates.len() != 1 {
        return false;
    }
    let Some((id, (_, sample, contact, prediction, context))) = candidates.into_iter().next()
    else {
        return false;
    };
    event.provisional_track_id = Some(id);
    event.body_contact_supported = true;
    event.inferred_body_reach_supported = false;
    event.contact_landmark = Some(contact);
    event.contact_pose_sample_frame = Some(sample);
    event.action_prediction = Some(prediction);
    event.court = Some(context);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use volley_core::action_prediction::{ActionPrediction, PredictionMethod};

    fn court() -> CourtTransform {
        CourtTransform::new([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]).unwrap()
    }

    fn frame(index: u64, foot_x: f64, ball: bool) -> TouchFrame {
        use volley_core::touch_pipeline::{TouchBall, TouchPlayer};
        let x1 = foot_x - 0.03;
        let x2 = foot_x + 0.03;
        TouchFrame {
            frame_index: index,
            players: vec![TouchPlayer {
                track_id: 3,
                score: 0.96,
                touch_probability: None,
                bbox_xyxy_norm: [x1, 0.35, x2, 0.75],
            }],
            selected_ball: ball.then_some(TouchBall {
                source_model: "vballnet".into(),
                score: 0.95,
                bbox_xyxy_norm: [foot_x - 0.01, 0.48, foot_x + 0.01, 0.50],
            }),
        }
    }

    fn event(index: u64) -> PhysicalTouchCandidate {
        PhysicalTouchCandidate {
            frame_index: index,
            provisional_track_id: Some(3),
            probability: 0.9,
            ball_x_px: 320.0,
            ball_y_px: 100.0,
            motion_frame_index: index,
            direction_change_deg: 45.0,
            speed_before_px_frame: 2.0,
            speed_after_px_frame: 9.0,
            velocity_residual_px_frame: 8.0,
            technique: "unknown_touch",
            action_prediction: Some(ActionPrediction {
                kind: ActionKind::Attack,
                evidence_score: 0.92,
                method: PredictionMethod::PoseKeypointsMotionV1,
                depth_ratio: -0.1,
                posture_ratio: 0.3,
            }),
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
    fn early_serve_flight_excludes_opponents_until_observed_crossing() {
        let court = court().with_net(Some([[0.5, 0.05], [0.5, 0.65]])).unwrap();
        let mut frames = (0..20).map(|i| frame(i, 0.30, true)).collect::<Vec<_>>();
        for row in &mut frames {
            row.players.push(volley_core::touch_pipeline::TouchPlayer {
                track_id: 8,
                score: 0.9,
                touch_probability: None,
                bbox_xyxy_norm: [0.40, 0.25, 0.72, 0.75],
            });
            row.players.push(volley_core::touch_pipeline::TouchPlayer {
                track_id: 11,
                score: 0.9,
                touch_probability: None,
                bbox_xyxy_norm: [0.35, 0.35, 0.43, 0.75],
            });
            let x = 0.30 + row.frame_index as f64 * 0.007;
            row.selected_ball.as_mut().unwrap().bbox_xyxy_norm = [x - 0.005, 0.4, x + 0.005, 0.41];
        }
        let mut hit = event(19);
        hit.ball_x_px = 0.433 * 1920.0;
        hit.ball_y_px = 0.405 * 1080.0;
        let allowed = pre_crossing_tracks(std::slice::from_ref(&hit), &frames, &court, 0, 3);
        assert_eq!(allowed.get(&19), Some(&vec![3, 11]));
        let mut baseline = hit.clone();
        volley_core::touch_pipeline::attach_touch_source_evidence(
            std::slice::from_mut(&mut baseline),
            &frames,
            1920,
            1080,
            0.75,
            |_, _| true,
        );
        assert_eq!(baseline.provisional_track_id, Some(8));
        let mut constrained = hit.clone();
        volley_core::touch_pipeline::attach_touch_source_evidence(
            std::slice::from_mut(&mut constrained),
            &frames,
            1920,
            1080,
            0.75,
            |frame, track| allowed.get(&frame).is_none_or(|ids| ids.contains(&track)),
        );
        assert_eq!(constrained.provisional_track_id, Some(11));
        frames[10].selected_ball.as_mut().unwrap().bbox_xyxy_norm = [0.55, 0.4, 0.56, 0.41];
        assert!(pre_crossing_tracks(std::slice::from_ref(&hit), &frames, &court, 0, 3).is_empty());
        frames[10].selected_ball = None;
        frames[11].selected_ball = None;
        frames[12].selected_ball = None;
        frames[13].selected_ball = None;
        assert!(pre_crossing_tracks(&[hit], &frames, &court, 0, 3).is_empty());
    }
    #[test]
    fn occluded_followup_stays_with_previous_contact_side_below_net_tape() {
        let court = CourtTransform::new([[0.0, 1.05], [0.40, 0.48], [0.64, 0.46], [0.93, 1.05]])
            .unwrap()
            .with_net(Some([[0.26, 0.41], [0.73, 0.39]]))
            .unwrap();
        let mut frames = (0..32).map(|i| frame(i, 0.55, false)).collect::<Vec<_>>();
        for row in &mut frames {
            row.players[0].bbox_xyxy_norm = [0.49, 0.76, 0.59, 0.99];
            row.players.push(volley_core::touch_pipeline::TouchPlayer {
                track_id: 11,
                score: 0.9,
                touch_probability: None,
                bbox_xyxy_norm: [0.503, 0.493, 0.543, 0.702],
            });
            row.players.push(volley_core::touch_pipeline::TouchPlayer {
                track_id: 8,
                score: 0.9,
                touch_probability: None,
                bbox_xyxy_norm: [0.485, 0.45, 0.529, 0.53],
            });
            let i = row.frame_index;
            if i <= 15 || i >= 23 {
                let y = if i <= 15 {
                    0.78 - (i as f64 / 15.0) * 0.26
                } else {
                    0.448 - (i - 23) as f64 * 0.025
                };
                row.selected_ball = Some(volley_core::touch_pipeline::TouchBall {
                    source_model: "vballnet".into(),
                    score: 0.8,
                    bbox_xyxy_norm: [0.485, y - 0.012, 0.505, y + 0.012],
                });
            }
        }
        let mut previous = event(0);
        previous.body_contact_supported = true;
        previous.ball_x_px = 0.495 * 1920.0;
        previous.ball_y_px = 0.78 * 1080.0;
        let mut hit = event(21);
        hit.provisional_track_id = None;
        hit.action_prediction = None;
        hit.ball_occluded_at_contact = true;
        hit.ball_x_px = 0.497 * 1920.0;
        hit.ball_y_px = 0.508 * 1080.0;
        hit.speed_before_px_frame = 2.1;
        hit.speed_after_px_frame = 31.0;
        hit.velocity_residual_px_frame = 29.0;
        let mut events = vec![hit.clone()];
        attach_occluded_followups(&mut events, &frames, &court, (1920, 1080), Some(&previous));
        assert_eq!(events[0].provisional_track_id, Some(11));
        assert!(events[0].inferred_body_reach_supported);
        assert!(!events[0].body_contact_supported);
        assert!(events[0].contact_landmark.is_none());
        assert!(events[0].ball_occluded_at_contact);
        frames[10].selected_ball.as_mut().unwrap().bbox_xyxy_norm = [0.485, 0.35, 0.505, 0.37];
        let mut events = vec![hit];
        attach_occluded_followups(&mut events, &frames, &court, (1920, 1080), Some(&previous));
        assert_eq!(events[0].provisional_track_id, None);
    }
    #[test]
    fn spectators_are_excluded_but_source_track_ids_remain() {
        let mut frames = vec![frame(0, 0.5, false)];
        frames[0]
            .players
            .push(volley_core::touch_pipeline::TouchPlayer {
                track_id: 91,
                score: 0.92,
                touch_probability: None,
                bbox_xyxy_norm: [0.97, 0.32, 1.0, 0.74],
            });
        let kept = filter_on_court(&frames, &court());
        assert_eq!(kept[0].players.len(), 2);
        frames[0].players[1].bbox_xyxy_norm = [0.01, 0.32, 0.04, 0.74];
        let kept = filter_on_court(&frames, &court());
        assert_eq!(kept[0].players.len(), 2); // still within 3.5 m end margin
        frames[0].players[1].bbox_xyxy_norm = [0.01, 0.02, 0.04, 0.06];
        let kept = filter_on_court(&frames, &court());
        assert_eq!(kept[0].players.len(), 1);
        assert_eq!(kept[0].players[0].track_id, 3);
    }

    #[test]
    fn blocks_need_overhead_hit_and_net_proximity() {
        let frames = (0..60).map(|i| frame(i, 0.5, false)).collect::<Vec<_>>();
        let mut events = vec![event(30)];
        annotate(&mut events, &frames, &court());
        let side = events[0].court.unwrap().side;
        let other = match side {
            CourtSide::Near => CourtSide::Far,
            CourtSide::Far => CourtSide::Near,
        };
        reconcile_block_origin(&mut events[0], Some(other), true);
        assert_eq!(
            events[0].action_prediction.as_ref().unwrap().kind,
            ActionKind::Block
        );
        assert!(events[0].court.unwrap().near_net);
        let far = (0..60).map(|i| frame(i, 0.2, false)).collect::<Vec<_>>();
        let mut events = vec![event(30)];
        annotate(&mut events, &far, &court());
        let other = match events[0].court.unwrap().side {
            CourtSide::Near => CourtSide::Far,
            CourtSide::Far => CourtSide::Near,
        };
        reconcile_block_origin(&mut events[0], Some(other), true);
        assert_eq!(
            events[0].action_prediction.as_ref().unwrap().kind,
            ActionKind::Attack
        );
    }

    #[test]
    fn block_requires_opponent_contact_in_same_rally() {
        let frames = (0..60).map(|i| frame(i, 0.5, false)).collect::<Vec<_>>();
        let mut events = vec![event(30)];
        annotate(&mut events, &frames, &court());
        let attack = events.pop().unwrap();
        let side = attack.court.unwrap().side;
        let other = match side {
            CourtSide::Near => CourtSide::Far,
            CourtSide::Far => CourtSide::Near,
        };
        let mut same_attack = attack.clone();
        reconcile_block_origin(&mut same_attack, Some(side), true);
        assert_eq!(
            same_attack.action_prediction.unwrap().kind,
            ActionKind::Attack
        );
        let mut pass = attack.clone();
        pass.action_prediction.as_mut().unwrap().kind = ActionKind::OverheadPass;
        let mut same = pass.clone();
        reconcile_block_origin(&mut same, Some(side), true);
        assert_eq!(
            same.action_prediction.unwrap().kind,
            ActionKind::OverheadPass
        );
        let mut opposite = pass.clone();
        reconcile_block_origin(&mut opposite, Some(other), true);
        assert_eq!(opposite.action_prediction.unwrap().kind, ActionKind::Block);
        let mut far = pass.clone();
        reconcile_block_origin(&mut far, Some(other), false);
        assert_eq!(
            far.action_prediction.unwrap().kind,
            ActionKind::OverheadPass
        );
        let mut unknown = pass.clone();
        reconcile_block_origin(&mut unknown, None, true);
        assert_eq!(
            unknown.action_prediction.unwrap().kind,
            ActionKind::OverheadPass
        );
        let mut jump = pass;
        jump.action_prediction.as_mut().unwrap().kind = ActionKind::Block;
        let mut own_jump = jump.clone();
        reconcile_block_origin(&mut own_jump, Some(side), true);
        assert_eq!(
            own_jump.action_prediction.unwrap().kind,
            ActionKind::OverheadPass
        );
        let mut opponent_jump = jump.clone();
        reconcile_block_origin(&mut opponent_jump, Some(other), true);
        assert_eq!(
            opponent_jump.action_prediction.unwrap().kind,
            ActionKind::Block
        );
        reconcile_block_origin(&mut jump, None, true);
        assert!(jump.action_prediction.is_none());
    }

    #[test]
    fn serve_requires_preparation_not_merely_baseline_location() {
        let mut frames = (0..60).map(|i| frame(i, 0.09, true)).collect::<Vec<_>>();
        for frame in frames
            .iter_mut()
            .filter(|f| (26..=44).contains(&f.frame_index))
        {
            frame.selected_ball.as_mut().unwrap().bbox_xyxy_norm = [0.65, 0.1, 0.67, 0.12];
        }
        let mut events = vec![event(50)];
        annotate(&mut events, &frames, &court());
        assert_eq!(
            events[0].action_prediction.as_ref().unwrap().kind,
            ActionKind::Serve
        );
        let frames = (0..60).map(|i| frame(i, 0.09, false)).collect::<Vec<_>>();
        let mut events = vec![event(50)];
        annotate(&mut events, &frames, &court());
        assert_eq!(
            events[0].action_prediction.as_ref().unwrap().kind,
            ActionKind::Attack
        );
    }
    #[test]
    fn serve_survives_short_ball_detection_holes() {
        let mut frames = (0..65).map(|i| frame(i, 0.09, true)).collect::<Vec<_>>();
        for frame in frames
            .iter_mut()
            .filter(|f| (27..=45).contains(&f.frame_index))
        {
            frame.selected_ball.as_mut().unwrap().bbox_xyxy_norm = [0.65, 0.1, 0.67, 0.12];
        }
        for index in [4, 9, 15, 20, 29, 35, 41] {
            frames[index].selected_ball = None;
        }
        let mut events = vec![event(52)];
        annotate(&mut events, &frames, &court());
        assert_eq!(
            events[0].action_prediction.as_ref().unwrap().kind,
            ActionKind::Serve
        );
    }

    #[test]
    fn jump_serve_uses_observed_takeoff_behind_end_line() {
        let mut frames = (0..70).map(|i| frame(i, 0.09, true)).collect::<Vec<_>>();
        for sample in &mut frames {
            if (28..=49).contains(&sample.frame_index) {
                sample.selected_ball.as_mut().unwrap().bbox_xyxy_norm = [0.65, 0.1, 0.67, 0.12];
            }
            if sample.frame_index >= 50 {
                sample.players[0].bbox_xyxy_norm = [0.13, 0.35, 0.19, 0.75];
            }
        }
        let mut events = vec![event(55)];
        annotate(&mut events, &frames, &court());
        assert!(!events[0].court.unwrap().in_serve_zone);
        assert_eq!(
            events[0].action_prediction.as_ref().unwrap().kind,
            ActionKind::Serve
        );
    }

    #[test]
    fn jump_block_overrides_projected_rear_shoulder_only_with_measured_hand_contact() {
        use volley_core::touch_pipeline::{
            PoseFrame, PosePlayer, TouchBall, TouchPlayer, TouchSettings,
        };
        let court = CourtTransform::new([[0.06, 1.02], [0.34, 0.29], [0.59, 0.28], [0.89, 1.02]])
            .unwrap()
            .with_net(Some([[0.245, 0.278], [0.689, 0.272]]))
            .unwrap();
        let mut frames = Vec::new();
        let mut poses = Vec::new();
        for f in 0..=32 {
            let at = (19..=21).contains(&f);
            let bbox = if at {
                [0.54, 0.277, 0.58, 0.499]
            } else {
                [0.53, 0.35, 0.57, 0.535]
            };
            let mut k = [[0.0; 3]; 17];
            let j = if at {
                [
                    (5, 0.559, 0.336),
                    (6, 0.576, 0.333),
                    (7, 0.559, 0.305),
                    (8, 0.579, 0.307),
                    (9, 0.563, 0.280),
                    (10, 0.578, 0.280),
                    (11, 0.560, 0.395),
                    (12, 0.570, 0.394),
                    (15, 0.557, 0.460),
                    (16, 0.567, 0.465),
                ]
            } else {
                [
                    (5, 0.545, 0.386),
                    (6, 0.563, 0.386),
                    (7, 0.540, 0.401),
                    (8, 0.570, 0.400),
                    (9, 0.539, 0.421),
                    (10, 0.566, 0.421),
                    (11, 0.550, 0.439),
                    (12, 0.560, 0.439),
                    (15, 0.548, 0.520),
                    (16, 0.558, 0.520),
                ]
            };
            for (i, x, y) in j {
                k[i] = [x, y, 0.8];
            }
            let ball = if f == 21 {
                [0.565, 0.281]
            } else if f <= 20 {
                [
                    0.547 - (20 - f) as f64 * 0.004,
                    0.292 - (20 - f) as f64 * 0.012,
                ]
            } else {
                [0.55, 0.28 - (f - 21) as f64 * 0.015]
            };
            frames.push(TouchFrame {
                frame_index: f,
                players: vec![
                    TouchPlayer {
                        track_id: 62,
                        score: 0.9,
                        touch_probability: None,
                        bbox_xyxy_norm: bbox,
                    },
                    TouchPlayer {
                        track_id: 64,
                        score: 0.9,
                        touch_probability: None,
                        bbox_xyxy_norm: [0.525, 0.248, 0.551, 0.368],
                    },
                ],
                selected_ball: Some(TouchBall {
                    source_model: "vballnet".into(),
                    score: 0.9,
                    bbox_xyxy_norm: [
                        ball[0] - 0.005,
                        ball[1] - 0.009,
                        ball[0] + 0.005,
                        ball[1] + 0.009,
                    ],
                }),
            });
            poses.push(PoseFrame {
                frame_index: f,
                players: vec![PosePlayer {
                    track_id: 62,
                    keypoints: k,
                    interpolated: false,
                }],
            });
        }
        let mut hit = event(20);
        hit.provisional_track_id = Some(64);
        hit.ball_x_px = 0.547 * 3840.0;
        hit.ball_y_px = 0.292 * 2160.0;
        hit.direction_change_deg = 138.0;
        hit.speed_before_px_frame = 32.0;
        hit.speed_after_px_frame = 15.0;
        hit.velocity_residual_px_frame = 45.0;
        hit.action_prediction = None;
        hit.body_contact_supported = true;
        hit.contact_landmark = Some(volley_core::pose_contact::BodyContact {
            joint_indices: [5, 5],
            point_px: [2100., 616.],
            distance_px: 14.,
            normalized_distance: 0.0,
        });
        let original = hit.clone();
        let settings = TouchSettings::default();
        assert!(recognize_jump_block(
            &mut hit,
            &frames,
            &poses,
            &court,
            (3840, 2160),
            settings,
            None
        ));
        assert_eq!(hit.provisional_track_id, Some(62));
        assert_eq!(hit.action_prediction.unwrap().kind, ActionKind::Block);
        assert!(hit
            .contact_landmark
            .unwrap()
            .joint_indices
            .iter()
            .all(|i| (9..=10).contains(i)));
        let mut no_jump = poses.clone();
        for p in &mut no_jump {
            if p.frame_index < 19 {
                let k = &mut p.players[0].keypoints;
                for i in [11, 12] {
                    k[i][1] = 0.395;
                }
                for i in [15, 16] {
                    k[i][1] = 0.462;
                }
            }
        }
        let mut hit = original.clone();
        assert!(!recognize_jump_block(
            &mut hit,
            &frames,
            &no_jump,
            &court,
            (3840, 2160),
            settings,
            None
        ));
        let mut hit = original.clone();
        assert!(!recognize_jump_block(
            &mut hit,
            &frames,
            &poses,
            &court,
            (3840, 2160),
            settings,
            Some(&[64])
        ));
        let mut no_arms = poses.clone();
        for p in &mut no_arms {
            p.players[0].keypoints[10][1] = 0.36;
        }
        let mut hit = original.clone();
        assert!(!recognize_jump_block(
            &mut hit,
            &frames,
            &no_arms,
            &court,
            (3840, 2160),
            settings,
            None
        ));
        let mut hit = original;
        hit.ball_x_px = 0.40 * 3840.0;
        assert!(!recognize_jump_block(
            &mut hit,
            &frames,
            &poses,
            &court,
            (3840, 2160),
            settings,
            None
        ));
    }
}
