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

pub(crate) const TOUCH_SIDE_ATTRIBUTION_VERSION: &str = "observed_serve_side_v2";

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
        evidence.side = sides.get(&track_id).copied().unwrap_or(evidence.side);
        event.court = Some(evidence);
        let Some(ref mut prediction) = event.action_prediction else {
            continue;
        };
        // Original Python: above the head AND within 1.6 m of net -> block.
        if evidence.near_net
            && matches!(
                prediction.method,
                PredictionMethod::PoseKeypointsMotionV1 | PredictionMethod::LearnedPoseV1
            )
            && matches!(
                prediction.kind,
                ActionKind::Attack | ActionKind::OverheadPass
            )
            && prediction.depth_ratio < -0.05
            && frame.selected_ball.as_ref().is_none_or(|ball| {
                let [x1, y1, x2, y2] = ball.bbox_xyxy_norm;
                court.supports_net_contact([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
            })
        {
            prediction.kind = ActionKind::Block;
            prediction.method = PredictionMethod::CourtPoseMotionV1;
        }
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
        assert_eq!(
            events[0].action_prediction.as_ref().unwrap().kind,
            ActionKind::Block
        );
        assert!(events[0].court.unwrap().near_net);
        let far = (0..60).map(|i| frame(i, 0.2, false)).collect::<Vec<_>>();
        let mut events = vec![event(30)];
        annotate(&mut events, &far, &court());
        assert_eq!(
            events[0].action_prediction.as_ref().unwrap().kind,
            ActionKind::Attack
        );
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
}
