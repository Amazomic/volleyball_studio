//! Conservative RTMPose Body17 volleyball technique refinement.
//! Pose evidence is only used after a motion-supported ball contact exists.
//! No serving or blocking verdict without verified rally/court context.
use crate::action_prediction::{ActionKind, ActionPrediction, PredictionMethod};

pub const ACTION_PREDICTION_POLICY_VERSION: &str = "pose_keypoints_motion_v8_or_unknown_touch";

#[derive(Clone, Copy, Debug)]
pub struct PoseEvidence {
    /// COCO17 points [x_norm,y_norm,confidence]; input contains no athlete ID.
    pub keypoints: [[f64; 3]; 17],
    pub ball_x_norm: f64,
    pub ball_y_norm: f64,
    pub video_width: f64,
    pub video_height: f64,
    pub player_bbox_norm: [f64; 4],
}

fn pixel(e: &PoseEvidence, index: usize) -> Option<[f64; 2]> {
    let [x, y, confidence] = e.keypoints[index];
    if x.is_finite()
        && y.is_finite()
        && confidence.is_finite()
        && (0.0..=1.0).contains(&x)
        && (0.0..=1.0).contains(&y)
        && confidence >= 0.35
    {
        Some([x * e.video_width, y * e.video_height])
    } else {
        None
    }
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn segment_distance(point: [f64; 2], from: [f64; 2], to: [f64; 2]) -> f64 {
    let delta = [to[0] - from[0], to[1] - from[1]];
    let length_sq = delta[0] * delta[0] + delta[1] * delta[1];
    if length_sq <= f64::EPSILON {
        return distance(point, from);
    }
    let fraction = (((point[0] - from[0]) * delta[0] + (point[1] - from[1]) * delta[1])
        / length_sq)
        .clamp(0.0, 1.0);
    distance(
        point,
        [from[0] + delta[0] * fraction, from[1] + delta[1] * fraction],
    )
}

pub fn contact_distance(e: &PoseEvidence, min_confidence: f64) -> Option<f64> {
    if !min_confidence.is_finite()
        || !(0.0..=1.0).contains(&min_confidence)
        || !e.video_width.is_finite()
        || !e.video_height.is_finite()
        || e.video_width < 64.0
        || e.video_height < 64.0
        || !e.ball_x_norm.is_finite()
        || !e.ball_y_norm.is_finite()
        || !(0.0..=1.0).contains(&e.ball_x_norm)
        || !(0.0..=1.0).contains(&e.ball_y_norm)
    {
        return None;
    }
    let height = (e.player_bbox_norm[3] - e.player_bbox_norm[1]) * e.video_height;
    if !height.is_finite() || height < 26.0 {
        return None;
    }
    let ball = [
        e.ball_x_norm * e.video_width,
        e.ball_y_norm * e.video_height,
    ];
    (0..17)
        .filter_map(|i| {
            let [x, y, confidence] = e.keypoints[i];
            if !x.is_finite()
                || !y.is_finite()
                || !confidence.is_finite()
                || !(0.0..=1.0).contains(&x)
                || !(0.0..=1.0).contains(&y)
                || !(min_confidence..=1.0).contains(&confidence)
            {
                return None;
            }
            Some(distance([x * e.video_width, y * e.video_height], ball) / height)
        })
        .min_by(f64::total_cmp)
}

/// Use wrists, elbows and shoulders; when anything is ambiguous, keep
/// the older, source-backed contact geometry classification unchanged.
#[must_use]
pub fn refine(e: &PoseEvidence, geometry: &ActionPrediction) -> Option<ActionPrediction> {
    if !geometry.valid()
        || ![e.ball_x_norm, e.ball_y_norm, e.video_width, e.video_height]
            .iter()
            .all(|v| v.is_finite())
        || !(0.0..=1.0).contains(&e.ball_x_norm)
        || !(0.0..=1.0).contains(&e.ball_y_norm)
        || e.video_width < 64.0
        || e.video_height < 64.0
    {
        return None;
    }
    let [x1, y1, x2, y2] = e.player_bbox_norm;
    if ![x1, y1, x2, y2]
        .iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        || x1 >= x2
        || y1 >= y2
    {
        return None;
    }
    let height = (y2 - y1) * e.video_height;
    if height < 26.0 {
        return None;
    }
    let ball = [
        e.ball_x_norm * e.video_width,
        e.ball_y_norm * e.video_height,
    ];
    let shoulder_left = pixel(e, 5);
    let shoulder_right = pixel(e, 6);
    let pairs = [(5, 7, 9), (6, 8, 10)];
    let mut arms = Vec::with_capacity(2);
    for (shoulder, elbow, wrist) in pairs {
        let (Some(s), Some(el), Some(w)) = (pixel(e, shoulder), pixel(e, elbow), pixel(e, wrist))
        else {
            continue;
        };
        let proximity = (distance(w, ball) / height).min(segment_distance(ball, el, w) / height);
        if proximity > 0.30 || distance(w, el) / height > 0.85 {
            continue;
        }
        arms.push((s, el, w, proximity));
    }
    if arms.is_empty() {
        return None;
    }
    arms.sort_by(|a, b| a.3.total_cmp(&b.3));
    let (_, elbow, wrist, nearest) = arms[0];
    let shoulders = [shoulder_left, shoulder_right]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if shoulders.is_empty() {
        return None;
    }
    let shoulder_y = shoulders.iter().map(|p| p[1]).sum::<f64>() / shoulders.len() as f64;
    let high = wrist[1] < shoulder_y - height * 0.11;
    let above_head = wrist[1] < (y1 * e.video_height) + height * 0.12;
    let low = wrist[1] > shoulder_y + height * 0.18;
    let both_hands_near =
        arms.len() == 2 && arms[1].3 <= 0.30 && distance(arms[0].2, arms[1].2) / height < 0.35;
    let classification = if high {
        if both_hands_near {
            // Both hands near the ball above the shoulders: typical set.
            ActionKind::OverheadPass
        } else if above_head && elbow[1] > wrist[1] {
            // Elevated, extended one-arm strike, rather than a two-hand set.
            ActionKind::Attack
        } else {
            return None;
        }
    } else if low {
        if wrist[1] > (y1 + 0.93 * (y2 - y1)) * e.video_height {
            return None;
        }
        ActionKind::Dig
    } else if both_hands_near
        || (elbow[1] >= shoulder_y + height * 0.08
            && wrist[1] >= shoulder_y - height * 0.04
            && wrist[1] <= elbow[1] + height * 0.10
            && nearest <= 0.18)
    {
        ActionKind::Dig
    } else {
        return None;
    };
    let wrist_confidence = if arms.len() == 2 {
        e.keypoints[9][2].min(e.keypoints[10][2])
    } else {
        e.keypoints[9][2].max(e.keypoints[10][2])
    };
    let prediction = ActionPrediction {
        kind: classification,
        // Evidence score is an uncalibrated support measure, never a probability.
        evidence_score: (geometry.evidence_score
            * wrist_confidence
            * (1.0 - nearest / 0.6).clamp(0.0, 1.0))
        .clamp(0.0, 1.0),
        method: PredictionMethod::PoseKeypointsV1,
        depth_ratio: geometry.depth_ratio,
        posture_ratio: geometry.posture_ratio,
    };
    prediction.valid().then_some(prediction)
}

pub(crate) fn refine_waist_rise(
    e: &PoseEvidence,
    geometry: &ActionPrediction,
    before: &[[f64; 3]; 17],
    after: &[[f64; 3]; 17],
) -> Option<ActionPrediction> {
    if !geometry.valid()
        || matches!(geometry.kind, ActionKind::Serve | ActionKind::Block)
        || ![e.video_width, e.video_height, e.ball_x_norm, e.ball_y_norm]
            .iter()
            .all(|v| v.is_finite())
        || e.video_width < 64.0
        || e.video_height < 64.0
        || !(0.0..=1.0).contains(&e.ball_x_norm)
        || !(0.0..=1.0).contains(&e.ball_y_norm)
    {
        return None;
    }
    let [x1, y1, x2, y2] = e.player_bbox_norm;
    let height = (y2 - y1) * e.video_height;
    if ![x1, y1, x2, y2].iter().all(|v| (0.0..=1.0).contains(v)) || x1 >= x2 || height < 26.0 {
        return None;
    }
    let start = PoseEvidence {
        keypoints: *before,
        ..*e
    };
    let end = PoseEvidence {
        keypoints: *after,
        ..*e
    };
    let ball = [
        e.ball_x_norm * e.video_width,
        e.ball_y_norm * e.video_height,
    ];
    let extended = |pose: &PoseEvidence, shoulder, elbow, wrist| {
        let s = pixel(pose, shoulder)?;
        let el = pixel(pose, elbow)?;
        let w = pixel(pose, wrist)?;
        let upper = distance(s, el);
        let lower = distance(el, w);
        let length = upper + lower;
        (upper >= height * 0.05
            && lower >= height * 0.05
            && length >= height * 0.16
            && distance(s, w) >= length * 0.94)
            .then_some((s, el, w))
    };
    let support = [(5, 7, 9, 11), (6, 8, 10, 12)]
        .into_iter()
        .filter_map(|(shoulder, elbow, wrist, hip)| {
            let (s0, _, w0) = extended(&start, shoulder, elbow, wrist)?;
            let (s, el, w) = extended(e, shoulder, elbow, wrist)?;
            let (s1, _, w1) = extended(&end, shoulder, elbow, wrist)?;
            let h0 = pixel(&start, hip)?;
            let a = (w0[1] - s0[1]) / height;
            let b = (w[1] - s[1]) / height;
            let c = (w1[1] - s1[1]) / height;
            let proximity = segment_distance(ball, el, w) / height;
            (a >= 0.16
                && (w0[1] - h0[1]).abs() <= height * 0.12
                && (-0.12..=0.35).contains(&b)
                && (-0.20..=0.12).contains(&c)
                && a - b >= 0.035
                && b - c >= 0.035
                && a - c >= 0.18
                && proximity <= 0.24)
                .then_some(e.keypoints[wrist][2] * (1.0 - proximity / 0.6))
        })
        .max_by(f64::total_cmp)?;
    Some(ActionPrediction {
        kind: ActionKind::Dig,
        method: PredictionMethod::PoseKeypointsMotionV1,
        evidence_score: (geometry.evidence_score * support).clamp(0.0, 1.0),
        ..geometry.clone()
    })
}
pub(crate) fn refine_overhead_peak(
    e: &PoseEvidence,
    geometry: &ActionPrediction,
    before: &[[f64; 3]; 17],
    after: &[[f64; 3]; 17],
) -> Option<ActionPrediction> {
    if !geometry.valid() || matches!(geometry.kind, ActionKind::Serve | ActionKind::Block) {
        return None;
    }
    let height = (e.player_bbox_norm[3] - e.player_bbox_norm[1]) * e.video_height;
    let ball = [
        e.ball_x_norm * e.video_width,
        e.ball_y_norm * e.video_height,
    ];
    if !height.is_finite()
        || height < 26.0
        || ball.iter().any(|v| !v.is_finite())
        || !e.video_width.is_finite()
        || !e.video_height.is_finite()
        || e.video_width < 64.0
        || e.video_height < 64.0
    {
        return None;
    }
    let point = |p: &[[f64; 3]; 17], i: usize, min: f64| {
        let [x, y, c] = p[i];
        (p[i]
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            && c >= min)
            .then_some([x * e.video_width, y * e.video_height])
    };
    let cur = &e.keypoints;
    let head = point(cur, 0, 0.28)?;
    let left = point(cur, 9, 0.25)?;
    let right = point(cur, 10, 0.25)?;
    let near = distance(left, ball).min(distance(right, ball)) / height;
    if near > 0.20
        || distance(left, right) > 0.30 * height
        || ball[1] > head[1] + 0.08 * height
        || cur[9][2].max(cur[10][2]) < 0.35
        || (distance(left, ball) > 0.22 * height && cur[10][2] < 0.35)
        || (distance(right, ball) > 0.22 * height && cur[9][2] < 0.35)
    {
        return None;
    }
    let mut movement = 0.0_f64;
    for (s, el, w) in [(5, 7, 9), (6, 8, 10)] {
        let shoulder = point(cur, s, 0.28)?;
        let elbow = point(cur, el, 0.28)?;
        let wrist = point(cur, w, 0.25)?;
        if shoulder[1] - elbow[1] < 0.04 * height
            || shoulder[1] - wrist[1] < 0.12 * height
            || elbow[1] - wrist[1] < 0.035 * height
        {
            return None;
        }
        let relative = (elbow[1] - shoulder[1]) / height;
        let prior = (point(before, el, 0.28)?[1] - point(before, s, 0.28)?[1]) / height;
        let next = (point(after, el, 0.28)?[1] - point(after, s, 0.28)?[1]) / height;
        let change = (prior - relative).abs().max((next - relative).abs());
        if change < 0.035 {
            return None;
        }
        movement = movement.max(change);
    }
    if movement < 0.065 {
        return None;
    }
    Some(ActionPrediction {
        kind: ActionKind::OverheadPass,
        method: PredictionMethod::PoseKeypointsMotionV1,
        evidence_score: (geometry.evidence_score * cur[9][2].min(cur[10][2]) * (1.0 - near))
            .clamp(0.0, 1.0),
        ..geometry.clone()
    })
}

#[must_use]
pub fn refine_with_motion(
    e: &PoseEvidence,
    geometry: &ActionPrediction,
    before: &[[f64; 3]; 17],
    after: &[[f64; 3]; 17],
) -> Option<ActionPrediction> {
    if let Some(prediction) = refine_waist_rise(e, geometry, before, after) {
        return Some(prediction);
    }
    if let Some(prediction) = refine_overhead_peak(e, geometry, before, after) {
        return Some(prediction);
    }
    let prediction = refine(e, geometry)?;
    let height = (e.player_bbox_norm[3] - e.player_bbox_norm[1]) * e.video_height;
    if !height.is_finite() || height < 26.0 {
        return None;
    }
    let relative = |pose: &[[f64; 3]; 17], shoulder: usize, wrist: usize| {
        let s = pose[shoulder];
        let w = pose[wrist];
        if [s[0], s[1], s[2], w[0], w[1], w[2]]
            .iter()
            .any(|value| !value.is_finite())
            || s[2] < 0.35
            || w[2] < 0.35
            || [s[0], s[1], w[0], w[1]]
                .iter()
                .any(|value| !(0.0..=1.0).contains(value))
        {
            return None;
        }
        Some([
            (w[0] - s[0]) * e.video_width / height,
            (w[1] - s[1]) * e.video_height / height,
        ])
    };
    let movement = [(5, 9), (6, 10)]
        .into_iter()
        .filter_map(|(shoulder, wrist)| {
            let a = relative(before, shoulder, wrist)?;
            let b = relative(after, shoulder, wrist)?;
            Some((a[0] - b[0]).hypot(a[1] - b[1]))
        })
        .fold(0.0_f64, f64::max);
    if movement < 0.055 {
        return None;
    }
    let ball = [
        e.ball_x_norm * e.video_width,
        e.ball_y_norm * e.video_height,
    ];
    let rising_forearms = [(5, 7, 9), (6, 8, 10)]
        .into_iter()
        .filter(|&(shoulder, elbow, wrist)| {
            let (Some(start), Some(end), Some(current), Some(s), Some(el), Some(w)) = (
                relative(before, shoulder, wrist),
                relative(after, shoulder, wrist),
                relative(&e.keypoints, shoulder, wrist),
                pixel(e, shoulder),
                pixel(e, elbow),
                pixel(e, wrist),
            ) else {
                return false;
            };
            let arm_length = distance(s, el) + distance(el, w);
            arm_length >= height * 0.16
                && distance(s, w) >= arm_length * 0.95
                && current[0].abs() >= (current[1].abs() * 1.35).max(0.16)
                && current[1] >= -0.24
                && start[1] >= -0.12
                && start[1] - current[1] >= 0.02
                && current[1] - end[1] >= 0.02
                && start[1] - end[1] >= 0.07
                && segment_distance(ball, el, w) <= height * 0.18
        })
        .count();
    let forearm_rise = match prediction.kind {
        ActionKind::Attack => rising_forearms >= 1,
        ActionKind::OverheadPass | ActionKind::Dig => rising_forearms == 2,
        _ => false,
    };
    let overhead_rise = prediction.kind == ActionKind::Dig
        && [(5, 7, 9), (6, 8, 10)]
            .into_iter()
            .all(|(shoulder, elbow, wrist)| {
                let (Some(start), Some(end), Some(current), Some(elbow), Some(wrist)) = (
                    relative(before, shoulder, wrist),
                    relative(after, shoulder, wrist),
                    relative(&e.keypoints, shoulder, wrist),
                    pixel(e, elbow),
                    pixel(e, wrist),
                ) else {
                    return false;
                };
                start[1] >= -0.07
                    && end[1] <= -0.12
                    && start[1] - end[1] >= 0.12
                    && current[1] <= 0.04
                    && elbow[1] - wrist[1] >= height * 0.04
                    && distance(wrist, ball) <= height * 0.22
            });
    Some(ActionPrediction {
        kind: if forearm_rise {
            ActionKind::Dig
        } else if overhead_rise {
            ActionKind::OverheadPass
        } else {
            prediction.kind
        },
        method: PredictionMethod::PoseKeypointsMotionV1,
        ..prediction
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bilateral_elbow_lift_accepts_one_partly_uncertain_wrist_without_calling_attack() {
        let mut current = [[0.0; 3]; 17];
        current[0] = [0.50, 0.42, 0.7];
        current[5] = [0.49, 0.45, 0.7];
        current[6] = [0.51, 0.45, 0.7];
        current[7] = [0.48, 0.42, 0.4];
        current[8] = [0.52, 0.42, 0.31];
        current[9] = [0.498, 0.39, 0.5];
        current[10] = [0.502, 0.39, 0.27];
        let mut before = current;
        let mut after = current;
        before[7][1] = 0.46;
        before[8][1] = 0.46;
        after[7][1] = 0.46;
        after[8][1] = 0.46;
        let mut evidence = PoseEvidence {
            keypoints: current,
            ball_x_norm: 0.5,
            ball_y_norm: 0.39,
            video_width: 1000.0,
            video_height: 1000.0,
            player_bbox_norm: [0.46, 0.37, 0.54, 0.57],
        };
        assert_eq!(
            refine_overhead_peak(&evidence, &geom(), &before, &after)
                .unwrap()
                .kind,
            ActionKind::OverheadPass
        );
        evidence.keypoints[8][1] = 0.47;
        assert!(refine_overhead_peak(&evidence, &geom(), &before, &after).is_none());
    }

    fn geom() -> ActionPrediction {
        ActionPrediction {
            kind: ActionKind::Attack,
            evidence_score: 0.85,
            method: PredictionMethod::ContactGeometryV1,
            depth_ratio: -0.1,
            posture_ratio: 0.4,
        }
    }
    fn data() -> PoseEvidence {
        let mut p = [[0.5, 0.5, 0.0]; 17];
        // A standing subject with extended arms near the overhead ball.
        p[5] = [0.43, 0.44, 0.95];
        p[6] = [0.57, 0.44, 0.95];
        p[7] = [0.41, 0.31, 0.96];
        p[8] = [0.59, 0.31, 0.96];
        p[9] = [0.49, 0.22, 0.96];
        p[10] = [0.52, 0.22, 0.96];
        PoseEvidence {
            keypoints: p,
            ball_x_norm: 0.5,
            ball_y_norm: 0.20,
            video_width: 640.0,
            video_height: 360.0,
            player_bbox_norm: [0.35, 0.17, 0.65, 0.90],
        }
    }
    #[test]
    fn contact_distance_requires_confident_nearby_body_evidence() {
        let mut evidence = data();
        assert!(contact_distance(&evidence, 0.35).unwrap() < 0.1);
        evidence.ball_x_norm = 0.95;
        assert!(contact_distance(&evidence, 0.35).unwrap() > 0.3);
        for point in &mut evidence.keypoints {
            point[2] = 0.0;
        }
        assert!(contact_distance(&evidence, 0.35).is_none());
    }

    #[test]
    fn distinguishes_two_hand_set_from_single_arm_attack() {
        let mut d = data();
        assert_eq!(refine(&d, &geom()).unwrap().kind, ActionKind::OverheadPass);
        d.keypoints[10][2] = 0.0;
        assert_eq!(refine(&d, &geom()).unwrap().kind, ActionKind::Attack);
    }
    #[test]
    fn missing_or_distant_wrist_never_overrides_geometry() {
        let mut d = data();
        d.keypoints[9][2] = 0.0;
        d.keypoints[10][2] = 0.0;
        assert!(refine(&d, &geom()).is_none());
        d = data();
        d.ball_x_norm = 0.97;
        assert!(refine(&d, &geom()).is_none());
    }

    #[test]
    fn motion_is_required_for_named_technique() {
        let e = data();
        let no_motion = e.keypoints;
        assert!(refine_with_motion(&e, &geom(), &no_motion, &no_motion).is_none());
        let mut before = e.keypoints;
        let mut after = e.keypoints;
        before[9][1] = 0.31;
        before[10][1] = 0.31;
        after[9][1] = 0.18;
        after[10][1] = 0.18;
        let predicted = refine_with_motion(&e, &geom(), &before, &after).unwrap();
        assert_eq!(predicted.kind, ActionKind::OverheadPass);
        assert_eq!(predicted.method, PredictionMethod::PoseKeypointsMotionV1);
        let whole_body = before;
        let mut whole_body_after = after;
        for joint in [5, 6, 7, 8, 9, 10] {
            whole_body_after[joint][1] = whole_body[joint][1] + 0.13;
        }
        assert!(refine_with_motion(&e, &geom(), &whole_body, &whole_body_after).is_none());
    }

    #[test]
    fn crouched_forearm_receive_with_one_visible_wrist_is_dig() {
        let mut e = data();
        e.player_bbox_norm = [0.35, 0.17, 0.65, 0.90];
        e.keypoints[5] = [0.43, 0.48, 0.95];
        e.keypoints[6] = [0.57, 0.48, 0.95];
        e.keypoints[7] = [0.45, 0.61, 0.95];
        e.keypoints[9] = [0.50, 0.57, 0.95];
        e.keypoints[10][2] = 0.0;
        e.ball_x_norm = 0.49;
        e.ball_y_norm = 0.57;
        let mut before = e.keypoints;
        let mut after = e.keypoints;
        before[9][1] = 0.62;
        after[9][1] = 0.52;
        let predicted = refine_with_motion(&e, &geom(), &before, &after).unwrap();
        assert_eq!(predicted.kind, ActionKind::Dig);
        assert_eq!(predicted.method, PredictionMethod::PoseKeypointsMotionV1);
    }

    #[test]
    fn forearm_receive_uses_contact_segment_not_only_wrist_location() {
        let mut e = data();
        e.player_bbox_norm = [0.20, 0.10, 0.92, 0.95];
        e.keypoints[5] = [0.43, 0.45, 0.95];
        e.keypoints[6] = [0.56, 0.45, 0.95];
        e.keypoints[7] = [0.50, 0.63, 0.95];
        e.keypoints[8] = [0.54, 0.64, 0.95];
        e.keypoints[9] = [0.80, 0.56, 0.95];
        e.keypoints[10] = [0.81, 0.57, 0.95];
        e.ball_x_norm = 0.64;
        e.ball_y_norm = 0.60;
        assert_eq!(refine(&e, &geom()).unwrap().kind, ActionKind::Dig);
    }

    #[test]
    fn shoulder_level_contact_with_two_rising_wrists_is_overhead_pass() {
        let mut e = data();
        e.keypoints[7] = [0.41, 0.54, 0.96];
        e.keypoints[8] = [0.59, 0.54, 0.96];
        e.keypoints[9] = [0.49, 0.42, 0.96];
        e.keypoints[10] = [0.52, 0.42, 0.96];
        e.ball_y_norm = 0.40;
        let mut before = e.keypoints;
        let mut after = e.keypoints;
        before[9][1] = 0.46;
        before[10][1] = 0.46;
        after[9][1] = 0.28;
        after[10][1] = 0.28;
        assert_eq!(
            refine_with_motion(&e, &geom(), &before, &after)
                .unwrap()
                .kind,
            ActionKind::OverheadPass
        );
        assert_ne!(
            refine_with_motion(&e, &geom(), &after, &before)
                .unwrap()
                .kind,
            ActionKind::OverheadPass
        );
        after[10][2] = 0.0;
        assert_ne!(
            refine_with_motion(&e, &geom(), &before, &after)
                .unwrap()
                .kind,
            ActionKind::OverheadPass
        );
    }

    #[test]
    fn rising_extended_forearm_of_crouched_player_is_dig_not_attack() {
        let mut e = data();
        e.player_bbox_norm = [0.24, 0.40, 0.76, 0.95];
        e.keypoints[5] = [0.60, 0.53, 0.92];
        e.keypoints[6] = [0.66, 0.53, 0.91];
        e.keypoints[7] = [0.52, 0.49, 0.94];
        e.keypoints[9] = [0.44, 0.46, 0.95];
        e.keypoints[8][2] = 0.0;
        e.keypoints[10][2] = 0.0;
        e.keypoints[0] = [0.62, 0.44, 0.95];
        e.ball_x_norm = 0.43;
        e.ball_y_norm = 0.45;
        let mut before = e.keypoints;
        let mut after = e.keypoints;
        before[7][1] = 0.55;
        before[9][1] = 0.57;
        after[7][1] = 0.47;
        after[9][1] = 0.41;
        assert_eq!(refine(&e, &geom()).unwrap().kind, ActionKind::Attack);
        assert_eq!(
            refine_with_motion(&e, &geom(), &before, &after)
                .unwrap()
                .kind,
            ActionKind::Dig
        );
        assert_ne!(
            refine_with_motion(&e, &geom(), &after, &before).map(|p| p.kind),
            Some(ActionKind::Dig)
        );
    }

    #[test]
    fn overhead_single_arm_swing_remains_attack() {
        let mut e = data();
        e.keypoints[10][2] = 0.0;
        e.keypoints[0] = [0.50, 0.33, 0.95];
        let mut before = e.keypoints;
        let mut after = e.keypoints;
        before[9][1] = 0.16;
        after[9][1] = 0.32;
        assert_eq!(
            refine_with_motion(&e, &geom(), &before, &after)
                .unwrap()
                .kind,
            ActionKind::Attack
        );
    }

    #[test]
    fn waist_to_shoulder_extended_receive_is_dig_even_with_one_reliable_arm() {
        let mut e = data();
        e.player_bbox_norm = [0.30, 0.20, 0.70, 0.80];
        e.keypoints[5] = [0.49, 0.35, 0.8];
        e.keypoints[6] = [0.55, 0.35, 0.8];
        e.keypoints[8] = [0.51, 0.39, 0.8];
        e.keypoints[10] = [0.47, 0.43, 0.8];
        e.keypoints[7][2] = 0.1;
        e.keypoints[9][2] = 0.1;
        e.keypoints[12] = [0.55, 0.54, 0.8];
        e.ball_x_norm = 0.47;
        e.ball_y_norm = 0.33;
        let mut before = e.keypoints;
        before[8] = [0.53, 0.45, 0.8];
        before[10] = [0.51, 0.54, 0.8];
        let mut after = e.keypoints;
        after[8] = [0.50, 0.34, 0.8];
        after[10] = [0.45, 0.33, 0.8];
        assert_eq!(
            refine_with_motion(&e, &geom(), &before, &after).map(|p| p.kind),
            Some(ActionKind::Dig)
        );
        assert_ne!(
            refine_with_motion(&e, &geom(), &after, &before).map(|p| p.kind),
            Some(ActionKind::Dig)
        );
        use crate::touch_pipeline::{refine_observed_pose_motion, PoseFrame, PosePlayer};
        let row = |frame_index, keypoints, track_id, interpolated| PoseFrame {
            frame_index,
            players: vec![PosePlayer {
                track_id,
                keypoints,
                interpolated,
            }],
        };
        let poses = [
            row(6, before, 3, false),
            row(9, e.keypoints, 3, false),
            row(10, e.keypoints, 3, false),
            row(11, e.keypoints, 3, false),
            row(14, after, 3, false),
        ];
        assert_eq!(
            refine_observed_pose_motion(&poses, 10, 3, &e, &geom()).map(|p| p.kind),
            Some(ActionKind::Dig)
        );
        let mut wrong_track = poses.clone();
        wrong_track[0].players[0].track_id = 4;
        assert_ne!(
            refine_observed_pose_motion(&wrong_track, 10, 3, &e, &geom()).map(|p| p.kind),
            Some(ActionKind::Dig)
        );
        let mut synthetic = poses;
        synthetic[4].players[0].interpolated = true;
        assert_ne!(
            refine_observed_pose_motion(&synthetic, 10, 3, &e, &geom()).map(|p| p.kind),
            Some(ActionKind::Dig)
        );

        let mut bent = e;
        bent.keypoints[8] = [0.55, 0.33, 0.8];
        assert_ne!(
            refine_with_motion(&bent, &geom(), &before, &after).map(|p| p.kind),
            Some(ActionKind::Dig)
        );
    }

    #[test]
    fn low_wrist_infers_dig() {
        let mut d = data();
        d.keypoints[7] = [0.42, 0.52, 0.9];
        d.keypoints[9] = [0.47, 0.67, 0.9];
        d.keypoints[10][2] = 0.0;
        d.ball_x_norm = 0.47;
        d.ball_y_norm = 0.67;
        assert_eq!(refine(&d, &geom()).unwrap().kind, ActionKind::Dig);
    }
}
