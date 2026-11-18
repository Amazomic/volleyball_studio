//! CPU-only VBallNet trajectory input and strict, provisional neural peaks.
//! An ONNX score is not a confirmed athlete touch or an autonomous rally event.
use super::{PhysicalTouchCandidate, TouchFrame, TouchSettings};

fn point(frame: &TouchFrame) -> Option<[f64; 2]> {
    let ball = frame.selected_ball.as_ref()?;
    if ball.source_model != "vballnet"
        || !ball
            .bbox_xyxy_norm
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        || ball.bbox_xyxy_norm[2] <= ball.bbox_xyxy_norm[0]
        || ball.bbox_xyxy_norm[3] <= ball.bbox_xyxy_norm[1]
    {
        return None;
    }
    Some([
        (ball.bbox_xyxy_norm[0] + ball.bbox_xyxy_norm[2]) * 0.5,
        (ball.bbox_xyxy_norm[1] + ball.bbox_xyxy_norm[3]) * 0.5,
    ])
}

/// Paper input: image-width/height-normalized (X,Y,dX,dY).
/// Gaps are explicitly zero-masked; no interpolation becomes GPU evidence.
pub fn trajectory_features(frames: &[TouchFrame]) -> Vec<[f32; 4]> {
    let mut result = Vec::with_capacity(frames.len());
    let mut previous: Option<[f64; 2]> = None;
    for frame in frames {
        let current = point(frame);
        let feature = match current {
            Some([x, y]) => {
                let (dx, dy) = match previous {
                    Some([px, py]) => (x - px, y - py),
                    None => (0.0, 0.0),
                };
                [x as f32, y as f32, dx as f32, dy as f32]
            }
            None => [0.0; 4],
        };
        result.push(feature);
        previous = current;
    }
    result
}

/// A learned deflection is an *unverified candidate*. Require observed ball
/// on impact and both adjacent flights, physical velocity discontinuity, and
/// a reachable same-frame provisional track. Pose confirmation happens later.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_tcn_peaks(
    frames: &[TouchFrame],
    scores: &[f32],
    width: u32,
    height: u32,
    emit_start: u64,
    emit_end: u64,
    settings: TouchSettings,
) -> Result<Vec<PhysicalTouchCandidate>, String> {
    settings.validate()?;
    if !(64..=16384).contains(&width)
        || !(64..=16384).contains(&height)
        || frames.len() > 1200
        || frames.len() != scores.len()
        || frames
            .windows(2)
            .any(|p| p[0].frame_index + 1 != p[1].frame_index)
        || scores
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err("Invalid TCN frame/score evidence".into());
    }
    let cutoff = (settings.threshold as f32).max(0.35);
    let mut ranked = (2..frames.len().saturating_sub(2))
        .filter(|&i| {
            scores[i] >= cutoff && scores[i] >= scores[i - 1] && scores[i] >= scores[i + 1]
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
    let scale = f64::from(width.max(height)) / 1280.0;
    let mut output: Vec<PhysicalTouchCandidate> = Vec::new();
    let gap = settings.min_gap.max(2) as u64;
    for i in ranked {
        let frame = &frames[i];
        if frame.frame_index < emit_start
            || frame.frame_index >= emit_end
            || output
                .iter()
                .any(|p| p.frame_index.abs_diff(frame.frame_index) <= gap)
        {
            continue;
        }
        let (Some(before), Some(contact), Some(after)) =
            (point(&frames[i - 2]), point(frame), point(&frames[i + 2]))
        else {
            continue;
        };
        let w = f64::from(width);
        let h = f64::from(height);
        let x = contact[0] * w;
        let y = contact[1] * h;
        let v1 = [
            (contact[0] - before[0]) * w / 2.0,
            (contact[1] - before[1]) * h / 2.0,
        ];
        let v2 = [
            (after[0] - contact[0]) * w / 2.0,
            (after[1] - contact[1]) * h / 2.0,
        ];
        let s1 = v1[0].hypot(v1[1]);
        let s2 = v2[0].hypot(v2[1]);
        if s1 < 1.0 * scale || s2 < 1.0 * scale {
            continue;
        }
        let cosine = ((v1[0] * v2[0] + v1[1] * v2[1]) / (s1 * s2)).clamp(-1.0, 1.0);
        let angle = cosine.acos().to_degrees();
        let residual = (v2[0] - v1[0]).hypot(v2[1] - v1[1]);
        // Gravity and tracking jitter should not become athlete touches.
        if angle < 14.0 && residual < 4.0 * scale {
            continue;
        }
        let actor = frame
            .players
            .iter()
            .filter(|p| {
                p.track_id > 0
                    && p.score.is_finite()
                    && p.score > 0.0
                    && p.bbox_xyxy_norm.iter().all(|v| v.is_finite())
                    && p.bbox_xyxy_norm[2] > p.bbox_xyxy_norm[0]
                    && p.bbox_xyxy_norm[3] > p.bbox_xyxy_norm[1]
            })
            .filter_map(|p| {
                let b = p.bbox_xyxy_norm;
                let dx = (b[0] * w - x).max(x - b[2] * w).max(0.0);
                let dy = (b[1] * h - y).max(y - b[3] * h).max(0.0);
                let distance = dx.hypot(dy);
                let actor_height = (b[3] - b[1]) * h;
                (distance <= actor_height * settings.max_reach_ratio).then_some((
                    p.track_id,
                    distance / actor_height.max(1.0),
                    p.score,
                ))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.2.total_cmp(&a.2)));
        let Some((track_id, _, _)) = actor else {
            continue;
        };
        output.push(PhysicalTouchCandidate {
            frame_index: frame.frame_index,
            provisional_track_id: Some(track_id),
            probability: f64::from(scores[i]).min(0.95),
            ball_x_px: x,
            ball_y_px: y,
            motion_frame_index: frame.frame_index,
            direction_change_deg: angle,
            speed_before_px_frame: s1,
            speed_after_px_frame: s2,
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
        });
    }
    output.sort_by_key(|p| p.frame_index);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::touch_pipeline::{TouchBall, TouchPlayer};

    fn frames(bounce: bool) -> Vec<TouchFrame> {
        (0..31)
            .map(|i| {
                let x = if bounce && i > 15 {
                    0.25 - (i - 15) as f64 * 0.008
                } else {
                    0.1 + i as f64 * 0.01
                };
                TouchFrame {
                    frame_index: i as u64,
                    selected_ball: Some(TouchBall {
                        source_model: "vballnet".to_owned(),
                        score: 0.95,
                        bbox_xyxy_norm: [x - 0.005, 0.48, x + 0.005, 0.50],
                    }),
                    players: vec![TouchPlayer {
                        track_id: 9,
                        score: 0.9,
                        touch_probability: None,
                        bbox_xyxy_norm: [0.12, 0.2, 0.32, 0.95],
                    }],
                }
            })
            .collect()
    }
    #[test]
    fn trajectory_features_are_normalized_and_first_order() {
        let mut f = frames(false);
        f[3].selected_ball = None;
        let v = trajectory_features(&f);
        assert!((v[1][2] - 0.01).abs() < 0.00001);
        assert_eq!(v[3], [0.0; 4]);
        assert_eq!(v[4][2], 0.0);
    }
    #[test]
    fn trained_scores_still_require_a_real_motion_break() {
        let scores = (0..31)
            .map(|i| if i == 15 { 0.92 } else { 0.01 })
            .collect::<Vec<_>>();
        let settings = TouchSettings::default();
        let hits = evaluate_tcn_peaks(&frames(true), &scores, 1280, 720, 0, 31, settings).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].frame_index, 15);
        assert_eq!(hits[0].provisional_track_id, Some(9));
        assert!(
            evaluate_tcn_peaks(&frames(false), &scores, 1280, 720, 0, 31, settings)
                .unwrap()
                .is_empty()
        );
    }
    #[test]
    fn no_actor_or_no_observed_ball_means_no_touch() {
        let scores = (0..31)
            .map(|i| if i == 15 { 0.92 } else { 0.01 })
            .collect::<Vec<_>>();
        let mut f = frames(true);
        f[15].players.clear();
        assert!(
            evaluate_tcn_peaks(&f, &scores, 1280, 720, 0, 31, TouchSettings::default())
                .unwrap()
                .is_empty()
        );
        f = frames(true);
        f[15].selected_ball = None;
        assert!(
            evaluate_tcn_peaks(&f, &scores, 1280, 720, 0, 31, TouchSettings::default())
                .unwrap()
                .is_empty()
        );
    }
}
