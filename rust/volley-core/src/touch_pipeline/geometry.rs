//! Independent geometric touch proposals: ball-trajectory breaks near an observed player.
//! Unlike confidence-only touch peaks, this path can recover contacts without neural Touch logits.
//! It is deliberately conservative: a ball flight near a person alone is not a Touch.

use super::{center, pixel_bbox, standing_height, TouchFrame};
use crate::motion::{BallObservation, MotionEvidence};

#[derive(Clone, Copy, Debug)]
pub(super) struct GeometryContact {
    pub frame: usize,
    pub track_id: u64,
    pub probability: f64,
    pub motion: MotionEvidence,
    pub ball_x: f64,
    pub ball_y: f64,
}

/// Do not interpret a switch to an unsupported ball source as an impact.
pub(super) fn stable_ball_source(frames: &[TouchFrame], index: usize) -> bool {
    stable_ball_source_with_min_samples(frames, index, 3)
}

/// Pose-confirmed contacts can tolerate a few missing VballNet frames: two
/// observations per side still give the motion fitter independent samples.
/// Legacy geometry keeps the stricter three-per-side gate.
pub(super) fn stable_ball_source_with_min_samples(
    frames: &[TouchFrame],
    index: usize,
    min_samples: usize,
) -> bool {
    if index < 5 || index + 5 >= frames.len() || !(2..=3).contains(&min_samples) {
        return false;
    }
    let source = match frames[index].selected_ball.as_ref() {
        Some(ball) => ball.source_model.as_str(),
        None => return false,
    };
    let mut left = 0;
    let mut right = 0;
    for (relative, frame) in frames[index - 5..=index + 5].iter().enumerate() {
        if let Some(ball) = frame.selected_ball.as_ref() {
            if ball.source_model != source {
                return false;
            }
            if relative < 5 {
                left += 1;
            } else if relative > 5 {
                right += 1;
            }
        }
    }
    left >= min_samples && right >= min_samples
}

/// Pose-confirmed contact may span a short missing-ball island. Two actual
/// observations on each side within nine frames remain mandatory.
pub(super) fn stable_pose_ball_source(frames: &[TouchFrame], index: usize) -> bool {
    const WINDOW: usize = 9;
    if index < WINDOW || index + WINDOW >= frames.len() {
        return false;
    }
    let Some(source) = frames[index].selected_ball.as_ref() else {
        return false;
    };
    let mut before = 0;
    let mut after = 0;
    for (relative, row) in frames[index - WINDOW..=index + WINDOW].iter().enumerate() {
        if let Some(ball) = &row.selected_ball {
            if ball.source_model != source.source_model {
                return false;
            }
            if relative < WINDOW {
                before += 1;
            } else if relative > WINDOW {
                after += 1;
            }
        }
    }
    before >= 2 && after >= 2
}

/// A single wrong ball localization can create two huge fictitious velocity changes.
fn single_frame_outlier(frames: &[TouchFrame], index: usize, width: f64, height: f64) -> bool {
    if index < 2 || index + 2 >= frames.len() {
        return false;
    }
    let center = |i: usize| -> Option<(f64, f64)> {
        let box_ = frames[i].selected_ball.as_ref()?.bbox_xyxy_norm;
        Some((
            (box_[0] + box_[2]) * width * 0.5,
            (box_[1] + box_[3]) * height * 0.5,
        ))
    };
    let Some((ax, ay)) = center(index - 2) else {
        return false;
    };
    let Some((bx, by)) = center(index - 1) else {
        return false;
    };
    let Some((cx, cy)) = center(index) else {
        return false;
    };
    let Some((dx, dy)) = center(index + 1) else {
        return false;
    };
    let Some((ex, ey)) = center(index + 2) else {
        return false;
    };
    let normal_step = (ax - bx).hypot(ay - by).max((dx - ex).hypot(dy - ey));
    let expected = (bx + dx) * 0.5;
    let expected_y = (by + dy) * 0.5;
    (cx - expected).hypot(cy - expected_y) > 16.0_f64.max(4.0 * normal_step)
}

/// The motion fitter sees up to five observations on either side of an impact.
/// A single bad localization must not become a nearby false geometric Touch.
pub(super) fn motion_window_has_outlier(
    frames: &[TouchFrame],
    index: usize,
    width: f64,
    height: f64,
) -> bool {
    (index.saturating_sub(6)..=(index + 6).min(frames.len() - 1))
        .any(|neighbor| single_frame_outlier(frames, neighbor, width, height))
}

/// Physical footprint of one observed ball, in a single CPU source frame.
#[derive(Clone, Copy, Debug)]
struct ReachEvidence {
    width: f64,
    height: f64,
    ball_x: f64,
    ball_y: f64,
    ball_radius: f64,
    max_reach_ratio: f64,
    /// Only a measured strong direction reversal can expand overhead reach
    /// beyond the detector's truncated body box.
    strong_reversal: bool,
}

/// Returns the most plausible SAME-FRAME player; no invented or interpolated Track IDs.
fn touching_player(
    frames: &[TouchFrame],
    index: usize,
    reach_evidence: ReachEvidence,
) -> Option<(u64, f64, f64)> {
    let ReachEvidence {
        width,
        height,
        ball_x,
        ball_y,
        ball_radius,
        max_reach_ratio,
        strong_reversal,
    } = reach_evidence;
    let mut best: Option<(u64, f64, f64)> = None;
    for player in &frames[index].players {
        if player.score < 0.25 {
            continue;
        }
        let [x1, y1, x2, y2] = pixel_bbox(player.bbox_xyxy_norm, width, height);
        let bbox_height = y2 - y1;
        if bbox_height < 14.0 || x2 - x1 < 6.0 {
            continue;
        }
        let Some(standing) = standing_height(frames, index, player.track_id, height) else {
            continue;
        };
        let standing = standing.max(bbox_height);
        let reach_x = ((x2 - x1) * 0.70).max(standing * 0.50) + ball_radius;
        // Far-court detector boxes often truncate raised arms. Allow extra
        // overhead reach only when independent ball motion shows a sharp
        // reversal; mere proximity or acceleration cannot expand the zone.
        let reach_top = y2 - standing * if strong_reversal { 1.70 } else { 1.45 } - ball_radius;
        let reach_bottom = y2 + standing * 0.15 + ball_radius;
        let reach_mid = (reach_top + reach_bottom) * 0.5;
        let reach_y = (reach_bottom - reach_top) * 0.5;
        let reach =
            ((ball_x - (x1 + x2) * 0.5) / reach_x).hypot((ball_y - reach_mid) / reach_y.max(1.0));
        // A geometry-only proposal must be closer than the broader distance gate.
        let reach_limit = (0.55 + max_reach_ratio * 0.65).clamp(0.58, 1.0);
        if reach > reach_limit {
            continue;
        }
        let body_x = (ball_x - (x1 + x2) * 0.5).abs() - (x2 - x1) * 0.5;
        let body_y = (y1.min(y2 - standing) - ball_y).max(ball_y - y2);
        let body = body_x.max(0.0).hypot(body_y.max(0.0)) / standing;
        let ratio = ball_radius / standing;
        if ratio <= 0.0 {
            continue;
        }
        let mismatch = if ratio < 0.05675 * 0.75 {
            (ratio / (0.05675 * 0.75)).log2().abs()
        } else if ratio > 0.05675 * 1.25 {
            (ratio / (0.05675 * 1.25)).log2().abs()
        } else {
            0.0
        };
        let cost = body + 0.25 * reach + 1.25 * mismatch;
        if best.is_none_or(|(_, old_cost, _)| cost < old_cost) {
            best = Some((player.track_id, cost, player.score));
        }
    }
    best
}

/// A ball that decelerates to rest is not a player contact merely because a
/// tracked athlete is nearby. Require repeated *observed* VBallNet positions
/// that remain in a tight cluster after the proposed impact. The body-contact
/// path remains independent and can still confirm a genuine interception.
fn settled_ball_after(frames: &[TouchFrame], index: usize, width: f64, height: f64) -> bool {
    const MIN_SETTLED_SAMPLES: usize = 4;
    const MAX_SETTLED_SPAN_FRAMES: usize = 6;
    const MAX_CLUSTER_RADIUS_PX: f64 = 35.0;
    let Some(origin) = frames.get(index).and_then(|row| row.selected_ball.as_ref()) else {
        return false;
    };
    if origin.source_model != "vballnet" {
        return false;
    }
    let point = center(origin.bbox_xyxy_norm);
    let anchor = [point[0] * width, point[1] * height];
    let mut samples = 0;
    let mut far = 0;
    for row in frames.iter().skip(index + 1).take(MAX_SETTLED_SPAN_FRAMES) {
        let Some(ball) = row
            .selected_ball
            .as_ref()
            .filter(|b| b.source_model == "vballnet")
        else {
            continue;
        };
        let point = center(ball.bbox_xyxy_norm);
        let distance = (point[0] * width - anchor[0]).hypot(point[1] * height - anchor[1]);
        if distance > MAX_CLUSTER_RADIUS_PX {
            far += 1;
        } else {
            samples += 1;
        }
    }
    samples >= MIN_SETTLED_SAMPLES && far == 0
}
fn strong_forward_speed_impulse(motion: MotionEvidence) -> bool {
    motion.angle < 25.0
        && motion.speed_before >= 2.0
        && motion.speed_after >= 18.0
        && motion.speed_after >= 2.5 * motion.speed_before
        && motion.speed_after - motion.speed_before >= 12.0
        && motion.residual >= 12.0
}

/// Scan the bounded chunk. Unobserved ball/player frames never become proposals.
pub(super) fn detect(
    frames: &[TouchFrame],
    balls: &[BallObservation],
    width: f64,
    height: f64,
    max_reach_ratio: f64,
) -> Vec<GeometryContact> {
    if balls.len() < 7 {
        return Vec::new();
    }
    let mut result = Vec::new();
    for change in crate::motion::trajectory_breaks(balls) {
        let Ok(index) = usize::try_from(change.frame) else {
            continue;
        };
        let Some(frame) = frames.get(index) else {
            continue;
        };
        if !stable_ball_source(frames, index) {
            continue;
        }
        let Some(ball) = frame
            .selected_ball
            .as_ref()
            .filter(|ball| ball.source_model == "vballnet")
        else {
            continue;
        };
        let motion = change.motion;
        let speed = motion.speed_before.max(motion.speed_after);
        // An observed ball settling on the ground is an endpoint, not a
        // second player contact. Never discard a real body-confirmed hit here.
        if motion.speed_after < 5.0
            && motion.speed_after < motion.speed_before * 0.40
            && settled_ball_after(frames, index, width, height)
        {
            continue;
        }
        let acceleration_impulse = strong_forward_speed_impulse(motion);
        if speed < 2.0
            || motion.residual < 8.0_f64.max(0.45 * speed)
            || (motion.angle < 5.0 && !acceleration_impulse)
            || (acceleration_impulse && motion_window_has_outlier(frames, index, width, height))
        {
            continue;
        }
        let box_ = ball.bbox_xyxy_norm;
        let radius = ((box_[2] - box_[0]) * width + (box_[3] - box_[1]) * height) * 0.25;
        let x = (box_[0] + box_[2]) * 0.5 * width;
        let y = (box_[1] + box_[3]) * 0.5 * height;
        let Some((track_id, cost, player_score)) = touching_player(
            frames,
            index,
            ReachEvidence {
                width,
                height,
                ball_x: x,
                ball_y: y,
                ball_radius: radius,
                max_reach_ratio,
                strong_reversal: motion.angle >= 120.0,
            },
        ) else {
            continue;
        };
        // An evidence score, NOT a calibrated P(touch) or P(action-kind).
        let probability = (0.30
            + 0.23 * motion.residual / (motion.residual + 12.0)
            + 0.16 * (1.0 - cost.min(1.0))
            + 0.11 * ball.score
            + 0.10 * player_score)
            .clamp(0.0, 0.85);
        result.push(GeometryContact {
            frame: index,
            track_id,
            probability,
            motion,
            ball_x: x,
            ball_y: y,
        });
    }
    result
}

/// Recover a sharp vertical flight reversal from consecutive *real* VBallNet
/// samples when a moving body overlaps the ball near its upper half.
/// The legacy motion fitter can treat the exact reversal as a localization
/// outlier and miss it; this independent witness only issues an UNCONFIRMED
/// proposal. A ground landing without an outgoing upward flight cannot pass.
pub(super) fn detect_observed_vertical_rebounds(
    frames: &[TouchFrame],
    width: f64,
    height: f64,
) -> Vec<GeometryContact> {
    if frames.len() < 9 || width < 64.0 || height < 64.0 {
        return Vec::new();
    }
    let observed = |index: usize| -> Option<[f64; 2]> {
        let ball = frames.get(index)?.selected_ball.as_ref()?;
        if ball.source_model != "vballnet" || ball.score < 0.35 {
            return None;
        }
        let p = center(ball.bbox_xyxy_norm);
        Some([p[0] * width, p[1] * height])
    };
    let mut proposals = Vec::new();
    for index in 3..frames.len() - 4 {
        let (Some(a), Some(b), Some(c), Some(d)) = (
            observed(index - 2),
            observed(index),
            observed(index + 1),
            observed(index + 3),
        ) else {
            continue;
        };
        let incoming = [(b[0] - a[0]) * 0.5, (b[1] - a[1]) * 0.5];
        let outgoing = [(d[0] - c[0]) * 0.5, (d[1] - c[1]) * 0.5];
        let speed_in = incoming[0].hypot(incoming[1]);
        let speed_out = outgoing[0].hypot(outgoing[1]);
        // A comparatively slow inbound ball can still be sharply attacked.
        // Require a decisive outgoing vertical flight; weak floor bounces
        // remain ineligible.
        if incoming[1] < 8.0
            || outgoing[1] > -13.0
            || speed_in > 60.0
            || speed_out > 60.0
            || speed_in < 8.0
            || speed_out < 13.0
        {
            continue;
        }
        let angle = ((incoming[0] * outgoing[0] + incoming[1] * outgoing[1])
            / (speed_in * speed_out))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        if angle < 115.0 || settled_ball_after(frames, index, width, height) {
            continue;
        }
        // The ball must pass through the upper portion of a persistent
        // observed player, rather than align with someone's feet as it
        // rebounds off the floor. No inferred/absent player is invented.
        let mut best: Option<(f64, u64, f64)> = None;
        for player in &frames[index].players {
            if player.score < 0.40 {
                continue;
            }
            let [x1, y1, x2, y2] = pixel_bbox(player.bbox_xyxy_norm, width, height);
            let ph = y2 - y1;
            let pw = x2 - x1;
            if ph < 35.0
                || pw < 12.0
                || b[0] < x1 - 0.18 * pw
                || b[0] > x2 + 0.18 * pw
                || b[1] < y1 - 0.16 * ph
                || b[1] > y1 + 0.46 * ph
            {
                continue;
            }
            let count = frames[index - 2..=index + 2]
                .iter()
                .filter(|row| row.players.iter().any(|p| p.track_id == player.track_id))
                .count();
            if count < 3 {
                continue;
            }
            let cost = (b[0] - (x1 + x2) * 0.5).abs() / pw + (b[1] - y1).abs() / ph;
            if best.is_none_or(|old| cost < old.0) {
                best = Some((cost, player.track_id, player.score));
            }
        }
        let Some((_, track_id, player_score)) = best else {
            continue;
        };
        let residual = (outgoing[0] - incoming[0]).hypot(outgoing[1] - incoming[1]);
        proposals.push(GeometryContact {
            frame: index,
            track_id,
            probability: (0.50 + 0.16 * angle / 180.0 + 0.10 * player_score).min(0.79),
            motion: MotionEvidence {
                motion_frame: index as i64,
                speed_before: speed_in,
                speed_after: speed_out,
                angle,
                residual,
            },
            ball_x: b[0],
            ball_y: b[1],
        });
    }
    proposals
}

/// A VBallNet localization island may hide an actual impact between the last
/// good observation and the first good one after it. Fit both *observed* flight
/// segments independently; only a strong velocity reversal whose extrapolated
/// paths intersect near the same player becomes a contact proposal. The event
/// is anchored to the last observed VBallNet frame, not a fabricated ball box.
/// Recover an *unverified* strike when a brief player occlusion accompanies a
/// sharp ball-flight discontinuity. The candidate is anchored to a genuine
/// earlier frame containing both the ball and the SAME observed Track ID; no
/// inferred Track ID or synthesized ball is published to downstream stages.
fn witnessed_strike_before_gap(
    frames: &[TouchFrame],
    before: usize,
    after: usize,
    predicted: [f64; 2],
    ball_radius: f64,
    width: f64,
    height: f64,
) -> Option<(usize, u64, f64, f64)> {
    let mut best: Option<(f64, usize, u64, f64, f64)> = None;
    for anchor in before.saturating_sub(5)..before {
        let Some(ball) = frames[anchor]
            .selected_ball
            .as_ref()
            .filter(|ball| ball.source_model == "vballnet" && ball.score >= 0.35)
        else {
            continue;
        };
        let observed = center(ball.bbox_xyxy_norm);
        let observed = [observed[0] * width, observed[1] * height];
        if (observed[0] - predicted[0]).hypot(observed[1] - predicted[1])
            > (ball_radius * 9.0).max(85.0)
        {
            continue;
        }
        for player in &frames[anchor].players {
            if player.score < 0.25
                || frames[before]
                    .players
                    .iter()
                    .any(|p| p.track_id == player.track_id)
                || frames[after]
                    .players
                    .iter()
                    .any(|p| p.track_id == player.track_id)
            {
                continue;
            }
            // Independent observations of this ID must bracket its dropout.
            let Some(next) = frames[after + 1..(after + 7).min(frames.len())]
                .iter()
                .flat_map(|row| &row.players)
                .find(|p| p.track_id == player.track_id && p.score >= 0.25)
            else {
                continue;
            };
            let bbox = pixel_bbox(player.bbox_xyxy_norm, width, height);
            let next_box = pixel_bbox(next.bbox_xyxy_norm, width, height);
            let (bw, bh) = (bbox[2] - bbox[0], bbox[3] - bbox[1]);
            let (nw, nh) = (next_box[2] - next_box[0], next_box[3] - next_box[1]);
            if bw < 10.0
                || bh < 35.0
                || nw < 10.0
                || nh < 35.0
                || bh.max(nh) / bh.min(nh) > 2.2
                || ((bbox[0] + bbox[2] - next_box[0] - next_box[2]) * 0.5)
                    .hypot((bbox[1] + bbox[3] - next_box[1] - next_box[3]) * 0.5)
                    > bh.max(nh) * 0.75
            {
                continue;
            }
            let standing = standing_height(frames, anchor, player.track_id, height)
                .unwrap_or(bh)
                .max(bh);
            let center_x = (bbox[0] + bbox[2]) * 0.5;
            let reach_x = (bw * 0.75).max(standing * 0.48) + ball_radius;
            let reach_top = bbox[3] - standing * 1.90 - ball_radius;
            let reach_bottom = bbox[3] + standing * 0.12 + ball_radius;
            if (predicted[0] - center_x).abs() > reach_x
                || predicted[1] < reach_top
                || predicted[1] > reach_bottom
            {
                continue;
            }
            let cost = ((predicted[0] - center_x).abs() / reach_x
                + ((predicted[1] - (reach_top + reach_bottom) * 0.5).abs()
                    / ((reach_bottom - reach_top) * 0.5).max(1.0)))
                * 0.5;
            let rank = cost + (before - anchor) as f64 * 0.025;
            if best.as_ref().is_none_or(|old| rank < old.0) {
                best = Some((rank, anchor, player.track_id, cost, player.score));
            }
        }
    }
    best.map(|(_, frame, track_id, cost, score)| (frame, track_id, cost, score))
}

pub(super) fn detect_gap_rebounds(
    frames: &[TouchFrame],
    width: f64,
    height: f64,
    max_reach_ratio: f64,
) -> Vec<GeometryContact> {
    if frames.len() < 15 {
        return Vec::new();
    }
    let center_px = |i: usize| -> Option<[f64; 2]> {
        let ball = frames[i].selected_ball.as_ref()?;
        if ball.source_model != "vballnet" {
            return None;
        }
        let [x1, y1, x2, y2] = ball.bbox_xyxy_norm;
        Some([(x1 + x2) * width * 0.5, (y1 + y2) * height * 0.5])
    };
    let mut result = Vec::new();
    for before in 4..frames.len().saturating_sub(8) {
        let Some(a) = center_px(before) else { continue };
        if center_px(before + 1).is_some() {
            continue;
        }
        // The gap must have been explicitly cleared by VBallNet-only
        // outlier filtering, or represent genuinely missing detections.
        let after =
            (before + 3..=(before + 10).min(frames.len() - 5)).find(|&i| center_px(i).is_some());
        let Some(after) = after else { continue };
        if (before + 1..after).any(|i| frames[i].selected_ball.is_some()) {
            continue;
        }
        let Some(b) = center_px(after) else { continue };
        // VBallNet can produce two consecutive false-landmark islands with a
        // single *real* ball observation between them. Requiring four
        // consecutive samples discards the incoming flight precisely when
        // an athlete hits the ball. Use three or more independent observed
        // positions on each side, within an eight-frame bounded window;
        // never construct synthetic ball observations for a Touch.
        let fit_velocity = |anchor: usize, backwards: bool| -> Option<[f64; 2]> {
            let mut samples = Vec::with_capacity(5);
            for offset in 0..=8 {
                let index = match if backwards {
                    anchor.checked_sub(offset)
                } else {
                    anchor.checked_add(offset)
                } {
                    Some(index) if index < frames.len() => index,
                    _ => break,
                };
                if let Some(point) = center_px(index) {
                    samples.push((index, point));
                    if samples.len() == 5 {
                        break;
                    }
                }
            }
            if samples.len() < 3
                || samples
                    .windows(2)
                    .any(|pair| pair[0].0.abs_diff(pair[1].0) > 3)
            {
                return None;
            }
            let n = samples.len() as f64;
            let mean_t = samples.iter().map(|(t, _)| *t as f64).sum::<f64>() / n;
            let mean = std::array::from_fn::<_, 2, _>(|axis| {
                samples.iter().map(|(_, p)| p[axis]).sum::<f64>() / n
            });
            let variance = samples
                .iter()
                .map(|(t, _)| (*t as f64 - mean_t).powi(2))
                .sum::<f64>();
            if variance <= 0.0 {
                return None;
            }
            let velocity: [f64; 2] = std::array::from_fn(|axis| {
                samples
                    .iter()
                    .map(|(t, p)| (*t as f64 - mean_t) * (p[axis] - mean[axis]))
                    .sum::<f64>()
                    / variance
            });
            // Reject another VBallNet jump masquerading as a smooth flight.
            if samples.iter().any(|(t, p)| {
                let delta = *t as f64 - mean_t;
                (p[0] - mean[0] - delta * velocity[0]).hypot(p[1] - mean[1] - delta * velocity[1])
                    > 12.0
            }) {
                return None;
            }
            Some(velocity)
        };
        let Some(v_after) = fit_velocity(after, false) else {
            continue;
        };
        let v_before = fit_velocity(before, true).or_else(|| {
            let previous = before.checked_sub(1)?;
            if v_after[1] > -15.0 {
                return None;
            }
            let velocity = fit_velocity(previous, true)?;
            if velocity[1] < 18.0 {
                return None;
            }
            let last = center_px(previous)?;
            let deviation = (a[0] - last[0] - velocity[0]).hypot(a[1] - last[1] - velocity[1]);
            (deviation <= (velocity[0].hypot(velocity[1]) * 1.15).max(16.0)).then_some(velocity)
        });
        let Some(v_before) = v_before else {
            continue;
        };
        let speed_before = v_before[0].hypot(v_before[1]);
        let speed_after = v_after[0].hypot(v_after[1]);
        let top_speed = speed_before.max(speed_after);
        if speed_before < 3.0 || speed_after < 3.0 {
            continue;
        }
        let cosine = (v_before[0] * v_after[0] + v_before[1] * v_after[1])
            / (speed_before * speed_after).max(1e-9);
        let angle = cosine.clamp(-1.0, 1.0).acos().to_degrees();
        let residual = (v_after[0] - v_before[0]).hypot(v_after[1] - v_before[1]);
        if angle < 70.0 || residual < 10.0_f64.max(0.65 * top_speed) {
            continue;
        }
        let ball_before = frames[before].selected_ball.as_ref().unwrap();
        let ball_after = frames[after].selected_ball.as_ref().unwrap();
        let radius = {
            let size = |ball: &super::TouchBall| {
                let [x1, y1, x2, y2] = ball.bbox_xyxy_norm;
                ((x2 - x1) * width + (y2 - y1) * height) * 0.25
            };
            (size(ball_before) + size(ball_after)) * 0.5
        };
        let mut best: Option<(f64, usize, [f64; 2])> = None;
        for impact in before + 1..after {
            let da = (impact - before) as f64;
            let db = (impact as i64 - after as i64) as f64;
            let p = [a[0] + da * v_before[0], a[1] + da * v_before[1]];
            let q = [b[0] + db * v_after[0], b[1] + db * v_after[1]];
            let error = (p[0] - q[0]).hypot(p[1] - q[1]);
            if best.as_ref().is_none_or(|v| error < v.0) {
                best = Some((error, impact, [(p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5]));
            }
        }
        let Some((error, impact, predicted)) = best else {
            continue;
        };
        let direct = error <= (radius * 3.0).max(22.0);
        let contact = if direct {
            touching_player(
                frames,
                impact,
                ReachEvidence {
                    width,
                    height,
                    ball_x: predicted[0],
                    ball_y: predicted[1],
                    ball_radius: radius,
                    max_reach_ratio,
                    strong_reversal: angle >= 120.0,
                },
            )
            .map(|(id, cost, score)| (before, id, cost, score))
        } else if after - before <= 4
            && angle >= 95.0
            && residual >= 18.0
            && speed_after >= 12.0
            && error <= (radius * 11.0).max(85.0)
        {
            // A sharp departure with one or two missing VBallNet positions
            // may not extrapolate to the same pixel. Only a source-observed
            // athlete that disappears at impact AND reappears on the
            // outbound flight can support this weaker proposal.
            witnessed_strike_before_gap(frames, before, after, predicted, radius, width, height)
        } else if after - before <= 10
            && impact <= before + 2
            && v_before[1] >= 18.0
            && v_after[1] <= -15.0
            && angle >= 130.0
            && residual >= 35.0
            && (b[0] - a[0]).abs() <= (radius * 10.0).max(85.0)
        {
            touching_player(
                frames,
                before,
                ReachEvidence {
                    width,
                    height,
                    ball_x: a[0],
                    ball_y: a[1],
                    ball_radius: radius,
                    max_reach_ratio,
                    strong_reversal: true,
                },
            )
            .and_then(|(track_id, cost, score)| {
                let player = frames[before]
                    .players
                    .iter()
                    .find(|player| player.track_id == track_id)?;
                let bbox = pixel_bbox(player.bbox_xyxy_norm, width, height);
                let player_height = bbox[3] - bbox[1];
                let observed_player_frames = frames[before.saturating_sub(2)..=after]
                    .iter()
                    .filter(|frame| {
                        frame
                            .players
                            .iter()
                            .any(|player| player.track_id == track_id)
                    })
                    .count();
                if a[1] > bbox[1] + 0.45 * player_height
                    || error > 0.90 * player_height.max(100.0)
                    || observed_player_frames < 6
                    || !frames[after]
                        .players
                        .iter()
                        .any(|player| player.track_id == track_id)
                {
                    return None;
                }
                Some((before, track_id, cost, score))
            })
        } else {
            None
        };
        let Some((anchor, track_id, cost, player_score)) = contact else {
            continue;
        };
        // Never publish a Touch assigned to a reconstructed-only Track ID.
        if !frames[anchor]
            .players
            .iter()
            .any(|p| p.track_id == track_id)
        {
            continue;
        }
        let probability = (0.37
            + 0.20 * residual / (residual + 12.0)
            + 0.18 * (1.0 - cost.min(1.0))
            + 0.12 * ball_before.score
            + 0.10 * player_score)
            .clamp(0.0, 0.87);
        let anchor_ball = center_px(anchor).expect("source-observed contact anchor");
        result.push(GeometryContact {
            frame: anchor,
            track_id,
            probability,
            motion: MotionEvidence {
                motion_frame: impact as i64,
                speed_before,
                speed_after,
                angle,
                residual,
            },
            ball_x: anchor_ball[0],
            ball_y: anchor_ball[1],
        });
    }
    result
}

#[cfg(test)]
mod gap_tests {
    use super::*;
    #[test]
    fn nearby_striker_reappears_after_short_ball_gap_without_inventing_track_ids() {
        // Rounded source-backed centers around a 30-FPS 23.2s net attack.
        // A spurious far-right VBallNet box is discarded before this stage.
        let observed: &[(u64, f64, f64)] = &[
            (690, 918.8, 333.8),
            (691, 918.8, 345.0),
            (692, 918.8, 360.0),
            (693, 918.8, 375.0),
            (694, 918.8, 390.0),
            (695, 915.0, 401.3),
            (696, 911.2, 412.5),
            (697, 918.8, 420.0),
            (698, 922.5, 427.5),
            (701, 847.5, 382.5),
            (702, 828.8, 375.0),
            (703, 817.5, 371.2),
            (704, 798.8, 371.2),
            (705, 791.2, 367.5),
            (706, 776.2, 367.5),
            (707, 757.5, 360.0),
        ];
        let frames = (680..715u64)
            .map(|frame_index| {
                let players = if frame_index == 695 || (703..=707).contains(&frame_index) {
                    vec![
                        TouchPlayer {
                            track_id: 4,
                            score: 0.67,
                            touch_probability: None,
                            bbox_xyxy_norm: [0.45, 0.46, 0.49, 0.60],
                        },
                        TouchPlayer {
                            track_id: 6,
                            score: 0.35,
                            touch_probability: None,
                            bbox_xyxy_norm: if frame_index == 695 {
                                [0.473, 0.446, 0.490, 0.503]
                            } else {
                                [0.472, 0.426, 0.497, 0.497]
                            },
                        },
                    ]
                } else {
                    vec![TouchPlayer {
                        track_id: 4,
                        score: 0.67,
                        touch_probability: None,
                        bbox_xyxy_norm: [0.45, 0.46, 0.49, 0.60],
                    }]
                };
                let selected_ball =
                    observed
                        .iter()
                        .find(|r| r.0 == frame_index)
                        .map(|r| TouchBall {
                            source_model: "vballnet".into(),
                            score: 0.65,
                            bbox_xyxy_norm: [
                                (r.1 - 8.0) / 1920.0,
                                (r.2 - 8.0) / 1080.0,
                                (r.1 + 8.0) / 1920.0,
                                (r.2 + 8.0) / 1080.0,
                            ],
                        });
                TouchFrame {
                    frame_index,
                    players,
                    selected_ball,
                }
            })
            .collect::<Vec<_>>();
        let detected = detect_gap_rebounds(&frames, 1920.0, 1080.0, 0.75);
        assert!(
            detected.iter().any(|v| v.frame == (695 - 680)
                && v.track_id == 6
                && v.motion.motion_frame >= (699 - 680)),
            "{detected:?}"
        );
        let proposals =
            evaluate_block(&frames, 1920, 1080, 680, 715, TouchSettings::default()).unwrap();
        assert!(
            proposals.iter().any(|v| v.frame_index == 695
                && v.provisional_track_id == Some(6)
                && v.physical_motion_supported),
            "{proposals:?}"
        );
        let mut no_reappearance = frames.clone();
        for frame in &mut no_reappearance {
            if frame.frame_index > 700 {
                frame.players.retain(|player| player.track_id != 6);
            }
        }
        assert!(
            detect_gap_rebounds(&no_reappearance, 1920.0, 1080.0, 0.75)
                .iter()
                .all(|event| event.track_id != 6),
            "An unbracketed player must not get a contact"
        );
        let mut uninterrupted_flight = frames;
        for row in &mut uninterrupted_flight {
            if row.frame_index > 700 {
                let dt = (row.frame_index - 698) as f64;
                let (x, y) = (922.5 + 1.125 * dt, 427.5 + 9.375 * dt);
                row.selected_ball = Some(TouchBall {
                    source_model: "vballnet".into(),
                    score: 0.65,
                    bbox_xyxy_norm: [
                        (x - 8.0) / 1920.0,
                        (y - 8.0) / 1080.0,
                        (x + 8.0) / 1920.0,
                        (y + 8.0) / 1080.0,
                    ],
                });
            }
        }
        assert!(
            detect_gap_rebounds(&uninterrupted_flight, 1920.0, 1080.0, 0.75)
                .iter()
                .all(|event| event.track_id != 6),
            "Player dropout alone must not produce an impact during uninterrupted flight"
        );
    }

    use crate::motion::{ball_motion_change, TOUCH_MOTION_CONFIG};
    use crate::touch_pipeline::{
        evaluate_block, exclude_ball_teleports, TouchBall, TouchPlayer, TouchSettings,
    };

    #[test]
    fn irek_33s_flight_break_is_associated_after_trajectory_detection() {
        let positions: Vec<(u64, f64, f64)> =
            serde_json::from_str(include_str!("../../tests/fixtures/irek_rebound_33s.json"))
                .unwrap();
        let frames: Vec<_> = positions
            .into_iter()
            .map(|(frame_index, x, y)| TouchFrame {
                frame_index,
                players: vec![TouchPlayer {
                    track_id: 14,
                    score: 0.6,
                    touch_probability: None,
                    bbox_xyxy_norm: [0.54, 0.425, 0.565, 0.537],
                }],
                selected_ball: Some(TouchBall {
                    source_model: "vballnet".into(),
                    score: 0.6,
                    bbox_xyxy_norm: [
                        (x - 5.0) / 1280.0,
                        (y - 5.0) / 720.0,
                        (x + 5.0) / 1280.0,
                        (y + 5.0) / 720.0,
                    ],
                }),
            })
            .collect();
        let events =
            evaluate_block(&frames, 1280, 720, 980, 1006, TouchSettings::default()).unwrap();
        assert!(events.iter().any(|e| (990..=994).contains(&e.frame_index) && e.provisional_track_id==Some(14)),"{events:?}");
        assert!(crate::touch_pipeline::evaluate_rally_blocks(
            &frames,
            1280,
            720,
            980,
            1006,
            TouchSettings::default(),
            &[992]
        )
        .unwrap()
        .is_empty());
        let mut no_players = frames;
        for row in &mut no_players {
            row.players.clear();
        }
        assert!(
            evaluate_block(&no_players, 1280, 720, 980, 1006, TouchSettings::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn observed_far_court_reversal_at_31_4s_is_a_touch_candidate() {
        // VBallNet centers from the immutable CPU replay (30 FPS, 1280x720),
        // frames 928..955. The ball reverses 40+ pixels above the small far
        // athlete's detector bbox, inside plausible extended overhead reach.
        // This is a geometry hypothesis, not a verified joint contact.
        let centers: [[f64; 2]; 28] = [
            [547.5, 187.5],
            [550.0, 192.5],
            [550.0, 200.0],
            [552.5, 205.0],
            [555.0, 210.0],
            [555.0, 217.5],
            [557.5, 225.0],
            [560.0, 230.0],
            [562.5, 237.5],
            [562.5, 250.0],
            [565.0, 257.5],
            [570.0, 267.5],
            [570.0, 272.5],
            [570.0, 272.5],
            [570.0, 267.5],
            [572.5, 260.0],
            [577.5, 247.5],
            [575.0, 237.5],
            [580.0, 227.5],
            [582.5, 217.5],
            [582.5, 207.5],
            [585.0, 197.5],
            [585.0, 190.0],
            [590.0, 177.5],
            [590.0, 172.5],
            [595.0, 160.0],
            [595.0, 157.5],
            [597.5, 150.0],
        ];
        let frames: Vec<TouchFrame> = centers
            .iter()
            .enumerate()
            .map(|(i, &[x, y])| TouchFrame {
                frame_index: 928 + i as u64,
                players: vec![TouchPlayer {
                    track_id: 12,
                    score: 0.34,
                    touch_probability: None,
                    bbox_xyxy_norm: [0.428, 0.436, 0.458, 0.526],
                }],
                selected_ball: Some(TouchBall {
                    source_model: "vballnet".into(),
                    score: 0.60,
                    bbox_xyxy_norm: [
                        (x - 5.0) / 1280.0,
                        (y - 5.0) / 720.0,
                        (x + 5.0) / 1280.0,
                        (y + 5.0) / 720.0,
                    ],
                }),
            })
            .collect();
        let hits = evaluate_block(&frames, 1280, 720, 936, 947, TouchSettings::default())
            .expect("replay-derived frames valid");
        assert!(
            hits.iter().any(|hit| (939..=944).contains(&hit.frame_index)
                && hit.provisional_track_id == Some(12)),
            "real observed turn must create a player-linked proposal: {hits:?}"
        );
    }

    #[test]
    fn near_straight_acceleration_at_21_7s_is_a_touch_only_near_player() {
        let make_frames = |speed_out: f64| -> Vec<TouchFrame> {
            (0..27u64)
                .map(|i| {
                    let y = if i <= 12 {
                        490.0 + (12.0 - i as f64) * 4.0
                    } else {
                        490.0 - (i as f64 - 12.0) * speed_out
                    };
                    TouchFrame {
                        frame_index: 639 + i,
                        players: vec![TouchPlayer {
                            track_id: 11,
                            score: 0.90,
                            touch_probability: None,
                            bbox_xyxy_norm: [0.47, 0.48, 0.54, 0.76],
                        }],
                        selected_ball: Some(TouchBall {
                            source_model: "vballnet".into(),
                            score: 0.80,
                            bbox_xyxy_norm: [
                                (960.0 - 8.0) / 1920.0,
                                (y - 8.0) / 1080.0,
                                (960.0 + 8.0) / 1920.0,
                                (y + 8.0) / 1080.0,
                            ],
                        }),
                    }
                })
                .collect()
        };
        let accelerated = make_frames(28.0);
        let events =
            evaluate_block(&accelerated, 1920, 1080, 639, 666, TouchSettings::default()).unwrap();
        assert!(
            events.iter().any(|e| {
                (649..=654).contains(&e.frame_index)
                    && e.provisional_track_id == Some(11)
                    && e.speed_after_px_frame > e.speed_before_px_frame * 2.5
                    && e.direction_change_deg < 5.0
            }),
            "near-straight hand acceleration must be proposed: {events:?}"
        );
        let mut no_player = accelerated.clone();
        for frame in &mut no_player {
            frame.players.clear();
        }
        assert!(
            evaluate_block(&no_player, 1920, 1080, 639, 666, TouchSettings::default())
                .unwrap()
                .is_empty()
        );
        let continuous = make_frames(4.0);
        assert!(
            evaluate_block(&continuous, 1920, 1080, 639, 666, TouchSettings::default())
                .unwrap()
                .is_empty()
        );
        let slowing = make_frames(0.8);
        assert!(
            evaluate_block(&slowing, 1920, 1080, 639, 666, TouchSettings::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn straight_deceleration_near_player_is_not_a_touch_at_4_9s() {
        // Reproduces the physics shape of an observed false event: the ball
        // slows sharply but keeps flying in the same direction beside a player.
        // This must not create an unconfirmed geometry-only Touch.
        let frames: Vec<TouchFrame> = (0..26u64)
            .map(|i| {
                let y = if i <= 12 {
                    0.86 - i as f64 * 0.037
                } else {
                    0.416 - (i - 12) as f64 * 0.009
                };
                TouchFrame {
                    frame_index: 132 + i,
                    players: vec![TouchPlayer {
                        track_id: 1,
                        score: 0.95,
                        touch_probability: None,
                        bbox_xyxy_norm: [0.37, 0.35, 0.47, 0.70],
                    }],
                    selected_ball: Some(TouchBall {
                        source_model: "vballnet".into(),
                        score: 0.90,
                        bbox_xyxy_norm: [0.403, y - 0.013, 0.421, y + 0.013],
                    }),
                }
            })
            .collect();
        let balls: Vec<BallObservation> = frames
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let bbox = row.selected_ball.as_ref().unwrap().bbox_xyxy_norm;
                BallObservation {
                    frame: i as i64,
                    x: (bbox[0] + bbox[2]) * 640.0,
                    y: (bbox[1] + bbox[3]) * 360.0,
                }
            })
            .collect();
        let observed = ball_motion_change(&balls, 12, 0, TOUCH_MOTION_CONFIG)
            .expect("observed flights on both sides of a velocity break");
        assert!(
            observed.angle < 5.0,
            "expected near-straight flight: {observed:?}"
        );
        assert!(
            (observed.speed_after - observed.speed_before).abs() > 6.0,
            "the existing speed-change exception should be exercised: {observed:?}"
        );
        let contacts = evaluate_block(&frames, 1280, 720, 132, 158, TouchSettings::default())
            .expect("valid selected VBallNet frames");
        assert!(
            contacts.is_empty(),
            "deceleration without deflection is not a Touch: {contacts:?}"
        );

        // A modest but real 5-35° turn must now be eligible when the other
        // physics and player-reach evidence is strong.
        let mut turning = frames.clone();
        for (i, frame) in turning.iter_mut().enumerate().skip(13) {
            let dx = (i - 12) as f64 * 0.00085;
            let bbox = &mut frame.selected_ball.as_mut().unwrap().bbox_xyxy_norm;
            bbox[0] += dx;
            bbox[2] += dx;
        }
        let contacts = evaluate_block(&turning, 1280, 720, 132, 158, TouchSettings::default())
            .expect("valid turned VBallNet frames");
        assert!(
            contacts
                .iter()
                .any(|event| (5.0..35.0).contains(&event.direction_change_deg)),
            "a supported small turn should pass the new 5° threshold: {contacts:?}"
        );
    }

    #[test]
    fn stationary_out_of_play_ball_cannot_be_player_touch_at_25_3s() {
        // Saved new3 VBallNet observations around frame 759. The ball stops
        // near x=0.305, y=0.556; Track 15 is geometrically nearby.
        let positions = [
            [0.275, 0.472],
            [0.279, 0.483],
            [0.281, 0.490],
            [0.287, 0.503],
            [0.291, 0.517],
            [0.293, 0.528],
            [0.297, 0.538],
            [0.299, 0.549],
            [0.305, 0.556],
            [0.305, 0.559],
            [0.305, 0.559],
            [0.305, 0.559],
            [0.305, 0.556],
            [0.305, 0.556],
            [0.303, 0.556],
            [0.303, 0.549],
            [0.303, 0.545],
        ];
        let frames: Vec<TouchFrame> = positions
            .iter()
            .enumerate()
            .map(|(i, &xy)| TouchFrame {
                frame_index: 751 + i as u64,
                players: vec![TouchPlayer {
                    track_id: 15,
                    score: 0.70,
                    touch_probability: None,
                    bbox_xyxy_norm: [0.269, 0.636, 0.328, 0.876],
                }],
                selected_ball: Some(TouchBall {
                    source_model: "vballnet".to_string(),
                    score: 0.62,
                    bbox_xyxy_norm: [xy[0] - 0.005, xy[1] - 0.006, xy[0] + 0.005, xy[1] + 0.006],
                }),
            })
            .collect();
        assert!(settled_ball_after(&frames, 7, 1280.0, 720.0));
        assert!(settled_ball_after(&frames, 8, 1280.0, 720.0));
        assert!(
            settled_ball_after(&frames, 10, 1280.0, 720.0),
            "following frames of the same settled phase must not become contacts"
        );
        let observations = frames
            .iter()
            .map(|row| {
                let b = row.selected_ball.as_ref().unwrap().bbox_xyxy_norm;
                BallObservation {
                    frame: (row.frame_index - 751) as i64,
                    x: (b[0] + b[2]) * 640.0,
                    y: (b[1] + b[3]) * 360.0,
                }
            })
            .collect::<Vec<_>>();
        let hits = detect(&frames, &observations, 1280.0, 720.0, 0.75);
        assert!(
            !hits.iter().any(|event| (7..=11).contains(&event.frame)),
            "ball stopping near a player is not contact: {hits:?}"
        );
    }

    #[test]
    fn volleyball_40_1s_slow_inbound_fast_vertical_reversal_is_touch() {
        let pts = [
            [0.645, 0.299],
            [0.646, 0.309],
            [0.648, 0.319],
            [0.648, 0.333],
            [0.646, 0.347],
            [0.646, 0.361],
            [0.646, 0.375],
            [0.646, 0.385],
            [0.645, 0.389],
            [0.646, 0.410],
            [0.645, 0.417],
            [0.643, 0.413],
            [0.637, 0.344],
            [0.637, 0.340],
            [0.631, 0.288],
            [0.623, 0.253],
            [0.619, 0.236],
        ];
        let frames = pts
            .iter()
            .enumerate()
            .map(|(i, &p)| TouchFrame {
                frame_index: 1194 + i as u64,
                players: vec![TouchPlayer {
                    track_id: 17,
                    score: 0.50,
                    touch_probability: None,
                    bbox_xyxy_norm: [0.629, 0.419, 0.664, 0.539],
                }],
                selected_ball: Some(TouchBall {
                    source_model: "vballnet".into(),
                    score: 0.65,
                    bbox_xyxy_norm: [p[0] - 0.005, p[1] - 0.006, p[0] + 0.005, p[1] + 0.006],
                }),
            })
            .collect::<Vec<_>>();
        let proposed = detect_observed_vertical_rebounds(&frames, 1280.0, 720.0);
        assert!(
            proposed
                .iter()
                .any(|item| (9..=12).contains(&item.frame) && item.track_id == 17),
            "40.1s rebound must be kept: {proposed:?}"
        );
    }

    #[test]
    fn rebound_at_36s_is_not_lost_and_27s_ground_settling_is_not_rebound() {
        let points = [
            [0.445, 0.354],
            [0.447, 0.372],
            [0.447, 0.385],
            [0.453, 0.403],
            [0.455, 0.427],
            [0.459, 0.451],
            [0.459, 0.476],
            [0.463, 0.507],
            [0.465, 0.535],
            [0.467, 0.566],
            [0.469, 0.597],
            [0.475, 0.635],
            [0.477, 0.681],
            [0.475, 0.677],
            [0.473, 0.653],
            [0.465, 0.569],
            [0.461, 0.569],
            [0.461, 0.528],
            [0.453, 0.500],
            [0.447, 0.462],
            [0.443, 0.434],
        ];
        let frames: Vec<TouchFrame> = points
            .iter()
            .enumerate()
            .map(|(i, &p)| TouchFrame {
                frame_index: 1068 + i as u64,
                players: (i >= 11)
                    .then_some(TouchPlayer {
                        track_id: 15,
                        score: 0.65,
                        touch_probability: None,
                        bbox_xyxy_norm: [0.409, 0.679, 0.492, 0.891],
                    })
                    .into_iter()
                    .collect(),
                selected_ball: Some(TouchBall {
                    source_model: "vballnet".into(),
                    score: 0.62,
                    bbox_xyxy_norm: [p[0] - 0.005, p[1] - 0.006, p[0] + 0.005, p[1] + 0.006],
                }),
            })
            .collect();
        let found = detect_observed_vertical_rebounds(&frames, 1280.0, 720.0);
        assert!(
            found
                .iter()
                .any(|v| (11..=14).contains(&v.frame) && v.track_id == 15),
            "upper-body reversal at 36s must be proposed: {found:?}"
        );
        let mut no_player = frames.clone();
        for row in &mut no_player {
            row.players.clear();
        }
        assert!(detect_observed_vertical_rebounds(&no_player, 1280.0, 720.0).is_empty());
        let floor = [
            [0.326, 0.698],
            [0.330, 0.722],
            [0.332, 0.740],
            [0.334, 0.764],
            [0.340, 0.774],
            [0.344, 0.785],
            [0.344, 0.778],
            [0.344, 0.778],
            [0.348, 0.788],
            [0.352, 0.792],
        ]
        .into_iter()
        .enumerate()
        .map(|(i, p)| TouchFrame {
            frame_index: 813 + i as u64,
            players: vec![TouchPlayer {
                track_id: 15,
                score: 0.72,
                touch_probability: None,
                bbox_xyxy_norm: [0.35, 0.568, 0.422, 0.793],
            }],
            selected_ball: Some(TouchBall {
                source_model: "vballnet".into(),
                score: 0.62,
                bbox_xyxy_norm: [p[0] - 0.005, p[1] - 0.006, p[0] + 0.005, p[1] + 0.006],
            }),
        })
        .collect::<Vec<_>>();
        assert!(detect_observed_vertical_rebounds(&floor, 1280.0, 720.0).is_empty());
        assert!(settled_ball_after(&floor, 3, 1280.0, 720.0));
        assert!(settled_ball_after(&floor, 4, 1280.0, 720.0));
    }

    fn flight_at_13_seconds(with_rebound: bool) -> Vec<TouchFrame> {
        (380_u64..=420)
            .map(|i| {
                let (x, y) = if i <= 395 {
                    (
                        652.0 - 3.0 * (i as f64 - 395.0),
                        262.0 + 13.0 * (i as f64 - 395.0),
                    )
                } else if i >= 402 {
                    if with_rebound {
                        (
                            645.0 + 0.5 * (i as f64 - 402.0),
                            260.0 - 12.75 * (i as f64 - 402.0),
                        )
                    } else {
                        (
                            652.0 - 3.0 * (i as f64 - 395.0),
                            262.0 + 13.0 * (i as f64 - 395.0),
                        )
                    }
                } else {
                    (1190.0, 417.0) // VBallNet incorrectly latched onto a far landmark
                };
                let valid_ball = TouchBall {
                    source_model: "vballnet".to_owned(),
                    score: 0.70,
                    bbox_xyxy_norm: [
                        (x - 7.0) / 1280.0,
                        (y - 7.0) / 720.0,
                        (x + 7.0) / 1280.0,
                        (y + 7.0) / 720.0,
                    ],
                };
                TouchFrame {
                    frame_index: i,
                    players: vec![
                        TouchPlayer {
                            track_id: 4,
                            score: 0.53,
                            touch_probability: None,
                            bbox_xyxy_norm: [
                                629.0 / 1280.0,
                                304.0 / 720.0,
                                665.0 / 1280.0,
                                371.0 / 720.0,
                            ],
                        },
                        TouchPlayer {
                            track_id: 1,
                            score: 0.96,
                            touch_probability: None,
                            bbox_xyxy_norm: [
                                635.0 / 1280.0,
                                342.0 / 720.0,
                                688.0 / 1280.0,
                                484.0 / 720.0,
                            ],
                        },
                    ],
                    selected_ball: Some(valid_ball),
                }
            })
            .collect()
    }

    #[test]
    fn real_vballnet_flight_reversal_across_removed_outlier_is_detected() {
        let frames = flight_at_13_seconds(true);
        let clean = exclude_ball_teleports(&frames, 1280, 720);
        assert!((396..=401).all(|frame| clean[(frame - 380) as usize].selected_ball.is_none()));
        let events = evaluate_block(&clean, 1280, 720, 380, 421, TouchSettings::default()).unwrap();
        assert!(
            events.iter().any(|e| {
                e.frame_index == 395
                    && e.motion_frame_index >= 396
                    && e.motion_frame_index <= 401
                    && e.physical_motion_supported
            }),
            "Real VBallNet rebound around 13.3 seconds must survive the outlier gap: {events:?}"
        );
        assert!(events.iter().all(|e| {
            clean[(e.frame_index - 380) as usize]
                .selected_ball
                .as_ref()
                .is_some_and(|b| b.source_model == "vballnet")
        }));
    }

    /// A landmark at frame 475 and another at 477..=484 must not erase
    /// the actual observed ball at 476. The two observed ballistic segments
    /// meet near player 6 at ~16 seconds, where VBallNet is temporarily wrong.
    fn interrupted_flight_at_16_seconds(rebound: bool, player_present: bool) -> Vec<TouchFrame> {
        (463_u64..=495)
            .map(|frame| {
                let t = frame as f64;
                let (x, y) = if frame == 475 || (477..=484).contains(&frame) {
                    (1192.0, 420.0) // false landmark, not the ball
                } else if frame <= 476 {
                    (820.0 + 6.0 * (t - 476.0), 407.0 + 15.0 * (t - 476.0))
                } else if rebound {
                    (808.0 - 10.0 * (t - 485.0), 400.0 - 29.0 * (t - 485.0))
                } else {
                    (820.0 + 6.0 * (t - 476.0), 407.0 + 15.0 * (t - 476.0))
                };
                TouchFrame {
                    frame_index: frame,
                    players: if player_present {
                        vec![TouchPlayer {
                            track_id: 6,
                            score: 0.66,
                            touch_probability: None,
                            bbox_xyxy_norm: [
                                817.0 / 1280.0,
                                480.0 / 720.0,
                                888.0 / 1280.0,
                                710.0 / 720.0,
                            ],
                        }]
                    } else {
                        vec![]
                    },
                    selected_ball: Some(TouchBall {
                        source_model: "vballnet".into(),
                        score: 0.65,
                        bbox_xyxy_norm: [
                            (x - 7.0) / 1280.0,
                            (y - 7.0) / 720.0,
                            (x + 7.0) / 1280.0,
                            (y + 7.0) / 720.0,
                        ],
                    }),
                }
            })
            .collect()
    }

    #[test]
    fn two_remote_islands_with_one_real_ball_sample_still_recover_player_rebound() {
        let raw = interrupted_flight_at_16_seconds(true, true);
        let clean = exclude_ball_teleports(&raw, 1280, 720);
        assert!(clean[(475 - 463) as usize].selected_ball.is_none());
        assert!(clean[(477 - 463) as usize..=(484 - 463) as usize]
            .iter()
            .all(|row| row.selected_ball.is_none()));
        assert!(clean[(476 - 463) as usize].selected_ball.is_some());
        let events = evaluate_block(&clean, 1280, 720, 463, 496, TouchSettings::default()).unwrap();
        assert_eq!(events.len(), 1, "Expected the physical rebound: {events:?}");
        assert_eq!(events[0].frame_index, 476);
        assert!((477..=484).contains(&events[0].motion_frame_index));
        assert_eq!(events[0].provisional_track_id, Some(6));
        assert!(events[0].physical_motion_supported);
        // An unrelated player's absence does not authorize a fictitious touch.
        let no_player =
            exclude_ball_teleports(&interrupted_flight_at_16_seconds(true, false), 1280, 720);
        assert!(
            evaluate_block(&no_player, 1280, 720, 463, 496, TouchSettings::default(),)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn broken_observation_is_not_a_touch_without_physical_reversal() {
        let frames =
            exclude_ball_teleports(&interrupted_flight_at_16_seconds(false, true), 1280, 720);
        assert!(detect_gap_rebounds(&frames, 1280.0, 720.0, 0.75).is_empty());
    }

    #[test]
    fn straight_flight_and_unsupported_ball_are_not_gap_contacts() {
        let frames = exclude_ball_teleports(&flight_at_13_seconds(false), 1280, 720);
        assert!(detect_gap_rebounds(&frames, 1280.0, 720.0, 0.70).is_empty());
        let mut frames = exclude_ball_teleports(&flight_at_13_seconds(true), 1280, 720);
        frames[(395 - 380) as usize]
            .selected_ball
            .as_mut()
            .unwrap()
            .source_model = "unsupported".into();
        assert!(detect_gap_rebounds(&frames, 1280.0, 720.0, 0.70).is_empty());
    }
}

#[cfg(test)]
mod stabilized_gap_rebound_tests {
    use super::*;
    use crate::touch_pipeline::{evaluate_block, TouchBall, TouchPlayer, TouchSettings};

    fn interrupted_overhead_rebound(with_player: bool, reversal: bool) -> Vec<TouchFrame> {
        (407..=440_u64)
            .map(|frame_index| {
                let y = if frame_index <= 418 {
                    Some(500.0 + 26.0 * (frame_index as f64 - 418.0))
                } else if frame_index == 419 {
                    Some(500.0)
                } else if frame_index < 428 {
                    None
                } else if reversal {
                    Some(250.0 - 18.0 * (frame_index as f64 - 428.0))
                } else {
                    Some(250.0 + 18.0 * (frame_index as f64 - 428.0))
                };
                TouchFrame {
                    frame_index,
                    players: if with_player {
                        vec![TouchPlayer {
                            track_id: 6,
                            score: 0.90,
                            touch_probability: None,
                            bbox_xyxy_norm: [
                                630.0 / 1280.0,
                                505.0 / 720.0,
                                710.0 / 1280.0,
                                700.0 / 720.0,
                            ],
                        }]
                    } else {
                        vec![]
                    },
                    selected_ball: y.map(|y| TouchBall {
                        source_model: "vballnet".into(),
                        score: 0.7,
                        bbox_xyxy_norm: [
                            (650.0 - 7.0) / 1280.0,
                            (y - 7.0) / 720.0,
                            (650.0 + 7.0) / 1280.0,
                            (y + 7.0) / 720.0,
                        ],
                    }),
                }
            })
            .collect()
    }

    #[test]
    fn overhead_rebound_survives_ball_stabilizer_gap_near_14_seconds() {
        let frames = interrupted_overhead_rebound(true, true);
        let witness = touching_player(
            &frames,
            12,
            ReachEvidence {
                width: 1280.0,
                height: 720.0,
                ball_x: 650.0,
                ball_y: 500.0,
                ball_radius: 7.0,
                max_reach_ratio: 0.75,
                strong_reversal: true,
            },
        );
        assert!(
            witness.is_some(),
            "Expected observed body near the rebound: {witness:?}"
        );
        let contacts = detect_gap_rebounds(&frames, 1280.0, 720.0, 0.75);
        assert_eq!(
            contacts.len(),
            1,
            "Gap must retain the physical rebound: {contacts:?}"
        );
        assert_eq!(contacts[0].frame, 12);
        assert_eq!(contacts[0].motion.motion_frame, 13);
        assert_eq!(contacts[0].track_id, 6);
        assert!(
            evaluate_block(&frames, 1280, 720, 407, 441, TouchSettings::default())
                .unwrap()
                .iter()
                .any(|contact| contact.frame_index == 419 && contact.motion_frame_index == 420)
        );
        assert!(detect_gap_rebounds(
            &interrupted_overhead_rebound(false, true),
            1280.0,
            720.0,
            0.75
        )
        .is_empty());
        assert!(detect_gap_rebounds(
            &interrupted_overhead_rebound(true, false),
            1280.0,
            720.0,
            0.75
        )
        .is_empty());
        let mut foot_level = interrupted_overhead_rebound(true, true);
        for frame in &mut foot_level {
            frame.players[0].bbox_xyxy_norm =
                [630.0 / 1280.0, 290.0 / 720.0, 710.0 / 1280.0, 600.0 / 720.0];
        }
        assert!(detect_gap_rebounds(&foot_level, 1280.0, 720.0, 0.75).is_empty());
    }
}
