//! Remove short VballNet teleports from *temporary* postprocessing samples.
//! The immutable fusion/replay remains unchanged. A false stationary landmark
//! between two compatible ball observations must not erase nearby contacts.
use super::{center, valid_bbox, TouchFrame};
mod leading_ball;
pub use leading_ball::{LeadingBallObservation, LeadingBallSettings, LeadingBallTracker};

fn location(frame: &TouchFrame, width: f64, height: f64) -> Option<[f64; 2]> {
    let ball = frame
        .selected_ball
        .as_ref()
        .filter(|b| b.source_model == "vballnet" && valid_bbox(b.bbox_xyxy_norm))?;
    let [x, y] = center(ball.bbox_xyxy_norm);
    Some([x * width, y * height])
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Repeated detections of a fixed background feature can become a sticky
/// high-confidence "ball", including across brief returns to the real flight.
/// Detect that *temporal* pattern before rejecting isolated teleport islands.
/// Local neighborhoods fit both the streaming CPU halo and preview paging.
fn flag_stationary_landmarks(samples: &[Option<[f64; 2]>], width: f64, height: f64) -> Vec<bool> {
    const NEIGHBORHOOD: usize = 16;
    const MIN_REPEAT: usize = 7;
    const MIN_SPAN: usize = 8;
    const LONG_DWELL: usize = 14;
    let mut flagged = vec![false; samples.len()];
    let stationary_radius = (width.max(height) * 0.008).max(7.0);
    let teleport = (width * 0.09).max(90.0);
    for (index, candidate) in samples.iter().enumerate() {
        let Some(point) = candidate else { continue };
        let origin = [point[0] * width, point[1] * height];
        let begin = index.saturating_sub(NEIGHBORHOOD);
        let end = index.saturating_add(NEIGHBORHOOD + 1).min(samples.len());
        let mut count = 0;
        let mut first = None;
        let mut last = index;
        let mut min_xy = [f64::INFINITY; 2];
        let mut max_xy = [f64::NEG_INFINITY; 2];
        let (mut sum_t, mut sum_tt) = (0.0, 0.0);
        let (mut sum_x, mut sum_y, mut sum_tx, mut sum_ty) = (0.0, 0.0, 0.0, 0.0);
        let mut abrupt_switch = false;
        for near in begin..end {
            let Some(other) = samples[near] else { continue };
            let other = [other[0] * width, other[1] * height];
            if distance(origin, other) > stationary_radius {
                continue;
            }
            count += 1;
            first.get_or_insert(near);
            last = near;
            let t = (near - begin) as f64;
            sum_t += t;
            sum_tt += t * t;
            sum_x += other[0];
            sum_y += other[1];
            sum_tx += t * other[0];
            sum_ty += t * other[1];
            for axis in 0..2 {
                min_xy[axis] = min_xy[axis].min(other[axis]);
                max_xy[axis] = max_xy[axis].max(other[axis]);
            }
            // A real slow-moving ball cannot jump a tenth of the image
            // within 1-3 frames and return repeatedly to the same pixel.
            for delta in 1..=3 {
                for adjacent in [near.checked_sub(delta), near.checked_add(delta)] {
                    let Some(adjacent) = adjacent.filter(|&j| j >= begin && j < end) else {
                        continue;
                    };
                    if let Some(remote) = samples[adjacent] {
                        abrupt_switch |=
                            distance(other, [remote[0] * width, remote[1] * height]) >= teleport;
                    }
                }
            }
        }
        // A slowly traveling real ball may spend many frames within this
        // neighborhood. Its full path still spans more than a fixed pixel.
        let spread = distance(min_xy, max_xy);
        let denominator = count as f64 * sum_tt - sum_t * sum_t;
        let drift = if denominator > 0.0 {
            let vx = (count as f64 * sum_tx - sum_t * sum_x) / denominator;
            let vy = (count as f64 * sum_ty - sum_t * sum_y) / denominator;
            vx.hypot(vy)
        } else {
            f64::INFINITY
        };
        if count >= MIN_REPEAT
            && spread <= stationary_radius * 1.25
            && drift < 0.55
            && last - first.unwrap_or(index) >= MIN_SPAN
            && (abrupt_switch
                || (count >= LONG_DWELL && last - first.unwrap_or(index) >= LONG_DWELL))
        {
            flagged[index] = true;
        }
    }
    flagged
}

/// A wrong VBallNet location can have *higher* confidence than the real
/// ball. Compare motion on both sides instead of deleting every low-score
/// sample. Reject both persistent stationary landmarks and brief out-and-back islands.
fn suppress_local_ball_artifacts(
    samples: &mut [Option<[f64; 2]>],
    width: u32,
    height: u32,
) -> Vec<bool> {
    let mut rejected = vec![false; samples.len()];
    if width < 64 || height < 64 || samples.len() < 3 {
        return rejected;
    }
    let (w, h) = (f64::from(width), f64::from(height));
    // Reject single-frame out-and-back jumps BEFORE removing stationary
    // landmarks. Otherwise both neighbors can be suppressed as a sticky
    // background target, leaving the isolated false detection as a touch.
    let teleport = (w * 0.09).max(90.0);
    let observed = samples.to_vec();
    let stationary = flag_stationary_landmarks(samples, w, h);
    for index in 1..samples.len() - 1 {
        let (Some(before), Some(point), Some(after)) =
            (observed[index - 1], observed[index], observed[index + 1])
        else {
            continue;
        };
        let a = [before[0] * w, before[1] * h];
        let p = [point[0] * w, point[1] * h];
        let b = [after[0] * w, after[1] * h];
        if distance(a, b) <= teleport * 0.5
            && distance(a, p) >= teleport
            && distance(p, b) >= teleport
        {
            // A real flight may have one valid frame between two false
            // stationary readings. Preserve that frame when the real flight
            // independently supports it just before or after the switch.
            let locally_supported = (index.saturating_sub(3)
                ..index.saturating_add(4).min(observed.len()))
                .filter(|&near| near != index && !stationary[near])
                .any(|near| {
                    observed[near].is_some_and(|other| {
                        distance(p, [other[0] * w, other[1] * h])
                            <= 26.0 + 32.0 * index.abs_diff(near) as f64
                    })
                });
            if !locally_supported {
                samples[index] = None;
                rejected[index] = true;
            }
        }
    }
    // Work on the remaining observations so one false landmark cannot
    // repeatedly become an anchor for a reconstructed trajectory.
    for (index, is_landmark) in stationary.into_iter().enumerate() {
        if is_landmark {
            samples[index] = None;
            rejected[index] = true;
        }
    }
    // At 30 FPS a one-frame move of >~100 px at 720p is suspect,
    // but it is *not* removed unless observations on both sides
    // independently agree that a short remote island is false.
    let min_teleport = (w * 0.09).max(90.0);
    const MAX_BRIDGE_FRAMES: usize = 11;
    for before in 0..samples.len() - 2 {
        let Some(a) = samples[before] else { continue };
        // A dropout just before the false detection is common. Compare the
        // first *observed* position within three frames, not only the next
        // frame, so an isolated remote detection cannot anchor a new flight.
        let Some(first_remote) =
            (before + 1..(before + 4).min(samples.len())).find(|&index| samples[index].is_some())
        else {
            continue;
        };
        let next = samples[first_remote].expect("observed remote candidate");
        let a_px = [a[0] * w, a[1] * h];
        let next_px = [next[0] * w, next[1] * h];
        if distance(a_px, next_px) < min_teleport {
            continue;
        }
        let end = (before + MAX_BRIDGE_FRAMES + 1).min(samples.len());
        for after in first_remote + 1..end {
            let Some(b) = samples[after] else { continue };
            let b_px = [b[0] * w, b[1] * h];
            let gap = (after - before) as f64;
            if distance(a_px, b_px) > 60.0 + 22.0 * gap {
                continue;
            }
            let separated = (before + 1..after).all(|index| {
                let Some(p) = samples[index] else { return true };
                let t = (index - before) as f64 / gap;
                let bridge = [
                    a_px[0] + t * (b_px[0] - a_px[0]),
                    a_px[1] + t * (b_px[1] - a_px[1]),
                ];
                distance([p[0] * w, p[1] * h], bridge) > min_teleport
            });
            if separated {
                for index in before + 1..after {
                    if samples[index].is_some() {
                        rejected[index] = true;
                        samples[index] = None;
                    }
                }
                break;
            }
        }
    }
    rejected
}

pub fn flag_local_ball_artifacts(
    samples: &[Option<[f64; 2]>],
    width: u32,
    height: u32,
) -> Vec<bool> {
    let mut working = samples.to_vec();
    suppress_local_ball_artifacts(&mut working, width, height)
}

fn suppress_false_ball_islands(
    samples: &mut [Option<[f64; 2]>],
    width: u32,
    height: u32,
) -> Vec<bool> {
    let mut rejected = suppress_local_ball_artifacts(samples, width, height);
    if width < 64 || height < 64 {
        return rejected;
    }
    let mut tracker = LeadingBallTracker::new(width, height);
    let observed = samples.to_vec();
    for index in 0..samples.len() {
        let decision = tracker.observe(index as u64, observed[index].as_slice());
        if samples[index].is_some() && decision.selected.is_none() {
            samples[index] = None;
            rejected[index] = true;
        }
        if let Some(anchors) = decision.acquired_frames {
            for anchor in anchors {
                samples[anchor as usize] = observed[anchor as usize];
                rejected[anchor as usize] = false;
            }
        }
    }
    rejected
}

/// Identify unreliable observed positions without inventing a ball location.
/// Streaming CPU stages can apply this boolean mask to their own source data.
pub fn flag_ball_teleports(samples: &[Option<[f64; 2]>], width: u32, height: u32) -> Vec<bool> {
    let mut working = samples.to_vec();
    suppress_false_ball_islands(&mut working, width, height)
}

/// Read-only reconstruction for *display*, never physical Touch evidence.
/// Correct detections stay anchored exactly. Physically plausible short gaps
/// are filled between VBallNet observations; ambiguous flight reversals,
/// out-of-frame arcs and long gaps intentionally stay empty.
pub fn repair_vballnet_trajectory(
    samples: &[Option<[f64; 2]>],
    width: u32,
    height: u32,
) -> (Vec<Option<[f64; 2]>>, Vec<bool>) {
    repair_vballnet_trajectory_with_players(
        samples,
        &vec![Vec::new(); samples.len()],
        width,
        height,
    )
}

/// The player-aware path also joins two genuinely different flights *only*
/// when their observed velocities meet inside a tracked athlete's reach.
/// This branch is exclusively for processed preview visualization.
pub fn repair_vballnet_trajectory_with_players(
    samples: &[Option<[f64; 2]>],
    players: &[Vec<[f64; 4]>],
    width: u32,
    height: u32,
) -> (Vec<Option<[f64; 2]>>, Vec<bool>) {
    let mut points = samples.to_vec();
    suppress_false_ball_islands(&mut points, width, height);
    let mut synthetic = vec![false; points.len()];
    crate::ball_interpolation::interpolate_player_rebound_gaps(
        &mut points,
        &mut synthetic,
        players,
        width,
        height,
    );
    let ballistic = crate::ball_interpolation::interpolate_short_gaps(&mut points);
    for (flag, fitted) in synthetic.iter_mut().zip(ballistic) {
        *flag |= fitted;
    }
    (points, synthetic)
}

pub fn repair_accepted_vballnet_trajectory(
    samples: &[Option<[f64; 2]>],
    players: &[Vec<[f64; 4]>],
    track_ids: &[u32],
    width: u32,
    height: u32,
) -> (Vec<Option<[f64; 2]>>, Vec<bool>) {
    let mut points = samples.to_vec();
    let mut synthetic = vec![false; samples.len()];
    if samples.len() != players.len() || samples.len() != track_ids.len() {
        return (points, synthetic);
    }
    let mut begin = 0;
    while begin < samples.len() {
        let mut end = begin + 1;
        while end < samples.len() && track_ids[end] == track_ids[begin] {
            end += 1;
        }
        if track_ids[begin] > 0 {
            crate::ball_interpolation::interpolate_player_rebound_gaps(
                &mut points[begin..end],
                &mut synthetic[begin..end],
                &players[begin..end],
                width,
                height,
            );
            let ballistic =
                crate::ball_interpolation::interpolate_short_gaps(&mut points[begin..end]);
            for (flag, fitted) in synthetic[begin..end].iter_mut().zip(ballistic) {
                *flag |= fitted;
            }
            let mut left = begin;
            while left + 1 < end {
                if points[left].is_none() || synthetic[left] {
                    left += 1;
                    continue;
                }
                let mut right = left + 1;
                while right < end && points[right].is_none() {
                    right += 1;
                }
                if right == end {
                    break;
                }
                if (1..=6).contains(&(right - left - 1)) && !synthetic[right] {
                    if let (Some(a), Some(b)) = (points[left], points[right]) {
                        let duration = (right - left) as f64;
                        let displacement = ((b[0] - a[0]) * f64::from(width))
                            .hypot((b[1] - a[1]) * f64::from(height));
                        if displacement <= f64::from(width) * 0.12 * duration {
                            for index in left + 1..right {
                                let t = (index - left) as f64 / duration;
                                points[index] =
                                    Some([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
                                synthetic[index] = true;
                            }
                        }
                    }
                }
                left = right;
            }
            let mut tracker = LeadingBallTracker::new(width, height);
            for index in begin..end {
                if synthetic[index] {
                    if points[index].is_none_or(|point| !tracker.allows(index as u64, point)) {
                        points[index] = None;
                        synthetic[index] = false;
                    }
                } else {
                    if let Some(point) = samples[index] {
                        tracker.accept_anchor(index as u64, point);
                    }
                }
            }
        }
        begin = end;
    }
    (points, synthetic)
}

/// Remove spurious ball locations from a temporary *observed-only* working
/// copy. Synthetic reconstruction must NEVER become player contact evidence.
#[must_use]
pub fn exclude_ball_teleports(frames: &[TouchFrame], width: u32, height: u32) -> Vec<TouchFrame> {
    let mut filtered = frames.to_vec();
    let mut samples = frames
        .iter()
        .map(|frame| location(frame, f64::from(width), f64::from(height)))
        .map(|point| point.map(|p| [p[0] / f64::from(width), p[1] / f64::from(height)]))
        .collect::<Vec<_>>();
    let rejected = suppress_false_ball_islands(&mut samples, width, height);
    for (frame, invalid) in filtered.iter_mut().zip(rejected) {
        if invalid {
            frame.selected_ball = None;
        }
    }
    filtered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::touch_pipeline::TouchBall;

    #[test]
    fn leading_ball_rejects_moving_remote_candidates_without_waiting_for_return() {
        let samples = (0..50)
            .map(|i| {
                Some(if i < 10 {
                    [0.28 + i as f64 * 0.002, 0.3]
                } else {
                    [0.85 + (i - 10) as f64 * 0.002, 0.65]
                })
            })
            .collect::<Vec<_>>();
        let rejected = flag_ball_teleports(&samples, 1280, 720);
        assert!(rejected[10..].iter().all(|flag| *flag));
        assert!(rejected[..10].iter().all(|flag| !flag));
    }

    #[test]
    fn leading_ball_rejects_recorded_13_second_right_edge_detour() {
        let mut samples = (380..=410)
            .map(|i| {
                let t = (i - 387) as f64;
                Some([(432.5 + 8.0 * t) / 1280.0, (5.3 + 0.65 * t * t) / 720.0])
            })
            .collect::<Vec<_>>();
        let false_points = [
            [1182.5, 412.5],
            [1192.5, 417.5],
            [1192.5, 415.0],
            [1192.5, 415.0],
            [1190.0, 412.5],
            [1190.0, 415.0],
            [1190.0, 417.5],
            [1190.0, 417.5],
            [1190.0, 422.5],
            [1192.5, 422.5],
            [1190.0, 422.5],
            [1190.0, 422.5],
        ];
        for (offset, point) in false_points.into_iter().enumerate() {
            samples[8 + offset] = Some([point[0] / 1280.0, point[1] / 720.0]);
        }
        let rejected = flag_ball_teleports(&samples, 1280, 720);
        assert!(rejected[8..20].iter().all(|flag| *flag));
        assert!(!rejected[7] && !rejected[20]);
    }

    #[test]
    fn interpolation_never_crosses_a_new_rally_even_with_matching_motion() {
        let samples = vec![
            Some([0.20, 0.4]),
            Some([0.21, 0.4]),
            Some([0.22, 0.4]),
            None,
            Some([0.24, 0.4]),
            Some([0.25, 0.4]),
            Some([0.26, 0.4]),
        ];
        let (points, inferred) = repair_accepted_vballnet_trajectory(
            &samples,
            &vec![Vec::new(); 7],
            &[1, 1, 1, 2, 2, 2, 2],
            1280,
            720,
        );
        assert!(points[3].is_none() && !inferred[3]);
    }

    #[test]
    fn accepted_preview_fills_short_dropouts_despite_endpoint_velocity_noise() {
        for missing in 1..=6 {
            let mut samples = vec![Some([0.40, 0.48]), Some([0.41, 0.43])];
            samples.extend(vec![None; missing]);
            samples.extend([Some([0.43, 0.41]), Some([0.44, 0.44])]);
            let (points, inferred) = repair_accepted_vballnet_trajectory(
                &samples,
                &vec![Vec::new(); samples.len()],
                &vec![2; samples.len()],
                1280,
                720,
            );
            for index in 2..2 + missing {
                let t = (index - 1) as f64 / (missing + 1) as f64;
                let point = points[index].expect("short dropout restored");
                assert!((point[0] - (0.41 + 0.02 * t)).abs() < 1e-12);
                assert!((point[1] - (0.43 - 0.02 * t)).abs() < 1e-12);
                assert!(inferred[index]);
            }
            for index in [0, 1, missing + 2, missing + 3] {
                assert_eq!(points[index], samples[index]);
                assert!(!inferred[index]);
            }
        }
    }

    #[test]
    fn accepted_preview_does_not_join_long_ambiguous_or_remote_dropouts() {
        for (missing, end) in [(7, [0.43, 0.41]), (1, [0.95, 0.95])] {
            let mut samples = vec![Some([0.40, 0.48]), Some([0.41, 0.43])];
            samples.extend(vec![None; missing]);
            samples.extend([Some(end), Some([0.44, 0.44])]);
            let (points, inferred) = repair_accepted_vballnet_trajectory(
                &samples,
                &vec![Vec::new(); samples.len()],
                &vec![2; samples.len()],
                1280,
                720,
            );
            assert!(points[2..2 + missing].iter().all(Option::is_none));
            assert!(inferred.iter().all(|flag| !flag));
        }
    }

    fn sample(i: u64, x: f64) -> TouchFrame {
        TouchFrame {
            frame_index: i,
            players: Vec::new(),
            selected_ball: Some(TouchBall {
                source_model: "vballnet".into(),
                score: 0.7,
                bbox_xyxy_norm: [x, 0.35, x + 0.02, 0.37],
            }),
        }
    }

    #[test]
    fn persistent_right_side_landmark_is_not_a_ball_track() {
        // A false stationary detection can win for seconds, beyond the
        // old 11-frame out-and-back bridge. Neither confidence nor duration
        // makes this a genuine ball trajectory.
        let observed = (0..92)
            .map(|i| {
                Some(if (16..=73).contains(&i) {
                    [0.932 + (i % 3) as f64 * 0.001, 0.587]
                } else {
                    [0.28 + i as f64 * 0.001, 0.34]
                })
            })
            .collect::<Vec<_>>();
        let flags = flag_ball_teleports(&observed, 1280, 720);
        assert!(flags[16..=73].iter().all(|flag| *flag));
        assert!(!flags[15] && !flags[74]);
        let (repaired, _) = repair_vballnet_trajectory(&observed, 1280, 720);
        assert!(repaired[16..=73]
            .iter()
            .all(|p| p.is_none_or(|p| p[0] < 0.6)));
        assert_eq!(repaired[15], observed[15]);
        assert_eq!(repaired[74], observed[74]);
    }

    #[test]
    fn alternating_real_ball_and_sticky_background_target_preserves_real_ball() {
        // The landmark repeatedly steals the selected VBallNet candidate,
        // while the real ball is still observed between false streaks.
        let observed = (0..52)
            .map(|i| {
                Some(if (9..42).contains(&i) && (i - 9) % 10 < 7 {
                    [0.932, 0.586 + (i % 3) as f64 * 0.002]
                } else {
                    [0.35 + i as f64 * 0.0015, 0.38]
                })
            })
            .collect::<Vec<_>>();
        let flags = flag_ball_teleports(&observed, 1280, 720);
        for (i, flagged) in flags.iter().enumerate() {
            if (9..42).contains(&i) && (i - 9) % 10 < 7 {
                assert!(*flagged, "sticky false position must be rejected at {i}");
            } else {
                assert!(!*flagged, "real observation must survive at {i}");
            }
        }
    }

    #[test]
    fn short_stationary_pause_without_teleport_is_not_a_false_landmark() {
        let observed = (0..40)
            .map(|i| {
                Some(if (13..22).contains(&i) {
                    [0.423, 0.36]
                } else if i < 13 {
                    [0.423 - (13 - i) as f64 * 0.002, 0.36]
                } else {
                    [0.423 + (i - 21) as f64 * 0.002, 0.36]
                })
            })
            .collect::<Vec<_>>();
        assert!(flag_ball_teleports(&observed, 1280, 720).iter().all(|f| !f));
    }

    #[test]
    fn remote_detection_after_dropout_does_not_become_new_ball_track() {
        let mut observed = (0..24)
            .map(|i| Some([0.31 + i as f64 * 0.003, 0.36]))
            .collect::<Vec<_>>();
        observed[9] = None;
        observed[10] = Some([0.94, 0.65]);
        observed[11] = Some([0.94, 0.65]);
        observed[12] = None;
        let (repaired, synthetic) = repair_vballnet_trajectory(&observed, 1280, 720);
        for i in 10..=11 {
            assert!(repaired[i].is_none_or(|p| p[0] < 0.6));
            if repaired[i].is_some() {
                assert!(synthetic[i]);
            }
        }
        assert_eq!(observed[10], Some([0.94, 0.65]));
        assert_eq!(repaired[8], observed[8]);
        assert_eq!(repaired[13], observed[13]);
    }

    #[test]
    fn remote_landmark_island_removed_without_changing_source() {
        let frames: Vec<_> = (0..30_u64)
            .map(|i| {
                let x = if (10..=17).contains(&i) {
                    0.93
                } else {
                    0.34 + (i as f64) * 0.001
                };
                sample(i, x)
            })
            .collect();
        let filtered = exclude_ball_teleports(&frames, 1280, 720);
        assert_eq!(
            filtered
                .iter()
                .filter(|f| f.selected_ball.is_none())
                .count(),
            8
        );
        assert!(filtered[9].selected_ball.is_some());
        assert!(filtered[18].selected_ball.is_some());
        assert!(frames.iter().all(|f| f.selected_ball.is_some()));
    }

    #[test]
    fn recovers_volley_23_4s_false_ball_island_with_high_attack() {
        // Real observations around frames 696..707; the GPU briefly chose
        // a stationary right-edge false ball at 697..700 and 703..706.
        let mut observations = vec![None; 19];
        observations[0] = Some([0.481, 0.352]);
        observations[1] = Some([0.479, 0.362]);
        let samples = [
            (0, [0.477, 0.372]),
            (1, [0.475, 0.382]),
            (2, [0.932, 0.587]),
            (3, [0.932, 0.583]),
            (4, [0.930, 0.583]),
            (5, [0.930, 0.587]),
            (6, [0.443, 0.354]),
            (7, [0.430, 0.347]),
            (8, [0.932, 0.587]),
            (9, [0.932, 0.587]),
            (10, [0.934, 0.583]),
            (11, [0.936, 0.580]),
            (12, [0.389, 0.337]),
            (13, [0.385, 0.337]),
        ];
        for (i, xy) in samples {
            observations[i + 2] = Some(xy);
        }
        let mut players = vec![vec![]; observations.len()];
        for row in &mut players {
            row.push([0.448, 0.452, 0.493, 0.605]);
        }
        let (repaired, synthetic) =
            repair_vballnet_trajectory_with_players(&observations, &players, 1280, 720);
        for i in 4..=7 {
            assert!(
                repaired[i].is_some_and(|p| p[0] < 0.55),
                "false ball island at frame {} must be reconstructed: {:?}",
                693 + i,
                repaired[i]
            );
            assert!(synthetic[i]);
        }
        for i in 10..=13 {
            assert!(
                repaired[i].is_some_and(|p| p[0] < 0.55),
                "second false island at frame {} must be reconstructed: {:?}",
                693 + i,
                repaired[i]
            );
            assert!(synthetic[i]);
        }
        assert_eq!(observations[4], Some([0.932, 0.587]));
        assert_eq!(repaired[3], observations[3]);
        assert_eq!(repaired[8], observations[8]);
    }

    #[test]
    fn repairs_high_confidence_false_island_without_changing_real_anchors() {
        let observed = (0..30)
            .map(|i| {
                Some(if (10..=17).contains(&i) {
                    [0.93, 0.59] // The false right-side stationary landmark.
                } else {
                    [0.34 + i as f64 * 0.001, 0.35]
                })
            })
            .collect::<Vec<_>>();
        let (restored, synthetic) = repair_vballnet_trajectory(&observed, 1280, 720);
        assert_eq!(observed[10], Some([0.93, 0.59]));
        for i in 10..=17 {
            assert!(restored[i].is_none_or(|point| point[0] < 0.5));
            if restored[i].is_some() {
                assert!(
                    synthetic[i],
                    "Inferred location must have a synthetic marker"
                );
            }
        }
        assert_eq!(restored[9], observed[9]);
        assert_eq!(restored[18], observed[18]);
        assert!(!synthetic[9] && !synthetic[18]);
    }

    #[test]
    fn does_not_invent_flight_across_a_real_player_reversal() {
        let observed = (463..=495)
            .map(|f| {
                let t = f as f64;
                let [x, y] = if f == 475 || (477..=484).contains(&f) {
                    [1192.0, 420.0]
                } else if f <= 476 {
                    [820.0 + 6.0 * (t - 476.0), 407.0 + 15.0 * (t - 476.0)]
                } else {
                    [808.0 - 10.0 * (t - 485.0), 400.0 - 29.0 * (t - 485.0)]
                };
                Some([x / 1280.0, y / 720.0])
            })
            .collect::<Vec<_>>();
        let (points, synthetic) = repair_vballnet_trajectory(&observed, 1280, 720);
        assert_eq!(points[476 - 463], observed[476 - 463]);
        assert!(!synthetic[476 - 463]);
        for frame in [475, 477, 478, 479, 480, 481, 482, 483, 484] {
            assert!(
                points[frame - 463].is_none_or(|p| p[0] < 0.83),
                "Bad ball at frame {frame} must not survive"
            );
        }
        // A hidden collision has incompatible velocities: do not bridge it
        // with a single straight line that would hide the actual Touch.
        assert!(points[(480 - 463) as usize].is_none());
    }

    #[test]
    fn player_supported_rebound_joins_two_flights_without_false_landmark() {
        let observed = (463..=495)
            .map(|f| {
                let t = f as f64;
                let [x, y] = if f == 475 || (477..=484).contains(&f) {
                    [1192.0, 420.0]
                } else if f <= 476 {
                    [820.0 + 6.0 * (t - 476.0), 407.0 + 15.0 * (t - 476.0)]
                } else {
                    [808.0 - 10.0 * (t - 485.0), 400.0 - 29.0 * (t - 485.0)]
                };
                Some([x / 1280.0, y / 720.0])
            })
            .collect::<Vec<_>>();
        let mut players = vec![Vec::new(); observed.len()];
        for frame in 479..=483 {
            players[frame - 463].push([
                817.0 / 1280.0,
                480.0 / 720.0,
                888.0 / 1280.0,
                710.0 / 720.0,
            ]);
        }
        let (filled, inferred) =
            repair_vballnet_trajectory_with_players(&observed, &players, 1280, 720);
        for frame in 477..=484 {
            let index = frame - 463;
            let pos = filled[index].expect("body-supported short rebound should be displayable");
            assert!(inferred[index]);
            assert!(
                pos[0] < 0.83,
                "Must never use the false right-side landmark"
            );
        }
        assert_eq!(filled[476 - 463], observed[476 - 463]);
        assert!(!inferred[476 - 463]);
        // A true flight reversal without any physically nearby player
        // remains unresolved, instead of manufacturing a new Touch.
        let (unconfirmed, _) = repair_vballnet_trajectory(&observed, 1280, 720);
        assert!(unconfirmed[480 - 463].is_none());
        assert!(observed[480 - 463].unwrap()[0] > 0.9);
    }

    #[test]
    fn realistic_rebound_not_suppressed() {
        let frames: Vec<_> = (0..25_u64)
            .map(|i| {
                sample(
                    i,
                    if i <= 12 {
                        0.3 + i as f64 * 0.004
                    } else {
                        0.348 - (i - 12) as f64 * 0.004
                    },
                )
            })
            .collect();
        let filtered = exclude_ball_teleports(&frames, 1280, 720);
        assert!(filtered.iter().all(|f| f.selected_ball.is_some()));
    }
    #[test]
    fn isolated_ball_at_player_after_stationary_right_edge_is_rejected() {
        // Real project 00:55.5: VBallNet stays at the right edge before and
        // after frame 1665, but momentarily selects a false ball by the player.
        // Even pose proximity must not convert this one-frame jump into Touch.
        let mut frames: Vec<_> = (0..48_u64).map(|i| sample(i, 0.932)).collect();
        let false_ball = frames[25].selected_ball.as_mut().unwrap();
        false_ball.bbox_xyxy_norm = [0.273, 0.895, 0.293, 0.931];
        let cleaned = exclude_ball_teleports(&frames, 1280, 720);
        assert!(cleaned[25].selected_ball.is_none());
        assert!(
            frames[25].selected_ball.is_some(),
            "GPU replay is immutable"
        );
    }
}
