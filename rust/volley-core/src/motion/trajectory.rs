use super::{BallObservation, MotionEvidence};

#[derive(Clone, Debug, serde::Serialize)]
pub struct TrajectoryBreak {
    pub frame: i64,
    pub position: [f64; 2],
    pub motion: MotionEvidence,
    pub fit_error: f64,
    pub before_frames: Vec<i64>,
    pub after_frames: Vec<i64>,
    pub contact_observed: bool,
}

pub fn trajectory_breaks_in_rallies(
    points: &[BallObservation],
    intervals: &[(i64, i64)],
) -> Vec<TrajectoryBreak> {
    if points.windows(2).any(|p| p[0].frame >= p[1].frame)
        || intervals.iter().any(|(a, b)| *a < 0 || a > b)
        || intervals.windows(2).any(|p| p[0].1 >= p[1].0)
    {
        return Vec::new();
    }
    intervals
        .iter()
        .flat_map(|&(start, end)| {
            let lo = points.partition_point(|p| p.frame < start);
            let hi = points.partition_point(|p| p.frame <= end);
            trajectory_breaks(&points[lo..hi])
        })
        .collect()
}

struct Flight {
    x: [f64; 3],
    y: [f64; 3],
    error: f64,
}

fn fit(points: &[BallObservation], at: i64) -> Option<Flight> {
    if points.len() < 4 {
        return None;
    }
    let mut matrix = [[0.0; 5]; 3];
    for p in points {
        let t = (p.frame - at) as f64;
        let basis = [1.0, t, t * t];
        for row in 0..3 {
            for column in 0..3 {
                matrix[row][column] += basis[row] * basis[column];
            }
            matrix[row][3] += basis[row] * p.x;
            matrix[row][4] += basis[row] * p.y;
        }
    }
    for column in 0..3 {
        let pivot = (column..3)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))?;
        matrix.swap(column, pivot);
        let scale = matrix[column][column];
        if scale.abs() < 1e-9 {
            return None;
        }
        for entry in &mut matrix[column][column..5] {
            *entry /= scale;
        }
        for row in 0..3 {
            if row == column {
                continue;
            }
            let scale = matrix[row][column];
            for j in column..5 {
                matrix[row][j] -= scale * matrix[column][j];
            }
        }
    }
    let x = [matrix[0][3], matrix[1][3], matrix[2][3]];
    let y = [matrix[0][4], matrix[1][4], matrix[2][4]];
    let error = (points
        .iter()
        .map(|p| {
            let t = (p.frame - at) as f64;
            (p.x - x[0] - x[1] * t - x[2] * t * t).powi(2)
                + (p.y - y[0] - y[1] * t - y[2] * t * t).powi(2)
        })
        .sum::<f64>()
        / points.len() as f64)
        .sqrt();
    Some(Flight { x, y, error })
}

pub fn trajectory_breaks(points: &[BallObservation]) -> Vec<TrajectoryBreak> {
    if points.len() < 9
        || points.windows(2).any(|p| p[1].frame <= p[0].frame)
        || points.iter().any(|p| {
            p.frame < 0 || p.frame > i32::MAX as i64 || !p.x.is_finite() || !p.y.is_finite()
        })
    {
        return Vec::new();
    }
    let mut found = Vec::<TrajectoryBreak>::new();
    let mut times = std::collections::BTreeSet::new();
    for i in 4..points.len() - 4 {
        times.insert(points[i].frame);
        let gap = points[i + 1].frame - points[i].frame;
        if (2..=4).contains(&gap) {
            times.insert(points[i].frame + gap / 2);
        }
    }
    for at in times {
        let left_end = points.partition_point(|p| p.frame < at);
        let right_begin = points.partition_point(|p| p.frame <= at);
        let left = &points[left_end.saturating_sub(7)..left_end];
        let right = &points[right_begin..(right_begin + 7).min(points.len())];
        if left.last().is_none_or(|p| at - p.frame > 3)
            || right.first().is_none_or(|p| p.frame - at > 3)
            || left.first().is_none_or(|p| at - p.frame > 12)
            || right.last().is_none_or(|p| p.frame - at > 12)
            || left
                .windows(2)
                .chain(right.windows(2))
                .any(|p| p[1].frame - p[0].frame > 4)
        {
            continue;
        }
        let (Some(before), Some(after)) = (fit(left, at), fit(right, at)) else {
            continue;
        };
        let error = before.error.max(after.error);
        if error > 4.0 {
            continue;
        }
        let join = (before.x[0] - after.x[0]).hypot(before.y[0] - after.y[0]);
        if join > 12.0 {
            continue;
        }
        let speed_before = before.x[1].hypot(before.y[1]);
        let speed_after = after.x[1].hypot(after.y[1]);
        let residual = (after.x[1] - before.x[1]).hypot(after.y[1] - before.y[1]);
        if speed_before.min(speed_after) < 2.0
            || speed_before.max(speed_after) > 60.0
            || residual
                < 8.0_f64
                    .max(0.45 * speed_before.max(speed_after))
                    .max(3.0 * error)
        {
            continue;
        }
        let joined: Vec<_> = left.iter().chain(right).copied().collect();
        let Some(smooth) = fit(&joined, at) else {
            continue;
        };
        if smooth.error < 2.0 || error > smooth.error * 0.65 {
            continue;
        }
        let angle = ((before.x[1] * after.x[1] + before.y[1] * after.y[1])
            / (speed_before * speed_after))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        let candidate = TrajectoryBreak {
            frame: at,
            position: [
                (before.x[0] + after.x[0]) * 0.5,
                (before.y[0] + after.y[0]) * 0.5,
            ],
            motion: MotionEvidence {
                motion_frame: at,
                speed_before,
                speed_after,
                angle,
                residual,
            },
            fit_error: error + join * 0.25,
            before_frames: left.iter().map(|p| p.frame).collect(),
            after_frames: right.iter().map(|p| p.frame).collect(),
            contact_observed: points.binary_search_by_key(&at, |p| p.frame).is_ok(),
        };
        if let Some(last) = found.last_mut().filter(|old| at - old.frame <= 6) {
            if candidate.motion.residual / (1.0 + candidate.fit_error)
                > last.motion.residual / (1.0 + last.fit_error)
            {
                *last = candidate;
            }
        } else {
            found.push(candidate);
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flight(change: bool) -> Vec<BallObservation> {
        (0..61)
            .map(|frame| {
                let t = frame as f64 - 30.0;
                let velocity = if change && t > 0.0 { -8.0 } else { 8.0 };
                BallObservation {
                    frame,
                    x: 400.0 + 2.0 * t,
                    y: 250.0 + velocity * t + 0.2 * t * t,
                }
            })
            .collect()
    }

    #[test]
    fn smooth_ballistic_apex_and_linear_interpolation_are_not_contacts() {
        assert!(trajectory_breaks(&flight(false)).is_empty());
        let line: Vec<_> = flight(false)
            .into_iter()
            .map(|mut p| {
                p.y = 100.0 + p.frame as f64;
                p
            })
            .collect();
        assert!(trajectory_breaks(&line).is_empty());
    }

    #[test]
    fn velocity_break_is_found_without_any_players() {
        let found = trajectory_breaks(&flight(true));
        assert_eq!(found.len(), 1, "{found:?}");
        assert!((found[0].frame - 30).abs() <= 1, "{found:?}");
    }

    #[test]
    fn rally_boundaries_and_scene_jumps_are_not_contacts() {
        assert!(trajectory_breaks_in_rallies(&flight(true), &[(0, 29), (30, 60)]).is_empty());
        let mut jump = flight(false);
        for p in &mut jump[30..] {
            p.x += 100.0;
        }
        assert!(trajectory_breaks(&jump).is_empty());
    }

    #[test]
    fn separated_contacts_are_not_merged_and_small_localization_noise_is_ignored() {
        let mut points = flight(false);
        for p in &mut points {
            let t = p.frame as f64;
            p.y = 200.0 + 8.0 * t - 16.0 * (t - 20.0).max(0.0)
                + 16.0 * (t - 40.0).max(0.0)
                + 0.1 * t * t;
        }
        let found = trajectory_breaks(&points);
        assert_eq!(found.len(), 2, "{found:?}");
        let mut smooth = flight(false);
        for p in &mut smooth {
            p.y += match p.frame % 5 {
                0 => 2.0,
                1 => -2.0,
                _ => 0.0,
            };
        }
        assert!(trajectory_breaks(&smooth).is_empty());
    }

    #[test]
    fn malformed_timeline_is_rejected() {
        let mut points = flight(true);
        points[30].frame = points[29].frame;
        assert!(trajectory_breaks(&points).is_empty());
        points[30].frame = 30;
        points[30].x = f64::NAN;
        assert!(trajectory_breaks(&points).is_empty());
    }

    #[test]
    fn isolated_coordinate_spike_does_not_become_a_contact() {
        let mut points = flight(false);
        points[30].y += 45.0;
        assert!(trajectory_breaks(&points).is_empty());
    }

    #[test]
    fn missing_sample_after_observed_contact_does_not_shift_or_duplicate_it() {
        let mut points = flight(true);
        points.retain(|p| p.frame != 31);
        let found = trajectory_breaks(&points);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!((found[0].frame - 30).abs() <= 1, "{found:?}");
    }

    #[test]
    fn short_missing_contact_sample_can_be_bridged_but_long_gap_cannot() {
        let mut points = flight(true);
        points.retain(|p| p.frame != 30);
        assert_eq!(trajectory_breaks(&points).len(), 1);
        points.retain(|p| !(25..=35).contains(&p.frame));
        assert!(trajectory_breaks(&points).is_empty());
    }

    #[test]
    fn irek_recorded_rebound_survives_a_noisy_contact_coordinate() {
        let positions: Vec<(i64, f64, f64)> =
            serde_json::from_str(include_str!("../../tests/fixtures/irek_rebound_33s.json"))
                .unwrap();
        let points: Vec<_> = positions
            .into_iter()
            .map(|(frame, x, y)| BallObservation { frame, x, y })
            .collect();
        let found = trajectory_breaks(&points);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!((found[0].frame - 992).abs() <= 2, "{found:?}");
    }
}
