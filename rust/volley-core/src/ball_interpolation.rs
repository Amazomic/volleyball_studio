//! Bounded VBallNet-only reconstruction of invisible portions of a flight.
//! Every inferred point is marked synthetic; it supports motion hypotheses, not observed contacts.

/// Maximum interior dropout (0.9 s at 30 fps); never invent an unanchored tail.
pub const MAX_MISSING_BALL_FRAMES: usize = 27;
const MAX_DISPLACEMENT_PER_FRAME: f64 = 0.12;
const MAX_ENDPOINT_VELOCITY_MISMATCH: f64 = 0.018;
const MAX_HORIZONTAL_ACCEL: f64 = 0.006;
const MAX_VERTICAL_ACCEL: f64 = 0.010;

/// Fit one continuous flight between real observations. The left/right
/// velocity estimates use only other *real* VBallNet observations.
/// Two-sided velocity agreement, nonnegative downward gravity and an entirely
/// in-frame curve are required when enough evidence exists. Missing endpoint
/// velocities fall back to a straight chord, never to unsupported model boxes or extrapolation
/// beyond the last real observation.
#[derive(Clone, Debug, Default)]
pub struct BallMotionReconstruction {
    pub observations: Vec<crate::motion::BallObservation>,
    pub rejected_frames: Vec<i64>,
    pub interpolated_frames: Vec<i64>,
}

pub fn reconstruct_short_ball_motion(
    observations: &[crate::motion::BallObservation],
    width: u32,
    height: u32,
    max_join_error_px: f64,
) -> Result<BallMotionReconstruction, &'static str> {
    if width < 64
        || height < 64
        || !max_join_error_px.is_finite()
        || max_join_error_px < 1.0
        || observations.windows(2).any(|w| w[1].frame <= w[0].frame)
        || observations.iter().any(|p| {
            p.frame < 0
                || !p.x.is_finite()
                || !p.y.is_finite()
                || p.x < 0.0
                || p.x > f64::from(width)
                || p.y < 0.0
                || p.y > f64::from(height)
        })
    {
        return Err("Invalid observed ball motion coordinates");
    }
    if observations.is_empty() {
        return Ok(BallMotionReconstruction::default());
    }
    let first = observations[0].frame;
    let last = observations[observations.len() - 1].frame;
    let size = usize::try_from(last - first + 1).map_err(|_| "Ball motion span overflow")?;
    if size > 1200 {
        return Err("Ball motion reconstruction exceeds bounded scan window");
    }
    let mut retained = Vec::with_capacity(observations.len());
    let mut rejected_frames = Vec::new();
    for (index, &point) in observations.iter().enumerate() {
        if index > 0 && index + 1 < observations.len() {
            let previous = observations[index - 1];
            let next = observations[index + 1];
            let span = next.frame - previous.frame;
            if span <= 4 {
                let fraction = (point.frame - previous.frame) as f64 / span as f64;
                let x = previous.x + fraction * (next.x - previous.x);
                let y = previous.y + fraction * (next.y - previous.y);
                let deviation = (point.x - x).hypot(point.y - y);
                let chord = (next.x - previous.x).hypot(next.y - previous.y);
                if deviation > (4.0 * max_join_error_px).max(3.0 * chord) {
                    rejected_frames.push(point.frame);
                    continue;
                }
            }
        }
        retained.push(point);
    }
    let mut points = vec![None; size];
    for point in retained {
        let index = (point.frame - first) as usize;
        points[index] = Some([point.x / f64::from(width), point.y / f64::from(height)]);
    }
    let synthetic = interpolate_short_gaps_bounded(&mut points, 2);
    let mut interpolated_frames = Vec::new();
    let observations = points
        .into_iter()
        .enumerate()
        .filter_map(|(index, point)| {
            let point = point?;
            if synthetic[index] {
                interpolated_frames.push(first + index as i64);
            }
            Some(crate::motion::BallObservation {
                frame: first + index as i64,
                x: point[0] * f64::from(width),
                y: point[1] * f64::from(height),
            })
        })
        .collect();
    Ok(BallMotionReconstruction {
        observations,
        rejected_frames,
        interpolated_frames,
    })
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(from = "BallStabilizerSettingsInput")]
pub struct BallStabilizerSettings {
    pub max_gap_frames: usize,
    pub max_step_fraction: f64,
}

#[derive(serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct BallStabilizerSettingsInput {
    #[serde(rename = "min_confidence")]
    _legacy_min_confidence: Option<f32>,
    max_gap_frames: usize,
    max_step_fraction: f64,
}

impl Default for BallStabilizerSettingsInput {
    fn default() -> Self {
        let settings = BallStabilizerSettings::default();
        Self {
            _legacy_min_confidence: None,
            max_gap_frames: settings.max_gap_frames,
            max_step_fraction: settings.max_step_fraction,
        }
    }
}

impl From<BallStabilizerSettingsInput> for BallStabilizerSettings {
    fn from(value: BallStabilizerSettingsInput) -> Self {
        Self {
            max_gap_frames: value.max_gap_frames,
            max_step_fraction: value.max_step_fraction,
        }
    }
}

impl Default for BallStabilizerSettings {
    fn default() -> Self {
        Self {
            max_gap_frames: 12,
            max_step_fraction: 0.12,
        }
    }
}

impl BallStabilizerSettings {
    pub fn validate(self) -> Result<(), &'static str> {
        if !(1..=MAX_MISSING_BALL_FRAMES).contains(&self.max_gap_frames)
            || !self.max_step_fraction.is_finite()
            || !(0.01..=0.25).contains(&self.max_step_fraction)
        {
            return Err("Invalid Ball Stabilizer settings");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ScoredBallSample {
    pub frame: usize,
    pub point: [f64; 2],
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub struct StabilizedBallMotion {
    pub points: Vec<Option<[f64; 2]>>,
    pub interpolated: Vec<bool>,
    pub rejected_frames: Vec<usize>,
    pub confidence_frames: usize,
    pub observed_frames: usize,
}

pub const BALL_STABILIZER_ALGORITHM_VERSION: &str = "ball_stabilizer_v16";

fn reject_preexisting_remote_ball(
    points: &[Option<[f64; 2]>],
    max_step_fraction: f64,
) -> Vec<usize> {
    const HISTORY_FRAMES: usize = 600;
    const HISTORY_RADIUS: f64 = 0.045;
    const EXCLUSION_RADIUS: f64 = 0.065;
    const EXCLUSION_TIMEOUT: usize = 90;

    let distance = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    let mut latest: Option<(usize, [f64; 2])> = None;
    let mut foreign: Option<(usize, [f64; 2])> = None;
    let mut rejected = Vec::new();
    for (frame, &candidate) in points.iter().enumerate() {
        let Some(point) = candidate else { continue };
        foreign = foreign.filter(|(last_seen, _)| frame - last_seen <= EXCLUSION_TIMEOUT);
        if let Some((_, center)) = foreign {
            if distance(point, center) <= EXCLUSION_RADIUS {
                let continuous_return = latest.is_some_and(|(last_frame, last_point)| {
                    let dt = frame - last_frame;
                    dt <= 3
                        && distance(last_point, center) <= EXCLUSION_RADIUS + 0.025
                        && distance(point, last_point) <= max_step_fraction * dt as f64
                });
                if !continuous_return {
                    rejected.push(frame);
                    foreign = Some((frame, center));
                    continue;
                }
                foreign = None;
            }
        }
        let impossible_switch = latest.is_some_and(|(last_frame, last_point)| {
            let dt = frame - last_frame;
            dt <= 4
                && distance(point, last_point) > max_step_fraction * dt as f64
                && (last_frame.saturating_sub(4)..last_frame)
                    .filter(|&prior_frame| {
                        points[prior_frame].is_some_and(|prior| {
                            distance(prior, last_point)
                                <= max_step_fraction * (last_frame - prior_frame) as f64
                        }) && rejected
                            .last()
                            .is_none_or(|&rejected_frame| rejected_frame < prior_frame)
                    })
                    .take(2)
                    .count()
                    == 2
        });
        if impossible_switch {
            let begin = frame.saturating_sub(HISTORY_FRAMES);
            let end = frame.saturating_sub(12);
            let mut visits: Vec<(usize, [f64; 2])> = Vec::new();
            let established = points[begin..end]
                .iter()
                .enumerate()
                .filter_map(|(offset, sample)| {
                    sample
                        .filter(|&prior| distance(prior, point) <= HISTORY_RADIUS)
                        .map(|prior| (begin + offset, prior))
                })
                .any(|(prior_frame, prior)| {
                    let repeated = visits.len() >= 2
                        && visits.iter().any(|&(earlier_frame, earlier)| {
                            prior_frame - earlier_frame >= 12 && distance(prior, earlier) <= 0.012
                        });
                    visits.push((prior_frame, prior));
                    repeated
                });
            if established {
                rejected.push(frame);
                foreign = Some((frame, point));
                continue;
            }
        }
        latest = Some((frame, point));
    }
    rejected
}

fn supports_moving_source_after_idle(
    raw_points: &[Option<[f64; 2]>],
    points: &[Option<[f64; 2]>],
    frame: usize,
    last_frame: usize,
    last_point: [f64; 2],
    max_step: f64,
) -> bool {
    const IDLE_HISTORY: usize = 30;
    const FLIGHT_SUPPORT: usize = 12;
    const IDLE_RADIUS: f64 = 0.025;
    let distance = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    if last_frame < IDLE_HISTORY - 1 || frame - last_frame > 4 {
        return false;
    }
    let history = &points[last_frame + 1 - IDLE_HISTORY..=last_frame];
    if history.iter().flatten().count() < 24
        || history
            .iter()
            .flatten()
            .any(|&point| distance(point, last_point) > IDLE_RADIUS)
    {
        return false;
    }
    let Some(flight) = points.get(frame..frame + FLIGHT_SUPPORT) else {
        return false;
    };
    let Some(first) = flight[0] else { return false };
    let Some(last) = flight[FLIGHT_SUPPORT - 1] else {
        return false;
    };
    let displacement = distance(first, last);
    if displacement < 0.045 || distance(first, last_point) <= max_step * (frame - last_frame) as f64
    {
        return false;
    }
    let direction = [
        (last[0] - first[0]) / displacement,
        (last[1] - first[1]) / displacement,
    ];
    let mut previous = first;
    let mut forward_steps = 0;
    for &candidate in &flight[1..] {
        let Some(point) = candidate else { return false };
        if distance(point, previous) > max_step || distance(point, last_point) <= IDLE_RADIUS * 2.0
        {
            return false;
        }
        let progress =
            (point[0] - previous[0]) * direction[0] + (point[1] - previous[1]) * direction[1];
        if progress < -0.006 {
            return false;
        }
        forward_steps += usize::from(progress > 0.001);
        previous = point;
    }
    forward_steps >= 8
        && !raw_points[frame + FLIGHT_SUPPORT..(frame + 30).min(raw_points.len())]
            .iter()
            .flatten()
            .any(|&point| distance(point, last_point) <= IDLE_RADIUS * 2.0)
}
fn reject_unlinked_motion(
    raw_points: &[Option<[f64; 2]>],
    points: &mut [Option<[f64; 2]>],
    settings: BallStabilizerSettings,
) -> Vec<usize> {
    let mut rejected = Vec::new();
    let mut previous: Option<(usize, [f64; 2])> = None;
    let mut latest: Option<(usize, [f64; 2])> = None;
    for frame in 0..points.len() {
        let Some(point) = points[frame] else {
            continue;
        };
        let Some((last_frame, last_point)) = latest else {
            latest = Some((frame, point));
            continue;
        };
        let delta_frames = frame - last_frame;
        if delta_frames > settings.max_gap_frames + 1 {
            previous = None;
            latest = Some((frame, point));
            continue;
        }
        let velocity = previous
            .filter(|(previous_frame, _)| last_frame - previous_frame <= 4)
            .map(|(previous_frame, previous_point)| {
                let dt = (last_frame - previous_frame) as f64;
                [
                    (last_point[0] - previous_point[0]) / dt,
                    (last_point[1] - previous_point[1]) / dt,
                ]
            });
        let dt = delta_frames as f64;
        let predicted = velocity.map_or(last_point, |v| {
            [last_point[0] + v[0] * dt, last_point[1] + v[1] * dt]
        });
        let limit = if let Some(v) = velocity {
            (0.012
                + settings.max_step_fraction * 0.30
                + v[0].hypot(v[1]) * 0.65
                + settings.max_step_fraction * 0.08 * (dt.sqrt() - 1.0))
                .min(settings.max_step_fraction * 0.80)
        } else {
            settings.max_step_fraction * dt
        };
        let deviation = (point[0] - predicted[0]).hypot(point[1] - predicted[1]);
        if deviation > limit {
            let plausible_step = (point[0] - last_point[0]).hypot(point[1] - last_point[1])
                <= settings.max_step_fraction * dt;
            let supported_turn = velocity.is_some_and(|old_velocity| {
                if !plausible_step {
                    return false;
                }
                let new_velocity = [
                    (point[0] - last_point[0]) / dt,
                    (point[1] - last_point[1]) / dt,
                ];
                let mut confirmations = 0;
                for (future_frame, future) in points
                    .iter()
                    .enumerate()
                    .take((frame + 4).min(points.len()))
                    .skip(frame + 1)
                {
                    let Some(future) = future else {
                        continue;
                    };
                    let elapsed = (future_frame - frame) as f64;
                    let expected = [
                        point[0] + new_velocity[0] * elapsed,
                        point[1] + new_velocity[1] * elapsed,
                    ];
                    if (future[0] - expected[0]).hypot(future[1] - expected[1]) <= limit {
                        confirmations += 1;
                    }
                }
                if confirmations < 2 {
                    return false;
                }
                ((frame + 1)..(frame + settings.max_gap_frames + 2).min(points.len()))
                    .filter(|&candidate_frame| {
                        points[candidate_frame].is_some_and(|future| {
                            let offset = (candidate_frame - last_frame) as f64;
                            let old = [
                                last_point[0] + old_velocity[0] * offset,
                                last_point[1] + old_velocity[1] * offset,
                            ];
                            (future[0] - old[0]).hypot(future[1] - old[1]) <= limit * 1.35
                        })
                    })
                    .take(2)
                    .count()
                    < 2
            });
            if !supported_turn {
                if supports_moving_source_after_idle(
                    raw_points,
                    points,
                    frame,
                    last_frame,
                    last_point,
                    settings.max_step_fraction,
                ) {
                    let mut idle_end = last_frame;
                    for idle_frame in (0..=last_frame).rev() {
                        let Some(idle_point) = points[idle_frame] else {
                            if idle_end - idle_frame > settings.max_gap_frames {
                                break;
                            }
                            continue;
                        };
                        if (idle_point[0] - last_point[0]).hypot(idle_point[1] - last_point[1])
                            > 0.025
                        {
                            break;
                        }
                        points[idle_frame] = None;
                        rejected.push(idle_frame);
                        idle_end = idle_frame;
                    }
                    previous = None;
                    latest = Some((frame, point));
                    continue;
                }
                points[frame] = None;
                rejected.push(frame);
                continue;
            }
        }
        previous = latest;
        latest = Some((frame, point));
    }
    rejected
}

fn reject_interleaved_outliers(points: &[Option<[f64; 2]>], max_step_fraction: f64) -> Vec<usize> {
    let mut rejected = Vec::new();
    let support_radius = max_step_fraction * 1.5;
    let outlier_radius = max_step_fraction * 1.75;
    for (index, point) in points.iter().enumerate() {
        let Some(point) = point else {
            continue;
        };
        let begin = index.saturating_sub(8);
        let end = (index + 9).min(points.len());
        let neighbors = points[begin..end]
            .iter()
            .enumerate()
            .filter_map(|(offset, value)| {
                let frame = begin + offset;
                (frame != index).then_some((frame, value.as_ref()?))
            })
            .collect::<Vec<_>>();
        if neighbors.len() < 8 {
            continue;
        }
        let mut xs = neighbors.iter().map(|(_, p)| p[0]).collect::<Vec<_>>();
        let mut ys = neighbors.iter().map(|(_, p)| p[1]).collect::<Vec<_>>();
        xs.sort_by(f64::total_cmp);
        ys.sort_by(f64::total_cmp);
        let mid = xs.len() / 2;
        let center = if xs.len() % 2 == 0 {
            [(xs[mid - 1] + xs[mid]) * 0.5, (ys[mid - 1] + ys[mid]) * 0.5]
        } else {
            [xs[mid], ys[mid]]
        };
        let mut before = 0;
        let mut after = 0;
        for &(frame, candidate) in &neighbors {
            if (candidate[0] - center[0]).hypot(candidate[1] - center[1]) <= support_radius {
                if frame < index {
                    before += 1;
                } else {
                    after += 1;
                }
            }
        }
        if before >= 3
            && after >= 3
            && before + after >= 8
            && (point[0] - center[0]).hypot(point[1] - center[1]) > outlier_radius
        {
            rejected.push(index);
        }
    }
    rejected
}

fn reject_recurring_static_hotspots(
    points: &[Option<[f64; 2]>],
    max_step_fraction: f64,
) -> Vec<usize> {
    #[derive(Default)]
    struct Cell {
        count: usize,
        episodes: usize,
        first: usize,
        last: usize,
        sum: [f64; 2],
        square_sum: [f64; 2],
        min: [f64; 2],
        max: [f64; 2],
    }

    let mut cells = std::collections::HashMap::<(u8, u8), Cell>::new();
    for (frame, point) in points.iter().enumerate() {
        let Some(point) = point else {
            continue;
        };
        let key = (
            (point[0] * 40.0).round() as u8,
            (point[1] * 40.0).round() as u8,
        );
        let cell = cells.entry(key).or_default();
        if cell.count == 0 {
            cell.first = frame;
            cell.episodes = 1;
            cell.min = *point;
            cell.max = *point;
        } else if frame - cell.last > 120 {
            cell.episodes += 1;
        }
        cell.last = frame;
        cell.count += 1;
        for (axis, &coordinate) in point.iter().enumerate() {
            cell.sum[axis] += coordinate;
            cell.square_sum[axis] += coordinate * coordinate;
            cell.min[axis] = cell.min[axis].min(coordinate);
            cell.max[axis] = cell.max[axis].max(coordinate);
        }
    }

    let hotspots = cells
        .values()
        .filter_map(|cell| {
            if cell.count < 40 || cell.episodes < 3 || cell.last - cell.first < 300 {
                return None;
            }
            let n = cell.count as f64;
            let center = [cell.sum[0] / n, cell.sum[1] / n];
            let spread = [
                (cell.square_sum[0] / n - center[0] * center[0])
                    .max(0.0)
                    .sqrt(),
                (cell.square_sum[1] / n - center[1] * center[1])
                    .max(0.0)
                    .sqrt(),
            ];
            (spread[0] <= 0.0065
                && spread[1] <= 0.0065
                && cell.max[0] - cell.min[0] <= 0.021
                && cell.max[1] - cell.min[1] <= 0.024)
                .then_some(center)
        })
        .collect::<Vec<_>>();

    let mut rejected = Vec::new();
    for (frame, point) in points.iter().enumerate() {
        let Some(point) = point else {
            continue;
        };
        for center in &hotspots {
            if (point[0] - center[0]).hypot(point[1] - center[1]) > 0.025 {
                continue;
            }
            let moving_neighbor = |neighbor: usize| {
                points[neighbor].is_some_and(|other| {
                    let distance = (point[0] - other[0]).hypot(point[1] - other[1]);
                    distance > 0.018
                        && distance <= max_step_fraction * frame.abs_diff(neighbor) as f64
                })
            };
            let moving_before = (frame.saturating_sub(2)..frame).any(moving_neighbor);
            let moving_after = ((frame + 1)..(frame + 3).min(points.len())).any(moving_neighbor);
            if !(moving_before && moving_after) {
                rejected.push(frame);
            }
            break;
        }
    }
    rejected
}

fn restore_supported_vertical_rebounds(
    raw: &[Option<[f64; 2]>],
    points: &mut [Option<[f64; 2]>],
    max_step: f64,
) {
    let distance = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    for contact in 4..points.len().saturating_sub(8) {
        if points[contact].is_some() {
            continue;
        }
        let (Some(impact), Some(before), Some(earlier)) =
            (raw[contact], points[contact - 1], points[contact - 4])
        else {
            continue;
        };
        if points[contact - 4..contact].iter().any(Option::is_none)
            || before[1] - earlier[1] < 0.065
            || impact[1] - before[1] < 0.015
            || distance(impact, before) > max_step
        {
            continue;
        }
        for start in contact + 1..=contact + 3 {
            let (Some(first), Some(second), Some(third)) =
                (raw[start], raw[start + 1], raw[start + 2])
            else {
                continue;
            };
            let velocity = [(third[0] - first[0]) * 0.5, (third[1] - first[1]) * 0.5];
            let elapsed = (start - contact) as f64;
            let origin = [
                first[0] - velocity[0] * elapsed,
                first[1] - velocity[1] * elapsed,
            ];
            if velocity[1] > -0.025
                || first[1] >= impact[1] - 0.05
                || second[1] >= first[1] - 0.015
                || third[1] >= second[1] - 0.015
                || distance(first, second).max(distance(second, third)) > max_step
                || distance(
                    second,
                    [(first[0] + third[0]) * 0.5, (first[1] + third[1]) * 0.5],
                ) > 0.018
                || distance(impact, first) > max_step * elapsed
                || distance(impact, origin) > 0.065
            {
                continue;
            }
            let mut branch = vec![(start, first), (start + 1, second), (start + 2, third)];
            let mut anchor = None;
            for (frame, candidate) in raw
                .iter()
                .enumerate()
                .take((contact + 25).min(raw.len()))
                .skip(start + 3)
            {
                let (last_frame, last) = branch[branch.len() - 1];
                let (previous_frame, previous) = branch[branch.len() - 2];
                let gap = frame - last_frame;
                if gap > 4 {
                    break;
                }
                let Some(point) = *candidate else { continue };
                let dt = gap as f64;
                let prior_dt = (last_frame - previous_frame) as f64;
                let predicted = [
                    last[0] + (last[0] - previous[0]) / prior_dt * dt,
                    last[1] + (last[1] - previous[1]) / prior_dt * dt,
                ];
                if point[1] > last[1] + 0.004
                    || distance(last, point) > max_step * dt
                    || distance(predicted, point) > 0.018 + 0.004 * dt * dt
                {
                    continue;
                }
                branch.push((frame, point));
                if branch.len() >= 5 && points[frame] == Some(point) {
                    anchor = Some(frame);
                    break;
                }
            }
            let Some(end) = anchor else { continue };
            points[contact..=end].fill(None);
            points[contact] = Some(impact);
            for (frame, point) in branch {
                points[frame] = Some(point);
            }
            break;
        }
    }
}

fn reject_unsupported_weak_branches(
    points: &mut [Option<[f64; 2]>],
    scores: &[f32],
    settings: BallStabilizerSettings,
) {
    const ANCHOR_SCORE: f32 = 0.35;
    const ANCHOR_CONTEXT: usize = 90;
    const EDGE_SUPPORT: usize = 4;
    let reliable: Vec<_> = (0..points.len())
        .filter_map(|frame| {
            let point = points[frame]?;
            (scores[frame] >= ANCHOR_SCORE
                && (frame.saturating_sub(4)..(frame + 5).min(points.len()))
                    .filter(|&neighbor| {
                        points[neighbor].is_some() && scores[neighbor] >= ANCHOR_SCORE
                    })
                    .take(3)
                    .count()
                    == 3)
                .then_some((frame, point))
        })
        .collect();
    let mut next = 0;
    for (frame, slot) in points.iter_mut().enumerate() {
        while next < reliable.len() && reliable[next].0 < frame {
            next += 1;
        }
        let Some(point) = *slot else { continue };
        if scores[frame] >= ANCHOR_SCORE {
            continue;
        }
        let before = next.checked_sub(1).map(|i| reliable[i]);
        let after = reliable.get(next).copied();
        let distance = |(i, anchor): (usize, [f64; 2])| {
            let elapsed = frame.abs_diff(i);
            let reachable = (point[0] - anchor[0]).hypot(point[1] - anchor[1])
                <= settings.max_step_fraction * elapsed as f64;
            (elapsed, reachable)
        };
        let left = before.map(distance);
        let right = after.map(distance);
        let nearest = left
            .into_iter()
            .chain(right)
            .map(|(elapsed, _)| elapsed)
            .min()
            .unwrap_or(usize::MAX);
        let bounded = left
            .zip(right)
            .is_some_and(|((l, a), (r, b))| a && b && l + r <= settings.max_gap_frames + 1);
        let edge = left
            .into_iter()
            .chain(right)
            .any(|(elapsed, reachable)| reachable && elapsed <= EDGE_SUPPORT);
        if nearest <= ANCHOR_CONTEXT && !edge && !bounded {
            *slot = None;
        }
    }
}

pub fn stabilize_ball_motion(
    samples: &[ScoredBallSample],
    frames: usize,
    settings: BallStabilizerSettings,
) -> Result<StabilizedBallMotion, &'static str> {
    settings.validate()?;
    if frames == 0 || frames > 48 * 3600 * 30 {
        return Err("Invalid Ball Stabilizer frame count");
    }
    let mut points = vec![None; frames];
    let mut scores = vec![f32::NEG_INFINITY; frames];
    for sample in samples {
        if sample.frame >= frames
            || !sample.confidence.is_finite()
            || sample
                .point
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err("Invalid Ball Stabilizer sample");
        }
        if sample.confidence >= scores[sample.frame] {
            points[sample.frame] = Some(sample.point);
            scores[sample.frame] = sample.confidence;
        }
    }
    let confidence_frames = points.iter().filter(|p| p.is_some()).count();
    let raw_points = points.clone();
    let mut rejected_frames = reject_preexisting_remote_ball(&points, settings.max_step_fraction);
    for &index in &rejected_frames {
        points[index] = None;
    }
    rejected_frames.extend(reject_recurring_static_hotspots(
        &points,
        settings.max_step_fraction,
    ));
    for &index in &rejected_frames {
        points[index] = None;
    }
    for index in 0..frames {
        let Some(point) = points[index] else {
            continue;
        };
        let start = index.saturating_sub(4);
        let end = (index + 5).min(frames);
        let mut support = 0;
        for (neighbor, &candidate) in points.iter().enumerate().take(end).skip(start) {
            if neighbor == index {
                continue;
            }
            if let Some(other) = candidate {
                let limit = settings.max_step_fraction * index.abs_diff(neighbor) as f64;
                if (point[0] - other[0]).hypot(point[1] - other[1]) <= limit {
                    support += 1;
                    if support == 2 {
                        break;
                    }
                }
            }
        }
        if support < 2 {
            rejected_frames.push(index);
        }
    }
    for &index in &rejected_frames {
        points[index] = None;
    }
    for index in reject_interleaved_outliers(&points, settings.max_step_fraction) {
        points[index] = None;
        rejected_frames.push(index);
    }
    let mut segments = Vec::<(usize, usize, usize)>::new();
    for index in 0..frames {
        if points[index].is_none() {
            continue;
        }
        if let Some(last) = segments.last_mut() {
            let gap = index - last.1;
            let prev = points[last.1].expect("segment end is observed");
            let current = points[index].expect("anchor is observed");
            if gap <= 4
                && (prev[0] - current[0]).hypot(prev[1] - current[1])
                    <= settings.max_step_fraction * gap as f64
            {
                last.1 = index;
                last.2 += 1;
                continue;
            }
        }
        segments.push((index, index, 1));
    }
    let mut rejected_islands = Vec::new();
    for triple in segments.windows(3) {
        let (_, before_end, before_count) = triple[0];
        let (island_start, island_end, island_count) = triple[1];
        let (after_start, _, after_count) = triple[2];
        if after_count < 4
            || island_count > 4
            || after_start - before_end > settings.max_gap_frames + 5
        {
            continue;
        }
        let before = points[before_end].expect("segment end is observed");
        let after = points[after_start].expect("segment start is observed");
        let island_first = points[island_start].expect("island start is observed");
        let island_last = points[island_end].expect("island end is observed");
        let bounded_static_island = before_count >= 1
            && (2..=4).contains(&island_count)
            && island_end - island_start <= 4
            && island_start - before_end <= 2
            && after_start - island_end <= 2
            && (before[0] - island_first[0]).hypot(before[1] - island_first[1])
                > settings.max_step_fraction * (island_start - before_end) as f64
            && (island_last[0] - after[0]).hypot(island_last[1] - after[1])
                > settings.max_step_fraction * (after_start - island_end) as f64
            && (island_first[0] - island_last[0]).hypot(island_first[1] - island_last[1])
                <= settings.max_step_fraction * 0.25;
        if (before_count >= 4 || bounded_static_island)
            && (before[0] - after[0]).hypot(before[1] - after[1])
                <= settings.max_step_fraction * (after_start - before_end) as f64
        {
            rejected_islands.push((island_start, island_end));
        }
    }
    for (island_start, island_end) in rejected_islands {
        for (index, slot) in points
            .iter_mut()
            .enumerate()
            .take(island_end + 1)
            .skip(island_start)
        {
            if slot.take().is_some() {
                rejected_frames.push(index);
            }
        }
    }
    reject_unsupported_weak_branches(&mut points, &scores, settings);
    reject_unlinked_motion(&raw_points, &mut points, settings);
    restore_supported_vertical_rebounds(&raw_points, &mut points, settings.max_step_fraction);
    let weak_repairs = reject_weak_rebound_islands(&mut points, &scores, settings.max_gap_frames);
    reject_unsupported_weak_branches(&mut points, &scores, settings);
    let rejected_frames: Vec<_> = raw_points
        .iter()
        .zip(&points)
        .enumerate()
        .filter_map(|(frame, (raw, point))| (raw.is_some() && point.is_none()).then_some(frame))
        .collect();
    let observed_frames = confidence_frames - rejected_frames.len();
    let mut interpolated = interpolate_short_gaps_bounded(&mut points, settings.max_gap_frames);
    for (start, estimates) in weak_repairs {
        for (offset, point) in estimates.into_iter().enumerate() {
            points[start + offset] = Some(point);
            interpolated[start + offset] = true;
        }
    }
    interpolate_stabilizer_rebounds(&mut points, &mut interpolated, settings.max_gap_frames);
    Ok(StabilizedBallMotion {
        points,
        interpolated,
        rejected_frames,
        confidence_frames,
        observed_frames,
    })
}

pub fn interpolate_short_gaps(points: &mut [Option<[f64; 2]>]) -> Vec<bool> {
    interpolate_short_gaps_bounded(points, MAX_MISSING_BALL_FRAMES)
}

fn reject_weak_rebound_islands(
    points: &mut [Option<[f64; 2]>],
    scores: &[f32],
    max_missing_frames: usize,
) -> Vec<(usize, Vec<[f64; 2]>)> {
    let mut repairs = Vec::new();
    let synthetic = vec![false; points.len()];
    let mut left = 3;
    while left + 6 < points.len() {
        let start = left + 1;
        if points[left - 3..=left].iter().any(Option::is_none)
            || scores[left - 3..=left]
                .iter()
                .filter(|&&s| s >= 0.5)
                .count()
                < 2
            || scores[start] >= 0.45
        {
            left += 1;
            continue;
        }
        let max_right = (left + 1 + max_missing_frames.min(12)).min(points.len() - 4);
        let mut repair = None;
        for right in left + 4..=max_right {
            if scores[start..right].iter().any(|&s| s >= 0.45) {
                break;
            }
            if points[right..right + 4].iter().any(Option::is_none)
                || scores[right..right + 4]
                    .iter()
                    .filter(|&&s| s >= 0.5)
                    .count()
                    < 2
            {
                continue;
            }
            let Some(estimates) = reconstruct_occluded_rebound(
                points,
                left,
                right,
                &synthetic,
                max_missing_frames,
                true,
            ) else {
                continue;
            };
            let mut contradictions = 0;
            let incompatible =
                points[start..right]
                    .iter()
                    .zip(&estimates)
                    .all(|(point, estimate)| {
                        let Some(point) = point else { return true };
                        let contradicts = estimate[1] - point[1] >= 0.014;
                        contradictions += usize::from(contradicts);
                        contradicts
                    });
            if incompatible && contradictions >= 3 {
                repair = Some((right, estimates));
                break;
            }
        }
        if let Some((right, estimates)) = repair {
            points[start..right].fill(None);
            repairs.push((start, estimates));
            left = right;
        } else {
            left += 1;
        }
    }
    repairs
}

fn reconstruct_occluded_rebound(
    points: &[Option<[f64; 2]>],
    left: usize,
    right: usize,
    synthetic: &[bool],
    max_missing_frames: usize,
    weak_island: bool,
) -> Option<Vec<[f64; 2]>> {
    let missing = right.checked_sub(left + 1)?;
    if !(1..=12.min(max_missing_frames)).contains(&missing)
        || left < 2
        || right + 2 >= points.len()
        || synthetic[left - 1]
        || synthetic[left - 2]
        || synthetic[right + 1]
        || synthetic[right + 2]
    {
        return None;
    }
    let begin = points[left]?;
    let end = points[right]?;
    let before = points[left - 1]?;
    let before_prev = points[left - 2]?;
    let after = points[right + 1]?;
    let after_next = points[right + 2]?;
    let incoming_step = [begin[0] - before[0], begin[1] - before[1]];
    let incoming_previous = [before[0] - before_prev[0], before[1] - before_prev[1]];
    let outgoing_step = [after[0] - end[0], after[1] - end[1]];
    let outgoing_next = [after_next[0] - after[0], after_next[1] - after[1]];
    let incoming = [
        (begin[0] - before_prev[0]) * 0.5,
        (begin[1] - before_prev[1]) * 0.5,
    ];
    let outgoing = [
        (after_next[0] - end[0]) * 0.5,
        (after_next[1] - end[1]) * 0.5,
    ];
    let norm = |v: [f64; 2]| v[0].hypot(v[1]);
    let speed_in = norm(incoming);
    let speed_out = norm(outgoing);
    if speed_in < if weak_island { 0.009 } else { 0.012 }
        || speed_out < if weak_island { 0.004 } else { 0.012 }
        || speed_in > MAX_DISPLACEMENT_PER_FRAME
        || speed_out > MAX_DISPLACEMENT_PER_FRAME
        || norm([
            incoming_step[0] - incoming_previous[0],
            incoming_step[1] - incoming_previous[1],
        ]) > 0.012
        || norm([
            outgoing_step[0] - outgoing_next[0],
            outgoing_step[1] - outgoing_next[1],
        ]) > 0.012
    {
        return None;
    }
    let cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (speed_in * speed_out);
    if cosine > 0.15
        || norm([outgoing[0] - incoming[0], outgoing[1] - incoming[1]])
            < if weak_island { 0.015 } else { 0.028 }
        || (weak_island && (incoming[1] < 0.009 || outgoing[1] > -0.002))
        || (incoming[1] < -0.004 && outgoing[1] > 0.004)
    {
        return None;
    }
    let duration = (right - left) as f64;
    let mut closest: Option<(f64, usize, [f64; 2])> = None;
    for impact in left..right {
        let t = (impact - left) as f64;
        let predicted_in = [begin[0] + incoming[0] * t, begin[1] + incoming[1] * t];
        let predicted_out = [
            end[0] - outgoing[0] * (duration - t),
            end[1] - outgoing[1] * (duration - t),
        ];
        let separation = norm([
            predicted_in[0] - predicted_out[0],
            predicted_in[1] - predicted_out[1],
        ]);
        if (predicted_in[0] - predicted_out[0]).abs() > if weak_island { 0.012 } else { 0.024 }
            || separation > if weak_island { 0.025 } else { 0.055 }
        {
            continue;
        }
        let joint = if impact == left {
            begin
        } else {
            [
                (predicted_in[0] + predicted_out[0]) * 0.5,
                (predicted_in[1] + predicted_out[1]) * 0.5,
            ]
        };
        if !valid_point(joint) {
            continue;
        }
        if closest.is_none_or(|(error, _, _)| separation < error) {
            closest = Some((separation, impact, joint));
        }
    }
    let (_, impact, joint) = closest?;
    let slope_after = [
        (end[0] - joint[0]) / (right - impact) as f64,
        (end[1] - joint[1]) / (right - impact) as f64,
    ];
    if norm([slope_after[0] - outgoing[0], slope_after[1] - outgoing[1]]) > 0.025
        || norm(slope_after) > MAX_DISPLACEMENT_PER_FRAME
    {
        return None;
    }
    let slope_before = if impact > left {
        [
            (joint[0] - begin[0]) / (impact - left) as f64,
            (joint[1] - begin[1]) / (impact - left) as f64,
        ]
    } else {
        incoming
    };
    if norm(slope_before) > MAX_DISPLACEMENT_PER_FRAME {
        return None;
    }
    let estimates = (left + 1..right)
        .map(|frame| {
            if frame <= impact {
                let t = (frame - left) as f64;
                [
                    begin[0] + slope_before[0] * t,
                    begin[1] + slope_before[1] * t,
                ]
            } else {
                let t = (frame - impact) as f64;
                [joint[0] + slope_after[0] * t, joint[1] + slope_after[1] * t]
            }
        })
        .collect::<Vec<_>>();
    estimates
        .iter()
        .all(|point| valid_point(*point))
        .then_some(estimates)
}

fn interpolate_stabilizer_rebounds(
    points: &mut [Option<[f64; 2]>],
    synthetic: &mut [bool],
    max_missing_frames: usize,
) {
    let mut previous: Option<usize> = None;
    for right in 0..points.len() {
        if points[right].is_none() {
            continue;
        }
        if let Some(left) = previous {
            if !synthetic[left] && !synthetic[right] {
                if let Some(estimates) = reconstruct_occluded_rebound(
                    points,
                    left,
                    right,
                    synthetic,
                    max_missing_frames,
                    false,
                ) {
                    for (offset, point) in estimates.into_iter().enumerate() {
                        let frame = left + offset + 1;
                        points[frame] = Some(point);
                        synthetic[frame] = true;
                    }
                }
            }
        }
        previous = Some(right);
    }
}

fn interpolate_short_gaps_bounded(
    points: &mut [Option<[f64; 2]>],
    max_missing_frames: usize,
) -> Vec<bool> {
    let mut synthetic = vec![false; points.len()];
    let mut previous: Option<usize> = None;
    for right in 0..points.len() {
        let Some(end) = points[right].filter(|p| valid_point(*p)) else {
            if points[right].is_some() {
                previous = None;
            }
            continue;
        };
        if let Some(left) = previous {
            let missing = right - left - 1;
            if (1..=max_missing_frames.min(MAX_MISSING_BALL_FRAMES)).contains(&missing) {
                let begin = points[left].expect("observed left anchor");
                let duration = (right - left) as f64;
                let delta = [end[0] - begin[0], end[1] - begin[1]];
                if delta[0].hypot(delta[1]) <= MAX_DISPLACEMENT_PER_FRAME * duration {
                    // Velocity at each edge is measured outside the gap only.
                    let left_velocity = left
                        .checked_sub(1)
                        .filter(|&i| !synthetic[i])
                        .and_then(|i| points[i].filter(|p| valid_point(*p)))
                        .map(|p| [begin[0] - p[0], begin[1] - p[1]]);
                    let right_velocity = (right + 1 < points.len())
                        .then_some(right + 1)
                        .and_then(|i| points[i].filter(|p| valid_point(*p)))
                        .map(|p| [p[0] - end[0], p[1] - end[1]]);
                    // Fit x(t), y(t) = P_left + v_left*t + 0.5*a*t^2,
                    // exactly constrained by the other observed endpoint.
                    let v_left =
                        left_velocity.unwrap_or([delta[0] / duration, delta[1] / duration]);
                    let acceleration = [
                        2.0 * (delta[0] - v_left[0] * duration) / duration.powi(2),
                        2.0 * (delta[1] - v_left[1] * duration) / duration.powi(2),
                    ];
                    // Long gaps need observed velocities from *both* sides; a
                    // chord alone does not determine the flight parabola.
                    let enough_anchors =
                        missing <= 12 || (left_velocity.is_some() && right_velocity.is_some());
                    let compatible = enough_anchors
                        && acceleration[0].abs() <= MAX_HORIZONTAL_ACCEL
                        && acceleration[1] >= -0.0015
                        && acceleration[1] <= MAX_VERTICAL_ACCEL
                        && right_velocity.is_none_or(|v| {
                            (v[0] - v_left[0] - acceleration[0] * duration)
                                .hypot(v[1] - v_left[1] - acceleration[1] * duration)
                                <= MAX_ENDPOINT_VELOCITY_MISMATCH
                        });
                    if compatible {
                        let estimates: Vec<_> = (1..=missing)
                            .map(|offset| {
                                let t = offset as f64;
                                [
                                    begin[0] + v_left[0] * t + 0.5 * acceleration[0] * t * t,
                                    begin[1] + v_left[1] * t + 0.5 * acceleration[1] * t * t,
                                ]
                            })
                            .collect();
                        if estimates.iter().all(|p| valid_point(*p)) {
                            for (offset, point) in estimates.into_iter().enumerate() {
                                let index = left + offset + 1;
                                points[index] = Some(point);
                                synthetic[index] = true;
                            }
                        }
                    }
                }
            }
        }
        previous = Some(right);
    }
    synthetic
}

/// Display-only reconstruction of a short gap that may contain a player
/// contact. Unlike one ballistic parabola, the two observed flights are
/// joined at a single velocity-changing kink within a tracked player's body
/// reach. Neither the interpolated samples nor this candidate impact frame
/// are authoritative Touch evidence.
pub fn interpolate_player_rebound_gaps(
    points: &mut [Option<[f64; 2]>],
    synthetic: &mut [bool],
    players: &[Vec<[f64; 4]>],
    width: u32,
    height: u32,
) {
    if points.len() < 6
        || points.len() != synthetic.len()
        || points.len() != players.len()
        || width < 64
        || height < 64
    {
        return;
    }
    let w = f64::from(width);
    let h = f64::from(height);
    let mut left = 0;
    while left + 2 < points.len() {
        let Some(begin) = points[left].filter(|p| valid_point(*p)) else {
            left += 1;
            continue;
        };
        let mut right = left + 1;
        while right < points.len() && points[right].is_none() {
            right += 1;
        }
        if right >= points.len() {
            break;
        }
        let missing = right - left - 1;
        if !(2..=MAX_MISSING_BALL_FRAMES).contains(&missing) {
            left = right;
            continue;
        }
        let Some(end) = points[right].filter(|p| valid_point(*p)) else {
            left = right;
            continue;
        };
        // Both velocities come from real VBallNet samples only, not from
        // earlier interpolation or a single questionable position.
        let previous = (left.saturating_sub(3)..left)
            .rev()
            .find(|&i| !synthetic[i] && points[i].is_some());
        let next = (right + 1..(right + 4).min(points.len()))
            .find(|&i| !synthetic[i] && points[i].is_some());
        let (Some(previous), Some(next)) = (previous, next) else {
            left = right;
            continue;
        };
        let Some(before) = points[previous] else {
            left = right;
            continue;
        };
        let Some(after) = points[next] else {
            left = right;
            continue;
        };
        let px = |p: [f64; 2]| [p[0] * w, p[1] * h];
        let (a, b) = (px(begin), px(end));
        let before = px(before);
        let after = px(after);
        let duration_before = (left - previous) as f64;
        let duration_after = (next - right) as f64;
        let incoming = [
            (a[0] - before[0]) / duration_before,
            (a[1] - before[1]) / duration_before,
        ];
        let outgoing = [
            (after[0] - b[0]) / duration_after,
            (after[1] - b[1]) / duration_after,
        ];
        let speed_before = incoming[0].hypot(incoming[1]);
        let speed_after = outgoing[0].hypot(outgoing[1]);
        if speed_before < 3.0 || speed_after < 3.0 {
            left = right;
            continue;
        }
        let cosine =
            (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (speed_before * speed_after);
        let turn = cosine.clamp(-1.0, 1.0).acos().to_degrees();
        let residual = (incoming[0] - outgoing[0]).hypot(incoming[1] - outgoing[1]);
        if turn < 70.0 || residual < 10.0 {
            left = right;
            continue;
        }
        let mut best: Option<(f64, usize, [f64; 2])> = None;
        for (impact, nearby_players) in players.iter().enumerate().take(right).skip(left + 1) {
            let da = (impact - left) as f64;
            let db = (impact as i64 - right as i64) as f64;
            let from_left = [a[0] + da * incoming[0], a[1] + da * incoming[1]];
            let from_right = [b[0] + db * outgoing[0], b[1] + db * outgoing[1]];
            let error = (from_left[0] - from_right[0]).hypot(from_left[1] - from_right[1]);
            if error > 32.0 {
                continue;
            }
            let joint = [
                (from_left[0] + from_right[0]) * 0.5,
                (from_left[1] + from_right[1]) * 0.5,
            ];
            if !valid_point([joint[0] / w, joint[1] / h]) {
                continue;
            }
            let near_athlete = nearby_players.iter().any(|rect| {
                let [x1, y1, x2, y2] = *rect;
                if ![x1, y1, x2, y2].iter().all(|v| v.is_finite()) || x2 <= x1 || y2 <= y1 {
                    return false;
                }
                let athlete_height = (y2 - y1) * h;
                let reach = (athlete_height * 0.42).clamp(12.0, 140.0);
                joint[0] >= x1 * w - reach
                    && joint[0] <= x2 * w + reach
                    // Jump attacks may contact the ball well above the
                    // detected upper-body box while hands are extended.
                    && joint[1] >= y1 * h - athlete_height * 0.55
                    && joint[1] <= y2 * h + athlete_height * 0.15
            });
            if near_athlete && best.is_none_or(|value| error < value.0) {
                best = Some((error, impact, joint));
            }
        }
        let Some((_, impact, joint)) = best else {
            left = right;
            continue;
        };
        // A piecewise flight is valid only if the reconstructed samples
        // remain in frame and move at bounded, physically plausible speeds.
        let reconstructed = (left + 1..right)
            .map(|index| {
                let candidate = if index <= impact {
                    let t = (index - left) as f64 / (impact - left) as f64;
                    [a[0] + (joint[0] - a[0]) * t, a[1] + (joint[1] - a[1]) * t]
                } else {
                    let t = (index - impact) as f64 / (right - impact) as f64;
                    [
                        joint[0] + (b[0] - joint[0]) * t,
                        joint[1] + (b[1] - joint[1]) * t,
                    ]
                };
                [candidate[0] / w, candidate[1] / h]
            })
            .collect::<Vec<_>>();
        let mut previous_point = a;
        let mut plausible = true;
        for point in reconstructed.iter().copied().chain(std::iter::once(end)) {
            let value = px(point);
            if !valid_point(point)
                || (value[0] - previous_point[0]).hypot(value[1] - previous_point[1]) > 65.0
            {
                plausible = false;
                break;
            }
            previous_point = value;
        }
        if plausible {
            for (index, sample) in (left + 1..right).zip(reconstructed) {
                points[index] = Some(sample);
                synthetic[index] = true;
            }
        }
        left = right;
    }
}

fn valid_point(point: [f64; 2]) -> bool {
    point
        .iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_rebound_survives_short_interleaved_false_ball_island() {
        let samples = (0..53)
            .map(|frame| {
                let t = frame as f64;
                let y = match frame {
                    0..=18 => 0.025 + 0.019 * t + 0.00115 * t * t,
                    19 => 0.75,
                    20 => 0.825,
                    21 => 0.82,
                    22..=24 => 0.675 - 0.055 * (t - 22.0),
                    25..=27 => 0.76,
                    28..=42 => 0.355 - 0.047 * (t - 28.0) + 0.0017 * (t - 28.0).powi(2),
                    _ => 0.03 + 0.0015 * (t - 42.0).powi(2),
                };
                ScoredBallSample {
                    frame,
                    point: [0.42 + 0.0065 * t.min(20.0) + 0.001 * (t - 20.0).max(0.0), y],
                    confidence: 0.6,
                }
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 53, BallStabilizerSettings::default()).unwrap();
        assert!(
            result.points[20].is_some_and(|p| p[1] > 0.8),
            "{:?}",
            &result.points[17..43]
        );
        assert!(result.points[24].is_some_and(|p| p[1] < 0.60));
        assert!(result.points[30].is_some_and(|p| p[1] < 0.35));
        assert!(result.rejected_frames.contains(&26));
        let points = result
            .points
            .iter()
            .enumerate()
            .filter_map(|(frame, point)| {
                let [x, y] = (*point)?;
                Some(crate::motion::BallObservation {
                    frame: frame as i64,
                    x: x * 1920.0,
                    y: y * 1080.0,
                })
            })
            .collect::<Vec<_>>();
        let breaks = crate::motion::trajectory_direction_breaks(&points, Default::default());
        assert!(
            breaks.iter().any(|e| (18..=22).contains(&e.frame)),
            "{breaks:?}"
        );
    }

    #[test]
    fn ball_stabilizer_does_not_panic_when_adjacent_segment_windows_share_a_rejected_island() {
        let mut samples = Vec::new();
        for (start, end, x, y) in [
            (0, 3, 0.20, 0.30),
            (8, 10, 0.80, 0.78),
            (15, 18, 0.23, 0.32),
            (23, 26, 0.55, 0.39),
        ] {
            for frame in start..=end {
                samples.push(ScoredBallSample {
                    frame,
                    point: [x + (frame - start) as f64 * 0.003, y],
                    confidence: 0.82,
                });
            }
        }
        let result = stabilize_ball_motion(&samples, 27, BallStabilizerSettings::default())
            .expect("overlapping segment windows must not access removed observations");
        assert!((8..=10).all(|frame| result.rejected_frames.contains(&frame)));
        assert!(result.points[15].is_some());
    }
    #[test]
    fn idle_source_switch_removes_remote_prefix_without_losing_flight() {
        let samples = (0..100)
            .map(|frame| ScoredBallSample {
                frame,
                point: if frame < 60 {
                    [0.93 + (frame % 3) as f64 * 0.002, 0.58]
                } else {
                    [
                        0.41 + (frame - 60) as f64 * 0.002,
                        0.90 - (frame - 60) as f64 * 0.012,
                    ]
                },
                confidence: 0.9,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 100, BallStabilizerSettings::default()).unwrap();
        assert!(result.points[..60].iter().all(Option::is_none));
        assert!(result.rejected_frames.contains(&59));
        assert!(result.points[60..].iter().all(Option::is_some));
        assert!(!result.interpolated[..60].iter().any(|&v| v));
    }

    #[test]
    fn idle_source_switch_survives_two_frame_remote_glitch() {
        let samples = (0..100)
            .map(|frame| ScoredBallSample {
                frame,
                point: match frame {
                    35..=36 => [0.25, 0.82],
                    0..=59 => [0.93 + (frame % 3) as f64 * 0.002, 0.58],
                    _ => [
                        0.41 + (frame - 60) as f64 * 0.002,
                        0.90 - (frame - 60) as f64 * 0.012,
                    ],
                },
                confidence: 0.9,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 100, BallStabilizerSettings::default()).unwrap();
        assert!(result.points[..60].iter().all(Option::is_none));
        assert!(result.points[60..].iter().all(Option::is_some));
        assert!(!result.interpolated[..60].iter().any(|&v| v));
    }

    #[test]
    fn idle_lock_rejects_remote_motion_that_returns_to_original_ball() {
        let samples = (0..100)
            .map(|frame| ScoredBallSample {
                frame,
                point: if (45..60).contains(&frame) {
                    [0.20 + (frame - 45) as f64 * 0.008, 0.35]
                } else {
                    [0.85, 0.75]
                },
                confidence: 0.9,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 100, BallStabilizerSettings::default()).unwrap();
        assert!(result.rejected_frames.contains(&45));
        assert!(result.points[45].is_none());
    }
    #[test]
    fn ball_stabilizer_keeps_lock_through_long_false_detection_burst() {
        let samples = (0..66)
            .map(|frame| ScoredBallSample {
                frame,
                point: if (24..=31).contains(&frame) {
                    [0.84 + (frame - 24) as f64 * 0.004, 0.82]
                } else {
                    [0.20 + frame as f64 * 0.004, 0.36]
                },
                confidence: if (24..=31).contains(&frame) {
                    0.99
                } else {
                    0.07
                },
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 66, BallStabilizerSettings::default()).unwrap();
        for frame in 24..=31 {
            assert!(result.rejected_frames.contains(&frame), "frame {frame}");
            assert!(result.interpolated[frame], "frame {frame}");
            assert!(
                (result.points[frame].unwrap()[0] - (0.20 + frame as f64 * 0.004)).abs() < 0.01
            );
        }
        assert_eq!(result.confidence_frames, 66);
        assert_eq!(result.observed_frames, 58);
    }

    #[test]
    fn ball_stabilizer_rejects_physically_reachable_false_detour_that_rejoins() {
        let samples = (0..62)
            .map(|frame| ScoredBallSample {
                frame,
                point: if (22..=29).contains(&frame) {
                    [0.18 + frame as f64 * 0.003, 0.48]
                } else {
                    [0.18 + frame as f64 * 0.003, 0.39]
                },
                confidence: 0.85,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 62, BallStabilizerSettings::default()).unwrap();
        for frame in 22..=29 {
            assert!(result.rejected_frames.contains(&frame), "frame {frame}");
            assert!(result.interpolated[frame], "frame {frame}");
        }
        assert!(result.rejected_frames.iter().all(|f| (22..=29).contains(f)));
        assert_eq!(result.observed_frames, 54);
    }

    #[test]
    fn ball_stabilizer_keeps_touch_when_ball_crosses_old_prediction_once() {
        let y_after = [0.36, 0.42, 0.48, 0.37, 0.26, 0.15, 0.04];
        let samples = (0..28)
            .map(|frame| ScoredBallSample {
                frame,
                point: [
                    0.38,
                    if frame <= 20 {
                        0.30
                    } else {
                        y_after[frame - 21]
                    },
                ],
                confidence: 0.12,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 28, BallStabilizerSettings::default()).unwrap();
        assert!(
            result.rejected_frames.is_empty(),
            "{:?}",
            result.rejected_frames
        );
        assert!(result.interpolated.iter().all(|flag| !flag));
        assert_eq!(result.points[25], Some([0.38, 0.26]));
    }

    #[test]
    fn ball_stabilizer_retains_supported_abrupt_touch_direction_change() {
        let samples = (0..29)
            .map(|frame| ScoredBallSample {
                frame,
                point: if frame <= 17 {
                    [0.15 + frame as f64 * 0.027, 0.27]
                } else {
                    [0.15 + 17.0 * 0.027 - (frame - 17) as f64 * 0.047, 0.27]
                },
                confidence: 0.11,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 42, BallStabilizerSettings::default()).unwrap();
        assert!(
            result.rejected_frames.is_empty(),
            "{:?}",
            result.rejected_frames
        );
        assert_eq!(result.observed_frames, 29);
        assert_eq!(result.points[18], Some([0.15 + 17.0 * 0.027 - 0.047, 0.27]));
    }

    #[test]
    fn ball_stabilizer_rejects_remote_false_positives_before_interpolation() {
        let samples: Vec<_> = (0..42)
            .filter(|&frame| frame != 17 && frame != 18)
            .map(|frame| ScoredBallSample {
                frame,
                point: if frame == 12 || frame == 13 {
                    [0.92, 0.88]
                } else {
                    [0.2 + frame as f64 * 0.003, 0.3]
                },
                confidence: if frame == 5 { 0.50 } else { 0.9 },
            })
            .collect();
        let result =
            stabilize_ball_motion(&samples, 42, BallStabilizerSettings::default()).unwrap();
        assert!(result.rejected_frames.contains(&12));
        assert!(result.rejected_frames.contains(&13));
        assert!(result.points[5].is_some());
        assert!(result.points[12].is_some());
        assert!(result.interpolated[12]);
        assert!(result.interpolated[17]);
        assert!(result.interpolated[18]);
        assert!(!result.interpolated[14]);
        assert!(result.points[41].is_some());
    }

    #[test]
    fn ball_stabilizer_rejects_short_consistent_false_island() {
        let samples = (0..45)
            .map(|frame| ScoredBallSample {
                frame,
                point: if (18..=20).contains(&frame) {
                    [0.88 + (frame - 18) as f64 * 0.004, 0.8]
                } else {
                    [0.20 + frame as f64 * 0.004, 0.4]
                },
                confidence: 0.95,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 45, BallStabilizerSettings::default()).unwrap();
        for frame in 18..=20 {
            assert!(result.rejected_frames.contains(&frame));
            assert!(result.interpolated[frame]);
            assert!(result.points[frame].unwrap()[0] < 0.4);
        }
    }

    #[test]
    fn ball_stabilizer_rejects_interleaved_corner_false_detections() {
        let samples = (0..48)
            .map(|frame| {
                let (point, confidence) = match frame {
                    19 | 20 => ([0.932, 0.576], 0.63),
                    22 => ([0.934, 0.576], 0.67),
                    _ => (
                        [0.42 - frame as f64 * 0.004, 0.43 + frame as f64 * 0.005],
                        0.68,
                    ),
                };
                ScoredBallSample {
                    frame,
                    point,
                    confidence,
                }
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 48, BallStabilizerSettings::default()).unwrap();
        for frame in [19, 20, 22] {
            assert!(
                result.rejected_frames.contains(&frame),
                "false corner detection at frame {frame} survived"
            );
            assert!(result.interpolated[frame]);
            assert!(result.points[frame].unwrap()[0] < 0.5);
        }
        for frame in [17, 18, 21, 23, 24] {
            assert!(!result.rejected_frames.contains(&frame));
            assert!(!result.interpolated[frame]);
        }
    }

    #[test]
    fn ball_stabilizer_keeps_real_sustained_corner_and_fast_motion() {
        for fast in [false, true] {
            let samples = (0..35)
                .map(|frame| ScoredBallSample {
                    frame,
                    point: if fast {
                        [0.06 + frame as f64 * 0.025, 0.12 + frame as f64 * 0.019]
                    } else {
                        [0.86 + frame as f64 * 0.002, 0.73 + frame as f64 * 0.002]
                    },
                    confidence: 0.72,
                })
                .collect::<Vec<_>>();
            let result =
                stabilize_ball_motion(&samples, 35, BallStabilizerSettings::default()).unwrap();
            assert!(result.rejected_frames.is_empty());
            assert_eq!(result.observed_frames, 35);
            assert!(result.interpolated.iter().all(|&value| !value));
        }
    }

    #[test]
    fn ball_stabilizer_rejects_recurrent_static_false_detection_in_any_image_area() {
        let mut samples = Vec::new();
        for frame in 0..900 {
            if (10..59).contains(&frame)
                || (180..226).contains(&frame)
                || (540..583).contains(&frame)
                || (750..775).contains(&frame)
            {
                samples.push(ScoredBallSample {
                    frame,
                    point: [
                        0.63 + (frame % 3) as f64 * 0.001,
                        0.55 + (frame % 4) as f64 * 0.001,
                    ],
                    confidence: 0.84,
                });
            } else if (300..343).contains(&frame) {
                samples.push(ScoredBallSample {
                    frame,
                    point: [0.15 + (frame - 300) as f64 * 0.004, 0.30],
                    confidence: 0.84,
                });
            }
        }
        let result =
            stabilize_ball_motion(&samples, 900, BallStabilizerSettings::default()).unwrap();
        for frame in [10, 25, 180, 200, 540, 560, 750] {
            assert!(result.rejected_frames.contains(&frame));
            assert_eq!(result.points[frame], None);
        }
        assert!((result.points[320].unwrap()[0] - 0.23).abs() < 1e-12);
        assert_eq!(result.points[320].unwrap()[1], 0.30);
    }

    #[test]
    fn ball_stabilizer_preserves_genuine_motion_through_a_recurrent_static_hotspot() {
        let mut samples = Vec::new();
        for frame in 0..900 {
            if (10..59).contains(&frame)
                || (180..226).contains(&frame)
                || (540..583).contains(&frame)
                || (750..775).contains(&frame)
            {
                samples.push(ScoredBallSample {
                    frame,
                    point: [0.63 + (frame % 3) as f64 * 0.001, 0.55],
                    confidence: 0.84,
                });
            } else if (310..321).contains(&frame) {
                samples.push(ScoredBallSample {
                    frame,
                    point: [0.505 + (frame - 310) as f64 * 0.025, 0.55],
                    confidence: 0.84,
                });
            }
        }
        let result =
            stabilize_ball_motion(&samples, 900, BallStabilizerSettings::default()).unwrap();
        assert_eq!(result.points[315], Some([0.63, 0.55]));
        assert!(!result.rejected_frames.contains(&315));
        assert!(!result.interpolated[315]);
    }

    #[test]
    fn ball_stabilizer_rejects_jittering_recurrent_hotspot_before_flight() {
        let mut samples = Vec::new();
        for frame in 0..900 {
            let point = if frame < 140 || (350..390).contains(&frame) || (700..760).contains(&frame)
            {
                [
                    0.932 + (frame % 3) as f64 * 0.002,
                    0.57 + (frame % 4) as f64 * 0.0055,
                ]
            } else if (140..172).contains(&frame) {
                [
                    0.408 + (frame - 140) as f64 * 0.002,
                    0.94 - (frame - 140) as f64 * 0.018,
                ]
            } else {
                continue;
            };
            samples.push(ScoredBallSample {
                frame,
                point,
                confidence: 0.8,
            });
        }
        let result =
            stabilize_ball_motion(&samples, 900, BallStabilizerSettings::default()).unwrap();
        assert!(result.points[..140].iter().all(Option::is_none));
        assert_eq!(result.points[140], Some([0.408, 0.94]));
        assert!(!result.interpolated[140]);
        assert!(!result.rejected_frames.contains(&140));
    }

    #[test]
    fn ball_stabilizer_does_not_reacquire_preexisting_remote_ball_after_loss() {
        let samples = (0..100)
            .filter_map(|frame| {
                let point = match frame {
                    0..=19 => [0.28, 0.21 + (frame % 5) as f64 * 0.004],
                    40..=59 => [0.68 - (frame - 40) as f64 * 0.008, 0.25],
                    60 => [0.28, 0.215],
                    78..=99 => [
                        0.28 + (frame % 3) as f64 * 0.002,
                        0.21 + (frame % 5) as f64 * 0.005,
                    ],
                    _ => return None,
                };
                Some(ScoredBallSample {
                    frame,
                    point,
                    confidence: 0.8,
                })
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 100, BallStabilizerSettings::default()).unwrap();
        assert_eq!(result.points[59], Some([0.528, 0.25]));
        assert!(result.points[60..].iter().all(Option::is_none));
        assert!(result.rejected_frames.contains(&78));
        assert!(!result.interpolated[60..].iter().any(|&v| v));
    }

    #[test]
    fn ball_stabilizer_keeps_continuous_return_through_preexisting_ball_area() {
        let samples = (0..85)
            .filter_map(|frame| {
                let point = match frame {
                    0..=19 => [0.28, 0.21 + (frame % 5) as f64 * 0.004],
                    40..=84 => [0.68 - (frame - 40) as f64 * 0.01, 0.22],
                    _ => return None,
                };
                Some(ScoredBallSample {
                    frame,
                    point,
                    confidence: 0.8,
                })
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 85, BallStabilizerSettings::default()).unwrap();
        assert_eq!(result.points[80], Some([0.28, 0.22]));
        assert!(!result.rejected_frames.contains(&80));
    }

    #[test]
    fn ball_stabilizer_keeps_a_stationary_ball_seen_in_one_episode() {
        let samples = (15..70)
            .map(|frame| ScoredBallSample {
                frame,
                point: [
                    0.63 + (frame % 3) as f64 * 0.001,
                    0.55 + (frame % 4) as f64 * 0.001,
                ],
                confidence: 0.85,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 300, BallStabilizerSettings::default()).unwrap();
        assert!(result.rejected_frames.is_empty());
        assert_eq!(result.observed_frames, 55);
    }

    #[test]
    fn ball_stabilizer_rejects_short_static_island_with_one_previous_anchor() {
        let samples = (24..37)
            .map(|frame| ScoredBallSample {
                frame,
                point: if frame == 24 {
                    [0.553, 0.562]
                } else if (25..=27).contains(&frame) {
                    [0.562, 0.76 + (frame % 2) as f64 * 0.004]
                } else {
                    [
                        0.553 + (frame - 28) as f64 * 0.002,
                        0.354 - (frame - 28) as f64 * 0.035,
                    ]
                },
                confidence: 0.8,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 48, BallStabilizerSettings::default()).unwrap();
        assert_eq!(result.points[24], Some([0.553, 0.562]));
        assert_eq!(result.points[28], Some([0.553, 0.354]));
        for frame in 25..=27 {
            assert!(result.rejected_frames.contains(&frame), "frame {frame}");
            assert!(result.interpolated[frame]);
            let point = result.points[frame].unwrap();
            assert!(point[1] < 0.56 && point[1] > 0.35);
        }
        assert!(result
            .rejected_frames
            .iter()
            .all(|&frame| (25..=27).contains(&frame)));
    }

    #[test]
    fn weak_false_island_does_not_erase_a_supported_rebound() {
        let data = [
            (0.4727, 0.1944, 0.67),
            (0.4707, 0.2049, 0.65),
            (0.4707, 0.2188, 0.62),
            (0.4707, 0.2326, 0.65),
            (0.4668, 0.2465, 0.67),
            (0.4668, 0.2639, 0.65),
            (0.4648, 0.2812, 0.67),
            (0.4668, 0.2986, 0.53),
            (0.4668, 0.3090, 0.39),
            (0.4668, 0.3264, 0.326),
            (0.4805, 0.3125, 0.32),
            (0.4805, 0.3090, 0.327),
            (0.4785, 0.3056, 318.0 / 1000.0),
            (0.4785, 0.3056, 0.357),
            (0.4805, 0.3056, 0.38),
            (0.4766, 0.2917, 0.433),
            (0.4883, 0.3299, 0.322),
            (0.4961, 0.3229, 0.394),
            (0.4980, 0.3194, 0.518),
            (0.5039, 0.3125, 0.616),
            (0.5098, 0.3125, 0.599),
            (0.5117, 0.3090, 0.477),
            (0.5156, 0.3056, 0.339),
            (0.5254, 0.3021, 0.34),
            (0.5410, 0.2917, 0.382),
            (0.5410, 0.3021, 0.644),
            (0.5469, 0.3056, 0.684),
            (0.5508, 0.3090, 0.656),
        ];
        let samples = data
            .iter()
            .enumerate()
            .map(|(frame, &(x, y, confidence))| ScoredBallSample {
                frame,
                point: [x, y],
                confidence,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, data.len(), BallStabilizerSettings::default()).unwrap();
        assert_eq!(result.points[9], Some(samples[9].point));
        assert_eq!(result.points[16], Some(samples[16].point));
        assert!(
            result.interpolated[10..16].iter().all(|&b| b),
            "{:?}",
            result.points
        );
        let peak = result.points[10..16]
            .iter()
            .flatten()
            .map(|p| p[1])
            .fold(0.0_f64, f64::max);
        assert!(peak > 0.345, "peak {peak}");
        let observations = result
            .points
            .iter()
            .enumerate()
            .filter_map(|(frame, point)| {
                point.map(|[x, y]| crate::motion::BallObservation {
                    frame: frame as i64,
                    x: x * 3840.0,
                    y: y * 2160.0,
                })
            })
            .collect::<Vec<_>>();
        let breaks = crate::motion::trajectory_direction_breaks(
            &observations,
            crate::motion::DirectionBreakConfig::default(),
        );
        assert!(
            breaks
                .iter()
                .any(|b| b.frame.abs_diff(11) <= 1 && b.motion.angle >= 90.0),
            "{breaks:?}"
        );
        assert!(
            result
                .rejected_frames
                .iter()
                .filter(|&&f| (10..16).contains(&f))
                .count()
                == 6
        );
        let strong = samples
            .iter()
            .cloned()
            .map(|mut p| {
                p.confidence = 0.9;
                p
            })
            .collect::<Vec<_>>();
        let protected =
            stabilize_ball_motion(&strong, data.len(), BallStabilizerSettings::default()).unwrap();
        assert!(!protected.interpolated[10..16].iter().all(|&b| b));
    }

    #[test]
    fn short_confirmed_rebound_gaps_preserve_the_turn() {
        for missing in 1..=4 {
            let samples = (0..=20)
                .filter(|frame| !(10..10 + missing).contains(frame))
                .map(|frame| {
                    let t = frame as f64 - 10.0;
                    let y = if t <= 0.0 {
                        0.54 + 0.022 * t
                    } else {
                        0.54 - 0.030 * t
                    };
                    ScoredBallSample {
                        frame,
                        point: [0.60 + 0.0015 * t, (y * 288.0).round() / 288.0],
                        confidence: 0.8,
                    }
                })
                .collect::<Vec<_>>();
            let result =
                stabilize_ball_motion(&samples, 21, BallStabilizerSettings::default()).unwrap();
            assert!(
                result.interpolated[10..10 + missing]
                    .iter()
                    .all(|&flag| flag),
                "gap {missing}: {:?}",
                result.points
            );
            let peak = result.points[9..=10 + missing]
                .iter()
                .flatten()
                .map(|p| p[1])
                .fold(0.0_f64, f64::max);
            assert!(
                peak > result.points[9].unwrap()[1] + 0.005,
                "gap {missing}: {peak}"
            );
            let points = result
                .points
                .iter()
                .enumerate()
                .filter_map(|(frame, point)| {
                    point.map(|p| crate::motion::BallObservation {
                        frame: frame as i64,
                        x: p[0] * 3840.0,
                        y: p[1] * 2160.0,
                    })
                })
                .collect::<Vec<_>>();
            let breaks = crate::motion::trajectory_direction_breaks(
                &points,
                crate::motion::DirectionBreakConfig::default(),
            );
            assert!(
                breaks.iter().any(|b| b.frame.abs_diff(10) <= 1),
                "gap {missing}: {breaks:?}"
            );
        }
    }
    #[test]
    fn ball_stabilizer_reconstructs_an_occluded_rebound_instead_of_erasing_it() {
        let samples = |offset: f64| {
            (407..=435)
                .filter(|frame| !(420..=427).contains(frame))
                .map(|frame| {
                    let t = (frame as i64 - 419) as f64;
                    ScoredBallSample {
                        frame,
                        point: [
                            (1035.0 + 2.5 * t + if frame > 419 { offset } else { 0.0 }) / 1920.0,
                            (if frame <= 419 {
                                820.0 + 48.0 * t + 0.15 * t * t
                            } else {
                                790.0 - 42.0 * t + 0.20 * t * t
                            }) / 1080.0,
                        ],
                        confidence: 0.85,
                    }
                })
                .collect::<Vec<_>>()
        };
        let result =
            stabilize_ball_motion(&samples(0.0), 436, BallStabilizerSettings::default()).unwrap();
        for frame in 420..=427 {
            assert!(
                result.interpolated[frame],
                "missing synthetic frame {frame}"
            );
            assert!(result.points[frame].is_some());
        }
        assert!(result.points[419].unwrap()[1] > result.points[418].unwrap()[1]);
        assert!(result.points[420].unwrap()[1] < result.points[419].unwrap()[1]);
        assert!(result.points[429].unwrap()[1] < result.points[428].unwrap()[1]);
        assert!(!result.interpolated[419]);
        assert!(!result.interpolated[428]);

        let switched =
            stabilize_ball_motion(&samples(260.0), 436, BallStabilizerSettings::default()).unwrap();
        assert!(!switched.interpolated[424]);
        assert!(switched.points[424].is_none());
        let apex = (407..=435)
            .filter(|frame| !(420..=427).contains(frame))
            .map(|frame| {
                let t = (frame as i64 - 420) as f64;
                ScoredBallSample {
                    frame,
                    point: [0.52, (350.0 + 0.20 * t * t) / 1080.0],
                    confidence: 0.90,
                }
            })
            .collect::<Vec<_>>();
        let smooth = stabilize_ball_motion(&apex, 436, BallStabilizerSettings::default()).unwrap();
        assert!(smooth.points[420..=427].iter().all(Option::is_some));
        assert!(smooth.points[418].unwrap()[1] > smooth.points[419].unwrap()[1]);
        assert!(smooth.points[429].unwrap()[1] > smooth.points[428].unwrap()[1]);
    }
    #[test]
    fn ball_stabilizer_keeps_real_turns_and_refuses_long_gaps() {
        let mut samples = Vec::new();
        for frame in (0..=80).filter(|f| !(30..=55).contains(f)) {
            let x = if frame <= 15 {
                0.20 + frame as f64 * 0.004
            } else {
                0.26 - (frame as f64 - 15.0) * 0.003
            };
            samples.push(ScoredBallSample {
                frame,
                point: [x, 0.35],
                confidence: 0.95,
            });
        }
        let result =
            stabilize_ball_motion(&samples, 81, BallStabilizerSettings::default()).unwrap();
        assert!(result.points[15].is_some());
        assert!(result.points[40].is_none());
        assert!(result.points[20].is_some());
    }

    #[test]
    fn weak_foreign_prefix_cannot_block_a_supported_fast_flight() {
        let samples = (0..60)
            .map(|frame| {
                let (point, confidence) = if frame < 16 {
                    ([0.20 + frame as f64 * 0.001, 0.65], 0.09)
                } else {
                    (
                        [
                            0.85 - (frame - 16) as f64 * 0.009,
                            0.8 - (frame - 16) as f64 * 0.015,
                        ],
                        0.65,
                    )
                };
                ScoredBallSample {
                    frame,
                    point,
                    confidence,
                }
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 60, BallStabilizerSettings::default()).unwrap();
        assert_eq!(result.points[16], Some(samples[16].point));
        assert_eq!(result.points[20], Some(samples[20].point));
        assert!(!result.interpolated[16]);
        assert!(result.points[10].is_none());
    }

    #[test]
    fn weak_motion_between_supported_flights_does_not_create_a_false_bridge() {
        let samples = (0..120)
            .map(|frame| {
                let (point, confidence) = if frame < 30 {
                    ([0.2 + frame as f64 * 0.003, 0.3], 0.65)
                } else if frame < 90 {
                    ([0.29 + (frame - 30) as f64 * 0.003, 0.3], 0.09)
                } else {
                    ([0.47 + (frame - 90) as f64 * 0.003, 0.3], 0.65)
                };
                ScoredBallSample {
                    frame,
                    point,
                    confidence,
                }
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 120, BallStabilizerSettings::default()).unwrap();
        assert!(result.points[45..75].iter().all(Option::is_none));
        assert!(result.rejected_frames.contains(&60));
        assert!(result.points[10].is_some());
        assert!(result.points[100].is_some());
    }

    #[test]
    fn ball_stabilizer_accepts_continuous_low_confidence_observations() {
        let samples = (0..40)
            .map(|frame| ScoredBallSample {
                frame,
                point: [0.2 + frame as f64 * 0.003, 0.4],
                confidence: 0.08,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 40, BallStabilizerSettings::default()).unwrap();
        assert_eq!(result.confidence_frames, 40);
        assert_eq!(result.observed_frames, 40);
        assert!(result.rejected_frames.is_empty());
        assert!(result.points.iter().all(Option::is_some));
    }

    #[test]
    fn ball_stabilizer_rejects_remote_false_detection_even_at_high_confidence() {
        let samples = (0..42)
            .map(|frame| {
                let expected = [0.20 + frame as f64 * 0.004, 0.35];
                ScoredBallSample {
                    frame,
                    point: match frame {
                        12 => [expected[0] + 0.014, expected[1] + 0.012],
                        13 => [expected[0] - 0.012, expected[1] - 0.010],
                        22 => [0.92, 0.83],
                        _ => expected,
                    },
                    confidence: if frame == 22 { 0.99 } else { 0.06 },
                }
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 42, BallStabilizerSettings::default()).unwrap();
        for frame in [12, 13] {
            assert!(!result.rejected_frames.contains(&frame));
            assert!(!result.interpolated[frame]);
            assert_eq!(result.points[frame], Some(samples[frame].point));
        }
        assert!(result.rejected_frames.contains(&22));
        assert!(result.interpolated[22]);
        assert!((result.points[22].unwrap()[0] - (0.20 + 22.0 * 0.004)).abs() < 1e-9);
        assert_eq!(result.observed_frames, 41);
    }

    #[test]
    fn ball_stabilizer_does_not_require_strong_anchors() {
        let samples = (0..37)
            .map(|frame| ScoredBallSample {
                frame,
                point: [0.2 + frame as f64 * 0.003, 0.4],
                confidence: 0.05,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 37, BallStabilizerSettings::default()).unwrap();
        assert_eq!(result.observed_frames, 37);
        assert!(result.points.iter().all(Option::is_some));
        assert!(result.interpolated.iter().all(|&flag| !flag));
    }

    #[test]
    fn ball_stabilizer_does_not_bridge_long_gaps_or_extrapolate() {
        let samples = (0..44)
            .filter(|frame| !(6..=37).contains(frame))
            .map(|frame| ScoredBallSample {
                frame,
                point: [0.22 + frame as f64 * 0.002, 0.32],
                confidence: 0.05,
            })
            .collect::<Vec<_>>();
        let result =
            stabilize_ball_motion(&samples, 44, BallStabilizerSettings::default()).unwrap();
        for frame in [8, 15, 22, 31, 36] {
            assert_eq!(result.points[frame], None, "frame {frame}");
        }
        assert_eq!(result.observed_frames, 12);
        assert!(result.points[0].is_some());
        assert!(result.points[43].is_some());
    }

    #[test]
    fn ball_stabilizer_accepts_old_profile_without_reintroducing_confidence_gate() {
        let settings: BallStabilizerSettings = serde_json::from_str(
            r#"{"min_confidence":0.70,"max_gap_frames":12,"max_step_fraction":0.12}"#,
        )
        .unwrap();
        assert_eq!(settings, BallStabilizerSettings::default());
        assert!(!serde_json::to_string(&settings)
            .unwrap()
            .contains("min_confidence"));
        assert!(serde_json::from_str::<BallStabilizerSettings>(
            r#"{"unsupported_threshold":0.70}"#
        )
        .is_err());
        assert!(BallStabilizerSettings {
            max_gap_frames: 99,
            ..Default::default()
        }
        .validate()
        .is_err());
        let points = (5..9)
            .map(|frame| ScoredBallSample {
                frame,
                point: [0.25 + frame as f64 * 0.001, 0.4],
                confidence: 0.05,
            })
            .collect::<Vec<_>>();
        let result = stabilize_ball_motion(&points, 20, BallStabilizerSettings::default()).unwrap();
        assert_eq!(result.points[0], None);
        assert_eq!(result.points[19], None);
        assert_eq!(result.observed_frames, 4);
    }

    #[test]
    fn reconstruction_removes_single_false_ball_and_bridges_short_smooth_gaps() {
        let raw: Vec<_> = (0..65)
            .filter(|frame| ![27, 28].contains(frame))
            .map(|frame| {
                let t = frame as f64 - 32.0;
                crate::motion::BallObservation {
                    frame,
                    x: 580.0 + 4.0 * t + if frame == 18 { 1100.0 } else { 0.0 },
                    y: 230.0 + 0.16 * t * t,
                }
            })
            .collect();
        let restored = reconstruct_short_ball_motion(&raw, 1920, 1080, 28.0).unwrap();
        assert_eq!(restored.rejected_frames, [18]);
        assert!(restored.interpolated_frames.contains(&27));
        assert!(restored.interpolated_frames.contains(&28));
        assert!(restored.observations.iter().any(|p| p.frame == 18));
        assert!(restored.observations.iter().all(|p| p.x < 1000.0));
        for &frame in &[17, 19, 26, 29] {
            let before = raw.iter().find(|p| p.frame == frame).unwrap();
            let after = restored
                .observations
                .iter()
                .find(|p| p.frame == frame)
                .unwrap();
            assert!((before.x - after.x).abs() < 1e-9);
            assert!((before.y - after.y).abs() < 1e-9);
        }
    }

    #[test]
    fn reconstruction_never_extrapolates_or_bridges_an_unobserved_large_gap() {
        let raw: Vec<_> = (0..44)
            .filter(|frame| ![0, 22, 23, 24, 43].contains(frame))
            .map(|frame| crate::motion::BallObservation {
                frame,
                x: 450.0 + 2.0 * frame as f64,
                y: 240.0,
            })
            .collect();
        let restored = reconstruct_short_ball_motion(&raw, 1920, 1080, 28.0).unwrap();
        assert!(restored
            .interpolated_frames
            .iter()
            .all(|frame| ![0, 22, 23, 24, 43].contains(frame)));
        assert!(restored
            .observations
            .iter()
            .all(|p| ![0, 22, 23, 24, 43].contains(&p.frame)));
    }

    #[test]
    fn reconstruction_does_not_splice_two_flights_across_a_contact() {
        let raw: Vec<_> = (0..80)
            .filter(|frame| *frame != 41)
            .map(|frame| {
                let t = frame as f64 - 40.0;
                let rebound = t.max(0.0);
                crate::motion::BallObservation {
                    frame,
                    x: 900.0 - 3.0 * t,
                    y: 600.0 + 6.0 * t - 15.0 * rebound + 0.06 * t * t,
                }
            })
            .collect();
        let restored = reconstruct_short_ball_motion(&raw, 1920, 1080, 28.0).unwrap();
        assert!(!restored.interpolated_frames.contains(&41));
    }

    #[test]
    fn fills_only_interior_gaps_preserving_observations() {
        let mut points = [None, Some([0.2, 0.4]), None, None, Some([0.5, 0.4]), None];
        let flags = interpolate_short_gaps(&mut points);
        assert_eq!(flags, [false, false, true, true, false, false]);
        assert_eq!(points[0], None);
        assert_eq!(points[1], Some([0.2, 0.4]));
        assert!((points[2].unwrap()[0] - 0.3).abs() < 1e-12);
        assert!((points[3].unwrap()[0] - 0.4).abs() < 1e-12);
        assert_eq!(points[4], Some([0.5, 0.4]));
        assert_eq!(points[5], None);
    }

    #[test]
    fn does_not_bridge_long_missing_segments() {
        let mut points = vec![Some([0.2, 0.2])];
        points.extend([None; MAX_MISSING_BALL_FRAMES + 1]);
        points.push(Some([0.3, 0.2]));
        assert!(!interpolate_short_gaps(&mut points).iter().any(|flag| *flag));
        assert!(points[1..points.len() - 1].iter().all(Option::is_none));
    }

    #[test]
    fn does_not_join_implausible_switch_or_invalid_coordinates() {
        let mut points = [
            Some([0.05, 0.05]),
            None,
            Some([0.95, 0.95]),
            Some([f64::NAN, 0.3]),
            None,
            Some([0.6, 0.6]),
        ];
        assert!(!interpolate_short_gaps(&mut points).iter().any(|flag| *flag));
        assert_eq!(points[1], None);
        assert_eq!(points[4], None);
    }

    #[test]
    fn recovers_parabolic_flight_using_both_real_boundaries() {
        let flight = |i: usize| {
            let t = i as f64;
            [0.31 + 0.004 * t, 0.22 + 0.003 * t + 0.00045 * t * t]
        };
        let mut points = (0..18).map(|i| Some(flight(i))).collect::<Vec<_>>();
        for row in &mut points[5..=9] {
            *row = None;
        }
        let mask = interpolate_short_gaps(&mut points);
        for i in 5..=9 {
            assert!(mask[i]);
            let point = points[i].unwrap();
            let actual = flight(i);
            assert!((point[0] - actual[0]).abs() < 0.003);
            assert!((point[1] - actual[1]).abs() < 0.003);
        }
        assert!(!mask[4] && !mask[10]);
    }

    #[test]
    fn refuses_gap_with_hidden_velocity_reversal() {
        let mut points = [
            Some([0.20, 0.30]),
            Some([0.21, 0.30]),
            None,
            None,
            Some([0.24, 0.30]),
            Some([0.20, 0.30]),
        ];
        let mask = interpolate_short_gaps(&mut points);
        assert!(!mask.iter().any(|flag| *flag));
        assert!(points[2].is_none() && points[3].is_none());
    }

    #[test]
    fn refuses_ballistic_path_leaving_image_and_never_fills_tails() {
        let mut points = [
            None,
            Some([0.50, 0.016]),
            Some([0.50, 0.004]),
            None,
            None,
            None,
            Some([0.50, 0.004]),
            Some([0.50, 0.016]),
            None,
        ];
        let mask = interpolate_short_gaps(&mut points);
        assert!(!mask.iter().any(|flag| *flag));
        assert!(points[0].is_none() && points[8].is_none());
        assert!(points[3..6].iter().all(Option::is_none));
    }
}
