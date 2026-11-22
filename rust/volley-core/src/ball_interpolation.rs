//! Bounded VBallNet-only reconstruction of invisible portions of a flight.
//! Every inferred point is marked synthetic and is NEVER Touch evidence.

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

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BallStabilizerSettings {
    pub min_confidence: f32,
    pub max_gap_frames: usize,
    pub max_step_fraction: f64,
}

impl Default for BallStabilizerSettings {
    fn default() -> Self {
        Self {
            min_confidence: 0.55,
            max_gap_frames: 12,
            max_step_fraction: 0.12,
        }
    }
}

impl BallStabilizerSettings {
    pub fn validate(self) -> Result<(), &'static str> {
        if !self.min_confidence.is_finite()
            || !(0.51..=1.0).contains(&self.min_confidence)
            || !(1..=MAX_MISSING_BALL_FRAMES).contains(&self.max_gap_frames)
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

pub const BALL_STABILIZER_ALGORITHM_VERSION: &str = "ball_stabilizer_v3";

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
            (spread[0] <= 0.005
                && spread[1] <= 0.005
                && cell.max[0] - cell.min[0] <= 0.015
                && cell.max[1] - cell.min[1] <= 0.021)
                .then_some(center)
        })
        .collect::<Vec<_>>();

    let mut rejected = Vec::new();
    for (frame, point) in points.iter().enumerate() {
        let Some(point) = point else {
            continue;
        };
        for center in &hotspots {
            if (point[0] - center[0]).hypot(point[1] - center[1]) > 0.018 {
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
    let mut scores = vec![settings.min_confidence; frames];
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
    let mut rejected_frames = reject_recurring_static_hotspots(&points, settings.max_step_fraction);
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
    }
    rejected_frames.sort_unstable();
    rejected_frames.dedup();
    let observed_frames = confidence_frames - rejected_frames.len();
    let interpolated = interpolate_short_gaps_bounded(&mut points, settings.max_gap_frames);
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
    fn ball_stabilizer_validates_strict_confidence_and_does_not_extrapolate() {
        assert!(BallStabilizerSettings {
            min_confidence: 0.5,
            ..Default::default()
        }
        .validate()
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
                confidence: 0.9,
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
