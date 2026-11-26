use super::{BallObservation, MotionEvidence};

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryBreakConfig {
    pub min_velocity_jump_px_frame: f64,
    pub relative_velocity_jump: f64,
    pub max_fit_error_px: f64,
    pub min_separation_frames: i64,
}

impl Default for TrajectoryBreakConfig {
    fn default() -> Self {
        Self {
            min_velocity_jump_px_frame: 8.0,
            relative_velocity_jump: 0.45,
            max_fit_error_px: 4.0,
            min_separation_frames: 6,
        }
    }
}

impl TrajectoryBreakConfig {
    pub fn valid(self) -> bool {
        (2.0..=60.0).contains(&self.min_velocity_jump_px_frame)
            && (0.0..=1.5).contains(&self.relative_velocity_jump)
            && (1.0..=20.0).contains(&self.max_fit_error_px)
            && (3..=30).contains(&self.min_separation_frames)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectionBreakConfig {
    pub min_turn_angle_deg: f64,
    pub max_join_error_px: f64,
    pub max_fit_error_px: f64,
    pub min_separation_frames: i64,
}

impl Default for DirectionBreakConfig {
    fn default() -> Self {
        Self {
            min_turn_angle_deg: 45.0,
            max_join_error_px: 28.0,
            max_fit_error_px: 4.0,
            min_separation_frames: 6,
        }
    }
}

impl DirectionBreakConfig {
    pub fn valid(self) -> bool {
        (15.0..=170.0).contains(&self.min_turn_angle_deg)
            && (8.0..=80.0).contains(&self.max_join_error_px)
            && (1.0..=20.0).contains(&self.max_fit_error_px)
            && (3..=30).contains(&self.min_separation_frames)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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
        let pivot_row = matrix[column];
        for (row, entries) in matrix.iter_mut().enumerate() {
            if row == column {
                continue;
            }
            let scale = entries[column];
            for (entry, pivot) in entries[column..5].iter_mut().zip(&pivot_row[column..5]) {
                *entry -= scale * pivot;
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
    trajectory_breaks_configured(points, TrajectoryBreakConfig::default())
}

pub fn trajectory_breaks_configured(
    points: &[BallObservation],
    config: TrajectoryBreakConfig,
) -> Vec<TrajectoryBreak> {
    if points.len() < 9
        || !config.valid()
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
        if error > config.max_fit_error_px {
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
                < config
                    .min_velocity_jump_px_frame
                    .max(config.relative_velocity_jump * speed_before.max(speed_after))
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
        if let Some(last) = found
            .last_mut()
            .filter(|old| at - old.frame <= config.min_separation_frames)
        {
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

fn smooth_gravity_apex_near(points: &[BallObservation], frame: i64, max_fit_error_px: f64) -> bool {
    let start = points.partition_point(|point| point.frame < frame - 12);
    let end = points.partition_point(|point| point.frame <= frame + 12);
    let Some((offset, _)) = points[start..end]
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.y.total_cmp(&b.y))
    else {
        return false;
    };
    let apex = start + offset;
    if apex < 5 || apex + 5 >= points.len() || points[apex].frame.abs_diff(frame) > 10 {
        return false;
    }
    let window = &points[apex - 5..=apex + 5];
    if window[10].frame - window[0].frame > 18
        || window
            .windows(2)
            .any(|pair| pair[1].frame - pair[0].frame > 3)
    {
        return false;
    }
    let Some(flight) = fit(window, points[apex].frame) else {
        return false;
    };
    if flight.error > 1.5 * max_fit_error_px
        || flight.y[2] <= 0.0
        || 2.0 * flight.x[2].abs() > 0.8_f64.max(flight.y[2])
    {
        return false;
    }
    let left_t = (window[0].frame - points[apex].frame) as f64;
    let right_t = (window[window.len() - 1].frame - points[apex].frame) as f64;
    let before_vertical = flight.y[1] + 2.0 * flight.y[2] * left_t;
    let after_vertical = flight.y[1] + 2.0 * flight.y[2] * right_t;
    let vertex = flight.y[0] - flight.y[1].powi(2) / (4.0 * flight.y[2]);
    if before_vertical >= -max_fit_error_px / 2.0
        || after_vertical <= max_fit_error_px / 2.0
        || (window[0].y - vertex) < 2.0 * max_fit_error_px
        || (window[window.len() - 1].y - vertex) < 2.0 * max_fit_error_px
    {
        return false;
    }
    let left_dx = window[4].x - window[0].x;
    let right_dx = window[10].x - window[6].x;
    !(left_dx * right_dx < 0.0 && left_dx.abs().min(right_dx.abs()) > 2.0 * max_fit_error_px)
}

fn sustained_down_to_up_rebounds(
    points: &[BallObservation],
    config: DirectionBreakConfig,
    extended: bool,
) -> Vec<TrajectoryBreak> {
    let mut candidates: Vec<TrajectoryBreak> = Vec::new();
    for at in 4..points.len().saturating_sub(4) {
        let contact = points[at];
        let displacement_needed = 10.0 * config.max_fit_error_px;
        let before_count = if extended {
            if at < 6 || contact.y - points[at - 4].y >= displacement_needed {
                continue;
            }
            6
        } else {
            4
        };
        let window = &points[at - before_count..=at + 4];
        let last = window.len() - 1;
        if window[last].frame - window[0].frame > (before_count + 6) as i64
            || window.windows(2).any(|pair| {
                pair[1].frame - pair[0].frame > 2
                    || (pair[1].x - pair[0].x).abs() > 4.0 * config.max_join_error_px
            })
        {
            continue;
        }
        if contact.y - window[0].y < displacement_needed
            || contact.y - window[last].y < displacement_needed
            || window[..=before_count]
                .windows(2)
                .any(|pair| pair[1].y + 2.0 * config.max_fit_error_px < pair[0].y)
            || window[before_count..]
                .windows(2)
                .any(|pair| pair[1].y > pair[0].y + 2.0 * config.max_fit_error_px)
        {
            continue;
        }
        if before_count > 4
            && (window.iter().any(|p| p.y > contact.y)
                || contact.y - points[at - 4].y < 4.0 * config.max_fit_error_px)
        {
            continue;
        }
        let before_span = (contact.frame - window[0].frame) as f64;
        let after_span = (window[last].frame - contact.frame) as f64;
        let before = [
            (contact.x - window[0].x) / before_span,
            (contact.y - window[0].y) / before_span,
        ];
        let after = [
            (window[last].x - contact.x) / after_span,
            (window[last].y - contact.y) / after_span,
        ];
        let before_length = before[0].hypot(before[1]);
        let after_length = after[0].hypot(after[1]);
        if before_length <= 1e-6 || after_length <= 1e-6 {
            continue;
        }
        let cosine = ((before[0] * after[0] + before[1] * after[1])
            / (before_length * after_length))
            .clamp(-1.0, 1.0);
        let angle = cosine.acos().to_degrees();
        if angle < config.min_turn_angle_deg {
            continue;
        }
        let local_x_error =
            (contact.x - 0.5 * (window[before_count - 1].x + window[before_count + 1].x)).abs();
        let candidate = TrajectoryBreak {
            frame: contact.frame,
            position: [contact.x, contact.y],
            motion: MotionEvidence {
                motion_frame: contact.frame,
                speed_before: before_length,
                speed_after: after_length,
                angle,
                residual: (after[0] - before[0]).hypot(after[1] - before[1]),
            },
            fit_error: local_x_error,
            before_frames: window[..before_count].iter().map(|p| p.frame).collect(),
            after_frames: window[before_count + 1..].iter().map(|p| p.frame).collect(),
            contact_observed: true,
        };
        if let Some(prior) = candidates.last_mut().filter(|event| {
            event.frame.abs_diff(candidate.frame) <= config.min_separation_frames as u64
        }) {
            if candidate.position[1] > prior.position[1] {
                *prior = candidate;
            }
        } else {
            candidates.push(candidate);
        }
    }
    candidates
}

fn occluded_vertical_rebounds(
    points: &[BallObservation],
    config: DirectionBreakConfig,
) -> Vec<TrajectoryBreak> {
    let mut results = Vec::new();
    for (index, boundary) in points.windows(2).enumerate() {
        let missing = boundary[1].frame - boundary[0].frame - 1;
        if !(3..=10).contains(&missing) || index < 4 || index + 5 >= points.len() {
            continue;
        }
        let before = &points[index - 4..=index];
        let after = &points[index + 1..=index + 5];
        if before
            .windows(2)
            .chain(after.windows(2))
            .any(|pair| pair[1].frame - pair[0].frame > 2)
        {
            continue;
        }
        let slope = |samples: &[BallObservation]| -> [f64; 2] {
            let duration = (samples.last().unwrap().frame - samples[0].frame) as f64;
            [
                (samples.last().unwrap().x - samples[0].x) / duration,
                (samples.last().unwrap().y - samples[0].y) / duration,
            ]
        };
        let incoming = slope(before);
        let outgoing = slope(after);
        let moderate = incoming[1] >= 10.0 && outgoing[1] <= -7.0;
        let strong = incoming[1] >= 18.0 && outgoing[1] <= -15.0;
        if !moderate
            || (!strong && missing <= 4)
            || before
                .windows(2)
                .any(|pair| pair[1].y + 2.0 * config.max_fit_error_px < pair[0].y)
            || after
                .windows(2)
                .any(|pair| pair[1].y > pair[0].y + 2.0 * config.max_fit_error_px)
        {
            continue;
        }
        let before_speed = incoming[0].hypot(incoming[1]);
        let after_speed = outgoing[0].hypot(outgoing[1]);
        let cosine =
            (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (before_speed * after_speed);
        let angle = cosine.clamp(-1.0, 1.0).acos().to_degrees();
        let residual = (outgoing[0] - incoming[0]).hypot(outgoing[1] - incoming[1]);
        if angle < config.min_turn_angle_deg.max(130.0)
            || residual < if strong { 35.0 } else { 21.0 }
        {
            continue;
        }
        let anchor = *before.last().unwrap();
        let first_outbound = after[0];
        if (first_outbound.x - anchor.x).abs() > (3.0 * config.max_join_error_px).max(85.0) {
            continue;
        }
        if missing <= 4 {
            let span = (first_outbound.frame - anchor.frame) as f64;
            let elapsed =
                (first_outbound.y - anchor.y - outgoing[1] * span) / (incoming[1] - outgoing[1]);
            if !(1.0..=span - 1.0).contains(&elapsed) {
                continue;
            }
            let inbound_x = anchor.x + incoming[0] * elapsed;
            let outbound_x = first_outbound.x - outgoing[0] * (span - elapsed);
            let mismatch = (inbound_x - outbound_x).abs();
            if mismatch > config.max_join_error_px {
                continue;
            }
            let frame = anchor.frame + elapsed.round() as i64;
            results.push(TrajectoryBreak {
                frame,
                position: [
                    (inbound_x + outbound_x) * 0.5,
                    anchor.y + incoming[1] * elapsed,
                ],
                motion: MotionEvidence {
                    motion_frame: frame,
                    speed_before: before_speed,
                    speed_after: after_speed,
                    angle,
                    residual,
                },
                fit_error: mismatch,
                before_frames: before.iter().map(|sample| sample.frame).collect(),
                after_frames: after.iter().map(|sample| sample.frame).collect(),
                contact_observed: false,
            });
            continue;
        }
        let impact_frame = if strong {
            anchor.frame + 1
        } else {
            if index < 6 || index + 7 >= points.len() {
                continue;
            }
            let approach = &points[index - 6..=index];
            let departure = &points[index + 1..=index + 7];
            let long_in = slope(approach);
            let long_out = slope(departure);
            if long_in[1] < 10.0
                || long_out[1] > -7.0
                || approach
                    .windows(2)
                    .chain(departure.windows(2))
                    .any(|p| p[1].frame - p[0].frame > 2)
                || (first_outbound.x - anchor.x).abs() > 45.0
                || approach.windows(2).any(|p| p[1].y + 8.0 < p[0].y)
                || departure.windows(2).any(|p| p[1].y > p[0].y + 8.0)
            {
                continue;
            }
            let span = (first_outbound.frame - anchor.frame) as f64;
            let t = (first_outbound.y - anchor.y - long_out[1] * span) / (long_in[1] - long_out[1]);
            if !(-1.5..=span + 0.5).contains(&t)
                || (anchor.x + long_in[0] * t - first_outbound.x + long_out[0] * (span - t)).abs()
                    > 35.0
            {
                continue;
            }
            anchor.frame + (t.round() as i64).clamp(0, missing + 1)
        };
        let flight_back = (first_outbound.frame - impact_frame) as f64;
        let dt = (impact_frame - anchor.frame) as f64;
        let predicted_before = [anchor.x + incoming[0] * dt, anchor.y + incoming[1] * dt];
        let predicted_after = [
            first_outbound.x - outgoing[0] * flight_back,
            first_outbound.y - outgoing[1] * flight_back,
        ];
        let mismatch = (predicted_before[0] - predicted_after[0])
            .hypot(predicted_before[1] - predicted_after[1]);
        if mismatch > (6.0 * config.max_join_error_px).max(85.0) {
            continue;
        }
        results.push(TrajectoryBreak {
            frame: if strong { anchor.frame } else { impact_frame },
            position: if strong {
                [anchor.x, anchor.y]
            } else {
                predicted_before
            },
            motion: MotionEvidence {
                motion_frame: impact_frame,
                speed_before: before_speed,
                speed_after: after_speed,
                angle,
                residual,
            },
            fit_error: mismatch / (missing as f64 + 1.0),
            before_frames: before.iter().map(|sample| sample.frame).collect(),
            after_frames: after.iter().map(|sample| sample.frame).collect(),
            contact_observed: strong || impact_frame == anchor.frame,
        });
    }
    results
}

fn occluded_lateral_rebounds(
    points: &[BallObservation],
    config: DirectionBreakConfig,
) -> Vec<TrajectoryBreak> {
    let mut found = Vec::new();
    for (index, boundary) in points.windows(2).enumerate() {
        let missing = boundary[1].frame - boundary[0].frame - 1;
        if !(3..=9).contains(&missing) || index < 5 || index + 5 >= points.len() {
            continue;
        }
        let before = &points[index - 5..index];
        let after = &points[index + 1..=index + 5];
        let anchor = before[4];
        let outlier = boundary[0];
        let next = after[0];
        if outlier.frame != anchor.frame + 1
            || next.frame - anchor.frame > 11
            || before
                .windows(2)
                .chain(after.windows(2))
                .any(|p| p[1].frame - p[0].frame != 1)
        {
            continue;
        }
        let incoming = [
            (anchor.x - before[0].x) / 4.0,
            (anchor.y - before[0].y) / 4.0,
        ];
        let outgoing = [(after[4].x - next.x) / 4.0, (after[4].y - next.y) / 4.0];
        let near_outgoing = [(after[2].x - next.x) / 2.0, (after[2].y - next.y) / 2.0];
        let before_speed = incoming[0].hypot(incoming[1]);
        let after_speed = outgoing[0].hypot(outgoing[1]);
        if incoming[0].abs() < 7.0
            || incoming[1] < 0.0
            || incoming[1] > incoming[0].abs()
            || outgoing[0].abs() < 3.0
            || incoming[0] * outgoing[0] >= 0.0
            || outgoing[1] > -15.0
            || near_outgoing[1] >= -5.0
            || before_speed > 60.0
            || after_speed > 60.0
            || (outlier.x - anchor.x).abs()
                < (4.0 * incoming[0].abs()).max(3.0 * config.max_join_error_px)
            || before.windows(2).any(|p| {
                (p[1].x - p[0].x) * incoming[0] < 0.0
                    || p[1].y + 2.0 * config.max_fit_error_px < p[0].y
            })
            || after
                .windows(2)
                .any(|p| (p[1].x - p[0].x) * outgoing[0] < 0.0 || p[1].y > p[0].y)
        {
            continue;
        }
        let angle = ((incoming[0] * outgoing[0] + incoming[1] * outgoing[1])
            / (before_speed * after_speed))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        if angle < config.min_turn_angle_deg.max(100.0) {
            continue;
        }
        let mut best = None::<(i64, [f64; 2], f64)>;
        for frame in outlier.frame..next.frame {
            let dt = (frame - anchor.frame) as f64;
            let back = (next.frame - frame) as f64;
            let a = [anchor.x + incoming[0] * dt, anchor.y + incoming[1] * dt];
            let b = [
                next.x - near_outgoing[0] * back,
                next.y - near_outgoing[1] * back,
            ];
            let error = (a[0] - b[0]).hypot(a[1] - b[1]);
            if (a[0] - b[0]).abs() <= config.max_join_error_px
                && error <= (2.0 * config.max_join_error_px).min(56.0)
                && best.is_none_or(|(_, _, old)| error < old)
            {
                best = Some((frame, [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5], error));
            }
        }
        let Some((frame, position, error)) = best else {
            continue;
        };
        found.push(TrajectoryBreak {
            frame,
            position,
            motion: MotionEvidence {
                motion_frame: frame,
                speed_before: before_speed,
                speed_after: after_speed,
                angle,
                residual: (outgoing[0] - incoming[0]).hypot(outgoing[1] - incoming[1]),
            },
            fit_error: error / (next.frame - anchor.frame) as f64,
            before_frames: before.iter().map(|p| p.frame).collect(),
            after_frames: after.iter().map(|p| p.frame).collect(),
            contact_observed: false,
        });
    }
    found
}

fn occluded_stationary_launches(
    points: &[BallObservation],
    config: DirectionBreakConfig,
) -> Vec<TrajectoryBreak> {
    let mut launches = Vec::new();
    for (index, boundary) in points.windows(2).enumerate() {
        let missing = boundary[1].frame - boundary[0].frame - 1;
        if !(3..=10).contains(&missing) || index < 4 || index + 5 >= points.len() {
            continue;
        }
        let before = &points[index - 4..=index];
        let after = &points[index + 1..=index + 5];
        if before
            .windows(2)
            .chain(after.windows(2))
            .any(|pair| pair[1].frame - pair[0].frame > 2)
        {
            continue;
        }
        let duration_before = (before[4].frame - before[0].frame) as f64;
        let duration_after = (after[4].frame - after[0].frame) as f64;
        let incoming = [
            (before[4].x - before[0].x) / duration_before,
            (before[4].y - before[0].y) / duration_before,
        ];
        let outgoing = [
            (after[4].x - after[0].x) / duration_after,
            (after[4].y - after[0].y) / duration_after,
        ];
        let before_speed = incoming[0].hypot(incoming[1]);
        let after_speed = outgoing[0].hypot(outgoing[1]);
        if before_speed >= 3.0 || !(14.0..=60.0).contains(&after_speed) {
            continue;
        }
        if before
            .iter()
            .any(|point| (point.x - before[4].x).hypot(point.y - before[4].y) > 14.0)
        {
            continue;
        }
        let Some(outbound_fit) = fit(after, after[0].frame) else {
            continue;
        };
        if outbound_fit.error > config.max_fit_error_px {
            continue;
        }
        let mut best = None::<(i64, [f64; 2], f64)>;
        for frame in boundary[0].frame + 1..boundary[1].frame {
            let dt_before = (frame - before[4].frame) as f64;
            let dt_after = (frame - after[0].frame) as f64;
            let projected_before = [
                before[4].x + incoming[0] * dt_before,
                before[4].y + incoming[1] * dt_before,
            ];
            let projected_after = [
                after[0].x + outgoing[0] * dt_after,
                after[0].y + outgoing[1] * dt_after,
            ];
            let mismatch = (projected_before[0] - projected_after[0])
                .hypot(projected_before[1] - projected_after[1]);
            if best.is_none_or(|(_, _, error)| mismatch < error) {
                best = Some((
                    frame,
                    [
                        (projected_before[0] + projected_after[0]) * 0.5,
                        (projected_before[1] + projected_after[1]) * 0.5,
                    ],
                    mismatch,
                ));
            }
        }
        let Some((frame, position, mismatch)) = best else {
            continue;
        };
        if mismatch > config.max_join_error_px.min(20.0) {
            continue;
        }
        let jump = (outgoing[0] - incoming[0]).hypot(outgoing[1] - incoming[1]);
        if jump < 14.0 {
            continue;
        }
        let angle = if before_speed > 1e-6 {
            ((incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (before_speed * after_speed))
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees()
        } else {
            0.0
        };
        launches.push(TrajectoryBreak {
            frame,
            position,
            motion: MotionEvidence {
                motion_frame: frame,
                speed_before: before_speed,
                speed_after: after_speed,
                angle,
                residual: jump,
            },
            fit_error: mismatch / (missing + 1) as f64,
            before_frames: before.iter().map(|sample| sample.frame).collect(),
            after_frames: after.iter().map(|sample| sample.frame).collect(),
            contact_observed: false,
        });
    }
    launches
}

fn descending_lateral_brakes(
    points: &[BallObservation],
    config: DirectionBreakConfig,
) -> Vec<TrajectoryBreak> {
    let mut found: Vec<TrajectoryBreak> = Vec::new();
    for at in 6..points.len().saturating_sub(5) {
        let contact = points[at];
        let left = &points[at - 6..at];
        let right = &points[at + 1..at + 6];
        let first = left[0];
        let last = left[5];
        let next = right[0];
        let end = right[4];
        if contact.frame - last.frame != 1
            || next.frame - contact.frame != 1
            || points[at - 6..=at + 5]
                .windows(2)
                .any(|p| p[1].frame - p[0].frame > 2)
        {
            continue;
        }
        let velocity = |a: BallObservation, b: BallObservation| {
            let dt = (b.frame - a.frame) as f64;
            [(b.x - a.x) / dt, (b.y - a.y) / dt]
        };
        let before = velocity(first, last);
        let after = velocity(next, end);
        let speed_before = before[0].hypot(before[1]);
        let speed_after = after[0].hypot(after[1]);
        let tolerance = config.max_fit_error_px;
        if before[1] < 3.5 * tolerance
            || after[1] < tolerance
            || speed_after > speed_before * 0.72
            || before[0] * after[0] >= 0.0
            || (last.x - first.x).abs().min((end.x - next.x).abs()) < 3.5 * tolerance
            || (contact.x - last.x).abs() > 3.0 * tolerance
            || (contact.x - next.x).abs() > 4.0 * tolerance
            || (contact.y - last.y - before[1]).abs() > 3.0 * tolerance
            || (next.y - contact.y - after[1]).abs() > 1.25 * config.max_join_error_px
        {
            continue;
        }
        let angle = ((before[0] * after[0] + before[1] * after[1]) / (speed_before * speed_after))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        if angle < (config.min_turn_angle_deg * 0.6).max(25.0) || config.min_turn_angle_deg > 60.0 {
            continue;
        }
        let error = |segment: &[BallObservation], v: [f64; 2]| {
            let origin = segment[0];
            (segment
                .iter()
                .map(|p| {
                    let t = (p.frame - origin.frame) as f64;
                    (p.x - origin.x - v[0] * t).powi(2) + (p.y - origin.y - v[1] * t).powi(2)
                })
                .sum::<f64>()
                / segment.len() as f64)
                .sqrt()
        };
        let fit_error = error(left, before).max(error(right, after));
        if fit_error > 2.5 * tolerance {
            continue;
        }
        let candidate = TrajectoryBreak {
            frame: contact.frame,
            position: [contact.x, contact.y],
            motion: MotionEvidence {
                motion_frame: contact.frame,
                speed_before,
                speed_after,
                angle,
                residual: (after[0] - before[0]).hypot(after[1] - before[1]),
            },
            fit_error,
            before_frames: left.iter().map(|p| p.frame).collect(),
            after_frames: right.iter().map(|p| p.frame).collect(),
            contact_observed: true,
        };
        if let Some(prior) = found
            .last_mut()
            .filter(|e| candidate.frame.abs_diff(e.frame) <= config.min_separation_frames as u64)
        {
            if candidate.motion.residual / (1.0 + candidate.fit_error)
                > prior.motion.residual / (1.0 + prior.fit_error)
            {
                *prior = candidate;
            }
        } else {
            found.push(candidate);
        }
    }
    found
}

pub fn trajectory_direction_breaks(
    points: &[BallObservation],
    config: DirectionBreakConfig,
) -> Vec<TrajectoryBreak> {
    if !config.valid()
        || points.len() < 9
        || points.windows(2).any(|pair| pair[1].frame <= pair[0].frame)
        || points.iter().any(|point| {
            point.frame < 0
                || point.frame > i32::MAX as i64
                || !point.x.is_finite()
                || !point.y.is_finite()
        })
    {
        return Vec::new();
    }
    let mut found: Vec<TrajectoryBreak> = Vec::new();
    let mut times = std::collections::BTreeSet::new();
    for i in 4..points.len() - 4 {
        times.insert(points[i].frame);
        let gap = points[i + 1].frame - points[i].frame;
        if (2..=4).contains(&gap) {
            times.insert(points[i].frame + gap / 2);
        }
    }
    for at in times {
        let left_end = points.partition_point(|point| point.frame < at);
        let right_begin = points.partition_point(|point| point.frame <= at);
        let mut left = &points[left_end.saturating_sub(7)..left_end];
        let mut right = &points[right_begin..(right_begin + 7).min(points.len())];
        if left.len() < 4
            || right.len() < 4
            || left.last().is_none_or(|point| at - point.frame > 3)
            || right.first().is_none_or(|point| point.frame - at > 3)
            || left.first().is_none_or(|point| at - point.frame > 12)
            || right.last().is_none_or(|point| point.frame - at > 12)
            || left
                .windows(2)
                .chain(right.windows(2))
                .any(|pair| pair[1].frame - pair[0].frame > 4)
        {
            continue;
        }
        let (Some(mut before), Some(mut after)) = (fit(left, at), fit(right, at)) else {
            continue;
        };
        if before.error > config.max_fit_error_px && left.len() > 5 {
            if let Some(shorter) = fit(&left[1..], at) {
                if shorter.error < before.error {
                    left = &left[1..];
                    before = shorter;
                }
            }
        }
        if after.error > config.max_fit_error_px && right.len() > 5 {
            if let Some(shorter) = fit(&right[..right.len() - 1], at) {
                if shorter.error < after.error {
                    right = &right[..right.len() - 1];
                    after = shorter;
                }
            }
        }
        let error = before.error.max(after.error);
        if error > config.max_fit_error_px {
            continue;
        }
        let join = (before.x[0] - after.x[0]).hypot(before.y[0] - after.y[0]);
        let (bx, by) = (before.x[1], before.y[1]);
        let (ax, ay) = (after.x[1], after.y[1]);
        let before_length = bx.hypot(by);
        let after_length = ax.hypot(ay);
        if before_length < 1e-6 || after_length < 1e-6 {
            continue;
        }
        let cosine = ((bx * ax + by * ay) / (before_length * after_length)).clamp(-1.0, 1.0);
        let angle = cosine.acos().to_degrees();
        if angle < config.min_turn_angle_deg {
            continue;
        }
        if join > config.max_join_error_px {
            let observed_contact = points
                .binary_search_by_key(&at, |point| point.frame)
                .is_ok();
            let missing =
                right[0].frame - left[left.len() - 1].frame - if observed_contact { 2 } else { 1 };
            let sustained_rebound = observed_contact
                && (1..=2).contains(&missing)
                && join <= (3.0 * config.max_join_error_px).min(84.0)
                && angle >= config.min_turn_angle_deg.max(110.0)
                && by >= 0.75 * config.max_fit_error_px
                && ay <= -config.max_fit_error_px
                && ax.abs() >= 2.0 * config.max_fit_error_px
                && left.len() >= 5
                && right.len() >= 5
                && left[left.len() - 4..]
                    .windows(2)
                    .all(|pair| pair[1].y + 2.0 * config.max_fit_error_px >= pair[0].y)
                && right[..4]
                    .windows(2)
                    .all(|pair| pair[1].y <= pair[0].y + 2.0 * config.max_fit_error_px);
            if !sustained_rebound {
                continue;
            }
        }
        let joined: Vec<_> = left.iter().chain(right).copied().collect();
        let Some(smooth) = fit(&joined, at) else {
            continue;
        };
        let lateral_before = left[left.len() - 1].x - left[0].x;
        let lateral_after = right[right.len() - 1].x - right[0].x;
        let sustained_lateral_reversal = lateral_before * lateral_after < 0.0
            && lateral_before.abs().min(lateral_after.abs()) > 2.0 * config.max_fit_error_px;
        let impossible_gravity_reversal = by > 0.0 && ay < 0.0;
        if (smooth.error <= config.max_fit_error_px
            && !sustained_lateral_reversal
            && !impossible_gravity_reversal)
            || error > smooth.error * 0.65
        {
            continue;
        }
        if smooth_gravity_apex_near(points, at, config.max_fit_error_px) {
            continue;
        }
        let candidate = TrajectoryBreak {
            frame: at,
            position: [
                (before.x[0] + after.x[0]) * 0.5,
                (before.y[0] + after.y[0]) * 0.5,
            ],
            motion: MotionEvidence {
                motion_frame: at,
                speed_before: before_length,
                speed_after: after_length,
                angle,
                residual: (ax - bx).hypot(ay - by),
            },
            fit_error: error + join * 0.25,
            before_frames: left.iter().map(|point| point.frame).collect(),
            after_frames: right.iter().map(|point| point.frame).collect(),
            contact_observed: points
                .binary_search_by_key(&at, |point| point.frame)
                .is_ok(),
        };
        let quality = |event: &TrajectoryBreak| event.motion.angle / (1.0 + event.fit_error);
        if let Some(previous) = found
            .last_mut()
            .filter(|prior| at - prior.frame <= config.min_separation_frames)
        {
            if quality(&candidate) > quality(previous) {
                *previous = candidate;
            }
        } else {
            found.push(candidate);
        }
    }
    for rebound in sustained_down_to_up_rebounds(points, config, false) {
        if let Some(nearby) = found.iter_mut().find(|event| {
            event.frame.abs_diff(rebound.frame) <= config.min_separation_frames as u64
        }) {
            if nearby.frame != rebound.frame {
                *nearby = rebound;
            }
        } else {
            found.push(rebound);
        }
    }
    for rebound in occluded_vertical_rebounds(points, config) {
        if let Some(existing) = found.iter_mut().find(|event| {
            event.frame.abs_diff(rebound.frame) <= config.min_separation_frames as u64
        }) {
            if rebound.motion.residual > existing.motion.residual {
                *existing = rebound;
            }
        } else {
            found.push(rebound);
        }
    }
    for launch in occluded_stationary_launches(points, config) {
        if let Some(existing) = found
            .iter_mut()
            .find(|event| event.frame.abs_diff(launch.frame) <= config.min_separation_frames as u64)
        {
            if !existing.contact_observed && launch.motion.residual > existing.motion.residual {
                *existing = launch;
            }
        } else {
            found.push(launch);
        }
    }
    for rebound in sustained_down_to_up_rebounds(points, config, true) {
        if !found
            .iter()
            .any(|event| event.frame.abs_diff(rebound.frame) <= config.min_separation_frames as u64)
        {
            found.push(rebound);
        }
    }
    for rebound in occluded_lateral_rebounds(points, config) {
        if !found
            .iter()
            .any(|event| event.frame.abs_diff(rebound.frame) <= config.min_separation_frames as u64)
        {
            found.push(rebound);
        }
    }
    for brake in descending_lateral_brakes(points, config) {
        if !found
            .iter()
            .any(|e| e.frame.abs_diff(brake.frame) <= config.min_separation_frames as u64)
        {
            found.push(brake);
        }
    }
    found.sort_by_key(|event| event.frame);
    found
}

pub fn downward_to_upward_breaks(points: &[BallObservation]) -> Vec<TrajectoryBreak> {
    trajectory_breaks(points)
        .into_iter()
        .filter(|impact| {
            let left_end = points.partition_point(|p| p.frame < impact.frame);
            let right_begin = points.partition_point(|p| p.frame <= impact.frame);
            let left = &points[left_end.saturating_sub(7)..left_end];
            let right = &points[right_begin..(right_begin + 7).min(points.len())];
            match (fit(left, impact.frame), fit(right, impact.frame)) {
                (Some(before), Some(after)) => before.y[1] > 2.0 && after.y[1] < -2.0,
                _ => false,
            }
        })
        .collect()
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
    fn short_occlusion_preserves_late_vertical_rebound() {
        let points: Vec<_> = (0..=30)
            .filter(|frame| !(12..=15).contains(frame))
            .map(|frame| {
                let t = (frame - 15) as f64;
                BallObservation {
                    frame,
                    x: 900.0 + 4.0 * t,
                    y: 1000.0 + if t < 0.0 { 40.0 * t } else { -45.0 * t },
                }
            })
            .collect();
        let events = trajectory_direction_breaks(&points, DirectionBreakConfig::default());
        assert!(
            events
                .iter()
                .any(|e| (14..=16).contains(&e.frame) && !e.contact_observed),
            "{events:?}"
        );
        let mut switched = points.clone();
        for p in &mut switched {
            if p.frame >= 16 {
                p.x += 250.0;
            }
        }
        assert!(trajectory_direction_breaks(&switched, DirectionBreakConfig::default()).is_empty());
    }

    #[test]
    fn perspective_curvature_on_quantized_gravity_arc_is_not_contact() {
        let points: Vec<_> = (0..=24)
            .map(|frame| {
                let t = (frame - 10) as f64;
                let quantize = |v: f64| (v / 7.5).round() * 7.5;
                BallObservation {
                    frame,
                    x: quantize(1800.0 + 18.0 * t - 0.65 * t * t),
                    y: quantize(400.0 + 1.8 * t * t - if frame == 10 { 7.5 } else { 0.0 }),
                }
            })
            .collect();
        assert!(smooth_gravity_apex_near(&points, 10, 4.0));
        assert!(trajectory_direction_breaks(&points, DirectionBreakConfig::default()).is_empty());
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
    fn tunable_collision_detection_ignores_gravity_and_detects_impulses() {
        let gravity_only: Vec<_> = (0..121)
            .map(|frame| {
                let t = frame as f64 - 60.0;
                BallObservation {
                    frame,
                    x: 380.0 + 3.0 * t,
                    y: 120.0 + 0.12 * t * t,
                }
            })
            .collect();
        for threshold in [2.0, 8.0, 25.0] {
            let config = TrajectoryBreakConfig {
                min_velocity_jump_px_frame: threshold,
                ..TrajectoryBreakConfig::default()
            };
            assert!(trajectory_breaks_configured(&gravity_only, config).is_empty());
        }
        let mut hit = gravity_only;
        for p in &mut hit {
            let t = (p.frame as f64 - 60.0).max(0.0);
            p.x -= 19.0 * t;
            p.y -= 12.0 * t;
        }
        let found = trajectory_breaks_configured(&hit, TrajectoryBreakConfig::default());
        assert_eq!(found.len(), 1, "{found:?}");
        assert!((found[0].frame - 60).abs() <= 1, "{found:?}");
        let threshold = TrajectoryBreakConfig {
            min_velocity_jump_px_frame: 50.0,
            ..TrajectoryBreakConfig::default()
        };
        assert!(trajectory_breaks_configured(&hit, threshold).is_empty());
        assert!(!TrajectoryBreakConfig {
            max_fit_error_px: f64::NAN,
            ..TrajectoryBreakConfig::default()
        }
        .valid());
    }

    #[test]
    fn rounded_slowing_ascent_before_gravity_apex_is_not_a_touch() {
        let points: Vec<BallObservation> = (0..46)
            .map(|frame| {
                let at = frame as f64 - 25.0;
                let early_ascent = (3.0 - at).max(0.0);
                BallObservation {
                    frame,
                    x: ((804.0 + 1.7 * at) / 3.75).round() * 3.75,
                    y: ((305.0 + 0.75 * at * at + 0.09 * early_ascent.powi(3)) / 3.75).round()
                        * 3.75,
                }
            })
            .collect();
        let found = trajectory_direction_breaks(&points, DirectionBreakConfig::default());
        assert!(
            found.is_empty(),
            "Smooth ascent through natural apex gave a false touch: {found:?}"
        );
    }

    fn two_touch_rally_with_six_missing_ball_frames() -> Vec<BallObservation> {
        (620..=668)
            .filter(|frame| !(648..=653).contains(frame))
            .map(|frame| {
                let (x, y) = if frame <= 631 {
                    let t = (frame - 631) as f64;
                    (980.0 - 3.0 * t, 840.0 + 34.0 * t + 0.25 * t * t)
                } else if frame <= 642 {
                    let t = (frame - 631) as f64;
                    (980.0 - 3.0 * t, 840.0 - 35.0 * t + t * t)
                } else if frame <= 647 {
                    let t = (frame - 643) as f64;
                    (947.0 + t * 0.7, 575.0 + t * 0.2)
                } else {
                    let t = (frame - 654) as f64;
                    (955.0 - t * 2.0, 485.0 - 32.0 * t + t * t)
                };
                BallObservation { frame, x, y }
            })
            .collect()
    }

    #[test]
    fn debug_touch_detector_preserves_210_and_217_separate_collisions() {
        let events = trajectory_direction_breaks(
            &two_touch_rally_with_six_missing_ball_frames(),
            DirectionBreakConfig::default(),
        );
        assert!(
            events.iter().any(|e| e.frame.abs_diff(631) <= 2),
            "{events:?}"
        );
        assert!(
            events.iter().any(|e| e.frame.abs_diff(651) <= 1),
            "{events:?}"
        );
        assert!(
            !events.iter().any(|e| (643..=648).contains(&e.frame)),
            "{events:?}"
        );
    }

    #[test]
    fn debug_touch_detector_rejects_spurious_launch_when_gap_is_not_continuous() {
        let base = two_touch_rally_with_six_missing_ball_frames();
        let changed = base
            .iter()
            .copied()
            .map(|mut item| {
                if item.frame >= 654 {
                    item.x += 135.0;
                }
                item
            })
            .collect::<Vec<_>>();
        assert!(
            trajectory_direction_breaks(&changed, DirectionBreakConfig::default())
                .iter()
                .all(|event| !(648..=653).contains(&event.frame))
        );
        let slow = base
            .into_iter()
            .map(|mut item| {
                if item.frame >= 654 {
                    item.x = 955.0;
                    item.y = 574.0 - (item.frame - 654) as f64 * 2.0;
                }
                item
            })
            .collect::<Vec<_>>();
        assert!(
            trajectory_direction_breaks(&slow, DirectionBreakConfig::default())
                .iter()
                .all(|event| !(648..=653).contains(&event.frame))
        );
    }

    #[test]
    fn quantized_gravity_apex_does_not_trigger_direction_break() {
        let observations = [
            (-7, 675.0, 266.25),
            (-6, 678.75, 262.5),
            (-5, 682.5, 258.75),
            (-4, 686.25, 255.0),
            (-3, 686.25, 251.25),
            (-2, 690.0, 251.25),
            (-1, 693.75, 247.5),
            (0, 700.0, 250.0),
            (1, 705.0, 247.5),
            (2, 708.75, 251.25),
            (3, 712.5, 255.0),
            (4, 712.5, 255.0),
            (5, 716.25, 258.75),
            (6, 720.0, 258.75),
            (7, 723.75, 262.5),
        ];
        let points: Vec<_> = observations
            .into_iter()
            .map(|(dt, x, y)| BallObservation {
                frame: 30 + dt,
                x,
                y,
            })
            .collect();
        let found = trajectory_direction_breaks(&points, DirectionBreakConfig::default());
        assert!(
            found.is_empty(),
            "smooth quantized flight generated false contact: {found:?}"
        );
    }

    #[test]
    fn false_track_switch_during_a_descent_is_not_a_rebound() {
        let points: Vec<_> = (0..64)
            .map(|frame| {
                let t = frame as f64 - 32.0;
                BallObservation {
                    frame,
                    x: 480.0 + 2.0 * t - if frame >= 32 { 900.0 } else { 0.0 },
                    y: 330.0 + 7.0 * t + 0.2 * t * t + if frame == 32 { 250.0 } else { 0.0 },
                }
            })
            .collect();
        assert!(
            trajectory_direction_breaks(&points, DirectionBreakConfig::default()).is_empty(),
            "A switched track or a single spurious position is not a confirmed contact"
        );
    }

    #[test]
    fn rebound_precedes_noisy_upper_apex_and_is_timed_at_first_turn() {
        let points: Vec<_> = (40..=108)
            .map(|frame| {
                let t = (frame - 70) as f64;
                let descending = 900.0 + 47.0 * t + 1.5 * t * t;
                let rising = 900.0 - 84.0 * t + 2.0 * t * t;
                let mut y = if t <= 0.0 { descending } else { rising };
                if frame == 71 {
                    y = 889.0;
                }
                if (75..=77).contains(&frame) {
                    y = 820.0;
                }
                BallObservation {
                    frame,
                    x: ((1050.0 + 2.2 * t) / 3.75).round() * 3.75,
                    y: (y / 3.75).round() * 3.75,
                }
            })
            .collect();
        let events = trajectory_direction_breaks(&points, DirectionBreakConfig::default());
        assert!(
            events.iter().any(|event| event.frame.abs_diff(70) <= 1),
            "The descending-to-rising collision was missed: {events:?}"
        );
        assert!(
            events.iter().all(|event| event.frame.abs_diff(91) > 5),
            "The next unforced upper apex is not another contact: {events:?}"
        );
    }

    #[test]
    fn descending_lateral_brake_survives_quantized_tracking() {
        let coords = [
            (0, 52., 0.),
            (1, 45., 23.),
            (2, 45., 52.),
            (3, 37., 75.),
            (4, 30., 105.),
            (5, 30., 135.),
            (6, 22., 165.),
            (7, 22., 195.),
            (8, 15., 225.),
            (9, 30., 265.),
            (10, 45., 287.),
            (11, 45., 295.),
            (12, 48., 306.),
            (13, 57., 322.),
            (14, 70., 344.),
            (15, 90., 373.),
            (16, 110., 405.),
            (17, 127., 420.),
        ];
        let points = coords
            .iter()
            .map(|&(frame, x, y)| BallObservation { frame, x, y })
            .collect::<Vec<_>>();
        let events = trajectory_direction_breaks(&points, DirectionBreakConfig::default());
        assert!(
            events.iter().any(|e| e.frame.abs_diff(8) <= 1),
            "{events:?}"
        );
        let mut smooth = points.clone();
        for p in &mut smooth {
            p.x = 100. - 5. * p.frame as f64;
            p.y = 20. * p.frame as f64 + 0.4 * (p.frame as f64).powi(2);
        }
        assert!(trajectory_direction_breaks(&smooth, DirectionBreakConfig::default()).is_empty());
    }
    #[test]
    fn directional_turn_is_detected_without_velocity_threshold() {
        let settings = DirectionBreakConfig::default();
        let smooth = flight(false);
        assert!(trajectory_direction_breaks(&smooth, settings).is_empty());
        let mut quantized = smooth.clone();
        for point in &mut quantized {
            point.x = (point.x / 2.0).round() * 2.0;
            point.y = (point.y / 3.0).round() * 3.0;
        }
        assert!(trajectory_direction_breaks(&quantized, settings).is_empty());
        let collision = trajectory_direction_breaks(&flight(true), settings);
        assert_eq!(collision.len(), 1, "{collision:?}");
        assert!((collision[0].frame - 30).abs() <= 1);
        assert!(collision[0].motion.angle >= 45.0);
        let stricter = DirectionBreakConfig {
            min_turn_angle_deg: 170.0,
            ..settings
        };
        assert!(trajectory_direction_breaks(&flight(true), stricter).is_empty());
        assert!(!DirectionBreakConfig {
            max_join_error_px: f64::NAN,
            ..settings
        }
        .valid());
    }

    #[test]
    fn direction_break_survives_isolated_false_ball_and_one_missing_detection() {
        let samples: Vec<_> = (0..84)
            .filter(|frame| *frame != 41)
            .map(|frame| {
                let t = frame as f64 - 40.0;
                let turn = t.max(0.0);
                let jitter = (frame % 5) as f64 - 2.0;
                BallObservation {
                    frame,
                    x: 400.0 - 4.0 * t
                        + 11.0 * turn
                        + jitter * 0.8
                        + if frame == 45 { 1100.0 } else { 0.0 },
                    y: 540.0 + 12.0 * t - 26.0 * turn
                        + 0.15 * t * t
                        + jitter * 1.1
                        + if frame == 45 { 80.0 } else { 0.0 },
                }
            })
            .collect();
        let cleaned =
            crate::ball_interpolation::reconstruct_short_ball_motion(&samples, 1920, 1080, 28.0)
                .unwrap();
        assert_eq!(cleaned.rejected_frames, [45]);
        let events =
            trajectory_direction_breaks(&cleaned.observations, DirectionBreakConfig::default());
        assert_eq!(
            events.len(),
            1,
            "unreliable single frame masked or duplicated the contact: {events:?}"
        );
        assert!((events[0].frame - 40).abs() <= 2, "{events:?}");
    }

    #[test]
    fn isolated_ball_jump_is_not_a_touch_in_smooth_gravity_flight() {
        let samples: Vec<_> = (0..84)
            .map(|frame| {
                let t = frame as f64 - 42.0;
                BallObservation {
                    frame,
                    x: 300.0 + 5.0 * t + if frame == 45 { 1200.0 } else { 0.0 },
                    y: 250.0 + 0.19 * t * t,
                }
            })
            .collect();
        assert!(
            trajectory_direction_breaks(&samples, DirectionBreakConfig::default()).is_empty(),
            "An isolated false ball detection is not an impact"
        );
    }

    #[test]
    fn direction_break_survives_moderate_extrapolation_gap() {
        let points: Vec<_> = (0..100)
            .map(|frame| {
                let t = frame as f64 - 50.0;
                let dy = 9.0 * t + 0.13 * t * t - 21.0 * t.max(0.0);
                BallObservation {
                    frame,
                    x: 420.0 + 2.0 * t - 3.0 * t.max(0.0),
                    y: 310.0 + dy,
                }
            })
            .collect();
        assert!(
            trajectory_direction_breaks(
                &points,
                DirectionBreakConfig {
                    max_join_error_px: 8.0,
                    ..DirectionBreakConfig::default()
                }
            )
            .len()
                <= 1
        );
        let found = trajectory_direction_breaks(&points, DirectionBreakConfig::default());
        assert_eq!(found.len(), 1, "{found:?}");
        assert!((found[0].frame - 50).abs() <= 1);
    }

    #[test]
    fn ball_stabilizer_eight_frame_dropout_preserves_down_to_up_contact() {
        let synthetic = |gap: std::ops::RangeInclusive<i64>, jump: f64| {
            (400..=444)
                .filter(|frame| !gap.contains(frame))
                .map(|frame| {
                    let t = (frame - 419) as f64;
                    let before = frame <= 419;
                    BallObservation {
                        frame,
                        x: 1035.0 + 2.5 * t + if before { 0.0 } else { jump },
                        y: if before {
                            820.0 + 48.0 * t + 0.15 * t * t
                        } else {
                            790.0 - 42.0 * t + 0.20 * t * t
                        },
                    }
                })
                .collect::<Vec<_>>()
        };
        let points = synthetic(420..=427, 0.0);
        let impacts = trajectory_direction_breaks(&points, DirectionBreakConfig::default());
        assert!(
            impacts
                .iter()
                .any(|contact| (418..=421).contains(&contact.frame)),
            "Eight rejected observations must not hide the impact near frame 420: {impacts:?}"
        );
        let switched = synthetic(420..=427, 260.0);
        assert!(trajectory_direction_breaks(&switched, DirectionBreakConfig::default()).is_empty());
        let too_long = synthetic(420..=431, 0.0);
        assert!(trajectory_direction_breaks(&too_long, DirectionBreakConfig::default()).is_empty());
        let apex = (400..=444)
            .filter(|frame| !(420..=427).contains(frame))
            .map(|frame| {
                let t = (frame - 420) as f64;
                BallObservation {
                    frame,
                    x: 1035.0 + 2.5 * t,
                    y: 350.0 + 0.2 * t * t,
                }
            })
            .collect::<Vec<_>>();
        assert!(trajectory_direction_breaks(&apex, DirectionBreakConfig::default()).is_empty());
    }
    #[test]
    fn moderate_rebound_with_short_quantized_pause_is_still_a_contact() {
        let raw: Vec<_> = (0..68)
            .map(|frame| {
                let t = frame as f64 - 30.0;
                let x = if frame < 28 {
                    600.0 + 7.0 * t
                } else if frame <= 30 {
                    600.0
                } else {
                    600.0 - 5.0 * t
                };
                let y = if frame < 28 {
                    450.0 + 11.0 * t + 0.05 * t * t
                } else if frame <= 30 {
                    450.0 - if frame == 29 { 4.0 } else { 0.0 }
                } else {
                    450.0 - 12.0 * t + 0.06 * t * t
                };
                BallObservation {
                    frame,
                    x: (x / 3.75).round() * 3.75,
                    y: (y / 3.75).round() * 3.75,
                }
            })
            .collect();
        let cleaned =
            crate::ball_interpolation::reconstruct_short_ball_motion(&raw, 1920, 1080, 28.0)
                .unwrap();
        let impacts =
            trajectory_direction_breaks(&cleaned.observations, DirectionBreakConfig::default());
        assert!(
            impacts.iter().any(|impact| impact.frame.abs_diff(30) <= 1),
            "A sustained direction reversal was rejected: {impacts:?}"
        );
        assert_eq!(
            impacts.len(),
            1,
            "Unexpected duplicate contacts: {impacts:?}"
        );
    }

    #[test]
    fn noisy_gap_at_a_sustained_sideways_rebound_still_yields_one_contact() {
        let raw: Vec<_> = (12..65)
            .map(|frame| {
                let t = frame as f64 - 25.0;
                let (x, y) = if frame <= 25 {
                    (900.0 + 4.0 * t, 430.0 + 5.0 * t - 0.6 * t * t)
                } else if frame == 26 {
                    (1790.0, 620.0)
                } else {
                    let after = frame as f64 - 27.0;
                    (
                        825.0 - 15.0 * after,
                        385.0 - 4.0 * after + 0.2 * after * after,
                    )
                };
                BallObservation {
                    frame,
                    x: (x / 3.75).round() * 3.75,
                    y: (y / 3.75).round() * 3.75,
                }
            })
            .collect();
        let cleaned =
            crate::ball_interpolation::reconstruct_short_ball_motion(&raw, 1920, 1080, 28.0)
                .unwrap();
        assert_eq!(cleaned.rejected_frames, [26]);
        let impacts =
            trajectory_direction_breaks(&cleaned.observations, DirectionBreakConfig::default());
        assert!(
            impacts.iter().any(|impact| (24..=26).contains(&impact.frame)),
            "A sustained downward-to-sideways reversal was missed after removing a false ball: {impacts:?}"
        );
        assert_eq!(
            impacts.len(),
            1,
            "Unexpected duplicate contacts: {impacts:?}"
        );

        let switched: Vec<_> = raw
            .iter()
            .copied()
            .map(|mut point| {
                if point.frame >= 27 {
                    point.x += 240.0;
                }
                point
            })
            .collect();
        let cleaned =
            crate::ball_interpolation::reconstruct_short_ball_motion(&switched, 1920, 1080, 28.0)
                .unwrap();
        assert_eq!(cleaned.rejected_frames, [26]);
        assert!(
            trajectory_direction_breaks(&cleaned.observations, DirectionBreakConfig::default())
                .is_empty(),
            "A distant track switch after a dropout is not a contact"
        );
    }

    #[test]
    fn velocity_break_is_found_without_any_players() {
        let found = trajectory_breaks(&flight(true));
        assert_eq!(found.len(), 1, "{found:?}");
        assert!((found[0].frame - 30).abs() <= 1, "{found:?}");
    }

    #[test]
    fn preliminary_impacts_only_count_descending_to_ascending_ball() {
        let rebound = downward_to_upward_breaks(&flight(true));
        assert_eq!(rebound.len(), 1, "{rebound:?}");
        assert!((rebound[0].frame - 30).abs() <= 1);
        assert!(downward_to_upward_breaks(&flight(false)).is_empty());
        let side_change: Vec<_> = flight(true)
            .into_iter()
            .map(|mut p| {
                let t = p.frame as f64 - 30.0;
                p.x = 400.0 + if t > 0.0 { -8.0 } else { 8.0 } * t;
                p.y = 200.0 + t * 5.0;
                p
            })
            .collect();
        assert!(downward_to_upward_breaks(&side_change).is_empty());
        let mut noise = flight(false);
        noise[30].y += 55.0;
        assert!(downward_to_upward_breaks(&noise).is_empty());
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

    #[test]
    fn slow_quantized_overhead_set_has_a_sustained_reversal() {
        let coords = [
            (1740., 360.),
            (1732.5, 367.5),
            (1725., 375.),
            (1717.5, 382.5),
            (1717.5, 390.),
            (1702.5, 420.),
            (1710., 427.5),
            (1702.5, 442.5),
            (1695., 442.5),
            (1695., 450.),
            (1695., 442.5),
            (1695., 420.),
            (1687.5, 405.),
            (1680., 367.5),
            (1680., 345.),
            (1687.5, 322.5),
            (1687.5, 300.),
            (1687.5, 277.5),
            (1687.5, 255.),
            (1687.5, 232.5),
        ];
        let points: Vec<_> = coords
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| BallObservation {
                frame: i as i64,
                x,
                y,
            })
            .collect();
        let events = trajectory_direction_breaks(&points, DirectionBreakConfig::default());
        assert_eq!(events.len(), 1, "{events:?}");
        assert!(events[0].frame.abs_diff(9) <= 1, "{events:?}");
    }

    fn recorded_occluded_forearm_points() -> Vec<BallObservation> {
        let coords = [
            (0, 1747.5, 570.),
            (1, 1725., 570.),
            (2, 1695., 570.),
            (3, 1672.5, 570.),
            (4, 1657.5, 570.),
            (5, 1635., 577.5),
            (6, 1612.5, 585.),
            (7, 1605., 592.5),
            (8, 1597.5, 592.5),
            (9, 1590., 592.5),
            (10, 1440., 600.),
            (17, 1522.5, 660.),
            (18, 1522.5, 652.5),
            (19, 1530., 637.5),
            (20, 1530., 615.),
            (21, 1537.5, 577.5),
            (22, 1552.5, 555.),
            (23, 1560., 517.5),
            (24, 1575., 480.),
            (25, 1582.5, 457.5),
            (26, 1597.5, 427.5),
            (27, 1605., 405.),
        ];
        coords
            .into_iter()
            .map(|(frame, x, y)| BallObservation { frame, x, y })
            .collect()
    }

    #[test]
    fn occluded_forearm_reversal_survives_isolated_pre_gap_outlier() {
        let points = recorded_occluded_forearm_points();
        let found = trajectory_direction_breaks(&points, DirectionBreakConfig::default());
        let target: Vec<_> = found
            .iter()
            .filter(|e| (10..17).contains(&e.frame))
            .collect();
        assert_eq!(target.len(), 1, "{found:?}");
        assert!(!target[0].contact_observed);
        assert!(!target[0].before_frames.contains(&10));
    }

    #[test]
    fn occluded_forearm_rejects_unrelated_or_nonreversing_track() {
        for shift in [300., -300.] {
            let mut points = recorded_occluded_forearm_points();
            for point in points.iter_mut().filter(|p| p.frame >= 17) {
                point.x += shift;
            }
            assert!(
                trajectory_direction_breaks(&points, DirectionBreakConfig::default())
                    .iter()
                    .all(|e| !(10..17).contains(&e.frame))
            );
        }
        let mut points = recorded_occluded_forearm_points();
        for point in points.iter_mut().filter(|p| p.frame >= 17) {
            point.x = 1522.5 - (point.frame - 17) as f64 * 8.;
        }
        assert!(
            trajectory_direction_breaks(&points, DirectionBreakConfig::default())
                .iter()
                .all(|e| !(10..17).contains(&e.frame))
        );
    }
    #[test]
    fn moderate_occluded_reversal_with_quantized_ball() {
        let samples = |shift: f64, reversed: bool| {
            (4..=36)
                .filter(|f| !(21..=26).contains(f) && *f != 18)
                .map(|frame| {
                    let t = (frame - 20) as f64;
                    let (x, y) = if frame <= 20 {
                        (2050.0 - 8.0 * t, 600.0 + 14.0 * t)
                    } else if reversed {
                        (2050.0 - t + shift, 624.0 - 10.0 * t)
                    } else {
                        (2050.0 - 8.0 * t, 600.0 + 14.0 * t)
                    };
                    BallObservation {
                        frame,
                        x: (x / 7.5).round() * 7.5,
                        y: (y / 7.5).round() * 7.5,
                    }
                })
                .collect::<Vec<_>>()
        };
        let find = |points: Vec<BallObservation>| {
            trajectory_direction_breaks(&points, DirectionBreakConfig::default())
                .iter()
                .filter(|impact| (19..=26).contains(&impact.frame))
                .count()
        };
        assert_eq!(find(samples(0.0, true)), 1);
        assert_eq!(find(samples(120.0, true)), 0);
        assert_eq!(find(samples(0.0, false)), 0);
    }
}
