use super::{BallObservation, TrajectoryBreak};
use crate::{
    court_geometry::CourtTransform,
    rally_reconstruction::serve_flight::fit_projected_ballistic_flight,
};

pub const GROUND_DETECTOR_VERSION: &str = "court_ball_roll_v13";

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GroundContact {
    pub frame: u64,
    pub position: [f64; 2],
    pub landing: String,
}

pub fn classify_ground_contact(
    points: &[BallObservation],
    impact: &TrajectoryBreak,
    court: &CourtTransform,
    width: u32,
    height: u32,
    rally_end_frame: u64,
) -> Option<GroundContact> {
    classify_ground_contact_with_observed(
        points,
        points,
        impact,
        court,
        width,
        height,
        rally_end_frame,
    )
}

pub fn classify_ground_contact_with_observed(
    points: &[BallObservation],
    observed: &[BallObservation],
    impact: &TrajectoryBreak,
    court: &CourtTransform,
    width: u32,
    height: u32,
    rally_end_frame: u64,
) -> Option<GroundContact> {
    if width == 0
        || height == 0
        || impact.frame < 0
        || impact.before_frames.len() < 4
        || impact.after_frames.len() < 4
    {
        return None;
    }
    if airborne_contact_supported(observed, impact.frame, court, width, height) {
        return None;
    }
    if let Some(contact) = short_damped_out_rebound(points, impact, court, width, height) {
        return Some(contact);
    }
    if let Some(contact) = distant_sand_rebound(points, impact, court, width, height) {
        return Some(contact);
    }
    let get = |frame: i64| -> Option<BallObservation> {
        points
            .binary_search_by_key(&frame, |point| point.frame)
            .ok()
            .map(|index| points[index])
    };
    let before_start = get(*impact.before_frames.first()?)?;
    let before_end = get(*impact.before_frames.last()?)?;
    let after_start = get(*impact.after_frames.first()?)?;
    let after_end = get(*impact.after_frames.last()?)?;
    if !(before_start.frame < before_end.frame
        && before_end.frame <= impact.frame
        && impact.frame < after_start.frame
        && after_start.frame < after_end.frame)
    {
        return None;
    }
    let incoming = (before_end.y - before_start.y) / (before_end.frame - before_start.frame) as f64;
    let outgoing = (after_end.y - after_start.y) / (after_end.frame - after_start.frame) as f64;
    let weak_rebound = outgoing <= -1.0 && -outgoing <= incoming * 0.75;
    let rebound_or_settle = weak_rebound || (incoming >= 3.0 && outgoing.abs() <= 1.5);
    if incoming < 2.0
        || !rebound_or_settle
        || impact.position[1] + 8.0 + incoming * ((impact.frame - before_end.frame) as f64)
            < before_end.y
        || impact.position[1] + 8.0 + outgoing.abs() * ((after_start.frame - impact.frame) as f64)
            < after_start.y
    {
        return None;
    }
    let position = [
        impact.position[0] / f64::from(width),
        impact.position[1] / f64::from(height),
    ];
    if position
        .iter()
        .any(|value| !value.is_finite() || !(-0.5..=1.5).contains(value))
    {
        return None;
    }
    // Avoid airborne turnarounds above the court and background
    // collisions far beyond the lines. Permit nearby OUT landings.
    let zones = court.zones()?;
    let polygon = [
        zones.near_half[0],
        zones.far_half[1],
        zones.far_half[2],
        zones.near_half[3],
    ];
    let top_y = polygon.iter().map(|p| p[1]).reduce(f64::min)?;
    if position[1] < top_y + 0.035 {
        return None;
    }
    let mut has_negative = false;
    let mut has_positive = false;
    let mut boundary_distance = f64::INFINITY;
    for index in 0..4 {
        let start = polygon[index];
        let end = polygon[(index + 1) % 4];
        let edge = [end[0] - start[0], end[1] - start[1]];
        let offset = [position[0] - start[0], position[1] - start[1]];
        let cross = edge[0] * offset[1] - edge[1] * offset[0];
        has_negative |= cross < -1e-9;
        has_positive |= cross > 1e-9;
        let squared_length = edge[0] * edge[0] + edge[1] * edge[1];
        if squared_length < 1e-10 {
            return None;
        }
        let t = ((offset[0] * edge[0] + offset[1] * edge[1]) / squared_length).clamp(0.0, 1.0);
        let dx = offset[0] - t * edge[0];
        let dy = offset[1] - t * edge[1];
        boundary_distance = boundary_distance.min(dx.hypot(dy));
    }
    if has_negative && has_positive && boundary_distance > 0.09 {
        return None;
    }
    let [x_m, y_m] = court.to_court(position)?;
    let distance_m = (x_m.abs() - crate::court_geometry::COURT_LENGTH_M / 2.0)
        .max(y_m.abs() - crate::court_geometry::COURT_WIDTH_M / 2.0);
    let landing = if distance_m.abs() <= 0.25 {
        "line_uncertain"
    } else {
        court.tentative_landing(position)?
    };
    if rally_end_frame.saturating_sub(impact.frame as u64) > 45
        && (!impact.contact_observed || !matches!(landing, "line_uncertain" | "out"))
    {
        return None;
    }
    Some(GroundContact {
        frame: impact.frame as u64,
        position,
        landing: landing.to_owned(),
    })
}

fn contact_flight_fit(
    points: &[BallObservation],
    frame: i64,
    before: bool,
    projection: &[[f64; 4]; 3],
    dimensions: [f64; 2],
    window: i64,
) -> Option<[f64; 6]> {
    if !(0..=i32::MAX as i64).contains(&frame) {
        return None;
    }
    let (start, end) = if before {
        (frame - window, frame)
    } else {
        (frame, frame + window)
    };
    let lo = points.partition_point(|p| p.frame < start);
    let hi = points.partition_point(|p| p.frame <= end);
    let samples = points.get(lo..hi)?;
    if samples.len() * 10 < (window as usize + 1) * 7
        || (samples.first()?.frame - start) > 3
        || (end - samples.last()?.frame) > 3
        || samples
            .windows(2)
            .any(|p| p[1].frame <= p[0].frame || p[1].frame - p[0].frame > 4)
    {
        return None;
    }
    let normalized = |p: &BallObservation| {
        (
            (p.frame - frame) as f64 / 30.0,
            [p.x / dimensions[0], p.y / dimensions[1]],
        )
    };
    let fit = fit_projected_ballistic_flight(projection, samples.iter().map(normalized))?;
    let [x, y, z, vx, vy, vz] = fit;
    if x.abs() > 20.0
        || y.abs() > 10.0
        || !(-1.0..=8.0).contains(&z)
        || vx.hypot(vy).hypot(vz) > 35.0
    {
        return None;
    }
    let mut squared_error = 0.0;
    for (t, point) in samples.iter().map(normalized) {
        let position = [x + vx * t, y + vy * t, z + vz * t - 4.905 * t * t, 1.0];
        let projected =
            projection.map(|row| row.iter().zip(position).map(|(a, b)| a * b).sum::<f64>());
        if projected[2] <= 0.0 {
            return None;
        }
        let residual =
            (projected[0] / projected[2] - point[0]).hypot(projected[1] / projected[2] - point[1]);
        if !residual.is_finite() || residual > 0.05 {
            return None;
        }
        squared_error += residual * residual;
    }
    ((squared_error / samples.len() as f64).sqrt() <= 0.015).then_some(fit)
}

fn consistent_contact_fits(
    points: &[BallObservation],
    frame: i64,
    before: bool,
    projection: &[[f64; 4]; 3],
    dimensions: [f64; 2],
) -> Option<[[f64; 6]; 2]> {
    let a = contact_flight_fit(points, frame, before, projection, dimensions, 24)?;
    let b = contact_flight_fit(points, frame, before, projection, dimensions, 30)?;
    ((a[2] - b[2]).abs() <= 0.6 && (a[0] - b[0]).hypot(a[1] - b[1]) <= 2.5).then_some([a, b])
}

fn airborne_contact_supported(
    points: &[BallObservation],
    frame: i64,
    court: &CourtTransform,
    width: u32,
    height: u32,
) -> bool {
    if width == 0 || height == 0 {
        return false;
    }
    let Some(projection) = court.flight_projection(2.43) else {
        return false;
    };
    let dimensions = [f64::from(width), f64::from(height)];
    let Some(before) = consistent_contact_fits(points, frame, true, &projection, dimensions) else {
        return false;
    };
    let Some(after) = consistent_contact_fits(points, frame, false, &projection, dimensions) else {
        return false;
    };
    before.iter().chain(&after).all(|fit| fit[2] >= 0.75)
        && (before[1][2] - after[1][2]).abs() <= 0.75
}

pub fn terminal_loss_ground_contacts(
    observed: &[BallObservation],
    court: &CourtTransform,
    width: u32,
    height: u32,
    rally_end_frame: u64,
) -> Vec<GroundContact> {
    if width == 0
        || height == 0
        || observed
            .iter()
            .any(|p| !(0..=i32::MAX as i64).contains(&p.frame))
        || observed.windows(2).any(|p| p[0].frame >= p[1].frame)
    {
        return Vec::new();
    }
    let Some(projection) = court.flight_projection(2.43) else {
        return Vec::new();
    };
    let dimensions = [f64::from(width), f64::from(height)];
    observed
        .windows(2)
        .filter_map(|pair| {
            let last = pair[0];
            if last.frame < 0
                || last.frame as u64 > rally_end_frame
                || pair[1].frame - last.frame < 12
            {
                return None;
            }
            let first = observed.partition_point(|p| p.frame < last.frame - 2);
            let stop = observed.partition_point(|p| p.frame <= last.frame);
            let (impact, x, y, position) = observed[first..stop].iter().find_map(|p| {
                let fits =
                    consistent_contact_fits(observed, p.frame, true, &projection, dimensions)?;
                if !fits
                    .iter()
                    .all(|f| (-0.25..=0.40).contains(&f[2]) && f[5] < -2.0)
                {
                    return None;
                }
                let [x, y, ..] = fits[1];
                let position = court.floor_to_image([x, y])?;
                let measured = [p.x / dimensions[0], p.y / dimensions[1]];
                ((position[0] - measured[0]).hypot(position[1] - measured[1]) <= 0.05
                    && x.abs() <= 12.0
                    && y.abs() <= 7.5)
                    .then_some((p.frame, x, y, position))
            })?;
            let distance = (x.abs() - crate::court_geometry::COURT_LENGTH_M / 2.0)
                .max(y.abs() - crate::court_geometry::COURT_WIDTH_M / 2.0);
            let landing = if distance.abs() <= 0.25 {
                "line_uncertain"
            } else {
                court.tentative_landing(position)?
            };
            Some(GroundContact {
                frame: impact as u64,
                position,
                landing: landing.to_owned(),
            })
        })
        .collect()
}

fn short_damped_out_rebound(
    points: &[BallObservation],
    impact: &TrajectoryBreak,
    court: &CourtTransform,
    width: u32,
    height: u32,
) -> Option<GroundContact> {
    if !impact.contact_observed {
        return None;
    }
    let [w, h] = [f64::from(width), f64::from(height)];
    let peak = points
        .iter()
        .filter(|p| p.frame.abs_diff(impact.frame) <= 2)
        .max_by(|a, b| a.y.total_cmp(&b.y))?;
    let position = [peak.x / w, peak.y / h];
    let zones = court.zones()?;
    let top = zones
        .near_half
        .iter()
        .chain(&zones.far_half)
        .map(|p| p[1])
        .reduce(f64::min)?;
    if !(top - 0.045..top - 0.005).contains(&position[1]) {
        return None;
    }
    let [x, y] = court.to_court(position)?;
    let outside = (x.abs() - crate::court_geometry::COURT_LENGTH_M / 2.0)
        .max(y.abs() - crate::court_geometry::COURT_WIDTH_M / 2.0);
    if !(0.25..=3.0).contains(&outside) {
        return None;
    }
    let frame = peak.frame;
    let lo = points.partition_point(|p| p.frame < frame - 17);
    let mid = points.partition_point(|p| p.frame <= frame);
    let hi = points.partition_point(|p| p.frame <= frame + 40);
    let before = &points[lo..mid];
    let after = &points[mid..hi];
    let tail_len = after
        .windows(2)
        .position(|p| p[1].frame - p[0].frame > 6)
        .map_or(after.len(), |i| i + 1);
    let tail = &after[..tail_len];
    let start = before.first()?;
    let last = tail.last()?;
    let descent = (peak.y - start.y) / h;
    if before.len() < 12
        || tail.len() < 10
        || frame - start.frame < 12
        || descent < 0.045
        || before
            .windows(2)
            .any(|p| p[1].frame - p[0].frame > 3 || p[1].y + h * 0.008 < p[0].y)
        || tail[0].frame - frame > 2
        || !(10..=20).contains(&(last.frame - frame))
        || after
            .get(tail_len)
            .is_some_and(|p| p.frame - last.frame < 16)
        || tail
            .iter()
            .any(|p| (p.x - peak.x).abs() > w * 0.025 || p.y > peak.y + h * 0.006)
    {
        return None;
    }
    let apex = tail.iter().min_by(|a, b| a.y.total_cmp(&b.y))?;
    let rise = (peak.y - apex.y) / h;
    if !(5..=13).contains(&(apex.frame - frame))
        || !(0.008..=0.024).contains(&rise)
        || rise > descent * 0.4
        || last.frame - apex.frame < 3
        || (last.y - apex.y) / h < rise * 0.35
        || (last.y - peak.y).abs() > h * 0.012
    {
        return None;
    }
    Some(GroundContact {
        frame: impact.frame as u64,
        position,
        landing: "out".to_owned(),
    })
}
fn distant_sand_rebound(
    points: &[BallObservation],
    impact: &TrajectoryBreak,
    court: &CourtTransform,
    width: u32,
    height: u32,
) -> Option<GroundContact> {
    let [w, h] = [f64::from(width), f64::from(height)];
    let position = [impact.position[0] / w, impact.position[1] / h];
    if position.iter().any(|p| !(-0.5..=1.5).contains(p)) {
        return None;
    }
    let zones = court.zones()?;
    let top = zones
        .near_half
        .iter()
        .chain(&zones.far_half)
        .map(|p| p[1])
        .reduce(f64::min)?;
    if !(top - 0.005..=top + 0.12).contains(&position[1]) {
        return None;
    }
    let [x, y] = court.to_court(position)?;
    let outside = (x.abs() - crate::court_geometry::COURT_LENGTH_M / 2.0)
        .max(y.abs() - crate::court_geometry::COURT_WIDTH_M / 2.0);
    if !(0.25..=5.0).contains(&outside) {
        return None;
    }
    let frame = impact.frame;
    let lo = points.partition_point(|p| p.frame < frame.saturating_sub(18));
    let hi = points.partition_point(|p| p.frame <= frame.saturating_add(48));
    let local = &points[lo..hi];
    let split = local.partition_point(|p| p.frame <= frame);
    let (before, after) = local.split_at(split);
    let contiguous_end = after
        .windows(2)
        .position(|p| p[1].frame - p[0].frame > 6)
        .map_or(after.len(), |index| index + 1);
    let after = &after[..contiguous_end];
    if before.len() < 8
        || after.len() < 12
        || before.windows(2).any(|p| p[1].frame - p[0].frame > 6)
        || frame - before.last()?.frame > 3
        || after.first()?.frame - frame > 3
    {
        return None;
    }
    let start = before.first()?;
    let approach = before.last()?;
    let descent = position[1] - start.y / h;
    if frame - start.frame < 12
        || descent < 0.075
        || (approach.y / h - position[1]).abs() > 0.018
        || before.windows(2).any(|p| p[1].y + h * 0.012 < p[0].y)
    {
        return None;
    }
    let apex = after
        .iter()
        .enumerate()
        .filter(|(_, p)| (frame + 5..=frame + 30).contains(&p.frame))
        .min_by(|(_, a), (_, b)| a.y.total_cmp(&b.y))?;
    let rise = position[1] - apex.1.y / h;
    if rise < 0.025
        || rise > descent * 0.75
        || after.iter().any(|p| (p.x / w - position[0]).abs() > 0.06)
    {
        return None;
    }
    let returned = after[apex.0 + 1..].windows(4).any(|window| {
        window[0].frame - apex.1.frame >= 6
            && window[3].frame - window[0].frame <= 8
            && window
                .iter()
                .all(|p| (p.y / h - position[1]).abs() <= 0.018)
            && (window[3].y - window[0].y).abs() <= h * 0.015
    });
    let damped_return = after[apex.0 + 1..].windows(2).any(|window| {
        let elapsed = (window[1].frame - apex.1.frame) as f64;
        let recovery = window[1].y / h - apex.1.y / h;
        let approach_speed = descent / (frame - start.frame) as f64;
        elapsed >= 8.0
            && window[1].frame - window[0].frame <= 6
            && window.iter().all(|p| p.y / h - apex.1.y / h >= rise * 0.65)
            && (window[1].y / h - position[1]).abs() <= 0.035
            && recovery / elapsed <= approach_speed * 0.65
            && window[1].y + h * 0.006 >= window[0].y
    });
    (returned || damped_return).then_some(GroundContact {
        frame: frame as u64,
        position,
        landing: "out".to_owned(),
    })
}

pub fn rolling_ground_contacts(
    points: &[BallObservation],
    court: &CourtTransform,
    width: u32,
    height: u32,
    rally_end_frame: u64,
) -> Vec<GroundContact> {
    if width == 0
        || height == 0
        || points.windows(2).any(|p| p[0].frame >= p[1].frame)
        || points.iter().any(|p| {
            p.frame < 0 || p.frame > i32::MAX as i64 || !p.x.is_finite() || !p.y.is_finite()
        })
    {
        return Vec::new();
    }
    let mut contacts: Vec<GroundContact> = Vec::new();
    for split in 5..points.len().saturating_sub(4) {
        let before = &points[split - 5..split];
        let left = before[4];
        let right = points[split];
        let gap = right.frame - left.frame - 1;
        if gap > 12 || before.windows(2).any(|p| p[1].frame - p[0].frame > 2) {
            continue;
        }
        let after_end = if gap <= 1 {
            points.partition_point(|p| p.frame <= right.frame + 12)
        } else {
            split + 5
        };
        let after = &points[split..after_end];
        if after.len() < 5
            || after.last().unwrap().frame - right.frame < if gap <= 1 { 10 } else { 4 }
            || after.last().unwrap().frame - right.frame > 12
            || after
                .windows(2)
                .any(|p| p[1].frame - p[0].frame > if gap <= 1 { 2 } else { 6 })
        {
            continue;
        }
        if let Some(contact) =
            rolling_transition(before, after, court, width, height, gap, rally_end_frame)
        {
            if contacts
                .last()
                .is_none_or(|prior| contact.frame.saturating_sub(prior.frame) > 12)
            {
                contacts.push(contact);
            }
        }
    }
    contacts
}

pub fn terminal_ground_contacts(
    points: &[BallObservation],
    court: &CourtTransform,
    width: u32,
    height: u32,
    rally_end_frame: u64,
) -> Vec<GroundContact> {
    if width == 0
        || height == 0
        || points.windows(2).any(|p| p[0].frame >= p[1].frame)
        || points.iter().any(|p| {
            p.frame < 0 || p.frame > i32::MAX as i64 || !p.x.is_finite() || !p.y.is_finite()
        })
    {
        return Vec::new();
    }
    let mut contacts: Vec<GroundContact> = Vec::new();
    for split in 5..points.len() {
        if let Some(contact) =
            short_terminal_rebound(points, split - 1, court, width, height, rally_end_frame)
        {
            if contacts
                .last()
                .is_none_or(|prior| contact.frame.saturating_sub(prior.frame) > 12)
            {
                contacts.push(contact);
            }
            continue;
        }
        let before = &points[split - 5..split];
        if points[split].frame - before[4].frame > 6
            || before.windows(2).any(|p| p[1].frame - p[0].frame > 2)
        {
            continue;
        }
        if let Some(contact) = terminal_rolling_transition(
            before,
            &points[split..],
            court,
            width,
            height,
            rally_end_frame,
        ) {
            if contacts
                .last()
                .is_none_or(|prior| contact.frame.saturating_sub(prior.frame) > 12)
            {
                contacts.push(contact);
            }
        }
    }
    contacts
}

fn short_terminal_rebound(
    points: &[BallObservation],
    at: usize,
    court: &CourtTransform,
    width: u32,
    height: u32,
    rally_end_frame: u64,
) -> Option<GroundContact> {
    let [w, h] = [f64::from(width), f64::from(height)];
    let peak = points[at];
    if rally_end_frame.saturating_sub(peak.frame as u64) > 45
        || points[at - 1].y > peak.y
        || points[at + 1].y >= peak.y
    {
        return None;
    }
    let lo = points.partition_point(|p| p.frame < peak.frame - 15);
    let before = &points[lo..=at];
    let start = before.first()?;
    let descent = (peak.y - start.y) / h;
    if before.len() < 10
        || peak.frame - start.frame < 12
        || descent < 0.06
        || before
            .windows(2)
            .any(|p| p[1].frame - p[0].frame > 2 || p[1].y + h * 0.004 < p[0].y)
    {
        return None;
    }
    let tail = &points[at + 1..];
    let limit = tail.partition_point(|p| p.frame <= peak.frame + 7);
    let n = tail[..limit]
        .windows(2)
        .position(|p| p[1].frame - p[0].frame > 2)
        .map_or(limit, |i| i + 1);
    let after = &tail[..n];
    let end = after.last()?;
    let rise = (peak.y - end.y) / h;
    if !(3..=7).contains(&(end.frame - peak.frame))
        || after.len() < 3
        || after[0].frame - peak.frame > 2
        || !(0.006..=0.024).contains(&rise)
        || rise > descent * 0.3
        || tail
            .get(n)
            .map_or(rally_end_frame.saturating_sub(end.frame as u64) < 15, |p| {
                p.frame - end.frame < 16
            })
        || after
            .iter()
            .any(|p| (p.x - peak.x).abs() > w * 0.018 || p.y > peak.y + h * 0.002)
        || after.windows(2).any(|p| p[1].y > p[0].y + h * 0.002)
    {
        return None;
    }
    let incoming_speed =
        ((peak.x - start.x) / w).hypot(descent) / (peak.frame - start.frame) as f64;
    let outgoing_speed = ((end.x - peak.x) / w).hypot(rise) / (end.frame - peak.frame) as f64;
    if outgoing_speed > incoming_speed * 0.8 {
        return None;
    }
    let position = [peak.x / w, peak.y / h];
    let zones = court.zones()?;
    let top = zones
        .near_half
        .iter()
        .chain(&zones.far_half)
        .map(|p| p[1])
        .reduce(f64::min)?;
    if position[1] < top + 0.10 {
        return None;
    }
    let [x, y] = court.to_court(position)?;
    let distance = (x.abs() - crate::court_geometry::COURT_LENGTH_M / 2.0)
        .max(y.abs() - crate::court_geometry::COURT_WIDTH_M / 2.0);
    if distance > 0.8 {
        return None;
    }
    let landing = if distance.abs() <= 0.25 {
        "line_uncertain"
    } else {
        court.tentative_landing(position)?
    };
    Some(GroundContact {
        frame: peak.frame as u64,
        position,
        landing: landing.to_owned(),
    })
}

fn terminal_rolling_transition(
    before: &[BallObservation],
    following: &[BallObservation],
    court: &CourtTransform,
    width: u32,
    height: u32,
    rally_end_frame: u64,
) -> Option<GroundContact> {
    let [w, h] = [f64::from(width), f64::from(height)];
    let first = before[0];
    let impact = *before.last()?;
    let span = (impact.frame - first.frame) as f64;
    let descent = (impact.y - first.y) / h;
    let incoming_speed = ((impact.x - first.x) / w).hypot(descent) / span;
    if descent < 0.075
        || descent / span < 0.012
        || before.windows(2).any(|p| p[1].y + h * 0.003 < p[0].y)
    {
        return None;
    }
    let mut previous = impact;
    let mut count = 0;
    for point in following {
        if point.frame - previous.frame > 6 || point.frame - impact.frame > 14 {
            break;
        }
        let speed = ((point.x - previous.x) / w).hypot((point.y - previous.y) / h)
            / (point.frame - previous.frame) as f64;
        if point.y > impact.y + h * 0.006
            || point.y < impact.y - h * 0.025
            || speed > incoming_speed * 0.5
        {
            return None;
        }
        previous = *point;
        count += 1;
    }
    if count < 2 || previous.frame - impact.frame < 3 {
        return None;
    }
    let mean_speed = ((previous.x - impact.x) / w).hypot((previous.y - impact.y) / h)
        / (previous.frame - impact.frame) as f64;
    let sustained_settle = count >= 8
        && previous.frame - impact.frame >= 12
        && mean_speed <= incoming_speed * 0.2
        && following[..count]
            .iter()
            .all(|p| (p.y - impact.y).abs() <= h * 0.012);
    let tracking_loss = following.get(count).map_or_else(
        || rally_end_frame.saturating_sub(previous.frame as u64) >= 15,
        |next| next.frame - previous.frame >= 16,
    );
    if mean_speed > incoming_speed * 0.35 || !(sustained_settle || tracking_loss) {
        return None;
    }
    let position = [impact.x / w, impact.y / h];
    let zones = court.zones()?;
    let top = zones
        .near_half
        .iter()
        .chain(&zones.far_half)
        .map(|p| p[1])
        .reduce(f64::min)?;
    if position[1] < top + 0.12 {
        return None;
    }
    let [x, y] = court.to_court(position)?;
    let distance = (x.abs() - crate::court_geometry::COURT_LENGTH_M / 2.0)
        .max(y.abs() - crate::court_geometry::COURT_WIDTH_M / 2.0);
    if distance > 0.8 {
        return None;
    }
    let landing = if distance.abs() <= 0.25 {
        "line_uncertain"
    } else {
        court.tentative_landing(position)?
    };
    Some(GroundContact {
        frame: impact.frame as u64,
        position,
        landing: landing.to_owned(),
    })
}

fn rolling_transition(
    before: &[BallObservation],
    after: &[BallObservation],
    court: &CourtTransform,
    width: u32,
    height: u32,
    gap: i64,
    rally_end_frame: u64,
) -> Option<GroundContact> {
    let [w, h] = [f64::from(width), f64::from(height)];
    let first = before[0];
    let left = *before.last()?;
    let right = after[0];
    let last = *after.last()?;
    let slope = |a: BallObservation, b: BallObservation| {
        let span = (b.frame - a.frame) as f64;
        [(b.x - a.x) / span / w, (b.y - a.y) / span / h]
    };
    let incoming = slope(first, left);
    let outgoing = slope(right, last);
    let incoming_speed = incoming[0].hypot(incoming[1]);
    let outgoing_speed = outgoing[0].hypot(outgoing[1]);
    if incoming[1] < 0.006
        || outgoing_speed > incoming_speed * 0.45
        || before.windows(2).any(|p| p[1].y + h * 0.003 < p[0].y)
    {
        return None;
    }
    let dt = (right.frame - left.frame) as f64;
    let difference = incoming[1] - outgoing[1];
    if difference <= 0.0 {
        return None;
    }
    let offset = ((right.y - left.y) / h - outgoing[1] * dt) / difference;
    if !(-0.5..=dt + 0.5).contains(&offset) {
        return None;
    }
    let offset = offset.clamp(0.0, dt);
    let left_x = left.x / w + incoming[0] * offset;
    let right_x = right.x / w - outgoing[0] * (dt - offset);
    if (left_x - right_x).abs() > 0.012 {
        return None;
    }
    let position = [(left_x + right_x) * 0.5, left.y / h + incoming[1] * offset];
    if position.iter().any(|p| !(-0.5..=1.5).contains(p)) {
        return None;
    }
    let origin = court.to_court(position)?;
    let boundary_distance = |point: [f64; 2]| {
        (point[0].abs() - crate::court_geometry::COURT_LENGTH_M / 2.0)
            .max(point[1].abs() - crate::court_geometry::COURT_WIDTH_M / 2.0)
    };
    let distance_m = boundary_distance(origin);
    if !(-0.8..=0.8).contains(&distance_m) && (gap > 1 || distance_m > 0.8) {
        return None;
    }
    let projected = after
        .iter()
        .map(|p| court.to_court([p.x / w, p.y / h]))
        .collect::<Option<Vec<_>>>()?;
    let end = *projected.last()?;
    let frame = (left.frame + offset.round() as i64) as u64;
    if rally_end_frame.saturating_sub(frame) > 45
        && (distance_m.abs() > 0.8 || boundary_distance(end) < -0.8)
    {
        return None;
    }
    let direction = [end[0] - origin[0], end[1] - origin[1]];
    let length = direction[0].hypot(direction[1]);
    if length < 0.3 || length / (last.frame - left.frame) as f64 > 0.4 {
        return None;
    }
    let mut previous_progress = 0.0_f64;
    for point in projected {
        if point[0].abs() > crate::court_geometry::COURT_LENGTH_M / 2.0 + 6.0
            || point[1].abs() > crate::court_geometry::COURT_WIDTH_M / 2.0 + 1.5
        {
            return None;
        }
        let delta = [point[0] - origin[0], point[1] - origin[1]];
        let progress = (delta[0] * direction[0] + delta[1] * direction[1]) / length;
        let lateral = (delta[0] * direction[1] - delta[1] * direction[0]).abs() / length;
        if lateral > 0.5 || progress + 0.25 < previous_progress {
            return None;
        }
        previous_progress = previous_progress.max(progress);
    }
    let landing = if distance_m.abs() <= 0.25 {
        "line_uncertain"
    } else {
        court.tentative_landing(position)?
    };
    Some(GroundContact {
        frame,
        position,
        landing: landing.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::MotionEvidence;

    fn court() -> CourtTransform {
        CourtTransform::new([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]).unwrap()
    }

    fn samples(x: f64, upward: bool) -> (Vec<BallObservation>, TrajectoryBreak) {
        let values = [46.0, 55.0, 65.0, 75.0, 85.0, 91.0, 96.0, 100.0, 103.0];
        let points = (0..9)
            .map(|i| BallObservation {
                frame: i,
                x,
                y: if i >= 5 && upward {
                    103.0 - (i - 4) as f64 * 6.0
                } else {
                    values[i as usize]
                },
            })
            .collect();
        let impact = TrajectoryBreak {
            frame: 4,
            position: [x, 85.0],
            motion: MotionEvidence::default(),
            fit_error: 0.5,
            before_frames: vec![0, 1, 2, 3],
            after_frames: vec![5, 6, 7, 8],
            contact_observed: true,
        };
        (points, impact)
    }

    #[test]
    fn calibrated_terminal_loss_requires_measured_descent_to_floor() {
        let project = |[x, y, z]: [f64; 3]| {
            let depth = 1.0 + 0.05 * x;
            [
                (0.5 + 0.03 * x + 0.04 * y) / depth,
                (0.5 - 0.01 * x - 0.09 * z) / depth,
            ]
        };
        let court = CourtTransform::new(
            [
                [-9., -4.5, 0.],
                [9., -4.5, 0.],
                [9., 4.5, 0.],
                [-9., 4.5, 0.],
            ]
            .map(project),
        )
        .unwrap()
        .with_net(Some([[0., -4.5, 2.43], [0., 4.5, 2.43]].map(project)))
        .unwrap();
        for (height, expected) in [(0.15, true), (1.3, false)] {
            let mut points = (0..=30)
                .map(|frame| {
                    let t = (frame - 30) as f64 / 30.0;
                    let [x, y] =
                        project([2.0 + 2.0 * t, 1.0 + t, height - 7.0 * t - 4.905 * t * t]);
                    BallObservation {
                        frame,
                        x: x * 1920.,
                        y: y * 1080.,
                    }
                })
                .collect::<Vec<_>>();
            points.push(BallObservation {
                frame: 51,
                x: 1200.,
                y: 450.,
            });
            let contacts = terminal_loss_ground_contacts(&points, &court, 1920, 1080, 51);
            assert_eq!(!contacts.is_empty(), expected, "{contacts:?}");
            if expected {
                assert!((28..=30).contains(&contacts[0].frame));
            }
        }
    }
    #[test]
    fn calibrated_airborne_reception_is_not_ground_contact() {
        let project = |[x, y, z]: [f64; 3]| {
            [
                (0.5 + 0.03 * x + 0.04 * y) / (1.0 + 0.05 * x),
                (0.5 - 0.01 * x - 0.09 * z) / (1.0 + 0.05 * x),
            ]
        };
        let court = CourtTransform::new(
            [
                [-9., -4.5, 0.],
                [9., -4.5, 0.],
                [9., 4.5, 0.],
                [-9., 4.5, 0.],
            ]
            .map(project),
        )
        .unwrap()
        .with_net(Some([[0., -4.5, 2.43], [0., 4.5, 2.43]].map(project)))
        .unwrap();
        let points = (0..=60)
            .map(|frame| {
                let t = (frame - 30) as f64 / 30.;
                let vz = if frame <= 30 { -8. } else { 4. };
                let [x, y] = project([-6. + t * 1.5, -3. + t * 0.5, 1.3 + vz * t - 4.905 * t * t]);
                BallObservation {
                    frame,
                    x: x * 1920.,
                    y: y * 1080.,
                }
            })
            .collect::<Vec<_>>();
        let peak = points[30];
        let impact = TrajectoryBreak {
            frame: 30,
            position: [peak.x, peak.y],
            motion: MotionEvidence::default(),
            fit_error: 0.5,
            before_frames: vec![25, 26, 27, 28],
            after_frames: vec![31, 32, 33, 34],
            contact_observed: true,
        };
        assert!(classify_ground_contact(&points, &impact, &court, 1920, 1080, 50).is_none());
    }

    #[test]
    fn rebound_on_each_calibrated_half_is_separate_from_player_touch() {
        let (mut points, mut impact) = samples(300.0, true);
        impact.position[1] = 103.0;
        assert_eq!(
            classify_ground_contact(&points, &impact, &court(), 1000, 200, 20)
                .unwrap()
                .landing,
            "near_half"
        );
        for p in &mut points {
            p.x = 700.0;
        }
        impact.position[0] = 700.0;
        assert_eq!(
            classify_ground_contact(&points, &impact, &court(), 1000, 200, 20)
                .unwrap()
                .landing,
            "far_half"
        );
    }

    #[test]
    fn early_boundary_rebound_and_out_settle_are_floor_candidates() {
        let (mut points, mut impact) = samples(100.0, true);
        impact.position[1] = 103.0;
        assert_eq!(
            classify_ground_contact(&points, &impact, &court(), 1000, 200, 250)
                .unwrap()
                .landing,
            "line_uncertain"
        );
        for p in &mut points {
            p.x = 50.0;
            if p.frame >= 5 {
                p.y = 103.0;
            }
        }
        impact.position[0] = 50.0;
        assert_eq!(
            classify_ground_contact(&points, &impact, &court(), 1000, 200, 250)
                .unwrap()
                .landing,
            "out"
        );
    }
    #[test]
    fn mid_rally_and_above_surface_impacts_are_not_sand() {
        let (points, mut impact) = samples(300.0, true);
        impact.position[1] = 103.0;
        assert!(classify_ground_contact(&points, &impact, &court(), 1000, 200, 99).is_none());
        impact.position[1] = 35.0;
        assert!(classify_ground_contact(&points, &impact, &court(), 1000, 200, 20).is_none());
    }

    #[test]
    fn energetic_reversal_is_not_a_sand_rebound_even_near_a_boundary() {
        let (mut points, mut impact) = samples(100.0, true);
        impact.position[1] = 103.0;
        for p in &mut points {
            if p.frame >= 5 {
                p.y = 103.0 - (p.frame - 4) as f64 * 14.0;
            }
        }
        assert!(classify_ground_contact(&points, &impact, &court(), 1000, 200, 250).is_none());
        for p in &mut points {
            if p.frame >= 5 {
                p.y = 103.0 - (p.frame - 4) as f64 * 6.0;
            }
        }
        assert!(classify_ground_contact(&points, &impact, &court(), 1000, 200, 250).is_some());
    }

    #[test]
    fn sand_can_settle_without_visible_upward_rebound() {
        let (mut points, mut impact) = samples(300.0, true);
        impact.position[1] = 103.0;
        for p in &mut points {
            if p.frame >= 5 {
                p.y = 103.0;
            }
        }
        assert!(classify_ground_contact(&points, &impact, &court(), 1000, 200, 20).is_some());
    }

    #[test]
    fn visible_horizontal_roll_is_not_required_to_rebound() {
        let mut points = (0..5)
            .map(|frame| BallObservation {
                frame,
                x: 300.0,
                y: 400.0 + frame as f64 * 12.0,
            })
            .collect::<Vec<_>>();
        points.extend((5..19).map(|frame| BallObservation {
            frame,
            x: 300.0 + (frame - 4) as f64 * 2.0,
            y: 448.0,
        }));
        let contacts = rolling_ground_contacts(&points, &court(), 1000, 1000, 30);
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].frame, 4);
        assert!(rolling_ground_contacts(&points, &court(), 1000, 1000, 200).is_empty());
        for p in &mut points[5..] {
            p.y -= (p.frame - 4) as f64 * 16.0;
        }
        assert!(rolling_ground_contacts(&points, &court(), 1000, 1000, 30).is_empty());
    }

    #[test]
    fn sand_settle_followed_by_visible_pickup_ends_at_landing() {
        let mut points = (0..5)
            .map(|frame| BallObservation {
                frame,
                x: 510.0,
                y: 520.0 + frame as f64 * 30.0,
            })
            .collect::<Vec<_>>();
        points.extend((5..=24).map(|frame| BallObservation {
            frame,
            x: 510.0 + (frame - 4) as f64 * 0.3,
            y: 640.0,
        }));
        points.extend((25..=40).map(|frame| BallObservation {
            frame,
            x: 516.0,
            y: 640.0 - (frame - 24) as f64 * 20.0,
        }));
        let contacts = terminal_ground_contacts(&points, &court(), 1000, 1000, 100);
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].frame, 4);
        for p in &mut points[5..] {
            p.y = 640.0 + (p.frame - 4) as f64 * 15.0;
        }
        assert!(terminal_ground_contacts(&points, &court(), 1000, 1000, 100).is_empty());
    }

    #[test]
    fn short_terminal_rebound_between_measured_anchors_is_floor() {
        let court =
            CourtTransform::new([[0.06, 1.02], [0.34, 0.29], [0.60, 0.29], [0.89, 1.03]]).unwrap();
        let coords = [
            (0, 0.416, 0.350),
            (2, 0.421, 0.361),
            (3, 0.422, 0.365),
            (4, 0.427, 0.369),
            (5, 0.428, 0.375),
            (6, 0.430, 0.379),
            (7, 0.432, 0.382),
            (8, 0.433, 0.389),
            (9, 0.435, 0.399),
            (10, 0.437, 0.403),
            (11, 0.439, 0.420),
            (12, 0.441, 0.424),
            (13, 0.443, 0.426),
            (14, 0.445, 0.428),
            (15, 0.447, 0.428),
            (16, 0.449, 0.426),
            (17, 0.451, 0.424),
            (18, 0.453, 0.420),
            (19, 0.457, 0.417),
            (50, 0.5, 0.3),
        ];
        let points = coords
            .iter()
            .map(|&(frame, x, y)| BallObservation {
                frame,
                x: x * 3840.,
                y: y * 2160.,
            })
            .collect::<Vec<_>>();
        let hits = terminal_ground_contacts(&points, &court, 3840, 2160, 35);
        assert!(hits.iter().any(|c| c.frame.abs_diff(15) <= 1), "{hits:?}");
        let mut continuing = points.clone();
        continuing.last_mut().unwrap().frame = 20;
        assert!(terminal_ground_contacts(&continuing, &court, 3840, 2160, 35).is_empty());
    }
    #[test]
    fn short_damped_tail_before_tracking_loss_ends_merged_rally() {
        let c = court();
        for tail in [
            vec![(5, 508.0, 644.0), (7, 505.0, 644.0)],
            vec![
                (8, 491.0, 627.0),
                (9, 490.0, 625.0),
                (10, 489.0, 625.0),
                (11, 487.0, 626.0),
                (12, 484.0, 625.0),
                (13, 483.0, 627.0),
                (14, 478.0, 631.0),
                (15, 476.0, 632.0),
            ],
        ] {
            let mut points = (0..5)
                .map(|frame| BallObservation {
                    frame,
                    x: 510.0,
                    y: 520.0 + frame as f64 * 30.0,
                })
                .collect::<Vec<_>>();
            points.extend(
                tail.into_iter()
                    .map(|(frame, x, y)| BallObservation { frame, x, y }),
            );
            points.push(BallObservation {
                frame: 70,
                x: 400.0,
                y: 300.0,
            });
            let contacts = terminal_ground_contacts(&points, &c, 1000, 1000, 100);
            assert_eq!(contacts.len(), 1);
            assert_eq!(contacts[0].frame, 4);
            for p in &mut points[5..] {
                p.y = 640.0 - (p.frame - 4) as f64 * 22.0;
            }
            assert!(terminal_ground_contacts(&points, &c, 1000, 1000, 100).is_empty());
        }
    }

    #[test]
    fn non_rebound_motion_and_uncalibrated_court_are_not_ground_contacts() {
        let (points, impact) = samples(300.0, false);
        assert!(classify_ground_contact(&points, &impact, &court(), 1000, 200, 20).is_none());
        let (points, impact) = samples(300.0, true);
        assert!(classify_ground_contact(&points, &impact, &court(), 0, 200, 20).is_none());
    }

    fn distant_sand_samples() -> (Vec<BallObservation>, TrajectoryBreak, CourtTransform) {
        let court =
            CourtTransform::new([[0.1, 0.9], [0.35, 0.28], [0.6, 0.28], [0.9, 0.9]]).unwrap();
        let points = (2..=64)
            .filter(|&frame| frame != 21)
            .map(|frame| {
                let y = match frame {
                    2..=14 => 0.14 + (frame - 2) as f64 * 0.012,
                    15..=20 => 0.284 + (frame - 14) as f64 * 0.0015,
                    22..=35 => 0.278 - (frame - 22) as f64 * 0.0045,
                    36..=56 => 0.2195 + (frame - 35) as f64 * 0.0034,
                    _ => 0.279,
                };
                BallObservation {
                    frame,
                    x: (0.69 - (frame - 20).max(0) as f64 * 0.0005) * 1000.0,
                    y: y * 1000.0,
                }
            })
            .collect();
        let impact = TrajectoryBreak {
            frame: 20,
            position: [690.0, 293.0],
            motion: MotionEvidence::default(),
            fit_error: 0.5,
            before_frames: vec![16, 17, 18, 19],
            after_frames: vec![22, 23, 24, 25],
            contact_observed: true,
        };
        (points, impact, court)
    }

    #[test]
    fn distant_out_sand_short_rebound_and_return_ends_rally() {
        let (points, impact, court) = distant_sand_samples();
        let contact = classify_ground_contact(&points, &impact, &court, 1000, 1000, 200).unwrap();
        assert_eq!(contact.frame, 20);
        assert_eq!(contact.landing, "out");
    }

    #[test]
    fn distant_out_damped_rebound_returns_toward_sand_without_settling() {
        let (mut points, impact, court) = distant_sand_samples();
        points.retain(|p| p.frame <= 54);
        for p in &mut points {
            if p.frame > 35 {
                p.y = 219.5 + (p.frame - 35) as f64 * 2.8;
            }
        }
        points.push(BallObservation {
            frame: 64,
            x: 800.0,
            y: 400.0,
        });
        let contact = classify_ground_contact(&points, &impact, &court, 1000, 1000, 200)
            .expect("damped out rebound with incomplete return");
        assert_eq!(contact.frame, 20);
        assert_eq!(contact.landing, "out");
        for p in &mut points {
            if p.frame > 35 {
                p.y = 219.5 + (p.frame - 35) as f64 * 0.8;
            }
        }
        assert!(classify_ground_contact(&points, &impact, &court, 1000, 1000, 200).is_none());
    }

    #[test]
    fn out_sand_damped_rebound_below_far_edge_ends_rally() {
        let (mut points, mut impact, court) = distant_sand_samples();
        for p in &mut points {
            p.x += 45.0;
            p.y += 80.0;
        }
        impact.position[0] += 45.0;
        impact.position[1] += 80.0;
        let contact = classify_ground_contact(&points, &impact, &court, 1000, 1000, 200)
            .expect("damped out rebound below the far edge");
        assert_eq!(contact.landing, "out");
    }

    #[test]
    fn distant_out_without_damped_return_is_not_floor() {
        let (mut points, impact, court) = distant_sand_samples();
        for p in &mut points {
            if p.frame > 35 {
                p.y = 219.5 - (p.frame - 35) as f64 * 3.0;
            }
        }
        assert!(classify_ground_contact(&points, &impact, &court, 1000, 1000, 200).is_none());
    }
    #[test]
    fn distant_short_damped_out_rebound_before_loss() {
        let court =
            CourtTransform::new([[0.1, 0.9], [0.35, 0.28], [0.6, 0.28], [0.9, 0.9]]).unwrap();
        let points = (2..=34)
            .map(|frame| {
                let t = (frame - 20) as f64;
                let y = if t <= 0.0 {
                    0.255 + 0.005 * t
                } else if t <= 9.0 {
                    0.255 - 0.0016 * t
                } else {
                    0.2406 + 0.0015 * (t - 9.0)
                };
                BallObservation {
                    frame,
                    x: (0.53 - t.max(0.0) * 0.0007) * 3840.0,
                    y: y * 2160.0,
                }
            })
            .collect::<Vec<_>>();
        let impact = TrajectoryBreak {
            frame: 20,
            position: [0.53 * 3840.0, 0.255 * 2160.0],
            motion: MotionEvidence::default(),
            fit_error: 0.5,
            before_frames: vec![16, 17, 18, 19],
            after_frames: vec![21, 22, 23, 24],
            contact_observed: true,
        };
        let hit = classify_ground_contact(&points, &impact, &court, 3840, 2160, 200)
            .expect("short damped rebound behind the far baseline");
        assert_eq!(hit.frame, 20);
        assert_eq!(hit.landing, "out");
        let mut continuing = points.clone();
        continuing.extend((35..=55).map(|frame| BallObservation {
            frame,
            x: 0.52 * 3840.0,
            y: (0.25 - (frame - 35) as f64 * 0.008) * 2160.0,
        }));
        assert!(classify_ground_contact(&continuing, &impact, &court, 3840, 2160, 200).is_none());
    }
}
