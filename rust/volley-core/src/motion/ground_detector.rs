use super::{BallObservation, TrajectoryBreak};
use crate::court_geometry::CourtTransform;

pub const GROUND_DETECTOR_VERSION: &str = "court_ball_roll_v6";

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
    if width == 0
        || height == 0
        || impact.frame < 0
        || impact.before_frames.len() < 4
        || impact.after_frames.len() < 4
    {
        return None;
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
    if !(top - 0.005..=top + 0.06).contains(&position[1]) {
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
    if before.len() < 8
        || after.len() < 12
        || local.windows(2).any(|p| p[1].frame - p[0].frame > 6)
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
    returned.then_some(GroundContact {
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
    fn distant_out_without_damped_return_is_not_floor() {
        let (mut points, impact, court) = distant_sand_samples();
        for p in &mut points {
            if p.frame > 35 {
                p.y = 219.5 - (p.frame - 35) as f64 * 3.0;
            }
        }
        assert!(classify_ground_contact(&points, &impact, &court, 1000, 1000, 200).is_none());
    }
}
