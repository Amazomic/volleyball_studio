use super::{BallObservation, TrajectoryBreak};
use crate::court_geometry::CourtTransform;

pub const GROUND_DETECTOR_VERSION: &str = "court_ball_rebound_v3";

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct GroundContact {
    pub frame: u64,
    pub position: [f64; 2],
    pub landing: &'static str,
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
    let rebound_or_settle = outgoing <= -1.0 || (incoming >= 3.0 && outgoing.abs() <= 1.5);
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
        landing,
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
    fn non_rebound_motion_and_uncalibrated_court_are_not_ground_contacts() {
        let (points, impact) = samples(300.0, false);
        assert!(classify_ground_contact(&points, &impact, &court(), 1000, 200, 20).is_none());
        let (points, impact) = samples(300.0, true);
        assert!(classify_ground_contact(&points, &impact, &court(), 0, 200, 20).is_none());
    }
}
