use crate::{
    ball_interpolation::StabilizedBallMotion,
    court_geometry::{CourtSide, CourtTransform},
};

pub const SERVE_FLIGHT_VERSION: &str = "calibrated_opening_serve_flight_v2";
const NET_HEIGHT_M: f64 = 2.43;
const GRAVITY_HALF: f64 = 4.905;

pub struct ServeFlightDetector {
    projection: [[f64; 4]; 3],
}

impl ServeFlightDetector {
    pub fn new(court: &CourtTransform) -> Option<Self> {
        let projection = court.flight_projection(NET_HEIGHT_M)?;
        Some(Self { projection })
    }

    pub fn supports_opening(
        &self,
        trajectory: &StabilizedBallMotion,
        interval: (u64, u64),
        earliest: u64,
    ) -> bool {
        self.opening_origin(trajectory, interval, earliest)
            .is_some()
    }

    pub fn opening_origin(
        &self,
        trajectory: &StabilizedBallMotion,
        interval: (u64, u64),
        earliest: u64,
    ) -> Option<(u64, CourtSide)> {
        self.opening_flight(trajectory, interval, earliest)
            .map(|(launch, side, _)| (launch, side))
    }

    pub fn opening_flight(
        &self,
        trajectory: &StabilizedBallMotion,
        interval: (u64, u64),
        earliest: u64,
    ) -> Option<(u64, CourtSide, u64)> {
        let (start, end) = interval;
        if start > end
            || end >= trajectory.points.len() as u64
            || trajectory.interpolated.len() != trajectory.points.len()
        {
            return None;
        }
        let first = start.saturating_sub(150).max(earliest);
        let last = start.saturating_add(45).min(end.saturating_sub(24));
        if first > last {
            return None;
        }
        let mut regular = None;
        'regular: for launch in (first..=last).step_by(3) {
            for span in [24, 30, 36, 45, 54, 63] {
                if launch + span <= end {
                    if let Some((side, crossing)) = self.flight_origin(
                        &trajectory.points[launch as usize..=(launch + span) as usize],
                        &trajectory.interpolated[launch as usize..=(launch + span) as usize],
                        None,
                    ) {
                        regular = Some((launch, side, launch + crossing));
                        break 'regular;
                    }
                }
            }
        }
        let high_last = regular.map_or(last, |(launch, _, _)| launch.saturating_sub(1));
        if regular.is_none_or(|(launch, _, _)| launch > first) {
            if let Some(high) = self.supports_high_flight(
                trajectory,
                first as usize,
                high_last as usize,
                end as usize,
            ) {
                return Some(high);
            }
        }
        regular
    }

    fn supports_high_flight(
        &self,
        trajectory: &StabilizedBallMotion,
        first: usize,
        last: usize,
        end: usize,
    ) -> Option<(u64, CourtSide, u64)> {
        let points = &trajectory.points;
        for exit in first.saturating_add(8)..=last.saturating_add(28).min(end) {
            let (Some(upper), Some(before)) = (points[exit], points[exit - 6]) else {
                continue;
            };
            if upper[1] > 0.035 || before[1] - upper[1] < 0.045 {
                continue;
            }
            for entry in exit + 6..=exit.saturating_add(54).min(end.saturating_sub(3)) {
                let (Some(returned), Some(after)) = (points[entry], points[entry + 3]) else {
                    continue;
                };
                if after[1] - returned[1] < 0.025 {
                    continue;
                }
                let hidden = &points[exit + 1..entry];
                if hidden.iter().filter(|point| point.is_none()).count() * 2 < hidden.len()
                    || !hidden
                        .windows(5)
                        .any(|window| window.iter().all(Option::is_none))
                {
                    continue;
                }
                for launch in
                    (exit.saturating_sub(30).max(first)..=exit.saturating_sub(8).min(last)).rev()
                {
                    for tail in [3, 5, 7, 10, 12] {
                        let finish = entry + tail;
                        if finish > end || finish - launch > 84 {
                            continue;
                        }
                        if let Some((side, crossing)) = self.flight_origin(
                            &points[launch..=finish],
                            &trajectory.interpolated[launch..=finish],
                            Some((exit + 1 - launch, entry - launch)),
                        ) {
                            return Some((launch as u64, side, launch as u64 + crossing));
                        }
                    }
                }
            }
        }
        None
    }

    fn flight_origin(
        &self,
        points: &[Option<[f64; 2]>],
        interpolated: &[bool],
        gap: Option<(usize, usize)>,
    ) -> Option<(CourtSide, u64)> {
        let real = |range: std::ops::Range<usize>| {
            range
                .filter(|&i| points[i].is_some() && !interpolated[i])
                .count()
        };
        let continuous = |slice: &[Option<[f64; 2]>]| {
            !slice
                .windows(5)
                .any(|window| window.iter().all(Option::is_none))
        };
        if points.first().is_none_or(Option::is_none) || points.last().is_none_or(Option::is_none) {
            return None;
        }
        if let Some((left, right)) = gap {
            if real(0..left) < 8
                || real(right..points.len()) < 4
                || !continuous(&points[..left])
                || !continuous(&points[right..])
            {
                return None;
            }
        } else if !continuous(points) || real(0..points.len()) * 10 < (points.len() - 1) * 7 {
            return None;
        }
        let hidden = |frame| gap.is_some_and(|(left, right)| (left..right).contains(&frame));
        let [x, y, z, vx, vy, vz] = fit_projected_ballistic_flight(
            &self.projection,
            points.iter().enumerate().filter_map(|(frame, point)| {
                if hidden(frame) {
                    return None;
                }
                Some((frame as f64 / 30.0, (*point)?))
            }),
        )?;
        let duration = (points.len() - 1) as f64 / 30.0;
        if !(6.5..=17.0).contains(&x.abs())
            || !(0.3..=4.0).contains(&z)
            || y.abs() > 8.0
            || vx.abs() < 1.0
        {
            return None;
        }
        let crossing = -x / vx;
        let height = |t: f64| z + vz * t - GRAVITY_HALF * t * t;
        let final_x = x + vx * duration;
        if crossing <= 0.25
            || crossing >= duration
            || height(crossing) < NET_HEIGHT_M - 0.43
            || (y + vy * crossing).abs() > 5.5
            || final_x * x >= 0.0
            || final_x.abs() < 1.0
            || height(duration) < 0.0
        {
            return None;
        }
        let mut error = 0.0;
        let mut count = 0;
        let mut hidden_high = 0;
        let mut hidden_top = false;
        for (frame, point) in points.iter().enumerate() {
            let t = frame as f64 / 30.0;
            let position = [x + vx * t, y + vy * t, height(t), 1.0];
            let projected = self
                .projection
                .map(|row| row.iter().zip(position).map(|(a, b)| a * b).sum::<f64>());
            if projected[2] <= 0.0 {
                return None;
            }
            if hidden(frame) {
                let py = projected[1] / projected[2];
                hidden_high += usize::from(py < 0.08);
                hidden_top |= py < 0.025;
                continue;
            }
            let Some(point) = point else { continue };
            let residual = (projected[0] / projected[2] - point[0])
                .hypot(projected[1] / projected[2] - point[1]);
            if !residual.is_finite() || residual > 0.07 {
                return None;
            }
            error += residual * residual;
            count += 1;
        }
        (count > 0
            && (error / f64::from(count)).sqrt() <= 0.018
            && gap.is_none_or(|(left, right)| hidden_top && hidden_high * 3 >= right - left))
        .then_some((
            if x < 0.0 {
                CourtSide::Near
            } else {
                CourtSide::Far
            },
            (crossing * 30.0).ceil() as u64,
        ))
    }
}

pub(crate) fn fit_projected_ballistic_flight(
    projection: &[[f64; 4]; 3],
    observations: impl IntoIterator<Item = (f64, [f64; 2])>,
) -> Option<[f64; 6]> {
    let mut normal = [[0.0; 7]; 6];
    for (t, point) in observations {
        if !t.is_finite() || !point.iter().all(|coordinate| coordinate.is_finite()) {
            return None;
        }
        for (axis, coordinate) in point.iter().copied().enumerate() {
            let q = std::array::from_fn::<_, 4, _>(|j| {
                projection[axis][j] - coordinate * projection[2][j]
            });
            let row = [q[0], q[1], q[2], q[0] * t, q[1] * t, q[2] * t];
            let rhs = -q[3] + q[2] * GRAVITY_HALF * t * t;
            for (i, &a) in row.iter().enumerate() {
                for (j, &b) in row.iter().enumerate() {
                    normal[i][j] += a * b;
                }
                normal[i][6] += a * rhs;
            }
        }
    }
    solve(normal)
}

fn solve(mut rows: [[f64; 7]; 6]) -> Option<[f64; 6]> {
    for pivot in 0..6 {
        let best =
            (pivot..6).max_by(|&a, &b| rows[a][pivot].abs().total_cmp(&rows[b][pivot].abs()))?;
        if rows[best][pivot].abs() < 1e-10 {
            return None;
        }
        rows.swap(pivot, best);
        let divisor = rows[pivot][pivot];
        for value in &mut rows[pivot][pivot..] {
            *value /= divisor;
        }
        let normalized = rows[pivot];
        for (i, row) in rows.iter_mut().enumerate() {
            if i == pivot {
                continue;
            }
            let factor = row[pivot];
            for j in pivot..7 {
                row[j] -= factor * normalized[j];
            }
        }
    }
    let result = rows.map(|row| row[6]);
    result.iter().all(|v| v.is_finite()).then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project([x, y, z]: [f64; 3]) -> [f64; 2] {
        let depth = 1.0 + 0.05 * x;
        [
            (0.5 + 0.03 * x + 0.04 * y) / depth,
            (0.5 - 0.01 * x - 0.09 * z) / depth,
        ]
    }

    fn detector() -> ServeFlightDetector {
        let corners = [
            [-9.0, -4.5, 0.0],
            [9.0, -4.5, 0.0],
            [9.0, 4.5, 0.0],
            [-9.0, 4.5, 0.0],
        ]
        .map(project);
        let net = [[0.0, -4.5, NET_HEIGHT_M], [0.0, 4.5, NET_HEIGHT_M]].map(project);
        ServeFlightDetector::new(
            &CourtTransform::new(corners)
                .unwrap()
                .with_net(Some(net))
                .unwrap(),
        )
        .unwrap()
    }

    fn trajectory(position: [f64; 3], velocity: [f64; 3]) -> StabilizedBallMotion {
        let points: Vec<_> = (0..=39)
            .map(|frame| {
                let t = f64::from(frame) / 30.0;
                let [x, y, z] = std::array::from_fn(|i| position[i] + velocity[i] * t);
                Some(project([x, y, z - GRAVITY_HALF * t * t]))
            })
            .collect();
        StabilizedBallMotion {
            observed_frames: points.len(),
            confidence_frames: points.len(),
            interpolated: vec![false; points.len()],
            rejected_frames: Vec::new(),
            points,
        }
    }

    #[test]
    fn opening_flight_crosses_net_from_both_service_zones() {
        let detector = detector();
        for direction in [-1.0, 1.0] {
            let flight = trajectory([direction * 9.0, 0.0, 2.2], [-direction * 12.0, 1.0, 5.5]);
            assert!(detector.supports_opening(&flight, (0, 39), 0));
            let (launch, side) = detector.opening_origin(&flight, (0, 39), 0).unwrap();
            assert_eq!(launch, 0);
            let (_, _, crossing) = detector.opening_flight(&flight, (0, 39), 0).unwrap();
            assert!((22..=23).contains(&crossing));
            assert_eq!(
                side,
                if direction < 0.0 {
                    crate::court_geometry::CourtSide::Near
                } else {
                    crate::court_geometry::CourtSide::Far
                }
            );
        }
    }

    #[test]
    fn opening_after_stationary_remote_ball_keeps_observed_launch() {
        use crate::ball_interpolation::{
            stabilize_ball_motion, BallStabilizerSettings, ScoredBallSample,
        };

        let flight = trajectory([-9.0, 0.0, 2.2], [12.0, 1.0, 5.5]);
        let mut samples = (0..45)
            .map(|frame| ScoredBallSample {
                frame,
                point: [0.95, 0.80],
                confidence: 0.9,
            })
            .collect::<Vec<_>>();
        samples.extend(
            flight
                .points
                .iter()
                .enumerate()
                .map(|(index, point)| ScoredBallSample {
                    frame: 45 + index,
                    point: point.unwrap(),
                    confidence: 0.8,
                }),
        );
        let motion =
            stabilize_ball_motion(&samples, 85, BallStabilizerSettings::default()).unwrap();
        assert_eq!(motion.points[45], flight.points[0]);
        assert!(!motion.interpolated[45]);
        assert!(!motion.rejected_frames.contains(&45));
        assert!(detector().supports_opening(&motion, (45, 84), 0));
    }
    #[test]
    fn local_toss_and_flight_below_net_do_not_start_rally() {
        let detector = detector();
        let toss = trajectory([8.5, 0.0, 1.8], [-0.5, 0.5, 5.5]);
        assert!(!detector.supports_opening(&toss, (0, 39), 0));
        let low = trajectory([9.0, 0.0, 1.0], [-12.0, 0.0, 3.2]);
        assert!(!detector.supports_opening(&low, (0, 39), 0));
        let in_court = trajectory([3.0, 0.0, 2.2], [-8.0, 0.0, 5.5]);
        assert!(!detector.supports_opening(&in_court, (0, 39), 0));
    }

    #[test]
    fn high_serve_can_leave_frame_without_accepting_unrelated_ball() {
        let detector = detector();
        let points: Vec<_> = (0..=60)
            .map(|frame| {
                let t = f64::from(frame) / 30.0;
                let point = project([9.0 - 10.0 * t, 0.0, 2.2 + 9.0 * t - GRAVITY_HALF * t * t]);
                (point[1] >= 0.0).then_some(point)
            })
            .collect();
        let mut flight = StabilizedBallMotion {
            observed_frames: points.len(),
            confidence_frames: points.len(),
            interpolated: vec![false; points.len()],
            rejected_frames: Vec::new(),
            points,
        };
        assert!(detector.supports_opening(&flight, (0, 60), 0));
        let gap = flight.points.iter().position(Option::is_none).unwrap();
        flight.points[gap + 5..gap + 8].fill(Some([0.8, 0.5]));
        assert!(detector.supports_opening(&flight, (0, 60), 0));
        flight.points[40..].fill(None);
        assert!(!detector.supports_opening(&flight, (0, 60), 0));
    }

    #[test]
    fn opening_recovers_earlier_high_serve_instead_of_later_opposite_flight() {
        let early = (0..=60).map(|frame| {
            let t = f64::from(frame) / 30.0;
            let point = project([9.0 - 10.0 * t, 0.0, 2.2 + 9.0 * t - GRAVITY_HALF * t * t]);
            (point[1] >= 0.0).then_some(point)
        });
        let mut points = vec![None; 180];
        for (frame, point) in early.enumerate() {
            points[frame + 10] = point;
        }
        let later = trajectory([-9.0, 0.0, 2.2], [12.0, 1.0, 5.5]);
        points[130..170].copy_from_slice(&later.points);
        let motion = StabilizedBallMotion {
            observed_frames: points.iter().flatten().count(),
            confidence_frames: points.iter().flatten().count(),
            interpolated: vec![false; points.len()],
            rejected_frames: Vec::new(),
            points,
        };
        let detector = detector();
        let (launch, side, crossing) = detector.opening_flight(&motion, (130, 179), 0).unwrap();
        assert!((10..=20).contains(&launch), "launch={launch}");
        assert_eq!(side, CourtSide::Far);
        assert!(crossing < 100);
        let (launch, side, _) = detector.opening_flight(&motion, (130, 179), 100).unwrap();
        assert_eq!(launch, 130);
        assert_eq!(side, CourtSide::Near);
    }

    #[test]
    fn supported_interpolation_is_allowed_but_synthetic_only_crossing_is_not() {
        let detector = detector();
        let mut flight = trajectory([9.0, 0.0, 2.2], [-12.0, 1.0, 5.5]);
        flight.interpolated[17..22].fill(true);
        assert!(detector.supports_opening(&flight, (0, 39), 0));
        flight.interpolated.fill(true);
        assert!(!detector.supports_opening(&flight, (0, 39), 0));
        flight.interpolated.fill(false);
        flight.points[17..22].fill(None);
        assert!(!detector.supports_opening(&flight, (0, 39), 0));
    }
}
