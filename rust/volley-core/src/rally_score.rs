use crate::court_geometry::{CourtSide, CourtTransform};

pub const RALLY_SCORE_VERSION: &str = "court_line_ball_radius_evidence_v4";

pub fn scoring_landing<'a>(
    landing: &'a str,
    position: Option<[f64; 2]>,
    court: Option<&CourtTransform>,
) -> &'a str {
    scoring_landing_with_observed_center(landing, position, court, None)
}

pub fn scoring_landing_with_observed_center<'a>(
    landing: &'a str,
    position: Option<[f64; 2]>,
    court: Option<&CourtTransform>,
    observed_center: Option<[f64; 2]>,
) -> &'a str {
    if landing != "line_uncertain" {
        return landing;
    }
    let Some((court, point)) = court.zip(position) else {
        return landing;
    };
    let Some([x, y]) = court.to_court(point) else {
        return landing;
    };
    let side = |x: f64| {
        if x < -0.15 {
            "near_half"
        } else if x > 0.15 {
            "far_half"
        } else {
            "line_uncertain"
        }
    };
    if x.abs() > 0.15 {
        return side(x);
    }
    if y.abs() <= crate::court_geometry::COURT_WIDTH_M / 2.0 {
        if let Some(center) =
            observed_center.filter(|[u, v]| (u - point[0]).hypot(v - point[1]) <= 0.006)
        {
            if let Some((corrected, measured)) =
                ball_center_footprint(court, point).zip(ball_center_footprint(court, center))
            {
                let in_court = |[cx, cy]: [f64; 2]| {
                    cy.abs() <= crate::court_geometry::COURT_WIDTH_M / 2.0
                        && cx.abs() <= crate::court_geometry::COURT_LENGTH_M / 2.0
                };
                if in_court(corrected)
                    && in_court(measured)
                    && side(corrected[0]) == side(measured[0])
                {
                    return side(corrected[0]);
                }
            }
        }
    }
    landing
}

fn ball_center_footprint(court: &CourtTransform, [u, v]: [f64; 2]) -> Option<[f64; 2]> {
    const BALL_RADIUS_M: f64 = 0.105;
    let p = court.flight_projection(2.43)?;
    let a = p[0][0] - u * p[2][0];
    let b = p[0][1] - u * p[2][1];
    let c = p[1][0] - v * p[2][0];
    let d = p[1][1] - v * p[2][1];
    let e = u * p[2][3] - p[0][3] + (u * p[2][2] - p[0][2]) * BALL_RADIUS_M;
    let f = v * p[2][3] - p[1][3] + (v * p[2][2] - p[1][2]) * BALL_RADIUS_M;
    let determinant = a * d - b * c;
    if !determinant.is_finite() || determinant.abs() < 1e-10 {
        return None;
    }
    let x = (e * d - b * f) / determinant;
    let y = (a * f - e * c) / determinant;
    (x.is_finite() && y.is_finite()).then_some([x, y])
}

pub fn winning_side(landing: Option<&str>, last_touch: Option<CourtSide>) -> Option<CourtSide> {
    match landing {
        Some("near_half") => Some(CourtSide::Far),
        Some("far_half") => Some(CourtSide::Near),
        Some("out") => last_touch.map(|side| match side {
            CourtSide::Near => CourtSide::Far,
            CourtSide::Far => CourtSide::Near,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_ball_awards_point_to_opposite_half_not_last_hitter() {
        let court = crate::court_geometry::CourtTransform::new([
            [0.1, 0.2],
            [0.9, 0.2],
            [0.9, 0.8],
            [0.1, 0.8],
        ])
        .unwrap();
        for (x, winner) in [(-3.0, CourtSide::Far), (3.0, CourtSide::Near)] {
            let point = court.floor_to_image([x, -4.5]).unwrap();
            let landing = scoring_landing("line_uncertain", Some(point), Some(&court));
            assert_eq!(
                winning_side(Some(landing), Some(CourtSide::Near)),
                Some(winner)
            );
        }
        assert_eq!(
            scoring_landing("line_uncertain", None, Some(&court)),
            "line_uncertain"
        );
        assert_eq!(
            scoring_landing(
                "line_uncertain",
                court.floor_to_image([0.0, 4.5]),
                Some(&court)
            ),
            "line_uncertain"
        );
        assert_eq!(
            scoring_landing("out", court.floor_to_image([-3.0, -4.5]), Some(&court)),
            "out"
        );
    }

    #[test]
    fn ambiguous_net_footprint_uses_ball_center_height() {
        let court = CourtTransform::new([[0.1, 1.05], [0.38, 0.34], [0.62, 0.34], [0.9, 1.05]])
            .unwrap()
            .with_net(Some([[0.28, 0.325], [0.72, 0.325]]))
            .unwrap();
        let point = court.floor_to_image([0.04, -4.3]).unwrap();
        assert_eq!(
            scoring_landing("line_uncertain", Some(point), Some(&court)),
            "line_uncertain"
        );
        let landing = scoring_landing_with_observed_center(
            "line_uncertain",
            Some(point),
            Some(&court),
            Some(point),
        );
        assert_eq!(landing, "near_half");
        assert_eq!(winning_side(Some(landing), None), Some(CourtSide::Far));
        assert_eq!(
            scoring_landing_with_observed_center(
                "line_uncertain",
                Some(point),
                Some(&court),
                Some([0.8, 0.8])
            ),
            "line_uncertain"
        );
    }

    #[test]
    fn rally_score_uses_landing_half_or_last_touch_for_out() {
        assert_eq!(winning_side(Some("near_half"), None), Some(CourtSide::Far));
        assert_eq!(winning_side(Some("far_half"), None), Some(CourtSide::Near));
        assert_eq!(
            winning_side(Some("out"), Some(CourtSide::Near)),
            Some(CourtSide::Far)
        );
        assert_eq!(
            winning_side(Some("out"), Some(CourtSide::Far)),
            Some(CourtSide::Near)
        );
        assert_eq!(winning_side(Some("out"), None), None);
        assert_eq!(
            winning_side(Some("line_uncertain"), Some(CourtSide::Near)),
            None
        );
        assert_eq!(winning_side(None, Some(CourtSide::Near)), None);
    }
}
