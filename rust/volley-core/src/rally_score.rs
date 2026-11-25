use crate::court_geometry::{CourtSide, CourtTransform};

pub const RALLY_SCORE_VERSION: &str = "court_line_half_v2";

pub fn scoring_landing<'a>(
    landing: &'a str,
    position: Option<[f64; 2]>,
    court: Option<&CourtTransform>,
) -> &'a str {
    if landing != "line_uncertain" {
        return landing;
    }
    match court
        .zip(position)
        .and_then(|(court, point)| court.to_court(point))
    {
        Some([x, _]) if x < -0.15 => "near_half",
        Some([x, _]) if x > 0.15 => "far_half",
        _ => landing,
    }
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
