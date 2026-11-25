use crate::court_geometry::CourtSide;

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
