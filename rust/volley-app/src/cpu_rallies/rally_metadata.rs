use serde_json::{json, Value};
use std::collections::BTreeMap;
use volley_core::{
    court_geometry::{CourtSide, CourtTransform},
    rally_score::{scoring_landing, winning_side},
};

pub(crate) fn refresh_line_winner(
    metadata: &mut Value,
    outcome: &str,
    position: Option<[f64; 2]>,
    court: &CourtTransform,
) {
    if outcome != "line_uncertain" {
        return;
    }
    let winner = if ["team_1", "team_2"].iter().all(|team| {
        metadata[*team]
            .as_array()
            .is_some_and(|ids| !ids.is_empty())
    }) {
        winning_side(Some(scoring_landing(outcome, position, Some(court))), None)
    } else {
        None
    };
    metadata["winning_team"] = json!(winner.map(team_label));
}

fn team_label(side: CourtSide) -> &'static str {
    match side {
        CourtSide::Near => "team_1",
        CourtSide::Far => "team_2",
    }
}

pub(crate) fn build(
    start: u64,
    end: u64,
    sides: &BTreeMap<u64, CourtSide>,
    roster: &[u64],
    observed: &[(u64, Option<u64>, &str)],
    floor: Option<(u64, &str)>,
    offset_sec: f64,
) -> Value {
    let team = |side| {
        roster
            .iter()
            .copied()
            .filter(|id| sides.get(id) == Some(&side))
            .collect::<Vec<_>>()
    };
    let team_1 = team(CourtSide::Near);
    let team_2 = team(CourtSide::Far);
    let last_touch = floor.and_then(|(at, _)| {
        observed
            .iter()
            .rev()
            .find(|(frame, _, _)| *frame < at)
            .and_then(|(_, id, _)| {
                id.and_then(|id| roster.contains(&id).then(|| sides.get(&id)).flatten())
                    .copied()
            })
    });
    let winner = if !team_1.is_empty() && !team_2.is_empty() {
        winning_side(floor.map(|(_, outcome)| outcome), last_touch)
    } else {
        None
    };
    let touches = observed.iter().filter(|(frame, _, _)| floor.is_none_or(|(at, _)| *frame < at))
        .filter_map(|(frame, id, kind)| {
            id.filter(|id| roster.contains(id)).map(|id| json!({"track_id":id,"kind":kind,"time_sec":offset_sec + *frame as f64 / 30.0}))
        }).collect::<Vec<_>>();
    json!({
        "start_frame":start,
        "end_frame":end,
        "team_1":team_1,
        "team_2":team_2,
        "touches":touches,
        "winning_team":winner.map(team_label)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saved_line_score_uses_geometry_without_changing_roster_or_touches() {
        let court = CourtTransform::new([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]).unwrap();
        let sides = BTreeMap::from([(1, CourtSide::Near), (2, CourtSide::Far)]);
        let mut saved = build(
            0,
            40,
            &sides,
            &[1, 2],
            &[(20, Some(1), "attack")],
            Some((25, "line_uncertain")),
            0.0,
        );
        let touches = saved["touches"].clone();
        for (x, expected) in [(-3.0, "team_2"), (3.0, "team_1")] {
            refresh_line_winner(
                &mut saved,
                "line_uncertain",
                court.floor_to_image([x, -4.5]),
                &court,
            );
            assert_eq!(saved["winning_team"], expected);
            assert_eq!(saved["touches"], touches);
        }
        refresh_line_winner(&mut saved, "line_uncertain", None, &court);
        assert!(saved["winning_team"].is_null());
    }

    #[test]
    fn metadata_uses_only_roster_and_first_floor() {
        let sides = BTreeMap::from([
            (1, CourtSide::Near),
            (2, CourtSide::Far),
            (9, CourtSide::Far),
        ]);
        let observed = [
            (10, Some(1), "serve"),
            (20, Some(2), "attack"),
            (30, Some(1), "dig"),
        ];
        let result = build(0, 40, &sides, &[1, 2, 3], &observed, Some((25, "out")), 2.0);
        assert_eq!(result["team_1"], json!([1]));
        assert_eq!(result["team_2"], json!([2]));
        assert_eq!(result["winning_team"], "team_1");
        assert_eq!(result["touches"].as_array().unwrap().len(), 2);
        assert_eq!(result["touches"][0]["kind"], "serve");
        assert_eq!(result["touches"][0]["time_sec"], json!(2.0 + 10.0 / 30.0));
        let unknown = build(
            0,
            40,
            &sides,
            &[1, 2],
            &[(20, None, "unknown_touch")],
            Some((25, "out")),
            0.0,
        );
        assert!(unknown["winning_team"].is_null());
    }
}
