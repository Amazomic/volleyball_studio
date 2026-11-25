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

pub(crate) fn parse_initial_serves(
    data: &Value,
    manifest: &Value,
    intervals: &[(u64, u64)],
) -> Result<BTreeMap<u64, (u64, u64)>, String> {
    let mut result = BTreeMap::new();
    if data["initial_serve_proposals"].is_null()
        && manifest["initial_serve_proposal_count"].is_null()
    {
        return Ok(result);
    }
    let rows = data["initial_serve_proposals"]
        .as_array()
        .ok_or("Invalid initial serve proposals")?;
    if manifest["initial_serve_proposal_count"].as_u64() != Some(rows.len() as u64)
        || rows.len() > intervals.len()
    {
        return Err("Initial serve count mismatch".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    for row in rows {
        let start = row["rally_start_frame"]
            .as_u64()
            .ok_or("Invalid initial serve start")?;
        let index = intervals
            .binary_search_by_key(&start, |&(from, _)| from)
            .map_err(|_| "Initial serve is outside rally intervals")?;
        let at = row["trajectory_start_frame"].as_u64();
        let id = row["provisional_track_id"].as_u64();
        if !seen.insert(start)
            || row["kind"] != "serve"
            || row["source"] != "rally_trajectory_start_v1"
            || row["identity_verified"] != false
            || (!row["trajectory_start_frame"].is_null() && at.is_none())
            || (!row["provisional_track_id"].is_null() && id.is_none())
            || at.is_some_and(|at| {
                at < start
                    || at > intervals[index].1
                    || at >= start.saturating_add(crate::cpu_touches::INITIAL_SERVE_SEARCH_FRAMES)
            })
            || id.is_some_and(|id| id == 0 || id > i64::MAX as u64 || at.is_none())
        {
            return Err("Invalid initial serve proposal".into());
        }
        if let Some(pair) = at.zip(id) {
            result.insert(start, pair);
        }
    }
    Ok(result)
}

pub(crate) fn include_initial_serve(
    metadata: &mut Value,
    roster: &[u64],
    interval: (u64, u64),
    first_floor: Option<u64>,
    offset_sec: f64,
    serve: Option<(u64, u64)>,
) {
    let Some((at, id)) = serve else { return };
    if !(interval.0..=interval.1).contains(&at)
        || first_floor.is_some_and(|floor| at >= floor)
        || !roster.contains(&id)
    {
        return;
    }
    let Some(touches) = metadata["touches"].as_array_mut() else {
        return;
    };
    if touches.iter().any(|event| event["kind"] == "serve") {
        return;
    }
    let time = offset_sec + at as f64 / 30.0;
    if let Some(event) = touches.iter_mut().find(|event| {
        event["track_id"].as_u64() == Some(id)
            && event["time_sec"]
                .as_f64()
                .is_some_and(|t| (t - time).abs() < 1e-8)
    }) {
        event["kind"] = json!("serve");
    } else {
        let index =
            touches.partition_point(|event| event["time_sec"].as_f64().is_some_and(|t| t < time));
        touches.insert(index, json!({"track_id":id,"kind":"serve","time_sec":time}));
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
    fn initial_serve_source_is_scoped_and_unassigned_is_not_guessed() {
        let proposal = |start, at, id| {
            json!({
                "rally_start_frame":start,"trajectory_start_frame":at,
                "provisional_track_id":id,"kind":"serve",
                "source":"rally_trajectory_start_v1","identity_verified":false
            })
        };
        let data = json!({"initial_serve_proposals":[
            proposal(10, 15, Some(1)), proposal(50, 52, None)
        ]});
        let manifest = json!({"initial_serve_proposal_count":2});
        let ranges = [(10, 40), (50, 80)];
        assert_eq!(
            parse_initial_serves(&data, &manifest, &ranges).unwrap(),
            BTreeMap::from([(10, (15, 1))])
        );
        let mut invalid = data.clone();
        invalid["initial_serve_proposals"][1]["rally_start_frame"] = json!(10);
        assert!(parse_initial_serves(&invalid, &manifest, &ranges).is_err());
        invalid = data.clone();
        invalid["initial_serve_proposals"][0]["trajectory_start_frame"] = json!(50);
        assert!(parse_initial_serves(&invalid, &manifest, &ranges).is_err());
    }

    #[test]
    fn initial_serve_is_added_to_saved_metadata_once_and_only_for_roster() {
        let mut saved = json!({"touches":[{"track_id":2,"kind":"dig","time_sec":3.0}]});
        include_initial_serve(&mut saved, &[1, 2], (10, 100), Some(80), 2.0, Some((15, 1)));
        assert_eq!(
            saved["touches"],
            json!([
                {"track_id":1,"kind":"serve","time_sec":2.5},
                {"track_id":2,"kind":"dig","time_sec":3.0}
            ])
        );
        let expected = saved.clone();
        include_initial_serve(&mut saved, &[1, 2], (10, 100), Some(80), 2.0, Some((15, 1)));
        assert_eq!(saved, expected);
        let mut other = json!({"touches":[]});
        include_initial_serve(&mut other, &[1, 2], (10, 100), Some(80), 2.0, Some((15, 7)));
        include_initial_serve(&mut other, &[1, 2], (10, 100), Some(12), 2.0, Some((15, 1)));
        assert_eq!(other["touches"], json!([]));
    }

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
