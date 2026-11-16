use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    Track(u64),
    Athlete(String),
}

fn key(
    player: &Value,
    owners: &BTreeMap<u64, String>,
    conflicts: &BTreeSet<String>,
) -> Option<Key> {
    let id = player["id"].as_u64()?;
    Some(
        match owners.get(&id).filter(|owner| !conflicts.contains(*owner)) {
            Some(owner) => Key::Athlete(owner.clone()),
            None => Key::Track(id),
        },
    )
}

fn blend(a: &Value, b: &Value, t: f64, size: usize) -> Option<Value> {
    let a = a.as_array().filter(|v| v.len() == size)?;
    let b = b.as_array().filter(|v| v.len() == size)?;
    let values = a
        .iter()
        .zip(b)
        .map(|(a, b)| {
            let (a, b) = (a.as_f64()?, b.as_f64()?);
            (a.is_finite() && b.is_finite()).then_some(a * (1.0 - t) + b * t)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(json!(values))
}

fn joints(a: &Value, b: &Value, t: f64) -> Option<Value> {
    let a = a.as_array().filter(|v| v.len() == 17)?;
    let b = b.as_array().filter(|v| v.len() == 17)?;
    let points = a
        .iter()
        .zip(b)
        .map(|(a, b)| {
            let mut p = blend(a, b, t, 3)?;
            p[2] = json!(a[2].as_f64()?.min(b[2].as_f64()?));
            Some(p)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(json!(points))
}

pub(super) fn context(start: u64, end: u64, intervals: &[(u64, u64)]) -> (u64, u64) {
    intervals
        .iter()
        .filter(|(a, b)| *a < end && *b >= start)
        .fold((start, end), |(lo, hi), (a, b)| {
            (lo.min(*a), hi.max(b.saturating_add(1)))
        })
}

fn same_joints(a: &Value, b: &Value) -> bool {
    let (Some(a), Some(b)) = (a.as_array(), b.as_array()) else {
        return false;
    };
    if a.len() != 17 || b.len() != 17 {
        return false;
    }
    let mut matches = 0;
    let mut distance = 0.0;
    for (a, b) in a.iter().zip(b) {
        let (Some(ax), Some(ay), Some(ac), Some(bx), Some(by), Some(bc)) = (
            a[0].as_f64(),
            a[1].as_f64(),
            a[2].as_f64(),
            b[0].as_f64(),
            b[1].as_f64(),
            b[2].as_f64(),
        ) else {
            continue;
        };
        if ac >= 0.3 && bc >= 0.3 {
            distance += (ax - bx).hypot(ay - by);
            matches += 1;
        }
    }
    matches >= 6 && distance / (matches as f64) < 0.01
}

pub(super) fn append_poses(frames: &mut [Value]) {
    for row in frames {
        if row["in_rally"] != true {
            continue;
        }
        let mut poses = row["pose_points"].as_array().cloned().unwrap_or_default();
        let observed = !poses.is_empty() && row["pose_points_interpolated"] != true;
        let mut measured = observed;
        for p in row["players"].as_array().into_iter().flatten() {
            if !p["pose"].is_array()
                || poses
                    .iter()
                    .any(|pose| same_joints(&p["pose"], &pose["keypoints"]))
            {
                continue;
            }
            let interpolated = p["pose_interpolated"] == true;
            measured |= !interpolated;
            poses.push(json!({"box":p["box"],"keypoints":p["pose"],"track_id":p["id"],
                "interpolated":interpolated,"source":if interpolated {"cpu_rally_linear_v1"} else {"rtmpose_rgb"}}));
        }
        if !poses.is_empty() {
            row["pose_points"] = json!(poses);
            row["pose_points_interpolated"] = json!(!measured);
        }
    }
}

pub(super) fn fill(
    frames: &mut [Value],
    intervals: &[(u64, u64)],
    owners: &BTreeMap<u64, String>,
    interpolate_players: bool,
) {
    for &(start, end) in intervals {
        let lo = frames.partition_point(|row| row["frame"].as_u64().unwrap_or(u64::MAX) < start);
        let hi = frames.partition_point(|row| row["frame"].as_u64().unwrap_or(u64::MAX) <= end);
        let rows = &mut frames[lo..hi];
        let offset = rows
            .first()
            .and_then(|row| row["frame"].as_u64())
            .unwrap_or(start);
        let mut conflicts = BTreeSet::new();
        for row in rows.iter() {
            let mut seen = BTreeSet::new();
            for p in row["players"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|p| p["interpolated"] != true)
            {
                if let Some(owner) = p["id"].as_u64().and_then(|id| owners.get(&id)) {
                    if !seen.insert(owner) {
                        conflicts.insert(owner.clone());
                    }
                }
            }
        }
        let mut boxes = BTreeMap::<Key, Vec<(usize, Value)>>::new();
        let mut poses = BTreeMap::<Key, Vec<(usize, Value)>>::new();
        let mut balls = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            if row["ball"].is_array() && row["ball_interpolated"] != true {
                balls.push((i, row["ball"].clone()));
            }
            for p in row["players"].as_array().into_iter().flatten() {
                if !interpolate_players {
                    break;
                }
                let Some(k) = key(p, owners, &conflicts) else {
                    continue;
                };
                if p["interpolated"] != true {
                    boxes.entry(k.clone()).or_default().push((i, p.clone()));
                }
                if p["pose"].is_array() && p["pose_interpolated"] != true {
                    poses.entry(k).or_default().push((i, p.clone()));
                }
            }
        }
        for (k, anchors) in boxes {
            for pair in anchors.windows(2) {
                let ((a, left), (b, right)) = (&pair[0], &pair[1]);
                if b <= &(a + 1) {
                    continue;
                }
                for (i, row) in rows.iter_mut().enumerate().take(*b).skip(a + 1) {
                    let t = (i - a) as f64 / (b - a) as f64;
                    let Some(bb) = blend(&left["box"], &right["box"], t, 4) else {
                        continue;
                    };
                    let Some(players) = row["players"].as_array_mut() else {
                        continue;
                    };
                    if players
                        .iter()
                        .any(|p| key(p, owners, &conflicts).as_ref() == Some(&k))
                    {
                        continue;
                    }
                    if players.len() >= 128 {
                        continue;
                    }
                    if super::box_norm(&bb).is_some_and(|bb| {
                        players
                            .iter()
                            .filter(|p| p["interpolated"] != true)
                            .filter_map(|p| super::box_norm(&p["box"]))
                            .any(|measured| super::boxes_clash(bb, measured))
                    }) {
                        continue;
                    }
                    players.push(json!({"id":left["id"],"box":bb,"interpolated":true,
                        "interpolation_source":"cpu_rally_linear_v1",
                        "interpolation_endpoints":[left["id"],right["id"]],
                        "interpolation_frames":[offset+*a as u64,offset+*b as u64]}));
                }
            }
        }
        for (k, anchors) in poses {
            for pair in anchors.windows(2) {
                let ((a, left), (b, right)) = (&pair[0], &pair[1]);
                if b <= &(a + 1) {
                    continue;
                }
                for (i, row) in rows.iter_mut().enumerate().take(*b).skip(a + 1) {
                    let t = (i - a) as f64 / (b - a) as f64;
                    let Some(pose) = joints(&left["pose"], &right["pose"], t) else {
                        continue;
                    };
                    if let Some(p) = row["players"].as_array_mut().and_then(|players| {
                        players
                            .iter_mut()
                            .find(|p| key(p, owners, &conflicts).as_ref() == Some(&k))
                    }) {
                        if p["pose"].is_null() {
                            p["pose"] = pose;
                            p["pose_interpolated"] = json!(true);
                            p["pose_interpolation_frames"] =
                                json!([offset + *a as u64, offset + *b as u64]);
                        }
                    }
                }
            }
        }
        for pair in balls.windows(2) {
            let ((a, left), (b, right)) = (&pair[0], &pair[1]);
            if b <= &(a + 1) {
                continue;
            }
            for (i, row) in rows.iter_mut().enumerate().take(*b).skip(a + 1) {
                if !row["ball"].is_null() {
                    continue;
                }
                let t = (i - a) as f64 / (b - a) as f64;
                if let Some(point) = blend(left, right, t, 2) {
                    row["ball"] = point;
                    row["ball_interpolated"] = json!(true);
                    row["ball_origin"] = json!("vballnet_interpolated");
                    row["ball_interpolation_source"] = json!("cpu_rally_linear_v1");
                    row["ball_interpolation_frames"] =
                        json!([offset + *a as u64, offset + *b as u64]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::collections::BTreeMap;

    fn fill(frames: &mut [Value], intervals: &[(u64, u64)], owners: &BTreeMap<u64, String>) {
        super::fill(frames, intervals, owners, true);
    }

    fn player(id: u64, x: f64) -> Value {
        json!({"id":id,"box":[x,0.2,x+0.1,0.8],"interpolated":false,
            "pose":vec![[x+0.05,0.4,0.8];17],"pose_interpolated":false})
    }

    fn frames(count: u64) -> Vec<Value> {
        (0..count)
            .map(|f| json!({"frame":f,"in_rally":true,"players":[],"ball":null}))
            .collect()
    }

    #[test]
    fn fills_long_internal_box_pose_and_ball_gaps_without_extrapolation() {
        let mut rows = frames(125);
        rows[1]["players"] = json!([player(7, 0.1)]);
        rows[121]["players"] = json!([player(7, 0.5)]);
        rows[1]["ball"] = json!([0.1, 0.2]);
        rows[121]["ball"] = json!([0.7, 0.8]);
        fill(&mut rows, &[(0, 124)], &BTreeMap::new());
        assert_eq!(rows[61]["players"][0]["box"], json!([0.3, 0.2, 0.4, 0.8]));
        assert!((rows[61]["players"][0]["pose"][0][0].as_f64().unwrap() - 0.35).abs() < 1e-9);
        assert_eq!(rows[61]["players"][0]["interpolated"], true);
        assert_eq!(rows[61]["players"][0]["pose_interpolated"], true);
        assert!((rows[61]["ball"][0].as_f64().unwrap() - 0.4).abs() < 1e-9);
        assert!((rows[61]["ball"][1].as_f64().unwrap() - 0.5).abs() < 1e-9);
        assert_eq!(rows[61]["ball_interpolated"], true);
        assert_eq!(rows[0]["players"], json!([]));
        assert_eq!(rows[124]["ball"], Value::Null);
        assert_eq!(rows[1]["players"][0], player(7, 0.1));
    }

    #[test]
    fn bridges_only_confirmed_owner_and_keeps_real_observations() {
        let mut rows = frames(5);
        rows[0]["players"] = json!([player(7, 0.1)]);
        rows[2]["players"] = json!([{"id":9,"box":[0.2,0.2,0.3,0.8],"interpolated":false}]);
        rows[4]["players"] = json!([player(9, 0.5)]);
        let owners = BTreeMap::from([(7, "vlad".into()), (9, "vlad".into())]);
        let mut unlinked = rows.clone();
        fill(&mut unlinked, &[(0, 4)], &BTreeMap::new());
        assert_eq!(unlinked[1]["players"], json!([]));
        fill(&mut rows, &[(0, 4)], &owners);
        assert_eq!(rows[1]["players"][0]["id"], 7);
        assert_eq!(rows[3]["players"][0]["id"], 9);
        assert_eq!(rows[2]["players"][0]["box"], json!([0.2, 0.2, 0.3, 0.8]));
        assert_eq!(rows[2]["players"][0]["pose_interpolated"], true);
    }

    #[test]
    fn never_bridges_rallies_or_conflicting_athletes() {
        let mut rows = frames(8);
        rows[1]["players"] = json!([player(7, 0.1)]);
        rows[6]["players"] = json!([player(7, 0.5)]);
        rows[1]["ball"] = json!([0.1, 0.2]);
        rows[6]["ball"] = json!([0.7, 0.8]);
        fill(&mut rows, &[(0, 3), (4, 7)], &BTreeMap::new());
        assert_eq!(rows[3]["players"], json!([]));
        assert_eq!(rows[4]["ball"], Value::Null);
        let mut rows = frames(5);
        rows[0]["players"] = json!([player(7, 0.1)]);
        rows[2]["players"] = json!([player(7, 0.2), player(9, 0.6)]);
        rows[4]["players"] = json!([player(9, 0.5)]);
        fill(
            &mut rows,
            &[(0, 4)],
            &BTreeMap::from([(7, "vlad".into()), (9, "vlad".into())]),
        );
        assert!(rows[1]["players"].as_array().unwrap().len() <= 1);
        assert!(rows[3]["players"].as_array().unwrap().len() <= 1);
        assert_eq!(rows[2]["players"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn full_rally_context_keeps_paged_results_identical() {
        assert_eq!(context(180, 360, &[(20, 380), (400, 500)]), (20, 381));
        let mut whole = frames(401);
        whole[20]["players"] = json!([player(7, 0.1)]);
        whole[380]["players"] = json!([player(7, 0.5)]);
        let (a, b) = context(180, 360, &[(20, 380)]);
        let mut page = whole[a as usize..b as usize].to_vec();
        fill(&mut whole, &[(20, 380)], &BTreeMap::new());
        fill(&mut page, &[(20, 380)], &BTreeMap::new());
        assert_eq!(whole[180..360], page[160..340]);
        assert_eq!(page[180]["players"][0]["interpolated"], true);
    }

    #[test]
    fn adds_missing_skeleton_without_duplicating_independent_measured_pose() {
        let measured = json!({"box":[0.1,0.2,0.2,0.8],"keypoints":vec![[0.15,0.4,0.8];17]});
        let mut rows = vec![json!({"frame":1,"in_rally":true,
            "players":[player(7,0.1),player(9,0.5)],"pose_points":[measured]})];
        rows[0]["players"][1]["pose_interpolated"] = json!(true);
        append_poses(&mut rows);
        assert_eq!(rows[0]["pose_points"].as_array().unwrap().len(), 2);
        assert_eq!(rows[0]["pose_points"][0], measured);
        assert_eq!(rows[0]["pose_points"][1]["interpolated"], true);
    }

    #[test]
    fn measured_alternate_id_blocks_a_synthetic_ghost() {
        let mut rows = frames(5);
        rows[0]["players"] = json!([player(7, 0.1)]);
        rows[4]["players"] = json!([player(7, 0.1)]);
        for row in &mut rows[1..4] {
            row["players"] = json!([player(9, 0.1)]);
        }
        let mut linked = rows.clone();
        fill(&mut rows, &[(0, 4)], &BTreeMap::new());
        fill(
            &mut linked,
            &[(0, 4)],
            &BTreeMap::from([(7, "vlad".into())]),
        );
        assert!(rows
            .iter()
            .all(|row| row["players"].as_array().unwrap().len() == 1));
        assert!(linked
            .iter()
            .all(|row| row["players"].as_array().unwrap().len() == 1));
    }

    #[test]
    fn disabled_player_interpolation_keeps_gaps_while_ball_continues() {
        let mut rows = frames(5);
        rows[0]["players"] = json!([player(7, 0.1)]);
        rows[4]["players"] = json!([player(7, 0.5)]);
        rows[0]["ball"] = json!([0.2, 0.3]);
        rows[4]["ball"] = json!([0.6, 0.7]);
        super::fill(&mut rows, &[(0, 4)], &BTreeMap::new(), false);
        assert_eq!(rows[2]["players"], json!([]));
        assert!(rows[2]["ball"].is_array());
    }
}
