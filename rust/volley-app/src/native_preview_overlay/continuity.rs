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

const MAX_TORSO_MISSING_FRAMES: usize = 2 * 30;

pub(super) fn fill_torso_points(frames: &mut [Value], intervals: &[(u64, u64)]) {
    for &(start, end) in intervals {
        let lo = frames.partition_point(|row| row["frame"].as_u64().unwrap_or(u64::MAX) < start);
        let hi = frames.partition_point(|row| row["frame"].as_u64().unwrap_or(u64::MAX) <= end);
        let rows = &mut frames[lo..hi];
        let mut anchors = BTreeMap::<u64, Vec<(usize, Value)>>::new();
        for (index, row) in rows.iter().enumerate() {
            for player in row["players"].as_array().into_iter().flatten() {
                if let Some(id) = player["id"].as_u64() {
                    if player["torso_point"].is_array() {
                        anchors
                            .entry(id)
                            .or_default()
                            .push((index, player["torso_point"].clone()));
                    }
                }
            }
        }
        let mut points = vec![Vec::new(); rows.len()];
        for (id, positions) in anchors {
            for (index, (frame, point)) in positions.iter().enumerate() {
                points[*frame].push(json!({"track_id":id,"point":point}));
                if let Some((next, target)) = positions.get(index + 1) {
                    let span = next.saturating_sub(*frame);
                    if (2..=MAX_TORSO_MISSING_FRAMES + 1).contains(&span) {
                        for (step, row) in points[*frame + 1..*next].iter_mut().enumerate() {
                            if let Some(position) =
                                blend(point, target, (step + 1) as f64 / span as f64, 2)
                            {
                                row.push(
                                    json!({"track_id":id,"point":position,"interpolated":true}),
                                );
                            }
                        }
                    }
                } else {
                    let stop = (*frame + 1 + MAX_TORSO_MISSING_FRAMES).min(points.len());
                    for row in &mut points[*frame + 1..stop] {
                        row.push(json!({"track_id":id,"point":point,"predicted":true}));
                    }
                }
            }
        }
        for (row, points) in rows.iter_mut().zip(points) {
            if !points.is_empty() {
                row["torso_points"] = json!(points);
            }
        }
    }
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
        let players = row["players"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default();
        let refined = players.iter().any(|p| {
            matches!(
                p["pose_source"].as_str(),
                Some(
                    crate::cpu_retrack::POSE_REFINEMENT_POLICY
                        | crate::cpu_retrack::LEGACY_POSE_REFINEMENT_POLICY
                        | crate::cpu_retrack::PREVIOUS_POSE_REFINEMENT_POLICY
                        | crate::cpu_retrack::PRIOR_CROP_REFINEMENT_POLICY
                        | crate::cpu_retrack::HISTORICAL_LEADING_CROP_POLICY
                        | crate::cpu_retrack::PREVIOUS_TRACKED_CROP_REFINEMENT_POLICY
                )
            )
        });
        let mut poses = if refined {
            Vec::new()
        } else {
            row["pose_points"].as_array().cloned().unwrap_or_default()
        };
        let observed = !poses.is_empty() && row["pose_points_interpolated"] != true;
        let mut measured = observed;
        for p in players {
            if !p["pose"].is_array()
                || poses
                    .iter()
                    .any(|pose| same_joints(&p["pose"], &pose["keypoints"]))
            {
                continue;
            }
            let interpolated = p["pose_interpolated"] == true;
            measured |= !interpolated;
            let source = p["pose_source"].as_str().unwrap_or(if interpolated {
                "cpu_rally_linear_v1"
            } else {
                "rtmpose_rgb"
            });
            poses.push(
                json!({"box":p["box"],"keypoints":p["pose"],"track_id":p["id"],
                "interpolated":interpolated,"source":source}),
            );
        }
        if !poses.is_empty() {
            row["pose_points"] = json!(poses);
            row["pose_points_interpolated"] = json!(!measured);
        }
    }
}

/// Two independent interpolations can fill the same missing body in the same
/// frame when measured Track IDs alternated. Never collapse real detections:
/// the saved replay and athlete assignments remain authoritative.
fn same_synthetic_body(a: &Value, b: &Value) -> bool {
    let (Some(a), Some(b)) = (super::box_norm(a), super::box_norm(b)) else {
        return false;
    };
    let area = |r: [f64; 4]| (r[2] - r[0]) * (r[3] - r[1]);
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let a_area = area(a);
    let b_area = area(b);
    let iou = intersection / (a_area + b_area - intersection).max(1e-9);
    let contained = intersection / a_area.min(b_area).max(1e-9);
    let centers_close = ((a[0] + a[2]) - (b[0] + b[2])).abs() < 0.030
        && ((a[1] + a[3]) - (b[1] + b[3])).abs() < 0.060;
    iou >= 0.55 && contained >= 0.80 && centers_close
}

/// Keep the locally best supported interpolated track, not whichever athlete
/// happens to sort first. This operates on preview-only rows; measured source
/// rows are never changed and IDs with distinct physical boxes stay separate.
fn remove_synthetic_duplicates(rows: &mut [Value], support: &BTreeMap<(usize, u64), usize>) {
    for (frame, row) in rows.iter_mut().enumerate() {
        let Some(players) = row["players"].as_array_mut() else {
            continue;
        };
        if players.len() < 2 || !players.iter().any(|p| p["interpolated"] == true) {
            continue;
        }
        let priority = |p: &Value| {
            let id = p["id"].as_u64().unwrap_or(0);
            let gap = p["interpolation_frames"][1]
                .as_u64()
                .unwrap_or(u64::MAX)
                .saturating_sub(p["interpolation_frames"][0].as_u64().unwrap_or(0));
            (
                p["interpolated"] != true,
                support.get(&(frame, id)).copied().unwrap_or(0),
                std::cmp::Reverse(gap),
                std::cmp::Reverse(id),
            )
        };
        let mut ranked = (0..players.len()).collect::<Vec<_>>();
        ranked.sort_by(|&a, &b| priority(&players[b]).cmp(&priority(&players[a])));
        let mut retained: Vec<usize> = Vec::new();
        for index in ranked {
            let candidate = &players[index];
            if candidate["interpolated"] == true
                && retained.iter().any(|&kept| {
                    candidate["occluding_track_id"].as_u64() != players[kept]["id"].as_u64()
                        && same_synthetic_body(&candidate["box"], &players[kept]["box"])
                })
            {
                continue;
            }
            retained.push(index);
        }
        retained.sort_unstable();
        *players = retained
            .into_iter()
            .map(|index| players[index].clone())
            .collect();
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
        let mut support = BTreeMap::<(usize, u64), usize>::new();
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
                    let count_before =
                        anchors.partition_point(|(at, _)| *at < i.saturating_sub(15));
                    let count_after =
                        anchors.partition_point(|(at, _)| *at <= i.saturating_add(15));
                    if let Some(id) = left["id"].as_u64() {
                        support.insert((i, id), count_after - count_before);
                    }
                    players.push(json!({"id":left["id"],"box":bb,"interpolated":true,
                        "interpolation_source":"cpu_rally_linear_v1",
                        "interpolation_endpoints":[left["id"],right["id"]],
                        "interpolation_frames":[offset+*a as u64,offset+*b as u64]}));
                }
            }
        }
        remove_synthetic_duplicates(rows, &support);
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

    #[test]
    fn refined_pose_replaces_old_unassigned_skeleton_in_processed_preview() {
        let mut frames = [json!({
            "in_rally":true,
            "pose_points":[{"keypoints":vec![[0.2,0.3,0.95];17]}],
            "players":[{"id":7,"pose":vec![[0.4,0.5,0.95];17],
                "pose_source":crate::cpu_retrack::POSE_REFINEMENT_POLICY}]
        })];
        append_poses(&mut frames);
        let poses = frames[0]["pose_points"].as_array().unwrap();
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0]["track_id"], 7);
        assert_eq!(poses[0]["keypoints"][0], json!([0.4, 0.5, 0.95]));
    }

    #[test]
    fn torso_points_fill_two_seconds_and_resume_after_longer_break() {
        let mut frames = (0..=185)
            .map(|frame| json!({"frame":frame,"in_rally":true,"players":[]}))
            .collect::<Vec<_>>();
        for (frame, x) in [(0, 0.2), (61, 0.4), (123, 0.5)] {
            frames[frame]["players"] = json!([{"id":7,"torso_point":[x,0.5]}]);
        }
        fill_torso_points(&mut frames, &[(0, 184)]);
        assert_eq!(frames[1]["torso_points"][0]["track_id"], 7);
        assert_eq!(frames[60]["torso_points"][0]["interpolated"], true);
        assert_eq!(frames[61]["torso_points"][0]["point"], json!([0.4, 0.5]));
        assert!(frames[62]["torso_points"].is_null());
        assert_eq!(frames[123]["torso_points"][0]["track_id"], 7);
        assert_eq!(frames[183]["torso_points"][0]["predicted"], true);
        assert!(frames[184]["torso_points"].is_null());
        assert!(frames[185]["torso_points"].is_null());
    }
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
    fn alternating_vlad_irek_31_7_does_not_draw_two_synthetic_players() {
        // Production CPU replay in project "uu": Track 6 belongs to Irek,
        // Track 10 to Vlad. The measured IDs alternate at 31.5-32.0s,
        // although their boxes describe one moving body. Do not globally
        // merge the IDs: both have independent observations elsewhere.
        let mut rows = (940..=965_u64)
            .map(|frame| json!({"frame":frame,"in_rally":true,"players":[],"ball":null}))
            .collect::<Vec<_>>();
        let put = |rows: &mut Vec<Value>, frame: u64, id, bbox: [f64; 4]| {
            rows[(frame - 940) as usize]["players"] =
                json!([{"id":id,"box":bbox,"interpolated":false}]);
        };
        put(&mut rows, 940, 6, [0.4350, 0.3799, 0.4514, 0.5234]);
        put(&mut rows, 945, 6, [0.4277, 0.4062, 0.4454, 0.5179]);
        put(&mut rows, 950, 10, [0.4242, 0.4196, 0.4465, 0.4954]);
        put(&mut rows, 955, 6, [0.4152, 0.4195, 0.4495, 0.5197]);
        put(&mut rows, 960, 10, [0.4162, 0.4249, 0.4385, 0.5216]);
        put(&mut rows, 965, 6, [0.4126, 0.4271, 0.4351, 0.5217]);
        let linked = BTreeMap::from([(6, "irek".to_string()), (10, "vlad".to_string())]);
        fill(&mut rows, &[(940, 965)], &linked);
        for at in 951..955 {
            let players = rows[(at - 940) as usize]["players"].as_array().unwrap();
            assert_eq!(players.len(), 1, "duplicate synthetic player at frame {at}");
            assert_eq!(players[0]["id"], 6, "use the locally supported body");
            assert_eq!(players[0]["interpolated"], true);
        }
        // The true, immutable detections and linked Track IDs are intact.
        assert_eq!(rows[10]["players"][0]["id"], 10);
        assert_eq!(rows[10]["players"][0]["interpolated"], false);
        assert_eq!(rows[15]["players"][0]["id"], 6);
        assert_eq!(rows[15]["players"][0]["interpolated"], false);
    }

    #[test]
    fn independent_neighboring_players_and_real_overlaps_are_preserved() {
        let mut rows = frames(3);
        rows[0]["players"] = json!([
            {"id":6,"box":[0.20,0.4,0.28,0.8],"interpolated":false},
            {"id":10,"box":[0.27,0.4,0.35,0.8],"interpolated":false}
        ]);
        rows[2]["players"] = rows[0]["players"].clone();
        fill(&mut rows, &[(0, 2)], &BTreeMap::new());
        assert_eq!(rows[1]["players"].as_array().unwrap().len(), 2);
        // Even identical measured boxes remain untouched: they must be
        // resolved by tracking/review, never silently removed from preview.
        let mut measured = frames(1);
        measured[0]["players"] = json!([
            {"id":6,"box":[0.2,0.4,0.3,0.8],"interpolated":false},
            {"id":10,"box":[0.2,0.4,0.3,0.8],"interpolated":false}
        ]);
        fill(&mut measured, &[(0, 0)], &BTreeMap::new());
        assert_eq!(measured[0]["players"].as_array().unwrap().len(), 2);
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
