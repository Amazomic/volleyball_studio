use super::{BoundedReader, PlayerFrameScope, MAX_ROW_BYTES};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
};
use volley_core::{bbox_iou, fusion::Candidate, tracker::PoseEvidence};

#[derive(Deserialize)]
struct Frame {
    frame_index: u64,
    tracked_players: Vec<Observation>,
}

#[derive(Deserialize)]
struct Observation {
    track_id: u64,
    detection: Candidate,
    pose: Option<[[f64; 3]; 17]>,
}

#[derive(Default)]
struct Track {
    start: Option<u64>,
    end: u64,
    observed: u64,
    first_head: Option<(u64, [f64; 2])>,
    last_head: Option<(u64, [f64; 2])>,
    first_pose: Option<(u64, PoseEvidence)>,
    last_pose: Option<(u64, PoseEvidence)>,
    first_box: Option<[f64; 4]>,
    last_box: Option<[f64; 4]>,
}

#[derive(Default)]
struct Simultaneous {
    total: u32,
    complementary: u32,
}

fn complementary_parts(a: &Candidate, b: &Candidate) -> bool {
    let (a, b) = (a.bbox_xyxy_norm, b.bbox_xyxy_norm);
    let (width_a, width_b) = (a[2] - a[0], b[2] - b[0]);
    let (height_a, height_b) = (a[3] - a[1], b[3] - b[1]);
    if width_a.min(width_b) < 0.010 || height_a.min(height_b) < 0.040 {
        return false;
    }
    let horizontal = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let center_x = ((a[0] + a[2]) - (b[0] + b[2])).abs() * 0.5;
    let overlap_y = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let gap_y = (a[1].max(b[1]) - a[3].min(b[3])).max(0.0);
    horizontal >= 0.70 * width_a.min(width_b)
        && center_x <= 0.25 * width_a.min(width_b)
        && overlap_y <= 0.12 * height_a.min(height_b)
        && gap_y <= 0.012
}

fn observed_row(reader: &mut BoundedReader, expected_frame: u64) -> Result<Frame, String> {
    let line = reader
        .line()?
        .ok_or("Confirmed Track ID replay truncated during handoff stitching")?;
    let row: Frame = serde_json::from_slice(&line)
        .map_err(|e| format!("Invalid confirmed tracking row: {e}"))?;
    if row.frame_index != expected_frame || row.tracked_players.len() > 128 {
        return Err("Track ID stitching requires ordered confirmed frames".into());
    }
    Ok(row)
}

fn find_handoffs(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    scope: Option<&PlayerFrameScope>,
) -> Result<BTreeMap<u64, u64>, String> {
    let mut tracks = BTreeMap::<u64, Track>::new();
    let mut pairs = BTreeMap::<(u64, u64), Simultaneous>::new();
    let mut reader = BoundedReader::open(source)?;
    for frame in 0..frames {
        let row = observed_row(&mut reader, frame)?;
        let mut in_frame = BTreeSet::new();
        for observation in &row.tracked_players {
            if observation.track_id == 0 || !in_frame.insert(observation.track_id) {
                return Err("Duplicate or invalid Track ID before stitching".into());
            }
            if !tracks.contains_key(&observation.track_id) && tracks.len() >= 1_000_000 {
                return Err("Too many Track IDs for bounded stitching".into());
            }
            let track = tracks.entry(observation.track_id).or_default();
            track.start.get_or_insert(frame);
            track.end = frame;
            track.observed += 1;
            track
                .first_box
                .get_or_insert(observation.detection.bbox_xyxy_norm);
            track.last_box = Some(observation.detection.bbox_xyxy_norm);
            if let Some(pose) = observation
                .pose
                .map(|joints| PoseEvidence { joints })
                .filter(PoseEvidence::valid)
            {
                track.first_pose.get_or_insert((frame, pose));
                track.last_pose = Some((frame, pose));
            }
            if let Some(head) = observation.detection.head_point_norm {
                if head.iter().all(|value| value.is_finite()) {
                    track.first_head.get_or_insert((frame, head));
                    track.last_head = Some((frame, head));
                }
            }
        }
        for (i, a) in row.tracked_players.iter().enumerate() {
            for b in row.tracked_players.iter().skip(i + 1) {
                let pair = (a.track_id.min(b.track_id), a.track_id.max(b.track_id));
                if !pairs.contains_key(&pair) && pairs.len() >= 200_000 {
                    return Err("Too many independently visible Track ID pairs".into());
                }
                let evidence = pairs.entry(pair).or_default();
                evidence.total += 1;
                let distinct_heads =
                    match (a.detection.head_point_norm, b.detection.head_point_norm) {
                        (Some(ha), Some(hb)) => (ha[0] - hb[0]).hypot(ha[1] - hb[1]) > 0.018,
                        _ => false,
                    };
                if !distinct_heads && complementary_parts(&a.detection, &b.detection) {
                    evidence.complementary += 1;
                }
            }
        }
    }
    if reader.line()?.is_some() || reader.finish() != expected_sha {
        return Err("Confirmed tracking source SHA-256 mismatch before Track ID stitching".into());
    }
    let mut eligible = Vec::<(u64, u64, u32, f64)>::new();
    for (&(first, second), evidence) in &pairs {
        if !(2..=8).contains(&evidence.total) || evidence.complementary != evidence.total {
            continue;
        }
        let (Some(a), Some(b)) = (tracks.get(&first), tracks.get(&second)) else {
            continue;
        };
        let (old_id, old, new_id, new) = match (a.start, b.start) {
            (Some(a_start), Some(b_start)) if a_start < b_start && a.end < b.end => {
                (first, a, second, b)
            }
            (Some(a_start), Some(b_start)) if b_start < a_start && b.end < a.end => {
                (second, b, first, a)
            }
            _ => continue,
        };
        let start = new.start.unwrap_or(0);
        if start > old.end
            || start.saturating_sub(old.start.unwrap_or(0)) < 12
            || new.end.saturating_sub(old.end) < 12
            || old.observed < 12
            || new.observed < 12
            || old.end.saturating_sub(start) > 8
        {
            continue;
        }
        let (Some((old_at, old_head)), Some((new_at, new_head))) = (old.last_head, new.first_head)
        else {
            continue;
        };
        if old_at > old.end
            || start.saturating_sub(old_at) > 45
            || new_at.saturating_sub(start) > 20
            || old_at > new_at
        {
            continue;
        }
        let displacement = (old_head[0] - new_head[0]).hypot(old_head[1] - new_head[1]);
        if displacement <= 0.018 {
            eligible.push((old_id, new_id, evidence.complementary, displacement));
        }
    }
    eligible.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.3.total_cmp(&b.3)));
    let mut aliases = BTreeMap::new();
    let mut used = BTreeSet::new();
    for (old_id, new_id, _, _) in eligible {
        if used.insert(old_id) {
            if used.insert(new_id) {
                aliases.insert(new_id, old_id);
            } else {
                used.remove(&old_id);
            }
        }
    }
    let mut gap_candidates = Vec::<(f64, u64, u64)>::new();
    for (&old_id, old) in &tracks {
        if used.contains(&old_id) || old.observed < 12 {
            continue;
        }
        for (&new_id, new) in &tracks {
            if old_id == new_id || used.contains(&new_id) || new.observed < 12 {
                continue;
            }
            let Some(start) = new.start else {
                continue;
            };
            if start <= old.end
                || start - old.end > 120
                || !scope.is_some_and(|mask| mask.same_interval(old.end, start))
            {
                continue;
            }
            if let (Some((old_at, old_pose)), Some((new_at, new_pose))) =
                (old.last_pose, new.first_pose)
            {
                if old.end - old_at <= 3 && new_at - start <= 3 {
                    if let Some((a, b)) = old_pose.torso().zip(new_pose.torso()) {
                        let distance = (a[0] - b[0]).hypot(a[1] - b[1]);
                        if distance <= (0.012 + (start - old.end) as f64 * 0.0006).min(0.035)
                            && old_pose
                                .similarity(&new_pose, new_at - old_at)
                                .is_some_and(|score| score >= 0.72)
                        {
                            gap_candidates.push((distance, old_id, new_id));
                        }
                    }
                    continue;
                }
            }
            let (Some((old_at, old_head)), Some((new_at, new_head)), Some(old_box), Some(new_box)) =
                (old.last_head, new.first_head, old.last_box, new.first_box)
            else {
                continue;
            };
            if start - old.end > 13
                || old.end - old_at > 3
                || new_at - start > 3
                || bbox_iou(old_box, new_box) < 0.25
            {
                continue;
            }
            let displacement = (old_head[0] - new_head[0]).hypot(old_head[1] - new_head[1]);
            if displacement <= 0.012 + (start - old.end) as f64 * 0.0006 {
                gap_candidates.push((displacement, old_id, new_id));
            }
        }
    }
    gap_candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    for &(distance, old_id, new_id) in &gap_candidates {
        if used.contains(&old_id) || used.contains(&new_id) {
            continue;
        }
        if gap_candidates
            .iter()
            .any(|&(other_distance, other_old, other_new)| {
                (other_old == old_id && other_new != new_id
                    || other_new == new_id && other_old != old_id)
                    && other_distance <= distance + 0.006
            })
        {
            continue;
        }
        used.insert(old_id);
        used.insert(new_id);
        aliases.insert(new_id, old_id);
    }
    Ok(aliases)
}

pub(super) fn stitch(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    scope: Option<&PlayerFrameScope>,
) -> Result<(String, u64, u64), String> {
    let aliases = find_handoffs(source, expected_sha, frames, scope)?;
    if aliases.is_empty() {
        return Ok((expected_sha.to_owned(), 0, 0));
    }
    let temp = source.with_extension("stitch.partial");
    let result = (|| -> Result<(String, u64), String> {
        let mut reader = BoundedReader::open(source)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| format!("Create Track ID stitch output: {e}"))?;
        let mut writer = BufWriter::new(file);
        let mut sha = Sha256::new();
        let mut removed = 0_u64;
        for frame_index in 0..frames {
            let line = reader
                .line()?
                .ok_or("Confirmed tracking truncated during Track ID rewrite")?;
            let mut row: Value = serde_json::from_slice(&line)
                .map_err(|e| format!("Invalid Track ID rewrite row: {e}"))?;
            if row["frame_index"].as_u64() != Some(frame_index) {
                return Err("Track ID stitching changed frame order".into());
            }
            let players = row["tracked_players"]
                .as_array_mut()
                .ok_or("Missing confirmed Track ID array")?;
            let mut canonical = BTreeMap::<u64, Value>::new();
            for mut player in std::mem::take(players) {
                let id = player["track_id"]
                    .as_u64()
                    .ok_or("Missing Track ID in stitching")?;
                let id = aliases.get(&id).copied().unwrap_or(id);
                player["track_id"] = json!(id);
                let quality = |player: &Value| {
                    let pose = u8::from(player["measurement_source"] == "rtmpose_rgb");
                    let score = player["detection"]["score"].as_f64().unwrap_or(0.0);
                    (pose, score)
                };
                match canonical.get_mut(&id) {
                    Some(previous) => {
                        removed += 1;
                        let bounds = serde_json::from_value::<[f64; 4]>(
                            previous["detection"]["bbox_xyxy_norm"].clone(),
                        )
                        .ok()
                        .zip(
                            serde_json::from_value::<[f64; 4]>(
                                player["detection"]["bbox_xyxy_norm"].clone(),
                            )
                            .ok(),
                        );
                        if quality(&player) > quality(previous) {
                            *previous = player;
                        }
                        if let Some((a, b)) = bounds {
                            previous["detection"]["bbox_xyxy_norm"] = json!([
                                a[0].min(b[0]),
                                a[1].min(b[1]),
                                a[2].max(b[2]),
                                a[3].max(b[3]),
                            ]);
                            previous["pose_crop_origin"] = json!("combined_track_box");
                        }
                    }
                    None => {
                        canonical.insert(id, player);
                    }
                }
            }
            *players = canonical.into_values().collect();
            if let Some(diagnostics) = row["new_track_diagnostics"].as_array_mut() {
                diagnostics.retain(|record| {
                    !record["track_id"]
                        .as_u64()
                        .is_some_and(|id| aliases.contains_key(&id))
                });
                for record in diagnostics {
                    if let Some(id) = record["closest_previous_track_id"].as_u64() {
                        if let Some(canonical) = aliases.get(&id) {
                            record["closest_previous_track_id"] = json!(canonical);
                        }
                    }
                }
            }
            let mut bytes = serde_json::to_vec(&row)
                .map_err(|e| format!("Serialize stitched Track IDs: {e}"))?;
            bytes.push(b'\n');
            if bytes.len() > MAX_ROW_BYTES {
                return Err("Track ID stitched row exceeds 512 KiB".into());
            }
            writer
                .write_all(&bytes)
                .map_err(|e| format!("Write stitched Track IDs: {e}"))?;
            sha.update(&bytes);
        }
        if reader.line()?.is_some() || reader.finish() != expected_sha {
            return Err("Confirmed Track ID source changed during stitching".into());
        }
        writer
            .flush()
            .map_err(|e| format!("Flush stitched Track IDs: {e}"))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|e| format!("Sync stitched Track IDs: {e}"))?;
        drop(writer);
        fs::rename(&temp, source).map_err(|e| format!("Publish stitched Track IDs: {e}"))?;
        Ok((format!("{:x}", sha.finalize()), removed))
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result.map(|(sha, removed)| (sha, aliases.len() as u64, removed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn rally_scope(selected: Vec<(u64, u64)>) -> PlayerFrameScope {
        PlayerFrameScope {
            selected,
            ball_job_id: "fixture".into(),
            mask_sha256: "a".repeat(64),
        }
    }

    fn candidate(frame: u64, bbox: [f64; 4], head: Option<[f64; 2]>) -> Value {
        json!({
            "frame_index":frame,
            "class_name":"player",
            "query_index":0,
            "score":0.70,
            "bbox_xyxy_norm":bbox,
            "head_point_norm":head
        })
    }

    fn fixture(
        different_heads: bool,
        long_overlap: bool,
        no_head: bool,
    ) -> (std::path::PathBuf, String, u64) {
        let path =
            std::env::temp_dir().join(format!("volley-track-stitch-{}.jsonl", Uuid::new_v4()));
        let frames = 65_u64;
        let mut raw = Vec::new();
        for frame in 0..frames {
            let mut players = Vec::new();
            let overlap_start = if long_overlap { 12 } else { 21 };
            let old_end = 24;
            if frame <= old_end {
                let (bbox, head) = if frame >= overlap_start {
                    ([0.5828536, 0.5042917, 0.5995204, 0.5638375], None)
                } else {
                    (
                        [0.576917, 0.444316, 0.592963, 0.560484],
                        (!no_head).then_some([0.579157, 0.453301]),
                    )
                };
                players.push(json!({
                    "track_id":3,
                    "measurement_source":if head.is_some() {"rtmpose_rgb"} else {"detector_bbox"},
                    "detection":candidate(frame,bbox,head)
                }));
            }
            if frame >= overlap_start {
                let head = if frame >= old_end + 2 && !no_head {
                    Some(if different_heads {
                        [0.585066, 0.529]
                    } else {
                        [0.585066, 0.451639]
                    })
                } else {
                    None
                };
                let bbox = if frame <= old_end {
                    [0.582925, 0.435986, 0.597286, 0.499780]
                } else {
                    [0.582951, 0.442672, 0.596922, 0.559037]
                };
                players.push(json!({
                    "track_id":7,
                    "measurement_source":if head.is_some() {"rtmpose_rgb"} else {"detector_bbox"},
                    "detection":candidate(frame,bbox,head)
                }));
            }
            players.push(json!({
                "track_id":15,
                "measurement_source":"rtmpose_rgb",
                "detection":candidate(frame,[0.21,0.45,0.28,0.69],Some([0.25,0.46]))
            }));
            let row = json!({
                "frame_index":frame,
                "tracked_players":players,
                "new_track_diagnostics": if frame == overlap_start {
                    vec![json!({"track_id":7,"reason":"new_track"})]
                } else {Vec::new()},
                "track_ids_provisional":true,
                "ball_candidates":[]
            });
            raw.extend(serde_json::to_vec(&row).unwrap());
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        (path, format!("{:x}", Sha256::digest(&raw)), frames)
    }

    #[test]
    fn stitches_id3_to_id7_before_reid_using_head_and_complementary_crops() {
        let (path, sha, count) = fixture(false, false, false);
        assert_eq!(
            find_handoffs(&path, &sha, count, None).unwrap().get(&7),
            Some(&3)
        );
        let (rewritten, merged, dropped) = stitch(&path, &sha, count, None).unwrap();
        assert_eq!(merged, 1);
        assert_eq!(dropped, 4);
        let contents = fs::read_to_string(&path).unwrap();
        assert_eq!(
            rewritten,
            format!("{:x}", Sha256::digest(contents.as_bytes()))
        );
        for (frame, line) in contents.lines().enumerate() {
            let row: Value = serde_json::from_str(line).unwrap();
            let ids: Vec<_> = row["tracked_players"]
                .as_array()
                .unwrap()
                .iter()
                .map(|player| player["track_id"].as_u64().unwrap())
                .collect();
            assert_eq!(ids, vec![3, 15], "frame {frame}");
            assert!(row["new_track_diagnostics"].as_array().unwrap().is_empty());
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn separate_heads_cannot_be_stitched() {
        let (path, sha, frames) = fixture(true, false, false);
        assert!(find_handoffs(&path, &sha, frames, None).unwrap().is_empty());
        assert_eq!(stitch(&path, &sha, frames, None).unwrap(), (sha, 0, 0));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn sustained_independent_overlap_cannot_be_stitched() {
        let (path, sha, frames) = fixture(false, true, false);
        assert!(find_handoffs(&path, &sha, frames, None).unwrap().is_empty());
        assert_eq!(stitch(&path, &sha, frames, None).unwrap(), (sha, 0, 0));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn no_independent_head_anchors_means_no_stitch() {
        let (path, sha, frames) = fixture(false, false, true);
        assert!(find_handoffs(&path, &sha, frames, None).unwrap().is_empty());
        assert_eq!(stitch(&path, &sha, frames, None).unwrap(), (sha, 0, 0));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn short_pose_gap_stitches_same_body_without_shared_observations() {
        let path = std::env::temp_dir().join(format!("volley-gap-stitch-{}.jsonl", Uuid::new_v4()));
        let mut raw = Vec::new();
        for frame in 0..48_u64 {
            let person = if frame <= 17 {
                Some(3)
            } else if (23..=47).contains(&frame) {
                Some(7)
            } else {
                None
            };
            let players = person
                .map(|id| {
                    vec![json!({
                        "track_id": id,
                        "measurement_source": "rtmpose_rgb",
                        "detection": candidate(frame, [0.50, 0.35, 0.58, 0.67], Some([0.54, 0.39]))
                    })]
                })
                .unwrap_or_default();
            raw.extend(
                serde_json::to_vec(&json!({
                    "frame_index":frame,
                    "tracked_players":players,
                    "new_track_diagnostics":[],
                    "ball_candidates":[]
                }))
                .unwrap(),
            );
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let separated = rally_scope(vec![(0, 17), (23, 47)]);
        assert_eq!(
            stitch(&path, &sha, 48, Some(&separated)).unwrap(),
            (sha.clone(), 0, 0)
        );
        let rally = rally_scope(vec![(0, 47)]);
        let (updated, merged, removed) = stitch(&path, &sha, 48, Some(&rally)).unwrap();
        assert_eq!((merged, removed), (1, 0));
        assert_ne!(updated, sha);
        for line in fs::read_to_string(&path).unwrap().lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            for player in row["tracked_players"].as_array().unwrap() {
                assert_eq!(player["track_id"], 3);
            }
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn measured_torso_stitches_different_crops_only_within_one_rally() {
        let path =
            std::env::temp_dir().join(format!("volley-torso-stitch-{}.jsonl", Uuid::new_v4()));
        let mut raw = Vec::new();
        let joints = [[0.54, 0.47, 0.9]; 17];
        for frame in 0..106 {
            let players = if frame <= 17 {
                vec![json!({"track_id":3,"pose":joints,
                    "detection":candidate(frame,[0.50,0.35,0.58,0.67],None)})]
            } else if frame >= 82 {
                vec![json!({"track_id":7,"pose":joints,
                    "detection":candidate(frame,[0.52,0.43,0.56,0.51],None)})]
            } else {
                vec![]
            };
            serde_json::to_writer(
                &mut raw,
                &json!({
                    "frame_index":frame,"tracked_players":players,
                }),
            )
            .unwrap();
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let separate = rally_scope(vec![(0, 17), (82, 105)]);
        assert!(find_handoffs(&path, &sha, 106, Some(&separate))
            .unwrap()
            .is_empty());
        let rally = rally_scope(vec![(0, 105)]);
        assert_eq!(
            find_handoffs(&path, &sha, 106, Some(&rally)).unwrap(),
            BTreeMap::from([(7, 3)])
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn ambiguous_short_gap_does_not_steal_player_identity() {
        let path =
            std::env::temp_dir().join(format!("volley-ambiguous-gap-{}.jsonl", Uuid::new_v4()));
        let mut raw = Vec::new();
        for frame in 0..48_u64 {
            let mut players = Vec::new();
            for (id, start, end, head) in [
                (3, 0, 17, [0.530, 0.39]),
                (4, 0, 17, [0.550, 0.39]),
                (7, 23, 47, [0.540, 0.39]),
            ] {
                if (start..=end).contains(&frame) {
                    players.push(json!({
                        "track_id":id,
                        "measurement_source":"rtmpose_rgb",
                        "detection":candidate(frame, [head[0]-0.035, 0.35, head[0]+0.035, 0.67], Some(head))
                    }));
                }
            }
            raw.extend(
                serde_json::to_vec(&json!({
                    "frame_index":frame,
                    "tracked_players":players,
                    "new_track_diagnostics":[],
                    "ball_candidates":[]
                }))
                .unwrap(),
            );
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let rally = rally_scope(vec![(0, 47)]);
        assert!(find_handoffs(&path, &sha, 48, Some(&rally))
            .unwrap()
            .is_empty());
        assert_eq!(stitch(&path, &sha, 48, Some(&rally)).unwrap(), (sha, 0, 0));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn hash_mismatch_never_modifies_source() {
        let (path, sha, frames) = fixture(false, false, false);
        let actual = fs::read(&path).unwrap();
        assert!(stitch(&path, &"0".repeat(64), frames, None).is_err());
        assert_eq!(actual, fs::read(&path).unwrap());
        fs::remove_file(path).unwrap();
        assert_eq!(sha.len(), 64);
    }
}
