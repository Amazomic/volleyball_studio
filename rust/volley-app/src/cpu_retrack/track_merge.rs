//! Merge only independently observed duplicate boxes of the same physical body.
//! Never merge by geometry alone: candidate pairs require several independent
//! OpenVINO ReID crops BEFORE any player-name assignment. Operates only on
//! an unpublished CPU replay; unavailable photo evidence fails closed.
use super::{BoundedReader, MAX_ROW_BYTES};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
};
use volley_core::fusion::Candidate;

#[derive(Deserialize)]
pub(super) struct Row {
    pub(super) frame_index: u64,
    pub(super) tracked_players: Vec<Observation>,
}
#[derive(Deserialize)]
pub(super) struct Observation {
    pub(super) track_id: u64,
    pub(super) detection: Candidate,
}
#[derive(Default)]
struct PairEvidence {
    simultaneous: u32,
    same_body: u32,
    nearby_body: u32,
}

fn box_overlap(a: [f64; 4], b: [f64; 4]) -> (f64, f64) {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let area = |b: [f64; 4]| (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let common = iw * ih;
    (
        common / (area(a) + area(b) - common).max(1e-9),
        common / area(a).min(area(b)).max(1e-9),
    )
}

/// Stronger evidence than matching photos: independent detections must
/// describe almost the same head AND torso, or an aligned nested crop.
fn same_body(a: &Candidate, b: &Candidate) -> bool {
    let (ab, bb) = (a.bbox_xyxy_norm, b.bbox_xyxy_norm);
    let (iou, contained) = box_overlap(ab, bb);
    let width = (ab[2] - ab[0]).min(bb[2] - bb[0]);
    let ha = ab[3] - ab[1];
    let hb = bb[3] - bb[1];
    if width < 0.012 || ha.min(hb) < 0.04 || iou < 0.28 {
        return false;
    }
    match (a.head_point_norm, b.head_point_norm) {
        (Some(a), Some(b)) => {
            (a[0] - b[0]).hypot(a[1] - b[1]) <= 0.012 && (iou >= 0.43 || contained >= 0.78)
        }
        _ => {
            // Without two noses only nearly concentric nested boxes count.
            // A person standing behind another is not merged by simple IoU.
            let ac = (ab[0] + ab[2]) * 0.5;
            let bc = (bb[0] + bb[2]) * 0.5;
            contained >= 0.86
                && (ac - bc).abs() < 0.16 * width
                && (ab[1] - bb[1]).abs() < 0.014
                && ha.max(hb) / ha.min(hb) <= 1.8
        }
    }
}

fn nearby_body(a: &Candidate, b: &Candidate) -> bool {
    let (Some(ha), Some(hb)) = (a.head_point_norm, b.head_point_norm) else {
        return false;
    };
    let (iou, contained) = box_overlap(a.bbox_xyxy_norm, b.bbox_xyxy_norm);
    iou >= 0.25
        && contained >= 0.70
        && (ha[0] - hb[0]).abs() <= 0.025
        && (ha[1] - hb[1]).abs() <= 0.020
}

struct Handoff {
    pair: (u64, u64),
    short: u64,
    long: u64,
    start: u64,
    end: u64,
    short_start: Option<[f64; 2]>,
    short_end: Option<[f64; 2]>,
    long_before: Option<[f64; 2]>,
    long_after: Option<[f64; 2]>,
}

fn verified_handoffs(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    mut candidates: Vec<Handoff>,
) -> Result<BTreeSet<(u64, u64)>, String> {
    if candidates.is_empty() || candidates.len() > 64 {
        return Ok(BTreeSet::new());
    }
    let mut reader = BoundedReader::open(source)?;
    for frame in 0..frames {
        let row = read_row(&mut reader, frame)?;
        for candidate in &mut candidates {
            if frame < candidate.start.saturating_sub(30)
                || frame > candidate.end.saturating_add(30)
            {
                continue;
            }
            let head = |id| {
                row.tracked_players
                    .iter()
                    .find(|player| player.track_id == id)
                    .and_then(|player| player.detection.head_point_norm)
            };
            if frame == candidate.start {
                candidate.short_start = head(candidate.short);
            }
            if frame == candidate.end {
                candidate.short_end = head(candidate.short);
            }
            if frame < candidate.start && candidate.start - frame <= 30 {
                if let Some(head) = head(candidate.long) {
                    candidate.long_before = Some(head);
                }
            }
            if frame > candidate.end
                && frame - candidate.end <= 30
                && candidate.long_after.is_none()
            {
                candidate.long_after = head(candidate.long);
            }
        }
    }
    if reader.line()?.is_some() || reader.finish() != expected_sha {
        return Err("Handoff source SHA-256 mismatch".into());
    }
    let close =
        |a: [f64; 2], b: [f64; 2], tolerance: f64| (a[0] - b[0]).hypot(a[1] - b[1]) <= tolerance;
    Ok(candidates
        .into_iter()
        .filter(|c| {
            matches!(
                (c.short_start, c.short_end, c.long_before, c.long_after),
                (Some(start), Some(end), Some(before), Some(after))
                    if close(start, before, 0.035) && close(end, after, 0.055)
            )
        })
        .map(|c| c.pair)
        .collect())
}

fn read_row(reader: &mut BoundedReader, frame: u64) -> Result<Row, String> {
    let bytes = reader.line()?.ok_or("Duplicate-track source truncated")?;
    let row: Row =
        serde_json::from_slice(&bytes).map_err(|e| format!("Bad duplicate-track row: {e}"))?;
    if row.frame_index != frame || row.tracked_players.len() > 128 {
        return Err("Duplicate-track input frames are incompatible".into());
    }
    Ok(row)
}

fn find_aliases(
    source: &Path,
    expected_sha: &str,
    frames: u64,
) -> Result<BTreeMap<u64, u64>, String> {
    let mut reader = BoundedReader::open(source)?;
    let mut totals = BTreeMap::<u64, u64>::new();
    let mut bounds = BTreeMap::<u64, (u64, u64)>::new();
    let mut pair_evidence = BTreeMap::<(u64, u64), PairEvidence>::new();
    for frame in 0..frames {
        let row = read_row(&mut reader, frame)?;
        let mut in_frame = BTreeSet::new();
        for player in &row.tracked_players {
            if player.track_id == 0 || !in_frame.insert(player.track_id) {
                return Err("Duplicate or invalid Track ID in source frame".into());
            }
            *totals.entry(player.track_id).or_default() += 1;
            bounds
                .entry(player.track_id)
                .and_modify(|(_, last)| *last = frame)
                .or_insert((frame, frame));
        }
        for (i, a) in row.tracked_players.iter().enumerate() {
            for b in row.tracked_players.iter().skip(i + 1) {
                let key = (a.track_id.min(b.track_id), a.track_id.max(b.track_id));
                let evidence = pair_evidence.entry(key).or_default();
                evidence.simultaneous += 1;
                evidence.same_body += u32::from(same_body(&a.detection, &b.detection));
                evidence.nearby_body += u32::from(nearby_body(&a.detection, &b.detection));
            }
        }
    }
    if reader.line()?.is_some() || reader.finish() != expected_sha {
        return Err("Duplicate-track source SHA-256 mismatch".into());
    }
    let transition_candidates = pair_evidence
        .iter()
        .filter_map(|(&pair, evidence)| {
            if !(2..=3).contains(&evidence.simultaneous)
                || evidence.nearby_body != evidence.simultaneous
            {
                return None;
            }
            let (short, long) = if totals[&pair.0] < totals[&pair.1] {
                (pair.0, pair.1)
            } else {
                (pair.1, pair.0)
            };
            let (start, end) = bounds[&short];
            let (before, after) = bounds[&long];
            (totals[&short] >= 6
                && totals[&short].saturating_mul(2) <= totals[&long]
                && end.saturating_sub(start) <= 120
                && before < start
                && after > end)
                .then_some(Handoff {
                    pair,
                    short,
                    long,
                    start,
                    end,
                    short_start: None,
                    short_end: None,
                    long_before: None,
                    long_after: None,
                })
        })
        .collect::<Vec<_>>();
    let handoffs = verified_handoffs(source, expected_sha, frames, transition_candidates)?;
    let mut ranked = pair_evidence
        .into_iter()
        .filter(|(pair, e)| {
            (e.same_body >= 3 && e.same_body as u64 * 5 >= e.simultaneous as u64 * 4)
                || handoffs.contains(pair)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.1.same_body
            .cmp(&a.1.same_body)
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut used = BTreeSet::new();
    let mut aliases = BTreeMap::new();
    for ((a, b), _) in ranked {
        // No transitive chains without independent three-way body proof.
        if !used.insert(a) {
            continue;
        }
        if !used.insert(b) {
            used.remove(&a);
            continue;
        }
        let keep = if totals[&a] >= totals[&b] { a } else { b };
        let drop = if keep == a { b } else { a };
        aliases.insert(drop, keep);
    }
    Ok(aliases)
}

/// Canonicalize new, unpublished replay data only; each output frame can
/// contain one and only one observation for a canonical Track ID.
pub(super) fn merge(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    photos: Option<&super::track_appearance::Source<'_>>,
) -> Result<(String, u64), String> {
    let proposed = find_aliases(source, expected_sha, frames)?;
    let aliases =
        super::track_appearance::approved_aliases(source, expected_sha, frames, &proposed, photos);
    write_aliases(source, expected_sha, frames, aliases)
}

fn write_aliases(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    aliases: BTreeMap<u64, u64>,
) -> Result<(String, u64), String> {
    if aliases.is_empty() {
        return Ok((expected_sha.to_owned(), 0));
    }
    let tmp = source.with_extension("merge.partial");
    let result = (|| -> Result<String, String> {
        let mut reader = BoundedReader::open(source)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| format!("Create duplicate-track output: {e}"))?;
        let mut writer = BufWriter::new(file);
        let mut digest = Sha256::new();
        for frame_index in 0..frames {
            let bytes = reader.line()?.ok_or("Duplicate-track rewrite truncated")?;
            let mut frame: Value = serde_json::from_slice(&bytes)
                .map_err(|e| format!("Decode duplicate-track row: {e}"))?;
            if frame["frame_index"].as_u64() != Some(frame_index) {
                return Err("Duplicate-track rewrite frames out of order".into());
            }
            let players = frame["tracked_players"]
                .as_array_mut()
                .ok_or("Missing Track ID rows during merge")?;
            let mut canonical = BTreeMap::<u64, Value>::new();
            for mut player in std::mem::take(players) {
                let id = player["track_id"]
                    .as_u64()
                    .ok_or("Missing Track ID in merge")?;
                let id = aliases.get(&id).copied().unwrap_or(id);
                player["track_id"] = json!(id);
                let score = player["detection"]["score"].as_f64().unwrap_or(0.0);
                let previous = canonical
                    .get(&id)
                    .and_then(|row| row["detection"]["score"].as_f64())
                    .unwrap_or(-1.0);
                if score > previous {
                    canonical.insert(id, player);
                }
            }
            *players = canonical.into_values().collect();
            let mut bytes =
                serde_json::to_vec(&frame).map_err(|e| format!("Encode merged tracks: {e}"))?;
            bytes.push(b'\n');
            if bytes.len() > MAX_ROW_BYTES {
                return Err("Merged Track ID row exceeds safety limit".into());
            }
            writer
                .write_all(&bytes)
                .map_err(|e| format!("Write merged tracks: {e}"))?;
            digest.update(&bytes);
        }
        if reader.line()?.is_some() || reader.finish() != expected_sha {
            return Err("Source Track ID checksum changed during merge".into());
        }
        writer
            .flush()
            .map_err(|e| format!("Flush merged tracks: {e}"))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|e| format!("Sync merged tracks: {e}"))?;
        drop(writer);
        fs::rename(&tmp, source).map_err(|e| format!("Publish merged tracks: {e}"))?;
        Ok(format!("{:x}", digest.finalize()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result.map(|sha| (sha, aliases.len() as u64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn candidate(frame: u64, box_xy: [f64; 4], head: [f64; 2]) -> Value {
        json!({"frame_index":frame,"class_name":"player","score":0.90,
            "bbox_xyxy_norm":box_xy,"query_index":0,"head_point_norm":head})
    }
    #[test]
    fn geometry_canonicalizes_same_head_after_independent_verification() {
        let dir = std::env::temp_dir().join(format!("track-merge-{}", Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        let path = dir.join("tracks.jsonl");
        let mut raw = Vec::new();
        for frame in 0..12_u64 {
            let player = candidate(frame, [0.20, 0.20, 0.30, 0.60], [0.25, 0.22]);
            let duplicate = candidate(frame, [0.21, 0.205, 0.305, 0.60], [0.251, 0.221]);
            let other = candidate(frame, [0.55, 0.20, 0.65, 0.60], [0.60, 0.22]);
            let record = json!({"frame_index":frame,"tracked_players":[
                {"track_id":1,"detection":player},
                {"track_id":2,"detection":duplicate},
                {"track_id":3,"detection":other}
            ],"raw_player_candidates":[]});
            raw.extend(serde_json::to_vec(&record).unwrap());
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let (unverified, rejected) = merge(&path, &sha, 12, None).unwrap();
        assert_eq!((unverified, rejected), (sha.clone(), 0));
        let aliases = find_aliases(&path, &sha, 12).unwrap();
        let (after, merged) = write_aliases(&path, &sha, 12, aliases).unwrap();
        assert_eq!(merged, 1);
        let rows = fs::read_to_string(&path).unwrap();
        for line in rows.lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            let ids: Vec<_> = row["tracked_players"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["track_id"].as_u64().unwrap())
                .collect();
            assert_eq!(ids, vec![1, 3]);
        }
        assert_eq!(after, format!("{:x}", Sha256::digest(rows.as_bytes())));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn joins_short_measured_pose_fragment_bracketed_by_same_head_track() {
        let dir = std::env::temp_dir().join(format!("track-handoff-{}", Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        let path = dir.join("tracks.jsonl");
        let mut raw = Vec::new();
        for frame in 0..76_u64 {
            let mut players = Vec::new();
            if frame <= 11 || frame >= 34 {
                let head = if frame <= 11 {
                    [0.52, 0.22]
                } else {
                    [0.54, 0.24]
                };
                players.push(json!({"track_id":6,"detection":candidate(
                    frame,[0.50,0.20,0.56,0.70],head
                )}));
            }
            if (10..=32).contains(&frame) {
                players.push(json!({"track_id":7,"detection":candidate(
                    frame,[0.51,0.21,0.56,0.70],[0.533,0.23]
                )}));
            }
            let row = json!({"frame_index":frame,"tracked_players":players});
            raw.extend(serde_json::to_vec(&row).unwrap());
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        assert_eq!(find_aliases(&path, &sha, 76).unwrap().get(&7), Some(&6));
        let aliases = find_aliases(&path, &sha, 76).unwrap();
        let (_, count) = write_aliases(&path, &sha, 76, aliases).unwrap();
        assert_eq!(count, 1);
        for row in fs::read_to_string(&path).unwrap().lines() {
            let doc: Value = serde_json::from_str(row).unwrap();
            assert!(doc["tracked_players"]
                .as_array()
                .unwrap()
                .iter()
                .all(|p| p["track_id"] == 6));
            assert!(doc["tracked_players"].as_array().unwrap().len() <= 1);
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn brief_overlapping_independent_players_are_not_merged() {
        let dir = std::env::temp_dir().join(format!("track-handoff-negative-{}", Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        let path = dir.join("tracks.jsonl");
        let mut raw = Vec::new();
        for frame in 0..44_u64 {
            let mut players = Vec::new();
            if frame <= 11 || frame >= 34 {
                players.push(json!({"track_id":6,"detection":candidate(
                    frame,[0.50,0.20,0.56,0.70],[0.52,0.22]
                )}));
            }
            if (10..=32).contains(&frame) {
                players.push(json!({"track_id":7,"detection":candidate(
                    frame,[0.51,0.21,0.56,0.70],
                    if frame >= 32 {[0.63,0.33]} else {[0.533,0.23]}
                )}));
            }
            let row = json!({"frame_index":frame,"tracked_players":players});
            raw.extend(serde_json::to_vec(&row).unwrap());
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        assert!(find_aliases(&path, &sha, 44).unwrap().is_empty());
        let (unchanged, count) = merge(&path, &sha, 44, None).unwrap();
        assert_eq!(count, 0);
        assert_eq!(unchanged, sha);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn does_not_merge_tracks_that_meet_only_in_interpolated_positions() {
        let mut a: Candidate =
            serde_json::from_value(candidate(0, [0.20, 0.20, 0.30, 0.60], [0.25, 0.22])).unwrap();
        let mut b = a.clone();
        b.head_point_norm = Some([0.35, 0.22]);
        assert!(!same_body(&a, &b));
        a.bbox_xyxy_norm = [0.12, 0.20, 0.22, 0.60];
        assert!(!same_body(&a, &b));
    }
}
