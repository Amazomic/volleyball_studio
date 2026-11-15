//! Pre-pose bounding-box preparation. No athlete IDs and no synthetic keypoints.
//! Saved immutable GPU detections remain unchanged. All frames are emitted in
//! source order, with a bounded two-sided interpolation for missing crops.
use super::{bbox, iou, MAX_LINE, MAX_POSES_PER_FRAME};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, VecDeque},
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

const MAX_GAP: u64 = 120; // 4 seconds, 30 fps
const MAX_PLAN_ROW: usize = 8192;

#[derive(Clone, Copy, Debug)]
struct Observation {
    bbox: [f64; 4],
    head: Option<[f64; 2]>,
}
#[derive(Clone, Copy)]
struct Anchor {
    frame: u64,
    observation: Observation,
}
struct Frame {
    index: u64,
    actual: Vec<[f64; 4]>,
    inferred: Vec<[f64; 4]>,
}
#[derive(Default, Debug)]
pub(super) struct Summary {
    pub frames: u64,
    pub actual: u64,
    pub inferred: u64,
    pub sha256: String,
}

fn observed(frame: &Value) -> Vec<Observation> {
    let mut found: Vec<Observation> = Vec::new();
    for candidate in frame["player_candidates"].as_array().into_iter().flatten() {
        if found.len() >= MAX_POSES_PER_FRAME {
            break;
        }
        let Some(rect) = bbox(candidate) else {
            continue;
        };
        if found.iter().any(|v| iou(v.bbox, rect) >= 0.60) {
            continue;
        }
        let head = candidate["head_point_norm"].as_array().and_then(|value| {
            let [x, y] = value.as_slice() else {
                return None;
            };
            let point = [x.as_f64()?, y.as_f64()?];
            point
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                .then_some(point)
        });
        found.push(Observation { bbox: rect, head });
    }
    found
}
fn center(rect: [f64; 4]) -> [f64; 2] {
    [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0]
}
fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn pairing_score(left: Anchor, frame: u64, right: Observation) -> Option<f64> {
    let gap = frame.checked_sub(left.frame)?;
    if gap == 0 || gap > MAX_GAP {
        return None;
    }
    let a = left.observation.bbox;
    let b = right.bbox;
    let ha = a[3] - a[1];
    let hb = b[3] - b[1];
    let wa = a[2] - a[0];
    let wb = b[2] - b[0];
    let ratio = (ha / hb).max(hb / ha).max(wa / wb).max(wb / wa);
    if ratio > 1.9 {
        return None;
    }
    let max_move = (0.035 + gap as f64 * 0.005).min(0.32);
    let displacement = distance(center(a), center(b));
    if displacement > max_move {
        return None;
    }
    let (head_bonus, head_distance) = match (left.observation.head, right.head) {
        (Some(a), Some(b)) => {
            let d = distance(a, b);
            let limit = (0.025 + gap as f64 * 0.005).min(0.25);
            if d > limit {
                return None;
            }
            (0.20 * (1.0 - d / limit), Some(d))
        }
        _ => (0.0, None),
    };
    // A match over a long disappearance must still have an independent
    // head anchor or a nearly stationary/re-overlapping geometry.
    if gap > 30 && head_distance.is_none() && iou(a, b) < 0.18 && displacement > 0.055 {
        return None;
    }
    let location = (1.0 - displacement / max_move).clamp(0.0, 1.0);
    let shape = (1.0 - (ratio - 1.0) / 0.9).clamp(0.0, 1.0);
    Some(0.47 * location + 0.20 * iou(a, b) + 0.13 * shape + head_bonus)
}
fn paired(active: &[Anchor], now: &[Observation], frame: u64) -> Vec<(usize, usize)> {
    let scores: Vec<Vec<Option<f64>>> = active
        .iter()
        .map(|a| now.iter().map(|b| pairing_score(*a, frame, *b)).collect())
        .collect();
    let mut pairs = Vec::new();
    for (ai, scores_for_anchor) in scores.iter().enumerate() {
        let mut ranked = scores_for_anchor
            .iter()
            .enumerate()
            .filter_map(|(j, v)| v.map(|s| (j, s)))
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let Some(&(bi, score)) = ranked.first() else {
            continue;
        };
        if score < 0.47 || ranked.get(1).is_some_and(|other| score - other.1 < 0.10) {
            continue;
        }
        let reverse = scores.iter().filter_map(|row| row[bi]).collect::<Vec<_>>();
        if reverse.iter().any(|&other| other > score + 1e-8)
            || reverse
                .iter()
                .filter(|&&other| other >= score - 0.10)
                .count()
                > 1
        {
            continue;
        }
        pairs.push((ai, bi));
    }
    pairs.sort_by_key(|v| v.0);
    pairs
}
fn interpolate(a: [f64; 4], b: [f64; 4], fraction: f64) -> [f64; 4] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * fraction)
}
fn safe_inferred(candidate: [f64; 4], frame: &Frame) -> bool {
    if candidate
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || candidate[0] >= candidate[2]
        || candidate[1] >= candidate[3]
    {
        return false;
    }
    let width = candidate[2] - candidate[0];
    let height = candidate[3] - candidate[1];
    width * 640.0 >= 8.0
        && height * 360.0 >= 20.0
        && frame
            .actual
            .iter()
            .chain(&frame.inferred)
            .all(|&other| iou(other, candidate) < 0.38)
        && frame.actual.len() + frame.inferred.len() < MAX_POSES_PER_FRAME
}
fn finish_frame(
    writer: &mut BufWriter<File>,
    hash: &mut Sha256,
    frame: Frame,
    stats: &mut Summary,
) -> Result<(), String> {
    let actual_count = frame.actual.len();
    let inferred_count = frame.inferred.len();
    let mut boxes = frame.actual;
    boxes.extend(frame.inferred);
    let mut bytes = serde_json::to_vec(&json!({
        "frame_index":frame.index,
        "boxes":boxes,
        "observed_box_count":actual_count,
        "interpolated_box_indices":(actual_count..actual_count+inferred_count).collect::<Vec<_>>()
    }))
    .map_err(|e| format!("Encode pose-box plan: {e}"))?;
    bytes.push(b'\n');
    if bytes.len() > MAX_PLAN_ROW {
        return Err("Prepared pose boxes exceed maximum row size".into());
    }
    writer
        .write_all(&bytes)
        .map_err(|e| format!("Write pose-box plan: {e}"))?;
    hash.update(&bytes);
    stats.frames += 1;
    stats.actual += actual_count as u64;
    stats.inferred += inferred_count as u64;
    Ok(())
}

/// Build a separate SHA-bound crop plan *before* running pose inference.
/// Uses only actual independent detector observations for matching, and only
/// fills gaps bracketed by two unambiguous observations of the same body.
pub(super) fn prepare(
    source: &Path,
    output: &Path,
    expected_source_sha256: &str,
    expected_frames: u64,
    cancel: &AtomicBool,
    mut progress: impl FnMut(f64) -> Result<(), String>,
) -> Result<Summary, String> {
    let reader =
        File::open(source).map_err(|e| format!("Open fusion for crop preparation: {e}"))?;
    let output_file = File::create(output).map_err(|e| format!("Create pose-box plan: {e}"))?;
    let mut writer = BufWriter::new(output_file);
    let mut source_hash = Sha256::new();
    let mut plan_hash = Sha256::new();
    let mut pending = VecDeque::<Frame>::new();
    let mut anchors = Vec::<Anchor>::new();
    let mut summary = Summary::default();
    let mut source_reader = BufReader::new(reader);
    let mut line = Vec::new();
    for frame_index in 0..expected_frames {
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        line.clear();
        let n = source_reader
            .read_until(b'\n', &mut line)
            .map_err(|e| format!("Read fusion for crop preparation: {e}"))?;
        if n == 0 || n as u64 > MAX_LINE || line.last() != Some(&b'\n') {
            return Err("Fusion crop-preparation stream truncated or malformed".into());
        }
        source_hash.update(&line);
        let frame: Value = serde_json::from_slice(&line)
            .map_err(|e| format!("Decode fusion for crop preparation: {e}"))?;
        if frame["frame_index"].as_u64() != Some(frame_index) {
            return Err("Unordered fusion frame in pose-box preparation".into());
        }
        let detected = observed(&frame);
        anchors.retain(|a| frame_index.saturating_sub(a.frame) <= MAX_GAP);
        let pairs = paired(&anchors, &detected, frame_index);
        // Reserve the actual detections before adding inferred shapes, so
        // nearby real people always take precedence.
        pending.push_back(Frame {
            index: frame_index,
            actual: detected.iter().map(|v| v.bbox).collect(),
            inferred: Vec::new(),
        });
        let mut occupied = BTreeSet::new();
        for (ai, bi) in pairs {
            let prev = anchors[ai];
            let next = detected[bi];
            if prev.frame + 1 < frame_index {
                for row in pending.iter_mut() {
                    if row.index <= prev.frame || row.index >= frame_index {
                        continue;
                    }
                    let fraction =
                        (row.index - prev.frame) as f64 / (frame_index - prev.frame) as f64;
                    let bb = interpolate(prev.observation.bbox, next.bbox, fraction);
                    if safe_inferred(bb, row) {
                        row.inferred.push(bb);
                    }
                }
            }
            anchors[ai] = Anchor {
                frame: frame_index,
                observation: next,
            };
            occupied.insert(bi);
        }
        for (bi, observation) in detected.into_iter().enumerate() {
            if !occupied.contains(&bi) {
                anchors.push(Anchor {
                    frame: frame_index,
                    observation,
                });
            }
        }
        while pending
            .front()
            .is_some_and(|row| frame_index.saturating_sub(row.index) >= MAX_GAP)
        {
            let oldest = pending.pop_front().expect("checked frame buffer");
            finish_frame(&mut writer, &mut plan_hash, oldest, &mut summary)?;
        }
        if frame_index % 300 == 0 {
            progress(frame_index as f64 / expected_frames.max(1) as f64)?;
        }
    }
    while let Some(row) = pending.pop_front() {
        finish_frame(&mut writer, &mut plan_hash, row, &mut summary)?;
    }
    if source_reader
        .read_until(b'\n', &mut line)
        .map_err(|e| format!("Verify fusion crop source end: {e}"))?
        != 0
    {
        return Err("Fusion has extra frames after crop preparation".into());
    }
    if format!("{:x}", source_hash.finalize()) != expected_source_sha256 {
        return Err("Fusion changed during pose-box preparation".into());
    }
    writer
        .flush()
        .map_err(|e| format!("Flush pose-box plan: {e}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| format!("Sync pose-box plan: {e}"))?;
    summary.sha256 = format!("{:x}", plan_hash.finalize());
    progress(1.0)?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use uuid::Uuid;
    fn candidate(frame: u64, x: f64, head: bool) -> Value {
        json!({"frame_index":frame,"class_name":"player","score":0.9,
        "bbox_xyxy_norm":[x,0.20,x+0.12,0.80],
        "head_point_norm":if head {json!([x+0.06,0.22])} else {Value::Null}})
    }
    fn run(frames: &[Vec<Value>]) -> Vec<Value> {
        let dir = std::env::temp_dir().join(format!("volley-pose-boxes-{}", Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        let source = dir.join("fusion.jsonl");
        let output = dir.join("crops.jsonl");
        let mut bytes = Vec::new();
        for (frame, candidates) in frames.iter().enumerate() {
            let mut row =
                serde_json::to_vec(&json!({"frame_index":frame,"player_candidates":candidates}))
                    .unwrap();
            row.push(b'\n');
            bytes.extend(row);
        }
        fs::write(&source, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let report = prepare(
            &source,
            &output,
            &sha,
            frames.len() as u64,
            &AtomicBool::new(false),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(report.frames, frames.len() as u64);
        let rows = fs::read_to_string(&output)
            .unwrap()
            .lines()
            .map(|row| serde_json::from_str::<Value>(row).unwrap())
            .collect();
        fs::remove_dir_all(dir).unwrap();
        rows
    }
    #[test]
    fn fills_only_two_sided_sparse_detection_gaps() {
        let mut samples = vec![Vec::new(); 15];
        samples[0].push(candidate(0, 0.2, true));
        samples[10].push(candidate(10, 0.25, true));
        let rows = run(&samples);
        for (index, row) in rows.iter().enumerate().take(10).skip(1) {
            assert_eq!(row["observed_box_count"], 0);
            assert_eq!(row["interpolated_box_indices"], json!([0]));
            let x = row["boxes"][0][0].as_f64().unwrap();
            assert!((x - (0.2 + index as f64 * 0.005)).abs() < 1e-8);
        }
        assert_eq!(rows[0]["observed_box_count"], 1);
        assert_eq!(rows[10]["observed_box_count"], 1);
        assert!(rows[11]["boxes"].as_array().unwrap().is_empty());
    }
    #[test]
    fn one_missing_athlete_is_filled_while_another_is_observed() {
        let mut frames = vec![Vec::new(); 13];
        frames[0] = vec![candidate(0, 0.2, true), candidate(0, 0.65, true)];
        frames[6] = vec![candidate(6, 0.65, true)];
        frames[12] = vec![candidate(12, 0.23, true), candidate(12, 0.65, true)];
        let rows = run(&frames);
        assert_eq!(rows[6]["observed_box_count"], 1);
        assert_eq!(rows[6]["interpolated_box_indices"], json!([1]));
        assert_eq!(rows[6]["boxes"].as_array().unwrap().len(), 2);
    }
    #[test]
    fn refuses_crossings_and_unbracketed_gaps() {
        let mut frames = vec![Vec::new(); 16];
        frames[0] = vec![candidate(0, 0.2, true), candidate(0, 0.5, true)];
        frames[10] = vec![candidate(10, 0.34, true), candidate(10, 0.36, true)];
        let rows = run(&frames);
        assert!(rows[5]["interpolated_box_indices"]
            .as_array()
            .unwrap()
            .is_empty());
        assert!(rows[11]["boxes"].as_array().unwrap().is_empty());
    }
    #[test]
    fn four_second_bridge_requires_both_real_anchors() {
        let mut frames = vec![Vec::new(); 123];
        frames[0].push(candidate(0, 0.20, true));
        frames[120].push(candidate(120, 0.22, true));
        let rows = run(&frames);
        assert_eq!(rows[119]["interpolated_box_indices"], json!([0]));
        assert_eq!(rows[120]["observed_box_count"], 1);
        assert!(rows[121]["boxes"].as_array().unwrap().is_empty());

        let mut too_long = vec![Vec::new(); 123];
        too_long[0].push(candidate(0, 0.20, true));
        too_long[121].push(candidate(121, 0.22, true));
        let rows = run(&too_long);
        assert!(rows[60]["boxes"].as_array().unwrap().is_empty());
    }

    #[test]
    fn ambiguous_neighbor_stays_unfilled_without_identity_stealing() {
        let mut frames = vec![Vec::new(); 16];
        frames[0] = vec![candidate(0, 0.20, false), candidate(0, 0.24, false)];
        frames[10] = vec![candidate(10, 0.22, false)];
        let rows = run(&frames);
        assert_eq!(rows[0]["observed_box_count"], 2);
        assert!(rows[5]["boxes"].as_array().unwrap().is_empty());
        assert_eq!(rows[10]["observed_box_count"], 1);
    }

    #[test]
    fn corrupt_source_hash_is_rejected() {
        let dir = std::env::temp_dir().join(format!("volley-pose-boxes-{}", Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        let source = dir.join("source");
        fs::write(&source, b"{\"frame_index\":0,\"player_candidates\":[]}\n").unwrap();
        let bad = prepare(
            &source,
            &dir.join("crops"),
            &"0".repeat(64),
            1,
            &AtomicBool::new(false),
            |_| Ok(()),
        );
        assert!(bad.is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
