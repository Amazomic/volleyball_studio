use super::BoundedReader;
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::SystemTime,
};
use volley_core::touch_pipeline::{flag_local_ball_artifacts, LeadingBallTracker};

pub(crate) const POLICY: &str = "vballnet_leading_ball_rally_v1";

pub(crate) fn saved_scope(manifest: &serde_json::Value) -> Result<Option<(Vec<u64>, f64)>, String> {
    if manifest["ball_tracking_policy"] != POLICY {
        return Ok(None);
    }
    let starts: Vec<u64> = serde_json::from_value(manifest["leading_ball_rally_starts"].clone())
        .map_err(|_| "Invalid saved leading-ball rally starts")?;
    let score = manifest["leading_ball_min_score"]
        .as_f64()
        .filter(|score| score.is_finite() && (0.0..=1.0).contains(score))
        .ok_or("Invalid saved leading-ball confidence")?;
    let frames = manifest["frames"]
        .as_u64()
        .ok_or("Missing leading-ball frame count")?;
    if starts.first() != Some(&0)
        || starts.iter().any(|start| *start >= frames)
        || starts.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("Invalid saved leading-ball reset order".into());
    }
    Ok(Some((starts, score)))
}

pub(crate) struct PreliminaryRallyScope {
    pub impacts: Vec<u64>,
    pub intervals: Vec<(u64, u64)>,
}

pub(crate) fn continuous_ball_intervals(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    min_score: f64,
    min_seconds: f64,
) -> Result<Vec<(u64, u64)>, String> {
    if frames == 0
        || !min_score.is_finite()
        || !(0.01..=1.0).contains(&min_score)
        || !min_seconds.is_finite()
        || !(0.1..=120.0).contains(&min_seconds)
    {
        return Err("Invalid continuous ball trajectory settings".into());
    }
    let minimum_frames = (min_seconds * 30.0).floor() as u64 + 1;
    let mut reader = BoundedReader::open(source)?;
    let mut intervals = Vec::new();
    let mut started = None::<u64>;
    let mut close = |end: u64, started: &mut Option<u64>| {
        if let Some(start) = started.take() {
            if end >= start && end - start + 1 >= minimum_frames {
                intervals.push((start, end));
            }
        }
    };
    for frame in 0..frames {
        let line = reader.line()?.ok_or("Truncated continuous ball source")?;
        let row: BallOnlyFrame =
            serde_json::from_slice(&line).map_err(|_| "Malformed continuous ball source")?;
        if row.frame_index != frame {
            return Err("Continuous ball source frames out of order".into());
        }
        let present = row.selected_ball.as_ref().is_some_and(|ball| {
            let bbox = &ball.detection.bbox_xyxy_norm;
            ball.source_model == "vballnet"
                && ball.detection.score.is_finite()
                && ball.detection.score >= min_score
                && bbox
                    .iter()
                    .all(|coord| coord.is_finite() && (0.0..=1.0).contains(coord))
                && bbox[0] < bbox[2]
                && bbox[1] < bbox[3]
        });
        if present {
            started.get_or_insert(frame);
        } else if frame > 0 {
            close(frame - 1, &mut started);
        }
    }
    close(frames - 1, &mut started);
    if reader.line()?.is_some() || reader.finish() != expected_sha {
        return Err("Continuous ball source SHA-256 mismatch".into());
    }
    if intervals.len() > 4096 {
        return Err("Too many continuous ball intervals".into());
    }
    Ok(intervals)
}

pub(crate) fn preliminary_rally_scope(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    dimensions: (u32, u32),
    min_score: f64,
    accepted: &[bool],
    settings: volley_core::rally_reconstruction::RallySettings,
) -> Result<PreliminaryRallyScope, String> {
    let (width, height) = dimensions;
    use volley_core::motion::{downward_to_upward_breaks, BallObservation};

    if settings.validate().is_err()
        || accepted.len() != frames as usize
        || !(64..=16384).contains(&width)
        || !(64..=16384).contains(&height)
        || !min_score.is_finite()
        || !(0.01..=1.0).contains(&min_score)
    {
        return Err("Invalid preliminary rally detection settings".into());
    }
    let mut reader = BoundedReader::open(source)?;
    let mut segment = Vec::<BallObservation>::with_capacity(640);
    let mut impacts = Vec::new();
    let mut emit = |observations: &[BallObservation]| {
        impacts.extend(
            downward_to_upward_breaks(observations)
                .into_iter()
                .filter_map(|item| u64::try_from(item.frame).ok()),
        );
    };
    for frame in 0..frames {
        let bytes = reader.line()?.ok_or("Truncated preliminary ball source")?;
        let row: BallOnlyFrame =
            serde_json::from_slice(&bytes).map_err(|_| "Malformed preliminary ball source")?;
        if row.frame_index != frame {
            return Err("Preliminary ball source frames out of order".into());
        }
        let point = row.selected_ball.and_then(|ball| {
            if ball.source_model != "vballnet"
                || ball.detection.score < min_score
                || !accepted[frame as usize]
            {
                return None;
            }
            let [x1, y1, x2, y2] = ball.detection.bbox_xyxy_norm;
            if ![x1, y1, x2, y2].iter().all(|v| v.is_finite()) {
                return None;
            }
            Some(BallObservation {
                frame: frame as i64,
                x: (x1 + x2) * 0.5 * width as f64,
                y: (y1 + y2) * 0.5 * height as f64,
            })
        });
        if point.is_none()
            && segment
                .last()
                .is_some_and(|last| frame > last.frame as u64 + 4)
        {
            emit(&segment);
            segment.clear();
        }
        if let Some(point) = point {
            if segment
                .last()
                .is_some_and(|last| point.frame - last.frame > 4)
            {
                emit(&segment);
                segment.clear();
            }
            segment.push(point);
            if segment.len() >= 640 {
                emit(&segment);
                let keep = segment.split_off(segment.len() - 20);
                segment = keep;
            }
        }
    }
    emit(&segment);
    if reader.line()?.is_some() || reader.finish() != expected_sha {
        return Err("Preliminary ball source SHA-256 mismatch".into());
    }
    impacts.sort_unstable();
    impacts.dedup_by(|next, earlier| *next - *earlier <= 6);
    let intervals =
        volley_core::rally_reconstruction::preliminary_rally_intervals(&impacts, frames, settings);
    Ok(PreliminaryRallyScope { impacts, intervals })
}

pub(crate) struct BallTrackMask {
    pub accepted: Vec<bool>,
    pub rejected: Vec<bool>,
    pub track_ids: Vec<u32>,
    pub candidate_counts: Vec<u8>,
}

#[derive(PartialEq)]
struct CacheKey {
    path: PathBuf,
    sha: String,
    frames: u64,
    width: u32,
    height: u32,
    min_score: u64,
    rally_starts: Vec<u64>,
    bytes: u64,
    modified: Option<SystemTime>,
    change: Option<(u64, u64, i64, i64)>,
}

type BallTrackCache = Vec<(CacheKey, Arc<BallTrackMask>)>;
static CACHE: OnceLock<Mutex<BallTrackCache>> = OnceLock::new();

#[derive(Deserialize)]
struct BallOnlyFrame {
    frame_index: u64,
    selected_ball: Option<BallOnlySelected>,
}
#[derive(Deserialize)]
struct BallOnlySelected {
    source_model: String,
    detection: BallOnlyDetection,
}
#[derive(Deserialize)]
struct BallOnlyDetection {
    score: f64,
    bbox_xyxy_norm: [f64; 4],
}

pub(crate) fn read(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    width: u32,
    height: u32,
    min_score: f64,
    rally_starts: &[u64],
) -> Result<Arc<BallTrackMask>, String> {
    if frames == 0
        || frames > 48 * 3600 * 30
        || width < 64
        || height < 64
        || !min_score.is_finite()
        || !(0.0..=1.0).contains(&min_score)
        || rally_starts.iter().any(|frame| *frame >= frames)
        || rally_starts.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("Invalid leading-ball source dimensions, threshold or rally starts".into());
    }
    let meta = fs::symlink_metadata(source).map_err(|e| format!("Ball source metadata: {e}"))?;
    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 64 * 1024 * 1024 * 1024 {
        return Err("Unsafe leading-ball replay source".into());
    }
    let key = CacheKey {
        path: source.to_owned(),
        sha: expected_sha.to_owned(),
        frames,
        width,
        height,
        min_score: min_score.to_bits(),
        rally_starts: rally_starts.to_vec(),
        bytes: meta.len(),
        modified: meta.modified().ok(),
        change: volley_media::preparation::file_change_stamp(source, &meta),
    };
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    if key.change.is_some() {
        if let Some((_, mask)) = cache
            .lock()
            .map_err(|_| "Ball mask cache poisoned")?
            .iter()
            .find(|(entry, _)| *entry == key)
        {
            return Ok(mask.clone());
        }
    }
    let mask = Arc::new(scan(
        source,
        expected_sha,
        frames,
        width,
        height,
        min_score,
        rally_starts,
    )?);
    let after = fs::symlink_metadata(source).map_err(|e| format!("Ball source changed: {e}"))?;
    if after.len() != key.bytes
        || after.modified().ok() != key.modified
        || volley_media::preparation::file_change_stamp(source, &after) != key.change
    {
        return Err("Replay changed during leading-ball scan".into());
    }
    if key.change.is_some() {
        let mut saved = cache.lock().map_err(|_| "Ball mask cache poisoned")?;
        saved.retain(|(entry, _)| {
            entry.path != key.path
                || entry.rally_starts != key.rally_starts
                || entry.min_score != key.min_score
        });
        while saved.len() >= 4
            || saved.iter().map(|(entry, _)| entry.frames).sum::<u64>() + frames > 8_000_000
        {
            if saved.is_empty() {
                break;
            }
            saved.remove(0);
        }
        saved.push((key, mask.clone()));
    }
    Ok(mask)
}

fn scan(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    width: u32,
    height: u32,
    min_score: f64,
    rally_starts: &[u64],
) -> Result<BallTrackMask, String> {
    const CHUNK: usize = 240;
    const HALO: usize = 16;
    let total = frames as usize;
    let mut mask = BallTrackMask {
        accepted: vec![false; total],
        rejected: vec![false; total],
        track_ids: vec![0; total],
        candidate_counts: vec![0; total],
    };
    let mut tracker = LeadingBallTracker::new(width, height);
    let mut next_rally = 0;
    let mut reader = BoundedReader::open(source)?;
    let mut buffer = std::collections::VecDeque::<Option<[f64; 2]>>::new();
    let (mut buffer_start, mut next_read, mut core_start) = (0, 0, 0);
    while core_start < total {
        let core_end = (core_start + CHUNK).min(total);
        let region_start = core_start.saturating_sub(HALO);
        let region_end = (core_end + HALO).min(total);
        while next_read < region_end {
            let bytes = reader.line()?.ok_or("Truncated ball source pre-scan")?;
            let row: BallOnlyFrame =
                serde_json::from_slice(&bytes).map_err(|_| "Malformed ball pre-scan frame")?;
            if row.frame_index != next_read as u64 {
                return Err("Ball source frames out of order".into());
            }
            let point = row.selected_ball.and_then(|entry| {
                let b = entry.detection.bbox_xyxy_norm;
                (entry.source_model == "vballnet"
                    && entry.detection.score.is_finite()
                    && entry.detection.score >= min_score
                    && entry.detection.score <= 1.0
                    && b.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                    && b[2] > b[0]
                    && b[3] > b[1])
                    .then_some([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5])
            });
            buffer.push_back(point);
            next_read += 1;
        }
        let points = buffer
            .iter()
            .skip(region_start - buffer_start)
            .take(region_end - region_start)
            .copied()
            .collect::<Vec<_>>();
        let local_flags = flag_local_ball_artifacts(&points, width, height);
        for index in core_start..core_end {
            if rally_starts.get(next_rally) == Some(&(index as u64)) {
                if index > 0 {
                    tracker.begin_rally();
                }
                next_rally += 1;
            }
            let point = points[index - region_start];
            mask.candidate_counts[index] = u8::from(point.is_some());
            let observed = point.filter(|_| !local_flags[index - region_start]);
            let leading = tracker.observe(index as u64, observed.as_slice());
            mask.rejected[index] = point.is_some()
                && (local_flags[index - region_start]
                    || !rally_starts.is_empty() && leading.selected.is_none());
            mask.accepted[index] =
                observed.is_some() && (rally_starts.is_empty() || leading.selected.is_some());
            mask.track_ids[index] = leading.track_id.unwrap_or(0);
            if !rally_starts.is_empty() {
                if let Some(anchors) = leading.acquired_frames {
                    for frame in anchors[0] as usize..=index {
                        mask.track_ids[frame] = leading.track_id.unwrap_or(0);
                    }
                    for anchor in anchors {
                        mask.rejected[anchor as usize] = false;
                        mask.accepted[anchor as usize] = true;
                        mask.track_ids[anchor as usize] = leading.track_id.unwrap_or(0);
                    }
                }
            }
        }
        core_start = core_end;
        let keep_from = core_start.saturating_sub(HALO);
        while buffer_start < keep_from {
            buffer.pop_front();
            buffer_start += 1;
        }
    }
    if reader.line()?.is_some() || reader.finish() != expected_sha {
        return Err("Ball source pre-scan SHA-256 mismatch".into());
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sha2::{Digest, Sha256};

    #[test]
    fn continuous_ball_requires_more_than_threshold_without_dropped_frames() {
        let root = std::env::temp_dir().join(format!("continuous-ball-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let path = root.join("tracked.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..310_u64 {
            let present = frame < 90
                || (100..191).contains(&frame)
                || ((200..310).contains(&frame) && frame != 260);
            let score = if frame == 220 { 0.2 } else { 0.9 };
            bytes.extend(
                serde_json::to_vec(&json!({
                    "frame_index":frame,
                    "selected_ball":if present {Some(json!({
                        "source_model":"vballnet",
                        "detection":{"score":score,"bbox_xyxy_norm":[0.4,0.4,0.5,0.5]}
                    }))} else {None}
                }))
                .unwrap(),
            );
            bytes.push(b'\n');
        }
        fs::write(&path, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(
            continuous_ball_intervals(&path, &sha, 310, 0.35, 3.0).unwrap(),
            vec![(100, 190)]
        );
        assert_eq!(
            continuous_ball_intervals(&path, &sha, 310, 0.35, 1.0).unwrap(),
            vec![(0, 89), (100, 190), (221, 259), (261, 309)]
        );
        assert!(continuous_ball_intervals(&path, &"0".repeat(64), 310, 0.35, 1.0).is_err());
        assert!(continuous_ball_intervals(&path, &sha, 310, 0.35, f64::NAN).is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn preliminary_ball_only_reversals_define_gameplay_scope_without_player_tracks() {
        let root =
            std::env::temp_dir().join(format!("preliminary-rallies-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let path = root.join("tracked.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..210_u64 {
            let y = if frame <= 70 {
                100.0 + 7.0 * frame as f64
            } else if frame <= 100 {
                590.0 - 7.0 * (frame - 70) as f64
            } else if frame <= 130 {
                380.0 + 7.0 * (frame - 100) as f64
            } else {
                590.0 - 7.0 * (frame - 130) as f64
            };
            bytes.extend(
                serde_json::to_vec(&json!({
                    "frame_index":frame,
                    "selected_ball":{
                        "source_model":"vballnet",
                        "detection":{
                            "score":0.9,
                            "bbox_xyxy_norm":[0.49,(y-4.0)/720.0,0.51,(y+4.0)/720.0]
                        }
                    }
                }))
                .unwrap(),
            );
            bytes.push(b'\n');
        }
        fs::write(&path, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let scope = preliminary_rally_scope(
            &path,
            &sha,
            210,
            (1280, 720),
            0.35,
            &vec![true; 210],
            volley_core::rally_reconstruction::RallySettings::default(),
        )
        .unwrap();
        assert_eq!(scope.impacts.len(), 2, "{:?}", scope.impacts);
        assert!((scope.impacts[0] as i64 - 70).abs() <= 2);
        assert!((scope.impacts[1] as i64 - 130).abs() <= 2);
        assert!(scope.intervals[0].0 <= 25 && scope.intervals[0].1 >= 175);
        let missing = preliminary_rally_scope(
            &path,
            &sha,
            210,
            (1280, 720),
            0.35,
            &(0..210).map(|i| i < 90).collect::<Vec<_>>(),
            volley_core::rally_reconstruction::RallySettings::default(),
        )
        .unwrap();
        assert!(missing.intervals.is_empty());
        assert!(preliminary_rally_scope(
            &path,
            &"0".repeat(64),
            210,
            (1280, 720),
            0.35,
            &vec![true; 210],
            volley_core::rally_reconstruction::RallySettings::default(),
        )
        .is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn saved_tracking_scope_rejects_missing_threshold_or_invalid_resets() {
        let mut manifest = json!({"ball_tracking_policy":POLICY,"frames":300,
            "leading_ball_rally_starts":[0,260],"leading_ball_min_score":0.67});
        assert_eq!(saved_scope(&manifest).unwrap(), Some((vec![0, 260], 0.67)));
        manifest["leading_ball_min_score"] = serde_json::Value::Null;
        assert!(saved_scope(&manifest).is_err());
        manifest["leading_ball_min_score"] = json!(0.67);
        manifest["leading_ball_rally_starts"] = json!([0, 260, 200]);
        assert!(saved_scope(&manifest).is_err());
    }

    #[test]
    fn chunk_seam_keeps_leader_until_explicit_rally_and_cache_keeps_scope() {
        let root = std::env::temp_dir().join(format!("ball-mask-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let path = root.join("frames.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..300 {
            let x = if frame < 230 {
                0.2 + frame as f64 * 0.0007
            } else {
                0.85 + (frame - 230) as f64 * 0.001
            };
            bytes.extend(serde_json::to_vec(&json!({"frame_index":frame,"selected_ball":{
                "source_model":"vballnet","detection":{"score":0.9,"bbox_xyxy_norm":[x,0.4,x+0.005,0.405]}}})).unwrap());
            bytes.push(b'\n');
        }
        fs::write(&path, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let same_rally = read(&path, &sha, 300, 1280, 720, 0.35, &[0]).unwrap();
        assert!(same_rally.rejected[230..].iter().all(|flag| *flag));
        let next_rally = read(&path, &sha, 300, 1280, 720, 0.35, &[0, 260]).unwrap();
        assert!(next_rally.rejected[230..260].iter().all(|flag| *flag));
        assert!(!next_rally.rejected[260]);
        assert_eq!(next_rally.track_ids[260], 2);
        assert_eq!(next_rally.candidate_counts[260], 1);
        assert!(read(&path, &"0".repeat(64), 300, 1280, 720, 0.35, &[0]).is_err());
        assert!(read(&path, &sha, 300, 1280, 720, 0.35, &[0, 0]).is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "requires an isolated VOLLEY_LEADING_BALL_FIXTURE replay"]
    fn recorded_13_9_seconds_has_only_the_leading_ball() {
        let root = PathBuf::from(std::env::var("VOLLEY_LEADING_BALL_FIXTURE").unwrap());
        let path = root.join("tracked_frames.jsonl");
        let bytes = fs::read(&path).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["sha256"], sha);
        let frames = manifest["frames"].as_u64().unwrap();
        let mask = read(&path, &sha, frames, 1280, 720, 0.35, &[0, 130, 901]).unwrap();
        assert!(mask.rejected[388..400].iter().all(|flag| *flag));
        assert!(!mask.rejected[387] && !mask.rejected[400] && !mask.rejected[417]);
        assert_eq!(mask.track_ids[387], mask.track_ids[417]);
        eprintln!(
            "recorded replay: candidates={}, rejected={}",
            mask.candidate_counts
                .iter()
                .map(|n| u64::from(*n))
                .sum::<u64>(),
            mask.rejected.iter().filter(|flag| **flag).count()
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }
}
