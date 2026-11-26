use super::BoundedReader;
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::SystemTime,
};
use volley_core::touch_pipeline::{flag_local_ball_artifacts, LeadingBallTracker};

pub(crate) const POLICY: &str = "vballnet_leading_ball_rally_v1";
pub(crate) const LOCAL_POLICY: &str = "vballnet_local_artifacts_v1";
pub(crate) const CONTINUOUS_MIN_SECONDS: f64 = 1.5;
pub(crate) const CONTINUOUS_MERGE_SECONDS: f64 = 4.0;
pub(crate) const MIN_RALLY_FRAMES: u64 = 5 * 30;
pub(crate) const MOVEMENT_FILTER_VERSION: &str = "stationary_ball_min_rally_5s_v4";

pub(crate) fn movement_filter_supported(version: Option<&str>) -> bool {
    matches!(
        version,
        Some("stationary_ball_min_rally_5s_v3" | MOVEMENT_FILTER_VERSION)
    )
}

pub(crate) fn filter_short_rallies(intervals: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    intervals
        .into_iter()
        .filter(|&(start, end)| end - start + 1 >= MIN_RALLY_FRAMES)
        .collect()
}

pub(crate) fn merge_continuous_intervals(
    intervals: &[(u64, u64)],
    gap_seconds: f64,
) -> Result<Vec<(u64, u64)>, String> {
    if !gap_seconds.is_finite() || !(0.0..=60.0).contains(&gap_seconds) {
        return Err("Invalid continuous rally merge gap".into());
    }
    let gap_frames = gap_seconds * 30.0;
    let mut sorted = intervals.to_vec();
    sorted.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, end) in sorted {
        if start > end {
            return Err("Invalid continuous rally interval".into());
        }
        if let Some(last) = merged.last_mut() {
            if start <= last.1.saturating_add(1) || ((start - last.1 - 1) as f64) < gap_frames {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    Ok(merged)
}

pub(crate) const DEFAULT_STATIONARY_BALL_MS: u32 = 1000;
const STATIONARY_REGION_SIZE: f64 = 0.035;
const MIN_RALLY_FIELD_TRAVEL: f64 = 0.10;
const MIN_UNOBSERVED_STOP_MS: u64 = 3000;

#[derive(Debug)]
pub(crate) struct MovingBallIntervals {
    pub detected: Vec<(u64, u64)>,
    pub merged: Vec<(u64, u64)>,
    pub stationary: Vec<(u64, u64)>,
    pub unobserved: Vec<(u64, u64)>,
}

pub(crate) fn stationary_ball_intervals(
    points: &[Option<[f64; 2]>],
    stationary_ball_ms: u32,
) -> Result<Vec<(u64, u64)>, String> {
    if !(200..=10_000).contains(&stationary_ball_ms) {
        return Err("Время неподвижности мяча должно быть от 200 до 10000 мс".into());
    }
    let window = (u64::from(stationary_ball_ms) * 30).div_ceil(1000) as usize;
    let mut x_min = std::collections::VecDeque::<(usize, f64)>::new();
    let mut x_max = std::collections::VecDeque::<(usize, f64)>::new();
    let mut y_min = std::collections::VecDeque::<(usize, f64)>::new();
    let mut y_max = std::collections::VecDeque::<(usize, f64)>::new();
    let mut observations = std::collections::VecDeque::<usize>::new();
    let mut regions = Vec::<(u64, u64)>::new();
    for index in 0..points.len() {
        let point = points[index].filter(|p| p.iter().all(|v| v.is_finite()));
        let isolated_jump = index > 0
            && index + 1 < points.len()
            && point.is_some_and(|p| {
                points[index - 1]
                    .zip(points[index + 1])
                    .is_some_and(|(before, after)| {
                        (p[0] - before[0]).hypot(p[1] - before[1]) > 0.08
                            && (after[0] - before[0]).hypot(after[1] - before[1]) < 0.02
                    })
            });
        if let Some([x, y]) = point.filter(|_| !isolated_jump) {
            observations.push_back(index);
            while x_min.back().is_some_and(|entry| entry.1 >= x) {
                x_min.pop_back();
            }
            while x_max.back().is_some_and(|entry| entry.1 <= x) {
                x_max.pop_back();
            }
            while y_min.back().is_some_and(|entry| entry.1 >= y) {
                y_min.pop_back();
            }
            while y_max.back().is_some_and(|entry| entry.1 <= y) {
                y_max.pop_back();
            }
            x_min.push_back((index, x));
            x_max.push_back((index, x));
            y_min.push_back((index, y));
            y_max.push_back((index, y));
        }
        let start = (index + 1).saturating_sub(window);
        for deque in [&mut x_min, &mut x_max, &mut y_min, &mut y_max] {
            while deque.front().is_some_and(|entry| entry.0 < start) {
                deque.pop_front();
            }
        }
        while observations.front().is_some_and(|frame| *frame < start) {
            observations.pop_front();
        }
        if index + 1 < window
            || observations.len() * 100 < window * 70
            || observations.front() != Some(&start)
            || observations.back() != Some(&index)
        {
            continue;
        }
        let stable = x_min
            .front()
            .zip(x_max.front())
            .zip(y_min.front().zip(y_max.front()))
            .is_some_and(|((left, right), (top, bottom))| {
                right.1 - left.1 <= STATIONARY_REGION_SIZE
                    && bottom.1 - top.1 <= STATIONARY_REGION_SIZE
            });
        if stable {
            let interval = (start as u64, index as u64);
            if let Some(last) = regions.last_mut() {
                if interval.0 <= last.1.saturating_add(1) {
                    last.1 = interval.1;
                    continue;
                }
            }
            regions.push(interval);
        }
    }
    Ok(regions)
}

fn prolonged_unobserved_ball_intervals(
    points: &[Option<[f64; 2]>],
    stationary_ball_ms: u32,
) -> Vec<(u64, u64)> {
    let minimum =
        (u64::from(stationary_ball_ms).max(MIN_UNOBSERVED_STOP_MS) * 30).div_ceil(1000) as usize;
    let mut ranges = Vec::new();
    let mut first = None;
    for (frame, point) in points
        .iter()
        .chain(std::iter::once(&Some([0.0, 0.0])))
        .enumerate()
    {
        if point.is_none() {
            first.get_or_insert(frame);
        } else if let Some(start) = first.take() {
            if frame - start >= minimum {
                ranges.push((start as u64, frame as u64 - 1));
            }
        }
    }
    ranges
}

fn is_field_motion(points: &[Option<[f64; 2]>], start: u64, end: u64) -> bool {
    let (mut low_x, mut high_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut low_y, mut high_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in points[start as usize..=end as usize].iter().flatten() {
        low_x = low_x.min(p[0]);
        high_x = high_x.max(p[0]);
        low_y = low_y.min(p[1]);
        high_y = high_y.max(p[1]);
    }
    (high_x - low_x).max(high_y - low_y) >= MIN_RALLY_FIELD_TRAVEL
}

fn extend_high_ball_returns(
    points: &[Option<[f64; 2]>],
    intervals: &mut [(u64, u64)],
    barriers: &[(u64, u64)],
    merge_gap_seconds: f64,
) {
    let max_gap = (merge_gap_seconds * 30.0).ceil().min(60.0) as usize;
    for index in 0..intervals.len() {
        let end = intervals[index].1 as usize;
        let limit = intervals
            .get(index + 1)
            .map_or(points.len(), |&(start, _)| start as usize);
        let Some(exit) = (end..(end + 9).min(limit)).find(|&frame| {
            points[frame].is_some() && points.get(frame + 1).is_some_and(Option::is_none)
        }) else {
            continue;
        };
        let Some([exit_x, exit_y]) = points[exit] else {
            continue;
        };
        let approach = points[exit.saturating_sub(7)..=exit]
            .iter()
            .flatten()
            .collect::<Vec<_>>();
        if exit_y > 0.05 || approach.len() < 4 || approach[0][1] - exit_y < 0.05 {
            continue;
        }
        let Some(entry) =
            (exit + 1..(exit + max_gap + 1).min(limit)).find(|&frame| points[frame].is_some())
        else {
            continue;
        };
        let Some([entry_x, entry_y]) = points[entry] else {
            continue;
        };
        if entry_y > 0.10
            || (entry_x - exit_x).abs() > 0.12
            || barriers
                .iter()
                .any(|&(start, stop)| start <= entry as u64 && stop > end as u64)
        {
            continue;
        }
        let mut last = entry;
        let mut descending = Vec::new();
        for (frame, point) in points.iter().enumerate().take(limit).skip(entry) {
            if frame - last > 4 {
                break;
            }
            if let Some(point) = point {
                last = frame;
                if frame < entry + 8 {
                    descending.push(point);
                }
            }
        }
        if descending.len() >= 4
            && descending
                .last()
                .is_some_and(|point| point[1] - entry_y >= 0.05)
        {
            intervals[index].1 = last as u64;
        }
    }
}

pub(crate) fn analyze_moving_ball(
    points: &[Option<[f64; 2]>],
    min_seconds: f64,
    merge_gap_seconds: f64,
    stationary_ball_ms: u32,
) -> Result<MovingBallIntervals, String> {
    let stationary = stationary_ball_intervals(points, stationary_ball_ms)?;
    let unobserved = prolonged_unobserved_ball_intervals(points, stationary_ball_ms);
    let mut moving = points.to_vec();
    for &(start, end) in &stationary {
        moving[start as usize..=end as usize].fill(None);
    }
    let minimum = (min_seconds * 30.0).ceil() as u64;
    let mut detected = Vec::new();
    for (start, end) in continuous_intervals_from_positions(&moving, min_seconds)? {
        let mut next_start = start;
        for &(idle_start, idle_end) in &stationary {
            if idle_end < next_start {
                continue;
            }
            if idle_start > end {
                break;
            }
            if idle_start > next_start {
                let final_frame = idle_start - 1;
                if final_frame - next_start + 1 >= minimum
                    && is_field_motion(&moving, next_start, final_frame)
                {
                    detected.push((next_start, final_frame));
                }
            }
            next_start = idle_end.saturating_add(1);
        }
        if next_start <= end
            && end - next_start + 1 >= minimum
            && is_field_motion(&moving, next_start, end)
        {
            detected.push((next_start, end));
        }
    }
    let mut barriers = stationary.clone();
    barriers.extend(unobserved.iter().copied());
    let mut merged =
        merge_continuous_intervals_with_pauses(&detected, merge_gap_seconds, &barriers)?;
    extend_high_ball_returns(&moving, &mut merged, &barriers, merge_gap_seconds);
    let merged = filter_short_rallies(merged);
    Ok(MovingBallIntervals {
        detected,
        merged,
        stationary,
        unobserved,
    })
}

pub(crate) fn merge_continuous_intervals_with_pauses(
    intervals: &[(u64, u64)],
    gap_seconds: f64,
    pauses: &[(u64, u64)],
) -> Result<Vec<(u64, u64)>, String> {
    let mut merged = merge_continuous_intervals(&[], gap_seconds)?;
    for &(start, end) in intervals {
        if start > end {
            return Err("Invalid continuous rally interval".into());
        }
        if let Some(last) = merged.last_mut() {
            let barrier = pauses.iter().any(|&(pause_start, pause_end)| {
                pause_start <= start
                    && pause_end >= last.1
                    && pause_start <= start.saturating_sub(1)
                    && pause_end >= last.1.saturating_add(1)
            });
            let gap_frames = gap_seconds * 30.0;
            if !barrier
                && (start <= last.1.saturating_add(1) || ((start - last.1 - 1) as f64) < gap_frames)
            {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    Ok(merged)
}

pub(crate) fn saved_scope(manifest: &serde_json::Value) -> Result<Option<(Vec<u64>, f64)>, String> {
    if manifest["ball_tracking_policy"] == LOCAL_POLICY {
        let starts: Vec<u64> =
            serde_json::from_value(manifest["leading_ball_rally_starts"].clone())
                .map_err(|_| "Invalid local ball policy marker")?;
        let score = manifest["leading_ball_min_score"]
            .as_f64()
            .filter(|score| score.is_finite() && (0.0..=1.0).contains(score))
            .ok_or("Invalid local ball confidence threshold")?;
        if !starts.is_empty() {
            return Err("Local ball policy cannot reset a leading tracker".into());
        }
        return Ok(Some((Vec::new(), score)));
    }
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
    #[cfg_attr(not(test), expect(dead_code))]
    pub intervals: Vec<(u64, u64)>,
}

fn median_ball_position(points: &[Option<[f64; 2]>], start: usize) -> Option<[f64; 2]> {
    let mut xs = [0.0; 3];
    let mut ys = [0.0; 3];
    let mut count = 0;
    for [x, y] in points.get(start..start + 3)?.iter().flatten() {
        xs[count] = *x;
        ys[count] = *y;
        count += 1;
    }
    if count == 0 {
        return None;
    }
    xs[..count].sort_by(f64::total_cmp);
    ys[..count].sort_by(f64::total_cmp);
    Some([xs[count / 2], ys[count / 2]])
}

pub(crate) fn continuous_intervals_from_positions(
    points: &[Option<[f64; 2]>],
    min_seconds: f64,
) -> Result<Vec<(u64, u64)>, String> {
    if !min_seconds.is_finite() || !(0.1..=120.0).contains(&min_seconds) {
        return Err("Invalid continuous ball trajectory duration".into());
    }
    let minimum_frames = (min_seconds * 30.0).ceil() as usize;
    let mut intervals = Vec::new();
    let mut started = None::<usize>;
    let mut last = 0usize;
    let mut evidence = 0usize;
    let mut close = |start: usize, end: usize, support: usize| {
        let start = start.saturating_sub(9);
        let end = end.saturating_add(9).min(points.len().saturating_sub(1));
        let slice = &points[start..=end];
        let (Some(first), Some(last)) = (
            slice.iter().position(Option::is_some),
            slice.iter().rposition(Option::is_some),
        ) else {
            return;
        };
        let start = start + first;
        let end = start + last - first;
        let duration = end - start + 1;
        let observed = slice[first..=last]
            .iter()
            .filter(|item| item.is_some())
            .count();
        if duration >= minimum_frames
            && support * 100 >= duration * 35
            && observed * 100 >= duration * 35
        {
            intervals.push((start as u64, end as u64));
        }
    };
    for frame in 9..points.len().saturating_sub(9) {
        let Some(before) = median_ball_position(points, frame - 9) else {
            continue;
        };
        let Some(after) = median_ball_position(points, frame + 7) else {
            continue;
        };
        let displacement = (after[0] - before[0]).hypot(after[1] - before[1]);
        if !(0.025..=0.3).contains(&displacement) {
            continue;
        }
        match started {
            None => {
                started = Some(frame);
                evidence = 1;
            }
            Some(start) if frame - last > 25 => {
                close(start, last, evidence);
                started = Some(frame);
                evidence = 1;
            }
            Some(_) => {
                evidence += 1;
            }
        }
        last = frame;
    }
    if let Some(start) = started {
        close(start, last, evidence);
    }
    if intervals.len() > 4096 {
        return Err("Too many continuous ball intervals".into());
    }
    Ok(intervals)
}

#[cfg(test)]
pub(crate) fn continuous_ball_intervals(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    min_score: f64,
    min_seconds: f64,
) -> Result<Vec<(u64, u64)>, String> {
    continuous_ball_intervals_with_cancel(
        source,
        expected_sha,
        frames,
        min_score,
        min_seconds,
        None,
    )
}

fn read_continuous_ball_points(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    min_score: f64,
    min_seconds: f64,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<Option<[f64; 2]>>, String> {
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        return Err("cancelled".into());
    }
    if frames == 0
        || frames > 48 * 3600 * 30
        || !min_score.is_finite()
        || !(0.01..=1.0).contains(&min_score)
        || !min_seconds.is_finite()
        || !(0.1..=120.0).contains(&min_seconds)
    {
        return Err("Invalid continuous ball trajectory settings".into());
    }
    let mut reader = BoundedReader::open(source)?;
    let mut points = Vec::with_capacity(frames as usize);
    for frame in 0..frames {
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        let line = reader.line()?.ok_or("Truncated continuous ball source")?;
        let row: BallOnlyFrame =
            serde_json::from_slice(&line).map_err(|_| "Malformed continuous ball source")?;
        if row.frame_index != frame {
            return Err("Continuous ball source frames out of order".into());
        }
        let point = row.selected_ball.and_then(|ball| {
            let bbox = ball.detection.bbox_xyxy_norm;
            (ball.source_model == "vballnet"
                && ball.detection.score.is_finite()
                && ball.detection.score >= min_score
                && bbox
                    .iter()
                    .all(|coord| coord.is_finite() && (0.0..=1.0).contains(coord))
                && bbox[0] < bbox[2]
                && bbox[1] < bbox[3])
                .then_some([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5])
        });
        points.push(point);
    }
    if reader.line()?.is_some() || reader.finish() != expected_sha {
        return Err("Continuous ball source SHA-256 mismatch".into());
    }

    Ok(points)
}

pub(crate) fn continuous_ball_intervals_with_cancel(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    min_score: f64,
    min_seconds: f64,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<(u64, u64)>, String> {
    let points =
        read_continuous_ball_points(source, expected_sha, frames, min_score, min_seconds, cancel)?;
    continuous_intervals_from_positions(&points, min_seconds)
}

pub(crate) fn continuous_ball_analysis(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    min_score: f64,
    min_seconds: f64,
    merge_gap_seconds: f64,
    stationary_ball_ms: u32,
) -> Result<MovingBallIntervals, String> {
    let points =
        read_continuous_ball_points(source, expected_sha, frames, min_score, min_seconds, None)?;
    analyze_moving_ball(&points, min_seconds, merge_gap_seconds, stationary_ball_ms)
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
    read_with_cancel(
        source,
        expected_sha,
        frames,
        (width, height),
        min_score,
        rally_starts,
        None,
    )
}

pub(crate) fn read_with_cancel(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    dimensions: (u32, u32),
    min_score: f64,
    rally_starts: &[u64],
    cancel: Option<&AtomicBool>,
) -> Result<Arc<BallTrackMask>, String> {
    let (width, height) = dimensions;
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        return Err("cancelled".into());
    }
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
        (width, height),
        min_score,
        rally_starts,
        cancel,
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
    dimensions: (u32, u32),
    min_score: f64,
    rally_starts: &[u64],
    cancel: Option<&AtomicBool>,
) -> Result<BallTrackMask, String> {
    let (width, height) = dimensions;
    const CHUNK: usize = 240;
    const HALO: usize = 16;
    let total = frames as usize;
    let mut mask = BallTrackMask {
        accepted: vec![false; total],
        rejected: vec![false; total],
        track_ids: vec![0; total],
        candidate_counts: vec![0; total],
    };
    let mut tracker = (!rally_starts.is_empty()).then(|| LeadingBallTracker::new(width, height));
    let mut next_rally = 0;
    let mut reader = BoundedReader::open(source)?;
    let mut buffer = std::collections::VecDeque::<Option<[f64; 2]>>::new();
    let (mut buffer_start, mut next_read, mut core_start) = (0, 0, 0);
    while core_start < total {
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
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
            if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
                return Err("cancelled".into());
            }
            let point = points[index - region_start];
            mask.candidate_counts[index] = u8::from(point.is_some());
            let observed = point.filter(|_| !local_flags[index - region_start]);
            mask.rejected[index] = point.is_some() && local_flags[index - region_start];
            mask.accepted[index] = observed.is_some();
            if let Some(leader) = tracker.as_mut() {
                if rally_starts.get(next_rally) == Some(&(index as u64)) {
                    if index > 0 {
                        leader.begin_rally();
                    }
                    next_rally += 1;
                }
                let leading = leader.observe(index as u64, observed.as_slice());
                mask.rejected[index] |= observed.is_some() && leading.selected.is_none();
                mask.accepted[index] &= leading.selected.is_some();
                mask.track_ids[index] = leading.track_id.unwrap_or(0);
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

    #[test]
    fn high_ball_returns_to_existing_rally_without_minimum_independent_duration() {
        let mut points = vec![None; 330];
        for (frame, point) in points.iter_mut().enumerate().take(200).skip(20) {
            let y = if frame < 185 {
                0.4
            } else {
                0.4 - (frame - 184) as f64 * 0.026
            };
            *point = Some([0.2 + (frame - 20) as f64 * 0.002, y]);
        }
        for (frame, point) in points.iter_mut().enumerate().take(244).skip(230) {
            *point = Some([
                0.59 + (frame - 230) as f64 * 0.001,
                0.02 + (frame - 230) as f64 * 0.02,
            ]);
        }
        let analysis = analyze_moving_ball(&points, 1.5, 4.0, 1000).unwrap();
        assert_eq!(analysis.merged, vec![(20, 243)], "{analysis:?}");
        let no_merge = analyze_moving_ball(&points, 1.5, 0.5, 1000).unwrap();
        assert_eq!(no_merge.merged, vec![(20, 196)]);
        for point in points[230..244].iter_mut().flatten() {
            point[1] += 0.2;
        }
        let unrelated = analyze_moving_ball(&points, 1.5, 4.0, 1000).unwrap();
        assert_eq!(unrelated.merged, vec![(20, 196)]);
    }

    #[test]
    fn final_rally_duration_is_checked_after_merge_at_exact_five_second_boundary() {
        let segments = [(20, 168), (300, 449), (600, 690), (750, 810)];
        assert_eq!(filter_short_rallies(segments.to_vec()), vec![(300, 449)],);
        let combined = merge_continuous_intervals_with_pauses(
            &[(600, 690), (750, 810)],
            CONTINUOUS_MERGE_SECONDS,
            &[],
        )
        .unwrap();
        assert_eq!(filter_short_rallies(combined), vec![(600, 810)]);
    }

    #[test]
    fn stationary_evidence_inside_unobserved_gap_blocks_interval_merge() {
        let detected = [(30, 120), (200, 290)];
        let pause = [(145, 170)];
        assert_eq!(
            merge_continuous_intervals_with_pauses(&detected, 4.0, &pause).unwrap(),
            detected.to_vec()
        );
        assert_eq!(
            merge_continuous_intervals_with_pauses(&detected, 4.0, &[]).unwrap(),
            vec![(30, 290)]
        );
    }

    #[test]
    fn cancelled_ball_prescan_does_not_open_source() {
        let cancelled = AtomicBool::new(true);
        let missing = std::env::temp_dir().join("missing-cancelled-ball-source.jsonl");
        assert_eq!(
            continuous_ball_intervals_with_cancel(
                &missing,
                "no-sha",
                9,
                0.5,
                1.5,
                Some(&cancelled)
            )
            .unwrap_err(),
            "cancelled"
        );
        assert!(
            matches!(read_with_cancel(&missing, "no-sha", 9, (640, 360), 0.5, &[], Some(&cancelled)), Err(ref error) if error == "cancelled")
        );
    }
    use serde_json::json;
    use sha2::{Digest, Sha256};

    #[test]
    fn continuous_ball_accepts_exact_one_and_a_half_seconds() {
        let mut samples = (0_u64..150)
            .map(|frame| {
                if (40..85).contains(&frame) {
                    Some([0.1 + (frame - 40) as f64 * 0.0035, 0.4])
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let exact = continuous_intervals_from_positions(&samples, 1.5).unwrap();
        assert_eq!(exact, vec![(40, 84)]);
        samples[84] = None;
        assert!(continuous_intervals_from_positions(&samples, 1.5)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn continuous_ball_rejects_unmoving_high_confidence_detections() {
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
        assert!(continuous_ball_intervals(&path, &sha, 310, 0.35, 3.0)
            .unwrap()
            .is_empty());
        assert!(continuous_ball_intervals(&path, &sha, 310, 0.35, 1.0)
            .unwrap()
            .is_empty());
        assert!(continuous_ball_intervals(&path, &"0".repeat(64), 310, 0.35, 1.0).is_err());
        assert!(continuous_ball_intervals(&path, &sha, 310, 0.35, f64::NAN).is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn continuous_ball_joins_short_gaps_and_ignores_static_and_short_flights() {
        let root = std::env::temp_dir().join(format!("continuous-motion-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let path = root.join("tracked.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..600_u64 {
            let flight = (70..240).contains(&frame)
                || (305..480).contains(&frame)
                || (510..567).contains(&frame);
            let rel = if frame < 240 {
                frame.saturating_sub(70)
            } else if frame < 480 {
                frame.saturating_sub(305)
            } else {
                frame.saturating_sub(510)
            };
            let false_detection = matches!(frame, 157 | 158 | 367 | 368);
            let missed_detection = flight && frame % 41 == 0;
            let (x, y) = if flight && !false_detection {
                (0.2 + rel as f64 * 0.001, 0.15 + rel as f64 * 0.003)
            } else {
                (0.92 + (frame % 5) as f64 * 0.0008, 0.57)
            };
            bytes.extend(
                serde_json::to_vec(&json!({
                    "frame_index":frame,
                    "selected_ball":if missed_detection {None} else {Some(json!({
                        "source_model":"vballnet",
                        "detection":{"score":0.9,
                            "bbox_xyxy_norm":[x-0.002,y-0.002,x+0.002,y+0.002]}
                    }))}
                }))
                .unwrap(),
            );
            bytes.push(b'\n');
        }
        fs::write(&path, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let intervals = continuous_ball_intervals(&path, &sha, 600, 0.35, 3.0).unwrap();
        assert_eq!(intervals.len(), 2, "{intervals:?}");
        for (actual, expected) in intervals.iter().zip([(70_u64, 239_u64), (305, 479)]) {
            assert!(actual.0.abs_diff(expected.0) <= 18, "{intervals:?}");
            assert!(actual.1.abs_diff(expected.1) <= 18, "{intervals:?}");
        }
        assert!(continuous_ball_intervals(&path, &sha, 600, 0.35, 7.0)
            .unwrap()
            .is_empty());
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
    fn local_ball_policy_never_resets_or_tracks_a_leading_ball() {
        let root = std::env::temp_dir().join(format!("local-ball-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("frames.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..300_u64 {
            let x = if frame < 200 {
                0.15 + frame as f64 * 0.0008
            } else {
                0.65 + (frame - 200) as f64 * 0.0008
            };
            bytes.extend(
                serde_json::to_vec(&json!({
                    "frame_index":frame,
                    "selected_ball":{
                        "source_model":"vballnet",
                        "detection":{"score":0.90,"bbox_xyxy_norm":[x,0.4,x+0.005,0.405]}
                    }
                }))
                .unwrap(),
            );
            bytes.push(b'\n');
        }
        fs::write(&path, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let local = read(&path, &sha, 300, 1280, 720, 0.35, &[]).unwrap();
        assert!(local.track_ids.iter().all(|id| *id == 0));
        assert!(local.accepted[260]);
        let saved = json!({
            "ball_tracking_policy":"vballnet_local_artifacts_v1",
            "leading_ball_rally_starts":[],
            "leading_ball_min_score":0.35,
            "frames":300
        });
        assert_eq!(saved_scope(&saved).unwrap(), Some((vec![], 0.35)));
        fs::remove_dir_all(root).unwrap();
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

#[cfg(test)]
mod continuous_rally_tests {
    use super::*;

    #[test]
    fn shared_merging_preserves_strict_four_second_boundary() {
        assert_eq!(
            merge_continuous_intervals(&[(221, 300), (30, 100)], 4.0).unwrap(),
            vec![(30, 100), (221, 300)]
        );
        assert_eq!(
            merge_continuous_intervals(&[(30, 100), (219, 300)], 4.0).unwrap(),
            vec![(30, 300)]
        );
        assert!(merge_continuous_intervals(&[], CONTINUOUS_MERGE_SECONDS)
            .unwrap()
            .is_empty());
        assert!(merge_continuous_intervals(&[(5, 3)], 4.0).is_err());
    }
}
