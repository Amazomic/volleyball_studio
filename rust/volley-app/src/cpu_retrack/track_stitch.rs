use super::{BoundedReader, PlayerFrameScope, MAX_ROW_BYTES};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};
use volley_core::fusion::Candidate;

#[derive(Deserialize)]
struct Frame {
    frame_index: u64,
    tracked_players: Vec<Observation>,
}

#[derive(Deserialize)]
struct Observation {
    track_id: u64,
    detection: Candidate,
}

#[derive(Default)]
struct Track {
    start: Option<u64>,
    end: u64,
    observed: u64,
    first_boxes: Vec<(u64, [f64; 4])>,
    last_boxes: VecDeque<(u64, [f64; 4])>,
    first_reliable: Option<(u64, [f64; 4])>,
    last_reliable: Option<(u64, [f64; 4])>,
    first_occluder: Option<u64>,
    last_occluder: Option<u64>,
}

#[derive(Default)]
struct Simultaneous {
    total: u32,
    compatible: u32,
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[derive(Clone, Copy)]
struct Motion {
    origin: f64,
    value: f64,
    velocity: f64,
    error: f64,
}

impl Motion {
    fn at(self, frame: f64) -> f64 {
        self.value + self.velocity * (frame - self.origin)
    }
}

fn fit_motion(samples: &[(u64, [f64; 4])], point: impl Fn([f64; 4]) -> f64) -> Option<Motion> {
    let first = samples.first()?.0;
    let last = samples.last()?.0;
    if samples.len() < 3 || last - first < 4 || last - first > 60 {
        return None;
    }
    let origin = first as f64;
    let mut slopes = Vec::new();
    for (i, &(at, a)) in samples.iter().enumerate() {
        for &(next, b) in &samples[i + 1..] {
            if next - at >= 3 {
                slopes.push((point(b) - point(a)) / (next - at) as f64);
            }
        }
    }
    if slopes.is_empty() {
        return None;
    }
    let velocity = median(slopes);
    let value = median(
        samples
            .iter()
            .map(|&(at, b)| point(b) - velocity * (at as f64 - origin))
            .collect(),
    );
    let error = median(
        samples
            .iter()
            .map(|&(at, b)| (point(b) - value - velocity * (at as f64 - origin)).abs())
            .collect(),
    );
    Some(Motion {
        origin,
        value,
        velocity,
        error,
    })
}

fn box_size(b: [f64; 4]) -> [f64; 2] {
    [b[2] - b[0], b[3] - b[1]]
}

fn box_center_x(b: [f64; 4]) -> f64 {
    (b[0] + b[2]) * 0.5
}

fn compatible_boxes(a: [f64; 4], b: [f64; 4]) -> bool {
    let [aw, ah] = box_size(a);
    let [bw, bh] = box_size(b);
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let width = aw.min(bw);
    let height = ah.min(bh);
    width >= 0.008
        && height >= 0.025
        && intersection >= 0.75 * (aw * ah).min(bw * bh)
        && (box_center_x(a) - box_center_x(b)).abs() <= (0.35 * width).max(0.004)
        && (a[1] - b[1]).abs().min((a[3] - b[3]).abs()) <= (0.12 * height).max(0.005)
}

fn motion_handoff(old: &Track, new: &Track, start: u64) -> Option<f64> {
    let a = old.last_boxes.iter().copied().collect::<Vec<_>>();
    let b = &new.first_boxes;
    let gap = start.saturating_sub(old.end);
    if gap > 90
        || old.end.saturating_sub(a.last()?.0) > 10
        || b.first()?.0.saturating_sub(start) > 10
    {
        return None;
    }
    let size = |samples: &[(u64, [f64; 4])], axis: usize| {
        median(
            samples
                .iter()
                .map(|&(_, bbox)| box_size(bbox)[axis])
                .collect(),
        )
    };
    let (aw, bw) = (size(&a, 0), size(b, 0));
    let (ah, bh) = (size(&a, 1), size(b, 1));
    if aw.min(bw) < 0.008
        || ah.min(bh) < 0.025
        || aw.max(bw) > 1.8 * aw.min(bw)
        || ah.max(bh)
            > if gap == 0 {
                2.5 * ah.min(bh)
            } else {
                2.0 * ah.min(bh)
            }
    {
        return None;
    }
    let stable = |samples: &[(u64, [f64; 4])], width: f64, height: f64| {
        samples
            .iter()
            .copied()
            .filter(|&(_, bbox)| {
                let [w, h] = box_size(bbox);
                w >= 0.7 * width && h >= 0.75 * height && h <= 1.35 * height
            })
            .collect::<Vec<_>>()
    };
    let a = stable(&a, aw, ah);
    let b = stable(b, bw, bh);
    let from = old.end as f64;
    let to = start as f64;
    let tolerance_x = (aw.min(bw) * 0.40 + 0.00015 * gap as f64).clamp(0.006, 0.025);
    let tolerance_y = (ah.min(bh) * 0.12 + 0.0002 * gap as f64).clamp(0.007, 0.035);
    let compare = |left: Motion, right: Motion, tolerance: f64| -> Option<f64> {
        if left.error > tolerance * 0.6 || right.error > tolerance * 0.6 {
            return None;
        }
        let forward = (left.at(to) - right.at(to)).abs();
        let backward = (left.at(from) - right.at(from)).abs();
        if forward.max(backward) > tolerance {
            return None;
        }
        Some((forward + backward) * 0.5)
    };
    let x = compare(
        fit_motion(&a, box_center_x)?,
        fit_motion(&b, box_center_x)?,
        tolerance_x,
    )?;
    let y = [1, 3]
        .into_iter()
        .filter_map(|edge| {
            compare(
                fit_motion(&a, |bbox| bbox[edge])?,
                fit_motion(&b, |bbox| bbox[edge])?,
                tolerance_y,
            )
        })
        .min_by(f64::total_cmp)?;
    Some(x.hypot(y) + 0.0006 * gap as f64)
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

#[derive(Clone, Copy)]
struct OcclusionBridge {
    left: u64,
    right: u64,
    occluder: u64,
}

fn occlusion_bridge(old: &Track, new: &Track, start: u64) -> Option<(f64, OcclusionBridge)> {
    let occluder = old.last_occluder?;
    if start - old.end > 16 || Some(occluder) != new.first_occluder {
        return None;
    }
    let ((left, a), (right, b)) = (old.last_reliable?, new.first_reliable?);
    if old.end - left > 4 || right - start > 4 || (left == old.end && right == start) {
        return None;
    }
    let (aw, ah) = (a[2] - a[0], a[3] - a[1]);
    let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
    let dx = ((a[0] + a[2]) - (b[0] + b[2])).abs() * 0.5;
    let dy = ((a[1] + a[3]) - (b[1] + b[3])).abs() * 0.5;
    let distance = dx.hypot(dy);
    (aw.min(bw) >= 0.005
        && ah.min(bh) >= 0.02
        && aw.max(bw) <= 1.65 * aw.min(bw)
        && ah.max(bh) <= 1.50 * ah.min(bh)
        && dx <= 1.55 * aw.max(bw)
        && dy <= 0.25 * ah.min(bh)
        && distance <= 0.045)
        .then_some((
            distance,
            OcclusionBridge {
                left,
                right,
                occluder,
            },
        ))
}

fn foreground_track(target: &Observation, players: &[Observation]) -> Option<u64> {
    let a = target.detection.bbox_xyxy_norm;
    let width = a[2] - a[0];
    let height = a[3] - a[1];
    let mut covering = players.iter().filter(|other| {
        let b = other.detection.bbox_xyxy_norm;
        other.track_id != target.track_id
            && other.detection.score >= 0.45
            && b[3] - b[1] >= 1.30 * height
            && b[3] > a[3] + 0.25 * height
            && a[2].min(b[2]) - a[0].max(b[0]) >= width * 0.30
            && a[3].min(b[3]) - a[1].max(b[1]) >= height * 0.35
    });
    let first = covering.next()?;
    covering.next().is_none().then_some(first.track_id)
}

#[derive(Default)]
struct Handoffs {
    aliases: BTreeMap<u64, u64>,
    occlusions: BTreeMap<u64, Vec<OcclusionBridge>>,
}

#[cfg(test)]
fn find_handoffs(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    scope: Option<&PlayerFrameScope>,
) -> Result<BTreeMap<u64, u64>, String> {
    find_links(source, expected_sha, frames, scope).map(|links| links.aliases)
}

fn find_links(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    scope: Option<&PlayerFrameScope>,
) -> Result<Handoffs, String> {
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
            let occluder = foreground_track(observation, &row.tracked_players);
            if track.start.is_none() {
                track.first_occluder = occluder;
            }
            track.last_occluder = occluder;
            track.start.get_or_insert(frame);
            track.end = frame;
            track.observed += 1;
            if observation.detection.score >= 0.35 {
                let reliable = (frame, observation.detection.bbox_xyxy_norm);
                track.first_reliable.get_or_insert(reliable);
                track.last_reliable = Some(reliable);
                if track.first_boxes.len() < 12 && frame - track.start.unwrap_or(frame) <= 60 {
                    track.first_boxes.push(reliable);
                }
                track.last_boxes.push_back(reliable);
                while track.last_boxes.len() > 12
                    || track
                        .last_boxes
                        .front()
                        .is_some_and(|(at, _)| frame - at > 60)
                {
                    track.last_boxes.pop_front();
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
                evidence.compatible += u32::from(
                    compatible_boxes(a.detection.bbox_xyxy_norm, b.detection.bbox_xyxy_norm)
                        || complementary_parts(&a.detection, &b.detection),
                );
            }
        }
    }
    if reader.line()?.is_some() || reader.finish() != expected_sha {
        return Err("Confirmed tracking source SHA-256 mismatch before Track ID stitching".into());
    }
    let mut aliases = BTreeMap::new();
    let mut gap_candidates = Vec::new();
    let mut starts = BTreeMap::<u64, Vec<u64>>::new();
    for (&id, track) in &tracks {
        if let Some(start) = track.start {
            starts.entry(start).or_default().push(id);
        }
    }
    for (&old_id, old) in &tracks {
        if old.observed < 3 {
            continue;
        }
        for &new_id in starts
            .range(old.end.saturating_sub(7)..=old.end.saturating_add(90))
            .flat_map(|(_, ids)| ids)
        {
            let new = &tracks[&new_id];
            if old_id == new_id || new.observed < 3 {
                continue;
            }
            let (Some(old_start), Some(start)) = (old.start, new.start) else {
                continue;
            };
            if start <= old_start
                || new.end <= old.end
                || old.end.saturating_sub(start) > 7
                || !scope.is_some_and(|mask| {
                    mask.same_confirmed_rally(old.end.min(start), old.end.max(start))
                })
            {
                continue;
            }
            if start <= old.end {
                let Some(evidence) = pairs.get(&(old_id.min(new_id), old_id.max(new_id))) else {
                    continue;
                };
                if evidence.total == 0 || evidence.total != evidence.compatible {
                    continue;
                }
            }
            if let Some(distance) = motion_handoff(old, new, start) {
                gap_candidates.push((distance, old_id, new_id, None));
            } else if start > old.end {
                if let Some((distance, bridge)) =
                    occlusion_bridge(old, new, start).filter(|&(_, bridge)| {
                        scope.is_some_and(|mask| {
                            mask.same_confirmed_rally(bridge.left, bridge.right)
                        })
                    })
                {
                    gap_candidates.push((distance, old_id, new_id, Some(bridge)));
                }
            }
        }
    }
    gap_candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut successors = BTreeMap::new();
    let mut predecessors = BTreeSet::new();
    for &(distance, old_id, new_id, bridge) in &gap_candidates {
        if successors.contains_key(&old_id) || predecessors.contains(&new_id) {
            continue;
        }
        if gap_candidates
            .iter()
            .any(|&(other_distance, other_old, other_new, _)| {
                (other_old == old_id && other_new != new_id
                    || other_new == new_id && other_old != old_id)
                    && other_distance <= distance + 0.006
            })
        {
            continue;
        }
        successors.insert(old_id, (new_id, bridge));
        predecessors.insert(new_id);
    }
    let mut occlusions = BTreeMap::<u64, Vec<OcclusionBridge>>::new();
    for &root in successors.keys().filter(|id| !predecessors.contains(id)) {
        let mut current = root;
        while let Some(&(next, bridge)) = successors.get(&current) {
            if let Some(bridge) = bridge {
                occlusions.entry(root).or_default().push(bridge);
            }
            if pairs
                .get(&(root.min(next), root.max(next)))
                .is_some_and(|p| p.total != p.compatible)
            {
                break;
            }
            aliases.insert(next, root);
            current = next;
        }
    }
    Ok(Handoffs {
        aliases,
        occlusions,
    })
}

#[derive(Clone, Copy)]
struct RosterObservation {
    id: u64,
    bbox: [f64; 4],
    score: f64,
}

#[derive(Clone, Copy)]
struct RosterSlot {
    id: u64,
    source_id: u64,
    frame: usize,
    bbox: [f64; 4],
    velocity: [f64; 2],
}

impl RosterSlot {
    fn new(observation: RosterObservation, frame: usize) -> Self {
        Self {
            id: observation.id,
            source_id: observation.id,
            frame,
            bbox: observation.bbox,
            velocity: [0.0; 2],
        }
    }

    fn predict(&self, frame: usize) -> [f64; 4] {
        let dt = (frame as f64 - self.frame as f64).clamp(-12.0, 12.0);
        std::array::from_fn(|i| self.bbox[i] + self.velocity[i % 2] * dt)
    }

    fn continuation_cost(&self, observation: RosterObservation, frame: usize) -> Option<f64> {
        if self.frame.abs_diff(frame) > 2 {
            return None;
        }
        let a = self.predict(frame);
        let b = observation.bbox;
        let [aw, ah] = box_size(a);
        let [bw, bh] = box_size(b);
        if aw.min(bw) < 0.003
            || ah.min(bh) < 0.015
            || ah.min(bh) < 0.78 * ah.max(bh)
            || aw.min(bw) < 0.5 * aw.max(bw)
        {
            return None;
        }
        let dx = (box_center_x(a) - box_center_x(b)).abs() / aw.max(bw);
        let dy = (a[3] - b[3]).abs() / ah.max(bh);
        if dx > 0.45 || dy > 0.16 {
            return None;
        }
        Some(dx + 2.0 * dy + 0.15 * (ah / bh).ln().abs())
    }
    fn score(&self, observation: RosterObservation, frame: usize) -> Option<f64> {
        let gap = self.frame.abs_diff(frame);
        let same_source = self.source_id == observation.id;
        if gap > 90 && !same_source {
            return None;
        }
        let a = self.predict(frame);
        let b = observation.bbox;
        let [aw, ah] = box_size(a);
        let [bw, bh] = box_size(b);
        if aw.min(bw) < 0.003 || ah.min(bh) < 0.015 {
            return None;
        }
        let dx = (box_center_x(a) - box_center_x(b)).abs();
        let bottom = (a[3] - b[3]).abs();
        let center_y = ((a[1] + a[3] - b[1] - b[3]) * 0.5).abs();
        let partial = ah.min(bh) < 0.78 * ah.max(bh);
        let contained_y = (a[1].max(b[1]) - a[3].min(b[3])).max(0.0);
        let dy = if partial && dx <= 0.65 * aw.max(bw) {
            bottom.min(center_y).min(contained_y + 0.18 * ah.min(bh))
        } else {
            bottom.min(center_y + 0.01)
        };
        let spread = gap.min(90) as f64 * if gap > 2 { 0.0020 } else { 0.0012 };
        let tx = (0.75 * aw.max(bw) + spread).clamp(0.012, 0.16);
        let ty = (0.35 * ah.max(bh) + spread).clamp(0.018, 0.16);
        let distance = (dx / tx).hypot(dy / ty);
        let size_cost = ((aw / bw).ln().abs() + (ah / bh).ln().abs()) * 0.09;
        if distance > 1.25 || (!same_source && aw.max(bw) > 2.5 * aw.min(bw)) {
            return None;
        }
        let continuity = if same_source {
            0.24
        } else if self.id == observation.id {
            0.08
        } else {
            0.0
        };
        let score = 0.95 - 0.55 * distance - size_cost + continuity + 0.06 * observation.score;
        (score > 0.20).then_some(score.min(1.0))
    }

    fn update(&mut self, observation: RosterObservation, frame: usize) {
        let previous = self.bbox;
        let predicted = self.predict(frame);
        let mut bbox = observation.bbox;
        let [pw, ph] = box_size(predicted);
        let [w, h] = box_size(bbox);
        if h < 0.75 * ph
            && w <= 1.3 * pw
            && (box_center_x(predicted) - box_center_x(bbox)).abs() <= 0.55 * pw
            && bbox[1] <= predicted[3]
            && bbox[3] >= predicted[1]
        {
            let dx = box_center_x(bbox) - box_center_x(predicted);
            let top = bbox[1] - predicted[1];
            let bottom = bbox[3] - predicted[3];
            let dy = if top.abs() < bottom.abs() {
                top
            } else {
                bottom
            };
            bbox = [
                predicted[0] + dx,
                predicted[1] + dy,
                predicted[2] + dx,
                predicted[3] + dy,
            ];
        }
        let dt = frame as f64 - self.frame as f64;
        if dt != 0.0 {
            let vx = ((box_center_x(bbox) - box_center_x(previous)) / dt).clamp(-0.008, 0.008);
            let vy = ((bbox[3] - previous[3]) / dt).clamp(-0.012, 0.012);
            self.velocity = [
                self.velocity[0] * 0.55 + vx * 0.45,
                self.velocity[1] * 0.55 + vy * 0.45,
            ];
        }
        self.bbox = bbox;
        self.frame = frame;
        self.source_id = observation.id;
    }
}

fn roster_fragment(a: [f64; 4], b: [f64; 4]) -> bool {
    let [aw, ah] = box_size(a);
    let [bw, bh] = box_size(b);
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    aw.min(bw) >= 0.003
        && ah.min(bh) >= 0.015
        && (box_center_x(a) - box_center_x(b)).abs() <= 0.35 * aw.max(bw)
        && (intersection >= 0.80 * (aw * ah).min(bw * bh)
            || (a[3].min(b[3]) - a[1].max(b[1])).abs() <= 0.008
                && (a[2].min(b[2]) - a[0].max(b[0])) >= 0.8 * aw.min(bw))
}

fn assign_roster_frame(
    observations: &[RosterObservation],
    slots: &mut [RosterSlot],
    frame: usize,
) -> Vec<Option<u64>> {
    let costs = observations
        .iter()
        .map(|&observation| {
            slots
                .iter()
                .map(|slot| slot.continuation_cost(observation, frame))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let unique_best = |values: &[(usize, f64)]| {
        let mut ordered = values.to_vec();
        ordered.sort_by(|a, b| a.1.total_cmp(&b.1));
        let &(index, cost) = ordered.first()?;
        if ordered
            .get(1)
            .is_some_and(|&(_, other)| other - cost < 0.15)
        {
            return None;
        }
        Some(index)
    };
    let mut result = vec![None; observations.len()];
    let mut matched = Vec::new();
    let mut reserved = vec![false; slots.len()];
    for (i, row) in costs.iter().enumerate() {
        let candidates = row
            .iter()
            .enumerate()
            .filter_map(|(j, c)| c.map(|c| (j, c)))
            .collect::<Vec<_>>();
        let Some(j) = unique_best(&candidates) else {
            continue;
        };
        let competitors = costs
            .iter()
            .enumerate()
            .filter_map(|(k, r)| r[j].map(|c| (k, c)))
            .collect::<Vec<_>>();
        if unique_best(&competitors) == Some(i) {
            result[i] = Some(slots[j].id);
            reserved[j] = true;
            matched.push((i, j));
        }
    }
    let scores = observations
        .iter()
        .enumerate()
        .map(|(i, &o)| {
            slots
                .iter()
                .enumerate()
                .map(|(j, slot)| {
                    if result[i].is_some() || reserved[j] {
                        None
                    } else {
                        slot.score(o, frame)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for (index, slot) in volley_core::tracker::assign_track_scores(&scores)
        .into_iter()
        .enumerate()
    {
        if let Some(slot) = slot {
            result[index] = Some(slots[slot].id);
            matched.push((index, slot));
        }
    }
    for (index, o) in observations.iter().enumerate() {
        if result[index].is_some() {
            continue;
        }
        let mut compatible = matched
            .iter()
            .filter(|&&(i, _)| roster_fragment(observations[i].bbox, o.bbox));
        if let Some(&(_, slot)) = compatible.next() {
            if compatible.next().is_none() {
                result[index] = Some(slots[slot].id);
            }
        }
    }
    for (index, slot) in matched {
        let mut observation = observations[index];
        for (i, &owner) in result.iter().enumerate() {
            if i != index && owner == Some(slots[slot].id) {
                let b = observations[i].bbox;
                let a = observation.bbox;
                observation.bbox = [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ];
            }
        }
        slots[slot].update(observation, frame);
    }
    result
}

fn roster_plan(
    frames: &[Vec<RosterObservation>],
    start: usize,
    end: usize,
) -> Option<Vec<Vec<Option<u64>>>> {
    let sample = frames.get(start..=end)?;
    let counts = sample.iter().map(|row| row.len() as u8).collect::<Vec<_>>();
    let target = volley_core::rally_reconstruction::stable_track_count(&counts) as usize;
    if target == 0 {
        return None;
    }
    let mut support = BTreeMap::<u64, usize>::new();
    for row in sample {
        for observation in row {
            *support.entry(observation.id).or_default() += 1;
        }
    }
    let mid = (start + end) / 2;
    let seed = (start..=end)
        .filter(|&i| frames[i].len() == target)
        .max_by(|&a, &b| {
            let quality = |i: usize| {
                let row = &frames[i];
                let duplicates = row
                    .iter()
                    .enumerate()
                    .map(|(j, p)| {
                        row.iter()
                            .skip(j + 1)
                            .filter(|q| compatible_boxes(p.bbox, q.bbox))
                            .count()
                    })
                    .sum::<usize>();
                let stability = row.iter().map(|p| support[&p.id]).sum::<usize>();
                (
                    std::cmp::Reverse(duplicates),
                    stability,
                    std::cmp::Reverse(i.abs_diff(mid)),
                )
            };
            quality(a).cmp(&quality(b))
        })?;
    let seed_slots = frames[seed]
        .iter()
        .copied()
        .map(|o| RosterSlot::new(o, seed))
        .collect::<Vec<_>>();
    let mut plan = vec![Vec::new(); frames.len()];
    plan[seed] = seed_slots.iter().map(|slot| Some(slot.id)).collect();
    let mut slots = seed_slots.clone();
    for frame in seed + 1..frames.len() {
        plan[frame] = assign_roster_frame(&frames[frame], &mut slots, frame);
    }
    let mut slots = seed_slots;
    for frame in (0..seed).rev() {
        plan[frame] = assign_roster_frame(&frames[frame], &mut slots, frame);
    }
    Some(plan)
}

type RosterPlan = (u64, u64, Vec<Vec<Option<u64>>>);

pub(super) struct RosterResult {
    pub sha: String,
    pub tracks: u64,
    pub merged_boxes: u64,
    pub unmatched_boxes: u64,
}

pub(super) fn reconcile_rally_rosters(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    scope: &PlayerFrameScope,
    cancel: Option<&AtomicBool>,
) -> Result<RosterResult, String> {
    let tmp = source.with_extension("rally-rosters");
    let result = (|| {
        let mut compact = BoundedReader::open(source)?;
        let mut reader = BoundedReader::open(source)?;
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)
                .map_err(|e| e.to_string())?,
        );
        let mut sha = Sha256::new();
        let mut tracks = BTreeSet::new();
        let mut merged_boxes = 0;
        let mut unmatched_boxes = 0;
        let mut windows = Vec::new();
        for &(selected_start, selected_end) in &scope.selected {
            let first = scope.rallies.partition_point(|&(a, _)| a < selected_start);
            let last = scope.rallies.partition_point(|&(_, b)| b <= selected_end);
            let rallies = scope
                .rallies
                .get(first..last)
                .ok_or("Invalid roster rally bounds")?;
            for (i, &(start, end)) in rallies.iter().enumerate() {
                let left = if i == 0 {
                    selected_start
                } else {
                    (rallies[i - 1].1 + start) / 2 + 1
                };
                let right = if i + 1 == rallies.len() {
                    selected_end
                } else {
                    (end + rallies[i + 1].0) / 2
                };
                windows.push((left, right, start, end));
            }
        }
        let mut index = 0;
        let mut plan: Option<RosterPlan> = None;
        for frame in 0..frames {
            if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
                return Err("cancelled".into());
            }
            if plan.as_ref().is_some_and(|(_, end, _)| frame > *end) {
                plan = None;
            }
            if windows.get(index).is_some_and(|w| w.0 == frame) {
                let (left, right, start, end) = windows[index];
                index += 1;
                if right - left >= 1_000_000 {
                    return Err("Rally roster exceeds bounded frame limit".into());
                }
                let mut observations = Vec::new();
                let mut total = 0;
                for at in left..=right {
                    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
                        return Err("cancelled".into());
                    }
                    let row = observed_row(&mut compact, at)?;
                    total += row.tracked_players.len();
                    if total > 1_000_000 {
                        return Err("Rally roster exceeds bounded observation limit".into());
                    }
                    observations.push(
                        row.tracked_players
                            .into_iter()
                            .map(|o| RosterObservation {
                                id: o.track_id,
                                bbox: o.detection.bbox_xyxy_norm,
                                score: o.detection.score,
                            })
                            .collect::<Vec<_>>(),
                    );
                }
                let owners = roster_plan(
                    &observations,
                    (start - left) as usize,
                    (end - left) as usize,
                )
                .unwrap_or_else(|| {
                    observations
                        .iter()
                        .map(|r| r.iter().map(|o| Some(o.id)).collect())
                        .collect()
                });
                plan = Some((left, right, owners));
            } else if plan.is_none() {
                observed_row(&mut compact, frame)?;
            }
            let mut row: Value =
                serde_json::from_slice(&reader.line()?.ok_or("Roster replay truncated")?)
                    .map_err(|e| e.to_string())?;
            if row["frame_index"].as_u64() != Some(frame) {
                return Err("Roster frame order mismatch".into());
            }
            if let Some((left, _, _)) = plan.as_ref() {
                row["rally_roster_window"] = json!(left);
            }
            let players = row["tracked_players"]
                .as_array_mut()
                .ok_or("Missing roster observations")?;
            if let Some((left, _, ref owners)) = plan {
                let owners = &owners[(frame - left) as usize];
                if owners.len() != players.len() {
                    return Err("Roster observations changed".into());
                }
                let mut canonical = BTreeMap::<u64, Value>::new();
                let mut dropped = 0;
                let mut rewritten = false;
                for (mut player, &owner) in std::mem::take(players).into_iter().zip(owners) {
                    let Some(id) = owner else {
                        unmatched_boxes += 1;
                        dropped += 1;
                        continue;
                    };
                    rewritten |= player["track_id"].as_u64() != Some(id);
                    player["track_id"] = json!(id);
                    match canonical.get_mut(&id) {
                        Some(previous) => {
                            merged_boxes += 1;
                            let a: [f64; 4] = serde_json::from_value(
                                previous["detection"]["bbox_xyxy_norm"].clone(),
                            )
                            .map_err(|e| e.to_string())?;
                            let b: [f64; 4] = serde_json::from_value(
                                player["detection"]["bbox_xyxy_norm"].clone(),
                            )
                            .map_err(|e| e.to_string())?;
                            if player["detection"]["score"].as_f64()
                                > previous["detection"]["score"].as_f64()
                            {
                                *previous = player;
                            }
                            previous["detection"]["bbox_xyxy_norm"] = json!([
                                a[0].min(b[0]),
                                a[1].min(b[1]),
                                a[2].max(b[2]),
                                a[3].max(b[3])
                            ]);
                            previous["pose_crop_origin"] = json!("combined_track_box");
                        }
                        None => {
                            canonical.insert(id, player);
                        }
                    }
                }
                *players = canonical.into_values().collect();
                for player in players.iter_mut() {
                    if rewritten {
                        player
                            .as_object_mut()
                            .ok_or("Invalid roster player")?
                            .remove("occlusion_bridge");
                    }
                }
                tracks.extend(players.iter().filter_map(|p| p["track_id"].as_u64()));
                if dropped > 0 {
                    row["unassigned_rally_boxes"] = json!(dropped);
                }
                if rewritten || dropped > 0 {
                    row["new_track_diagnostics"] = json!([]);
                }
            } else {
                tracks.extend(players.iter().filter_map(|p| p["track_id"].as_u64()));
            }
            let mut bytes = serde_json::to_vec(&row).map_err(|e| e.to_string())?;
            bytes.push(b'\n');
            if bytes.len() > MAX_ROW_BYTES {
                return Err("Roster row exceeds limit".into());
            }
            writer.write_all(&bytes).map_err(|e| e.to_string())?;
            sha.update(&bytes);
        }
        if index != windows.len()
            || reader.line()?.is_some()
            || compact.line()?.is_some()
            || reader.finish() != expected_sha
            || compact.finish() != expected_sha
        {
            return Err("Roster source SHA mismatch".into());
        }
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
        drop(writer);
        fs::rename(&tmp, source).map_err(|e| e.to_string())?;
        Ok(RosterResult {
            sha: format!("{:x}", sha.finalize()),
            tracks: tracks.len() as u64,
            merged_boxes,
            unmatched_boxes,
        })
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}
struct TrackShape {
    first: u64,
    last: u64,
    count: u64,
    min_edges: [f64; 4],
    max_edges: [f64; 4],
    max_aspect: f64,
}

impl TrackShape {
    fn new(frame: u64, bbox: [f64; 4]) -> Self {
        Self {
            first: frame,
            last: frame,
            count: 0,
            min_edges: bbox,
            max_edges: bbox,
            max_aspect: 0.0,
        }
    }

    fn observe(&mut self, frame: u64, bbox: [f64; 4]) {
        self.last = frame;
        self.count += 1;
        for (i, edge) in bbox.into_iter().enumerate() {
            self.min_edges[i] = self.min_edges[i].min(edge);
            self.max_edges[i] = self.max_edges[i].max(edge);
        }
        let [width, height] = box_size(bbox);
        self.max_aspect = self.max_aspect.max(width / height);
    }

    fn stationary_post(&self) -> bool {
        let span = self.last - self.first + 1;
        span >= 90
            && self.count >= 60
            && self.count * 3 >= span * 2
            && self.max_aspect <= 0.17
            && self
                .min_edges
                .iter()
                .zip(self.max_edges)
                .all(|(min, max)| max - min <= 0.004)
    }
}

pub(super) struct PrunedTracks {
    pub sha: String,
    pub short_tracks: u64,
    pub short_detections: u64,
    pub static_tracks: u64,
    pub static_detections: u64,
}

pub(super) fn prune_unreliable(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    minimum: u64,
    cancel: Option<&AtomicBool>,
) -> Result<PrunedTracks, String> {
    let mut counts = BTreeMap::<u64, TrackShape>::new();
    let mut reader = BoundedReader::open(source)?;
    for frame in 0..frames {
        if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            return Err("cancelled".into());
        }
        for player in observed_row(&mut reader, frame)?.tracked_players {
            let bbox = player.detection.bbox_xyxy_norm;
            counts
                .entry(player.track_id)
                .or_insert_with(|| TrackShape::new(frame, bbox))
                .observe(frame, bbox);
        }
    }
    if reader.line()?.is_some() || reader.finish() != expected_sha {
        return Err("Stitched track SHA mismatch".into());
    }
    let short = counts
        .iter()
        .filter(|(_, shape)| shape.count < minimum)
        .map(|(&id, _)| id)
        .collect::<BTreeSet<_>>();
    let stationary = counts
        .iter()
        .filter(|(id, shape)| !short.contains(id) && shape.stationary_post())
        .map(|(&id, _)| id)
        .collect::<BTreeSet<_>>();
    let mut pruned = PrunedTracks {
        sha: expected_sha.to_owned(),
        short_tracks: short.len() as u64,
        short_detections: short.iter().map(|id| counts[id].count).sum(),
        static_tracks: stationary.len() as u64,
        static_detections: stationary.iter().map(|id| counts[id].count).sum(),
    };
    let rejected = short.union(&stationary).copied().collect::<BTreeSet<_>>();
    if rejected.is_empty() {
        return Ok(pruned);
    }
    let tmp = source.with_extension("short-tracks");
    let result = (|| {
        let mut reader = BoundedReader::open(source)?;
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)
                .map_err(|e| e.to_string())?,
        );
        let mut sha = Sha256::new();
        let mut removed = 0;
        for frame in 0..frames {
            if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                return Err("cancelled".into());
            }
            let mut row: Value =
                serde_json::from_slice(&reader.line()?.ok_or("Stitched track truncated")?)
                    .map_err(|e| e.to_string())?;
            if row["frame_index"].as_u64() != Some(frame) {
                return Err("Stitched frame order mismatch".into());
            }
            let players = row["tracked_players"]
                .as_array_mut()
                .ok_or("Missing stitched players")?;
            let before = players.len();
            players.retain(|p| {
                p["track_id"]
                    .as_u64()
                    .is_some_and(|id| !rejected.contains(&id))
            });
            removed += (before - players.len()) as u64;
            if let Some(diagnostics) = row["new_track_diagnostics"].as_array_mut() {
                diagnostics.retain(|p| {
                    p["track_id"]
                        .as_u64()
                        .is_none_or(|id| !rejected.contains(&id))
                });
            }
            let mut bytes = serde_json::to_vec(&row).map_err(|e| e.to_string())?;
            bytes.push(b'\n');
            if bytes.len() > MAX_ROW_BYTES {
                return Err("Stitched row exceeds limit".into());
            }
            writer.write_all(&bytes).map_err(|e| e.to_string())?;
            sha.update(&bytes);
        }
        if reader.line()?.is_some() || reader.finish() != expected_sha {
            return Err("Stitched track SHA mismatch".into());
        }
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
        drop(writer);
        if removed != pruned.short_detections + pruned.static_detections {
            return Err("Pruned Track ID counts changed".into());
        }
        fs::rename(&tmp, source).map_err(|e| e.to_string())?;
        pruned.sha = format!("{:x}", sha.finalize());
        Ok(pruned)
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}
pub(super) fn stitch(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    scope: Option<&PlayerFrameScope>,
    interpolate_players: bool,
) -> Result<(String, u64, u64), String> {
    let Handoffs {
        aliases,
        occlusions,
    } = find_links(source, expected_sha, frames, scope)?;
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
                if interpolate_players
                    && player["detection"]["score"]
                        .as_f64()
                        .is_some_and(|score| score < 0.35)
                    && occlusions.get(&id).is_some_and(|bridges| {
                        bridges
                            .iter()
                            .any(|bridge| bridge.left < frame_index && frame_index < bridge.right)
                    })
                {
                    continue;
                }
                if interpolate_players {
                    if let Some(bridge) = occlusions
                        .get(&id)
                        .into_iter()
                        .flatten()
                        .find(|bridge| bridge.right == frame_index)
                    {
                        player["occlusion_bridge"] =
                            json!({"left":bridge.left,"occluder":bridge.occluder});
                    }
                }
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
            rallies: selected.clone(),
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

    #[test]
    fn prune_stationary_net_posts_but_keep_stationary_players() {
        let root = std::env::temp_dir().join(format!("volley-posts-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("tracks.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..180 {
            let jitter = (frame % 5) as f64 * 0.0004;
            let movement = (frame % 30) as f64 * 0.001;
            let players = [
                (
                    4,
                    [0.80 - movement * 0.4, 0.47, 0.84 + movement * 0.3, 0.66],
                ),
                (5, [0.283, 0.451 + jitter, 0.302 + jitter, 0.595]),
                (6, [0.661, 0.444 + jitter, 0.676, 0.581 + jitter]),
                (7, [0.50 + movement, 0.44, 0.515 + movement, 0.58]),
                (8, [0.10, 0.30, 0.15, 0.50]),
            ]
            .into_iter()
            .map(|(id, bbox)| {
                json!({
                    "track_id":id, "detection":candidate(frame,bbox,None)
                })
            })
            .collect::<Vec<_>>();
            serde_json::to_writer(
                &mut bytes,
                &json!({
                    "frame_index":frame,"tracked_players":players
                }),
            )
            .unwrap();
            bytes.push(b'\n');
        }
        fs::write(&source, &bytes).unwrap();
        let pruned = prune_unreliable(
            &source,
            &format!("{:x}", Sha256::digest(&bytes)),
            180,
            12,
            None,
        )
        .unwrap();
        assert_eq!(pruned.static_tracks, 2);
        assert_eq!(pruned.static_detections, 360);
        assert_eq!(pruned.short_tracks, 0);
        let result = fs::read_to_string(&source).unwrap();
        for line in result.lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            let ids = row["tracked_players"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["track_id"].as_u64().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(ids, vec![4, 7, 8]);
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn visible_player_keeps_id_when_occluded_neighbor_reappears() {
        for reverse in [false, true] {
            let at = if reverse { 40 } else { 60 };
            let prior = if reverse { 41 } else { 59 };
            let missing = if reverse { 55 } else { 45 };
            let near = RosterObservation {
                id: 4,
                bbox: [0.44, 0.46, 0.482, 0.615],
                score: 0.9,
            };
            let far = RosterObservation {
                id: 7,
                bbox: [0.468, 0.44, 0.49, 0.52],
                score: 0.8,
            };
            let mut slots = [RosterSlot::new(near, prior), RosterSlot::new(far, missing)];
            slots[1].velocity = [if reverse { -0.001 } else { 0.001 }, 0.0];
            let observations = [
                RosterObservation {
                    id: 7,
                    bbox: [0.447, 0.465, 0.486, 0.617],
                    score: 0.88,
                },
                RosterObservation {
                    id: 4,
                    bbox: [0.423, 0.448, 0.455, 0.536],
                    score: 0.61,
                },
            ];
            let assigned = assign_roster_frame(&observations, &mut slots, at);
            assert_eq!(
                assigned[0],
                Some(4),
                "reverse={reverse}, assigned={assigned:?}"
            );
            assert_eq!(assigned[1], Some(7));
            assert_eq!(slots[0].frame, at);
            assert!(slots[0].bbox[3] > 0.60);
        }
    }
    #[test]
    fn rally_roster_does_not_force_an_independent_far_box_into_existing_id() {
        let frames = (0..70)
            .map(|frame| {
                let mut row = vec![
                    RosterObservation {
                        id: 1,
                        bbox: [0.2, 0.3, 0.25, 0.6],
                        score: 0.9,
                    },
                    RosterObservation {
                        id: 2,
                        bbox: [0.6, 0.3, 0.65, 0.6],
                        score: 0.9,
                    },
                ];
                if (30..45).contains(&frame) {
                    row.push(RosterObservation {
                        id: 3,
                        bbox: [0.85, 0.3, 0.90, 0.6],
                        score: 0.8,
                    });
                }
                row
            })
            .collect::<Vec<_>>();
        let owners = roster_plan(&frames, 0, 69).unwrap();
        for (frame, assigned) in owners.iter().enumerate() {
            assert_eq!(assigned[0], Some(1));
            assert_eq!(assigned[1], Some(2));
            if (30..45).contains(&frame) {
                assert_eq!(assigned[2], None);
            }
        }
    }

    #[test]
    fn roster_window_prevents_interpolation_of_id_across_another_rally() {
        let path =
            std::env::temp_dir().join(format!("volley-roster-boundary-{}.jsonl", Uuid::new_v4()));
        let mut raw = Vec::new();
        for frame in 0..20_u64 {
            let window = if frame < 5 {
                0
            } else if frame < 15 {
                5
            } else {
                15
            };
            let players = if !(5..15).contains(&frame) {
                vec![json!({"track_id":1,"detection":candidate(frame,[0.2,0.3,0.25,0.6],None)})]
            } else {
                vec![json!({"track_id":2,"detection":candidate(frame,[0.6,0.3,0.65,0.6],None)})]
            };
            serde_json::to_writer(&mut raw,&json!({"frame_index":frame,"rally_roster_window":window,"tracked_players":players})).unwrap();
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let (_, filled) =
            super::super::fill_confirmed_track_gaps(&path, &sha, 20, None, None).unwrap();
        assert_eq!(filled, 0);
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn rally_roster_reconnects_fragments_and_keeps_only_two_ids() {
        let path = std::env::temp_dir().join(format!("volley-roster-{}.jsonl", Uuid::new_v4()));
        let mut raw = Vec::new();
        for frame in 0..100_u64 {
            let x = 0.25 + frame as f64 * 0.0004;
            let mut players =
                vec![json!({"track_id":2,"detection":candidate(frame,[0.65,0.3,0.70,0.55],None)})];
            if !(50..65).contains(&frame) {
                let id = if frame < 50 { 1 } else { 8 };
                players.push(json!({"track_id":id,"detection":candidate(frame,[x-0.02,0.30,x+0.02,0.52],None)}));
            }
            if (70..85).contains(&frame) {
                players.push(json!({"track_id":9,"detection":candidate(frame,[x-0.015,0.39,x+0.015,0.52],None)}));
            }
            serde_json::to_writer(
                &mut raw,
                &json!({"frame_index":frame,"tracked_players":players}),
            )
            .unwrap();
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let scope = rally_scope(vec![(0, 99)]);
        let (sha, _, _) = stitch(&path, &sha, 100, Some(&scope), true).unwrap();
        let result = reconcile_rally_rosters(&path, &sha, 100, &scope, None).unwrap();
        let sha = result.sha;
        let (_, _) =
            super::super::fill_confirmed_track_gaps(&path, &sha, 100, Some(&scope), None).unwrap();
        let rows = fs::read_to_string(&path).unwrap();
        let mut ids = BTreeSet::new();
        for line in rows.lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            let players = row["tracked_players"].as_array().unwrap();
            assert_eq!(players.len(), 2, "frame {}", row["frame_index"]);
            ids.extend(players.iter().map(|p| p["track_id"].as_u64().unwrap()));
        }
        assert_eq!(ids, BTreeSet::from([1, 2]));
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn single_frame_partial_pose_handoff_keeps_first_id_across_rally_context() {
        for distinct in [false, true] {
            let path = std::env::temp_dir()
                .join(format!("volley-partial-handoff-{}.jsonl", Uuid::new_v4()));
            let mut raw = Vec::new();
            for frame in 0..90_u64 {
                let mut players = Vec::new();
                for (id, start, end) in [(5, 0, 25), (7, 25, 89)] {
                    if !(start..=end).contains(&frame) {
                        continue;
                    }
                    let x = 0.43 + if distinct && id == 7 { 0.035 } else { 0.0 };
                    let joints: [[f64; 3]; 17] = std::array::from_fn(|i| {
                        [
                            x + (i % 2) as f64 * 0.010,
                            0.44 + (i / 2) as f64 * 0.009,
                            if i < 11 { 0.43 } else { 0.20 },
                        ]
                    });
                    players.push(json!({"track_id":id,"pose":joints,"measurement_source":"rtmpose_rgb",
                        "detection":candidate(frame, [x-0.01,0.43,x+0.023,if id==5 {0.49} else {0.56}], Some([x,0.44]))}));
                }
                serde_json::to_writer(
                    &mut raw,
                    &json!({"frame_index":frame,"tracked_players":players}),
                )
                .unwrap();
                raw.push(b'\n');
            }
            fs::write(&path, &raw).unwrap();
            let sha = format!("{:x}", Sha256::digest(&raw));
            let scope = rally_scope(vec![(0, 49), (60, 89)]);
            let (_, merged, removed) = stitch(&path, &sha, 90, Some(&scope), true).unwrap();
            assert_eq!((merged, removed), if distinct { (0, 0) } else { (1, 1) });
            if !distinct {
                let rows = fs::read_to_string(&path).unwrap();
                for line in rows.lines() {
                    let row: Value = serde_json::from_str(line).unwrap();
                    assert_eq!(row["tracked_players"].as_array().unwrap().len(), 1);
                    assert_eq!(row["tracked_players"][0]["track_id"], 5);
                }
            }
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn stable_box_motion_reconnects_without_joints_or_head() {
        for gap in [0_u64, 1, 20] {
            for distinct in [false, true] {
                let path = std::env::temp_dir()
                    .join(format!("volley-box-motion-{}.jsonl", Uuid::new_v4()));
                let mut raw = Vec::new();
                for frame in 0..80_u64 {
                    let x = 0.40 + frame as f64 * 0.001;
                    let mut players = Vec::new();
                    if frame <= 25 {
                        let top = if frame >= 23 { 0.50 } else { 0.46 };
                        players.push(json!({"track_id":15,
                            "detection":candidate(frame,[x-0.015,top,x+0.015,0.58],None)}));
                    }
                    if frame >= 25 + gap {
                        let x = x + if distinct { 0.050 } else { 0.0 };
                        players.push(json!({"track_id":21,
                            "detection":candidate(frame,[x-0.015,0.46,x+0.015,0.58],None)}));
                    }
                    serde_json::to_writer(
                        &mut raw,
                        &json!({"frame_index":frame,"tracked_players":players}),
                    )
                    .unwrap();
                    raw.push(b'\n');
                }
                fs::write(&path, &raw).unwrap();
                let sha = format!("{:x}", Sha256::digest(&raw));
                let scope = rally_scope(vec![(0, 79)]);
                let expected = if distinct {
                    BTreeMap::new()
                } else {
                    BTreeMap::from([(21, 15)])
                };
                assert_eq!(
                    find_handoffs(&path, &sha, 80, Some(&scope)).unwrap(),
                    expected,
                    "gap={gap}, distinct={distinct}"
                );
                fs::remove_file(path).unwrap();
            }
        }
    }
    #[test]
    fn moving_pose_gap_reconnects_and_interpolates_under_original_id() {
        let path = std::env::temp_dir().join(format!("volley-moving-gap-{}.jsonl", Uuid::new_v4()));
        let mut raw = Vec::new();
        for frame in 0..85_u64 {
            let id = if frame <= 24 {
                Some(5)
            } else if frame >= 55 {
                Some(7)
            } else {
                None
            };
            let x = 0.30 + frame as f64 * 0.002;
            let players = id
                .map(|id| {
                    let joints: [[f64; 3]; 17] = std::array::from_fn(|i| {
                        [
                            x + (i % 2) as f64 * 0.018,
                            0.35 + (i / 2) as f64 * 0.025,
                            0.85,
                        ]
                    });
                    vec![
                        json!({"track_id":id,"pose":joints,"measurement_source":"rtmpose_rgb",
                    "detection":candidate(frame,[x-0.02,0.33,x+0.04,0.65],Some([x,0.35]))}),
                    ]
                })
                .unwrap_or_default();
            serde_json::to_writer(
                &mut raw,
                &json!({"frame_index":frame,"tracked_players":players}),
            )
            .unwrap();
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let scope = rally_scope(vec![(0, 84)]);
        let (sha, merged, removed) = stitch(&path, &sha, 85, Some(&scope), true).unwrap();
        assert_eq!((merged, removed), (1, 0));
        let (_, filled) =
            super::super::fill_confirmed_track_gaps(&path, &sha, 85, Some(&scope), None).unwrap();
        assert_eq!(filled, 30);
        let rows = fs::read_to_string(&path).unwrap();
        for line in rows.lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            assert_eq!(row["tracked_players"].as_array().unwrap().len(), 1);
            assert_eq!(row["tracked_players"][0]["track_id"], 5);
            let frame = row["frame_index"].as_u64().unwrap();
            if (25..55).contains(&frame) {
                assert_eq!(row["interpolated_track_ids"], json!([5]));
                assert_eq!(
                    row["tracked_players"][0]["measurement_source"],
                    "interpolated_track_bbox"
                );
            }
        }
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn long_overlap_is_not_merged_by_joint_agreement_alone() {
        for distinct in [false, true] {
            let path =
                std::env::temp_dir().join(format!("volley-shared-joints-{}.jsonl", Uuid::new_v4()));
            let mut raw = Vec::new();
            for frame in 0..80_u64 {
                let mut players = Vec::new();
                for (id, start, end) in [(2, 0, 49), (10, 20, 79)] {
                    if !(start..=end).contains(&frame) {
                        continue;
                    }
                    let mut joints = [[0.0; 3]; 17];
                    for (i, joint) in joints.iter_mut().enumerate().skip(7) {
                        let shift = if id == 10 && distinct { 0.07 } else { 0.0 };
                        *joint = [
                            0.55 + shift + (i % 2) as f64 * 0.018,
                            0.46 + (i / 2) as f64 * 0.012,
                            0.9,
                        ];
                    }
                    if id == 2 {
                        joints[11][2] = 0.0;
                        joints[12][2] = 0.0;
                    }
                    players.push(json!({"track_id":id,"pose":joints,"measurement_source":"rtmpose_rgb",
                        "detection":candidate(frame, if id == 2 {[0.53,0.48,0.58,0.59]} else {[0.52,0.40,0.62,0.63]}, None)}));
                }
                serde_json::to_writer(
                    &mut raw,
                    &json!({"frame_index":frame,"tracked_players":players}),
                )
                .unwrap();
                raw.push(b'\n');
            }
            fs::write(&path, &raw).unwrap();
            let sha = format!("{:x}", Sha256::digest(&raw));
            let rally = rally_scope(vec![(0, 79)]);
            let (_, merged, dropped) = stitch(&path, &sha, 80, Some(&rally), true).unwrap();
            assert_eq!((merged, dropped), (0, 0));
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn short_occlusion_reconnects_without_pose_and_interpolates_moving_boxes() {
        let path = std::env::temp_dir().join(format!("volley-occlusion-{}.jsonl", Uuid::new_v4()));
        let frames = 60;
        let mut raw = Vec::new();
        for frame in 0..frames {
            let mut players = vec![json!({"track_id":4,"detection":
                candidate(frame,[0.444,0.464,0.485,0.620],None)})];
            let endpoint = match frame {
                0..=19 => Some((7, [0.468, 0.439, 0.490, 0.519], 0.67)),
                20..=22 => Some((7, [0.466, 0.464, 0.492, 0.528], 0.12)),
                32 => Some((14, [0.431, 0.443, 0.459, 0.538], 0.295)),
                34..=59 => Some((14, [0.423, 0.448, 0.455, 0.536], 0.61)),
                _ => None,
            };
            if let Some((id, bbox, score)) = endpoint {
                let mut detection = candidate(frame, bbox, None);
                detection["score"] = json!(score);
                players.push(json!({"track_id":id,"detection":detection}));
            }
            serde_json::to_writer(
                &mut raw,
                &json!({"frame_index":frame,"tracked_players":players}),
            )
            .unwrap();
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let rally = rally_scope(vec![(0, 59)]);
        let mut split = rally_scope(vec![(0, 59)]);
        split.rallies = vec![(0, 25), (29, 59)];
        assert!(find_handoffs(&path, &sha, frames, Some(&split))
            .unwrap()
            .is_empty());
        let no_blocker =
            std::env::temp_dir().join(format!("volley-no-occlusion-{}.jsonl", Uuid::new_v4()));
        let mut separate = Vec::new();
        for line in raw
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
        {
            let mut row: Value = serde_json::from_slice(line).unwrap();
            row["tracked_players"]
                .as_array_mut()
                .unwrap()
                .retain(|p| p["track_id"] != 4);
            serde_json::to_writer(&mut separate, &row).unwrap();
            separate.push(b'\n');
        }
        let separate_sha = format!("{:x}", Sha256::digest(&separate));
        fs::write(&no_blocker, &separate).unwrap();
        assert!(
            find_handoffs(&no_blocker, &separate_sha, frames, Some(&rally))
                .unwrap()
                .is_empty()
        );
        fs::remove_file(no_blocker).unwrap();
        let (stitched, count, _) = stitch(&path, &sha, frames, Some(&rally), true).unwrap();
        assert_eq!(count, 1);
        super::super::fill_confirmed_track_gaps(&path, &stitched, frames, Some(&rally), None)
            .unwrap();
        for line in fs::read_to_string(&path).unwrap().lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            let frame = row["frame_index"].as_u64().unwrap();
            let players = row["tracked_players"].as_array().unwrap();
            assert!(players.iter().any(|p| p["track_id"] == 4));
            assert!(players.iter().all(|p| p["track_id"] != 14));
            let player = players.iter().find(|p| p["track_id"] == 7).unwrap();
            if (20..34).contains(&frame) {
                assert_eq!(player["measurement_source"], "interpolated_track_bbox");
                assert_eq!(player["occluding_track_id"], 4);
                if frame == 30 {
                    let x = player["detection"]["bbox_xyxy_norm"][0].as_f64().unwrap();
                    assert!(x > 0.423 && x < 0.468);
                }
            }
        }
        fs::remove_file(path).unwrap();
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
                let bbox = if different_heads {
                    [bbox[0] + 0.05, bbox[1], bbox[2] + 0.05, bbox[3]]
                } else {
                    bbox
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
    fn stitches_id3_to_id7_before_pose_using_complementary_crops() {
        let (path, sha, count) = fixture(false, false, false);
        assert_eq!(
            find_handoffs(&path, &sha, count, Some(&rally_scope(vec![(0, count - 1)])))
                .unwrap()
                .get(&7),
            Some(&3)
        );
        let (rewritten, merged, dropped) = stitch(
            &path,
            &sha,
            count,
            Some(&rally_scope(vec![(0, count - 1)])),
            true,
        )
        .unwrap();
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
    fn separate_motion_cannot_be_stitched() {
        let (path, sha, frames) = fixture(true, false, false);
        let scope = rally_scope(vec![(0, frames - 1)]);
        assert!(find_handoffs(&path, &sha, frames, Some(&scope))
            .unwrap()
            .is_empty());
        assert_eq!(
            stitch(&path, &sha, frames, Some(&scope), true).unwrap(),
            (sha, 0, 0)
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn sustained_independent_overlap_cannot_be_stitched() {
        let (path, sha, frames) = fixture(false, true, false);
        assert!(find_handoffs(&path, &sha, frames, None).unwrap().is_empty());
        assert_eq!(
            stitch(&path, &sha, frames, None, true).unwrap(),
            (sha, 0, 0)
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn unscoped_fragments_are_not_stitched_across_unknown_rallies() {
        let (path, sha, frames) = fixture(false, false, true);
        assert!(find_handoffs(&path, &sha, frames, None).unwrap().is_empty());
        assert_eq!(
            stitch(&path, &sha, frames, None, true).unwrap(),
            (sha, 0, 0)
        );
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
            stitch(&path, &sha, 48, Some(&separated), true).unwrap(),
            (sha.clone(), 0, 0)
        );
        let rally = rally_scope(vec![(0, 47)]);
        let (updated, merged, removed) = stitch(&path, &sha, 48, Some(&rally), true).unwrap();
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
    fn stable_box_edge_stitches_different_crops_only_within_one_rally() {
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
                    "detection":candidate(frame,[0.515,0.35,0.565,0.53],None)})]
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
        assert_eq!(
            stitch(&path, &sha, 48, Some(&rally), true).unwrap(),
            (sha, 0, 0)
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn torso_handoffs_chain_to_first_id_inside_one_rally() {
        let path = std::env::temp_dir().join(format!("volley-chain-{}.jsonl", Uuid::new_v4()));
        let mut raw = Vec::new();
        for frame in 0..80_u64 {
            let mut players = Vec::new();
            for (id, start, end, x) in [
                (3, 0, 17, 0.54),
                (7, 21, 38, 0.54),
                (12, 42, 59, 0.54),
                (19, 62, 79, 0.54),
                (25, 0, 79, 0.65),
            ] {
                if (start..=end).contains(&frame) {
                    let joints = [[x, 0.47, 0.9]; 17];
                    players.push(json!({
                        "track_id":id,"pose":joints,
                        "detection":candidate(frame,[x-0.04,0.35,x+0.04,0.67],None)
                    }));
                }
            }
            serde_json::to_writer(
                &mut raw,
                &json!({"frame_index":frame,"tracked_players":players}),
            )
            .unwrap();
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let rally = rally_scope(vec![(0, 79)]);
        let (updated, merged, removed) = stitch(&path, &sha, 80, Some(&rally), true).unwrap();
        assert_eq!((merged, removed), (3, 0));
        assert_ne!(updated, sha);
        for line in fs::read_to_string(&path).unwrap().lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            for player in row["tracked_players"].as_array().unwrap() {
                assert!(matches!(player["track_id"].as_u64(), Some(3 | 25)));
            }
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn hash_mismatch_never_modifies_source() {
        let (path, sha, frames) = fixture(false, false, false);
        let actual = fs::read(&path).unwrap();
        assert!(stitch(&path, &"0".repeat(64), frames, None, true).is_err());
        assert_eq!(actual, fs::read(&path).unwrap());
        fs::remove_file(path).unwrap();
        assert_eq!(sha.len(), 64);
    }
}
