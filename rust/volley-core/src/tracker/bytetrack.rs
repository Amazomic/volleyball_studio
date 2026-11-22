//! Native Rust ByteTrack data association for immutable RF-DETR / RTMPose
//! observations. Two-stage high/low confidence matching, Kalman XYAH prediction,
//! Hungarian one-to-one assignment and bounded lost-track reactivation.
//!
//! ByteTrack reference: FoundationVision/ByteTrack, yolox/tracker/byte_tracker.py.
//! No synthetic/interpolated coordinate ever counts as an observed detection.

use super::{
    association, same_head_body_duplicate, NewTrackDiagnostic, PoseEvidence, TrackedFrame,
    TrackedPlayer, TrackingOptions,
};
use crate::{bbox_iou, fusion::Candidate};
use std::collections::{BTreeMap, BTreeSet};

/// Preserve weak RF-DETR detections solely for second-stage continuation.
/// The legacy 0.28 cutoff removed the very measurements ByteTrack needs.
pub const LOW_DETECTION_SCORE: f64 = 0.10;
// Preserve the former 0.28 activation sensitivity; only weaker boxes
// are restricted to continuation of an already observed player.
pub const HIGH_DETECTION_SCORE: f64 = 0.28;
const HIGH_IOU: f64 = 0.12;
const LOW_IOU: f64 = 0.20;
const UNCONFIRMED_IOU: f64 = 0.30;
const MAX_KF_TRACKS: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Tentative,
    Tracked,
    Lost,
}

fn xyah(bbox: [f64; 4]) -> [f64; 4] {
    let w = bbox[2] - bbox[0];
    let h = bbox[3] - bbox[1];
    [
        (bbox[0] + bbox[2]) / 2.0,
        (bbox[1] + bbox[3]) / 2.0,
        w / h,
        h,
    ]
}

fn tlbr(state: [f64; 4]) -> [f64; 4] {
    let w = state[2] * state[3];
    [
        state[0] - w / 2.0,
        state[1] - state[3] / 2.0,
        state[0] + w / 2.0,
        state[1] + state[3] / 2.0,
    ]
}

/// Kalman 8-state [cx, cy, aspect, h, vx, vy, va, vh] matching ByteTrack's
/// XYAH state layout. Noise scales with measured/predicted person height.
#[derive(Clone, Debug)]
struct Kalman {
    mean: [f64; 8],
    cov: [[f64; 8]; 8],
}
impl Kalman {
    fn new(bbox: [f64; 4]) -> Self {
        let xyah = xyah(bbox);
        let mut mean = [0.0; 8];
        mean[..4].copy_from_slice(&xyah);
        let h = xyah[3].max(0.01);
        let scale = [
            2.0 * h / 20.0,
            2.0 * h / 20.0,
            0.01,
            2.0 * h / 20.0,
            10.0 * h / 160.0,
            10.0 * h / 160.0,
            0.0001,
            10.0 * h / 160.0,
        ];
        let mut cov = [[0.0; 8]; 8];
        for i in 0..8 {
            cov[i][i] = scale[i] * scale[i];
        }
        Self { mean, cov }
    }
    fn predict(&mut self, delta: f64, lost: bool) {
        if lost {
            self.mean[7] = 0.0;
        }
        for i in 0..4 {
            self.mean[i] += delta * self.mean[i + 4];
        }
        let mut f = [[0.0; 8]; 8];
        for (i, row) in f.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        for i in 0..4 {
            f[i][i + 4] = delta;
        }
        let mut multiplied = [[0.0; 8]; 8];
        for (i, row) in multiplied.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                for (k, transform) in f[i].iter().enumerate() {
                    *value += transform * self.cov[k][j];
                }
            }
        }
        let mut next = [[0.0; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    next[i][j] += multiplied[i][k] * f[j][k];
                }
            }
        }
        let h = self.mean[3].abs().max(0.01);
        let pos = [h / 20.0, h / 20.0, 0.01, h / 20.0];
        let vel = [h / 160.0, h / 160.0, 0.0001, h / 160.0];
        for i in 0..4 {
            next[i][i] += pos[i] * pos[i] * delta.max(1.0);
            next[i + 4][i + 4] += vel[i] * vel[i] * delta.max(1.0);
        }
        self.cov = next;
    }
    fn update(&mut self, bbox: [f64; 4]) {
        let observation = xyah(bbox);
        let h = self.mean[3].abs().max(0.01);
        let std = [h / 20.0, h / 20.0, 0.1, h / 20.0];
        let mut innovation_cov = [[0.0; 4]; 4];
        for (i, row) in innovation_cov.iter_mut().enumerate() {
            row.copy_from_slice(&self.cov[i][..4]);
            row[i] += std[i] * std[i];
        }
        let Some(inv) = invert4(innovation_cov) else {
            *self = Self::new(bbox);
            return;
        };
        let prior = self.cov;
        let mut gain = [[0.0; 4]; 8];
        for (i, row) in gain.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                for (k, inverse) in inv.iter().enumerate() {
                    *value += prior[i][k] * inverse[j];
                }
            }
        }
        let difference = std::array::from_fn::<_, 4, _>(|j| observation[j] - self.mean[j]);
        for (mean, gains) in self.mean.iter_mut().zip(&gain) {
            for (gain, value) in gains.iter().zip(&difference) {
                *mean += gain * value;
            }
        }
        for (i, (row, gains)) in self.cov.iter_mut().zip(&gain).enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                for (gain, prior_row) in gains.iter().zip(&prior[..4]) {
                    *value -= gain * prior_row[j];
                }
            }
            row[i] = row[i].max(1e-12);
        }
    }
    fn bbox(&self) -> [f64; 4] {
        tlbr([
            self.mean[0],
            self.mean[1],
            self.mean[2],
            self.mean[3].max(0.005),
        ])
    }
}

fn invert4(mut source: [[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut inverse = [[0.0; 4]; 4];
    for (i, line) in inverse.iter_mut().enumerate() {
        line[i] = 1.0;
    }
    for column in 0..4 {
        let pivot = (column..4).max_by(|a, b| {
            source[*a][column]
                .abs()
                .total_cmp(&source[*b][column].abs())
        })?;
        if source[pivot][column].abs() <= 1e-14 {
            return None;
        }
        source.swap(column, pivot);
        inverse.swap(column, pivot);
        let divider = source[column][column];
        for j in 0..4 {
            source[column][j] /= divider;
            inverse[column][j] /= divider;
        }
        for row in 0..4 {
            if row == column {
                continue;
            }
            let scale = source[row][column];
            for j in 0..4 {
                source[row][j] -= scale * source[column][j];
                inverse[row][j] -= scale * inverse[column][j];
            }
        }
    }
    Some(inverse)
}

#[derive(Clone, Debug)]
struct Track {
    id: u64,
    kalman: Kalman,
    last_frame: u64,
    predicted_at: u64,
    status: Status,
    observations: u64,
}

#[derive(Clone, Debug)]
struct Observation {
    box_: Candidate,
    pose: Option<PoseEvidence>,
}

type AssociationOutcome = (Vec<(usize, usize)>, Vec<usize>, Vec<usize>, u64, Vec<usize>);

pub struct ByteTracker {
    config: TrackingOptions,
    tracks: Vec<Track>,
    next_id: u64,
    previous: Option<u64>,
    seen: u64,
    skipped: u64,
    ambiguous: u64,
    low_confidence_matches: u64,
    lost_low_confidence_matches: u64,
    separated_pairs: BTreeMap<(u64, u64), (u64, u8)>,
}

impl ByteTracker {
    pub fn new(config: TrackingOptions) -> Result<Self, String> {
        let config = config.validate()?;
        if config.min_player_score < LOW_DETECTION_SCORE {
            return Err("ByteTrack minimum detector score must be >= 0.10".into());
        }
        Ok(Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
            previous: None,
            seen: 0,
            skipped: 0,
            ambiguous: 0,
            low_confidence_matches: 0,
            lost_low_confidence_matches: 0,
            separated_pairs: BTreeMap::new(),
        })
    }
    fn independently_tracked_bodies(
        &self,
        frame: u64,
        first: &Candidate,
        second: &Candidate,
    ) -> bool {
        let recent = |track: &Track| {
            track.status == Status::Tracked
                && track.observations >= 2
                && frame.saturating_sub(track.last_frame) <= 2
        };
        self.tracks.iter().filter(|track| recent(track)).any(|a| {
            self.tracks.iter().filter(|track| recent(track)).any(|b| {
                if a.id == b.id
                    || !self
                        .separated_pairs
                        .get(&(a.id.min(b.id), a.id.max(b.id)))
                        .is_some_and(|(last, count)| {
                            *count >= 3 && frame.saturating_sub(*last) <= 60
                        })
                {
                    return false;
                }
                let (a_box, b_box) = (a.kalman.bbox(), b.kalman.bbox());
                (bbox_iou(a_box, first.bbox_xyxy_norm) >= 0.12
                    && bbox_iou(b_box, second.bbox_xyxy_norm) >= 0.12)
                    || (bbox_iou(a_box, second.bbox_xyxy_norm) >= 0.12
                        && bbox_iou(b_box, first.bbox_xyxy_norm) >= 0.12)
            })
        })
    }
    fn remember_separate_bodies(&mut self, frame: u64, observed: &[TrackedPlayer]) {
        for (index, first) in observed.iter().enumerate() {
            for second in observed.iter().skip(index + 1) {
                let (a, b) = (
                    first.detection.bbox_xyxy_norm,
                    second.detection.bbox_xyxy_norm,
                );
                let shorter = (a[2] - a[0]).min(b[2] - b[0]);
                let distinct_heads = match (
                    first.detection.head_point_norm,
                    second.detection.head_point_norm,
                ) {
                    (Some(head_a), Some(head_b)) => {
                        (head_a[0] - head_b[0]).hypot(head_a[1] - head_b[1])
                            > (0.7 * shorter).max(0.018)
                    }
                    _ => {
                        let center_a = (a[0] + a[2]) * 0.5;
                        let center_b = (b[0] + b[2]) * 0.5;
                        (center_a - center_b).abs() > (0.8 * shorter).max(0.020)
                    }
                };
                if bbox_iou(a, b) <= 0.25 && distinct_heads {
                    let pair = (
                        first.track_id.min(second.track_id),
                        first.track_id.max(second.track_id),
                    );
                    let evidence = self.separated_pairs.entry(pair).or_insert((frame, 0));
                    evidence.1 = if frame.saturating_sub(evidence.0) <= 30 {
                        evidence.1.saturating_add(1)
                    } else {
                        1
                    };
                    evidence.0 = frame;
                }
            }
        }
        let active = self
            .tracks
            .iter()
            .map(|track| track.id)
            .collect::<BTreeSet<_>>();
        self.separated_pairs.retain(|&(a, b), (last, _)| {
            frame.saturating_sub(*last) <= 60 && active.contains(&a) && active.contains(&b)
        });
    }
    pub fn process_frame_with_poses(
        &mut self,
        frame: u64,
        raw: &[Candidate],
        pose: &[Option<PoseEvidence>],
    ) -> Result<TrackedFrame, String> {
        if self.previous.is_some_and(|last| frame <= last) {
            return Err("ByteTrack input frame index must strictly increase".into());
        }
        if !pose.is_empty() && pose.len() != raw.len() {
            return Err("ByteTrack pose/candidate count mismatch".into());
        }
        if raw.len() > 128 || pose.iter().flatten().any(|p| !p.valid()) {
            return Err("ByteTrack input exceeds verified candidate limit".into());
        }
        if raw.iter().any(|c| {
            c.frame_index != frame
                || c.class_name != "player"
                || c.query_index.is_none()
                || !c.score.is_finite()
                || !(0.0..=1.0).contains(&c.score)
                || !valid_bbox(c.bbox_xyxy_norm)
        }) {
            return Err("Invalid observed ByteTrack candidate".into());
        }
        self.tracks
            .retain(|t| frame.saturating_sub(t.last_frame) <= self.config.max_gap_frames);
        for track in &mut self.tracks {
            if track.predicted_at < frame {
                track.kalman.predict(
                    (frame - track.predicted_at) as f64,
                    track.status == Status::Lost,
                );
                track.predicted_at = frame;
            }
        }
        let mut sorted = raw
            .iter()
            .enumerate()
            .filter(|(_, c)| c.score >= LOW_DETECTION_SCORE)
            .map(|(i, c)| Observation {
                box_: c.clone(),
                pose: pose.get(i).copied().flatten(),
            })
            .collect::<Vec<_>>();
        sorted.sort_by(|a, b| {
            b.box_
                .score
                .total_cmp(&a.box_.score)
                .then_with(|| a.box_.query_index.cmp(&b.box_.query_index))
        });
        let mut retained: Vec<Observation> = Vec::new();
        for item in sorted {
            if retained.iter().all(|other| {
                let duplicate = bbox_iou(other.box_.bbox_xyxy_norm, item.box_.bbox_xyxy_norm)
                    >= self.config.nms_iou
                    || same_head_body_duplicate(&other.box_, &item.box_);
                !duplicate || self.independently_tracked_bodies(frame, &other.box_, &item.box_)
            }) {
                retained.push(item);
            }
        }
        self.seen += retained.len() as u64;
        self.skipped += (raw.len() - retained.len()) as u64;
        let mut high = Vec::new();
        let mut low = Vec::new();
        for item in retained {
            if item.box_.score >= HIGH_DETECTION_SCORE {
                high.push(item);
            } else {
                low.push(item);
            }
        }

        let previous_predictions = self
            .tracks
            .iter()
            .map(|track| (track.id, track.kalman.bbox(), track.last_frame))
            .collect::<Vec<_>>();
        let mut emitted = Vec::<TrackedPlayer>::new();
        let mut new_track_diagnostics = Vec::new();
        // Stage 1: match high confidence to Kalman-predicted tracked/lost IDs.
        let all_ids: Vec<usize> = (0..self.tracks.len()).collect();
        let (matches_high, unmatched_high, unmatched_tracks, ambiguous_high, blocked_high) =
            self.associate(frame, &high, &all_ids, HIGH_IOU);
        self.ambiguous += ambiguous_high;
        for (d, t) in matches_high {
            self.observe(frame, &high[d], t, &mut emitted);
        }
        let recently_tracked = unmatched_tracks
            .iter()
            .copied()
            .filter(|&t| {
                self.tracks[t].status == Status::Tracked
                    && frame.saturating_sub(self.tracks[t].last_frame) <= 15
            })
            .collect::<Vec<_>>();
        let (matches_low, unmatched_low, _, ambiguous_low, blocked_low) =
            self.associate(frame, &low, &recently_tracked, LOW_IOU);
        self.ambiguous += ambiguous_low;
        self.low_confidence_matches += matches_low.len() as u64;
        for (d, t) in matches_low {
            self.observe(frame, &low[d], t, &mut emitted);
        }
        if self.config.lost_low_confidence_max_gap_frames > 0 && !unmatched_low.is_empty() {
            let lost_indices = unmatched_tracks
                .into_iter()
                .filter(|&t| {
                    self.tracks[t].status == Status::Lost
                        && frame.saturating_sub(self.tracks[t].last_frame)
                            <= self.config.lost_low_confidence_max_gap_frames
                })
                .collect::<Vec<_>>();
            let weak_remaining = unmatched_low
                .iter()
                .filter(|index| !blocked_low.contains(index))
                .map(|&index| low[index].clone())
                .collect::<Vec<_>>();
            let (recovered, _, _, ambiguous_lost, _) = self.associate(
                frame,
                &weak_remaining,
                &lost_indices,
                self.config.lost_low_confidence_min_iou,
            );
            self.ambiguous += ambiguous_lost;
            self.low_confidence_matches += recovered.len() as u64;
            self.lost_low_confidence_matches += recovered.len() as u64;
            for (d, t) in recovered {
                self.observe(frame, &weak_remaining[d], t, &mut emitted);
            }
        }
        // Only unmatched HIGH detections become new tentative Track IDs.
        let activation = self.config.min_player_score.max(HIGH_DETECTION_SCORE);
        for d in unmatched_high {
            if blocked_high.contains(&d) || high[d].box_.score < activation {
                continue;
            }
            if self.tracks.len() >= MAX_KF_TRACKS {
                return Err("ByteTrack active Track ID limit exceeded".into());
            }
            let id = self.next_id;
            self.next_id = id.checked_add(1).ok_or("ByteTrack Track ID overflow")?;
            let nearest = previous_predictions
                .iter()
                .filter(|(_, bbox, _)| valid_bbox_inferred(*bbox))
                .map(|(track_id, bbox, last)| {
                    (
                        *track_id,
                        bbox_iou(*bbox, high[d].box_.bbox_xyxy_norm),
                        distance(*bbox, high[d].box_.bbox_xyxy_norm),
                        frame.saturating_sub(*last),
                    )
                })
                .max_by(|a, b| a.1.total_cmp(&b.1));
            let reason = match nearest {
                None => "no_previous_track",
                Some((_, overlap, _, _)) if overlap < HIGH_IOU => "predicted_overlap_too_low",
                Some((_, _, movement, gap))
                    if movement > (0.055 + gap as f64 * 0.005).min(0.30) =>
                {
                    "predicted_movement_too_far"
                }
                Some(_) => "no_assignment_after_matching",
            };
            new_track_diagnostics.push(NewTrackDiagnostic {
                track_id: id,
                reason,
                closest_previous_track_id: nearest.map(|entry| entry.0),
                closest_predicted_iou: nearest.map(|entry| entry.1),
                frames_since_last_seen: nearest.map(|entry| entry.3),
            });
            self.tracks.push(Track {
                id,
                kalman: Kalman::new(high[d].box_.bbox_xyxy_norm),
                last_frame: frame,
                predicted_at: frame,
                status: Status::Tentative,
                observations: 1,
            });
            emitted.push(TrackedPlayer {
                track_id: id,
                detection: high[d].box_.clone(),
                is_verified_athlete: false,
            });
        }
        for track in &mut self.tracks {
            if track.last_frame < frame && track.status == Status::Tracked {
                track.status = Status::Lost;
            }
        }
        self.previous = Some(frame);
        self.remember_separate_bodies(frame, &emitted);
        emitted.sort_by_key(|p| p.track_id);
        Ok(TrackedFrame {
            frame_index: frame,
            retained_boxes: emitted.len(),
            rejected_boxes: raw.len().saturating_sub(emitted.len()),
            ambiguous_detections: (ambiguous_high + ambiguous_low) as usize,
            new_track_diagnostics,
            tracks: emitted,
        })
    }
    fn observe(
        &mut self,
        frame: u64,
        input: &Observation,
        index: usize,
        out: &mut Vec<TrackedPlayer>,
    ) {
        let track = &mut self.tracks[index];
        track.kalman.update(input.box_.bbox_xyxy_norm);
        track.last_frame = frame;
        track.status = Status::Tracked;
        track.observations += 1;
        out.push(TrackedPlayer {
            track_id: track.id,
            detection: input.box_.clone(),
            is_verified_athlete: false,
        });
        let _ = input.pose; // RTMPose measurements are preserved by the CPU replay.
    }
    fn associate(
        &self,
        frame: u64,
        observations: &[Observation],
        indices: &[usize],
        min_iou: f64,
    ) -> AssociationOutcome {
        let mut matrix = Vec::with_capacity(observations.len());
        let mut ambiguous = 0;
        let mut blocked = Vec::new();
        let strictness = (self.config.min_assignment_score - 0.43) * 0.7;
        for (index, observation) in observations.iter().enumerate() {
            let scores = indices
                .iter()
                .map(|&ti| {
                    let track = &self.tracks[ti];
                    let predicted = track.kalman.bbox();
                    if !valid_bbox_inferred(predicted) {
                        return None;
                    }
                    let overlap = bbox_iou(predicted, observation.box_.bbox_xyxy_norm);
                    let gap = frame.saturating_sub(track.last_frame);
                    // Reject teleportation even if a very uncertain Kalman state
                    // expands enough to overlap another player's bounding box.
                    let travel = distance(predicted, observation.box_.bbox_xyxy_norm);
                    let travel_limit = (0.055 + gap as f64 * 0.005).min(0.30);
                    let baseline = if track.status == Status::Tentative {
                        UNCONFIRMED_IOU
                    } else {
                        min_iou
                    };
                    let threshold = (baseline + strictness).clamp(0.03, 0.90);
                    (overlap >= threshold && travel <= travel_limit).then_some(overlap)
                })
                .collect::<Vec<_>>();
            let mut available = scores.iter().filter_map(|v| *v).collect::<Vec<_>>();
            available.sort_by(|a, b| b.total_cmp(a));
            let is_ambiguous =
                available.len() > 1 && available[0] - available[1] < self.config.ambiguity_margin;
            let independently_tracked_crossing = is_ambiguous
                && observations.iter().enumerate().any(|(other, row)| {
                    other != index
                        && self.independently_tracked_bodies(frame, &observation.box_, &row.box_)
                });
            if is_ambiguous && !independently_tracked_crossing {
                ambiguous += 1;
                blocked.push(matrix.len());
                matrix.push(vec![None; indices.len()]);
            } else {
                matrix.push(scores);
            }
        }
        let assigned = association::assign(&matrix);
        let mut matched = Vec::new();
        let mut pending_det = Vec::new();
        let mut used = vec![false; indices.len()];
        for (di, target) in assigned.into_iter().enumerate() {
            if let Some(col) = target {
                matched.push((di, indices[col]));
                used[col] = true;
            } else {
                pending_det.push(di);
            }
        }
        let unmatched_tracks = indices
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (!used[i]).then_some(id))
            .collect();
        (matched, pending_det, unmatched_tracks, ambiguous, blocked)
    }
    pub fn next_id(&self) -> u64 {
        self.next_id
    }
    pub fn low_confidence_matches(&self) -> u64 {
        self.low_confidence_matches
    }
    pub fn lost_low_confidence_matches(&self) -> u64 {
        self.lost_low_confidence_matches
    }
    pub fn totals(&self) -> (u64, u64, u64) {
        (self.seen, self.skipped, self.ambiguous)
    }
}

fn valid_bbox(b: [f64; 4]) -> bool {
    b.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)) && b[2] > b[0] && b[3] > b[1]
}
fn valid_bbox_inferred(b: [f64; 4]) -> bool {
    b.iter().all(|x| x.is_finite())
        && b[2] > b[0]
        && b[3] > b[1]
        && b[2] - b[0] < 1.0
        && b[3] - b[1] < 1.0
}
fn distance(a: [f64; 4], b: [f64; 4]) -> f64 {
    ((a[0] + a[2] - b[0] - b[2]) * 0.5).hypot((a[1] + a[3] - b[1] - b[3]) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn person(f: u64, q: usize, x: f64, score: f64) -> Candidate {
        Candidate {
            frame_index: f,
            class_name: "player".into(),
            score,
            bbox_xyxy_norm: [x, 0.20, x + 0.1, 0.80],
            query_index: Some(q),
            touch_probability: None,
            head_point_norm: None,
        }
    }
    fn tracker() -> ByteTracker {
        ByteTracker::new(TrackingOptions::default()).unwrap()
    }

    #[test]
    fn duplicate_pose_boxes_with_the_same_head_never_mint_an_extra_id() {
        let mut tracker = tracker();
        let before = Candidate {
            bbox_xyxy_norm: [0.519, 0.459, 0.548, 0.519],
            head_point_norm: Some([0.537, 0.472]),
            ..person(0, 0, 0.519, 0.92)
        };
        let id = tracker
            .process_frame_with_poses(0, std::slice::from_ref(&before), &[])
            .unwrap()
            .tracks[0]
            .track_id;
        let small = Candidate {
            frame_index: 1,
            query_index: Some(5),
            bbox_xyxy_norm: [0.5312153, 0.4684438, 0.5457274, 0.5359460],
            head_point_norm: Some([0.542465, 0.474742]),
            score: 0.62,
            ..before.clone()
        };
        let large = Candidate {
            frame_index: 1,
            query_index: Some(4),
            bbox_xyxy_norm: [0.5307123, 0.4664318, 0.5491365, 0.5847691],
            head_point_norm: Some([0.541959, 0.475287]),
            score: 0.90,
            ..before
        };
        let result = tracker
            .process_frame_with_poses(1, &[small, large], &[])
            .unwrap();
        assert_eq!(result.tracks.len(), 1);
        assert_eq!(result.tracks[0].track_id, id);
        assert_eq!(tracker.next_id(), 2);
    }

    #[test]
    fn two_crops_of_one_body_are_not_mistaken_for_previously_separate_players() {
        let mut tracker = tracker();
        let crop = |frame, query: usize, tall: bool, head| Candidate {
            frame_index: frame,
            class_name: "player".into(),
            score: if tall { 0.91 } else { 0.90 },
            query_index: Some(query),
            bbox_xyxy_norm: if tall {
                [0.53, 0.45, 0.55, 0.75]
            } else {
                [0.53, 0.45, 0.55, 0.52]
            },
            head_point_norm: head,
            touch_probability: None,
        };
        for frame in 0..3 {
            let row = tracker
                .process_frame_with_poses(
                    frame,
                    &[crop(frame, 0, false, None), crop(frame, 1, true, None)],
                    &[],
                )
                .unwrap();
            assert_eq!(row.tracks.len(), 2);
        }
        let identified = tracker
            .process_frame_with_poses(
                3,
                &[
                    crop(3, 0, false, Some([0.54, 0.46])),
                    crop(3, 1, true, Some([0.54, 0.46])),
                ],
                &[],
            )
            .unwrap();
        assert_eq!(
            identified.tracks.len(),
            1,
            "Vertical crop-size differences without measured separate heads are not two people"
        );
    }

    #[test]
    fn two_existing_players_are_not_deduplicated_during_a_crossing() {
        let mut tracker = tracker();
        let observed = |frame, query, left: f64, head_x: f64| Candidate {
            frame_index: frame,
            class_name: "player".into(),
            score: 0.90,
            query_index: Some(query),
            bbox_xyxy_norm: [left, 0.25, left + 0.10, 0.70],
            head_point_norm: Some([head_x, 0.26]),
            touch_probability: None,
        };
        let first = tracker
            .process_frame_with_poses(
                0,
                &[observed(0, 0, 0.20, 0.25), observed(0, 1, 0.43, 0.48)],
                &[],
            )
            .unwrap();
        assert_eq!(first.tracks.len(), 2);
        let previous = first.tracks.iter().map(|t| t.track_id).collect::<Vec<_>>();
        for (frame, a, b) in [(1, 0.235, 0.395), (2, 0.27, 0.36)] {
            let row = tracker
                .process_frame_with_poses(
                    frame,
                    &[
                        observed(frame, 0, a, a + 0.05),
                        observed(frame, 1, b, b + 0.05),
                    ],
                    &[],
                )
                .unwrap();
            assert_eq!(row.tracks.len(), 2);
        }
        let crossing_a = observed(3, 0, 0.308, 0.358);
        let crossing_b = observed(3, 1, 0.317, 0.363);
        let joined = tracker
            .process_frame_with_poses(3, &[crossing_a, crossing_b], &[])
            .unwrap();
        assert_eq!(
            joined.tracks.len(),
            2,
            "Two prior distinct tracks may approach with overlapping heads"
        );
        for (frame, a, b) in [(4, 0.35, 0.275), (5, 0.395, 0.235)] {
            let row = tracker
                .process_frame_with_poses(
                    frame,
                    &[
                        observed(frame, 0, a, a + 0.05),
                        observed(frame, 1, b, b + 0.05),
                    ],
                    &[],
                )
                .unwrap();
            assert_eq!(row.tracks.len(), 2);
        }
        assert_eq!(tracker.next_id(), 3);
        let last = tracker
            .process_frame_with_poses(
                6,
                &[observed(6, 0, 0.430, 0.480), observed(6, 1, 0.200, 0.250)],
                &[],
            )
            .unwrap();
        assert_eq!(
            last.tracks.iter().map(|t| t.track_id).collect::<Vec<_>>(),
            previous
        );
        assert_eq!(
            last.tracks
                .iter()
                .find(|t| t.detection.bbox_xyxy_norm[0] > 0.4)
                .unwrap()
                .track_id,
            previous[0],
            "The initially left player must retain its own ID after crossing"
        );
        assert_eq!(
            last.tracks
                .iter()
                .find(|t| t.detection.bbox_xyxy_norm[0] < 0.3)
                .unwrap()
                .track_id,
            previous[1],
            "The initially right player must retain its own ID after crossing"
        );
    }

    #[test]
    fn retains_same_id_after_short_missing_interval() {
        let mut tr = tracker();
        let id = tr
            .process_frame_with_poses(0, &[person(0, 0, 0.20, 0.95)], &[])
            .unwrap()
            .tracks[0]
            .track_id;
        for i in 1..=6 {
            assert!(tr
                .process_frame_with_poses(i, &[], &[])
                .unwrap()
                .tracks
                .is_empty());
        }
        let second = tr
            .process_frame_with_poses(7, &[person(7, 0, 0.205, 0.90)], &[])
            .unwrap();
        assert_eq!(second.tracks[0].track_id, id);
        assert_eq!(tr.next_id(), id + 1);
    }
    #[test]
    fn lost_track_reactivates_from_weak_detection_within_configured_window() {
        let mut tr = tracker();
        let id = tr
            .process_frame_with_poses(0, &[person(0, 0, 0.20, 0.95)], &[])
            .unwrap()
            .tracks[0]
            .track_id;
        tr.process_frame_with_poses(1, &[person(1, 0, 0.20, 0.90)], &[])
            .unwrap();
        for frame in 2..=20 {
            assert!(tr
                .process_frame_with_poses(frame, &[], &[])
                .unwrap()
                .tracks
                .is_empty());
        }
        let restored = tr
            .process_frame_with_poses(21, &[person(21, 0, 0.205, 0.17)], &[])
            .unwrap();
        assert_eq!(restored.tracks.len(), 1);
        assert_eq!(restored.tracks[0].track_id, id);
        assert_eq!(tr.next_id(), id + 1);
        assert_eq!(tr.lost_low_confidence_matches(), 1);
        assert!(restored.new_track_diagnostics.is_empty());
    }

    #[test]
    fn weak_recovery_does_not_reactivate_expired_or_disabled_lost_tracks() {
        for window in [0, 10] {
            let mut tr = ByteTracker::new(TrackingOptions {
                lost_low_confidence_max_gap_frames: window,
                ..TrackingOptions::default()
            })
            .unwrap();
            tr.process_frame_with_poses(0, &[person(0, 0, 0.20, 0.95)], &[])
                .unwrap();
            tr.process_frame_with_poses(1, &[person(1, 0, 0.20, 0.90)], &[])
                .unwrap();
            for frame in 2..=20 {
                tr.process_frame_with_poses(frame, &[], &[]).unwrap();
            }
            assert!(tr
                .process_frame_with_poses(21, &[person(21, 0, 0.20, 0.16)], &[])
                .unwrap()
                .tracks
                .is_empty());
            assert_eq!(tr.lost_low_confidence_matches(), 0);
        }
    }

    #[test]
    fn ambiguous_weak_reappearance_does_not_merge_two_lost_players() {
        let mut tr = tracker();
        tr.process_frame_with_poses(
            0,
            &[person(0, 0, 0.20, 0.95), person(0, 1, 0.24, 0.95)],
            &[],
        )
        .unwrap();
        tr.process_frame_with_poses(
            1,
            &[person(1, 0, 0.20, 0.95), person(1, 1, 0.24, 0.95)],
            &[],
        )
        .unwrap();
        for frame in 2..=16 {
            tr.process_frame_with_poses(frame, &[], &[]).unwrap();
        }
        let ambiguous = tr
            .process_frame_with_poses(17, &[person(17, 0, 0.22, 0.16)], &[])
            .unwrap();
        assert!(ambiguous.tracks.is_empty());
        assert_eq!(tr.next_id(), 3);
        assert_eq!(tr.lost_low_confidence_matches(), 0);
    }

    #[test]
    fn active_player_has_priority_over_lost_player_for_weak_matching() {
        let mut tr = tracker();
        tr.process_frame_with_poses(
            0,
            &[person(0, 0, 0.20, 0.95), person(0, 1, 0.25, 0.95)],
            &[],
        )
        .unwrap();
        let confirmed = tr
            .process_frame_with_poses(
                1,
                &[person(1, 0, 0.20, 0.95), person(1, 1, 0.25, 0.95)],
                &[],
            )
            .unwrap();
        let active_id = confirmed.tracks[1].track_id;
        for frame in 2..=16 {
            tr.process_frame_with_poses(frame, &[person(frame, 1, 0.25, 0.95)], &[])
                .unwrap();
        }
        let weak = tr
            .process_frame_with_poses(17, &[person(17, 0, 0.225, 0.16)], &[])
            .unwrap();
        assert_eq!(weak.tracks.len(), 1);
        assert_eq!(weak.tracks[0].track_id, active_id);
        assert_eq!(tr.lost_low_confidence_matches(), 0);
    }

    #[test]
    fn ambiguous_active_weak_detection_cannot_steal_a_lost_id() {
        let mut tr = tracker();
        for frame in 0..=1 {
            tr.process_frame_with_poses(
                frame,
                &[
                    person(frame, 0, 0.20, 0.95),
                    person(frame, 1, 0.24, 0.94),
                    person(frame, 2, 0.22, 0.93),
                ],
                &[],
            )
            .unwrap();
        }
        for frame in 2..=10 {
            tr.process_frame_with_poses(
                frame,
                &[person(frame, 0, 0.20, 0.95), person(frame, 1, 0.24, 0.94)],
                &[],
            )
            .unwrap();
        }
        let result = tr
            .process_frame_with_poses(11, &[person(11, 0, 0.22, 0.16)], &[])
            .unwrap();
        assert!(result.tracks.is_empty());
        assert_eq!(tr.lost_low_confidence_matches(), 0);
    }

    #[test]
    fn newly_spawned_ids_explain_nearest_failed_prediction() {
        let mut tr = tracker();
        let first = tr
            .process_frame_with_poses(0, &[person(0, 0, 0.20, 0.9)], &[])
            .unwrap();
        assert_eq!(first.new_track_diagnostics[0].reason, "no_previous_track");
        tr.process_frame_with_poses(1, &[person(1, 0, 0.20, 0.9)], &[])
            .unwrap();
        let switched = tr
            .process_frame_with_poses(2, &[person(2, 0, 0.60, 0.9)], &[])
            .unwrap();
        assert_eq!(switched.new_track_diagnostics.len(), 1);
        assert_eq!(
            switched.new_track_diagnostics[0].reason,
            "predicted_overlap_too_low"
        );
        assert_eq!(
            switched.new_track_diagnostics[0].closest_previous_track_id,
            Some(1)
        );
    }

    #[test]
    fn low_confidence_updates_existing_track_without_minting_an_id() {
        let mut tr = tracker();
        let id = tr
            .process_frame_with_poses(0, &[person(0, 0, 0.2, 0.9)], &[])
            .unwrap()
            .tracks[0]
            .track_id;
        tr.process_frame_with_poses(1, &[person(1, 0, 0.205, 0.9)], &[])
            .unwrap();
        for frame in 2..=9 {
            let matched = tr
                .process_frame_with_poses(
                    frame,
                    &[person(frame, 0, 0.205 + 0.001 * frame as f64, 0.17)],
                    &[],
                )
                .unwrap();
            assert_eq!(matched.tracks.len(), 1);
            assert_eq!(matched.tracks[0].track_id, id);
        }
        assert_eq!(tr.next_id(), 2, "Low score must never create new Track IDs");
        assert_eq!(tr.low_confidence_matches(), 8);
    }
    #[test]
    fn sparse_ten_frame_sampling_keeps_an_id_via_second_stage() {
        let mut tr = tracker();
        let first = tr
            .process_frame_with_poses(0, &[person(0, 0, 0.2, 0.91)], &[])
            .unwrap()
            .tracks[0]
            .track_id;
        let confirmed = tr
            .process_frame_with_poses(10, &[person(10, 0, 0.205, 0.89)], &[])
            .unwrap();
        assert_eq!(confirmed.tracks[0].track_id, first);
        for (frame, x) in [(20, 0.21), (30, 0.215), (40, 0.22)] {
            let recovered = tr
                .process_frame_with_poses(frame, &[person(frame, 0, x, 0.12)], &[])
                .unwrap();
            assert_eq!(recovered.tracks[0].track_id, first);
        }
        assert_eq!(tr.low_confidence_matches(), 3);
        assert_eq!(tr.next_id(), 2);
    }

    #[test]
    fn weak_false_positive_does_not_spawn_track_and_lost_track_expires() {
        let mut tr = tracker();
        assert!(tr
            .process_frame_with_poses(0, &[person(0, 0, 0.2, 0.14)], &[])
            .unwrap()
            .tracks
            .is_empty());
        let id = tr
            .process_frame_with_poses(1, &[person(1, 0, 0.2, 0.90)], &[])
            .unwrap()
            .tracks[0]
            .track_id;
        for frame in 2..=96 {
            tr.process_frame_with_poses(frame, &[], &[]).unwrap();
        }
        let new = tr
            .process_frame_with_poses(97, &[person(97, 0, 0.2, 0.91)], &[])
            .unwrap()
            .tracks[0]
            .track_id;
        assert_ne!(id, new);
    }
    #[test]
    fn overlapping_lookalikes_do_not_share_track_id_on_same_frame() {
        let mut tr = tracker();
        let first = tr
            .process_frame_with_poses(0, &[person(0, 0, 0.2, 0.99), person(0, 1, 0.36, 0.98)], &[])
            .unwrap();
        assert_eq!(first.tracks.len(), 2);
        // ByteTrack never promotes a tentative single-frame identity from
        // a weak detection. Confirm both with a second strong observation.
        tr.process_frame_with_poses(
            1,
            &[person(1, 0, 0.203, 0.90), person(1, 1, 0.363, 0.90)],
            &[],
        )
        .unwrap();
        let next = tr
            .process_frame_with_poses(
                2,
                &[person(2, 0, 0.205, 0.70), person(2, 1, 0.365, 0.16)],
                &[],
            )
            .unwrap();
        assert_eq!(next.tracks.len(), 2);
        assert_eq!(next.tracks[0].track_id, first.tracks[0].track_id);
        assert_eq!(next.tracks[1].track_id, first.tracks[1].track_id);
    }
    #[test]
    fn assignment_strictness_changes_track_continuity() {
        let run = |strict: f64| {
            let options = TrackingOptions {
                min_assignment_score: strict,
                ..TrackingOptions::default()
            };
            let mut tr = ByteTracker::new(options).unwrap();
            let id = tr
                .process_frame_with_poses(0, &[person(0, 0, 0.20, 0.99)], &[])
                .unwrap()
                .tracks[0]
                .track_id;
            tr.process_frame_with_poses(1, &[person(1, 0, 0.20, 0.99)], &[])
                .unwrap();
            let detected = tr
                .process_frame_with_poses(2, &[person(2, 0, 0.235, 0.99)], &[])
                .unwrap();
            (id, detected.tracks.first().map(|p| p.track_id))
        };
        let (original, relaxed) = run(0.43);
        let (_, strict) = run(1.00);
        assert_eq!(relaxed, Some(original));
        assert_ne!(
            strict,
            Some(original),
            "Stricter IoU must prevent the uncertain old ID join"
        );
    }

    #[test]
    fn ambiguity_margin_blocks_unsafe_assignment_and_duplicate_new_id() {
        let run = |margin: f64| {
            let options = TrackingOptions {
                ambiguity_margin: margin,
                ..TrackingOptions::default()
            };
            let mut tr = ByteTracker::new(options).unwrap();
            tr.process_frame_with_poses(
                0,
                &[person(0, 0, 0.20, 0.99), person(0, 1, 0.28, 0.98)],
                &[],
            )
            .unwrap();
            tr.process_frame_with_poses(
                1,
                &[person(1, 0, 0.20, 0.99), person(1, 1, 0.28, 0.98)],
                &[],
            )
            .unwrap();
            let match_at_middle = tr
                .process_frame_with_poses(2, &[person(2, 0, 0.24, 0.99)], &[])
                .unwrap();
            (
                match_at_middle.tracks.len(),
                match_at_middle.ambiguous_detections,
                tr.next_id(),
            )
        };
        let loose = run(0.0);
        let strict = run(0.15);
        assert_eq!(
            loose.0, 1,
            "No ambiguity gate should make the best available match"
        );
        assert_eq!(strict.0, 0, "Ambiguous match must not be silently assigned");
        assert_eq!(strict.1, 1);
        assert_eq!(
            strict.2, 3,
            "The rejected match must not create a new ghost Track ID"
        );
    }

    #[test]
    fn kalman_prediction_and_update_are_finite_and_close_to_observed() {
        let mut k = Kalman::new([0.2, 0.2, 0.3, 0.8]);
        for i in 0..15 {
            k.predict(1.0, false);
            k.update([0.2 + 0.001 * i as f64, 0.2, 0.3 + 0.001 * i as f64, 0.8]);
            assert!(valid_bbox_inferred(k.bbox()));
        }
        assert!((k.bbox()[0] - 0.214).abs() < 0.02);
    }
    #[test]
    fn rejects_malformed_pose_alignment_and_unordered_frames() {
        let mut t = tracker();
        assert!(t
            .process_frame_with_poses(0, &[person(0, 0, 0.2, 0.8)], &[None, None])
            .is_err());
        t.process_frame_with_poses(0, &[person(0, 0, 0.2, 0.8)], &[])
            .unwrap();
        assert!(t.process_frame_with_poses(0, &[], &[]).is_err());
    }
}
