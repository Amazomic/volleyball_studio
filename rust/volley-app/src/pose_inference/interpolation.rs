//! Conservative, bounded RF-DETR proposal interpolation for RTMPose crops
//! and the read-only raw detector overlay. Never modifies GPU detections or
//! asserts an athlete identity. Endpoints must both be real measurements.
use std::collections::VecDeque;

pub(crate) const MAX_GAP_FRAMES: u64 = 30;
pub(crate) const MAX_BOXES: usize = 12;

#[derive(Clone, Debug)]
pub(crate) struct FrameBoxes {
    pub frame: u64,
    pub boxes: Vec<[f64; 4]>,
}

fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = w * h;
    let area = |bb: [f64; 4]| (bb[2] - bb[0]) * (bb[3] - bb[1]);
    intersection / (area(a) + area(b) - intersection).max(1e-10)
}

fn center(b: [f64; 4]) -> [f64; 2] {
    [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]
}

/// Partial/cropped real detections can have a deceptively low IoU with the
/// inferred full-body box. Treat meaningful overlap with ANY measured box as
/// a conflict: it is safer to omit an inferred crop than draw a second player.
fn overlaps_measured(inferred: [f64; 4], measured: [f64; 4]) -> bool {
    let w = (inferred[2].min(measured[2]) - inferred[0].max(measured[0])).max(0.0);
    let h = (inferred[3].min(measured[3]) - inferred[1].max(measured[1])).max(0.0);
    let intersection = w * h;
    let area = |b: [f64; 4]| (b[2] - b[0]) * (b[3] - b[1]);
    let smaller = area(inferred).min(area(measured));
    // Overlap over the smaller box detects offset/cropped detections that
    // the usual 0.35 IoU check misses.
    iou(inferred, measured) > 0.20 || (smaller > 0.0 && intersection / smaller > 0.40)
}

fn score(a: [f64; 4], b: [f64; 4], frames: u64) -> Option<f64> {
    if !(2..=MAX_GAP_FRAMES).contains(&frames) {
        return None;
    }
    let wa = a[2] - a[0];
    let wb = b[2] - b[0];
    let ha = a[3] - a[1];
    let hb = b[3] - b[1];
    if wa.min(wb) < 0.012
        || ha.min(hb) < 0.05
        || wa.max(wb) / wa.min(wb) > 1.6
        || ha.max(hb) / ha.min(hb) > 1.6
    {
        return None;
    }
    let overlap = iou(a, b);
    let ca = center(a);
    let cb = center(b);
    let displacement = (ca[0] - cb[0]).hypot(ca[1] - cb[1]);
    let limit = (0.025 + 0.003 * frames as f64).min(0.13);
    if overlap < 0.22 || displacement > limit {
        return None;
    }
    Some(0.75 * overlap + 0.25 * (1.0 - displacement / limit))
}

/// Only bridge between unambiguous, real RF-DETR boxes. A live detector box
/// always wins; overlapping/competing people make a proposal ineligible.
pub(crate) fn interpolate_boxes(
    observations: &VecDeque<FrameBoxes>,
    at: u64,
    measured: &[[f64; 4]],
) -> Vec<[f64; 4]> {
    if measured.len() >= MAX_BOXES {
        return Vec::new();
    }
    let mut proposals: Vec<(u64, f64, [f64; 4])> = Vec::new();
    for left in observations
        .iter()
        .rev()
        .filter(|row| row.frame < at && at - row.frame < MAX_GAP_FRAMES)
    {
        for (i, &a) in left.boxes.iter().enumerate() {
            if measured.iter().any(|&box_| overlaps_measured(a, box_)) {
                continue;
            }
            for right in observations
                .iter()
                .filter(|row| row.frame > at && row.frame - left.frame <= MAX_GAP_FRAMES)
            {
                let gap = right.frame - left.frame;
                let mut scored = right
                    .boxes
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &b)| score(a, b, gap).map(|s| (j, s)))
                    .collect::<Vec<_>>();
                scored.sort_by(|x, y| y.1.total_cmp(&x.1));
                let Some(&(j, confidence)) = scored.first() else {
                    continue;
                };
                if scored
                    .get(1)
                    .is_some_and(|(_, next)| confidence - next < 0.12)
                {
                    continue;
                }
                let b = right.boxes[j];
                // Mutual, unique association. Never bridge a crossing or a
                // close ambiguous neighbor through geometry alone.
                let competing_left = left.boxes.iter().enumerate().any(|(k, &other)| {
                    k != i && score(other, b, gap).is_some_and(|s| s >= confidence - 0.12)
                });
                if competing_left {
                    continue;
                }
                let t = (at - left.frame) as f64 / gap as f64;
                let predicted = std::array::from_fn(|k| a[k] * (1.0 - t) + b[k] * t);
                if measured
                    .iter()
                    .any(|&actual| overlaps_measured(predicted, actual))
                {
                    break;
                }
                proposals.push((gap, confidence, predicted));
                break;
            }
        }
    }
    // Use the tightest real anchors first. Wider paths cannot create a
    // duplicate inferred player at the same target frame.
    proposals.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.total_cmp(&a.1)));
    let mut accepted = Vec::new();
    for (_, _, bbox) in proposals {
        if accepted.len() + measured.len() >= MAX_BOXES {
            break;
        }
        if accepted.iter().any(|&other| iou(bbox, other) > 0.35)
            || measured.iter().any(|&other| overlaps_measured(bbox, other))
        {
            continue;
        }
        accepted.push(bbox);
    }
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(frame: u64, boxes: &[[f64; 4]]) -> FrameBoxes {
        FrameBoxes {
            frame,
            boxes: boxes.to_vec(),
        }
    }

    #[test]
    fn fills_missing_detector_frame_without_replacing_measured_players() {
        let a = [0.15, 0.20, 0.25, 0.80];
        let b = [0.17, 0.20, 0.27, 0.80];
        let teammate = [0.65, 0.21, 0.75, 0.82];
        let frames = VecDeque::from([row(0, &[a]), row(10, &[teammate]), row(20, &[b])]);
        let generated = interpolate_boxes(&frames, 10, &[teammate]);
        assert_eq!(generated.len(), 1);
        assert!((generated[0][0] - 0.16).abs() < 1e-9);
        assert_eq!(
            interpolate_boxes(&frames, 10, &[teammate, generated[0]]).len(),
            0
        );
    }

    #[test]
    fn never_extrapolates_or_bridges_long_or_ambiguous_gaps() {
        let a = [0.15, 0.20, 0.25, 0.80];
        let b = [0.17, 0.20, 0.27, 0.80];
        assert!(interpolate_boxes(&VecDeque::from([row(0, &[a])]), 5, &[]).is_empty());
        assert!(
            interpolate_boxes(&VecDeque::from([row(0, &[a]), row(31, &[b])]), 15, &[]).is_empty()
        );
        let close = [0.151, 0.20, 0.251, 0.80];
        assert!(interpolate_boxes(
            &VecDeque::from([row(0, &[a, close]), row(10, &[b]),]),
            5,
            &[]
        )
        .is_empty());
    }

    #[test]
    fn hides_predicted_box_when_the_same_player_is_already_observed() {
        // Regression: the screenshot had an orange box offset ~half a
        // player's width from a 93%-confidence blue detection. The IoU
        // remains below the former 0.35 threshold, despite referring to
        // one player.
        let frames = VecDeque::from([
            row(0, &[[0.508, 0.54, 0.588, 0.80]]),
            row(10, &[[0.566, 0.565, 0.631, 0.815]]),
            row(20, &[[0.532, 0.54, 0.612, 0.80]]),
        ]);
        let real = [0.566, 0.565, 0.631, 0.815];
        let predicted = [0.520, 0.540, 0.600, 0.800];
        assert!(iou(predicted, real) < 0.35);
        assert!(overlaps_measured(predicted, real));
        assert!(interpolate_boxes(&frames, 10, &[real]).is_empty());
    }

    #[test]
    fn still_fills_only_missing_player_when_a_teammate_is_detected() {
        let teammate = [0.70, 0.15, 0.78, 0.8];
        let frames = VecDeque::from([
            row(0, &[[0.508, 0.54, 0.588, 0.80], teammate]),
            row(10, &[teammate]),
            row(20, &[[0.532, 0.54, 0.612, 0.80], teammate]),
        ]);
        let proposals = interpolate_boxes(&frames, 10, &[teammate]);
        assert_eq!(proposals.len(), 1);
        assert!((proposals[0][0] - 0.52).abs() < 1e-9);
    }

    #[test]
    fn synthetic_gap_coordinates_stay_between_real_anchors() {
        let frames = VecDeque::from([
            row(0, &[[0.10, 0.12, 0.20, 0.72]]),
            row(10, &[[0.11, 0.12, 0.21, 0.72]]),
        ]);
        let boxes = interpolate_boxes(&frames, 5, &[]);
        assert_eq!(boxes.len(), 1);
        assert!((boxes[0][0] - 0.105).abs() < 1e-9);
    }
}
