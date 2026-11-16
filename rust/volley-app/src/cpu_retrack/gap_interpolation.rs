//! Do not paint a disappeared Track ID over a person already observed
//! under another ID. Visual interpolation must follow real identity continuity.
use serde_json::Value;
use std::collections::{BTreeMap, VecDeque};
use volley_core::fusion::Candidate;

fn near_same_person_at_boundary(anchor: &Candidate, candidate: &Candidate, elapsed: u64) -> bool {
    // A head is required for cross-ID suppression: overlapping box centers
    // alone cannot distinguish two teammates standing close to each other.
    let (Some(head_a), Some(head_b)) = (anchor.head_point_norm, candidate.head_point_norm) else {
        return false;
    };
    if !(1..=15).contains(&elapsed) {
        return false;
    }
    let a = anchor.bbox_xyxy_norm;
    let b = candidate.bbox_xyxy_norm;
    let height_a = a[3] - a[1];
    let height_b = b[3] - b[1];
    if height_a.min(height_b) <= 0.02 || height_a.max(height_b) / height_a.min(height_b) > 2.5 {
        return false;
    }
    let elapsed = elapsed as f64;
    let head_distance = (head_a[0] - head_b[0]).hypot(head_a[1] - head_b[1]);
    let center_x_a = (a[0] + a[2]) * 0.5;
    let center_x_b = (b[0] + b[2]) * 0.5;
    // The center of a top-only RTMPose crop shifts vertically even though
    // the head stays in place. Require matching heads and overlapping torso.
    if head_distance > (0.014 + 0.002 * elapsed).min(0.040)
        || (center_x_a - center_x_b).abs() > (0.025 + 0.005 * elapsed).min(0.095)
    {
        return false;
    }
    let x = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let y = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let width = (a[2] - a[0]).min(b[2] - b[0]);
    let size = |bb: [f64; 4]| (bb[2] - bb[0]) * (bb[3] - bb[1]);
    let overlap = x * y / (size(a) + size(b) - x * y).max(1e-9);
    x > 0.65 * width && y > 0.35 * height_a.min(height_b) && overlap > 0.12
}

/// A different real Track ID observed next to BOTH endpoints is evidence
/// that the person changed labels during the gap. The vanished ID must not
/// be interpolated over those observations. We never merge IDs here.
pub(super) fn alternate_track_bridges_gap(
    queued: &VecDeque<Value>,
    source_id: u64,
    left: u64,
    prior: &Candidate,
    right: u64,
    next: &Candidate,
) -> bool {
    let mut boundaries = BTreeMap::<u64, (bool, bool)>::new();
    for frame in queued {
        let Some(at) = frame["frame_index"].as_u64() else {
            continue;
        };
        if at <= left || at >= right {
            continue;
        }
        let near_left = at - left <= 15;
        let near_right = right - at <= 15;
        if !near_left && !near_right {
            continue;
        }
        let inferred = frame["interpolated_track_ids"].as_array();
        let Some(players) = frame["tracked_players"].as_array() else {
            continue;
        };
        for player in players {
            let Some(other_id) = player["track_id"].as_u64() else {
                continue;
            };
            if other_id == source_id
                || inferred.is_some_and(|ids| ids.iter().any(|id| id.as_u64() == Some(other_id)))
            {
                continue;
            }
            let Ok(other) = serde_json::from_value::<Candidate>(player["detection"].clone()) else {
                continue;
            };
            let flags = boundaries.entry(other_id).or_default();
            if near_left && near_same_person_at_boundary(prior, &other, at - left) {
                flags.0 = true;
            }
            if near_right && near_same_person_at_boundary(next, &other, right - at) {
                flags.1 = true;
            }
        }
    }
    boundaries.values().any(|(left, right)| *left && *right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn player(frame: u64, x: f64, y: f64) -> Candidate {
        Candidate {
            frame_index: frame,
            class_name: "player".into(),
            score: 0.75,
            bbox_xyxy_norm: [x - 0.036, y - 0.12, x + 0.036, y + 0.12],
            query_index: Some(0),
            touch_probability: None,
            head_point_norm: Some([x, y - 0.09]),
        }
    }
    #[test]
    fn mm_25m54s_partial_body_ghost_is_blocked_by_real_full_body_track() {
        // Real detector/pose measurements from the saved GPU fusion.
        // ID 238: partial-body crops at frames 46590 and 46630.
        // ID 241: full-body crops at 46600, 46610, 46620.
        // The 0.155 -> 0.32 height ratio formerly blocked bridging.
        let pose = |frame, bbox, head| Candidate {
            frame_index: frame,
            class_name: "player".into(),
            score: 0.72,
            bbox_xyxy_norm: bbox,
            query_index: Some(0),
            touch_probability: None,
            head_point_norm: Some(head),
        };
        let prior = pose(
            46590,
            [0.527968, 0.634933, 0.600697, 0.789512],
            [0.562892, 0.652935],
        );
        let next = pose(
            46630,
            [0.523511, 0.615371, 0.590210, 0.770552],
            [0.548320, 0.709169],
        );
        let full = [
            pose(
                46600,
                [0.530560, 0.622365, 0.607799, 0.942932],
                [0.562159, 0.656500],
            ),
            pose(
                46610,
                [0.520413, 0.631329, 0.597414, 0.942371],
                [0.556352, 0.660752],
            ),
            pose(
                46620,
                [0.512124, 0.653987, 0.591641, 1.0],
                [0.549670, 0.686537],
            ),
        ];
        let mut queue = VecDeque::new();
        for current in full {
            queue.push_back(json!({
                "frame_index":current.frame_index,
                "tracked_players":[{"track_id":241,"detection":current}]
            }));
        }
        assert!(alternate_track_bridges_gap(
            &queue, 238, 46590, &prior, 46630, &next
        ));
    }

    #[test]
    fn real_alternate_track_between_both_anchors_blocks_visual_ghost() {
        let first = player(0, 0.453, 0.77);
        let last = player(70, 0.331, 0.76);
        let mut queue = VecDeque::new();
        for (frame, x) in [
            (10, 0.43),
            (20, 0.39),
            (30, 0.34),
            (40, 0.30),
            (50, 0.30),
            (60, 0.31),
        ] {
            let other = player(frame, x, 0.78);
            queue.push_back(json!({"frame_index":frame,
                "tracked_players":[{"track_id":792,"detection":other}]}));
        }
        assert!(alternate_track_bridges_gap(
            &queue, 795, 0, &first, 70, &last
        ));
        // A track that happens to overlap only at one endpoint is not
        // sufficient to suppress legitimate interpolation.
        queue.pop_back();
        assert!(!alternate_track_bridges_gap(
            &queue, 795, 0, &first, 70, &last
        ));
    }
    #[test]
    fn inferred_alternate_does_not_suppress_real_continuity() {
        let first = player(0, 0.45, 0.77);
        let last = player(70, 0.33, 0.76);
        let mut queue = VecDeque::new();
        for (frame, x) in [(10, 0.43), (60, 0.31)] {
            queue.push_back(json!({"frame_index":frame,
                "interpolated_track_ids":[792],
                "tracked_players":[{"track_id":792,"detection":player(frame,x,0.78)}]}));
        }
        assert!(!alternate_track_bridges_gap(
            &queue, 795, 0, &first, 70, &last
        ));
    }
}
