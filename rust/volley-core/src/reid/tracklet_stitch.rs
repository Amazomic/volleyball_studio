//! Appearance-gated, endpoint-based linking of measured player tracklets.
//! A geometry match alone must never identify a person or synthesize pose data.
//! Only current, model-compatible ReID embeddings can authorize a proposal.

pub const DEFAULT_REID_STITCH_THRESHOLD_PERCENT: u8 = 85;
pub const DEFAULT_STITCH_MAX_GAP_FRAMES: u64 = 90;
const MIN_LONG_TRACK_OBSERVATIONS: u32 = 6;
const MIN_LONG_TRACK_SPAN: u64 = 30;
const UNAMBIGUOUS_MARGIN: f64 = 0.035;

#[derive(Debug, Clone)]
pub struct Tracklet {
    pub id: u64,
    pub first_frame: u64,
    pub last_frame: u64,
    pub observations: u32,
    pub first_box: [f64; 4],
    pub last_box: [f64; 4],
    /// A normalized or unnormalized OpenVINO vector, all in the same model space.
    pub reid: Vec<f32>,
    pub confirmed_owner: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Link {
    pub previous_id: u64,
    pub next_id: u64,
    pub last_observed_frame: u64,
    pub first_resumed_frame: u64,
    pub last_observed_box: [f64; 4],
    pub first_resumed_box: [f64; 4],
    /// Cosine similarity, NOT the probability that two appearances belong to one person.
    pub cosine_similarity: f64,
}

impl Link {
    /// Interpolate each of the four bbox corners between actual observations.
    /// Endpoints have measured provenance; intermediates have inferred provenance.
    pub fn interpolated_box(&self, frame: u64) -> Option<[f64; 4]> {
        if frame <= self.last_observed_frame || frame >= self.first_resumed_frame {
            return None;
        }
        let t = (frame - self.last_observed_frame) as f64
            / (self.first_resumed_frame - self.last_observed_frame) as f64;
        Some(std::array::from_fn(|i| {
            self.last_observed_box[i] * (1.0 - t) + self.first_resumed_box[i] * t
        }))
    }
}

fn box_valid(b: [f64; 4]) -> bool {
    b.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) && b[0] < b[2] && b[1] < b[3]
}

fn cosine(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() < 2 || a.len() != b.len() || a.len() > 4096 {
        return None;
    }
    let mut dot = 0.0;
    let mut aa = 0.0;
    let mut bb = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    (aa > 1e-12 && bb > 1e-12).then(|| (dot / (aa * bb).sqrt()).clamp(-1.0, 1.0))
}

fn spatially_possible(left: &Tracklet, right: &Tracklet, gap: u64) -> bool {
    let a = left.last_box;
    let b = right.first_box;
    let h_a = a[3] - a[1];
    let h_b = b[3] - b[1];
    let w_a = a[2] - a[0];
    let w_b = b[2] - b[0];
    if w_a.min(w_b) < 0.010
        || h_a.min(h_b) < 0.04
        || w_a.max(w_b) / w_a.min(w_b) > 1.9
        || h_a.max(h_b) / h_a.min(h_b) > 1.9
    {
        return false;
    }
    let center = |v: [f64; 4]| [(v[0] + v[2]) * 0.5, (v[1] + v[3]) * 0.5];
    let a = center(a);
    let b = center(b);
    let motion = (a[0] - b[0]).hypot(a[1] - b[1]);
    motion <= (0.025 + gap as f64 * 0.003).min(0.17)
}

/// Link only long-to-new, nonoverlapping, nearby tracklets with unambiguous
/// verified ReID features. Missing ReID never falls back to box-only linking.
/// Each Track ID participates in at most one automatic link.
pub fn propose_links(
    tracklets: &[Tracklet],
    threshold_percent: u8,
    max_gap_frames: u64,
) -> Result<Vec<Link>, &'static str> {
    if !(55..=99).contains(&threshold_percent)
        || !(2..=180).contains(&max_gap_frames)
        || tracklets.len() > 1024
    {
        return Err("Invalid ReID stitch threshold, temporal horizon or number of tracks");
    }
    let mut seen = std::collections::BTreeSet::new();
    for t in tracklets {
        if t.id == 0
            || !seen.insert(t.id)
            || t.first_frame > t.last_frame
            || t.observations == 0
            || !box_valid(t.first_box)
            || !box_valid(t.last_box)
        {
            return Err("Invalid verified tracklet endpoint");
        }
    }
    let threshold = threshold_percent as f64 / 100.0;
    let mut candidates = Vec::<(usize, usize, f64)>::new();
    for (i, left) in tracklets.iter().enumerate() {
        if left.observations < MIN_LONG_TRACK_OBSERVATIONS
            || left.last_frame - left.first_frame < MIN_LONG_TRACK_SPAN
        {
            continue;
        }
        for (j, right) in tracklets.iter().enumerate() {
            if i == j || right.observations < 2 || right.first_frame <= left.last_frame + 1 {
                continue;
            }
            let gap = right.first_frame - left.last_frame;
            if gap > max_gap_frames || !spatially_possible(left, right, gap) {
                continue;
            }
            if let (Some(a), Some(b)) = (&left.confirmed_owner, &right.confirmed_owner) {
                if a != b {
                    continue;
                }
            }
            if let Some(score) = cosine(&left.reid, &right.reid) {
                if score >= threshold {
                    candidates.push((i, j, score));
                }
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| tracklets[a.0].id.cmp(&tracklets[b.0].id))
            .then_with(|| tracklets[a.1].id.cmp(&tracklets[b.1].id))
    });
    let mut used = std::collections::BTreeSet::new();
    let mut links = Vec::new();
    for (i, j, similarity) in &candidates {
        if used.contains(i) || used.contains(j) {
            continue;
        }
        // Two equally plausible lookalikes cannot be automatically joined.
        if candidates.iter().any(|(other_i, other_j, score)| {
            (*other_i == *i && *other_j != *j || *other_j == *j && *other_i != *i)
                && similarity - score < UNAMBIGUOUS_MARGIN
        }) {
            continue;
        }
        let previous = &tracklets[*i];
        let next = &tracklets[*j];
        used.insert(*i);
        used.insert(*j);
        links.push(Link {
            previous_id: previous.id,
            next_id: next.id,
            last_observed_frame: previous.last_frame,
            first_resumed_frame: next.first_frame,
            last_observed_box: previous.last_box,
            first_resumed_box: next.first_box,
            cosine_similarity: *similarity,
        });
    }
    Ok(links)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(id: u64, first: u64, last: u64, x: f64, feature: &[f32]) -> Tracklet {
        Tracklet {
            id,
            first_frame: first,
            last_frame: last,
            observations: ((last - first) / 5 + 1) as u32,
            first_box: [x, 0.20, x + 0.10, 0.76],
            last_box: [x + 0.01, 0.20, x + 0.11, 0.76],
            reid: feature.to_vec(),
            confirmed_owner: None,
        }
    }

    #[test]
    fn long_track_and_new_high_reid_id_interpolate_all_four_corners() {
        let tracks = [
            segment(4, 0, 90, 0.20, &[1., 0., 0.]),
            segment(17, 101, 135, 0.23, &[0.99, 0.05, 0.]),
        ];
        let linked = propose_links(&tracks, 85, 90).unwrap();
        assert_eq!(linked.len(), 1);
        let link = &linked[0];
        assert_eq!((link.previous_id, link.next_id), (4, 17));
        assert!(link.cosine_similarity > 0.98);
        assert_eq!(link.interpolated_box(90), None);
        assert_eq!(link.interpolated_box(101), None);
        let at = link.interpolated_box(95).unwrap();
        let t = 5.0 / 11.0;
        for (i, got) in at.iter().enumerate() {
            let expected = link.last_observed_box[i] * (1.0 - t) + link.first_resumed_box[i] * t;
            assert!((got - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn low_similarity_or_missing_features_never_stitch() {
        let first = segment(4, 0, 90, 0.20, &[1., 0., 0.]);
        let unrelated = segment(17, 101, 135, 0.23, &[0., 1., 0.]);
        assert!(propose_links(&[first.clone(), unrelated], 85, 90)
            .unwrap()
            .is_empty());
        let mut missing = segment(17, 101, 135, 0.23, &[]);
        assert!(propose_links(&[first.clone(), missing.clone()], 85, 90)
            .unwrap()
            .is_empty());
        missing.reid = vec![1., 0., 0.];
        assert!(propose_links(&[first, missing], 99, 90).unwrap().len() == 1);
    }

    #[test]
    fn adjacent_other_player_and_overlap_and_temporal_distance_are_vetoes() {
        let first = segment(4, 0, 90, 0.20, &[1., 0.]);
        let too_far = segment(17, 101, 135, 0.60, &[1., 0.]);
        let concurrent = segment(18, 85, 135, 0.23, &[1., 0.]);
        let late = segment(19, 230, 270, 0.23, &[1., 0.]);
        assert!(
            propose_links(&[first.clone(), too_far, concurrent, late], 85, 90)
                .unwrap()
                .is_empty()
        );
        let mut owner = segment(21, 102, 135, 0.22, &[1., 0.]);
        owner.confirmed_owner = Some("someone_else".into());
        let mut owned = first;
        owned.confirmed_owner = Some("original_person".into());
        assert!(propose_links(&[owned, owner], 85, 90).unwrap().is_empty());
    }

    #[test]
    fn equally_similar_candidates_are_ambiguous_and_not_joined() {
        let long = segment(4, 0, 90, 0.20, &[1., 0.]);
        let a = segment(17, 101, 135, 0.23, &[0.99, 0.04]);
        let b = segment(18, 101, 135, 0.19, &[0.99, 0.05]);
        assert!(propose_links(&[long, a, b], 85, 90).unwrap().is_empty());
    }
}
