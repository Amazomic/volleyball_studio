//! Detect a Track ID jumping onto a previously co-visible, independently
//! observed body. This is measured evidence only, before any visual interpolation.
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy)]
struct Observation {
    frame: u64,
    rect: [f64; 4],
    count: u64,
}

#[derive(Clone, Copy, Default)]
struct Distinct {
    count: u32,
    last: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Handoff {
    pub track: u64,
    pub at: u64,
    pub previous_observations: u64,
    pub donor: u64,
}

#[derive(Default)]
pub(super) struct Detector {
    last: BTreeMap<u64, Observation>,
    distinct: BTreeMap<(u64, u64), Distinct>,
    proposed: Vec<Handoff>,
}

fn valid_box(row: &Value) -> Option<[f64; 4]> {
    let raw = row["detection"]["bbox_xyxy_norm"].as_array()?;
    if raw.len() != 4 {
        return None;
    }
    let mut rect = [0.0; 4];
    for (i, v) in raw.iter().enumerate() {
        rect[i] = v.as_f64()?;
        if !rect[i].is_finite() || !(0.0..=1.0).contains(&rect[i]) {
            return None;
        }
    }
    (rect[0] < rect[2] && rect[1] < rect[3]).then_some(rect)
}

fn intersection_over_union(a: [f64; 4], b: [f64; 4]) -> f64 {
    let x = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let y = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    x * y / (area_a + area_b - x * y).max(1e-10)
}
fn pair(a: u64, b: u64) -> (u64, u64) {
    (a.min(b), a.max(b))
}

impl Detector {
    pub(super) fn observe(&mut self, frame: u64, rows: &[Value]) {
        let observed = rows
            .iter()
            .filter_map(|row| Some((row["track_id"].as_u64()?, valid_box(row)?)))
            .collect::<Vec<_>>();
        let present = observed.iter().map(|(id, _)| *id).collect::<BTreeSet<_>>();
        for &(track, rect) in &observed {
            let Some(previous) = self.last.get(&track).copied() else {
                continue;
            };
            if previous.count < 8 || frame.saturating_sub(previous.frame) > 9 {
                continue;
            }
            // A reliable switch must inherit the *other* person's last
            // measured location, not merely move toward another player.
            let donor = self
                .last
                .iter()
                .filter_map(|(&id, candidate)| {
                    if id == track
                        || present.contains(&id)
                        || candidate.count < 8
                        || frame.saturating_sub(candidate.frame) > 6
                    {
                        return None;
                    }
                    let proof = self.distinct.get(&pair(track, id))?;
                    if proof.count < 3 || frame.saturating_sub(proof.last) > 90 {
                        return None;
                    }
                    let donor_iou = intersection_over_union(rect, candidate.rect);
                    let own_iou = intersection_over_union(rect, previous.rect);
                    let old_pair_iou = intersection_over_union(previous.rect, candidate.rect);
                    (donor_iou >= 0.60
                        && own_iou <= 0.35
                        && old_pair_iou <= 0.30
                        && donor_iou - own_iou >= 0.30)
                        .then_some((id, donor_iou))
                })
                .max_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((donor, _)) = donor {
                if !self
                    .proposed
                    .iter()
                    .any(|cut| cut.track == track && frame.saturating_sub(cut.at) < 120)
                {
                    self.proposed.push(Handoff {
                        track,
                        at: frame,
                        previous_observations: previous.count,
                        donor,
                    });
                }
            }
        }
        // Require repeated earlier evidence that these were really distinct
        // bodies on the same measured frame, not a duplicated bounding box.
        for (index, &(a, first)) in observed.iter().enumerate() {
            for &(b, second) in observed.iter().skip(index + 1) {
                if intersection_over_union(first, second) <= 0.15 {
                    let entry = self.distinct.entry(pair(a, b)).or_default();
                    entry.count = entry.count.saturating_add(1);
                    entry.last = frame;
                }
            }
        }
        for (id, rect) in observed {
            let count = self.last.get(&id).map_or(1, |prev| prev.count + 1);
            self.last.insert(id, Observation { frame, rect, count });
        }
        if frame % 300 == 0 {
            self.distinct
                .retain(|_, item| frame.saturating_sub(item.last) <= 120);
        }
    }

    pub(super) fn valid_cuts(&self, spans: &BTreeMap<u64, (u64, u64, u64)>) -> Vec<Handoff> {
        self.proposed
            .iter()
            .copied()
            .filter(|p| {
                spans.get(&p.track).is_some_and(|&(first, last, count)| {
                    // Enough measured identity evidence on each side. Neither a
                    // one-frame false positive nor a brief jump may split a track.
                    p.at.saturating_sub(first) >= 45
                        && last.saturating_sub(p.at) >= 45
                        && p.previous_observations >= 8
                        && count.saturating_sub(p.previous_observations) >= 12
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(id: u64, rect: [f64; 4]) -> Value {
        json!({"track_id":id,"detection":{"bbox_xyxy_norm":rect}})
    }

    #[test]
    fn splits_measured_handoff_but_not_regular_disappearance() {
        let original = [0.4296, 0.4249, 0.4622, 0.5683];
        let other = [0.4435, 0.4749, 0.4876, 0.6601];
        let inherited = [0.4429, 0.4528, 0.4793, 0.6537];
        let mut detector = Detector::default();
        // Both athletes are genuinely different at earlier frames.
        for at in (860..1041).step_by(3) {
            let earlier = if at < 1038 {
                [0.395, 0.4249, 0.429, 0.5683]
            } else {
                original
            };
            detector.observe(at, &[row(12, other), row(13, earlier)]);
        }
        detector.observe(1041, &[row(12, other)]);
        detector.observe(1044, &[row(13, inherited)]);
        for at in (1047..1600).step_by(3) {
            detector.observe(at, &[row(13, inherited)]);
        }
        assert_eq!(detector.proposed[0].track, 13);
        assert_eq!(detector.proposed[0].at, 1044);
        assert_eq!(detector.proposed[0].donor, 12);
        let cuts = detector.valid_cuts(&BTreeMap::from([
            (12, (860, 1041, 62)),
            (13, (860, 1599, 246)),
        ]));
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0].at, 1044);
    }

    #[test]
    fn saved_replay_handoff_when_readonly_fixture_configured() {
        let Some(path) = std::env::var_os("VOLLEY_HANDOFF_READONLY_REPLAY") else {
            return;
        };
        use std::io::{BufRead, BufReader};
        let file = std::fs::File::open(path).unwrap();
        let mut detector = Detector::default();
        let mut spans = BTreeMap::<u64, (u64, u64, u64)>::new();
        for (frame, row) in BufReader::new(file).lines().enumerate() {
            let frame = frame as u64;
            let item: Value = serde_json::from_str(&row.unwrap()).unwrap();
            assert_eq!(item["frame_index"], frame);
            let inferred = item["interpolated_track_ids"]
                .as_array()
                .map(|rows| {
                    rows.iter()
                        .filter_map(Value::as_u64)
                        .collect::<BTreeSet<_>>()
                })
                .unwrap_or_default();
            let players = item["tracked_players"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|player| !inferred.contains(&player["track_id"].as_u64().unwrap_or(0)))
                .cloned()
                .collect::<Vec<_>>();
            detector.observe(frame, &players);
            for player in &players {
                let id = player["track_id"].as_u64().unwrap();
                let stat = spans.entry(id).or_insert((frame, frame, 0));
                stat.1 = frame;
                stat.2 += 1;
            }
        }
        let cuts = detector.valid_cuts(&spans);
        if let Some(expected) = std::env::var("VOLLEY_HANDOFF_EXPECTED_TRACK")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
        {
            assert!(
                cuts.iter().any(|cut| cut.track == expected),
                "expected a measured handoff for ID {expected}; found {cuts:?}"
            );
        }
        println!("Readonly GPU/CPU replay handoff diagnostics: {cuts:?}");
    }

    #[test]
    fn short_false_detection_does_not_split_track() {
        let mut detector = Detector::default();
        for at in (900..930).step_by(3) {
            detector.observe(
                at,
                &[
                    row(12, [0.50, 0.45, 0.55, 0.65]),
                    row(13, [0.40, 0.35, 0.43, 0.49]),
                ],
            );
        }
        detector.observe(933, &[row(12, [0.50, 0.45, 0.55, 0.65])]);
        detector.observe(936, &[row(13, [0.50, 0.45, 0.55, 0.65])]);
        assert!(detector
            .valid_cuts(&BTreeMap::from([
                (13, (900, 936, 11)),
                (12, (900, 933, 11))
            ]))
            .is_empty());
    }
}
