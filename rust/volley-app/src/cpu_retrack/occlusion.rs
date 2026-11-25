use super::{BoundedReader, MAX_ROW_BYTES};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};
use volley_core::fusion::Candidate;

type BBox = [f64; 4];

fn area(b: BBox) -> f64 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

fn intersection(a: BBox, b: BBox) -> BBox {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

fn visible_parts(b: BBox, front: BBox) -> Vec<BBox> {
    let i = intersection(b, front);
    if area(i) == 0.0 {
        return vec![b];
    }
    [
        [b[0], b[1], b[2], i[1]],
        [b[0], i[3], b[2], b[3]],
        [b[0], i[1], i[0], i[3]],
        [i[2], i[1], b[2], i[3]],
    ]
    .into_iter()
    .filter(|r| area(*r) > 0.0)
    .collect()
}

fn is_foreground(back: BBox, front: BBox) -> bool {
    let bh = back[3] - back[1];
    let fh = front[3] - front[1];
    bh >= 0.02
        && fh >= bh * 1.15
        && front[3] - back[3] >= bh * 0.20
        && area(intersection(back, front)) >= 0.10 * area(back)
}

#[derive(Clone)]
struct Anchor {
    frame: u64,
    confirmed: u64,
    detection: Candidate,
    support: u32,
    velocity: [f64; 2],
}

impl Anchor {
    fn predict(&self, frame: u64) -> BBox {
        let dt = (frame - self.frame).min(6) as f64;
        std::array::from_fn(|i| {
            (self.detection.bbox_xyxy_norm[i] + self.velocity[i % 2] * dt).clamp(0.0, 1.0)
        })
    }
}

fn observation(p: &Value) -> Option<(u64, Candidate)> {
    Some((
        p["track_id"].as_u64()?,
        serde_json::from_value(p["detection"].clone()).ok()?,
    ))
}

fn center(b: BBox, axis: usize) -> f64 {
    (b[axis] + b[axis + 2]) * 0.5
}

fn fragment_cost(predicted: BBox, b: BBox, score: f64) -> Option<f64> {
    let w = predicted[2] - predicted[0];
    let h = predicted[3] - predicted[1];
    let dx = (center(predicted, 0) - center(b, 0)).abs();
    let dy = (b[1] - predicted[1]).abs();
    (w > 0.0
        && h > 0.0
        && area(b) > 0.0
        && dx <= 0.85 * w + 0.004
        && dy <= 0.50 * h
        && b[2] - b[0] <= 1.65 * w
        && b[3] - b[1] <= 1.65 * h)
        .then_some(dx / w + dy / h - 0.3 * score)
}

fn align_fragment(predicted: BBox, b: BBox) -> BBox {
    let dx = center(b, 0) - center(predicted, 0);
    let top = b[1] - predicted[1];
    let bottom = b[3] - predicted[3];
    let dy = if top.abs() < bottom.abs() {
        top
    } else {
        bottom
    };
    std::array::from_fn(|i| (predicted[i] + [dx, dy][i % 2]).clamp(0.0, 1.0))
}

fn visible_regions(b: BBox, fronts: &[(u64, BBox)]) -> Vec<BBox> {
    let mut regions = vec![b];
    for (_, front) in fronts {
        regions = regions
            .into_iter()
            .flat_map(|r| visible_parts(r, *front))
            .collect();
        if regions.len() > 64 {
            return Vec::new();
        }
    }
    regions.retain(|r| r[2] - r[0] >= 0.001 && r[3] - r[1] >= 0.001);
    regions
}

struct Fragment {
    cost: f64,
    full: BBox,
    fronts: Vec<(u64, BBox)>,
    visible: Vec<BBox>,
    detection: Candidate,
}

fn track_frame(row: &mut Value, anchors: &mut BTreeMap<u64, Anchor>, frame: u64) -> u64 {
    let raw: Vec<Candidate> =
        serde_json::from_value(row["raw_player_candidates"].clone()).unwrap_or_default();
    let observed: BTreeMap<_, _> = row["tracked_players"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|p| p["measurement_source"] == "detector_bbox")
        .filter_map(observation)
        .map(|(id, detection)| {
            let original = raw.iter().find(|c| {
                c.query_index == detection.query_index
                    && (c.score - detection.score).abs() <= 1e-9
                    && volley_core::bbox_iou(c.bbox_xyxy_norm, detection.bbox_xyxy_norm) >= 0.40
            });
            (id, original.cloned().unwrap_or(detection))
        })
        .collect();
    let existing: BTreeMap<_, _> = row["tracked_players"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(observation)
        .collect();
    anchors.retain(|_, a| frame.saturating_sub(a.confirmed) <= 90);
    let mut additions = Vec::new();
    let mut metadata = BTreeMap::new();
    let mut updates = BTreeMap::new();
    for (&id, anchor) in anchors.iter() {
        if anchor.support < 6 {
            continue;
        }
        let predicted = anchor.predict(frame);
        let measured = observed.get(&id);
        let fronts_for = |full: BBox| {
            let mut fronts: Vec<_> = observed
                .iter()
                .filter_map(|(&other, c)| {
                    (other != id
                        && c.score >= 0.45
                        && anchors.get(&other).is_some_and(|a| a.support >= 6)
                        && is_foreground(full, c.bbox_xyxy_norm))
                    .then_some((other, c.bbox_xyxy_norm))
                })
                .collect();
            fronts.sort_by(|a, b| {
                area(intersection(full, b.1)).total_cmp(&area(intersection(full, a.1)))
            });
            fronts
        };
        let candidate = |c: &Candidate, assigned: bool| -> Option<Fragment> {
            let b = c.bbox_xyxy_norm;
            if c.score < 0.10 {
                return None;
            }
            let cost = fragment_cost(predicted, b, c.score)?;
            if !assigned
                && (observed.iter().any(|(&other, measured)| {
                    other != id && volley_core::bbox_iou(b, measured.bbox_xyxy_norm) > 0.50
                }) || anchors.iter().any(|(&other, other_anchor)| {
                    other != id
                        && other_anchor.support >= 6
                        && !observed.contains_key(&other)
                        && fragment_cost(other_anchor.predict(frame), b, c.score)
                            .is_some_and(|other_cost| other_cost <= cost + 0.12)
                }))
            {
                return None;
            }
            let aligned = align_fragment(predicted, b);
            let mut full = aligned;
            let initial_fronts = fronts_for(aligned);
            if initial_fronts.is_empty() {
                return None;
            }
            let exposed = visible_regions(b, &initial_fronts);
            let exposed_area = exposed.iter().map(|&r| area(r)).sum::<f64>();
            if exposed_area < 0.02 * area(b) {
                return None;
            }
            let fraction = exposed_area / area(b);
            let h = predicted[3] - predicted[1];
            let top = exposed.iter().map(|r| r[1]).fold(1.0_f64, f64::min);
            let bottom = exposed.iter().map(|r| r[3]).fold(0.0_f64, f64::max);
            if assigned && c.score >= 0.35 && fraction >= 0.65 && b[3] - b[1] >= 0.85 * h {
                full = b;
            } else if full[1] > predicted[1] {
                let max_drop = if fraction >= 0.35 && bottom - top >= 0.25 * h {
                    0.003
                } else {
                    0.0
                };
                let correction = (full[1] - predicted[1] - max_drop).max(0.0);
                full[1] -= correction;
                full[3] -= correction;
            }
            full[0] = full[0].min(b[0]);
            full[1] = full[1].min(top);
            full[2] = full[2].max(b[2]);
            let fronts = fronts_for(full);
            if fronts.is_empty() {
                return None;
            }
            let visible = visible_regions(intersection(full, b), &fronts);
            if visible.is_empty() {
                return None;
            }
            Some(Fragment {
                cost: cost - 0.85 * fraction - 0.25 * ((bottom - top) / h).min(1.0),
                full,
                fronts,
                visible,
                detection: c.clone(),
            })
        };
        if measured.is_some_and(|c| {
            c.score >= 0.35
                && c.bbox_xyxy_norm[3] - c.bbox_xyxy_norm[1] >= 0.85 * (predicted[3] - predicted[1])
                && fronts_for(c.bbox_xyxy_norm).is_empty()
        }) {
            continue;
        }
        let mut choices: Vec<_> = raw.iter().filter_map(|c| candidate(c, false)).collect();
        if let Some(assigned) = measured.and_then(|c| candidate(c, true)) {
            choices.retain(|c| {
                c.detection.query_index != assigned.detection.query_index
                    || volley_core::bbox_iou(
                        c.detection.bbox_xyxy_norm,
                        assigned.detection.bbox_xyxy_norm,
                    ) < 0.90
            });
            choices.push(assigned);
        }
        choices.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        let fragment = if choices.len() > 1
            && choices[1].cost - choices[0].cost <= 0.12
            && volley_core::bbox_iou(
                choices[0].detection.bbox_xyxy_norm,
                choices[1].detection.bbox_xyxy_norm,
            ) < 0.30
        {
            None
        } else {
            choices.into_iter().next()
        };
        let (full, fronts, regions, score) = if let Some(fragment) = fragment {
            let mut next = anchor.clone();
            next.frame = frame;
            next.detection.bbox_xyxy_norm = fragment.full;
            next.velocity = anchor.velocity.map(|v| v * 0.5);
            if measured.is_some_and(|c| c.score >= 0.35)
                || (fragment.detection.score >= 0.35
                    && fragment.visible.iter().map(|&r| area(r)).sum::<f64>()
                        >= 0.08 * area(fragment.detection.bbox_xyxy_norm))
            {
                next.confirmed = frame;
            }
            updates.insert(id, next);
            (
                fragment.full,
                fragment.fronts,
                fragment.visible,
                fragment.detection.score,
            )
        } else {
            let fronts = fronts_for(predicted);
            if fronts.is_empty() {
                continue;
            }
            (predicted, fronts, Vec::new(), 0.0)
        };
        if !existing.contains_key(&id) {
            if existing.len() + additions.len() >= 128 {
                continue;
            }
            let mut detection = anchor.detection.clone();
            detection.frame_index = frame;
            detection.bbox_xyxy_norm = full;
            detection.score = score;
            detection.head_point_norm = None;
            detection.touch_probability = None;
            detection.query_index = Some(0);
            additions.push(json!({
                "track_id":id, "detection":detection,
                "measurement_source":"interpolated_track_bbox", "is_verified_athlete":false
            }));
        }
        let front_ids = fronts.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let pose_regions = if regions.is_empty() {
            Vec::new()
        } else {
            let mut search = full;
            search[1] = (search[1] - 0.12 * (full[3] - full[1])).max(0.0);
            let mut blockers = fronts;
            for (&other, c) in &observed {
                if other != id
                    && !blockers.iter().any(|(front, _)| *front == other)
                    && area(intersection(search, c.bbox_xyxy_norm)) > 0.0
                {
                    blockers.push((other, c.bbox_xyxy_norm));
                }
            }
            visible_regions(search, &blockers)
        };
        metadata.insert(id, (full, front_ids, regions, pose_regions));
    }
    let inferred = additions.len() as u64;
    let ids: Vec<_> = additions
        .iter()
        .filter_map(|p| p["track_id"].as_u64())
        .collect();
    if let Some(players) = row["tracked_players"].as_array_mut() {
        players.extend(additions);
        for player in players {
            if let Some((full, fronts, regions, pose_regions)) =
                player["track_id"].as_u64().and_then(|id| metadata.get(&id))
            {
                player["occluding_track_id"] = json!(fronts[0]);
                player["occluding_track_ids"] = json!(fronts);
                player["visible_regions"] = json!(regions);
                player["full_track_box"] = json!(full);
                player["pose_visible_regions"] = json!(pose_regions);
            } else if let Some(id) = player["track_id"].as_u64() {
                if metadata
                    .values()
                    .any(|(_, fronts, _, _)| fronts.contains(&id))
                {
                    if let Some(measured) = observed.get(&id) {
                        let original = measured.bbox_xyxy_norm;
                        if existing.get(&id).is_some_and(|c| {
                            volley_core::bbox_iou(c.bbox_xyxy_norm, original) < 0.98
                        }) {
                            player["pose_visible_regions"] = json!([original]);
                        }
                    }
                }
            }
        }
    }
    if !ids.is_empty() {
        if !row["interpolated_track_ids"].is_array() {
            row["interpolated_track_ids"] = json!([]);
        }
        row["interpolated_track_ids"]
            .as_array_mut()
            .unwrap()
            .extend(ids.into_iter().map(|id| json!(id)));
    }
    for (id, detection) in observed {
        if detection.score < 0.35 || metadata.contains_key(&id) {
            continue;
        }
        let (support, velocity) = anchors.get(&id).map_or((1, [0.0; 2]), |a| {
            let dt = frame.saturating_sub(a.frame).max(1) as f64;
            let v = std::array::from_fn(|i| {
                let speed = (center(detection.bbox_xyxy_norm, i)
                    - center(a.detection.bbox_xyxy_norm, i))
                    / dt;
                (a.velocity[i] * 0.7 + speed * 0.3).clamp(-0.004, 0.004)
            });
            (a.support.saturating_add(1), v)
        });
        updates.insert(
            id,
            Anchor {
                frame,
                confirmed: frame,
                detection,
                support,
                velocity,
            },
        );
    }
    anchors.extend(updates);
    inferred
}

pub(super) fn apply(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    cancel: Option<&AtomicBool>,
) -> Result<(String, u64), String> {
    let temp = source.with_extension("occlusion.partial");
    let result = (|| {
        let mut reader = BoundedReader::open(source)?;
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)
                .map_err(|e| e.to_string())?,
        );
        let mut hash = Sha256::new();
        let mut anchors = BTreeMap::new();
        let mut roster = None;
        let mut inferred = 0;
        for frame in 0..frames {
            if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
                return Err("cancelled".into());
            }
            let mut row: Value =
                serde_json::from_slice(&reader.line()?.ok_or("Occlusion replay truncated")?)
                    .map_err(|e| e.to_string())?;
            if row["frame_index"].as_u64() != Some(frame) {
                return Err("Occlusion replay frame mismatch".into());
            }
            let next = row["rally_roster_window"].as_u64();
            if next != roster {
                anchors.clear();
                roster = next;
            }
            if roster.is_some() {
                inferred += track_frame(&mut row, &mut anchors, frame);
            }
            let mut bytes = serde_json::to_vec(&row).map_err(|e| e.to_string())?;
            bytes.push(b'\n');
            if bytes.len() > MAX_ROW_BYTES {
                return Err("Occlusion replay exceeds row limit".into());
            }
            writer.write_all(&bytes).map_err(|e| e.to_string())?;
            hash.update(&bytes);
        }
        if reader.line()?.is_some() || reader.finish() != expected_sha {
            return Err("Occlusion replay source changed".into());
        }
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
        drop(writer);
        fs::rename(&temp, source).map_err(|e| e.to_string())?;
        Ok((format!("{:x}", hash.finalize()), inferred))
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_regions_exclude_the_foreground() {
        let rear = [0.45, 0.40, 0.51, 0.54];
        let front = [0.46, 0.45, 0.53, 0.64];
        assert!(is_foreground(rear, front));
        assert!(!is_foreground(front, rear));
        let regions = visible_parts(rear, front);
        assert_eq!(regions.len(), 2);
        assert!(regions.iter().all(|r| area(intersection(*r, front)) == 0.0));
        assert!(visible_parts(rear, [0.4, 0.3, 0.6, 0.7]).is_empty());
    }

    #[test]
    fn small_visible_sliver_is_not_discarded_with_overlapping_players() {
        let rear = [0.47, 0.423, 0.496, 0.536];
        let front = [0.457, 0.451, 0.488, 0.598];
        let merged_front = [0.457, 0.443, 0.489, 0.598];
        let neighbor = [0.490, 0.481, 0.534, 0.689];
        let mut anchors = BTreeMap::from([
            (
                7,
                Anchor {
                    frame: 10,
                    confirmed: 10,
                    detection: detection(10, rear, 0.8),
                    support: 12,
                    velocity: [0.0; 2],
                },
            ),
            (
                4,
                Anchor {
                    frame: 10,
                    confirmed: 10,
                    detection: detection(10, front, 0.9),
                    support: 12,
                    velocity: [0.0; 2],
                },
            ),
            (
                2,
                Anchor {
                    frame: 10,
                    confirmed: 10,
                    detection: detection(10, neighbor, 0.9),
                    support: 12,
                    velocity: [0.0; 2],
                },
            ),
        ]);
        let mut row = json!({"tracked_players":[
            {"track_id":4,"measurement_source":"detector_bbox","detection":detection(11,merged_front,0.9)},
            {"track_id":2,"measurement_source":"detector_bbox","detection":detection(11,neighbor,0.9)}
        ],"raw_player_candidates":[
            detection(11,front,0.9),detection(11,neighbor,0.9),
            detection(11,[0.472,0.443,0.488,0.569],0.35)
        ]});
        assert_eq!(track_frame(&mut row, &mut anchors, 11), 1);
        let player = &row["tracked_players"][2];
        assert_eq!(player["track_id"], 7);
        let visible: Vec<BBox> = serde_json::from_value(player["visible_regions"].clone()).unwrap();
        assert!(!visible.is_empty());
        assert!(visible.iter().all(|r| area(intersection(*r, front)) == 0.0));
        assert!(visible
            .iter()
            .all(|r| area(intersection(*r, neighbor)) == 0.0));
    }

    #[test]
    fn low_sliver_does_not_drag_rear_box_down_and_higher_fragment_wins() {
        let rear = [0.45, 0.40, 0.49, 0.52];
        let front = [0.44, 0.44, 0.50, 0.61];
        let mut anchors = BTreeMap::from([
            (
                7,
                Anchor {
                    frame: 10,
                    confirmed: 10,
                    detection: detection(10, rear, 0.8),
                    support: 12,
                    velocity: [0.0; 2],
                },
            ),
            (
                4,
                Anchor {
                    frame: 10,
                    confirmed: 10,
                    detection: detection(10, front, 0.9),
                    support: 12,
                    velocity: [0.0; 2],
                },
            ),
        ]);
        let mut row = json!({"tracked_players":[{"track_id":4,"measurement_source":"detector_bbox","detection":detection(11,front,0.9)}],
            "raw_player_candidates":[detection(11,front,0.9),detection(11,[0.46,0.433,0.48,0.55],0.30)]});
        track_frame(&mut row, &mut anchors, 11);
        assert!(anchors[&7].detection.bbox_xyxy_norm[1] <= 0.405);
        let mut row = json!({"tracked_players":[{"track_id":4,"measurement_source":"detector_bbox","detection":detection(12,front,0.9)}],
            "raw_player_candidates":[detection(12,front,0.9),detection(12,[0.46,0.433,0.48,0.55],0.30),detection(12,[0.46,0.39,0.48,0.465],0.22)]});
        track_frame(&mut row, &mut anchors, 12);
        let rear = row["tracked_players"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["track_id"] == 7)
            .unwrap();
        assert!(rear["visible_regions"][0][1].as_f64().unwrap() < 0.40);
    }

    fn detection(frame: u64, bbox: BBox, score: f64) -> Candidate {
        serde_json::from_value(json!({
            "frame_index":frame,"class_name":"player","query_index":0,"score":score,
            "bbox_xyxy_norm":bbox
        }))
        .unwrap()
    }

    #[test]
    fn keeps_stable_rear_id_and_only_visible_fragment() {
        let rear = [0.45, 0.40, 0.51, 0.54];
        let front = [0.46, 0.45, 0.53, 0.64];
        let mut anchors = BTreeMap::from([
            (
                7,
                Anchor {
                    frame: 10,
                    confirmed: 10,
                    detection: detection(10, rear, 0.8),
                    support: 12,
                    velocity: [0.0; 2],
                },
            ),
            (
                4,
                Anchor {
                    frame: 10,
                    confirmed: 10,
                    detection: detection(10, front, 0.9),
                    support: 12,
                    velocity: [0.0; 2],
                },
            ),
        ]);
        let mut row = json!({"tracked_players":[{
            "track_id":4,"measurement_source":"detector_bbox","detection":detection(11,front,0.9)
        }],"raw_player_candidates":[detection(11,front,0.9),detection(11,[0.47,0.39,0.50,0.48],0.22)]});
        assert_eq!(track_frame(&mut row, &mut anchors, 11), 1);
        let hidden = &row["tracked_players"][1];
        assert_eq!(hidden["track_id"], 7);
        assert_eq!(hidden["occluding_track_id"], 4);
        assert_eq!(row["interpolated_track_ids"], json!([7]));
        let regions: Vec<BBox> = serde_json::from_value(hidden["visible_regions"].clone()).unwrap();
        assert!(!regions.is_empty());
        assert!(regions.iter().all(|r| area(intersection(*r, front)) == 0.0));
        row["tracked_players"].as_array_mut().unwrap().truncate(1);
        row["raw_player_candidates"] = json!([detection(12, front, 0.9)]);
        row["interpolated_track_ids"] = json!([]);
        assert_eq!(track_frame(&mut row, &mut anchors, 12), 1);
        assert_eq!(row["tracked_players"][1]["visible_regions"], json!([]));
    }
}
