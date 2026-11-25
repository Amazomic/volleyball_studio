use super::{BoundedReader, PlayerFrameScope, MAX_ROW_BYTES};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};
use volley_core::{bbox_iou, fusion::Candidate, tracker::PoseEvidence};
use volley_inference::{pose::PoseEngine, Device};
use volley_media::raw_decoder::{decode_original_frames_in_range, AnalysisRange};

pub(crate) const POLICY: &str = "original_rgb_stabilized_track_crops_v4";
pub(crate) const PRIOR_CROP_POLICY: &str = "original_rgb_stabilized_track_crops_v3";
pub(crate) const LEGACY_POLICY: &str = "original_rgb_stabilized_track_crops_v1";
pub(crate) const PREVIOUS_POLICY: &str = "original_rgb_stabilized_track_crops_v2";

pub(super) fn stabilize_crops(tracked: &mut [Value], raw: &[Candidate]) -> u64 {
    let anchors = tracked
        .iter()
        .map(|row| {
            let body: [f64; 4] =
                serde_json::from_value(row["detection"]["bbox_xyxy_norm"].clone()).ok()?;
            let head: [f64; 2] =
                serde_json::from_value(row["detection"]["head_point_norm"].clone()).ok()?;
            Some((body, head))
        })
        .collect::<Vec<_>>();
    let mut boxes = anchors
        .iter()
        .map(|a| a.map(|(body, _)| body))
        .collect::<Vec<_>>();
    for detection in raw.iter().filter(|c| c.score >= 0.35) {
        let b = detection.bbox_xyxy_norm;
        let mut matches = anchors.iter().enumerate().filter(|(_, anchor)| {
            let Some((a, head)) = anchor else {
                return false;
            };
            let w = a[2] - a[0];
            let h = a[3] - a[1];
            let bw = b[2] - b[0];
            let bh = b[3] - b[1];
            let body = bbox_iou(*a, b) >= 0.55 && bw <= 1.45 * w && bh <= 1.60 * h;
            let head_crop = bw <= 0.8 * w
                && bh <= 0.32 * h
                && (b[0]..=b[2]).contains(&head[0])
                && (b[1]..=b[3]).contains(&head[1]);
            body || head_crop
        });
        if let Some((index, _)) = matches.next() {
            if matches.next().is_none() {
                if let Some(a) = &mut boxes[index] {
                    *a = [
                        a[0].min(b[0]),
                        a[1].min(b[1]),
                        a[2].max(b[2]),
                        a[3].max(b[3]),
                    ];
                }
            }
        }
    }
    let mut combined = 0;
    for ((row, anchor), crop) in tracked.iter_mut().zip(anchors).zip(boxes) {
        if let (Some((original, _)), Some(crop)) = (anchor, crop) {
            if bbox_iou(original, crop) < 0.97 {
                row["detection"]["bbox_xyxy_norm"] = json!(crop);
                row["pose_crop_origin"] = json!("combined_track_box");
                combined += 1;
            }
        }
    }
    combined
}

fn recover_leading_crop(row: &mut Value, id: u64, anchor: [f64; 4]) {
    let w = anchor[2] - anchor[0];
    let h = anchor[3] - anchor[1];
    if w < 0.008 || h < 0.06 {
        return;
    }
    let Some(players) = row["tracked_players"].as_array() else {
        return;
    };
    if players.len() >= 128 || players.iter().any(|p| p["track_id"] == id) {
        return;
    }
    let Some(raw) = row["raw_player_candidates"].as_array() else {
        return;
    };
    let mut matching = raw.iter().filter_map(|entry| {
        let candidate: Candidate = serde_json::from_value(entry.clone()).ok()?;
        let b = candidate.bbox_xyxy_norm;
        let bw = b[2] - b[0];
        let bh = b[3] - b[1];
        if candidate.score < 0.20
            || !(0.55 * w..=1.5 * w).contains(&bw)
            || !(0.35 * h..=0.85 * h).contains(&bh)
            || ((b[0] + b[2] - anchor[0] - anchor[2]) * 0.5).abs() > 0.35 * w
            || (b[1] - anchor[1]).abs() > 0.40 * h
            || b[3] > anchor[3] + 0.10 * h
        {
            return None;
        }
        if players.iter().any(|p| {
            serde_json::from_value::<[f64; 4]>(p["detection"]["bbox_xyxy_norm"].clone())
                .is_ok_and(|other| bbox_iou(other, b) > 0.25 || bbox_iou(other, anchor) > 0.25)
        }) {
            return None;
        }
        Some(candidate)
    });
    let Some(mut candidate) = matching.next() else {
        return;
    };
    if matching.next().is_some() {
        return;
    }
    let b = candidate.bbox_xyxy_norm;
    candidate.bbox_xyxy_norm = [
        b[0].min(anchor[0]),
        b[1].min(anchor[1]),
        b[2].max(anchor[2]),
        b[3].max(anchor[3]),
    ];
    if let Some(players) = row["tracked_players"].as_array_mut() {
        players.push(json!({"track_id":id,"detection":candidate,
            "measurement_source":"interpolated_track_bbox",
            "pose_crop_origin":"restored_partial_track_box","is_verified_athlete":false}));
    }
}
pub(super) fn restore_partial_crops(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    scope: Option<&PlayerFrameScope>,
    cancel: Option<&AtomicBool>,
) -> Result<String, String> {
    let tmp = source.with_extension("partial-crops");
    let result = (|| {
        let mut reader = BoundedReader::open(source)?;
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)
                .map_err(|e| e.to_string())?,
        );
        let mut hash = Sha256::new();
        let mut queue = VecDeque::<Value>::new();
        let mut anchors = BTreeMap::<u64, (u64, [f64; 4])>::new();
        let write = |row: Value,
                     writer: &mut BufWriter<fs::File>,
                     hash: &mut Sha256|
         -> Result<(), String> {
            let mut bytes = serde_json::to_vec(&row).map_err(|e| e.to_string())?;
            bytes.push(b'\n');
            if bytes.len() > MAX_ROW_BYTES {
                return Err("Crop plan row too large".into());
            }
            writer.write_all(&bytes).map_err(|e| e.to_string())?;
            hash.update(bytes);
            Ok(())
        };
        let height = |b: [f64; 4]| b[3] - b[1];
        for frame in 0..frames {
            if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
                return Err("cancelled".into());
            }
            let row: Value = serde_json::from_slice(&reader.line()?.ok_or("Crop plan truncated")?)
                .map_err(|e| e.to_string())?;
            if row["frame_index"].as_u64() != Some(frame) {
                return Err("Crop plan frame mismatch".into());
            }
            anchors.retain(|_, (at, _)| frame.saturating_sub(*at) <= 90);
            for player in row["tracked_players"]
                .as_array()
                .ok_or("Missing crop tracks")?
            {
                let id = player["track_id"].as_u64().ok_or("Missing crop ID")?;
                let current: [f64; 4] =
                    serde_json::from_value(player["detection"]["bbox_xyxy_norm"].clone())
                        .map_err(|e| e.to_string())?;
                if player["detection"]["score"].as_f64().unwrap_or(0.0) < 0.35 {
                    continue;
                }
                if player["detection"]["score"].as_f64().unwrap_or(0.0) >= 0.50
                    && anchors
                        .get(&id)
                        .is_none_or(|(_, prior)| height(*prior) < height(current) * 0.80)
                {
                    for pending in queue.iter_mut().rev() {
                        let Some(at) = pending["frame_index"].as_u64() else {
                            continue;
                        };
                        if frame.saturating_sub(at) > 12
                            || scope.is_some_and(|s| !s.same_interval(at, frame))
                        {
                            break;
                        }
                        recover_leading_crop(pending, id, current);
                    }
                }
                if let Some(&(left, prior)) = anchors.get(&id) {
                    if height(current) < height(prior) * 0.72 {
                        continue;
                    }
                    let dt = frame - left;
                    if dt >= 2
                        && height(current).min(height(prior))
                            >= 0.75 * height(current).max(height(prior))
                        && scope.is_none_or(|s| s.same_interval(left, frame))
                        && ((current[0] + current[2] - prior[0] - prior[2]) * 0.5).abs() <= 0.08
                    {
                        for pending in queue.iter_mut().filter(|r| {
                            r["frame_index"]
                                .as_u64()
                                .is_some_and(|f| f > left && f < frame)
                        }) {
                            let at = pending["frame_index"].as_u64().unwrap_or(left);
                            let t = (at - left) as f64 / dt as f64;
                            let estimate: [f64; 4] =
                                std::array::from_fn(|i| prior[i] * (1.0 - t) + current[i] * t);
                            let Some(players) = pending["tracked_players"].as_array_mut() else {
                                continue;
                            };
                            let Some(index) = players
                                .iter()
                                .position(|p| p["track_id"].as_u64() == Some(id))
                            else {
                                continue;
                            };
                            let Ok(part) = serde_json::from_value::<[f64; 4]>(
                                players[index]["detection"]["bbox_xyxy_norm"].clone(),
                            ) else {
                                continue;
                            };
                            let area = (part[2] - part[0]) * height(part);
                            let overlap = (part[2].min(estimate[2]) - part[0].max(estimate[0]))
                                .max(0.0)
                                * (part[3].min(estimate[3]) - part[1].max(estimate[1])).max(0.0);
                            if height(part) > height(estimate) * 0.80 || overlap < 0.75 * area {
                                continue;
                            }
                            if players.iter().enumerate().any(|(j, p)| {
                                j != index
                                    && serde_json::from_value::<[f64; 4]>(
                                        p["detection"]["bbox_xyxy_norm"].clone(),
                                    )
                                    .is_ok_and(|b| bbox_iou(b, estimate) > 0.25)
                            }) {
                                continue;
                            }
                            players[index]["detection"]["bbox_xyxy_norm"] = json!([
                                part[0].min(estimate[0]),
                                part[1].min(estimate[1]),
                                part[2].max(estimate[2]),
                                part[3].max(estimate[3])
                            ]);
                            players[index]["pose_crop_origin"] =
                                json!("restored_partial_track_box");
                        }
                    }
                }
                anchors.insert(id, (frame, current));
            }
            queue.push_back(row);
            if queue.len() > 90 {
                write(
                    queue.pop_front().ok_or("Missing crop row")?,
                    &mut writer,
                    &mut hash,
                )?;
            }
        }
        for row in queue {
            write(row, &mut writer, &mut hash)?;
        }
        if reader.line()?.is_some() || reader.finish() != expected_sha {
            return Err("Crop plan SHA mismatch".into());
        }
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
        drop(writer);
        fs::rename(&tmp, source).map_err(|e| e.to_string())?;
        Ok(format!("{:x}", hash.finalize()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}

pub(crate) struct Refinement {
    pub threads: usize,
    pub memory_mb: u64,
    pub stride: usize,
}

pub(super) struct Source<'a> {
    pub original: &'a Path,
    pub replay: &'a Path,
    pub replay_sha: &'a str,
    pub frames: u64,
    pub decoded_frames: u64,
    pub range: Option<AnalysisRange>,
}

fn proposals(players: &[Value]) -> Vec<(usize, [f64; 4], &'static str)> {
    players
        .iter()
        .enumerate()
        .filter_map(|(index, player)| {
            let origin = if player["measurement_source"] == "interpolated_track_bbox"
                || player["pose_crop_origin"] == "restored_partial_track_box"
            {
                "interpolated_track_box"
            } else if player["pose_crop_origin"] == "combined_track_box" {
                "combined_track_box"
            } else {
                "stabilized_track_box"
            };
            let bbox =
                serde_json::from_value(player["detection"]["bbox_xyxy_norm"].clone()).ok()?;
            Some((index, bbox, origin))
        })
        .collect()
}

fn stabilize_pose_crop(current: [f64; 4], previous: Option<[f64; 4]>) -> [f64; 4] {
    let Some(previous) = previous.filter(|previous| bbox_iou(*previous, current) >= 0.80) else {
        return current;
    };
    let midpoint = std::array::from_fn::<_, 4, _>(|i| (current[i] + previous[i]) * 0.5);
    [
        current[0].min(midpoint[0]),
        current[1].min(midpoint[1]),
        current[2].max(midpoint[2]),
        current[3].max(midpoint[3]),
    ]
}

pub(super) fn refine(
    repository: &Path,
    source: Source<'_>,
    settings: Refinement,
    cancelled: &AtomicBool,
) -> Result<(String, u64), String> {
    let mut reader = BoundedReader::open(source.replay)?;
    let output = source.replay.with_file_name("poses.jsonl");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|e| format!("Create refined pose replay: {e}"))?;
    let mut writer = BufWriter::new(file);
    let mut hash = Sha256::new();
    let mut engine = None;
    let mut accepted = 0;
    let mut previous_crops = BTreeMap::<u64, [f64; 4]>::new();
    decode_original_frames_in_range(
        source.original,
        source.decoded_frames,
        source.range,
        cancelled,
        |index, pixels, width, height| {
            if index >= source.frames {
                return Ok(());
            }
            let row = reader
                .line()?
                .ok_or("Tracking replay truncated during RTMPose refinement")?;
            let row: Value = serde_json::from_slice(&row).map_err(|e| e.to_string())?;
            if row["frame_index"].as_u64() != Some(index) {
                return Err("Tracking replay frame mismatch during RTMPose refinement".into());
            }
            let players = row["tracked_players"]
                .as_array()
                .ok_or("Missing tracked players")?;
            let mut next_crops = BTreeMap::new();
            let mut crops = Vec::new();
            for (player, bbox, origin) in proposals(players) {
                let id = players[player]["track_id"]
                    .as_u64()
                    .ok_or("Missing Track ID for pose crop")?;
                if (index as usize).is_multiple_of(settings.stride) {
                    crops.push((
                        player,
                        stabilize_pose_crop(bbox, previous_crops.get(&id).copied()),
                        origin,
                    ));
                }
                next_crops.insert(id, bbox);
            }
            previous_crops = next_crops;
            if !crops.is_empty() && engine.is_none() {
                engine = Some(PoseEngine::load(
                    repository,
                    Device::Cuda,
                    settings.threads,
                    settings.memory_mb,
                )?);
            }
            let mut measured = Vec::new();
            if let Some(engine) = &mut engine {
                for group in crops.chunks(8) {
                    let boxes = group.iter().map(|(_, bbox, _)| *bbox).collect::<Vec<_>>();
                    let estimates = engine.infer_batch_original(pixels, width, height, &boxes)?;
                    for ((_player, bbox, origin), joints) in group.iter().zip(estimates) {
                        let joints = joints.map(|p| [p.x as f64, p.y as f64, p.score as f64]);
                        let evidence = PoseEvidence { joints };
                        if evidence.valid() && evidence.inside_crop(*bbox, 0.02) {
                            measured.push(json!({
                                "box": bbox,
                                "keypoints": joints,
                                "crop_origin": if *origin == "interpolated_track_box" {
                                    "interpolated_box"
                                } else {
                                    "observed"
                                },
                            }));
                            accepted += 1;
                        }
                    }
                }
            }
            if !(index as usize).is_multiple_of(settings.stride) {
                return Ok(());
            }
            let mut bytes = serde_json::to_vec(&json!({"frame":index,"players":measured}))
                .map_err(|e| e.to_string())?;
            bytes.push(b'\n');
            if bytes.len() > MAX_ROW_BYTES {
                return Err("Refined tracking row exceeds the replay limit".into());
            }
            writer.write_all(&bytes).map_err(|e| e.to_string())?;
            hash.update(&bytes);
            Ok(())
        },
    )?;
    if reader.line()?.is_some() || reader.finish() != source.replay_sha {
        return Err("Tracking replay changed during RTMPose refinement".into());
    }
    writer.flush().map_err(|e| e.to_string())?;
    writer.get_ref().sync_all().map_err(|e| e.to_string())?;
    drop(writer);
    Ok((format!("{:x}", hash.finalize()), accepted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn crop_plan_reconstructs_partial_box_before_pose_inference() {
        let path =
            std::env::temp_dir().join(format!("volley-crop-plan-{}.jsonl", uuid::Uuid::new_v4()));
        let full = [0.2, 0.3, 0.3, 0.7];
        let part = [0.21, 0.5, 0.29, 0.7];
        let mut raw = Vec::new();
        for frame in 0..5 {
            let bbox = if frame == 0 || frame == 4 { full } else { part };
            let players = if frame == 2 {
                vec![]
            } else {
                vec![json!({"track_id":1,"detection":{
                    "frame_index":frame,"class_name":"player","score":0.8,"query_index":0,"bbox_xyxy_norm":bbox,"head_point_norm":null
                }})]
            };
            serde_json::to_writer(
                &mut raw,
                &json!({"frame_index":frame,"tracked_players":players}),
            )
            .unwrap();
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let repaired = restore_partial_crops(&path, &sha, 5, None, None).unwrap();
        super::super::fill_confirmed_track_gaps(&path, &repaired, 5, None, None).unwrap();
        for (frame, line) in fs::read_to_string(&path).unwrap().lines().enumerate() {
            let row: Value = serde_json::from_str(line).unwrap();
            let crops = proposals(row["tracked_players"].as_array().unwrap());
            assert_eq!(crops.len(), 1);
            for (actual, expected) in crops[0].1.iter().zip(full) {
                assert!((actual - expected).abs() < 1e-9);
            }
            if (1..=3).contains(&frame) {
                assert_eq!(crops[0].2, "interpolated_track_box");
            }
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn crop_plan_recovers_unassigned_partial_detection_before_full_track() {
        let path =
            std::env::temp_dir().join(format!("volley-crop-start-{}.jsonl", uuid::Uuid::new_v4()));
        let mut raw = Vec::new();
        for frame in 0..=9 {
            let detection = |bbox: [f64; 4], score| {
                json!({
                    "frame_index":frame,"class_name":"player","score":score,
                    "query_index":0,"bbox_xyxy_norm":bbox,"head_point_norm":null
                })
            };
            let mut players =
                vec![json!({"track_id":18,"detection":detection([0.50,0.48,0.54,0.67],0.9)})];
            if frame == 9 {
                players.push(
                    json!({"track_id":21,"detection":detection([0.485,0.435,0.509,0.553],0.8)}),
                );
            }
            let candidates = if frame == 1 {
                vec![detection([0.489, 0.419, 0.511, 0.504], 0.24)]
            } else {
                vec![]
            };
            serde_json::to_writer(&mut raw,&json!({"frame_index":frame,"tracked_players":players,"raw_player_candidates":candidates})).unwrap();
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        restore_partial_crops(&path, &sha, 10, None, None).unwrap();
        let rows: Vec<Value> = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let players = rows[1]["tracked_players"].as_array().unwrap();
        let recovered = players
            .iter()
            .find(|p| p["track_id"] == 21)
            .expect("raw supported crop");
        assert_eq!(recovered["pose_crop_origin"], "restored_partial_track_box");
        assert_eq!(recovered["detection"]["bbox_xyxy_norm"][1], 0.419);
        assert_eq!(recovered["detection"]["bbox_xyxy_norm"][3], 0.553);
        assert_eq!(rows[0]["tracked_players"].as_array().unwrap().len(), 1);
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn crop_alignment_preserves_current_bounds_and_rejects_a_large_jump() {
        let current = [0.21, 0.26, 0.30, 0.66];
        let previous = [0.205, 0.26, 0.295, 0.67];
        let aligned = stabilize_pose_crop(current, Some(previous));
        assert!(aligned[0] < current[0] && aligned[3] > current[3]);
        assert_eq!(aligned[1..3], current[1..3]);
        assert_eq!(
            stabilize_pose_crop(current, Some([0.6, 0.5, 0.8, 0.9])),
            current
        );
    }

    #[test]
    fn single_pose_pass_accepts_unchanged_stable_track_crop() {
        let rows = [json!({
            "track_id": 7,
            "detection": {"bbox_xyxy_norm": [0.21, 0.26, 0.29, 0.66]}
        })];
        let crops = proposals(&rows);
        assert_eq!(
            crops,
            [(0, [0.21, 0.26, 0.29, 0.66], "stabilized_track_box")]
        );
    }

    #[test]
    fn measured_head_and_body_expand_one_crop_without_stealing_neighbor() {
        let mut tracked = vec![
            json!({"track_id":7,"detection":{"bbox_xyxy_norm":[0.21,0.26,0.29,0.66],"head_point_norm":[0.25,0.27]}}),
            json!({"track_id":8,"detection":{"bbox_xyxy_norm":[0.32,0.25,0.42,0.70],"head_point_norm":[0.37,0.27]}}),
        ];
        let raw = [
            [0.21, 0.30, 0.29, 0.72],
            [0.23, 0.23, 0.27, 0.31],
            [0.33, 0.25, 0.41, 0.69],
        ]
        .map(|bbox_xyxy_norm| volley_core::fusion::Candidate {
            frame_index: 0,
            class_name: "player".into(),
            score: 0.9,
            bbox_xyxy_norm,
            query_index: Some(0),
            touch_probability: None,
            head_point_norm: None,
        });
        stabilize_crops(&mut tracked, &raw);
        assert_eq!(
            tracked[0]["detection"]["bbox_xyxy_norm"],
            json!([0.21, 0.23, 0.29, 0.72])
        );
        assert_eq!(tracked[0]["pose_crop_origin"], "combined_track_box");
        assert!(tracked[1]["pose_crop_origin"].is_null());
    }
}
