use super::{BoundedReader, MAX_ROW_BYTES};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
    sync::atomic::AtomicBool,
};
use volley_core::{bbox_iou, fusion::Candidate, tracker::PoseEvidence};
use volley_inference::{pose::PoseEngine, Device};
use volley_media::raw_decoder::{decode_original_frames_in_range, AnalysisRange};

pub(crate) const POLICY: &str = "original_rgb_stabilized_track_crops_v2";
pub(crate) const LEGACY_POLICY: &str = "original_rgb_stabilized_track_crops_v1";

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
            let origin = if player["measurement_source"] == "interpolated_track_bbox" {
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
                        if evidence.valid()
                            && evidence.torso().is_some_and(|p| {
                                (bbox[0]..=bbox[2]).contains(&p[0])
                                    && (bbox[1]..=bbox[3]).contains(&p[1])
                            })
                        {
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
