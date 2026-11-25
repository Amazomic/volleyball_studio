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

pub(crate) const POLICY: &str = "original_rgb_stable_track_pose_attachment_v6";
pub(crate) const REJECTED_LEADING_POLICY: &str = "original_rgb_stabilized_track_crops_v4";
pub(crate) const PREVIOUS_TRACKED_CROP_POLICY: &str = "original_rgb_stabilized_track_crops_v5";
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
    pub scope: &'a PlayerFrameScope,
}

fn pose_visibility(player: &Value) -> Option<Vec<[f64; 4]>> {
    serde_json::from_value(
        player
            .get("pose_visible_regions")
            .unwrap_or(&player["visible_regions"])
            .clone(),
    )
    .ok()
}

fn valid_attached_pose(pose: &PoseEvidence, bbox: [f64; 4], mask: Option<&[[f64; 4]]>) -> bool {
    let Some(mask) = mask else {
        return pose.valid() && pose.inside_crop(bbox, 0.02);
    };
    let reliable = pose.joints.iter().filter(|p| p[2] >= 0.32);
    pose.joints
        .iter()
        .all(|p| p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)))
        && reliable.clone().all(|p| {
            mask.iter()
                .any(|r| p[0] >= r[0] && p[0] < r[2] && p[1] >= r[1] && p[1] < r[3])
        })
        && (reliable.count() >= 2 || pose.joints[5..11].iter().any(|p| p[2] >= 0.50))
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
            if pose_visibility(player).is_some_and(|regions| regions.is_empty()) {
                return None;
            }
            let bbox = serde_json::from_value(
                player
                    .get("full_track_box")
                    .unwrap_or(&player["detection"]["bbox_xyxy_norm"])
                    .clone(),
            )
            .ok()?;
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

fn read_crop_plan_row(
    reader: &mut BoundedReader,
    frame: u64,
    scope: &PlayerFrameScope,
) -> Result<Value, String> {
    let row: Value = serde_json::from_slice(
        &reader
            .line()?
            .ok_or("Tracking replay truncated during RTMPose refinement")?,
    )
    .map_err(|e| e.to_string())?;
    if row["frame_index"].as_u64() != Some(frame) {
        return Err("Tracking replay frame mismatch during RTMPose refinement".into());
    }
    let players = row["tracked_players"]
        .as_array()
        .ok_or("Missing tracked players")?;
    if !scope.contains(frame) && !players.is_empty() {
        return Err("RTMPose crop plan contains players outside selected rally intervals".into());
    }
    Ok(row)
}

fn pose_frame_total(
    source: &Source<'_>,
    stride: usize,
    cancelled: &AtomicBool,
) -> Result<u64, String> {
    if !(1..=30).contains(&stride) {
        return Err("Invalid RTMPose stride".into());
    }
    let mut reader = BoundedReader::open(source.replay)?;
    let mut total = 0;
    for frame in 0..source.frames {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let row = read_crop_plan_row(&mut reader, frame, source.scope)?;
        let players = row["tracked_players"]
            .as_array()
            .ok_or("Missing tracked players")?;
        if frame.is_multiple_of(stride as u64) && !proposals(players).is_empty() {
            total += 1;
        }
    }
    if reader.line()?.is_some() || reader.finish() != source.replay_sha {
        return Err("Tracking crop plan changed before RTMPose refinement".into());
    }
    Ok(total)
}

pub(super) fn refine(
    repository: &Path,
    source: Source<'_>,
    settings: Refinement,
    cancelled: &AtomicBool,
    mut on_frames: impl FnMut(u64, u64) -> Result<(), String>,
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
    let total = pose_frame_total(&source, settings.stride, cancelled)?;
    let mut processed = 0;
    on_frames(processed, total)?;
    decode_original_frames_in_range(
        source.original,
        source.decoded_frames,
        source.range,
        cancelled,
        |index, pixels, width, height| {
            if index >= source.frames {
                return Ok(());
            }
            let row = read_crop_plan_row(&mut reader, index, source.scope)?;
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
                    let visibility = group
                        .iter()
                        .map(|(player, _, _)| pose_visibility(&players[*player]))
                        .collect::<Vec<_>>();
                    let estimates = engine.infer_batch_original_visible(
                        pixels,
                        width,
                        height,
                        &boxes,
                        &visibility,
                    )?;
                    for (((player, bbox, origin), joints), visible) in
                        group.iter().zip(estimates).zip(visibility)
                    {
                        let joints = joints.map(|p| {
                            let (x, y) = (p.x as f64, p.y as f64);
                            let score = if visible.as_ref().is_some_and(|parts| {
                                !parts
                                    .iter()
                                    .any(|r| x >= r[0] && x < r[2] && y >= r[1] && y < r[3])
                            }) {
                                0.0
                            } else {
                                p.score as f64
                            };
                            [x, y, score]
                        });
                        let evidence = PoseEvidence { joints };
                        if valid_attached_pose(&evidence, *bbox, visible.as_deref()) {
                            measured.push(json!({
                                "track_id": players[*player]["track_id"],
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
            if !crops.is_empty() {
                processed += 1;
                if processed == 1 || processed.is_multiple_of(30) || processed == total {
                    on_frames(processed, total)?;
                }
            }
            Ok(())
        },
    )?;
    if reader.line()?.is_some() || reader.finish() != source.replay_sha {
        return Err("Tracking replay changed during RTMPose refinement".into());
    }
    if processed != total {
        return Err("RTMPose processed frame count differs from verified crop plan".into());
    }
    writer.flush().map_err(|e| e.to_string())?;
    writer.get_ref().sync_all().map_err(|e| e.to_string())?;
    drop(writer);
    Ok((format!("{:x}", hash.finalize()), accepted))
}

const MAX_POSE_GAP_SIDE: u64 = 15;
pub(crate) const INTERPOLATED_POSE_SOURCE: &str = "same_track_pose_interpolation_v1";

fn reliable_joint(player: &Value, joint: usize) -> Option<[f64; 3]> {
    if player["pose_interpolated"] == true || player["pose_source"] != POLICY {
        return None;
    }
    let p: [f64; 3] = serde_json::from_value(player["pose"].get(joint)?.clone()).ok()?;
    (p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) && p[2] >= 0.32).then_some(p)
}

fn fill_missing_pose(row: &mut Value, context: &VecDeque<Value>) -> u64 {
    let Some(frame) = row["frame_index"].as_u64() else {
        return 0;
    };
    let Some(roster) = row["rally_roster_window"].as_u64() else {
        return 0;
    };
    let Some(players) = row["tracked_players"].as_array_mut() else {
        return 0;
    };
    let mut count = 0;
    for player in players.iter_mut().filter(|p| !p["pose"].is_array()) {
        let Some(id) = player["track_id"].as_u64() else {
            continue;
        };
        let Ok(bbox) = serde_json::from_value::<[f64; 4]>(
            player
                .get("full_track_box")
                .unwrap_or(&player["detection"]["bbox_xyxy_norm"])
                .clone(),
        ) else {
            continue;
        };
        if !bbox
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            || bbox[0] >= bbox[2]
            || bbox[1] >= bbox[3]
        {
            continue;
        }
        let before = context
            .iter()
            .rev()
            .filter(|r| r["frame_index"].as_u64().is_some_and(|f| f < frame));
        let after = context
            .iter()
            .filter(|r| r["frame_index"].as_u64().is_some_and(|f| f > frame));
        let mut left = [None; 17];
        let mut right = [None; 17];
        for (rows, anchors) in [
            (before.collect::<Vec<_>>(), &mut left),
            (after.collect::<Vec<_>>(), &mut right),
        ] {
            for r in rows {
                let Some(at) = r["frame_index"].as_u64() else {
                    break;
                };
                if at.abs_diff(frame) > MAX_POSE_GAP_SIDE || r["rally_roster_window"] != roster {
                    break;
                }
                let Some(p) = r["tracked_players"]
                    .as_array()
                    .and_then(|ps| ps.iter().find(|p| p["track_id"] == id))
                else {
                    break;
                };
                for (joint, anchor) in anchors.iter_mut().enumerate() {
                    if anchor.is_none() {
                        *anchor = reliable_joint(p, joint).map(|point| (at, point));
                    }
                }
            }
        }
        let mut joints = [[0.0; 3]; 17];
        let mut first = frame;
        let mut last = frame;
        let mut supported = 0;
        let pad_x = (bbox[2] - bbox[0]) * 0.25;
        let pad_y = (bbox[3] - bbox[1]) * 0.20;
        for i in 0..17 {
            let (Some((a, p)), Some((b, q))) = (left[i], right[i]) else {
                continue;
            };
            let t = (frame - a) as f64 / (b - a) as f64;
            let point = [
                p[0] + (q[0] - p[0]) * t,
                p[1] + (q[1] - p[1]) * t,
                p[2].min(q[2]),
            ];
            if point[0] < bbox[0] - pad_x
                || point[0] > bbox[2] + pad_x
                || point[1] < bbox[1] - pad_y
                || point[1] > bbox[3] + pad_y
            {
                continue;
            }
            joints[i] = point;
            first = first.min(a);
            last = last.max(b);
            supported += 1;
        }
        if supported >= 2 {
            player["pose"] = json!(joints);
            player["pose_interpolated"] = json!(true);
            player["pose_source"] = json!(INTERPOLATED_POSE_SOURCE);
            player["pose_interpolation_frames"] = json!([first, last]);
            count += 1;
        }
    }
    count
}

pub(super) fn interpolate_gaps(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    cancel: Option<&AtomicBool>,
) -> Result<(String, u64), String> {
    let tmp = source.with_extension("pose-interpolation");
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
        let mut context = VecDeque::new();
        let mut next = 0;
        let mut count = 0;
        for frame in 0..frames {
            if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
                return Err("cancelled".into());
            }
            let row: Value = serde_json::from_slice(
                &reader
                    .line()?
                    .ok_or("Pose interpolation replay truncated")?,
            )
            .map_err(|e| e.to_string())?;
            if row["frame_index"] != frame {
                return Err("Pose interpolation frame mismatch".into());
            }
            context.push_back(row);
            while next + MAX_POSE_GAP_SIDE <= frame || (frame + 1 == frames && next < frames) {
                let start = next.saturating_sub(MAX_POSE_GAP_SIDE);
                while context
                    .front()
                    .is_some_and(|r| r["frame_index"].as_u64().is_some_and(|f| f < start))
                {
                    context.pop_front();
                }
                let mut output = context
                    .get((next - start) as usize)
                    .ok_or("Missing pose interpolation context")?
                    .clone();
                count += fill_missing_pose(&mut output, &context);
                let mut bytes = serde_json::to_vec(&output).map_err(|e| e.to_string())?;
                bytes.push(b'\n');
                if bytes.len() > MAX_ROW_BYTES {
                    return Err("Pose interpolation row exceeds limit".into());
                }
                writer.write_all(&bytes).map_err(|e| e.to_string())?;
                hash.update(&bytes);
                next += 1;
            }
        }
        if reader.line()?.is_some() || reader.finish() != expected_sha {
            return Err("Pose interpolation replay SHA mismatch".into());
        }
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
        drop(writer);
        fs::rename(&tmp, source).map_err(|e| e.to_string())?;
        Ok((format!("{:x}", hash.finalize()), count))
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}

pub(super) fn attach(
    source: &Path,
    expected_sha: &str,
    frames: u64,
    snapshot: &super::pose_source::Snapshot,
    cancel: Option<&AtomicBool>,
) -> Result<(String, u64, u64), String> {
    let tmp = source.with_extension("pose-attach");
    let result = (|| {
        let mut reader = BoundedReader::open(source)?;
        let mut poses = super::pose_source::PoseRows::open(snapshot)?;
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)
                .map_err(|e| e.to_string())?,
        );
        let mut sha = Sha256::new();
        let mut accepted = 0;
        let mut inferred_frames = 0;
        let mut previous = BTreeMap::new();
        for frame in 0..frames {
            if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
                return Err("cancelled".into());
            }
            let mut row: Value =
                serde_json::from_slice(&reader.line()?.ok_or("Stable crop plan truncated")?)
                    .map_err(|e| e.to_string())?;
            if row["frame_index"].as_u64() != Some(frame) {
                return Err("Stable crop plan frame mismatch".into());
            }
            let players = row["tracked_players"]
                .as_array_mut()
                .ok_or("Missing stable Track IDs")?;
            let mut expected = BTreeMap::new();
            let mut next = BTreeMap::new();
            for (index, bbox, _) in proposals(players) {
                let id = players[index]["track_id"]
                    .as_u64()
                    .ok_or("Missing stable Track ID")?;
                if expected
                    .insert(
                        id,
                        (index, stabilize_pose_crop(bbox, previous.get(&id).copied())),
                    )
                    .is_some()
                {
                    return Err("Duplicate stable Track ID".into());
                }
                next.insert(id, bbox);
            }
            previous = next;
            let sample = poses.take_sample(frame)?;
            if frame.is_multiple_of(snapshot.stride as u64) && sample.is_none() {
                return Err("Missing sampled RTMPose frame".into());
            }
            if let Some(sample) = sample {
                inferred_frames += u64::from(!expected.is_empty());
                for measured in sample["players"]
                    .as_array()
                    .ok_or("Missing measured poses")?
                {
                    let id = measured["track_id"]
                        .as_u64()
                        .ok_or("Missing measured crop Track ID")?;
                    let (index, bbox) = expected
                        .remove(&id)
                        .ok_or("Pose does not belong to a unique stable Track ID")?;
                    let actual: [f64; 4] = serde_json::from_value(measured["box"].clone())
                        .map_err(|e| e.to_string())?;
                    if actual
                        .iter()
                        .zip(bbox)
                        .any(|(a, b)| !a.is_finite() || (a - b).abs() > 1e-9)
                    {
                        return Err("Measured RTMPose crop differs from stable Track box".into());
                    }
                    let joints = serde_json::from_value(measured["keypoints"].clone())
                        .map_err(|e| e.to_string())?;
                    let pose = PoseEvidence { joints };
                    if !valid_attached_pose(
                        &pose,
                        bbox,
                        pose_visibility(&players[index]).as_deref(),
                    ) {
                        return Err("Invalid measured RTMPose evidence".into());
                    }
                    players[index]["pose"] = measured["keypoints"].clone();
                    players[index]["pose_interpolated"] = json!(false);
                    players[index]["pose_source"] = json!(POLICY);
                    accepted += 1;
                }
            }
            let mut bytes = serde_json::to_vec(&row).map_err(|e| e.to_string())?;
            bytes.push(b'\n');
            if bytes.len() > MAX_ROW_BYTES {
                return Err("Stable tracking row exceeds the replay limit".into());
            }
            writer.write_all(&bytes).map_err(|e| e.to_string())?;
            sha.update(&bytes);
        }
        poses.finish(frames)?;
        if reader.line()?.is_some() || reader.finish() != expected_sha {
            return Err("Stable crop plan SHA mismatch".into());
        }
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
        drop(writer);
        fs::rename(&tmp, source).map_err(|e| e.to_string())?;
        Ok((format!("{:x}", sha.finalize()), accepted, inferred_frames))
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}
#[cfg(test)]
mod tests {
    #[test]
    fn masked_partial_arm_is_saved_without_inventing_other_joints() {
        let mut joints = [[0.0; 3]; 17];
        joints[9] = [0.45, 0.30, 0.75];
        let evidence = super::PoseEvidence { joints };
        let bbox = [0.4, 0.2, 0.5, 0.6];
        assert!(!super::valid_attached_pose(&evidence, bbox, None));
        assert!(super::valid_attached_pose(
            &evidence,
            bbox,
            Some(&[[0.4, 0.2, 0.5, 0.35]])
        ));
        assert!(!super::valid_attached_pose(
            &evidence,
            bbox,
            Some(&[[0.4, 0.4, 0.5, 0.6]])
        ));
    }
    #[test]
    fn occlusion_pose_proposals_use_full_geometry_only_for_visible_players() {
        let players = vec![
            serde_json::json!({"track_id":7,"detection":{"bbox_xyxy_norm":[0.2,0.2,0.3,0.3]},
                "full_track_box":[0.2,0.2,0.3,0.6],"visible_regions":[]}),
            serde_json::json!({"track_id":4,"detection":{"bbox_xyxy_norm":[0.4,0.2,0.5,0.3]},
                "full_track_box":[0.4,0.2,0.5,0.6],"visible_regions":[[0.4,0.2,0.5,0.3]]}),
        ];
        let crops = super::proposals(&players);
        assert_eq!(crops.len(), 1);
        assert_eq!(crops[0].0, 1);
        assert_eq!(crops[0].1, [0.4, 0.2, 0.5, 0.6]);
    }

    use super::*;
    use serde_json::json;

    #[test]
    fn missing_detector_fragment_can_still_supply_masked_original_pixels() {
        let players = vec![
            json!({"track_id":7,"measurement_source":"interpolated_track_bbox",
            "full_track_box":[0.2,0.2,0.3,0.6],"visible_regions":[],
            "pose_visible_regions":[[0.2,0.2,0.22,0.6]]}),
        ];
        assert_eq!(proposals(&players).len(), 1);
        assert_eq!(proposals(&players)[0].1, [0.2, 0.2, 0.3, 0.6]);
    }

    #[test]
    fn pose_gap_uses_same_id_and_rally_real_joint_anchors_only() {
        let make = |frame, roster, id, pose: Option<[[f64; 3]; 17]>| {
            let mut p =
                json!({"track_id":id,"full_track_box":[0.2,0.2,0.4,0.6],"visible_regions":[]});
            if let Some(pose) = pose {
                p["pose"] = json!(pose);
                p["pose_source"] = json!(POLICY);
            }
            json!({"frame_index":frame,"rally_roster_window":roster,"tracked_players":[p]})
        };
        let mut a = [[0.0; 3]; 17];
        let mut b = a;
        a[0] = [0.25, 0.25, 0.8];
        b[0] = [0.27, 0.27, 0.6];
        a[5] = [0.3, 0.32, 0.7];
        b[5] = [0.32, 0.34, 0.7];
        a[9] = [0.3, 0.23, 0.8];
        let mut row = make(11, 1, 7, None);
        let mut context = VecDeque::from([
            make(10, 1, 7, Some(a)),
            row.clone(),
            make(12, 1, 7, Some(b)),
        ]);
        assert_eq!(fill_missing_pose(&mut row, &context), 1);
        let p = &row["tracked_players"][0];
        assert_eq!(p["pose_interpolated"], true);
        assert_eq!(p["pose_interpolation_frames"], json!([10, 12]));
        assert_eq!(p["pose"][9][2], 0.0);
        assert!((p["pose"][0][0].as_f64().unwrap() - 0.26).abs() < 1e-9);
        let mut row = make(11, 1, 7, None);
        context[2] = make(12, 2, 7, Some(b));
        assert_eq!(fill_missing_pose(&mut row, &context), 0);
        context[2] = make(12, 1, 8, Some(b));
        assert_eq!(fill_missing_pose(&mut row, &context), 0);
    }

    #[test]
    fn pose_progress_counts_only_sampled_frames_with_stable_crops() {
        let dir = std::env::temp_dir().join(format!("volley-pose-count-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tracked.jsonl");
        let mut raw = Vec::new();
        let bbox = [0.2, 0.3, 0.3, 0.7];
        for frame in 0..10 {
            let players = match frame {
                2 | 3 => vec![json!({"track_id": 1, "detection": {"bbox_xyxy_norm": bbox}})],
                4 => vec![
                    json!({"track_id": 1, "measurement_source": "interpolated_track_bbox",
                    "detection": {"bbox_xyxy_norm": bbox}}),
                ],
                8 => vec![
                    json!({"track_id": 1, "detection": {"bbox_xyxy_norm": bbox}}),
                    json!({"track_id": 2, "detection": {"bbox_xyxy_norm": [0.5, 0.3, 0.6, 0.7]}}),
                ],
                _ => vec![],
            };
            serde_json::to_writer(
                &mut raw,
                &json!({"frame_index": frame, "tracked_players": players}),
            )
            .unwrap();
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let scope = PlayerFrameScope {
            selected: vec![(2, 5), (8, 8)],
            rallies: vec![(2, 5), (8, 8)],
            ball_job_id: "test".into(),
            mask_sha256: String::new(),
        };
        let source = Source {
            original: &path,
            replay: &path,
            replay_sha: &sha,
            frames: 10,
            decoded_frames: 10,
            range: None,
            scope: &scope,
        };
        let cancelled = AtomicBool::new(false);
        let actual = pose_frame_total(&source, 2, &cancelled).unwrap();
        assert_eq!(pose_frame_total(&source, 1, &cancelled).unwrap(), 4);
        let invalid_scope = PlayerFrameScope {
            selected: vec![(2, 5)],
            rallies: vec![(2, 5)],
            ball_job_id: "test".into(),
            mask_sha256: String::new(),
        };
        let invalid_source = Source {
            scope: &invalid_scope,
            ..source
        };
        assert!(pose_frame_total(&invalid_source, 2, &cancelled).is_err());
        fs::remove_dir_all(dir).unwrap();
        assert_eq!(actual, 3);
    }
    #[test]
    fn measured_pose_attaches_without_changing_any_track_box_or_id() {
        let dir = std::env::temp_dir().join(format!("volley-attach-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tracked.jsonl");
        let bbox = [0.20, 0.20, 0.30, 0.70];
        let players = vec![
            json!({"track_id":15,"measurement_source":"interpolated_track_bbox",
            "detection":{"bbox_xyxy_norm":bbox}}),
            json!({"track_id":16,
            "measurement_source":"detector_bbox","detection":{"bbox_xyxy_norm":[0.5,0.2,0.6,0.7]}}),
        ];
        let mut raw =
            serde_json::to_vec(&json!({"frame_index":0,"tracked_players":players})).unwrap();
        raw.push(b'\n');
        fs::write(&path, &raw).unwrap();
        let joints: [[f64; 3]; 17] = std::array::from_fn(|i| {
            [
                0.23 + (i % 2) as f64 * 0.02,
                0.25 + (i / 2) as f64 * 0.04,
                0.9,
            ]
        });
        let mut bytes = serde_json::to_vec(
            &json!({"frame":0,"players":[{"track_id":15,"box":bbox,"keypoints":joints}]}),
        )
        .unwrap();
        bytes.push(b'\n');
        fs::write(dir.join("poses.jsonl"), &bytes).unwrap();
        let snapshot = super::super::pose_source::Snapshot {
            id: "test".into(),
            sha: format!("{:x}", Sha256::digest(&bytes)),
            stride: 1,
            folder: dir.clone(),
        };
        let sha = format!("{:x}", Sha256::digest(&raw));
        let (_, count, frames) = attach(&path, &sha, 1, &snapshot, None).unwrap();
        assert_eq!(count, 1);
        assert_eq!(frames, 1);
        let row: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let result = row["tracked_players"].as_array().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["track_id"], 15);
        assert_eq!(result[0]["detection"], players[0]["detection"]);
        assert_eq!(result[0]["measurement_source"], "interpolated_track_bbox");
        assert_eq!(result[0]["pose_interpolated"], false);
        let saved: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(result[0]["pose"], saved["players"][0]["keypoints"]);
        assert_eq!(result[1], players[1]);
        fs::remove_dir_all(dir).unwrap();
    }
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
    fn crop_plan_does_not_create_track_before_first_observation() {
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
        assert!(players.iter().all(|p| p["track_id"] != 21));
        for row in &rows[..9] {
            assert!(row["tracked_players"]
                .as_array()
                .unwrap()
                .iter()
                .all(|p| p["track_id"] != 21));
        }
        assert!(rows[9]["tracked_players"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["track_id"] == 21));
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
