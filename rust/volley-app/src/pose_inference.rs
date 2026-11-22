//! Immutable, versioned RTMPose evidence from observed player detector boxes.
//! Uses the prepared video and selected GPU fusion, never CPU Track IDs.
pub(crate) mod interpolation;
#[cfg(any(unix, windows))]
mod stream;
#[cfg(test)]
use crate::{inference_stream, preparation};
use crate::{projects, AppState};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
#[cfg(test)]
use std::io::{BufRead, BufReader, BufWriter};
#[cfg(test)]
use std::sync::atomic::Ordering;
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::Path,
    sync::atomic::AtomicBool,
};
#[cfg(any(unix, windows))]
pub(crate) use stream::{GpuPoseStage, PendingPose, PoseStream};
use uuid::Uuid;
#[cfg(any(test, unix, windows))]
use volley_inference::decode::RawDetection;
use volley_inference::{
    pose::{PoseEngine, MODEL_FILE, MODEL_SHA256},
    Device,
};
#[cfg(test)]
use volley_media::raw_decoder::{decode_original_frames_in_range, AnalysisRange};

const MAX_FUSION: u64 = 64 * 1024 * 1024 * 1024;
#[cfg(test)]
const MAX_LINE: u64 = 512 * 1024;
const MAX_POSES_PER_FRAME: usize = 12;
const POSE_FILE: &str = "poses.jsonl";

fn file_hash(path: &Path) -> Result<String, String> {
    let mut input = File::open(path).map_err(|e| format!("Hash open {}: {e}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let n = input
            .read(&mut buffer)
            .map_err(|e| format!("Hash read: {e}"))?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn safe_file(path: &Path, max: u64) -> Result<(), String> {
    let meta =
        fs::symlink_metadata(path).map_err(|e| format!("Pose source {}: {e}", path.display()))?;
    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > max {
        return Err(format!(
            "Unsafe or oversized pose source: {}",
            path.display()
        ));
    }
    Ok(())
}

fn bbox(candidate: &Value) -> Option<[f64; 4]> {
    // Smaller far-court players get weaker detection scores. Keep them as
    // pose proposals while retaining the existing geometry and IoU guards.
    if candidate["class_name"] != "player" || candidate["score"].as_f64()? < 0.35 {
        return None;
    }
    let values = candidate["bbox_xyxy_norm"].as_array()?;
    let [a, b, c, d] = values.as_slice() else {
        return None;
    };
    let rect = [a.as_f64()?, b.as_f64()?, c.as_f64()?, d.as_f64()?];
    if rect
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || rect[0] >= rect[2]
        || rect[1] >= rect[3]
    {
        return None;
    }
    if (rect[2] - rect[0]) * 640.0 < 8.0 || (rect[3] - rect[1]) * 360.0 < 20.0 {
        return None;
    }
    Some(rect)
}

fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = w * h;
    let sa = (a[2] - a[0]) * (a[3] - a[1]);
    let sb = (b[2] - b[0]) * (b[3] - b[1]);
    intersection / (sa + sb - intersection).max(1e-10)
}

pub(crate) fn boxes(frame: &Value) -> Vec<[f64; 4]> {
    let mut unique = Vec::new();
    if let Some(candidates) = frame["player_candidates"].as_array() {
        for candidate in candidates {
            if unique.len() >= MAX_POSES_PER_FRAME {
                break;
            }
            if let Some(bb) = bbox(candidate) {
                if unique.iter().all(|&existing| iou(existing, bb) < 0.6) {
                    unique.push(bb);
                }
            }
        }
    }
    unique
}

/// Canonical pose proposals directly from observed detector evidence before
/// fusion is written. This intentionally serializes the same RawDetection
/// payload that the existing fusion reader validates, so score rounding,
/// geometry, count and NMS guards all stay in this single owner.
#[cfg(any(test, unix, windows))]
pub(crate) fn boxes_from_raw_detections(
    detections: &[RawDetection],
    frame_index: u64,
) -> Result<Vec<[f64; 4]>, String> {
    let mut at_frame: Vec<_> = detections
        .iter()
        .filter(|record| record.frame_index == frame_index && record.class_name == "player")
        .collect();
    // Fusion sorts proposals before applying the pose deduplication guard.
    at_frame.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.query_index.cmp(&b.query_index))
    });
    let candidates = serde_json::to_value(at_frame)
        .map_err(|error| format!("Serialize observed pose boxes: {error}"))?;
    Ok(boxes(&json!({"player_candidates": candidates})))
}

/// Legacy sampling mode for stored GPU profiles; new primary analysis always
/// uses the detector's observed 10-frame sampling interval.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PoseInputMode {
    #[default]
    Photo,
    Video,
}
impl PoseInputMode {
    pub(crate) const fn effective_stride(self, photo_stride: usize) -> usize {
        match self {
            Self::Photo => photo_stride,
            Self::Video => 1,
        }
    }
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Photo => "photo",
            Self::Video => "video",
        }
    }
}

#[cfg(test)]
fn sampled_boxes(
    frame: &Value,
    expected_frame: u64,
    stride: usize,
) -> Result<Vec<[f64; 4]>, String> {
    if !expected_frame.is_multiple_of(stride as u64)
        || frame["frame_index"].as_u64() != Some(expected_frame)
    {
        return Err("RTMPose frame does not match the RF-DETR sampling timeline".into());
    }
    Ok(boxes(frame))
}

/// Reads a bounded detector neighborhood without materializing a whole video.
/// Only measured RF-DETR proposals can become interpolation anchors.
#[cfg(test)]
type PoseBoxPair = (Vec<[f64; 4]>, Vec<[f64; 4]>);

#[cfg(test)]
struct FusionBoxWindow {
    reader: BufReader<File>,
    frames: std::collections::VecDeque<interpolation::FrameBoxes>,
    next: u64,
    total: u64,
    stride: usize,
}

#[cfg(test)]
impl FusionBoxWindow {
    fn open(path: &Path, total: u64, stride: usize) -> Result<Self, String> {
        Ok(Self {
            reader: BufReader::new(File::open(path).map_err(|e| e.to_string())?),
            frames: std::collections::VecDeque::new(),
            next: 0,
            total,
            stride,
        })
    }

    fn at(&mut self, index: u64) -> Result<PoseBoxPair, String> {
        if index >= self.total || !index.is_multiple_of(self.stride as u64) {
            return Err("RTMPose sample exceeds verified detector timeline".into());
        }
        let until = index
            .saturating_add(interpolation::MAX_GAP_FRAMES)
            .min(self.total - 1);
        while self.next <= until {
            let mut line = Vec::new();
            let n = self
                .reader
                .read_until(b'\n', &mut line)
                .map_err(|e| e.to_string())?;
            if n == 0 || n as u64 > MAX_LINE || line.last() != Some(&b'\n') {
                return Err("GPU fusion stream truncated or malformed".into());
            }
            let frame: Value = serde_json::from_slice(&line)
                .map_err(|e| format!("Invalid GPU fusion row: {e}"))?;
            if frame["frame_index"].as_u64() != Some(self.next) {
                return Err("GPU fusion frame index mismatch".into());
            }
            let selected = if self.next.is_multiple_of(self.stride as u64) {
                sampled_boxes(&frame, self.next, self.stride)?
            } else {
                Vec::new()
            };
            self.frames.push_back(interpolation::FrameBoxes {
                frame: self.next,
                boxes: selected,
            });
            self.next += 1;
        }
        while self
            .frames
            .front()
            .is_some_and(|row| row.frame.saturating_add(interpolation::MAX_GAP_FRAMES) < index)
        {
            self.frames.pop_front();
        }
        let real = self
            .frames
            .iter()
            .find(|row| row.frame == index)
            .ok_or("Missing RTMPose detector sample")?
            .boxes
            .clone();
        let inferred = interpolation::interpolate_boxes(&self.frames, index, &real);
        Ok((real, inferred))
    }
}

/// Both measured and ReID/geometry-eligible interpolated crop coordinates
/// must be submitted to the same RTMPose model on the actual source frame.
/// Provenance survives inference for all downstream pose and motion consumers.
fn pose_proposals(
    observed: Vec<[f64; 4]>,
    interpolated: Vec<[f64; 4]>,
) -> Vec<([f64; 4], &'static str)> {
    let mut proposals = Vec::with_capacity(observed.len() + interpolated.len());
    proposals.extend(observed.into_iter().map(|bb| (bb, "observed")));
    proposals.extend(interpolated.into_iter().map(|bb| (bb, "interpolated_box")));
    proposals
}

/// Probe the selected fusion and run ONNX using one persistent CUDA session.
/// In-progress output is never exposed; returns an immutable manifest after sync.
#[cfg(test)]
#[expect(clippy::too_many_arguments)]
pub fn process(
    app: &AppState,
    project_id: &str,
    video: &str,
    fusion_id: &str,
    memory_mb: u64,
    threads: usize,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(&'static str, f64, u64, u64) -> Result<(), String>,
) -> Result<Value, String> {
    if !app.writes_enabled {
        return Err("Pose analysis requires writable isolated studio storage".into());
    }
    if !volley_core::contracts::job_id(fusion_id) {
        return Err("Invalid GPU fusion identifier".into());
    }
    let _ = projects::existing_project(app, project_id)
        .map_err(|e| format!("Pose project missing: {e:?}"))?;
    let root = projects::project_root(app).join(project_id);
    let dir = root.join("native_gpu_fusions").join(fusion_id);
    let manifest_file = dir.join("manifest.json");
    let frames_file = dir.join("frames.jsonl");
    safe_file(&manifest_file, 32 * 1024)?;
    safe_file(&frames_file, MAX_FUSION)?;
    let manifest: Value =
        serde_json::from_slice(&fs::read(&manifest_file).map_err(|e| e.to_string())?)
            .map_err(|e| format!("Pose fusion manifest: {e}"))?;
    if manifest["schema"] != "volley-native-gpu-fusion/1"
        || manifest["state"] != "raw_candidates_fused"
        || manifest["project_id"] != project_id
        || manifest["video"] != video
        || manifest["fusion_id"] != fusion_id
        || manifest["sha256"]
            .as_str()
            .is_none_or(|sha| !volley_core::contracts::sha256(sha))
    {
        return Err("RTMPose fusion provenance mismatch".into());
    }
    if file_hash(&frames_file)? != manifest["sha256"].as_str().unwrap_or_default() {
        return Err("RTMPose source fusion SHA mismatch".into());
    }
    let stride = manifest["player_sampling_period_frames"]
        .as_u64()
        .filter(|n| (1..=30).contains(n))
        .ok_or("RTMPose requires a verified shared 1..30-frame detector sampling period")?
        as usize;
    let plan = preparation::plan_for_run(app, project_id, video, "native_gpu_fusions", fusion_id)
        .map_err(|e| format!("{e:?}"))?;
    let prepared = plan.ready_path().ok_or("Prepared pose video missing")?;
    let source = inference_stream::source_stamp(prepared)?;
    volley_media::preparation::verify_source_provenance(
        &plan.source,
        &manifest["original_source"],
    )?;
    if manifest["source_identity"] != json!({"bytes":source.0,"mtime_ns":source.1}) {
        return Err("Prepared video changed since GPU fusion".into());
    }
    let model = app.repository_root.join("models").join(MODEL_FILE);
    safe_file(&model, 256 * 1024 * 1024)?;
    let hash = file_hash(&model)?;
    if hash != MODEL_SHA256 {
        return Err(
            "RTMPose-M Body7 model hash differs from verified official ONNX artifact".into(),
        );
    }
    let runs = root.join("native_gpu_pose_runs");
    fs::create_dir_all(&runs).map_err(|e| format!("Create pose run root: {e}"))?;
    let run_id = Uuid::new_v4().simple().to_string();
    let output = runs.join(&run_id);
    fs::create_dir(&output).map_err(|e| format!("Create pose run directory: {e}"))?;
    let result = (|| -> Result<Value, String> {
        let frames_expected = manifest["emitted_frames"]
            .as_u64()
            .filter(|n| *n > 0 && *n <= 48 * 3600 * 30)
            .ok_or("Fusion does not declare a valid number of frames")?;
        let expected_samples = frames_expected.div_ceil(stride as u64);
        on_progress("pose", 0.0, 0, expected_samples)?;
        let mut box_window = FusionBoxWindow::open(&frames_file, frames_expected, stride)?;
        let mut engine = PoseEngine::load(&app.repository_root, Device::Cuda, threads, memory_mb)?;
        let file = File::create(output.join("poses.jsonl.partial"))
            .map_err(|e| format!("Create pose evidence: {e}"))?;
        let mut writer = BufWriter::new(file);
        let mut checksum = Sha256::new();
        let mut samples = 0u64;
        let mut players = 0u64;
        let mut observed_crops = 0u64;
        let mut interpolated_crops = 0u64;
        let mut evidence_bytes = 0u64;
        let estimated = manifest["emitted_frames"]
            .as_u64()
            .filter(|v| *v > 0)
            .unwrap_or(1);
        let decoded_frames = manifest["decoded_frames"]
            .as_u64()
            .filter(|n| *n >= frames_expected && *n <= frames_expected + 8)
            .ok_or("GPU fusion decoded frame count is invalid")?;
        let native_video = volley_media::probe(&plan.source)?;
        let range: Option<AnalysisRange> =
            serde_json::from_value(manifest["analysis_range"].clone())
                .map_err(|e| format!("Invalid fused analysis interval: {e}"))?;
        if let Some(range) = range {
            range.validate(native_video.duration_sec)?;
        }
        decode_original_frames_in_range(
            &plan.source,
            decoded_frames,
            range,
            cancel,
            |index, rgb, width, height| {
                if index >= frames_expected || index % stride as u64 != 0 {
                    return Ok(());
                }
                let (observed, inferred) = box_window.at(index)?;
                let proposals = pose_proposals(observed, inferred);
                let mut poses = Vec::with_capacity(proposals.len());
                for group in proposals.chunks(8) {
                    let boxes = group.iter().map(|(bb, _)| *bb).collect::<Vec<_>>();
                    let measured = engine.infer_batch_original(rgb, width, height, &boxes)?;
                    for ((bb, origin), points) in group.iter().zip(measured) {
                        if *origin == "interpolated_box" {
                            interpolated_crops += 1;
                        } else {
                            observed_crops += 1;
                        }
                        let keypoints = points.map(|p| [p.x as f64, p.y as f64, p.score as f64]);
                        poses.push(json!({
                            "box":bb,
                            "keypoints":keypoints,
                            "crop_origin":origin
                        }));
                    }
                }
                players += poses.len() as u64;
                samples += 1;
                let mut row = serde_json::to_vec(&json!({
                    "frame":index,"players":poses
                }))
                .map_err(|e| e.to_string())?;
                row.push(b'\n');
                evidence_bytes += row.len() as u64;
                if row.len() > 64 * 1024 || evidence_bytes > MAX_FUSION {
                    return Err("RTMPose evidence exceeds bounded output".into());
                }
                writer
                    .write_all(&row)
                    .map_err(|e| format!("Write pose evidence: {e}"))?;
                checksum.update(&row);
                if samples.is_multiple_of(40) {
                    on_progress(
                        "pose",
                        (index as f64 / estimated as f64).clamp(0.0, 1.0),
                        samples,
                        expected_samples,
                    )?;
                }
                Ok(())
            },
        )?;
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        if samples != expected_samples {
            return Err("RTMPose sample timeline differs from RF-DETR".into());
        }
        if observed_crops + interpolated_crops != players {
            return Err("RTMPose pose inputs were not all passed through model inference".into());
        }
        if inference_stream::source_stamp(prepared)? != source {
            return Err("Prepared video changed during RTMPose analysis".into());
        }
        if file_hash(&frames_file)? != manifest["sha256"].as_str().unwrap_or_default() {
            return Err("GPU fusion changed during RTMPose analysis".into());
        }
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
        drop(writer);
        fs::rename(output.join("poses.jsonl.partial"), output.join(POSE_FILE))
            .map_err(|e| format!("Commit RTMPose evidence: {e}"))?;
        volley_media::preparation::verify_source_provenance(
            &plan.source,
            &manifest["original_source"],
        )?;
        let pose_manifest = json!({
            "schema":"volley-native-gpu-pose/1","state":"pose_evidence_complete",
            "pose_run_id":run_id, "fusion_id":fusion_id,
            "project_id":project_id, "video":video, "fps":30,
            "stride":stride, "pose_mode":"photo", "photo_stride":stride,
            "decoded_frames":decoded_frames,
            "sampled_frames":samples, "player_poses":players,
            "observed_pose_crops":observed_crops,
            "interpolated_pose_crops":interpolated_crops,
            "model":"RTMPose-M-Body7","model_sha256":hash,
            "original_source":manifest["original_source"],
            "working_stream":"original_rgb_fps30_v1",
            "original_source":manifest["original_source"],
            "source_fusion_sha256":manifest["sha256"], "source_identity":manifest["source_identity"],
            "analysis_range":manifest["analysis_range"],
            "box_source_version":"detector_observed_and_bounded_interpolated_v1",
            "pixel_source_version":"original_full_resolution_rgb24_v1",
            "pixel_dimensions":[native_video.width,native_video.height],
            "coordinate_system":"normalized_original_display_v1",
            "source_person_sampling_period_frames":stride,
            "interpolate_players":false,
            "file":POSE_FILE, "sha256":format!("{:x}",checksum.finalize())
        });
        crate::storage::write_json_atomic(&output.join("manifest.json"), &pose_manifest)
            .map_err(|e| format!("Write pose manifest: {e:?}"))?;
        on_progress("pose", 1.0, samples, expected_samples)?;
        Ok(pose_manifest)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&output);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pose_samples_follow_any_verified_detector_period() {
        for step in [1usize, 2, 3, 5, 10, 30] {
            let frame = (step * 4) as u64;
            let row = json!({
                "frame_index":frame,
                "player_candidates":[{
                    "class_name":"player","score":0.8,
                    "bbox_xyxy_norm":[0.1,0.2,0.3,0.8]
                }]
            });
            assert_eq!(sampled_boxes(&row, frame, step).unwrap().len(), 1);
            assert!(sampled_boxes(&row, frame + 1, step).is_err() || step == 1);
        }
    }

    #[test]
    fn observed_and_interpolated_boxes_are_both_rtmpose_inputs() {
        let measured = [0.11, 0.20, 0.20, 0.80];
        let filled = [0.45, 0.20, 0.55, 0.80];
        let crops = pose_proposals(vec![measured], vec![filled]);
        assert_eq!(
            crops,
            vec![(measured, "observed"), (filled, "interpolated_box")]
        );
        // Model inference loops over every returned crop in both the
        // standalone and the shared stream; an inferred box is never
        // represented as an interpolated joint array.
    }

    #[test]
    fn rtmpose_proposals_use_interpolated_boxes_only_for_missing_real_detection() {
        let root = std::env::temp_dir().join(format!("volley-pose-box-gap-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let file = root.join("frames.jsonl");
        let mut evidence = Vec::new();
        for frame in 0..21_u64 {
            let players = if frame == 0 || frame == 20 {
                vec![json!({
                    "frame_index":frame,"class_name":"player","score":0.8,
                    "bbox_xyxy_norm":if frame == 0 {
                        [0.2,0.2,0.3,0.8]
                    } else {
                        [0.22,0.2,0.32,0.8]
                    }
                })]
            } else {
                Vec::new()
            };
            evidence.extend(
                serde_json::to_vec(&json!({
                    "frame_index":frame,"player_candidates":players
                }))
                .unwrap(),
            );
            evidence.push(b'\n');
        }
        fs::write(&file, evidence).unwrap();
        let mut reader = FusionBoxWindow::open(&file, 21, 10).unwrap();
        let (real, inferred) = reader.at(10).unwrap();
        assert!(real.is_empty());
        assert_eq!(inferred.len(), 1);
        assert!((inferred[0][0] - 0.21).abs() < 1e-9);
        let (real, inferred) = reader.at(20).unwrap();
        assert_eq!(real.len(), 1);
        assert!(inferred.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn raw_detector_pose_proposals_follow_fusion_score_order() {
        let make = |score, bbox, query| RawDetection {
            frame_index: 20,
            class_name: "player",
            score,
            bbox_xyxy_norm: bbox,
            query_index: Some(query),
            touch_probability: None,
            head_point_norm: None,
        };
        // Input order is not ranked; fusion ranks descending before pose NMS.
        let low = make(0.42, [0.1, 0.1, 0.3, 0.8], 0);
        let high = make(0.95, [0.11, 0.11, 0.31, 0.81], 1);
        let winning_box = high.bbox_xyxy_norm.map(|v| v as f64);
        let expected = vec![high.clone(), low.clone()];
        let fused = json!({"player_candidates":expected});
        let filtered = boxes_from_raw_detections(&[low, high], 20).unwrap();
        assert_eq!(filtered, boxes(&fused));
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], winning_box);
    }

    #[test]
    fn raw_detector_pose_boxes_exactly_match_serialized_fusion_guards() {
        let sample = |frame_index, score, bbox| RawDetection {
            frame_index,
            class_name: "player",
            score,
            bbox_xyxy_norm: bbox,
            query_index: Some(0),
            touch_probability: None,
            head_point_norm: None,
        };
        let detections = vec![
            sample(20, 0.9, [0.1, 0.1, 0.3, 0.8]),
            sample(20, 0.8, [0.1, 0.1, 0.3, 0.8]),
            sample(20, 0.35, [0.5, 0.1, 0.7, 0.8]),
            sample(20, 0.9, [0.5, 0.5, 0.5001, 0.7]),
            sample(30, 0.9, [0.75, 0.1, 0.9, 0.8]),
        ];
        let filtered: Vec<_> = detections.iter().filter(|r| r.frame_index == 20).collect();
        let fused = json!({"player_candidates":filtered});
        assert_eq!(
            boxes_from_raw_detections(&detections, 20).unwrap(),
            boxes(&fused)
        );
        assert_eq!(boxes_from_raw_detections(&detections, 30).unwrap().len(), 1);
        assert!(boxes_from_raw_detections(&detections, 40)
            .unwrap()
            .is_empty());
    }
    #[test]
    fn pose_boxes_are_aligned_to_detector_and_never_synthesized() {
        let row = json!({"frame_index":20,"player_candidates":[
            {"class_name":"player","score":0.9,"bbox_xyxy_norm":[0.1,0.1,0.3,0.8]},
            {"class_name":"player","score":0.8,"bbox_xyxy_norm":[0.1,0.1,0.3,0.8]},
            {"class_name":"player","score":0.9,"bbox_xyxy_norm":[0.5,0.1,0.7,0.8]}]});
        assert_eq!(sampled_boxes(&row, 20, 10).unwrap().len(), 2);
        assert!(sampled_boxes(&row, 21, 10).is_err());
        assert!(sampled_boxes(&row, 10, 10).is_err());
        let gap = json!({"frame_index":30,"player_candidates":[]});
        assert!(sampled_boxes(&gap, 30, 10).unwrap().is_empty());
    }

    #[test]
    fn rejects_tiny_poses_and_deduplicates() {
        let f = json!({"player_candidates":[
            {"class_name":"player","score":0.9,"bbox_xyxy_norm":[0.1,0.1,0.3,0.8]},
            {"class_name":"player","score":0.8,"bbox_xyxy_norm":[0.1,0.1,0.3,0.8]},
            {"class_name":"player","score":0.4,"bbox_xyxy_norm":[0.6,0.1,0.8,0.8]}]});
        assert_eq!(boxes(&f).len(), 2);
        // Missing ball evidence must not remove valid player proposals.
        let without_ball = json!({"player_candidates":f["player_candidates"],"selected_ball":null});
        assert_eq!(boxes(&without_ball).len(), 2);
        assert!(bbox(
            &json!({"class_name":"player","score":0.34,"bbox_xyxy_norm":[0.6,0.1,0.8,0.8]})
        )
        .is_none());
        assert!(bbox(
            &json!({"class_name":"player","score":0.9,"bbox_xyxy_norm":[0.1,0.1,0.101,0.102]})
        )
        .is_none());
    }
}
