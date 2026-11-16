use crate::{
    Device,
    pose::{KPT_COUNT, Keypoint, PoseEngine},
};
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider},
    session::Session,
    value::Tensor,
};
use sha2::{Digest, Sha256};
use std::path::Path;

pub const PERSON_SAMPLE_PERIOD_FRAMES: usize = 10;
const W: usize = 640;
const H: usize = 360;
const MAX_PERSONS: usize = 12;

fn is_observation_frame(frame: u64, period: u64) -> bool {
    frame.is_multiple_of(period)
}

/// RF-DETR Medium is the sole supported player detector.
pub const RFDETR_MEDIUM_FILE: &str = "rf-detr-medium.onnx";
pub const RFDETR_MEDIUM_SHA256: &str =
    "18250b5a3a9552ae117d592f8e8c27fe52438d70c914a77a8913fc45d8f29e69";
const RFDETR_SIDE: usize = 576;
const RFDETR_MIN_SCORE: f32 = 0.28;
const RFDETR_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const RFDETR_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectorVariant {
    RfDetrMedium,
}

impl DetectorVariant {
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("VOLLEY_PERSON_DETECTOR") {
            Err(std::env::VarError::NotPresent) => Ok(Self::RfDetrMedium),
            Err(error) => Err(format!("Invalid VOLLEY_PERSON_DETECTOR: {error}")),
            Ok(value) if value == "rfdetr-medium" => Ok(Self::RfDetrMedium),
            Ok(value) => Err(format!(
                "Unknown VOLLEY_PERSON_DETECTOR={value:?}; expected rfdetr-medium"
            )),
        }
    }
    fn needs_pose_in_gpu_evidence(self) -> bool {
        false
    }
    fn file(self) -> &'static str {
        RFDETR_MEDIUM_FILE
    }
    fn sha256(self) -> &'static str {
        RFDETR_MEDIUM_SHA256
    }
}

#[derive(Clone, Debug)]
pub struct PersonObservation {
    pub bbox: [f64; 4],
    pub score: f32,
    pub head: Option<[f64; 2]>,
    pub keypoints: [[f64; 3]; KPT_COUNT],
}

/// Detector rectangles are coarse crop proposals only. At least eight
/// confident joint positions, including torso evidence, are mandatory.
pub fn skeleton_box(points: &[Keypoint; KPT_COUNT]) -> Option<([f64; 4], f32)> {
    skeleton_box_with_threshold(points, 0.36, 8)
}

/// A small athlete beyond the net may have joints at 0.3–0.35 even when
/// the detector finds a coherent person. Keep the strict path for normal
/// players, and relax only the *small crop* path with more joints required.
fn skeleton_box_with_threshold(
    points: &[Keypoint; KPT_COUNT],
    min_score: f32,
    min_joints: usize,
) -> Option<([f64; 4], f32)> {
    let valid = |p: &Keypoint| {
        p.score.is_finite()
            && p.score >= min_score
            && p.x.is_finite()
            && p.y.is_finite()
            && (0.0..=1.0).contains(&p.x)
            && (0.0..=1.0).contains(&p.y)
    };
    if !(valid(&points[5]) && valid(&points[6]) && (valid(&points[11]) || valid(&points[12]))) {
        return None;
    }
    let selected: Vec<_> = points.iter().filter(|p| valid(p)).collect();
    if selected.len() < min_joints {
        return None;
    }
    let mut bounds = [1.0_f64, 1.0_f64, 0.0_f64, 0.0_f64];
    let mut confidence = 0.0f32;
    for point in selected.iter().copied() {
        bounds[0] = bounds[0].min(point.x as f64);
        bounds[1] = bounds[1].min(point.y as f64);
        bounds[2] = bounds[2].max(point.x as f64);
        bounds[3] = bounds[3].max(point.y as f64);
        confidence += point.score;
    }
    let bw = bounds[2] - bounds[0];
    let bh = bounds[3] - bounds[1];
    if bw < 0.01 || bh < 0.035 || bw > 0.70 || bh > 0.96 {
        return None;
    }
    // Use the skeleton for all four sides. Extents include a small
    // perspective-independent margin for the torso and moving limbs.
    bounds[0] = (bounds[0] - bw * 0.13 - 0.003).max(0.0);
    bounds[1] = (bounds[1] - bh * 0.10).max(0.0);
    bounds[2] = (bounds[2] + bw * 0.13 + 0.003).min(1.0);
    bounds[3] = (bounds[3] + bh * 0.10).min(1.0);
    Some((bounds, confidence / selected.len() as f32))
}

pub struct PersonEngine {
    detector: Session,
    pose: Option<PoseEngine>,
}
impl PersonEngine {
    pub fn load(
        repo: &Path,
        device: Device,
        threads: usize,
        memory_mb: u64,
    ) -> Result<Self, String> {
        Self::load_with_detector(
            repo,
            device,
            threads,
            memory_mb,
            DetectorVariant::from_env()?,
        )
    }

    pub fn load_for_gpu_evidence(
        repo: &Path,
        device: Device,
        threads: usize,
        memory_mb: u64,
    ) -> Result<Self, String> {
        let variant = DetectorVariant::from_env()?;
        Self::load_with_pose_policy(
            repo,
            device,
            threads,
            memory_mb,
            variant,
            variant.needs_pose_in_gpu_evidence(),
        )
    }

    /// Explicit detector selection for tests and the CPU preview path.
    pub fn load_with_detector(
        repo: &Path,
        device: Device,
        threads: usize,
        memory_mb: u64,
        variant: DetectorVariant,
    ) -> Result<Self, String> {
        Self::load_with_pose_policy(repo, device, threads, memory_mb, variant, true)
    }

    fn load_with_pose_policy(
        repo: &Path,
        device: Device,
        threads: usize,
        memory_mb: u64,
        variant: DetectorVariant,
        with_pose: bool,
    ) -> Result<Self, String> {
        if !(1..=128).contains(&threads) || memory_mb > 131072 {
            return Err("Invalid person inference resource settings".into());
        }
        let models = repo
            .join("models")
            .canonicalize()
            .map_err(|e| format!("Models root: {e}"))?;
        let path = models
            .join(variant.file())
            .canonicalize()
            .map_err(|e| format!("Person detector {} not installed: {e}", variant.file()))?;
        if !path.starts_with(&models)
            || !path.is_file()
            || path.metadata().map_err(|e| e.to_string())?.len() > 256 * 1024 * 1024
        {
            return Err(format!(
                "Invalid person detector path or size: {}",
                variant.file()
            ));
        }
        let fingerprint = format!(
            "{:x}",
            Sha256::digest(std::fs::read(&path).map_err(|e| format!("RF-DETR model read: {e}"))?)
        );
        if fingerprint != variant.sha256() {
            return Err(format!(
                "{} SHA-256 does not match the tested ONNX artifact",
                variant.file()
            ));
        }
        // Both GPU sessions explicitly require CUDA; never silently fall back.
        #[cfg(target_os = "linux")]
        let _ = crate::runtime_library(device)?;
        ort::init()
            .commit()
            .map_err(|e| format!("ONNX Runtime init: {e}"))?;
        let mut builder = Session::builder()
            .map_err(|e| format!("RF-DETR session builder: {e}"))?
            .with_intra_threads(threads)
            .map_err(|e| e.to_string())?;
        if device == Device::Cuda {
            let mut cuda = CUDAExecutionProvider::default().with_device_id(0);
            if memory_mb > 0 {
                cuda = cuda.with_memory_limit((memory_mb * 1024 * 1024) as usize);
            }
            if !cuda.is_available().map_err(|e| e.to_string())? {
                return Err("RF-DETR CUDA unavailable".into());
            }
            builder = builder
                .with_execution_providers([cuda.build().error_on_failure()])
                .map_err(|e| format!("RF-DETR CUDA: {e}"))?;
        } else {
            builder = builder
                .with_execution_providers([CPUExecutionProvider::default().build()])
                .map_err(|e| e.to_string())?;
        }
        let detector = builder
            .commit_from_file(path)
            .map_err(|e| format!("Open person detector: {e}"))?;
        let required_outputs: &[&str] = &["pred_boxes", "pred_logits"];
        if !detector.inputs.iter().any(|i| i.name == "input")
            || required_outputs
                .iter()
                .any(|expected| !detector.outputs.iter().any(|o| o.name == *expected))
        {
            return Err(format!(
                "Unexpected person detector ONNX tensor ABI: {}",
                variant.file()
            ));
        }
        let pose = if with_pose {
            Some(PoseEngine::load(repo, device, threads, memory_mb)?)
        } else {
            None
        };
        Ok(Self { detector, pose })
    }

    pub fn detect(&mut self, rgb: &[u8]) -> Result<Vec<PersonObservation>, String> {
        self.detect_rfdetr_medium(rgb)
    }

    pub fn detect_frames(
        &mut self,
        frames: &[u8],
        mut consume: impl FnMut(usize, Vec<PersonObservation>) -> Result<(), String>,
    ) -> Result<(), String> {
        let frame_bytes = W * H * 3;
        if frames.is_empty() || !frames.len().is_multiple_of(frame_bytes) {
            return Err("Person detector expects complete RGB24 frames".into());
        }
        for (index, frame) in frames.chunks_exact(frame_bytes).enumerate() {
            consume(index, self.detect_rfdetr_medium(frame)?)?;
        }
        Ok(())
    }

    /// Run the expensive detector and pose model only on global frame anchors.
    /// Other frames produce no synthetic neural observations: confirmed CPU
    /// tracking fills short gaps after assigning stable provisional IDs.
    /// The global index keeps sampling aligned across 9-frame GPU windows.
    pub fn detect_frames_sampled(
        &mut self,
        frames: &[u8],
        first_frame: u64,
        sample_every: u64,
        mut consume: impl FnMut(usize, Vec<PersonObservation>) -> Result<(), String>,
    ) -> Result<(), String> {
        let frame_bytes = W * H * 3;
        if frames.is_empty()
            || !frames.len().is_multiple_of(frame_bytes)
            || !(1..=30).contains(&sample_every)
        {
            return Err("Invalid sampled person inference input".into());
        }
        for (index, frame) in frames.chunks_exact(frame_bytes).enumerate() {
            if is_observation_frame(first_frame + index as u64, sample_every) {
                consume(index, self.detect(frame)?)?;
            } else {
                consume(index, Vec::new())?;
            }
        }
        Ok(())
    }

    fn detect_rfdetr_medium(&mut self, rgb: &[u8]) -> Result<Vec<PersonObservation>, String> {
        let input = preprocess_rfdetr_medium(rgb)?;
        let tensor = Tensor::from_array((
            vec![1_i64, 3, RFDETR_SIDE as i64, RFDETR_SIDE as i64],
            input,
        ))
        .map_err(|e| format!("RF-DETR tensor: {e}"))?;
        let outputs = self
            .detector
            .run(ort::inputs!["input" => tensor])
            .map_err(|e| format!("RF-DETR inference: {e}"))?;
        let (box_shape, boxes) = outputs
            .get("pred_boxes")
            .ok_or("RF-DETR pred_boxes missing")?
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("RF-DETR boxes: {e}"))?;
        let (logit_shape, logits) = outputs
            .get("pred_logits")
            .ok_or("RF-DETR pred_logits missing")?
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("RF-DETR logits: {e}"))?;
        if box_shape.len() != 3
            || box_shape[0] != 1
            || box_shape[2] != 4
            || logit_shape.len() != 3
            || logit_shape[0] != 1
            || logit_shape[1] != box_shape[1]
            || logit_shape[2] != 91
        {
            return Err(format!(
                "RF-DETR tensor shape mismatch: {box_shape:?}/{logit_shape:?}"
            ));
        }
        let proposals = decode_rfdetr_medium(boxes, logits)?;
        drop(outputs);
        if self.pose.is_some() {
            self.collect_observations(rgb, proposals)
        } else {
            Ok(rfdetr_detector_observations(proposals))
        }
    }

    fn collect_observations(
        &mut self,
        rgb: &[u8],
        mut proposals: Vec<(f32, [f64; 4])>,
    ) -> Result<Vec<PersonObservation>, String> {
        proposals.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut distinct: Vec<(f32, [f64; 4])> = Vec::new();
        for candidate in proposals {
            distinct.push(candidate);
            if distinct.len() >= MAX_PERSONS * 3 {
                break;
            }
        }
        let mut results: Vec<PersonObservation> = Vec::new();
        for group in distinct.chunks(8) {
            if results.len() == MAX_PERSONS {
                break;
            }
            let valid: Vec<_> = group
                .iter()
                .copied()
                .filter(|(_, crop)| {
                    (crop[2] - crop[0]) * (W as f64) >= 8.0
                        && (crop[3] - crop[1]) * (H as f64) >= 20.0
                })
                .collect();
            let crops: Vec<_> = valid.iter().map(|(_, crop)| *crop).collect();
            let pose_result = self
                .pose
                .as_mut()
                .ok_or("Person engine missing requested pose model")?
                .infer_batch(rgb, &crops);
            let poses = match pose_result {
                Ok(poses) => poses,
                Err(_) => vec![
                    [Keypoint {
                        x: 0.0,
                        y: 0.0,
                        score: 0.0,
                    }; KPT_COUNT];
                    crops.len()
                ],
            };
            for ((det_score, crop), keypoints) in valid.into_iter().zip(poses) {
                if results.len() == MAX_PERSONS {
                    break;
                }
                // A very small or border-truncated proposal is not a runtime
                // failure; RTMPose needs at least an 8x20 original-pixel crop.
                if (crop[2] - crop[0]) * (W as f64) < 8.0 || (crop[3] - crop[1]) * (H as f64) < 20.0
                {
                    continue;
                }
                let small_far_player = crop[3] - crop[1] <= 0.17;
                let skeleton = skeleton_box(&keypoints).or_else(|| {
                    small_far_player
                        .then(|| skeleton_box_with_threshold(&keypoints, 0.28, 9))
                        .flatten()
                });
                let bbox = crop;
                let pose_score = skeleton.map_or(0.0, |(_, score)| score);
                // Very-low detector confidence is valid only with strong evidence
                // from a coherent pose; never admit a weak silhouette alone.
                if det_score < 0.24 && pose_score < 0.49 {
                    continue;
                }
                let head = (keypoints[0].score >= 0.32)
                    .then_some([keypoints[0].x as f64, keypoints[0].y as f64]);
                let observation = PersonObservation {
                    bbox,
                    score: det_score.max(pose_score).clamp(0.0, 1.0),
                    head,
                    keypoints: keypoints
                        .map(|point| [point.x as f64, point.y as f64, point.score as f64]),
                };
                results.push(observation);
            }
        }
        Ok(results)
    }
}

fn rfdetr_detector_observations(mut candidates: Vec<(f32, [f64; 4])>) -> Vec<PersonObservation> {
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    candidates
        .into_iter()
        .filter(|(score, bbox)| {
            score.is_finite()
                && *score >= RFDETR_MIN_SCORE
                && bbox.iter().all(|value| value.is_finite())
                && (bbox[2] - bbox[0]) * W as f64 >= 8.0
                && (bbox[3] - bbox[1]) * H as f64 >= 20.0
        })
        .take(MAX_PERSONS)
        .map(|(score, bbox)| PersonObservation {
            bbox,
            score,
            head: None,
            keypoints: [[0.0; 3]; KPT_COUNT],
        })
        .collect()
}

/// Decode the vetted RF-DETR Medium sparse COCO head: 300 cxcywh boxes
/// and 91 raw category logits, with COCO person in column 1.
fn decode_rfdetr_medium(boxes: &[f32], logits: &[f32]) -> Result<Vec<(f32, [f64; 4])>, String> {
    if boxes.len() != 300 * 4 || logits.len() != 300 * 91 {
        return Err(format!(
            "RF-DETR unexpected output lengths: boxes {}, logits {}",
            boxes.len(),
            logits.len()
        ));
    }
    let mut proposals = Vec::new();
    for (bbox, scores) in boxes.chunks_exact(4).zip(logits.chunks_exact(91)) {
        let [cx, cy, w, h] = [bbox[0], bbox[1], bbox[2], bbox[3]];
        let raw = scores[1];
        if ![cx, cy, w, h, raw].iter().all(|v| v.is_finite()) {
            continue;
        }
        let confidence = 1.0 / (1.0 + (-raw.clamp(-50.0, 50.0)).exp());
        if confidence < RFDETR_MIN_SCORE || w <= 0.0 || h <= 0.0 {
            continue;
        }
        let b = [
            (cx - w * 0.5) as f64,
            (cy - h * 0.5) as f64,
            (cx + w * 0.5) as f64,
            (cy + h * 0.5) as f64,
        ]
        .map(|v| v.clamp(0.0, 1.0));
        if (b[2] - b[0]) * (W as f64) < 8.0 || (b[3] - b[1]) * (H as f64) < 20.0 {
            continue;
        }
        proposals.push((confidence, b));
    }
    Ok(proposals)
}

/// Direct square RGB resize with ImageNet normalization, matching the tested
/// RF-DETR ONNX export. No letterboxing (the detector was tested this way).
fn preprocess_rfdetr_medium(rgb: &[u8]) -> Result<Vec<f32>, String> {
    if rgb.len() != W * H * 3 {
        return Err("RF-DETR expects 640x360 RGB24 frames".into());
    }
    let mut output = vec![0.0f32; 3 * RFDETR_SIDE * RFDETR_SIDE];
    for y in 0..RFDETR_SIDE {
        let source_y =
            ((y as f32 + 0.5) * H as f32 / RFDETR_SIDE as f32 - 0.5).clamp(0.0, (H - 1) as f32);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(H - 1);
        let fy = source_y - y0 as f32;
        for x in 0..RFDETR_SIDE {
            let source_x =
                ((x as f32 + 0.5) * W as f32 / RFDETR_SIDE as f32 - 0.5).clamp(0.0, (W - 1) as f32);
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(W - 1);
            let fx = source_x - x0 as f32;
            for channel in 0..3 {
                let tl = rgb[(y0 * W + x0) * 3 + channel] as f32;
                let tr = rgb[(y0 * W + x1) * 3 + channel] as f32;
                let bl = rgb[(y1 * W + x0) * 3 + channel] as f32;
                let br = rgb[(y1 * W + x1) * 3 + channel] as f32;
                let top = tl + (tr - tl) * fx;
                let bottom = bl + (br - bl) * fx;
                let pixel = (top + (bottom - top) * fy).round().clamp(0.0, 255.0);
                let index = channel * RFDETR_SIDE * RFDETR_SIDE + y * RFDETR_SIDE + x;
                output[index] = (pixel / 255.0 - RFDETR_MEAN[channel]) / RFDETR_STD[channel];
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_detection_uses_global_three_hz_anchors_across_window_boundaries() {
        let anchors = (0..36)
            .filter(|&frame| is_observation_frame(frame, 10))
            .collect::<Vec<_>>();
        assert_eq!(anchors, vec![0, 10, 20, 30]);
        assert_eq!(
            (0..9)
                .filter(|&offset| is_observation_frame(9 + offset, 10))
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(
            (0..9)
                .filter(|&offset| is_observation_frame(18 + offset, 10))
                .collect::<Vec<_>>(),
            vec![2]
        );
    }

    #[test]
    fn custom_sampling_intervals_remain_global_across_nine_frame_windows() {
        for step in [1u64, 2, 3, 5, 7, 10, 30] {
            let expected = (0..90)
                .filter(|frame| frame % step == 0)
                .collect::<Vec<_>>();
            let sampled = (0..10)
                .flat_map(|window| (0..9).map(move |offset| window * 9 + offset))
                .filter(|&frame| is_observation_frame(frame, step))
                .collect::<Vec<_>>();
            assert_eq!(sampled, expected, "sampling step {step}");
        }
    }

    #[test]
    fn rfdetr_full_gpu_evidence_defers_pose_while_other_entrypoints_keep_it() {
        assert!(!DetectorVariant::RfDetrMedium.needs_pose_in_gpu_evidence());
        assert_eq!(
            DetectorVariant::from_env().unwrap_or(DetectorVariant::RfDetrMedium),
            DetectorVariant::RfDetrMedium
        );
    }

    #[test]
    fn rfdetr_detection_returns_boxes_without_recomputing_or_inventing_pose() {
        let primary = [0.1, 0.1, 0.3, 0.7];
        let observations = rfdetr_detector_observations(vec![
            (0.4, [0.5, 0.5, 0.505, 0.6]),
            (0.5, [0.4, 0.2, 0.6, 0.9]),
            (0.9, primary),
            (f32::NAN, primary),
        ]);
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].bbox, primary);
        assert_eq!(observations[0].score, 0.9);
        for observation in &observations {
            assert!(observation.head.is_none());
            assert_eq!(observation.keypoints, [[0.0; 3]; KPT_COUNT]);
        }
    }

    #[test]
    fn rfdetr_rejects_invalid_frame_and_preserves_flat_color() {
        assert!(preprocess_rfdetr_medium(&[]).is_err());
        let input = vec![255u8; W * H * 3];
        let prepared = preprocess_rfdetr_medium(&input).unwrap();
        assert_eq!(prepared.len(), 3 * RFDETR_SIDE * RFDETR_SIDE);
        for channel in 0..3 {
            let pixel = prepared[channel * RFDETR_SIDE * RFDETR_SIDE];
            let expected = (1.0 - RFDETR_MEAN[channel]) / RFDETR_STD[channel];
            assert!((pixel - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn rfdetr_decodes_sparse_coco_person_column_and_filters_bad_boxes() {
        let mut boxes = vec![0.0f32; 300 * 4];
        let mut logits = vec![-20.0f32; 300 * 91];
        boxes[0..4].copy_from_slice(&[0.5, 0.5, 0.2, 0.6]);
        logits[1] = 5.0;
        boxes[4..8].copy_from_slice(&[0.2, 0.2, 0.2, 0.2]);
        logits[91 + 18] = 10.0; // a dog is not a person.
        boxes[8..12].copy_from_slice(&[f32::NAN, 0.5, 0.4, 0.4]);
        logits[2 * 91 + 1] = 20.0;
        let results = decode_rfdetr_medium(&boxes, &logits).unwrap();
        assert_eq!(results.len(), 1);
        let (score, bbox) = results[0];
        assert!(score > 0.99);
        for (actual, expected) in bbox.into_iter().zip([0.4, 0.2, 0.6, 0.8]) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert!(decode_rfdetr_medium(&boxes[..8], &logits).is_err());
    }

    #[test]
    #[ignore = "Run with PERSON_PREVIEW_RGB and CUDA runtime on host"]
    fn pipelined_frames_match_sequential_gpu_results() {
        let rgb = std::fs::read(std::env::var("PERSON_PREVIEW_RGB").unwrap()).unwrap();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let mut engine = PersonEngine::load(root, Device::Cuda, 8, 0).unwrap();
        let expected = format!("{:?}", engine.detect(&rgb).unwrap());
        let mut observed = Vec::new();
        engine
            .detect_frames(&rgb.repeat(3), |_, poses| {
                observed.push(format!("{poses:?}"));
                Ok(())
            })
            .unwrap();
        assert_eq!(observed, vec![expected; 3]);
    }
    fn sample(x: f32, y: f32, score: f32) -> Keypoint {
        Keypoint { x, y, score }
    }

    #[test]
    fn far_court_pose_threshold_recovers_occluded_small_athlete() {
        let mut points = [sample(0.0, 0.0, 0.0); KPT_COUNT];
        for (i, point) in points.iter_mut().enumerate() {
            *point = sample(
                0.43 + (i % 4) as f32 * 0.007,
                0.42 + (i / 4) as f32 * 0.02,
                if (5..=12).contains(&i) { 0.33 } else { 0.43 },
            );
        }
        assert!(skeleton_box(&points).is_none());
        assert!(skeleton_box_with_threshold(&points, 0.28, 9).is_some());
    }

    #[test]
    fn bbox_is_derived_from_body_points_not_detector_box() {
        let mut pts = [sample(0.0, 0.0, 0.0); KPT_COUNT];
        for (i, p) in pts.iter_mut().enumerate() {
            *p = sample(
                0.40 + (i % 4) as f32 * 0.02,
                0.25 + (i / 4) as f32 * 0.08,
                0.9,
            );
        }
        let (bbox, score) = skeleton_box(&pts).unwrap();
        assert!(bbox[0] < 0.4 && bbox[2] > 0.46);
        assert!(score > 0.8);
        assert!(skeleton_box(&[sample(0.5, 0.5, 0.9); KPT_COUNT]).is_none());
    }

    #[test]
    #[ignore = "Run with PERSON_PREVIEW_RGB and ORT_DYLIB_PATH on host"]
    fn actual_onnx_model_detects_pose_people_without_ravel() {
        let path = std::env::var("PERSON_PREVIEW_RGB").expect("raw RGB path");
        let rgb = std::fs::read(path).expect("prepared RGB frame");
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap();
        let mut engine = PersonEngine::load(root, Device::Cpu, 4, 0).unwrap();
        let result = engine.detect(&rgb).unwrap();
        println!(
            "RF-DETR + RTMPose players: {:?}",
            result.iter().map(|p| (p.bbox, p.score)).collect::<Vec<_>>()
        );
        assert!(
            !result.is_empty(),
            "Expected at least one player in match frame"
        );
    }

    #[test]
    fn refuses_missing_torso_or_incomplete_frame() {
        let mut points = [sample(0.5, 0.5, 0.9); KPT_COUNT];
        points[5].score = 0.1;
        assert!(skeleton_box(&points).is_none());
        assert!(preprocess_rfdetr_medium(&vec![0; 3 * W * H - 1]).is_err());
        assert_eq!(
            preprocess_rfdetr_medium(&vec![0; 3 * W * H]).unwrap().len(),
            3 * RFDETR_SIDE * RFDETR_SIDE
        );
    }
}
