use crate::{
    pose::{Keypoint, PoseEngine, KPT_COUNT},
    Device,
};
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider},
    io_binding::IoBinding,
    memory::{Allocator, MemoryInfo},
    session::Session,
    value::Tensor,
};

use std::path::Path;

pub const PERSON_SAMPLE_PERIOD_FRAMES: usize = 1;
const W: usize = 640;
const H: usize = 360;
const MAX_PERSONS: usize = 12;

fn is_observation_frame(frame: u64, period: u64) -> bool {
    frame.is_multiple_of(period)
}

pub const RFDETR_MEDIUM_FILE: &str = "rf-detr-medium.onnx";
pub const RFDETR_MEDIUM_SHA256: &str =
    "18250b5a3a9552ae117d592f8e8c27fe52438d70c914a77a8913fc45d8f29e69";
pub const RFDETR_MEDIUM_BATCH4_FILE: &str = "rf-detr-medium-batch4.onnx";
pub const RFDETR_MEDIUM_BATCH4_SHA256: &str =
    "9055ef9b1b76b79412069590654baf913cd173972be4ff4686c1535bb021e1f6";
pub const RFDETR_MEDIUM_BATCH8_FP16_FILE: &str = "rf-detr-medium-batch8-fp16.onnx";
pub const RFDETR_MEDIUM_BATCH8_FP16_SHA256: &str =
    "f6b2a35ee7f157d7f7d7a0d93f5a16e5e32dd05975ab05553464257456057b6e";
const RFDETR_LARGE_FILE: &str = "rf-detr-large.onnx";
const RFDETR_LARGE_SHA256: &str =
    "da7f1bc118c6071b48096149b5df01eb1caeb3bb60a262949ac0e41f7c96f705";
// ByteTrack stage two requires detector boxes that the old 0.28 cutoff discarded.
// Low-score boxes are NOT eligible to start a new Track ID.
const RFDETR_MIN_SCORE: f32 = 0.10;
const RFDETR_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const RFDETR_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum DetectorVariant {
    #[default]
    #[serde(rename = "rfdetr-medium")]
    RfDetrMedium,
    #[serde(rename = "rfdetr-large")]
    RfDetrLarge,
}

impl DetectorVariant {
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("VOLLEY_PERSON_DETECTOR") {
            Err(std::env::VarError::NotPresent) => Ok(Self::RfDetrMedium),
            Err(error) => Err(format!("Invalid VOLLEY_PERSON_DETECTOR: {error}")),
            Ok(value) if value == "rfdetr-medium" => Ok(Self::RfDetrMedium),
            Ok(value) if value == "rfdetr-large" => Ok(Self::RfDetrLarge),
            Ok(value) => Err(format!(
                "Unknown VOLLEY_PERSON_DETECTOR={value:?}; expected rfdetr-medium or rfdetr-large"
            )),
        }
    }
    fn needs_pose_in_gpu_evidence(self) -> bool {
        false
    }
    pub fn from_pipeline_tag(tag: &str) -> Option<Self> {
        if tag.starts_with("rfdetr_medium_") {
            Some(Self::RfDetrMedium)
        } else if tag.starts_with("rfdetr_large_") {
            Some(Self::RfDetrLarge)
        } else {
            None
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Self::RfDetrMedium => "rfdetr-medium",
            Self::RfDetrLarge => "rfdetr-large",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::RfDetrMedium => "RF-DETR Medium",
            Self::RfDetrLarge => "RF-DETR Large",
        }
    }
    fn input_shape(self) -> [i64; 4] {
        let side = match self {
            Self::RfDetrMedium => 576,
            Self::RfDetrLarge => 704,
        };
        [1, 3, side, side]
    }
    fn output_names(self) -> [&'static str; 2] {
        match self {
            Self::RfDetrMedium => ["pred_boxes", "pred_logits"],
            Self::RfDetrLarge => ["dets", "labels"],
        }
    }
    fn file(self) -> &'static str {
        match self {
            Self::RfDetrMedium => RFDETR_MEDIUM_FILE,
            Self::RfDetrLarge => RFDETR_LARGE_FILE,
        }
    }
    fn sha256(self) -> &'static str {
        match self {
            Self::RfDetrMedium => RFDETR_MEDIUM_SHA256,
            Self::RfDetrLarge => RFDETR_LARGE_SHA256,
        }
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
    batch: usize,
    variant: DetectorVariant,
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
        Self::load_for_gpu_evidence_with_detector(
            repo,
            device,
            threads,
            memory_mb,
            DetectorVariant::from_env()?,
        )
    }

    pub fn load_for_gpu_evidence_with_detector(
        repo: &Path,
        device: Device,
        threads: usize,
        memory_mb: u64,
        variant: DetectorVariant,
    ) -> Result<Self, String> {
        Self::load_with_pose_policy(
            repo,
            device,
            threads,
            memory_mb,
            variant,
            variant.needs_pose_in_gpu_evidence(),
            1,
            true,
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
        Self::load_with_pose_policy(repo, device, threads, memory_mb, variant, true, 1, true)
    }

    pub fn primary_batch_size(variant: DetectorVariant, memory_mb: u64) -> usize {
        if variant == DetectorVariant::RfDetrMedium && (memory_mb == 0 || memory_mb >= 4096) {
            8
        } else if variant == DetectorVariant::RfDetrMedium && memory_mb >= 3072 {
            4
        } else {
            1
        }
    }

    pub fn load_for_primary_gpu_evidence(
        repo: &Path,
        threads: usize,
        memory_mb: u64,
        variant: DetectorVariant,
    ) -> Result<Self, String> {
        Self::load_with_pose_policy(
            repo,
            Device::Cuda,
            threads,
            memory_mb,
            variant,
            false,
            Self::primary_batch_size(variant, memory_mb),
            variant != DetectorVariant::RfDetrMedium,
        )
    }

    pub fn batch_size(&self) -> usize {
        self.batch
    }

    pub fn primary_model_file(variant: DetectorVariant, memory_mb: u64) -> &'static str {
        if Self::primary_batch_size(variant, memory_mb) == 8 {
            RFDETR_MEDIUM_BATCH8_FP16_FILE
        } else if Self::primary_batch_size(variant, memory_mb) == 4 {
            RFDETR_MEDIUM_BATCH4_FILE
        } else {
            variant.file()
        }
    }

    fn input_shape(&self) -> [i64; 4] {
        let mut shape = self.variant.input_shape();
        shape[0] = self.batch as i64;
        shape
    }

    #[expect(clippy::too_many_arguments)]
    fn load_with_pose_policy(
        repo: &Path,
        device: Device,
        threads: usize,
        memory_mb: u64,
        variant: DetectorVariant,
        with_pose: bool,
        batch: usize,
        tf32: bool,
    ) -> Result<Self, String> {
        if threads > 128 || memory_mb > 131072 {
            return Err("Invalid person inference resource settings".into());
        }
        let models = repo
            .join("models")
            .canonicalize()
            .map_err(|e| format!("Models root: {e}"))?;
        let (filename, sha) = if batch == 8 {
            (
                RFDETR_MEDIUM_BATCH8_FP16_FILE,
                RFDETR_MEDIUM_BATCH8_FP16_SHA256,
            )
        } else if batch == 4 {
            (RFDETR_MEDIUM_BATCH4_FILE, RFDETR_MEDIUM_BATCH4_SHA256)
        } else {
            (variant.file(), variant.sha256())
        };
        let path = models
            .join(filename)
            .canonicalize()
            .map_err(|e| format!("Person detector {} not installed: {e}", variant.file()))?;
        let size = path
            .metadata()
            .map_err(|error| format!("RF-DETR model metadata: {error}"))?
            .len();
        if !path.is_file() || !(1024..=crate::MAX_MODEL_BYTES).contains(&size) {
            return Err(format!(
                "Invalid person detector path or size: {}",
                variant.file()
            ));
        }
        crate::verify_model_sha(&path, sha)?;
        // Both GPU sessions explicitly require CUDA; never silently fall back.
        #[cfg(any(target_os = "linux", windows))]
        let _ = crate::runtime_library(device)?;
        ort::init()
            .commit()
            .map_err(|e| format!("ONNX Runtime init: {e}"))?;
        let mut builder =
            Session::builder().map_err(|e| format!("RF-DETR session builder: {e}"))?;
        if threads != 0 {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|e| e.to_string())?;
        }
        if device == Device::Cuda {
            let mut cuda = CUDAExecutionProvider::default()
                .with_device_id(0)
                .with_tf32(tf32);
            if !tf32 {
                cuda = cuda.with_conv_algorithm_search(
                    ort::execution_providers::cuda::CuDNNConvAlgorithmSearch::Heuristic,
                );
            }
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
        let required_outputs = variant.output_names();
        let mut expected_shape = variant.input_shape();
        expected_shape[0] = batch as i64;
        if !detector.inputs.iter().any(|i| {
            i.name == "input"
                && i.input_type
                    .tensor_shape()
                    .is_some_and(|s| s.as_ref() == expected_shape)
        }) || required_outputs
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
        Ok(Self {
            detector,
            batch,
            variant,
            pose,
        })
    }

    pub fn allocate_cuda_input(&self) -> Result<crate::cuda_input::CudaInput, String> {
        crate::cuda_input::allocate(&self.detector, &self.input_shape(), 0)
    }

    pub fn create_cuda_detector_binding(&self) -> Result<IoBinding, String> {
        let mut binding = self
            .detector
            .create_binding()
            .map_err(|e| format!("Create RF-DETR binding: {e}"))?;
        let allocator = Allocator::default();
        for name in self.variant.output_names() {
            let shape = self
                .detector
                .outputs
                .iter()
                .find(|output| output.name == name)
                .and_then(|output| output.output_type.tensor_shape());
            if let Some(shape) = shape.filter(|shape| {
                shape.len() == 3
                    && shape[0] == self.batch as i64
                    && (1..=1024).contains(&shape[1])
                    && matches!(shape[2], 4 | 91)
            }) {
                let tensor = Tensor::<f32>::new(&allocator, shape.to_vec())
                    .map_err(|e| format!("Allocate RF-DETR output {name}: {e}"))?;
                binding
                    .bind_output(name, tensor)
                    .map_err(|e| format!("Bind RF-DETR output {name}: {e}"))?;
            } else {
                binding
                    .bind_output_to_device(name, &MemoryInfo::default())
                    .map_err(|e| format!("Bind RF-DETR output {name}: {e}"))?;
            }
        }
        Ok(binding)
    }

    pub fn detect_cuda_input_bound(
        &mut self,
        input: &Tensor<f32>,
        binding: &mut IoBinding,
    ) -> Result<Vec<PersonObservation>, String> {
        if self.batch != 1 {
            return Err("RF-DETR single-frame binding requires batch one".into());
        }
        Ok(self.detect_cuda_batch_bound(input, binding, 1)?.remove(0))
    }

    pub fn detect_cuda_batch_bound(
        &mut self,
        input: &Tensor<f32>,
        binding: &mut IoBinding,
        count: usize,
    ) -> Result<Vec<Vec<PersonObservation>>, String> {
        if self.pose.is_some() {
            return Err("CUDA detector-only input cannot substitute RTMPose source pixels".into());
        }
        crate::cuda_input::validate(input, &self.input_shape(), 0)?;
        let [box_name, logit_name] = self.variant.output_names();
        binding
            .bind_input("input", input)
            .map_err(|e| format!("Bind RF-DETR input: {e}"))?;
        let result = (|| {
            let outputs = self
                .detector
                .run_binding(binding)
                .map_err(|e| format!("Run bound RF-DETR: {e}"))?;
            let (box_shape, boxes) = outputs
                .get(box_name)
                .ok_or("CUDA RF-DETR boxes missing")?
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("RF-DETR boxes: {e}"))?;
            let (logit_shape, logits) = outputs
                .get(logit_name)
                .ok_or("CUDA RF-DETR logits missing")?
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("RF-DETR logits: {e}"))?;
            decode_detector_batch(
                box_shape.as_ref(),
                boxes,
                logit_shape.as_ref(),
                logits,
                count,
            )
        })();
        binding.clear_inputs();
        result
    }

    pub fn detect_cuda_input(
        &mut self,
        input: &Tensor<f32>,
    ) -> Result<Vec<PersonObservation>, String> {
        if self.pose.is_some() {
            return Err("CUDA detector-only input cannot substitute RTMPose source pixels".into());
        }
        let [box_name, logit_name] = self.variant.output_names();
        let tensors = crate::cuda_input::run(
            &mut self.detector,
            input,
            &self.variant.input_shape(),
            "input",
            &[box_name, logit_name],
            0,
        )?;
        let boxes = tensors
            .iter()
            .find(|tensor| tensor.name == box_name)
            .ok_or("CUDA RF-DETR boxes missing")?;
        let logits = tensors
            .iter()
            .find(|tensor| tensor.name == logit_name)
            .ok_or("CUDA RF-DETR logits missing")?;
        decode_detector_output(
            &boxes.dimensions,
            &boxes.data,
            &logits.dimensions,
            &logits.data,
        )
    }

    pub fn detect(&mut self, rgb: &[u8]) -> Result<Vec<PersonObservation>, String> {
        if self.batch == 1 {
            self.detect_rfdetr(rgb)
        } else {
            Ok(self.detect_rgb_batch(&[rgb])?.remove(0))
        }
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
            consume(index, self.detect(frame)?)?;
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
        consume: impl FnMut(usize, Vec<PersonObservation>) -> Result<(), String>,
    ) -> Result<(), String> {
        self.detect_frames_selected(frames, first_frame, sample_every, |_| true, consume)
    }

    pub fn detect_frames_selected(
        &mut self,
        frames: &[u8],
        first_frame: u64,
        sample_every: u64,
        selected: impl Fn(u64) -> bool,
        mut consume: impl FnMut(usize, Vec<PersonObservation>) -> Result<(), String>,
    ) -> Result<(), String> {
        let frame_bytes = W * H * 3;
        if frames.is_empty()
            || !frames.len().is_multiple_of(frame_bytes)
            || !(1..=30).contains(&sample_every)
        {
            return Err("Invalid sampled person inference input".into());
        }
        let indices: Vec<_> = (0..frames.len() / frame_bytes)
            .filter(|&i| {
                let frame = first_frame + i as u64;
                is_observation_frame(frame, sample_every) && selected(frame)
            })
            .collect();
        let mut next = 0;
        for group in indices.chunks(self.batch) {
            let images: Vec<_> = group
                .iter()
                .map(|&i| &frames[i * frame_bytes..(i + 1) * frame_bytes])
                .collect();
            let found = if self.batch == 1 {
                vec![self.detect_rfdetr(images[0])?]
            } else {
                self.detect_rgb_batch(&images)?
            };
            for (&i, observations) in group.iter().zip(found) {
                while next < i {
                    consume(next, Vec::new())?;
                    next += 1;
                }
                consume(i, observations)?;
                next = i + 1;
            }
        }
        while next < frames.len() / frame_bytes {
            consume(next, Vec::new())?;
            next += 1;
        }
        Ok(())
    }

    fn detect_rgb_batch(
        &mut self,
        frames: &[&[u8]],
    ) -> Result<Vec<Vec<PersonObservation>>, String> {
        if frames.is_empty() || frames.len() > self.batch || self.pose.is_some() {
            return Err("Invalid RF-DETR RGB batch".into());
        }
        let shape = self.input_shape();
        let mut values = Vec::new();
        for frame in frames {
            values.extend(preprocess_rfdetr(frame, shape[2] as usize)?);
        }
        let count = frames.len();
        let slot = 3 * shape[2] as usize * shape[3] as usize;
        for _ in count..self.batch {
            values.extend_from_within((count - 1) * slot..count * slot);
        }
        let tensor = Tensor::from_array((shape.to_vec(), values))
            .map_err(|e| format!("RF-DETR batch tensor: {e}"))?;
        let [box_name, logit_name] = self.variant.output_names();
        let outputs = self
            .detector
            .run(ort::inputs!["input"=>tensor])
            .map_err(|e| format!("RF-DETR batch inference: {e}"))?;
        let (bs, boxes) = outputs
            .get(box_name)
            .ok_or("RF-DETR boxes missing")?
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let (ls, logits) = outputs
            .get(logit_name)
            .ok_or("RF-DETR logits missing")?
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        decode_detector_batch(bs.as_ref(), boxes, ls.as_ref(), logits, count)
    }

    fn detect_rfdetr(&mut self, rgb: &[u8]) -> Result<Vec<PersonObservation>, String> {
        let shape = self.variant.input_shape();
        let [box_name, logit_name] = self.variant.output_names();
        let input = preprocess_rfdetr(rgb, shape[2] as usize)?;
        let tensor = Tensor::from_array((shape.to_vec(), input))
            .map_err(|e| format!("RF-DETR tensor: {e}"))?;
        let outputs = self
            .detector
            .run(ort::inputs!["input" => tensor])
            .map_err(|e| format!("RF-DETR inference: {e}"))?;
        let (box_shape, boxes) = outputs
            .get(box_name)
            .ok_or("RF-DETR pred_boxes missing")?
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("RF-DETR boxes: {e}"))?;
        let (logit_shape, logits) = outputs
            .get(logit_name)
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
        let proposals = decode_rfdetr(boxes, logits)?;
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
fn decode_detector_output(
    box_shape: &[i64],
    boxes: &[f32],
    logit_shape: &[i64],
    logits: &[f32],
) -> Result<Vec<PersonObservation>, String> {
    if box_shape.first() != Some(&1) {
        return Err("RF-DETR expected one frame".into());
    }
    Ok(decode_detector_batch(box_shape, boxes, logit_shape, logits, 1)?.remove(0))
}

fn decode_detector_batch(
    box_shape: &[i64],
    boxes: &[f32],
    logit_shape: &[i64],
    logits: &[f32],
    count: usize,
) -> Result<Vec<Vec<PersonObservation>>, String> {
    let batch = box_shape.first().copied().unwrap_or_default();
    if !matches!(batch, 1 | 4 | 8)
        || count == 0
        || count > batch as usize
        || box_shape != [batch, 300, 4]
        || logit_shape != [batch, 300, 91]
        || boxes.len() != batch as usize * 1200
        || logits.len() != batch as usize * 27300
    {
        return Err("CUDA RF-DETR output dimensions are incompatible".into());
    }
    boxes
        .as_chunks::<1200>()
        .0
        .iter()
        .zip(logits.as_chunks::<27300>().0.iter())
        .take(count)
        .map(|(b, l)| Ok(rfdetr_detector_observations(decode_rfdetr(b, l)?)))
        .collect()
}

fn decode_rfdetr(boxes: &[f32], logits: &[f32]) -> Result<Vec<(f32, [f64; 4])>, String> {
    if boxes.len() != 300 * 4 || logits.len() != 300 * 91 {
        return Err(format!(
            "RF-DETR unexpected output lengths: boxes {}, logits {}",
            boxes.len(),
            logits.len()
        ));
    }
    let mut proposals = Vec::new();
    for (bbox, scores) in boxes
        .as_chunks::<4>()
        .0
        .iter()
        .zip(logits.as_chunks::<91>().0.iter())
    {
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
fn preprocess_rfdetr(rgb: &[u8], side: usize) -> Result<Vec<f32>, String> {
    if rgb.len() != W * H * 3 {
        return Err("RF-DETR expects 640x360 RGB24 frames".into());
    }
    let mut output = vec![0.0f32; 3 * side * side];
    for y in 0..side {
        let source_y = ((y as f32 + 0.5) * H as f32 / side as f32 - 0.5).clamp(0.0, (H - 1) as f32);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(H - 1);
        let fy = source_y - y0 as f32;
        for x in 0..side {
            let source_x =
                ((x as f32 + 0.5) * W as f32 / side as f32 - 0.5).clamp(0.0, (W - 1) as f32);
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
                let index = channel * side * side + y * side + x;
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
    fn retains_low_score_person_proposal_for_bytetrack_second_pass() {
        let boxes = rfdetr_detector_observations(vec![
            (0.17, [0.2, 0.2, 0.3, 0.8]),
            (0.09, [0.4, 0.2, 0.5, 0.8]),
        ]);
        assert_eq!(boxes.len(), 1);
        assert!((boxes[0].score - 0.17).abs() < 0.0001);
        assert_eq!(boxes[0].bbox, [0.2, 0.2, 0.3, 0.8]);
    }

    #[test]
    fn rfdetr_rejects_invalid_frame_and_preserves_flat_color() {
        assert!(preprocess_rfdetr(&[], 576).is_err());
        let input = vec![255u8; W * H * 3];
        let prepared = preprocess_rfdetr(&input, 576).unwrap();
        assert_eq!(prepared.len(), 3 * 576 * 576);
        for channel in 0..3 {
            let pixel = prepared[channel * 576 * 576];
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
        let results = decode_rfdetr(&boxes, &logits).unwrap();
        assert_eq!(results.len(), 1);
        let (score, bbox) = results[0];
        assert!(score > 0.99);
        for (actual, expected) in bbox.into_iter().zip([0.4, 0.2, 0.6, 0.8]) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert!(decode_rfdetr(&boxes[..8], &logits).is_err());
    }

    #[test]
    fn batch_four_decodes_each_frame_independently() {
        let mut boxes = vec![0.0; 4 * 300 * 4];
        let mut logits = vec![-20.0; 4 * 300 * 91];
        for frame in 0..4 {
            let x = 0.15 + frame as f32 * 0.2;
            boxes[frame * 1200..frame * 1200 + 4].copy_from_slice(&[x, 0.5, 0.1, 0.4]);
            logits[frame * 27300 + 1] = 5.0;
        }
        let result =
            decode_detector_batch(&[4, 300, 4], &boxes, &[4, 300, 91], &logits, 3).unwrap();
        assert_eq!(result.len(), 3);
        for (frame, observations) in result.iter().enumerate() {
            assert_eq!(observations.len(), 1);
            assert!((observations[0].bbox[0] - (0.1 + frame as f64 * 0.2)).abs() < 1e-6);
        }
        assert!(decode_detector_batch(&[4, 300, 4], &boxes, &[4, 300, 91], &logits, 5).is_err());
        assert!(
            decode_detector_batch(&[4, 300, 4], &boxes[..1200], &[4, 300, 91], &logits, 1).is_err()
        );
    }

    #[test]
    fn batch_eight_decodes_independent_frames_and_partial_tail() {
        let mut boxes = vec![0.0; 8 * 300 * 4];
        let mut logits = vec![-20.0; 8 * 300 * 91];
        for frame in 0..8 {
            let x = 0.1 + frame as f32 * 0.1;
            boxes[frame * 1200..frame * 1200 + 4].copy_from_slice(&[x, 0.5, 0.1, 0.4]);
            logits[frame * 27300 + 1] = 5.0;
        }
        for count in [1, 6, 8] {
            let result =
                decode_detector_batch(&[8, 300, 4], &boxes, &[8, 300, 91], &logits, count).unwrap();
            assert_eq!(result.len(), count);
            for (frame, observations) in result.iter().enumerate() {
                assert_eq!(observations.len(), 1);
                assert!((observations[0].bbox[0] - (0.05 + frame as f64 * 0.1)).abs() < 1e-6);
            }
        }
        assert!(decode_detector_batch(&[8, 300, 4], &boxes, &[8, 300, 91], &logits, 9).is_err());
        assert_eq!(
            PersonEngine::primary_model_file(DetectorVariant::RfDetrMedium, 0),
            RFDETR_MEDIUM_BATCH8_FP16_FILE
        );
        assert_eq!(
            PersonEngine::primary_model_file(DetectorVariant::RfDetrMedium, 3072),
            RFDETR_MEDIUM_BATCH4_FILE
        );
    }

    #[test]
    fn primary_batch_respects_memory_limits_and_variant() {
        assert_eq!(
            PersonEngine::primary_batch_size(DetectorVariant::RfDetrMedium, 0),
            8
        );
        assert_eq!(
            PersonEngine::primary_batch_size(DetectorVariant::RfDetrMedium, 3072),
            4
        );
        assert_eq!(
            PersonEngine::primary_batch_size(DetectorVariant::RfDetrMedium, 4096),
            8
        );
        assert_eq!(
            PersonEngine::primary_batch_size(DetectorVariant::RfDetrMedium, 2048),
            1
        );
        assert_eq!(
            PersonEngine::primary_batch_size(DetectorVariant::RfDetrLarge, 0),
            1
        );
    }

    #[test]
    #[ignore = "requires verified batch model, CUDA and external media"]
    fn primary_rfdetr_batch4_fp32_matches_single_frame_cuda() -> Result<(), String> {
        use crate::gpu_preprocess::{GpuBuffer, GpuPreprocessor};
        let root = std::path::PathBuf::from(
            std::env::var("VOLLEY_AB_MODEL_ROOT").map_err(|e| e.to_string())?,
        );
        let video = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-i"])
            .arg(std::env::var_os("VOLLEY_AB_SOURCE").expect("Set VOLLEY_AB_SOURCE"))
            .args([
                "-vf",
                "fps=5,scale=640:360,format=nv12",
                "-frames:v",
                "11",
                "-threads:v",
                "1",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .output()
            .map_err(|e| e.to_string())?;
        let bytes = W * H * 3 / 2;
        assert!(video.status.success());
        assert_eq!(video.stdout.len(), 11 * bytes);
        let gpu = GpuPreprocessor::new()?;
        let driver = mini_cuda::driver::try_init_driver().ok_or("CUDA driver unavailable")?;
        let source = GpuBuffer::allocate(bytes)?;
        let view = source.view(W, W, H)?;
        let upload = |pixels: &[u8]| {
            assert_eq!(
                unsafe { (driver.cuMemcpyHtoD_v2)(view.ptr, pixels.as_ptr().cast(), bytes) },
                mini_cuda::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                unsafe { (driver.cuStreamSynchronize)(std::ptr::null_mut()) },
                mini_cuda::CUresult::CUDA_SUCCESS
            );
        };
        let download = |ptr: u64| {
            let mut values = vec![0.0f32; 3 * 576 * 576];
            assert_eq!(
                unsafe {
                    (driver.cuMemcpyDtoH_v2)(values.as_mut_ptr().cast(), ptr, values.len() * 4)
                },
                mini_cuda::CUresult::CUDA_SUCCESS
            );
            values
        };
        let mut expected_inputs = Vec::new();
        let mut reference = Vec::new();
        {
            let mut single = PersonEngine::load_for_primary_gpu_evidence(
                &root,
                2,
                2048,
                DetectorVariant::RfDetrMedium,
            )?;
            let input = single.allocate_cuda_input()?;
            for pixels in video.stdout.chunks_exact(bytes) {
                upload(pixels);
                gpu.person_input(view, &input, 1)?;
                gpu.finish()?;
                expected_inputs.push(download(input.data_ptr() as u64));
                reference.push(single.detect_cuda_input(&input)?);
            }
        }
        let mut batched = PersonEngine::load_for_primary_gpu_evidence(
            &root,
            2,
            3072,
            DetectorVariant::RfDetrMedium,
        )?;
        assert_eq!(batched.batch_size(), 4);
        let mut input = batched.allocate_cuda_input()?;
        let mut actual = Vec::new();
        let mut on_main = Vec::new();
        let mut offset = 0usize;
        let mut binding = batched.create_cuda_detector_binding()?;
        for chunk in video.stdout.chunks(4 * bytes) {
            let count = chunk.len() / bytes;
            for (slot, pixels) in chunk.chunks_exact(bytes).enumerate() {
                upload(pixels);
                gpu.person_input_slot(view, &input, 1, slot)?;
                gpu.finish()?;
            }
            gpu.pad_person_input(&input, count)?;
            for slot in 0..count {
                let actual_input =
                    download(input.data_ptr() as u64 + (slot * 3 * 576 * 576 * 4) as u64);
                let differing = actual_input
                    .iter()
                    .zip(&expected_inputs[offset + slot])
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(
                    differing,
                    0,
                    "GPU tensor differs for frame {}",
                    offset + slot
                );
            }
            offset += count;
            let serial = batched.detect_cuda_batch_bound(&input, &mut binding, count)?;
            on_main.extend(serial);
            let detections = std::thread::scope(|scope| {
                let input = &mut input;
                let batched = &mut batched;
                let binding = &mut binding;
                scope
                    .spawn(move || {
                        crate::gpu_preprocess::bind_cuda_thread()?;
                        batched.detect_cuda_batch_bound(input, binding, count)
                    })
                    .join()
                    .map_err(|_| "Batch CUDA worker panicked".to_owned())?
            })?;
            assert_eq!(detections.len(), count);
            actual.extend(detections);
        }
        assert_eq!(actual.len(), reference.len());
        assert!(
            format!("{actual:?}") == format!("{on_main:?}"),
            "cross-thread binding changed predictions"
        );
        let mut max_box_error = 0.0f64;
        let mut max_score_error = 0.0f32;
        let mut mismatched_counts = 0usize;
        assert!(reference.iter().any(|x| !x.is_empty()));
        let mut matched = 0;
        for (frame, (expected, observed)) in reference.iter().zip(actual.iter()).enumerate() {
            if expected.len() != observed.len() {
                println!(
                    "COUNT frame={frame} expected={} actual={}",
                    expected.len(),
                    observed.len()
                );
                mismatched_counts += 1;
                continue;
            }
            let mut used = vec![false; observed.len()];
            for a in expected {
                let (i, b) = observed
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !used[*i])
                    .max_by(|(_, a1), (_, b1)| {
                        let error = |item: &PersonObservation| {
                            a.bbox
                                .iter()
                                .zip(item.bbox)
                                .map(|(l, r)| (l - r).abs())
                                .fold(0.0, f64::max)
                        };
                        error(b1).total_cmp(&error(a1))
                    })
                    .ok_or("Missing paired RF-DETR box")?;
                used[i] = true;
                let error = a
                    .bbox
                    .iter()
                    .zip(b.bbox)
                    .map(|(l, r)| (l - r).abs())
                    .fold(0.0, f64::max);
                max_box_error = max_box_error.max(error);
                max_score_error = max_score_error.max((a.score - b.score).abs());
                matched += 1;
            }
        }
        println!("Batch 4 FP32: {matched} matching native CUDA boxes; max_box_error={max_box_error} max_score_error={max_score_error} mismatched_counts={mismatched_counts}");
        assert!(max_box_error < 0.002 && max_score_error < 0.01 && mismatched_counts == 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires local CUDA/ORT, model weights and external media"]
    fn reused_rfdetr_cuda_binding_matches_fresh_binding() {
        let root = std::path::PathBuf::from(std::env::var("VOLLEY_AB_MODEL_ROOT").unwrap());
        let rgb = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-i"])
            .arg(std::env::var_os("VOLLEY_AB_SOURCE").expect("Set VOLLEY_AB_SOURCE"))
            .args([
                "-vf",
                "fps=5,scale=640:360,format=rgb24",
                "-frames:v",
                "3",
                "-threads:v",
                "1",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .output()
            .unwrap();
        assert!(
            rgb.status.success(),
            "{}",
            String::from_utf8_lossy(&rgb.stderr)
        );
        assert_eq!(rgb.stdout.len(), 3 * W * H * 3);
        crate::gpu_preprocess::bind_cuda_thread().unwrap();
        let driver = mini_cuda::driver::try_init_driver().unwrap();
        let mut engine = PersonEngine::load_for_gpu_evidence_with_detector(
            &root,
            Device::Cuda,
            2,
            0,
            DetectorVariant::RfDetrMedium,
        )
        .unwrap();
        let inputs = [
            engine.allocate_cuda_input().unwrap(),
            engine.allocate_cuda_input().unwrap(),
        ];
        let prepared = rgb
            .stdout
            .as_chunks::<{ W * H * 3 }>()
            .0
            .iter()
            .map(|frame| preprocess_rfdetr(frame, 576).unwrap())
            .collect::<Vec<_>>();
        let upload = |input: &crate::cuda_input::CudaInput, data: &[f32]| {
            assert_eq!(
                unsafe {
                    (driver.cuMemcpyHtoD_v2)(
                        input.data_ptr() as u64,
                        data.as_ptr().cast(),
                        std::mem::size_of_val(data),
                    )
                },
                mini_cuda::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                unsafe { (driver.cuStreamSynchronize)(std::ptr::null_mut()) },
                mini_cuda::CUresult::CUDA_SUCCESS
            );
        };
        let mut expected = Vec::new();
        for (i, data) in prepared.iter().enumerate() {
            let input = &inputs[i % 2];
            upload(input, data);
            expected.push(engine.detect_cuda_input(input).unwrap());
        }
        assert!(expected.iter().any(|frame| !frame.is_empty()));
        let mut binding = engine.create_cuda_detector_binding().unwrap();
        for (i, data) in prepared.iter().enumerate().rev() {
            let input = &inputs[i % 2];
            upload(input, data);
            let actual = engine.detect_cuda_input_bound(input, &mut binding).unwrap();
            assert_eq!(format!("{actual:?}"), format!("{:?}", expected[i]));
        }
        println!("RF-DETR fresh/reused binding: identical detections for three distinct frames and two reused CUDA tensors");
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
    fn actual_onnx_model_detects_pose_people_with_rfdetr() {
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
        assert!(preprocess_rfdetr(&vec![0; 3 * W * H - 1], 576).is_err());
        assert_eq!(
            preprocess_rfdetr(&vec![0; 3 * W * H], 576).unwrap().len(),
            3 * 576 * 576
        );
    }
}
