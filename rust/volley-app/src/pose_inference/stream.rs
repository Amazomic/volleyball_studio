//! Deferred RTMPose publication over the same RGB frame stream as VBallNet
//! and the detector. Nothing becomes a reusable pose run until fusion SHA
//! and frame/source identities are verified.
use super::*;
use std::{
    cell::RefCell,
    collections::VecDeque,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::{atomic::Ordering, mpsc, Arc},
    thread,
    time::Duration,
};
use volley_inference::gpu_preprocess::{GpuBuffer, GpuPreprocessor};
use volley_media::raw_decoder::{PrimaryFrameBatch, MAX_NATIVE_RGB_FRAME_BYTES};

const DETECTOR_RGB_FRAME_BYTES: usize = 640 * 360 * 3;

#[derive(Debug)]
enum FramePixels {
    Skipped,
    Host {
        rgb: Arc<Vec<u8>>,
        offset: usize,
    },
    Cuda {
        frame: Arc<GpuBuffer>,
        pitch: usize,
        matrix: u8,
    },
}
#[derive(Debug)]
struct PendingFrame {
    index: u64,
    pixels: FramePixels,
    width: usize,
    height: usize,
    boxes: Vec<[f64; 4]>,
}

fn ready(pending: &mut VecDeque<PendingFrame>, latest: u64) -> Vec<PendingFrame> {
    let mut samples = Vec::new();
    while pending
        .front()
        .is_some_and(|s| s.index.saturating_add(interpolation::MAX_GAP_FRAMES) <= latest)
    {
        if let Some(sample) = pending.pop_front() {
            samples.push(sample);
        }
    }
    samples
}

#[derive(Debug)]
struct PoseArtifact {
    folder: PathBuf,
    committed: bool,
}
impl Drop for PoseArtifact {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_dir_all(&self.folder);
        }
    }
}

pub(crate) struct PoseStream {
    writer: Option<BufWriter<File>>,
    artifact: Option<PoseArtifact>,
    engine: PoseEngine,
    cuda_processor: Option<GpuPreprocessor>,
    cuda_pixels: bool,
    hash: Sha256,
    model_sha: String,
    sample_stride: usize,
    pending: VecDeque<PendingFrame>,
    history: VecDeque<interpolation::FrameBoxes>,
    next_frame: u64,
    native_width: usize,
    native_height: usize,
    sampled: u64,
    people: u64,
    observed_crops: u64,
    interpolated_crops: u64,
    bytes: u64,
    original_source: Value,
    source_identity: Value,
    project_id: String,
    video: String,
}

#[derive(Debug)]
pub(crate) struct PendingPose {
    artifact: PoseArtifact,
    run_id: String,
    checksum: String,
    model_sha: String,
    sample_stride: usize,
    emitted_frames: u64,
    decoded_frames: u64,
    native_width: usize,
    native_height: usize,
    sampled: u64,
    people: u64,
    observed_crops: u64,
    interpolated_crops: u64,
    cuda_pixels: bool,
    original_source: Value,
    source_identity: Value,
    project_id: String,
    video: String,
}

enum PoseTask {
    Observe {
        first_frame: u64,
        frames: Vec<Option<Arc<GpuBuffer>>>,
        pitch: usize,
        matrix: u8,
        width: usize,
        height: usize,
        detected: Vec<RawDetection>,
        allow_skipped_frames: bool,
    },
    Tail(u64),
    Finish {
        decoded_frames: u64,
        windows: usize,
    },
}

pub(crate) struct PoseWorker {
    sender: Option<mpsc::SyncSender<PoseTask>>,
    handle: Option<thread::JoinHandle<Result<PendingPose, String>>>,
    cancelled: Arc<AtomicBool>,
}

impl PoseWorker {
    #[expect(clippy::too_many_arguments)]
    fn start(
        app: &AppState,
        project_id: &str,
        video: &str,
        original_source: Value,
        source_identity: Value,
        sample_stride: usize,
        gpu_memory_mb: u64,
        cpu_threads: usize,
    ) -> Result<Self, String> {
        let repository = app.repository_root.clone();
        let project_root = projects::project_root(app);
        let writes_enabled = app.writes_enabled;
        let project_id = project_id.to_owned();
        let video = video.to_owned();
        let (sender, receiver) = mpsc::sync_channel::<PoseTask>(2);
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = Arc::clone(&cancelled);
        let handle = thread::Builder::new()
            .name("volley-rtmpose".into())
            .spawn(move || {
                volley_inference::gpu_preprocess::bind_cuda_thread()?;
                let stage = PoseStream::new_at(
                    &repository,
                    &project_root,
                    writes_enabled,
                    &project_id,
                    &video,
                    original_source,
                    source_identity,
                    sample_stride,
                    gpu_memory_mb,
                    cpu_threads,
                );
                let _ = ready_sender.send(stage.as_ref().map(|_| ()).map_err(Clone::clone));
                let mut stage = stage?;
                while let Ok(task) = receiver.recv() {
                    if worker_cancelled.load(Ordering::Acquire) {
                        return Err("cancelled".into());
                    }
                    match task {
                        PoseTask::Observe {
                            first_frame,
                            frames,
                            pitch,
                            matrix,
                            width,
                            height,
                            detected,
                            allow_skipped_frames,
                        } => stage.observe_cuda(
                            first_frame,
                            frames,
                            pitch,
                            matrix,
                            width,
                            height,
                            &detected,
                            allow_skipped_frames,
                            &worker_cancelled,
                        )?,
                        PoseTask::Tail(frames) => stage.tail_cuda(frames, &worker_cancelled)?,
                        PoseTask::Finish {
                            decoded_frames,
                            windows,
                        } => {
                            return stage.finish(decoded_frames, windows, &worker_cancelled);
                        }
                    }
                }
                Err("RTMPose pipeline closed without a finish message".into())
            })
            .map_err(|error| format!("Spawn parallel RTMPose worker: {error}"))?;
        match ready_receiver.recv() {
            Ok(Ok(())) => Ok(Self {
                sender: Some(sender),
                handle: Some(handle),
                cancelled,
            }),
            result => {
                cancelled.store(true, Ordering::Release);
                drop(sender);
                let joined = handle.join();
                let reason = match result {
                    Ok(Err(error)) => error,
                    _ => match joined {
                        Ok(Err(error)) => error,
                        Ok(Ok(_)) => "RTMPose worker exited during startup".into(),
                        Err(_) => "RTMPose worker panicked during startup".into(),
                    },
                };
                Err(reason)
            }
        }
    }

    fn worker_error(&mut self) -> String {
        self.sender.take();
        match self.handle.take() {
            Some(handle) => match handle.join() {
                Ok(Err(reason)) => reason,
                Ok(Ok(_)) => "RTMPose pipeline closed unexpectedly".into(),
                Err(_) => "RTMPose worker panicked".into(),
            },
            None => "RTMPose pipeline is unavailable".into(),
        }
    }

    fn send(&mut self, task: PoseTask, cancel: &AtomicBool) -> Result<(), String> {
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let mut task = task;
        loop {
            if cancel.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }
            let sender = self
                .sender
                .as_ref()
                .ok_or("RTMPose pipeline is already finished")?;
            match sender.try_send(task) {
                Ok(()) => return Ok(()),
                Err(mpsc::TrySendError::Disconnected(_)) => return Err(self.worker_error()),
                Err(mpsc::TrySendError::Full(pending)) => {
                    task = pending;
                    thread::sleep(Duration::from_millis(5));
                }
            }
        }
    }

    fn finish(
        mut self,
        decoded_frames: u64,
        windows: usize,
        cancel: &AtomicBool,
    ) -> Result<PendingPose, String> {
        self.send(
            PoseTask::Finish {
                decoded_frames,
                windows,
            },
            cancel,
        )?;
        self.sender.take();
        let handle = self.handle.take().ok_or("Missing RTMPose worker thread")?;
        match handle.join() {
            Ok(result) => result,
            Err(_) => Err("RTMPose worker panicked before completing the evidence".into()),
        }
    }
}

impl Drop for PoseWorker {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub(crate) enum GpuPoseStage {
    Inline(Box<RefCell<PoseStream>>),
    Pipelined(RefCell<PoseWorker>),
}

impl GpuPoseStage {
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new(
        app: &AppState,
        project_id: &str,
        video: &str,
        original_source: Value,
        source_identity: Value,
        sample_stride: usize,
        gpu_memory_mb: u64,
        cpu_threads: usize,
    ) -> Result<Self, String> {
        if !matches!(
            std::env::var("VOLLEY_PARALLEL_POSE").ok().as_deref(),
            Some("1" | "true"),
        ) {
            Ok(Self::Inline(Box::new(RefCell::new(PoseStream::new(
                app,
                project_id,
                video,
                original_source,
                source_identity,
                sample_stride,
                gpu_memory_mb,
                cpu_threads,
            )?))))
        } else {
            Ok(Self::Pipelined(RefCell::new(PoseWorker::start(
                app,
                project_id,
                video,
                original_source,
                source_identity,
                sample_stride,
                gpu_memory_mb,
                cpu_threads,
            )?)))
        }
    }

    #[expect(clippy::too_many_arguments)]
    pub(crate) fn observe_cuda(
        &self,
        first_frame: u64,
        frames: Vec<Option<Arc<GpuBuffer>>>,
        pitch: usize,
        matrix: u8,
        width: usize,
        height: usize,
        detected: &[RawDetection],
        allow_skipped_frames: bool,
        cancelled: &AtomicBool,
    ) -> Result<(), String> {
        match self {
            Self::Inline(stage) => stage.borrow_mut().observe_cuda(
                first_frame,
                frames,
                pitch,
                matrix,
                width,
                height,
                detected,
                allow_skipped_frames,
                cancelled,
            ),
            Self::Pipelined(worker) => worker.borrow_mut().send(
                PoseTask::Observe {
                    first_frame,
                    frames,
                    pitch,
                    matrix,
                    width,
                    height,
                    detected: detected.to_vec(),
                    allow_skipped_frames,
                },
                cancelled,
            ),
        }
    }

    pub(crate) fn tail_cuda(
        &self,
        decoded_frames: u64,
        cancelled: &AtomicBool,
    ) -> Result<(), String> {
        match self {
            Self::Inline(stage) => stage.borrow_mut().tail_cuda(decoded_frames, cancelled),
            Self::Pipelined(worker) => worker
                .borrow_mut()
                .send(PoseTask::Tail(decoded_frames), cancelled),
        }
    }

    pub(crate) fn finish(
        self,
        decoded_frames: u64,
        windows: usize,
        cancelled: &AtomicBool,
    ) -> Result<PendingPose, String> {
        match self {
            Self::Inline(stage) => stage
                .into_inner()
                .finish(decoded_frames, windows, cancelled),
            Self::Pipelined(worker) => {
                worker
                    .into_inner()
                    .finish(decoded_frames, windows, cancelled)
            }
        }
    }
}

impl PoseStream {
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new(
        app: &AppState,
        project_id: &str,
        video: &str,
        original_source: Value,
        source_identity: Value,
        sample_stride: usize,
        gpu_memory_mb: u64,
        cpu_threads: usize,
    ) -> Result<Self, String> {
        Self::new_at(
            &app.repository_root,
            &projects::project_root(app),
            app.writes_enabled,
            project_id,
            video,
            original_source,
            source_identity,
            sample_stride,
            gpu_memory_mb,
            cpu_threads,
        )
    }

    #[expect(clippy::too_many_arguments)]
    fn new_at(
        repository: &Path,
        project_root: &Path,
        writes_enabled: bool,
        project_id: &str,
        video: &str,
        original_source: Value,
        source_identity: Value,
        sample_stride: usize,
        gpu_memory_mb: u64,
        cpu_threads: usize,
    ) -> Result<Self, String> {
        if !(1..=30).contains(&sample_stride) || !writes_enabled {
            return Err("Invalid shared pose stream configuration".into());
        }
        let model = repository.join("models").join(MODEL_FILE);
        safe_file(&model, 256 * 1024 * 1024)?;
        let model_sha = file_hash(&model)?;
        if model_sha != MODEL_SHA256 {
            return Err("RTMPose model fingerprint does not match the official ONNX".into());
        }
        let engine = PoseEngine::load(repository, Device::Cuda, cpu_threads, gpu_memory_mb)?;
        let runs = project_root.join(project_id).join("native_gpu_pose_runs");
        fs::create_dir_all(&runs).map_err(|error| format!("Create pose runs: {error}"))?;
        let run_id = Uuid::new_v4().simple().to_string();
        let folder = runs.join(&run_id);
        fs::create_dir(&folder).map_err(|error| format!("Create shared pose run: {error}"))?;
        let artifact = PoseArtifact {
            folder,
            committed: false,
        };
        let writer = BufWriter::new(
            File::create(artifact.folder.join("poses.jsonl.partial"))
                .map_err(|error| format!("Create streamed pose output: {error}"))?,
        );
        Ok(Self {
            writer: Some(writer),
            artifact: Some(artifact),
            engine,
            cuda_processor: None,
            cuda_pixels: false,
            hash: Sha256::new(),
            model_sha,
            sample_stride,
            pending: VecDeque::new(),
            history: VecDeque::new(),
            next_frame: 0,
            native_width: 0,
            native_height: 0,
            sampled: 0,
            people: 0,
            observed_crops: 0,
            interpolated_crops: 0,
            bytes: 0,
            original_source,
            source_identity,
            project_id: project_id.to_owned(),
            video: video.to_owned(),
        })
    }

    pub(crate) fn observe(
        &mut self,
        batch: &PrimaryFrameBatch,
        detected: &[RawDetection],
        in_scope: Option<&dyn Fn(u64) -> bool>,
        cancelled: &AtomicBool,
    ) -> Result<(), String> {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        if batch.first_frame != self.next_frame {
            return Err("Shared RTMPose frames are not a contiguous nine-frame RGB window".into());
        }
        let native = batch
            .native
            .as_ref()
            .ok_or("RTMPose requires original-resolution RGB frames")?;
        let frame_bytes = native
            .width
            .checked_mul(native.height)
            .and_then(|n| n.checked_mul(3))
            .filter(|n| *n > 0 && *n <= MAX_NATIVE_RGB_FRAME_BYTES)
            .ok_or("Invalid original-resolution RTMPose frame dimensions")?;
        if native.rgb.len() != frame_bytes * 9 {
            return Err("Native RTMPose frames do not match original dimensions".into());
        }
        if self.native_width == 0 {
            (self.native_width, self.native_height) = (native.width, native.height);
        } else if (self.native_width, self.native_height) != (native.width, native.height) {
            return Err("Original RTMPose frame dimensions changed during decoding".into());
        }
        for offset in 0..9 {
            let index = batch.first_frame + offset as u64;
            if !index.is_multiple_of(self.sample_stride as u64) {
                continue;
            }
            let begin = offset * frame_bytes;
            let boxes = boxes_from_raw_detections(detected, index)?;
            let pixels = if in_scope.is_some_and(|allows| !allows(index)) {
                if !boxes.is_empty() {
                    return Err("Excluded RTMPose frame has player detections".into());
                }
                FramePixels::Skipped
            } else {
                FramePixels::Host {
                    rgb: native.rgb.clone(),
                    offset: begin,
                }
            };
            self.pending.push_back(PendingFrame {
                index,
                pixels,
                width: native.width,
                height: native.height,
                boxes,
            });
        }
        self.next_frame += 9;
        if self.pending.len() > 40 {
            return Err("Pose lookahead queue exceeded its bounded capacity".into());
        }
        self.flush_ready(batch.first_frame + 8, cancelled)
    }

    #[expect(clippy::too_many_arguments)]
    pub(crate) fn observe_cuda(
        &mut self,
        first_frame: u64,
        frames: Vec<Option<Arc<GpuBuffer>>>,
        pitch: usize,
        matrix: u8,
        width: usize,
        height: usize,
        detected: &[RawDetection],
        allow_skipped_frames: bool,
        cancelled: &AtomicBool,
    ) -> Result<(), String> {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        if first_frame != self.next_frame || frames.len() != 9 {
            return Err("CUDA RTMPose frames must be a contiguous nine-frame window".into());
        }
        if self.native_width == 0 {
            (self.native_width, self.native_height) = (width, height);
        } else if (self.native_width, self.native_height) != (width, height) {
            return Err("CUDA RTMPose original dimensions changed during decoding".into());
        }
        if self.cuda_processor.is_none() {
            self.cuda_processor = Some(GpuPreprocessor::new()?);
        }
        self.cuda_pixels = true;
        for (offset, frame) in frames.into_iter().enumerate() {
            let index = first_frame + offset as u64;
            if !index.is_multiple_of(self.sample_stride as u64) {
                if frame.is_some() {
                    return Err("CUDA RTMPose received unexpected nonsampled source".into());
                }
                continue;
            }
            let boxes = boxes_from_raw_detections(detected, index)?;
            let pixels = match frame {
                Some(frame) => FramePixels::Cuda {
                    frame,
                    pitch,
                    matrix,
                },
                None if allow_skipped_frames && boxes.is_empty() => FramePixels::Skipped,
                None => return Err("Missing sampled CUDA RTMPose original frame".into()),
            };
            self.pending.push_back(PendingFrame {
                index,
                pixels,
                width,
                height,
                boxes,
            });
        }
        self.next_frame += 9;
        if self.pending.len() > 40 {
            return Err("CUDA pose lookahead queue exceeded its bounded capacity".into());
        }
        self.flush_ready(first_frame + 8, cancelled)
    }

    pub(crate) fn tail_cuda(
        &mut self,
        decoded_frames: u64,
        cancelled: &AtomicBool,
    ) -> Result<(), String> {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        decoded_frames
            .checked_sub(self.next_frame)
            .filter(|count| *count < 9)
            .ok_or("CUDA pose tail frame indices exceed the original stream")?;
        self.flush_ready(decoded_frames.saturating_sub(1), cancelled)
    }

    pub(crate) fn tail(
        &mut self,
        rgb_tail: &[u8],
        decoded_frames: u64,
        cancelled: &AtomicBool,
    ) -> Result<(), String> {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let count = decoded_frames
            .checked_sub(self.next_frame)
            .filter(|count| *count < 9)
            .ok_or("Pose tail frame indices exceed the 30-fps source")?;
        if rgb_tail.len() != count as usize * DETECTOR_RGB_FRAME_BYTES {
            return Err("Pose tail RGB bytes do not match the final frame count".into());
        }
        self.flush_ready(decoded_frames.saturating_sub(1), cancelled)
    }

    fn flush_ready(&mut self, latest: u64, cancelled: &AtomicBool) -> Result<(), String> {
        // Retain previous real anchors while a bounded future lookahead is
        // available. Interpolated proposals never become new anchor points.
        let snapshots: VecDeque<interpolation::FrameBoxes> = self
            .history
            .iter()
            .cloned()
            .chain(self.pending.iter().map(|item| interpolation::FrameBoxes {
                frame: item.index,
                boxes: item.boxes.clone(),
            }))
            .collect();
        for frame in ready(&mut self.pending, latest) {
            if cancelled.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }
            let proposals = if matches!(frame.pixels, FramePixels::Skipped) {
                Vec::new()
            } else {
                let inferred =
                    interpolation::interpolate_boxes(&snapshots, frame.index, &frame.boxes);
                pose_proposals(frame.boxes.clone(), inferred)
            };
            let mut poses = Vec::with_capacity(proposals.len());
            for group in proposals.chunks(8) {
                if cancelled.load(Ordering::Acquire) {
                    return Err("cancelled".into());
                }
                let boxes: Vec<_> = group.iter().map(|(bb, _)| *bb).collect();
                let points = match &frame.pixels {
                    FramePixels::Skipped => {
                        return Err("Skipped RTMPose frame has unexpected proposals".into());
                    }
                    FramePixels::Host { rgb, offset } => {
                        let frame_bytes = frame.width * frame.height * 3;
                        self.engine.infer_batch_original(
                            &rgb[*offset..*offset + frame_bytes],
                            frame.width,
                            frame.height,
                            &boxes,
                        )?
                    }
                    FramePixels::Cuda {
                        frame: nv12,
                        pitch,
                        matrix,
                    } => {
                        let view = nv12.view(*pitch, frame.width, frame.height)?;
                        let input = self.engine.allocate_cuda_pose_input(boxes.len())?;
                        self.cuda_processor
                            .as_ref()
                            .ok_or("CUDA pose preprocessor was not initialized")?
                            .pose_input(view, &input, &boxes, *matrix)?;
                        self.engine.infer_cuda_pose_input(
                            &input,
                            frame.width,
                            frame.height,
                            &boxes,
                        )?
                    }
                };
                for ((bb, origin), keypoints) in group.iter().zip(points) {
                    if *origin == "interpolated_box" {
                        self.interpolated_crops += 1;
                    } else {
                        self.observed_crops += 1;
                    }
                    let keypoints = keypoints.map(|p| [p.x as f64, p.y as f64, p.score as f64]);
                    poses.push(json!({"box": bb, "keypoints":keypoints, "crop_origin":origin}));
                }
            }
            self.history.push_back(interpolation::FrameBoxes {
                frame: frame.index,
                boxes: frame.boxes,
            });
            while self.history.front().is_some_and(|record| {
                record.frame.saturating_add(interpolation::MAX_GAP_FRAMES) < frame.index
            }) {
                self.history.pop_front();
            }
            self.people += poses.len() as u64;
            self.sampled += 1;
            let mut row = serde_json::to_vec(&json!({"frame":frame.index,"players":poses}))
                .map_err(|error| format!("Serialize streamed pose evidence: {error}"))?;
            row.push(b'\n');
            self.bytes += row.len() as u64;
            if row.len() > 64 * 1024 || self.bytes > MAX_FUSION {
                return Err("Streamed RTMPose evidence exceeds its bounded output limit".into());
            }
            self.writer
                .as_mut()
                .ok_or("Pose writer already finished")?
                .write_all(&row)
                .map_err(|error| format!("Write pose evidence: {error}"))?;
            self.hash.update(&row);
        }
        Ok(())
    }

    pub(crate) fn finish(
        mut self,
        decoded_frames: u64,
        windows: usize,
        cancelled: &AtomicBool,
    ) -> Result<PendingPose, String> {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let emitted_frames = windows as u64 * 9;
        if emitted_frames != self.next_frame
            || decoded_frames < emitted_frames
            || decoded_frames > emitted_frames + 8
        {
            return Err("Shared pose frames do not match the detector timeline".into());
        }
        self.flush_ready(u64::MAX, cancelled)?;
        let expected_samples = emitted_frames.div_ceil(self.sample_stride as u64);
        if self.sampled != expected_samples {
            return Err("Streamed RTMPose sampled frame count mismatch".into());
        }
        if self.observed_crops + self.interpolated_crops != self.people {
            return Err("Streamed RTMPose crop provenance count mismatch".into());
        }
        let mut writer = self.writer.take().ok_or("Pose writer already finished")?;
        writer
            .flush()
            .map_err(|error| format!("Flush pose evidence: {error}"))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("Sync pose evidence: {error}"))?;
        drop(writer);
        let artifact = self.artifact.take().ok_or("Missing pending pose folder")?;
        fs::rename(
            artifact.folder.join("poses.jsonl.partial"),
            artifact.folder.join(POSE_FILE),
        )
        .map_err(|error| format!("Promote streamed pose evidence: {error}"))?;
        let run_id = artifact
            .folder
            .file_name()
            .and_then(|v| v.to_str())
            .ok_or("Invalid pose run directory name")?
            .to_owned();
        Ok(PendingPose {
            artifact,
            run_id,
            checksum: format!("{:x}", self.hash.finalize()),
            model_sha: self.model_sha,
            sample_stride: self.sample_stride,
            emitted_frames,
            decoded_frames,
            native_width: self.native_width,
            native_height: self.native_height,
            sampled: self.sampled,
            people: self.people,
            observed_crops: self.observed_crops,
            interpolated_crops: self.interpolated_crops,
            cuda_pixels: self.cuda_pixels,
            original_source: self.original_source,
            source_identity: self.source_identity,
            project_id: self.project_id,
            video: self.video,
        })
    }
}

impl PendingPose {
    pub(crate) fn publish(
        mut self,
        fusion: &Value,
        cancelled: &AtomicBool,
    ) -> Result<Value, String> {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        if fusion["schema"] != "volley-native-gpu-fusion/1"
            || fusion["state"] != "raw_candidates_fused"
            || fusion["project_id"] != self.project_id
            || fusion["video"] != self.video
            || fusion["source_identity"] != self.source_identity
            || fusion["original_source"] != self.original_source
            || fusion["emitted_frames"] != self.emitted_frames
            || fusion["decoded_frames"] != self.decoded_frames
            || fusion["player_sampling_period_frames"] != self.sample_stride as u64
            || fusion["sha256"]
                .as_str()
                .is_none_or(|v| !volley_core::contracts::sha256(v))
        {
            return Err("Shared pose evidence does not match verified fusion identity".into());
        }
        let fusion_id = fusion["fusion_id"]
            .as_str()
            .filter(|id| volley_core::contracts::job_id(id))
            .ok_or("Invalid fusion ID for shared pose publication")?;
        let pose_manifest = json!({
            "schema":"volley-native-gpu-pose/1",
            "state":"pose_evidence_complete",
            "pose_run_id":self.run_id,
            "fusion_id":fusion_id,
            "project_id":self.project_id,
            "video":self.video,
            "fps":30,
            "stride":self.sample_stride,
            "pose_mode":"photo",
            "photo_stride":self.sample_stride,
            "decoded_frames":self.decoded_frames,
            "sampled_frames":self.sampled,
            "player_poses":self.people,
            "observed_pose_crops":self.observed_crops,
            "interpolated_pose_crops":self.interpolated_crops,
            "model":"RTMPose-M-Body7",
            "model_sha256":self.model_sha,
            "original_source":self.original_source,
            "working_stream":if self.cuda_pixels { "original_nv12_cuda_fps30_v1" } else { "original_rgb_fps30_v1" },
            "pixel_source_version":if self.cuda_pixels { "original_nv12_cuda_v1" } else { "original_full_resolution_rgb24_v1" },
            "pixel_dimensions":[self.native_width,self.native_height],
            "coordinate_system":"normalized_original_display_v1",
            "source_fusion_sha256":fusion["sha256"],
            "source_identity":self.source_identity,
            "box_source_version":"detector_observed_and_bounded_interpolated_v1",
            "source_person_sampling_period_frames":self.sample_stride,
            "interpolate_players":false,
            "file":POSE_FILE,
            "sha256":self.checksum,
            "decode_pipeline":if self.cuda_pixels { "gpu_resident_nvdec_primary_v1" } else { "shared_primary_native_rgb_v2" }
        });
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        crate::storage::write_json_atomic(
            &self.artifact.folder.join("manifest.json"),
            &pose_manifest,
        )
        .map_err(|error| format!("Publish streamed RTMPose manifest: {error:?}"))?;
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        self.artifact.committed = true;
        Ok(pose_manifest)
    }
}

#[cfg(test)]
mod tests;
