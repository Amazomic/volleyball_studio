//! Deferred RTMPose publication over the same RGB frame stream as VBallNet
//! and the detector. Nothing becomes a reusable pose run until fusion SHA
//! and frame/source identities are verified.
use super::*;
use std::{collections::VecDeque, io::BufWriter, path::PathBuf};
use volley_media::raw_decoder::PrimaryFrameBatch;

const RGB_FRAME_BYTES: usize = 640 * 360 * 3;

#[derive(Debug)]
struct PendingFrame {
    index: u64,
    rgb: Vec<u8>,
    boxes: Vec<[f64; 4]>,
}

fn ready(pending: &mut VecDeque<PendingFrame>, latest: u64) -> Vec<PendingFrame> {
    let mut samples = Vec::new();
    while pending
        .front()
        .is_some_and(|s| s.index.saturating_add(8) <= latest)
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
    hash: Sha256,
    model_sha: String,
    sample_stride: usize,
    pending: VecDeque<PendingFrame>,
    next_frame: u64,
    sampled: u64,
    people: u64,
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
    sampled: u64,
    people: u64,
    original_source: Value,
    source_identity: Value,
    project_id: String,
    video: String,
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
        if !(1..=30).contains(&sample_stride) || !app.writes_enabled {
            return Err("Invalid shared pose stream configuration".into());
        }
        let model = app.repository_root.join("models").join(MODEL_FILE);
        safe_file(&model, 256 * 1024 * 1024)?;
        let model_sha = file_hash(&model)?;
        if model_sha != MODEL_SHA256 {
            return Err("RTMPose model fingerprint does not match the official ONNX".into());
        }
        let engine = PoseEngine::load(
            &app.repository_root,
            Device::Cuda,
            cpu_threads,
            gpu_memory_mb,
        )?;
        let runs = projects::project_root(app)
            .join(project_id)
            .join("native_gpu_pose_runs");
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
            hash: Sha256::new(),
            model_sha,
            sample_stride,
            pending: VecDeque::new(),
            next_frame: 0,
            sampled: 0,
            people: 0,
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
        cancelled: &AtomicBool,
    ) -> Result<(), String> {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        if batch.first_frame != self.next_frame || batch.rgb.len() != RGB_FRAME_BYTES * 9 {
            return Err("Shared RTMPose frames are not a contiguous nine-frame RGB window".into());
        }
        for offset in 0..9 {
            let index = batch.first_frame + offset as u64;
            if index % self.sample_stride as u64 != 0 {
                continue;
            }
            let begin = offset * RGB_FRAME_BYTES;
            self.pending.push_back(PendingFrame {
                index,
                rgb: batch.rgb[begin..begin + RGB_FRAME_BYTES].to_vec(),
                boxes: boxes_from_raw_detections(detected, index)?,
            });
        }
        self.next_frame += 9;
        if self.pending.len() > 18 {
            return Err("Pose lookahead queue exceeded its bounded capacity".into());
        }
        self.flush_ready(batch.first_frame + 8, cancelled)
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
        if rgb_tail.len() != count as usize * RGB_FRAME_BYTES {
            return Err("Pose tail RGB bytes do not match the final frame count".into());
        }
        self.flush_ready(decoded_frames.saturating_sub(1), cancelled)
    }

    fn flush_ready(&mut self, latest: u64, cancelled: &AtomicBool) -> Result<(), String> {
        for frame in ready(&mut self.pending, latest) {
            if cancelled.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }
            let mut poses = Vec::with_capacity(frame.boxes.len());
            for bb in frame.boxes {
                if cancelled.load(Ordering::Acquire) {
                    return Err("cancelled".into());
                }
                let points = self.engine.infer(&frame.rgb, bb)?;
                let keypoints = points.map(|p| [p.x as f64, p.y as f64, p.score as f64]);
                poses.push(json!({"box": bb, "keypoints":keypoints, "crop_origin":"observed"}));
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
            sampled: self.sampled,
            people: self.people,
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
        interpolate_pose: bool,
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
            "model":"RTMPose-M-Body7",
            "model_sha256":self.model_sha,
            "original_source":self.original_source,
            "working_stream":"original_rgb_fps30_v1",
            "source_fusion_sha256":fusion["sha256"],
            "source_identity":self.source_identity,
            "box_source_version":"detector_observed_sampling_v1",
            "source_person_sampling_period_frames":self.sample_stride,
            "interpolate_players":false,
            "interpolate_pose":interpolate_pose,
            "file":POSE_FILE,
            "sha256":self.checksum,
            "decode_pipeline":"shared_primary_v1"
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
