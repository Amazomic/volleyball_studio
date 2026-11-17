//! Experimental single-source GPU pair. The existing raw model manifests
//! and fusion contracts remain unchanged; no tracking or touches are invented.
use super::*;
use crate::pose_inference::{PendingPose, PoseStream};
use std::cell::RefCell;
use volley_media::raw_decoder::{
    decode_full_video_shared_with_preview, DecodeBackend, DecodePreference, PrimaryPreviewOutput,
};

#[derive(Debug)]
pub(crate) struct JointPair {
    pub ball_job_id: String,
    pub player_job_id: String,
    pub backend: DecodeBackend,
    pub fallback_reason: Option<String>,
    pub windows: usize,
    pub decoded_frames: u64,
    pub pending_pose: Option<PendingPose>,
}

#[derive(Default)]
struct PendingFolders {
    paths: Vec<PathBuf>,
    committed: bool,
}
impl Drop for PendingFolders {
    fn drop(&mut self) {
        if !self.committed {
            for folder in &self.paths {
                let _ = fs::remove_dir_all(folder);
            }
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn new_context(
    state: &AppState,
    project_id: &str,
    video: &str,
    source: &Path,
    provenance: &Value,
    duration: f64,
    config: Config,
    person_stride: usize,
) -> Result<JobContext, String> {
    let id = Uuid::new_v4().simple().to_string();
    let folder = job_root(state, project_id, true)
        .map_err(|error| format!("Shared GPU job root: {error:?}"))?
        .join(&id);
    fs::create_dir(&folder).map_err(|error| format!("Shared GPU job folder: {error}"))?;
    let status = Arc::new(StdMutex::new(JobStatus {
        job_id: id.clone(),
        project_id: project_id.to_owned(),
        video: video.to_owned(),
        model: config.model,
        phase: "running".into(),
        windows: 0,
        decoded_frames: 0,
        progress_percent: 0.0,
        error: None,
        output_manifest: None,
    }));
    Ok(JobContext {
        id,
        project_id: project_id.to_owned(),
        video: video.to_owned(),
        source: source.to_path_buf(),
        original: source.to_path_buf(),
        original_source: provenance.clone(),
        duration_sec: duration,
        config,
        stride: 9,
        person_stride,
        folder,
        repo: state.repository_root.clone(),
        status,
        cancelled: Arc::new(AtomicBool::new(false)),
    })
}

fn completed(ctx: &JobContext, summary: &DecodeSummary) -> Result<(), String> {
    let mut result = snapshot(&ctx.status);
    result.phase = "complete".into();
    result.progress_percent = 100.0;
    result.windows = summary.windows;
    result.decoded_frames = summary.decoded_frames;
    result.output_manifest = Some(MANIFEST_FILE.to_owned());
    persist(&ctx.folder, &result)
}

fn backend_preference() -> Result<DecodePreference, String> {
    match std::env::var("VOLLEY_SHARED_PRIMARY_BACKEND") {
        Err(std::env::VarError::NotPresent) => Ok(DecodePreference::Auto),
        Err(error) => Err(format!("Decode backend configuration: {error}")),
        Ok(value) => match value.as_str() {
            "auto" => Ok(DecodePreference::Auto),
            "software" => Ok(DecodePreference::Software),
            "nvdec" => Ok(DecodePreference::Nvdec),
            _ => Err("VOLLEY_SHARED_PRIMARY_BACKEND must be auto, software or nvdec".into()),
        },
    }
}

/// Keep immutable content provenance distinct from the prepared file stamp.
/// Fusion and CPU pose lineage validate these independent JSON contracts.
fn pose_manifest_identities(
    original_provenance: &Value,
    source_stamp: (u64, u128),
) -> (Value, Value) {
    (
        original_provenance.clone(),
        json!({"bytes":source_stamp.0,"mtime_ns":source_stamp.1}),
    )
}

/// Both model sessions consume exactly the same nine-frame timeline. The
/// CUDA arena limit is divided between persistent sessions (not a hard cap on
/// total GPU allocations). An error or cancellation removes both incomplete
/// runs; fusion can only observe two completed SHA-verified manifests.
#[expect(clippy::too_many_arguments)]
pub(crate) fn run_joint_pair(
    state: &AppState,
    project_id: &str,
    video: &str,
    original: &Path,
    duration: f64,
    person_stride: usize,
    pose_enabled: bool,
    gpu_memory_limit_mb: u64,
    cpu_threads: usize,
    cancelled: &AtomicBool,
    mut report: impl FnMut(f64) -> Result<(), String>,
) -> Result<JointPair, String> {
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    if !state.writes_enabled || !(1..=30).contains(&person_stride) {
        return Err("Invalid shared GPU job or person sampling interval".into());
    }
    // RF-DETR Medium is the sole supported person detector; keep its
    // explicit configuration validation but never revive deleted RTMDet.
    let _detector = volley_inference::person::DetectorVariant::from_env()?;
    let session_count = 2u64 + u64::from(pose_enabled);
    if gpu_memory_limit_mb > 0 && gpu_memory_limit_mb < session_count {
        return Err("GPU memory limit too small for shared sessions".into());
    }
    let original_provenance = volley_media::preparation::source_provenance(original)?;
    volley_media::preparation::verify_source_provenance(original, &original_provenance)?;
    let before = inference_stream::source_stamp(original)?;
    let session_memory = if gpu_memory_limit_mb == 0 {
        0
    } else {
        gpu_memory_limit_mb / session_count
    };
    let config = |model| Config {
        model,
        device: Device::Cuda,
        gpu_device_id: 0,
        gpu_memory_limit_mb: session_memory,
        cpu_threads,
    };
    let ball_config = config(ModelKind::Ball);
    let person_config = config(ModelKind::Players);
    ball_config.verify()?;
    person_config.verify()?;
    report(0.0)?;
    let preview_plan = volley_media::preparation::primary_preview::PrimaryPreviewPlan::new(
        original,
        &state.data_root,
        project_id,
        video,
    )?;
    let preview = if preview_plan.ready_path().is_some() {
        None
    } else {
        Some(preview_plan.begin()?)
    };
    let preview_hardware = if preview.is_some() {
        preview_nvenc_available(original, cancelled)?
    } else {
        false
    };
    let mut ball_engine = Engine::load(&state.repository_root, &ball_config)?;
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let mut person_engine = PersonEngine::load_for_gpu_evidence(
        &state.repository_root,
        Device::Cuda,
        cpu_threads,
        session_memory,
    )?;

    let mut pending = PendingFolders::default();
    let ball = new_context(
        state,
        project_id,
        video,
        original,
        &original_provenance,
        duration,
        ball_config,
        1,
    )?;
    pending.paths.push(ball.folder.clone());
    let players = new_context(
        state,
        project_id,
        video,
        original,
        &original_provenance,
        duration,
        person_config,
        person_stride,
    )?;
    pending.paths.push(players.folder.clone());

    let mut ball_writer = EvidenceWriter::create(&ball.folder)?;
    let mut person_writer = EvidenceWriter::create(&players.folder)?;
    let pose = if pose_enabled {
        let (pose_original, pose_source) = pose_manifest_identities(&original_provenance, before);
        Some(PoseStream::new(
            state,
            project_id,
            video,
            pose_original,
            pose_source,
            person_stride,
            session_memory,
            cpu_threads,
        )?)
    } else {
        None
    };
    let pose = RefCell::new(pose);
    let estimated = (duration * 30.0).max(9.0);
    let summary = decode_full_video_shared_with_preview(
        original,
        9,
        backend_preference()?,
        cpu_threads,
        cancelled,
        preview.as_ref().map(|write| PrimaryPreviewOutput {
            path: &write.path,
            hardware: preview_hardware,
        }),
        |batch| {
            if cancelled.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }
            let balls =
                detect_ball_window(&mut ball_engine, &batch.gray, batch.first_frame, cancelled)?;
            let persons = detect_player_window(
                &mut person_engine,
                &batch.rgb,
                batch.first_frame,
                person_stride as u64,
                cancelled,
            )?;
            ball_writer.push(batch.first_frame, &balls)?;
            person_writer.push(batch.first_frame, &persons)?;
            if let Some(stage) = pose.borrow_mut().as_mut() {
                stage.observe(&batch, &persons, cancelled)?;
            }
            if ball_writer.count % 10 == 0 {
                let frames = batch.first_frame + 9;
                update_progress(&ball.status, ball_writer.count, frames, estimated);
                update_progress(&players.status, person_writer.count, frames, estimated);
                persist(&ball.folder, &snapshot(&ball.status))?;
                persist(&players.folder, &snapshot(&players.status))?;
                report((frames as f64 / estimated).clamp(0.0, 0.99))?;
            }
            Ok(())
        },
        |tail, decoded| {
            if let Some(stage) = pose.borrow_mut().as_mut() {
                stage.tail(tail, decoded, cancelled)?;
            }
            Ok(())
        },
    )?;
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    if inference_stream::source_stamp(original)? != before {
        return Err("Original video changed during shared GPU analysis".into());
    }
    let backend = match summary.backend {
        DecodeBackend::Nvdec => "nvdec",
        DecodeBackend::Software => "software",
    };
    finalize_model_output(&ball, before, ball_writer, &summary.decode, Some(backend))?;
    finalize_model_output(
        &players,
        before,
        person_writer,
        &summary.decode,
        Some(backend),
    )?;
    completed(&ball, &summary.decode)?;
    completed(&players, &summary.decode)?;
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let pending_pose = pose
        .into_inner()
        .map(|stage| {
            stage.finish(
                summary.decode.decoded_frames,
                summary.decode.windows,
                cancelled,
            )
        })
        .transpose()?;
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    if let Some(preview) = preview {
        if let Err(error) = preview.publish(cancelled) {
            // Preview is an optional browser artifact. A bad MP4 must never
            // discard expensive, already validated ball/player GPU evidence.
            // Source integrity and cancellation remain fatal.
            if error == "cancelled" || error == "Preview source changed during analysis" {
                return Err(error);
            }
            eprintln!("Primary preview unavailable (GPU evidence retained): {error}");
        }
    }
    report(1.0)?;
    pending.committed = true;
    Ok(JointPair {
        ball_job_id: ball.id,
        player_job_id: players.id,
        backend: summary.backend,
        fallback_reason: summary.fallback_reason,
        windows: summary.decode.windows,
        decoded_frames: summary.decode.decoded_frames,
        pending_pose,
    })
}

fn preview_nvenc_available(source: &Path, cancelled: &AtomicBool) -> Result<bool, String> {
    let info = volley_media::probe(source)?;
    let scale = (1280.0 / info.width as f64)
        .min(1280.0 / info.height as f64)
        .min(1.0);
    let width = ((info.width as f64 * scale) as u32 / 2 * 2).max(2);
    let height = ((info.height as f64 * scale) as u32 / 2 * 2).max(2);
    let input = format!("color=size={width}x{height}:rate=30");
    let mut child = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            &input,
            "-frames:v",
            "1",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("Preview encoder preflight: {e}"))?;
    let start = std::time::Instant::now();
    loop {
        if cancelled.load(Ordering::Acquire) || start.elapsed() > std::time::Duration::from_secs(12)
        {
            let _ = child.kill();
            let _ = child.wait();
            return if cancelled.load(Ordering::Acquire) {
                Err("cancelled".into())
            } else {
                Ok(false)
            };
        }
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            return Ok(status.success());
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_provenance_does_not_swap_original_content_and_file_stamp() {
        let original = json!({"sha256":"a".repeat(64),"size_bytes":262927597u64});
        let (content, stamp) =
            pose_manifest_identities(&original, (262927597, 1794865957038721051));
        assert_eq!(content, original);
        assert_eq!(
            stamp,
            json!({"bytes":262927597u64,"mtime_ns":1794865957038721051u64})
        );
        assert_ne!(content, stamp);
    }

    #[test]
    fn uncommitted_joint_gpu_runs_are_removed_on_failure() {
        let dir = std::env::temp_dir().join(format!("volley-joint-clean-{}", Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        let folder = dir.join("partial");
        fs::create_dir(&folder).unwrap();
        let mut guard = PendingFolders::default();
        guard.paths.push(folder.clone());
        drop(guard);
        assert!(!folder.exists());
        fs::remove_dir(dir).unwrap();
    }
}

#[cfg(test)]
mod gpu_ab_tests;
