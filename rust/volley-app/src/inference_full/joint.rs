//! Experimental single-source GPU pair. The existing raw model manifests
//! and fusion contracts remain unchanged; no tracking or touches are invented.
use super::*;
use volley_media::raw_decoder::{
    decode_full_video_shared_with_backend, DecodeBackend, DecodePreference,
};

#[derive(Debug)]
pub(crate) struct JointPair {
    pub ball_job_id: String,
    pub player_job_id: String,
    pub backend: DecodeBackend,
    pub fallback_reason: Option<String>,
    pub windows: usize,
    pub decoded_frames: u64,
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
    if gpu_memory_limit_mb == 1 {
        return Err("GPU memory limit too small for shared sessions".into());
    }
    let source_identity = volley_media::preparation::source_provenance(original)?;
    volley_media::preparation::verify_source_provenance(original, &source_identity)?;
    let before = inference_stream::source_stamp(original)?;
    let session_memory = if gpu_memory_limit_mb == 0 {
        0
    } else {
        gpu_memory_limit_mb / 2
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
        &source_identity,
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
        &source_identity,
        duration,
        person_config,
        person_stride,
    )?;
    pending.paths.push(players.folder.clone());

    let mut ball_writer = EvidenceWriter::create(&ball.folder)?;
    let mut person_writer = EvidenceWriter::create(&players.folder)?;
    let estimated = (duration * 30.0).max(9.0);
    let summary = decode_full_video_shared_with_backend(
        original,
        9,
        backend_preference()?,
        cancelled,
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
    report(1.0)?;
    pending.committed = true;
    Ok(JointPair {
        ball_job_id: ball.id,
        player_job_id: players.id,
        backend: summary.backend,
        fallback_reason: summary.fallback_reason,
        windows: summary.decode.windows,
        decoded_frames: summary.decode.decoded_frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
