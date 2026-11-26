//! Experimental single-source GPU pair. The existing raw model manifests
//! and fusion contracts remain unchanged; no tracking or touches are invented.
use super::*;
mod rally_mask;
mod zero_copy;
use crate::cpu_retrack::{CompletedRallies, EarlyPoseOptions, EarlyPoseWorker};
use crate::pose_inference::{GpuPoseStage, PendingPose, PoseStream};
use crate::rally_handoff::{
    rally_windows, update_progress as update_rally_progress, RallyHandoff, RallyStageProgress,
};
pub(crate) use rally_mask::PrimaryRallySettings;
use rally_mask::RallyMask;
use std::cell::RefCell;
use std::sync::OnceLock;
use volley_media::nvdec::NvdecCodec;
use volley_media::raw_decoder::{DecodeBackend, DecodePreference, PrimaryDecodeSummary};

#[derive(Clone, Debug, PartialEq)]
struct ZeroCopySource {
    codec: NvdecCodec,
    matrix: u8,
    full_range: bool,
    bit_depth: u8,
    chroma: u8,
    rotation: i32,
    timeline: volley_media::nvdec::DisplayTimeline,
}

struct CachedNvdecSource {
    path: PathBuf,
    stamp: (u64, u128),
    source: ZeroCopySource,
}

fn zero_copy_color_matrix(space: Option<&str>, range: Option<&str>) -> Option<(u8, bool)> {
    let full_range = match range {
        Some("pc" | "jpeg") => true,
        None | Some("unknown" | "tv" | "mpeg") => false,
        _ => return None,
    };
    let matrix = match space {
        Some("bt709") => 1,
        Some("bt2020nc") => 2,
        None | Some("unknown" | "bt470bg" | "smpte170m" | "smpte240m") => 0,
        _ => return None,
    };
    Some((matrix, full_range))
}

fn source_color_matrix(source: &Path) -> Option<(u8, bool, u8, u8)> {
    let output = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=color_space,color_range,pix_fmt",
            "-of",
            "json",
        ])
        .arg(source)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let document: Value = serde_json::from_slice(&output.stdout).ok()?;
    let video = document["streams"].as_array()?.first()?;
    let (matrix, full_range) =
        zero_copy_color_matrix(video["color_space"].as_str(), video["color_range"].as_str())?;
    let (bit_depth, chroma) = match video["pix_fmt"].as_str()? {
        "yuv420p" | "yuvj420p" | "nv12" => (8, 1),
        "yuv420p10le" | "p010le" => (10, 1),
        "yuv420p12le" | "p012le" => (12, 1),
        "yuv422p" | "yuvj422p" | "nv16" => (8, 2),
        "yuv422p10le" | "p210le" => (10, 2),
        "yuv422p12le" | "p212le" => (12, 2),
        "yuv444p" | "yuvj444p" => (8, 3),
        "yuv444p10le" => (10, 3),
        "yuv444p12le" => (12, 3),
        _ => return None,
    };
    Some((matrix, full_range, bit_depth, chroma))
}
fn zero_copy_codec(original: &Path, cancelled: &AtomicBool) -> Result<ZeroCopySource, String> {
    if !matches!(
        std::env::var("VOLLEY_ZERO_COPY_PRIMARY").ok().as_deref(),
        None | Some("1") | Some("true")
    ) {
        return Err("CUDA zero-copy disabled by VOLLEY_ZERO_COPY_PRIMARY".into());
    }
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }

    static SOURCE_CACHE: OnceLock<StdMutex<Vec<CachedNvdecSource>>> = OnceLock::new();
    let stamp = inference_stream::source_stamp(original)?;
    let cache = SOURCE_CACHE.get_or_init(|| StdMutex::new(Vec::new()));
    let cached = cache
        .lock()
        .map_err(|_| "NVDEC source metadata cache was poisoned")?
        .iter()
        .find(|entry| entry.path == original && entry.stamp == stamp)
        .map(|entry| entry.source.clone());
    let source = if let Some(source) = cached {
        source
    } else {
        let info = volley_media::probe(original)
            .map_err(|error| format!("NVDEC source metadata probe: {error}"))?;
        let rotation = info.rotation.rem_euclid(360);
        if !matches!(rotation, 0 | 90 | 180 | 270) {
            return Err(format!("NVDEC rotation {rotation} is unsupported"));
        }
        let codec = match info.codec.as_str() {
            "h264" => NvdecCodec::H264,
            "hevc" => NvdecCodec::Hevc,
            "vp8" => NvdecCodec::Vp8,
            "vp9" => NvdecCodec::Vp9,
            "av1" => NvdecCodec::Av1,
            "mpeg1video" => NvdecCodec::Mpeg1,
            "mpeg2video" => NvdecCodec::Mpeg2,
            "mpeg4" => NvdecCodec::Mpeg4,
            "vc1" => NvdecCodec::Vc1,
            "mjpeg" => NvdecCodec::Mjpeg,
            other => return Err(format!("NVDEC unsupported video codec: {other}")),
        };
        let (matrix, full_range, bit_depth, chroma) = source_color_matrix(original)
            .ok_or("NVDEC unsupported source pixel format or color matrix")?;
        let timeline = volley_media::nvdec::display_timeline(original)
            .map_err(|error| format!("NVDEC display timeline: {error}"))?;
        let source = ZeroCopySource {
            codec,
            matrix,
            full_range,
            bit_depth,
            chroma,
            rotation,
            timeline,
        };
        let mut entries = cache
            .lock()
            .map_err(|_| "NVDEC source metadata cache was poisoned")?;
        entries.retain(|entry| entry.path != original);
        if entries.len() == 4 {
            entries.remove(0);
        }
        entries.push(CachedNvdecSource {
            path: original.to_path_buf(),
            stamp,
            source: source.clone(),
        });
        source
    };
    if !volley_media::nvdec::codec_supported_format(source.codec, source.bit_depth, source.chroma) {
        return Err(
            "NVDEC hardware does not support this codec/bit-depth/chroma combination".into(),
        );
    }
    volley_inference::gpu_preprocess::GpuPreprocessor::new()
        .map_err(|error| format!("NVDEC zero-copy CUDA preprocessing unavailable: {error}"))?;
    Ok(source)
}

#[derive(Debug)]
pub(crate) struct JointPair {
    pub ball_job_id: String,
    pub player_job_id: String,
    pub backend: DecodeBackend,
    pub zero_copy: bool,
    pub fallback_reason: Option<String>,
    pub windows: usize,
    pub decoded_frames: u64,
    pub pending_pose: Option<PendingPose>,
    pub early_poses: Option<CompletedRallies>,
    pub early_pose_error: Option<String>,
    pub source_provenance: Value,
    pub rally_filter: Option<Value>,
    pub checkpoint_key: Option<String>,
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
    range: Option<volley_media::raw_decoder::AnalysisRange>,
    config: Config,
    person_detector: volley_inference::person::DetectorVariant,
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
        analysis_range: range,
        config,
        person_detector,
        person_batch_size: 1,
        person_tf32: true,
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
    range: Option<volley_media::raw_decoder::AnalysisRange>,
    person_stride: usize,
    pose_enabled: bool,
    person_detector: volley_inference::person::DetectorVariant,
    gpu_memory_limit_mb: u64,
    cpu_threads: usize,
    rally_settings: PrimaryRallySettings,
    stabilizer_settings: volley_core::ball_interpolation::BallStabilizerSettings,
    early_pose: Option<EarlyPoseOptions>,
    cancelled: &AtomicBool,
    mut report: impl FnMut(f64, u64, u64, Option<(&Value, u64)>) -> Result<(), String>,
) -> Result<JointPair, String> {
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    rally_settings.validate()?;
    if !state.writes_enabled || !(1..=30).contains(&person_stride) {
        return Err("Invalid shared GPU job or person sampling interval".into());
    }
    let session_count = 2u64 + u64::from(pose_enabled || early_pose.is_some());
    if gpu_memory_limit_mb > 0 && gpu_memory_limit_mb < session_count {
        return Err("GPU memory limit too small for shared sessions".into());
    }
    if let Some(range) = range {
        range.validate(volley_media::probe(original)?.duration_sec)?;
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
    let total_frames = (duration * 30.0).round().max(9.0) as u64;
    report(0.0, 0, total_frames, None)?;
    let mut ball_engine = Engine::load(&state.repository_root, &ball_config)?;
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let mut person_engine = PersonEngine::load_for_primary_gpu_evidence(
        &state.repository_root,
        cpu_threads,
        session_memory,
        person_detector,
    )?;

    let mut pending = PendingFolders::default();
    let ball = new_context(
        state,
        project_id,
        video,
        original,
        &original_provenance,
        duration,
        range,
        ball_config,
        person_detector,
        1,
    )?;
    pending.paths.push(ball.folder.clone());
    let mut players = new_context(
        state,
        project_id,
        video,
        original,
        &original_provenance,
        duration,
        range,
        person_config,
        person_detector,
        person_stride,
    )?;
    pending.paths.push(players.folder.clone());
    players.person_batch_size = person_engine.batch_size();
    players.person_tf32 =
        person_detector != volley_inference::person::DetectorVariant::RfDetrMedium;

    let mut ball_writer = EvidenceWriter::create(&ball.folder)?;
    let mut person_writer = EvidenceWriter::create(&players.folder)?;
    let estimated = (duration * 30.0).max(9.0);
    let (zero_copy_codec, zero_copy_unavailable) = match zero_copy_codec(original, cancelled) {
        Ok(source) => (Some(source), None),
        Err(error) if error == "cancelled" => return Err(error),
        Err(error) => (None, Some(error)),
    };
    let zero_copy = zero_copy_codec.is_some();
    let court = crate::video_edits::court_snapshot(state, project_id, video)
        .map_err(|e| format!("{e:?}"))?;
    let preference = backend_preference()?;
    let checkpoint_root = early_pose
        .as_ref()
        .and_then(|options| options.checkpoint_root.as_ref())
        .filter(|_| !pose_enabled);
    let (preference, backend_tag) = if zero_copy {
        (preference, "nvdec_zero_copy")
    } else if checkpoint_root.is_some() {
        let selection =
            volley_media::raw_decoder::select_decode_backend(original, preference, cancelled)?;
        match selection.backend {
            DecodeBackend::Nvdec => (DecodePreference::Nvdec, "nvdec"),
            DecodeBackend::Software => (DecodePreference::Software, "software"),
        }
    } else {
        (preference, "no_checkpoint")
    };
    let checkpoint_dir = checkpoint_root.map(|root| {
        let options = early_pose.as_ref().ok_or("Missing rally checkpoint settings")?;
        let ball_model = "VballNetV4c_seq9_grayscale_20260908_213829.onnx";
        let detector_model = person_engine.model_file();
        let identity = json!({
            "schema":"volley-phase1-checkpoint/1",
            "build":env!("VOLLEY_BUILD_GIT_SHA"),
            "project":project_id,"video":video,
            "source":original_provenance,"stamp":[before.0,before.1],
            "range":range,"duration":duration,"backend":backend_tag,
            "person_stride":person_stride,"person_detector":person_detector,
            "person_batch_size":players.person_batch_size,"person_tf32":players.person_tf32,
            "ball_model_sha256":checkpoint::model_digest(&state.repository_root, ball_model)?,
            "ball_sidecar_sha256":checkpoint::model_digest(&state.repository_root, "VballNetV4c_seq9_grayscale_20260908_213829.json")?,
            "person_model_sha256":checkpoint::model_digest(&state.repository_root, detector_model)?,
            "pose_model_sha256":checkpoint::model_digest(&state.repository_root, volley_inference::pose::MODEL_FILE)?,
            "rally_settings":rally_settings,"stabilizer_settings":stabilizer_settings,
            "tracking":options.tracking,"interpolate_players":options.interpolate_players,
            "reid_capture":options.reid_crops.is_some(),
            "court_revision":court.as_ref().map(|c| &c.revision),
            "memory_mb":gpu_memory_limit_mb,"threads":cpu_threads,
        });
        checkpoint::directory(root, &identity)
    }).transpose()?;
    if let Some(directory) = &checkpoint_dir {
        ball_writer.checkpoint = Some(checkpoint::DetectionJournal::open(
            directory,
            "ball.jsonl.partial",
            "ball",
        )?);
    }
    let (pose_original, pose_source) = pose_manifest_identities(&original_provenance, before);
    let pose = RefCell::new(if pose_enabled && !zero_copy {
        Some(PoseStream::new(
            state,
            project_id,
            video,
            pose_original.clone(),
            pose_source.clone(),
            person_stride,
            session_memory,
            cpu_threads,
        )?)
    } else {
        None
    });
    let mut gpu_pose = None;
    let mut early_worker = None;
    let early_progress = early_pose.as_ref().map(|options| options.progress.clone());
    let early_pose_error = None;
    let rally_filter;
    let player_target;
    ball_writer.ball_samples = Some(Vec::new());
    let summary = {
        let ball_scan = if let Some(source_info) = zero_copy_codec.clone() {
            let decode = zero_copy::run(
                original,
                source_info,
                range,
                &mut ball_engine,
                &mut person_engine,
                gpu_pose.as_ref(),
                Some(&mut ball_writer),
                None,
                zero_copy::ScanPass::BallOnly,
                &ball,
                &players,
                person_stride,
                duration,
                cancelled,
                |fraction, frames| report(0.4 * fraction, frames, total_frames, None),
            )?;
            PrimaryDecodeSummary {
                decode,
                backend: DecodeBackend::Nvdec,
                fallback_reason: None,
            }
        } else {
            volley_media::raw_decoder::decode_full_video_shared_with_native_and_tail_in_range(
                original,
                9,
                preference,
                cpu_threads,
                range,
                cancelled,
                false,
                |batch| {
                    if cancelled.load(Ordering::Acquire) {
                        return Err("cancelled".into());
                    }
                    let balls = match ball_writer.cached(batch.first_frame)? {
                        Some(records) => records,
                        None => detect_ball_window(
                            &mut ball_engine,
                            &batch.gray,
                            batch.first_frame,
                            cancelled,
                        )?,
                    };
                    ball_writer.push(batch.first_frame, &balls)?;
                    if ball_writer.count.is_multiple_of(10) {
                        let frames = batch.first_frame + 9;
                        update_progress(&ball.status, ball_writer.count, frames, estimated);
                        persist(&ball.folder, &snapshot(&ball.status))?;
                        report(
                            0.4 * (frames as f64 / estimated).clamp(0.0, 0.99),
                            frames,
                            total_frames,
                            None,
                        )?;
                    }
                    Ok(())
                },
                |_, _| Ok(()),
            )?
        };
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        report(
            0.399999,
            ball_scan.decode.decoded_frames,
            total_frames,
            None,
        )?;
        let dimension = |key: &str| {
            original_provenance["video"][key]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .filter(|n| (64..=16384).contains(n))
                .ok_or("Invalid Ground Detector dimensions")
        };
        let dimensions = (dimension("width")?, dimension("height")?);
        let trajectory = RallyMask::stabilize(
            ball_writer.ball_samples.as_deref().unwrap_or(&[]),
            ball_scan.decode.decoded_frames,
            stabilizer_settings,
        )?;
        let trajectory_sha =
            crate::cpu_touches::write_ball_trajectory(&ball.folder, &trajectory, cancelled)?;
        report(
            0.3999995,
            ball_scan.decode.decoded_frames,
            total_frames,
            None,
        )?;
        let (mut mask, contacts) = RallyMask::detect(
            &trajectory,
            rally_settings,
            stabilizer_settings,
            court.as_ref().and_then(|c| c.transform.as_ref()),
            dimensions,
            cancelled,
        )?;
        let ground_summary = crate::cpu_touches::ground_stage::publish(
            &ball.folder,
            &contacts,
            &trajectory_sha,
            dimensions,
            trajectory.points.len() as u64,
            court.as_ref(),
        )?;
        if let Some(reason) = mask.fallback_reason {
            return Err(format!(
                "Не найдены надёжные розыгрыши ({reason}); поиск игроков и поз не запускался. Проверьте траекторию мяча и настройки розыгрышей"
            ));
        }
        if early_pose.is_some() && !pose_enabled {
            mask.restrict_to_confirmed();
        }
        ball_writer.ball_samples = None;
        let selected_frames = mask.included_frames();
        let mut mask_value = serde_json::to_value(&mask)
            .map_err(|error| format!("Serialize primary rally mask: {error}"))?;
        mask_value["selected_frames"] = json!(selected_frames);
        mask_value["ground_boundary_version"] = json!("ground_ends_rally_v3");
        mask_value["ground_detector"] = ground_summary;
        mask_value["ball_stabilizer"]["trajectory_sha256"] = json!(trajectory_sha);
        mask_value["ball_stabilizer"]["confidence_frames"] = json!(trajectory.confidence_frames);
        mask_value["ball_stabilizer"]["observed_frames"] = json!(trajectory.observed_frames);
        storage::write_json_atomic(
            &ball.folder.join("rally_mask.json"),
            &json!({
                "schema":"volley-primary-rally-mask-evidence/1",
                "ball_job_id":ball.id,
                "original_source":original_provenance,
                "analysis_range":range,
                "selected_frames":selected_frames,
                "mask":mask_value
            }),
        )
        .map_err(|error| format!("Save primary rally mask: {error:?}"))?;

        let player_total = mask.sampled_frames(person_stride);
        player_target = player_total;
        report(
            0.4,
            0,
            player_total,
            Some((&mask_value, ball_scan.decode.decoded_frames)),
        )?;
        rally_filter = Some(mask_value);
        let rally_checkpoint = checkpoint_dir
            .as_ref()
            .map(|directory| {
                checkpoint::directory(
                    directory,
                    &json!({
                        "schema":"volley-phase1-rally-scope/1",
                        "trajectory_sha256":trajectory_sha,
                        "selected":mask.selected,
                        "rallies":mask.rally_intervals,
                        "decoded_frames":ball_scan.decode.decoded_frames,
                    }),
                )
            })
            .transpose()?;
        if let Some(directory) = &rally_checkpoint {
            person_writer.checkpoint = Some(checkpoint::DetectionJournal::open(
                directory,
                "players.jsonl.partial",
                "player",
            )?);
        }
        let mut player_samples = 0u64;
        if let Some(mut options) = early_pose.filter(|_| !pose_enabled) {
            options.checkpoint_root = rally_checkpoint;
            options.tracking.min_track_observations = crate::cpu_retrack::confirmation_threshold(
                options.tracking.min_track_observations,
                Some(person_stride as u64),
            );
            let decoded_frames = ball_scan.decode.decoded_frames;
            let frames = decoded_frames / 9 * 9;
            let start = || -> Result<_, String> {
                let selected = crate::cpu_touches::align_primary_intervals(
                    &mask.selected,
                    decoded_frames,
                    frames,
                )?;
                let rallies = crate::cpu_touches::align_primary_intervals(
                    &mask.rally_intervals,
                    decoded_frames,
                    frames,
                )?;
                let windows = rally_windows(&selected, &rallies)?;
                if windows.is_empty() {
                    return Err("No confirmed rally work windows".into());
                }
                update_rally_progress(&options.progress, |p| p.initialize(&windows))?;
                let selected = windows.iter().map(|w| (w.left, w.right)).collect();
                let handoff = RallyHandoff::new(windows, frames)?;
                let (worker, sender) = EarlyPoseWorker::start(
                    options,
                    original.to_owned(),
                    state.repository_root.clone(),
                    range,
                    decoded_frames,
                    selected,
                    trajectory,
                    rallies,
                    dimensions,
                    session_memory,
                )?;
                Ok(Some((worker, handoff, sender)))
            };
            match start() {
                Ok(Some((worker, handoff, sender))) => {
                    person_writer.rally_handoff = Some((handoff, sender));
                    early_worker = Some(worker);
                }
                Ok(None) => {}
                Err(error) => return Err(error),
            }
        }
        if zero_copy && pose_enabled {
            gpu_pose = Some(GpuPoseStage::new(
                state,
                project_id,
                video,
                pose_original.clone(),
                pose_source.clone(),
                person_stride,
                session_memory,
                cpu_threads,
            )?);
        }
        let people_scan = if let Some(source_info) = zero_copy_codec.clone() {
            let decode = zero_copy::run(
                original,
                source_info,
                range,
                &mut ball_engine,
                &mut person_engine,
                gpu_pose.as_ref(),
                None,
                Some(&mut person_writer),
                zero_copy::ScanPass::PlayersOnly(&mask),
                &ball,
                &players,
                person_stride,
                duration,
                cancelled,
                |fraction, samples| report(0.4 + 0.6 * fraction, samples, player_total, None),
            )?;
            PrimaryDecodeSummary {
                decode,
                backend: DecodeBackend::Nvdec,
                fallback_reason: None,
            }
        } else {
            volley_media::raw_decoder::decode_full_video_shared_with_native_and_tail_in_range(
                original,
                9,
                preference,
                cpu_threads,
                range,
                cancelled,
                pose_enabled,
                |batch| {
                    if cancelled.load(Ordering::Acquire) {
                        return Err("cancelled".into());
                    }
                    let selected = (batch.first_frame..batch.first_frame + 9).any(|frame| {
                        frame.is_multiple_of(person_stride as u64) && mask.includes(frame)
                    });
                    let mut persons = match person_writer.cached(batch.first_frame)? {
                        Some(records) => records,
                        None if selected => detect_selected_player_window(
                            &mut person_engine,
                            &batch.rgb,
                            batch.first_frame,
                            person_stride as u64,
                            cancelled,
                            |frame| mask.includes(frame),
                        )?,
                        None => Vec::new(),
                    };
                    persons.retain(|person| mask.includes(person.frame_index));
                    player_samples += (batch.first_frame..batch.first_frame + 9)
                        .filter(|frame| {
                            frame.is_multiple_of(person_stride as u64) && mask.includes(*frame)
                        })
                        .count() as u64;
                    person_writer.push(batch.first_frame, &persons)?;
                    if let Some(stage) = pose.borrow_mut().as_mut() {
                        stage.observe(
                            &batch,
                            &persons,
                            Some(&|frame| mask.includes(frame)),
                            cancelled,
                        )?;
                    }
                    if person_writer.count.is_multiple_of(10) {
                        let frames = batch.first_frame + 9;
                        update_progress(&players.status, person_writer.count, frames, estimated);
                        persist(&players.folder, &snapshot(&players.status))?;
                        report(
                            0.4 + 0.6 * (frames as f64 / estimated).clamp(0.0, 0.99),
                            player_samples,
                            player_total,
                            None,
                        )?;
                    }
                    Ok(())
                },
                |tail, decoded| {
                    if let Some(stage) = pose.borrow_mut().as_mut() {
                        stage.tail(tail, decoded, cancelled)?;
                    }
                    Ok(())
                },
            )?
        };
        if people_scan.decode.decoded_frames != ball_scan.decode.decoded_frames
            || people_scan.decode.windows != ball_scan.decode.windows
            || people_scan.decode.first_frame != ball_scan.decode.first_frame
        {
            return Err("Primary ball and people scans disagree on video timeline".into());
        }
        people_scan
    };
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    if inference_stream::source_stamp(original)? != before {
        return Err("Original video changed during shared GPU analysis".into());
    }
    let backend = match summary.backend {
        DecodeBackend::Nvdec if zero_copy => "nvdec_zero_copy",
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
    let pending_pose = if let Some(stage) = gpu_pose {
        Some(stage.finish(
            summary.decode.decoded_frames,
            summary.decode.windows,
            cancelled,
        )?)
    } else {
        pose.into_inner()
            .map(|stage| {
                stage.finish(
                    summary.decode.decoded_frames,
                    summary.decode.windows,
                    cancelled,
                )
            })
            .transpose()?
    };
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    if let (Some(worker), Some(progress)) = (&early_worker, &early_progress) {
        update_rally_progress(progress, RallyStageProgress::detector_finished)?;
        report(1.0, player_target, player_target, None)?;
        while !worker.is_finished() {
            if cancelled.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
            report(1.0, player_target, player_target, None)?;
        }
    }
    let early_poses = early_worker.map(EarlyPoseWorker::finish).transpose()?;
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    report(1.0, player_target, player_target, None)?;
    pending.committed = true;
    Ok(JointPair {
        ball_job_id: ball.id,
        player_job_id: players.id,
        backend: summary.backend,
        zero_copy,
        fallback_reason: zero_copy_unavailable.or(summary.fallback_reason),
        windows: summary.decode.windows,
        decoded_frames: summary.decode.decoded_frames,
        pending_pose,
        early_poses,
        early_pose_error,
        source_provenance: original_provenance,
        rally_filter,
        checkpoint_key: checkpoint_dir
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_copy_source_color_matrix_preserves_range_and_bt2020() {
        assert_eq!(
            zero_copy_color_matrix(Some("bt709"), Some("tv")),
            Some((1, false))
        );
        assert_eq!(
            zero_copy_color_matrix(Some("bt470bg"), Some("tv")),
            Some((0, false))
        );
        assert_eq!(
            zero_copy_color_matrix(Some("smpte170m"), None),
            Some((0, false))
        );
        assert_eq!(zero_copy_color_matrix(None, None), Some((0, false)));
        assert_eq!(
            zero_copy_color_matrix(Some("bt709"), Some("pc")),
            Some((1, true))
        );
        assert_eq!(
            zero_copy_color_matrix(Some("bt2020nc"), Some("tv")),
            Some((2, false))
        );
        assert_eq!(
            zero_copy_color_matrix(Some("smpte240m"), Some("tv")),
            Some((0, false))
        );
        assert_eq!(zero_copy_color_matrix(Some("reserved"), Some("tv")), None);
    }
    #[test]
    #[ignore = "requires NVIDIA CUDA and FFmpeg"]
    fn zero_copy_hevc_preflight_proves_gpu_backend_ready() {
        let source = std::env::temp_dir().join(format!(
            "volley-nvdec-smoke-{}.mp4",
            Uuid::new_v4().simple()
        ));
        let encoded = std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x360:rate=30:duration=4",
                "-c:v",
                "libx265",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-y",
            ])
            .arg(&source)
            .status()
            .unwrap();
        assert!(encoded.success());
        let result = zero_copy_codec(&source, &AtomicBool::new(false));
        let _ = fs::remove_file(source);
        let detected = result.expect("NVDEC CUDA preflight must support HEVC 8-bit");
        assert_eq!(detected.codec, NvdecCodec::Hevc);
        assert_eq!(detected.bit_depth, 8);
        assert_eq!(detected.chroma, 1);
        assert!(detected.timeline.timestamps.len() >= 100);
    }

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
