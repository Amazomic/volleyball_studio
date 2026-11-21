//! Experimental single-source GPU pair. The existing raw model manifests
//! and fusion contracts remain unchanged; no tracking or touches are invented.
use super::*;
mod rally_mask;
mod zero_copy;
use crate::pose_inference::{GpuPoseStage, PendingPose, PoseStream};
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
    pub source_provenance: Value,
    pub rally_filter: Option<Value>,
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
    gpu_memory_limit_mb: u64,
    cpu_threads: usize,
    cancelled: &AtomicBool,
    mut report: impl FnMut(f64, u64, u64, Option<(&Value, u64)>) -> Result<(), String>,
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
        range,
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
        range,
        person_config,
        person_stride,
    )?;
    pending.paths.push(players.folder.clone());

    let mut ball_writer = EvidenceWriter::create(&ball.folder)?;
    let mut person_writer = EvidenceWriter::create(&players.folder)?;
    let estimated = (duration * 30.0).max(9.0);
    let (zero_copy_codec, zero_copy_unavailable) = match zero_copy_codec(original, cancelled) {
        Ok(source) => (Some(source), None),
        Err(error) if error == "cancelled" => return Err(error),
        Err(error) => (None, Some(error)),
    };
    let zero_copy = zero_copy_codec.is_some();
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
    let mut rally_filter = None;
    let mut player_target = total_frames;
    let prefilter = !matches!(
        std::env::var("VOLLEY_PRIMARY_RALLY_PREFILTER")
            .ok()
            .as_deref(),
        Some("0" | "false")
    );
    if prefilter {
        ball_writer.ball_samples = Some(Vec::new());
    }
    let summary = if prefilter {
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
                backend_preference()?,
                cpu_threads,
                range,
                cancelled,
                false,
                |batch| {
                    if cancelled.load(Ordering::Acquire) {
                        return Err("cancelled".into());
                    }
                    let balls = detect_ball_window(
                        &mut ball_engine,
                        &batch.gray,
                        batch.first_frame,
                        cancelled,
                    )?;
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
        let mask = RallyMask::from_ball_samples(
            ball_writer.ball_samples.as_deref().unwrap_or(&[]),
            ball_scan.decode.decoded_frames,
        )?;
        ball_writer.ball_samples = None;
        let selected_frames = mask.included_frames();
        let mut mask_value = serde_json::to_value(&mask)
            .map_err(|error| format!("Serialize primary rally mask: {error}"))?;
        mask_value["selected_frames"] = json!(selected_frames);
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
        let mut player_samples = 0u64;
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
                backend_preference()?,
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
                    let mut persons = if selected {
                        detect_player_window(
                            &mut person_engine,
                            &batch.rgb,
                            batch.first_frame,
                            person_stride as u64,
                            cancelled,
                        )?
                    } else {
                        Vec::new()
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
    } else if let Some(source_info) = zero_copy_codec {
        if pose_enabled {
            gpu_pose = Some(GpuPoseStage::new(
                state,
                project_id,
                video,
                pose_original,
                pose_source,
                person_stride,
                session_memory,
                cpu_threads,
            )?);
        }
        let decode = zero_copy::run(
            original,
            source_info,
            range,
            &mut ball_engine,
            &mut person_engine,
            gpu_pose.as_ref(),
            Some(&mut ball_writer),
            Some(&mut person_writer),
            zero_copy::ScanPass::Both,
            &ball,
            &players,
            person_stride,
            duration,
            cancelled,
            |fraction, frames| report(fraction, frames, total_frames, None),
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
            backend_preference()?,
            cpu_threads,
            range,
            cancelled,
            pose_enabled,
            |batch| {
                if cancelled.load(Ordering::Acquire) {
                    return Err("cancelled".into());
                }
                let balls = detect_ball_window(
                    &mut ball_engine,
                    &batch.gray,
                    batch.first_frame,
                    cancelled,
                )?;
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
                    stage.observe(&batch, &persons, None, cancelled)?;
                }
                if ball_writer.count % 10 == 0 {
                    let frames = batch.first_frame + 9;
                    update_progress(&ball.status, ball_writer.count, frames, estimated);
                    update_progress(&players.status, person_writer.count, frames, estimated);
                    persist(&ball.folder, &snapshot(&ball.status))?;
                    persist(&players.folder, &snapshot(&players.status))?;
                    report(
                        (frames as f64 / estimated).clamp(0.0, 0.99),
                        frames,
                        total_frames,
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
    let final_total = if prefilter {
        player_target
    } else {
        total_frames
    };
    report(1.0, final_total, final_total, None)?;
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
        source_provenance: original_provenance,
        rally_filter,
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
