use super::*;
use std::sync::Arc;
use volley_inference::gpu_preprocess::{GpuBuffer, GpuPreprocessor, Nv12View};
use volley_media::nvdec::{closest_display_frame, decode_video_file};

#[derive(Clone, Copy)]
pub(super) enum ScanPass<'a> {
    Both,
    BallOnly,
    PlayersOnly(&'a rally_mask::RallyMask),
}

#[expect(clippy::too_many_arguments)]
pub(super) fn run(
    original: &Path,
    source: ZeroCopySource,
    range: Option<volley_media::raw_decoder::AnalysisRange>,
    ball_engine: &mut Engine,
    person_engine: &mut PersonEngine,
    pose: &RefCell<Option<PoseStream>>,
    mut ball_writer: Option<&mut EvidenceWriter>,
    mut person_writer: Option<&mut EvidenceWriter>,
    pass: ScanPass<'_>,
    ball: &JobContext,
    players: &JobContext,
    person_stride: usize,
    duration: f64,
    cancelled: &AtomicBool,
    mut report: impl FnMut(f64) -> Result<(), String>,
) -> Result<DecodeSummary, String> {
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let gpu = GpuPreprocessor::new()?;
    let ball_input = (!matches!(pass, ScanPass::PlayersOnly(_)))
        .then(|| ball_engine.allocate_cuda_clip_input())
        .transpose()?;
    let people_input = (!matches!(pass, ScanPass::BallOnly))
        .then(|| person_engine.allocate_cuda_input())
        .transpose()?;
    let mut observations = Vec::<RawDetection>::new();
    let mut sampled_frames: Vec<Option<Arc<GpuBuffer>>> = Vec::with_capacity(9);
    let mut expected_index = 0u64;
    let mut native_index = 0u64;
    let mut windows = 0usize;
    let mut last_geometry: Option<(usize, usize, usize, u8)> = None;
    let estimated = (duration * 30.0).max(9.0);
    let target_frames = (duration * 30.0).round().max(0.0) as u64;
    let start = range.map_or(0.0, |r| r.start_sec);
    let timestamps = &source.timestamps;
    let keep_pose = pose.borrow().is_some() && !matches!(pass, ScanPass::BallOnly);
    let receipt = decode_video_file(original, source.codec, cancelled, |frame| {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        if frame.frame_index != native_index {
            return Err("NVDEC display frames are not contiguous".into());
        }
        native_index += 1;
        let geometry = (frame.width, frame.height, frame.pitch, frame.bit_depth);
        if let Some(previous) = last_geometry {
            if previous != geometry {
                return Err("NVDEC source geometry changed during inference".into());
            }
        } else {
            last_geometry = Some(geometry);
        }
        if expected_index >= target_frames {
            return Ok(());
        }
        let desired = closest_display_frame(timestamps, start + expected_index as f64 / 30.0)
            .ok_or("Cannot align NVDEC source presentation time")?;
        if desired > frame.frame_index {
            return Ok(());
        }
        let input = Nv12View {
            ptr: frame.device_ptr,
            pitch: frame.pitch,
            width: frame.width,
            height: frame.height,
        };
        let normalize =
            frame.bit_depth != 8 || frame.chroma != 1 || source.rotation != 0 || source.full_range;
        let canonical = if normalize {
            Some(gpu.canonical_frame(
                input,
                frame.bit_depth,
                frame.chroma,
                source.rotation,
                source.full_range,
            )?)
        } else {
            None
        };
        let (width, height, pitch) = if let Some((_, width, height)) = &canonical {
            (*width, *height, *width)
        } else {
            (frame.width, frame.height, frame.pitch)
        };
        let view = Nv12View {
            ptr: canonical
                .as_ref()
                .map_or(frame.device_ptr, |item| item.0.ptr),
            pitch,
            width,
            height,
        };
        while expected_index < target_frames {
            let desired = closest_display_frame(timestamps, start + expected_index as f64 / 30.0)
                .ok_or("Cannot align NVDEC source presentation time")?;
            if desired != frame.frame_index {
                break;
            }
            let frame_slot = (expected_index % 9) as usize;
            if let Some(input) = &ball_input {
                gpu.ball_input(view, input, frame_slot)?;
            }
            let include_person = match pass {
                ScanPass::Both => true,
                ScanPass::BallOnly => false,
                ScanPass::PlayersOnly(mask) => mask.includes(expected_index),
            };
            if include_person && expected_index.is_multiple_of(person_stride as u64) {
                let input = people_input.as_ref().ok_or("Missing GPU person input")?;
                gpu.person_input(view, input, source.matrix)?;
                let found = person_engine.detect_cuda_input(input)?;
                for (index, candidate) in found.into_iter().enumerate() {
                    observations.push(RawDetection {
                        frame_index: expected_index,
                        class_name: "player",
                        score: candidate.score,
                        bbox_xyxy_norm: candidate.bbox.map(|value| value as f32),
                        query_index: Some(index),
                        touch_probability: None,
                        head_point_norm: candidate
                            .head
                            .map(|point| point.map(|value| value as f32)),
                    });
                }
            }
            if keep_pose && expected_index.is_multiple_of(person_stride as u64) {
                sampled_frames.push(Some(Arc::new(GpuBuffer::copy_frame(view)?)));
            } else {
                sampled_frames.push(None);
            }
            expected_index += 1;
            if frame_slot == 8 {
                let first = expected_index - 9;
                if let Some(input) = &ball_input {
                    let outputs = ball_engine
                        .infer_cuda_clip(input)
                        .map_err(|error| window_error(ModelKind::Ball, first, &error))?;
                    let balls = decode_vball(&outputs, first, 0.05)?;
                    ball_writer
                        .as_deref_mut()
                        .ok_or("Missing GPU ball evidence writer")?
                        .push(first, &balls)?;
                }
                if let Some(writer) = person_writer.as_deref_mut() {
                    writer.push(first, &observations)?;
                }
                if keep_pose {
                    let mut active_pose = pose.borrow_mut();
                    let stage = active_pose.as_mut().ok_or("Missing primary pose stage")?;
                    stage.observe_cuda(
                        first,
                        std::mem::take(&mut sampled_frames),
                        pitch,
                        source.matrix,
                        width,
                        height,
                        &observations,
                        cancelled,
                    )?;
                } else {
                    sampled_frames.clear();
                }
                observations.clear();
                windows += 1;
                if windows.is_multiple_of(10) {
                    if ball_writer.is_some() {
                        update_progress(&ball.status, windows, expected_index, estimated);
                        persist(&ball.folder, &snapshot(&ball.status))?;
                    }
                    if person_writer.is_some() {
                        update_progress(&players.status, windows, expected_index, estimated);
                        persist(&players.folder, &snapshot(&players.status))?;
                    }
                    report((expected_index as f64 / estimated).clamp(0.0, 0.99))?;
                }
            }
        }
        Ok(())
    })?;
    if receipt.frames != native_index || timestamps.len() as u64 != receipt.frames {
        return Err("NVDEC display frames differ from original packet timestamps".into());
    }
    if expected_index != target_frames {
        return Err("NVDEC supplied fewer model frames than selected video interval".into());
    }
    if windows == 0 || expected_index < windows as u64 * 9 {
        return Err("NVDEC video does not contain a complete model window".into());
    }
    if keep_pose {
        let mut active_pose = pose.borrow_mut();
        let stage = active_pose.as_mut().ok_or("Missing primary pose stage")?;
        stage.tail_cuda(expected_index, cancelled)?;
    }
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    drop(people_input);
    drop(ball_input);
    Ok(DecodeSummary {
        first_frame: 0,
        windows,
        decoded_frames: expected_index,
        stride: 9,
    })
}
