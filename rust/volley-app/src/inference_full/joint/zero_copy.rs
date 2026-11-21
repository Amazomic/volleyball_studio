use super::*;
use std::sync::Arc;
use volley_inference::gpu_preprocess::{GpuBuffer, GpuPreprocessor, Nv12View};
use volley_media::nvdec::{
    closest_display_frame, decode_video_file_window, keyframe_window, NvdecDecodeWindow,
};

#[derive(Clone, Copy)]
pub(super) enum ScanPass<'a> {
    Both,
    BallOnly,
    PlayersOnly(&'a rally_mask::RallyMask),
}

fn plan_decode_windows(
    timeline: &volley_media::nvdec::DisplayTimeline,
    start: f64,
    target_frames: u64,
    selection: &[(u64, u64)],
    whole_source: bool,
) -> Result<Vec<Option<NvdecDecodeWindow>>, String> {
    if whole_source {
        return Ok(vec![None]);
    }
    let mut decode_windows = Vec::<Option<NvdecDecodeWindow>>::new();
    for &(first, last) in selection {
        if first > last || last >= target_frames {
            return Err("NVDEC rally interval is outside the selected video range".into());
        }
        let begin = start + first as f64 / 30.0;
        let end = start + (last + 1) as f64 / 30.0 + 0.3;
        let window = keyframe_window(timeline, begin, end)?;
        if let Some(Some(previous)) = decode_windows.last_mut() {
            if window.seek_sec <= previous.end_sec {
                previous.end_sec = previous.end_sec.max(window.end_sec);
                continue;
            }
        }
        decode_windows.push(Some(window));
    }
    Ok(decode_windows)
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
    mut report: impl FnMut(f64, u64) -> Result<(), String>,
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
    let mut windows = 0usize;
    let mut player_samples = 0u64;
    let mut last_geometry: Option<(usize, usize, usize, u8)> = None;
    let mut last_shape: Option<(usize, usize, usize)> = None;
    let estimated = (duration * 30.0).max(9.0);
    let target_frames = (duration * 30.0).round().max(0.0) as u64;
    let start = range.map_or(0.0, |r| r.start_sec);
    let timestamps = &source.timeline.timestamps;
    let keep_pose = pose.borrow().is_some() && !matches!(pass, ScanPass::BallOnly);
    let selection = match pass {
        ScanPass::PlayersOnly(mask) => mask.selected.clone(),
        ScanPass::Both | ScanPass::BallOnly => vec![(0, target_frames.saturating_sub(1))],
    };
    let whole_source = start == 0.0 && range.is_none() && !matches!(pass, ScanPass::PlayersOnly(_));
    let decode_windows = plan_decode_windows(
        &source.timeline,
        start,
        target_frames,
        &selection,
        whole_source,
    )?;
    let mut process_frame = |frame_index: u64,
                             view: Option<Nv12View>,
                             pitch: usize,
                             width: usize,
                             height: usize|
     -> Result<(), String> {
        let frame_slot = (frame_index % 9) as usize;
        if let Some(input) = &ball_input {
            let source = view.ok_or("Missing selected NVDEC ball frame")?;
            gpu.ball_input(source, input, frame_slot)?;
        }
        let include_person = match pass {
            ScanPass::Both => true,
            ScanPass::BallOnly => false,
            ScanPass::PlayersOnly(mask) => mask.includes(frame_index),
        };
        if include_person && frame_index.is_multiple_of(person_stride as u64) {
            let model_view = view.ok_or("Missing selected NVDEC player frame")?;
            let input = people_input.as_ref().ok_or("Missing GPU person input")?;
            gpu.person_input(model_view, input, source.matrix)?;
            let found = person_engine.detect_cuda_input(input)?;
            player_samples += 1;
            for (index, candidate) in found.into_iter().enumerate() {
                observations.push(RawDetection {
                    frame_index,
                    class_name: "player",
                    score: candidate.score,
                    bbox_xyxy_norm: candidate.bbox.map(|value| value as f32),
                    query_index: Some(index),
                    touch_probability: None,
                    head_point_norm: candidate.head.map(|point| point.map(|value| value as f32)),
                });
            }
        }
        if keep_pose && include_person && frame_index.is_multiple_of(person_stride as u64) {
            let source = view.ok_or("Missing selected CUDA pose source")?;
            sampled_frames.push(Some(Arc::new(GpuBuffer::copy_frame(source)?)));
        } else {
            sampled_frames.push(None);
        }
        let next_index = frame_index + 1;
        if frame_slot == 8 {
            let first = next_index - 9;
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
                    matches!(pass, ScanPass::PlayersOnly(_)),
                    cancelled,
                )?;
            } else {
                sampled_frames.clear();
            }
            observations.clear();
            windows += 1;
            if windows.is_multiple_of(10) {
                if ball_writer.is_some() {
                    update_progress(&ball.status, windows, next_index, estimated);
                    persist(&ball.folder, &snapshot(&ball.status))?;
                }
                if person_writer.is_some() {
                    update_progress(&players.status, windows, next_index, estimated);
                    persist(&players.folder, &snapshot(&players.status))?;
                }
                let done = if matches!(pass, ScanPass::PlayersOnly(_)) {
                    player_samples
                } else {
                    next_index
                };
                report((next_index as f64 / estimated).clamp(0.0, 0.99), done)?;
            }
        }
        Ok(())
    };
    for window in decode_windows {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let mut native_index = 0u64;
        let receipt = decode_video_file_window(
            original,
            source.codec,
            window,
            cancelled,
            |frame| {
                if frame.frame_index != native_index {
                    return Err(
                        "NVDEC display frames are not contiguous within the seek window".into(),
                    );
                }
                native_index += 1;
                let source_index = window.map_or(0, |item| item.first_frame) + frame.frame_index;
                if expected_index >= target_frames {
                    return Ok(());
                }
                let geometry = (frame.width, frame.height, frame.pitch, frame.bit_depth);
                if let Some(previous) = last_geometry {
                    if previous != geometry {
                        return Err("NVDEC source geometry changed during inference".into());
                    }
                } else {
                    last_geometry = Some(geometry);
                }
                let needed =
                    closest_display_frame(timestamps, start + expected_index as f64 / 30.0)
                        .ok_or("Cannot align NVDEC source presentation time")?;
                if needed > source_index {
                    return Ok(());
                }
                let input = Nv12View {
                    ptr: frame.device_ptr,
                    pitch: frame.pitch,
                    width: frame.width,
                    height: frame.height,
                };
                let normalize = frame.bit_depth != 8
                    || frame.chroma != 1
                    || source.rotation != 0
                    || source.full_range;
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
                last_shape = Some((pitch, width, height));
                let view = Nv12View {
                    ptr: canonical
                        .as_ref()
                        .map_or(frame.device_ptr, |item| item.0.ptr),
                    pitch,
                    width,
                    height,
                };
                while expected_index < target_frames {
                    let desired =
                        closest_display_frame(timestamps, start + expected_index as f64 / 30.0)
                            .ok_or("Cannot align NVDEC source presentation time")?;
                    if desired > source_index {
                        break;
                    }
                    if desired < source_index {
                        if !matches!(pass, ScanPass::PlayersOnly(mask) if !mask.includes(expected_index))
                        {
                            return Err(
                                "NVDEC seek omitted a source frame required for inference".into()
                            );
                        }
                        process_frame(expected_index, None, pitch, width, height)?;
                        expected_index += 1;
                    } else {
                        process_frame(expected_index, Some(view), pitch, width, height)?;
                        expected_index += 1;
                    }
                }
                Ok(())
            },
        )?;
        if receipt.frames != native_index {
            return Err("NVDEC seek window frame count differs from decoder output".into());
        }
    }
    if let Some((pitch, width, height)) = last_shape {
        while expected_index < target_frames {
            if !matches!(pass, ScanPass::PlayersOnly(mask) if !mask.includes(expected_index)) {
                return Err(
                    "NVDEC supplied fewer model frames than the selected video interval".into(),
                );
            }
            process_frame(expected_index, None, pitch, width, height)?;
            expected_index += 1;
        }
    }
    drop(process_frame);
    if expected_index != target_frames {
        return Err("NVDEC source did not supply the required video interval".into());
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

#[cfg(test)]
mod bounded_tests {
    use super::*;

    #[test]
    fn distant_rallies_decode_separate_gops_and_preserve_global_frames() {
        let timeline = volley_media::nvdec::DisplayTimeline {
            timestamps: (0..3600).map(|index| index as f64 / 30.0).collect(),
            keyframes: (0..3600).step_by(90).collect(),
        };
        let windows =
            plan_decode_windows(&timeline, 0.0, 3600, &[(120, 210), (2400, 2500)], false).unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].unwrap().first_frame, 90);
        assert_eq!(windows[1].unwrap().first_frame, 2340);
        assert!(windows[0].unwrap().end_sec < windows[1].unwrap().seek_sec);
        assert_eq!(
            plan_decode_windows(&timeline, 0.0, 3600, &[(120, 210)], true).unwrap(),
            vec![None]
        );
        assert!(plan_decode_windows(&timeline, 0.0, 3600, &[(4000, 4100)], false).is_err());
    }

    #[test]
    fn overlapping_decoder_gops_are_coalesced() {
        let timeline = volley_media::nvdec::DisplayTimeline {
            timestamps: (0..900).map(|index| index as f64 / 30.0).collect(),
            keyframes: vec![0, 150, 300, 450, 600, 750],
        };
        let windows =
            plan_decode_windows(&timeline, 0.0, 900, &[(170, 220), (270, 350)], false).unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].unwrap().first_frame, 150);
        assert!(windows[0].unwrap().end_sec > 11.0);
    }
}
