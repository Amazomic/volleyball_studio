use super::*;
use std::sync::Arc;
use volley_inference::cuda_input::CudaInput;
mod frame_queue;
use frame_queue::FrameQueue;

struct PersonBatch {
    input: CudaInput,
    frames: Vec<u64>,
}

use volley_inference::gpu_preprocess::{GpuBuffer, GpuPreprocessor, Nv12View};
use volley_media::nvdec::{
    closest_display_frame, decode_video_file_window_with_map_ranges, keyframe_window,
    NvdecDecodeWindow,
};

#[derive(Clone, Copy)]
pub(super) enum ScanPass<'a> {
    BallOnly,
    PlayersOnly(&'a rally_mask::RallyMask),
}

struct GpuFrameCompletion<'a> {
    gpu: &'a GpuPreprocessor,
    complete: bool,
}

impl Drop for GpuFrameCompletion<'_> {
    fn drop(&mut self) {
        if !self.complete {
            let _ = self.gpu.finish();
        }
    }
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

fn planned_map_ranges(
    timeline: &volley_media::nvdec::DisplayTimeline,
    window: Option<NvdecDecodeWindow>,
    selection: &[(u64, u64)],
    start: f64,
) -> Result<Option<Vec<(u64, u64)>>, String> {
    let Some(window) = window else {
        return Ok(None);
    };
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    for &(first, last) in selection {
        let left = closest_display_frame(&timeline.timestamps, start + first as f64 / 30.0)
            .ok_or("Cannot map first NVDEC rally frame")?
            .saturating_sub(1);
        let right = closest_display_frame(&timeline.timestamps, start + (last + 1) as f64 / 30.0)
            .ok_or("Cannot map last NVDEC rally frame")?
            .saturating_add(1);
        if right < window.first_frame || timeline.timestamps[left as usize] >= window.end_sec {
            continue;
        }
        let left = left.max(window.first_frame) - window.first_frame;
        let right = right - window.first_frame;
        if let Some(previous) = ranges.last_mut() {
            if left <= previous.1.saturating_add(1) {
                previous.1 = previous.1.max(right);
                continue;
            }
        }
        ranges.push((left, right));
    }
    if ranges.is_empty() {
        return Err("NVDEC seek window has no mapped frames for selected rallies".into());
    }
    Ok(Some(ranges))
}

#[expect(clippy::too_many_arguments)]
pub(super) fn run(
    original: &Path,
    source: ZeroCopySource,
    range: Option<volley_media::raw_decoder::AnalysisRange>,
    ball_engine: Option<&mut Engine>,
    person_engine: Option<&mut PersonEngine>,
    pose: Option<&GpuPoseStage>,
    ball_writer: Option<&mut EvidenceWriter>,
    person_writer: Option<&mut EvidenceWriter>,
    pass: ScanPass<'_>,
    ball: &JobContext,
    players: &JobContext,
    person_stride: usize,
    duration: f64,
    cancelled: &AtomicBool,
    report: impl FnMut(f64, u64) -> Result<(), String>,
) -> Result<DecodeSummary, String> {
    if matches!(pass, ScanPass::PlayersOnly(_)) {
        let person_engine = person_engine.ok_or("Missing RF-DETR engine")?;
        let batch_size = person_engine.batch_size();
        let allocate = || -> Result<PersonBatch, String> {
            Ok(PersonBatch {
                input: person_engine.allocate_cuda_input()?,
                frames: Vec::with_capacity(batch_size),
            })
        };
        let buffers = [allocate()?, allocate()?];
        return std::thread::scope(|scope| {
            let mut binding = None;
            let (mut queue, worker) = FrameQueue::start(scope, buffers, move |_, batch| {
                if cancelled.load(Ordering::Acquire) {
                    return Err("cancelled".into());
                }
                if binding.is_none() {
                    volley_inference::gpu_preprocess::bind_cuda_thread()?;
                    binding = Some(person_engine.create_cuda_detector_binding()?);
                }
                let found = person_engine.detect_cuda_batch_bound(
                    &batch.input,
                    binding.as_mut().ok_or("Missing RF-DETR binding")?,
                    batch.frames.len(),
                )?;
                Ok(found
                    .into_iter()
                    .zip(batch.frames.iter().copied())
                    .flat_map(|(items, frame_index)| {
                        items
                            .into_iter()
                            .enumerate()
                            .map(move |(index, candidate)| RawDetection {
                                frame_index,
                                class_name: "player",
                                score: candidate.score,
                                bbox_xyxy_norm: candidate.bbox.map(|value| value as f32),
                                query_index: Some(index),
                                touch_probability: None,
                                head_point_norm: candidate
                                    .head
                                    .map(|point| point.map(|value| value as f32)),
                            })
                    })
                    .collect())
            });
            let result = run_inner(
                original,
                source,
                range,
                ball_engine,
                Some(&mut queue),
                batch_size,
                pose,
                ball_writer,
                person_writer,
                pass,
                ball,
                players,
                person_stride,
                duration,
                cancelled,
                report,
            );
            drop(queue);
            worker.join().map_err(|_| "RF-DETR worker panicked")??;
            result
        });
    }
    run_inner(
        original,
        source,
        range,
        ball_engine,
        None,
        1,
        pose,
        ball_writer,
        person_writer,
        pass,
        ball,
        players,
        person_stride,
        duration,
        cancelled,
        report,
    )
}

#[expect(clippy::too_many_arguments)]
fn run_inner(
    original: &Path,
    source: ZeroCopySource,
    range: Option<volley_media::raw_decoder::AnalysisRange>,
    mut ball_engine: Option<&mut Engine>,
    mut person_queue: Option<&mut FrameQueue<PersonBatch, Vec<RawDetection>>>,
    batch_size: usize,
    pose: Option<&GpuPoseStage>,
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
    let ball_input = if matches!(pass, ScanPass::BallOnly) {
        Some(
            ball_engine
                .as_deref_mut()
                .ok_or("Missing VBallNet engine")?
                .allocate_cuda_clip_input()?,
        )
    } else {
        None
    };
    let mut observations = Vec::<RawDetection>::new();
    let mut prepared: Option<PersonBatch> = None;
    let mut cached_window: Option<Vec<RawDetection>> = None;
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
    let keep_pose = pose.is_some() && !matches!(pass, ScanPass::BallOnly);
    let selection = match pass {
        ScanPass::PlayersOnly(mask) => mask.selected.clone(),
        ScanPass::BallOnly => vec![(0, target_frames.saturating_sub(1))],
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
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let frame_slot = (frame_index % 9) as usize;
        if frame_slot == 0 {
            cached_window = match pass {
                ScanPass::BallOnly => ball_writer
                    .as_deref_mut()
                    .ok_or("Missing ball writer")?
                    .cached(frame_index)?,
                ScanPass::PlayersOnly(_) => person_writer
                    .as_deref_mut()
                    .ok_or("Missing person writer")?
                    .cached(frame_index)?,
            };
        }
        let mut needs_finish = view.is_some();
        if cached_window.is_none() {
            if let Some(input) = &ball_input {
                let source = view.ok_or("Missing selected NVDEC ball frame")?;
                gpu.ball_input(source, input, frame_slot)?;
            }
        }
        let include_person = match pass {
            ScanPass::BallOnly => false,
            ScanPass::PlayersOnly(mask) => mask.includes(frame_index),
        };
        if include_person && frame_index.is_multiple_of(person_stride as u64) {
            player_samples += 1;
        }
        if cached_window.is_none()
            && frame_index < target_frames / 9 * 9
            && include_person
            && frame_index.is_multiple_of(person_stride as u64)
        {
            let model_view = view.ok_or("Missing selected NVDEC player frame")?;
            let queue = person_queue
                .as_deref_mut()
                .ok_or("Missing RF-DETR inference queue")?;
            if prepared.is_none() {
                let (mut batch, completed) = queue.acquire()?;
                if let Some((_, mut found)) = completed {
                    observations.append(&mut found);
                }
                batch.frames.clear();
                prepared = Some(batch);
            }
            let batch = prepared.as_mut().ok_or("Missing RF-DETR prepared batch")?;
            let preparation =
                gpu.person_input_slot(model_view, &batch.input, source.matrix, batch.frames.len());
            let completion = gpu.finish();
            preparation?;
            completion?;
            needs_finish = false;
            batch.frames.push(frame_index);
            if batch.frames.len() == batch_size {
                let batch = prepared.take().ok_or("Missing full RF-DETR batch")?;
                queue.submit(batch.frames[0], batch)?;
            }
        }
        if keep_pose && include_person && frame_index.is_multiple_of(person_stride as u64) {
            let source = view.ok_or("Missing selected CUDA pose source")?;
            sampled_frames.push(Some(Arc::new(GpuBuffer::copy_frame(source)?)));
        } else {
            sampled_frames.push(None);
        }
        if needs_finish {
            gpu.finish()?;
        }
        let next_index = frame_index + 1;
        if frame_slot == 8 {
            let first = next_index - 9;
            if let Some(input) = &ball_input {
                let balls = match cached_window.take() {
                    Some(records) => records,
                    None => {
                        let outputs = ball_engine
                            .as_deref_mut()
                            .ok_or("Missing VBallNet engine")?
                            .infer_cuda_clip(input)
                            .map_err(|error| window_error(ModelKind::Ball, first, &error))?;
                        decode_vball(&outputs, first, 0.05)?
                    }
                };
                ball_writer
                    .as_deref_mut()
                    .ok_or("Missing GPU ball evidence writer")?
                    .push(first, &balls)?;
            }
            if let Some(writer) = person_writer.as_deref_mut() {
                if let Some(queue) = person_queue.as_deref_mut() {
                    if let Some(batch) = prepared.take() {
                        gpu.pad_person_input(&batch.input, batch.frames.len())?;
                        queue.submit(batch.frames[0], batch)?;
                    }
                    while queue.pending() > 0 {
                        let (_, mut found) = queue.receive()?;
                        observations.append(&mut found);
                    }
                }
                if let Some(records) = cached_window.take() {
                    observations = records;
                }
                writer.push(first, &observations)?;
            }
            if keep_pose {
                pose.ok_or("Missing primary pose stage")?.observe_cuda(
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
        let mut last_mapped = None;
        let map_ranges = planned_map_ranges(&source.timeline, window, &selection, start)?;
        let receipt = decode_video_file_window_with_map_ranges(
            original,
            source.codec,
            window,
            map_ranges,
            cancelled,
            |frame| {
                let mut completion = GpuFrameCompletion {
                    gpu: &gpu,
                    complete: false,
                };
                if last_mapped.is_some_and(|previous| frame.frame_index <= previous) {
                    return Err("NVDEC mapped source frames are out of presentation order".into());
                }
                last_mapped = Some(frame.frame_index);
                let source_index = window.map_or(0, |item| item.first_frame) + frame.frame_index;
                if expected_index >= target_frames {
                    completion.complete = true;
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
                    completion.complete = true;
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
                completion.complete = true;
                Ok(())
            },
        )?;
        if last_mapped.is_some_and(|index| index >= receipt.frames) {
            return Err("NVDEC mapped frame lies outside the decoder output".into());
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
        pose.ok_or("Missing primary pose stage")?
            .tail_cuda(expected_index, cancelled)?;
    }
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }

    if let Some(queue) = person_queue {
        while queue.pending() > 0 {
            queue.receive()?;
        }
    }
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
    fn mapped_frames_skip_gop_prelude_and_keep_rally_indices() {
        let timeline = volley_media::nvdec::DisplayTimeline {
            timestamps: (0..900).map(|index| index as f64 / 30.0).collect(),
            keyframes: vec![0, 150, 300, 450, 600, 750],
        };
        let windows = plan_decode_windows(&timeline, 0.0, 900, &[(170, 220)], false).unwrap();
        let ranges = planned_map_ranges(&timeline, windows[0], &[(170, 220)], 0.0)
            .unwrap()
            .unwrap();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], (19, 72));
        assert!(ranges[0].0 > 0);
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
