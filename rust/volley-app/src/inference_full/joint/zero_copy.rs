use super::*;
use std::sync::Arc;
use volley_inference::gpu_preprocess::{GpuBuffer, GpuPreprocessor, Nv12View};
use volley_media::nvdec::{decode_video_file, NvdecCodec};

#[expect(clippy::too_many_arguments)]
pub(super) fn run(
    original: &Path,
    codec: NvdecCodec,
    bt709: bool,
    ball_engine: &mut Engine,
    person_engine: &mut PersonEngine,
    pose: &RefCell<Option<PoseStream>>,
    ball_writer: &mut EvidenceWriter,
    person_writer: &mut EvidenceWriter,
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
    let ball_input = ball_engine.allocate_cuda_clip_input()?;
    let people_input = person_engine.allocate_cuda_input()?;
    let mut observations = Vec::<RawDetection>::new();
    let mut sampled_frames: Vec<Option<Arc<GpuBuffer>>> = Vec::with_capacity(9);
    let mut expected_index = 0u64;
    let mut windows = 0usize;
    let mut last_geometry: Option<(usize, usize, usize)> = None;
    let estimated = (duration * 30.0).max(9.0);
    let keep_pose = pose.borrow().is_some();
    let receipt = decode_video_file(original, codec, cancelled, |frame| {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        if frame.frame_index != expected_index {
            return Err("NVDEC display frames are not contiguous".into());
        }
        let geometry = (frame.width, frame.height, frame.pitch);
        if let Some(previous) = last_geometry {
            if previous != geometry {
                return Err("NVDEC source geometry changed during inference".into());
            }
        } else {
            last_geometry = Some(geometry);
        }
        let view = Nv12View {
            ptr: frame.device_ptr,
            pitch: frame.pitch,
            width: frame.width,
            height: frame.height,
        };
        let frame_slot = (expected_index % 9) as usize;
        gpu.ball_input(view, &ball_input, frame_slot)?;
        if expected_index.is_multiple_of(person_stride as u64) {
            gpu.person_input(view, &people_input, bt709)?;
            let found = person_engine.detect_cuda_input(&people_input)?;
            for (index, candidate) in found.into_iter().enumerate() {
                observations.push(RawDetection {
                    frame_index: expected_index,
                    class_name: "player",
                    score: candidate.score,
                    bbox_xyxy_norm: candidate.bbox.map(|value| value as f32),
                    query_index: Some(index),
                    touch_probability: None,
                    head_point_norm: candidate.head.map(|point| point.map(|value| value as f32)),
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
            let outputs = ball_engine
                .infer_cuda_clip(&ball_input)
                .map_err(|error| window_error(ModelKind::Ball, first, &error))?;
            let balls = decode_vball(&outputs, first, 0.05)?;
            ball_writer.push(first, &balls)?;
            person_writer.push(first, &observations)?;
            if let Some(stage) = pose.borrow_mut().as_mut() {
                stage.observe_cuda(
                    first,
                    std::mem::take(&mut sampled_frames),
                    frame.pitch,
                    bt709,
                    frame.width,
                    frame.height,
                    &observations,
                    cancelled,
                )?;
            } else {
                sampled_frames.clear();
            }
            observations.clear();
            windows += 1;
            if windows.is_multiple_of(10) {
                update_progress(&ball.status, windows, expected_index, estimated);
                update_progress(&players.status, windows, expected_index, estimated);
                persist(&ball.folder, &snapshot(&ball.status))?;
                persist(&players.folder, &snapshot(&players.status))?;
                report((expected_index as f64 / estimated).clamp(0.0, 0.99))?;
            }
        }
        Ok(())
    })?;
    if receipt.frames != expected_index {
        return Err("NVDEC receipt does not match observed display frames".into());
    }
    if windows == 0 || expected_index < windows as u64 * 9 {
        return Err("NVDEC video does not contain a complete model window".into());
    }
    if let Some(stage) = pose.borrow_mut().as_mut() {
        stage.tail_cuda(receipt.frames, cancelled)?;
    }
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    Ok(DecodeSummary {
        first_frame: 0,
        windows,
        decoded_frames: receipt.frames,
        stride: 9,
    })
}
