//! Bounded source-video FFmpeg sampling; pure Rust CPU geometry detection.
//! Detection is advisory: a user confirms/adjusts it in the editor.
use crate::{
    storage::{self, ApiResult},
    AppState,
};
use axum::{extract::State, http::HeaderMap, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{process::Stdio, sync::Arc, time::Duration};
use volley_core::court_detection::{combine_frames, combine_frames_advisory, detect_frame};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DetectionRequest {
    project_id: String,
    video: String,
    #[serde(default)]
    start_sec: f64,
}

/// Median background across a short fixed-camera window removes moving
/// players that otherwise obscure the same painted line on every frame.
fn temporal_median_rgb(frames: &[u8], frame_bytes: usize, count: usize) -> Option<Vec<u8>> {
    if count < 3 || count > 7 || frames.len() != frame_bytes.checked_mul(count)? {
        return None;
    }
    let mut result = vec![0_u8; frame_bytes];
    for (offset, channel) in result.iter_mut().enumerate() {
        let mut samples = [0_u8; 7];
        for (index, sample) in samples.iter_mut().enumerate().take(count) {
            *sample = frames[index * frame_bytes + offset];
        }
        samples[..count].sort_unstable();
        *channel = samples[count / 2];
    }
    Some(result)
}

pub(crate) async fn detect(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<DetectionRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    if !req.start_sec.is_finite() || !(0.0..=172800.0).contains(&req.start_sec) {
        return Err(storage::bad_request("Invalid calibration sample time"));
    }
    // Existing source-ownership validation is shared with video edits.
    let source = crate::video_edits::calibration_source(&state, &req.project_id, &req.video)?;
    const W: usize = 640;
    const H: usize = 360;
    const N: usize = 7;
    let mut command = tokio::process::Command::new("ffmpeg");
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "2",
        ])
        .args(["-ss", &format!("{:.3}", (req.start_sec - 3.0).max(0.0))])
        .arg("-i")
        .arg(source)
        .args([
            "-t",
            "8",
            "-vf",
            "fps=1,scale=640:360:flags=bicubic,format=rgb24",
        ])
        .args([
            "-frames:v",
            "7",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(35), command.output())
        .await
        .map_err(|_| storage::bad_request("Court frame extraction timed out"))?
        .map_err(|e| storage::bad_request(format!("FFmpeg unavailable: {e}")))?;
    if !output.status.success() {
        return Err(storage::bad_request("Unable to decode calibration frames"));
    }
    let frame_bytes = W * H * 3;
    if output.stdout.len() % frame_bytes != 0 || output.stdout.len() > N * frame_bytes {
        return Err(storage::bad_request(
            "Invalid decoded court frame dimensions",
        ));
    }
    let total = output.stdout.len() / frame_bytes;
    let detected: Vec<_> = output
        .stdout
        .chunks_exact(frame_bytes)
        .filter_map(|pixels| detect_frame(pixels, W, H))
        .collect();
    // Strong agreement is preferred. Two independent matching frames are
    // still useful as a reviewed suggestion when lines are briefly occluded.
    let strong = combine_frames(&detected);
    let weak = strong.is_none();
    let advisory = if weak {
        combine_frames_advisory(&detected)
    } else {
        None
    };
    // This last resort is a *single* composite observation, never counted
    // as N independent votes. Every result still needs manual confirmation.
    let temporal_composite = strong.is_none() && advisory.is_none();
    let result = strong.or(advisory).or_else(|| {
        temporal_median_rgb(&output.stdout, frame_bytes, total)
            .and_then(|median| detect_frame(&median, W, H))
    });
    Ok(Json(match result {
        Some(c) => json!({"detected":true,"court":c.court,"net":c.net,
            "confidence":c.confidence,"sampled_frames":total,
            "supporting_frames":detected.len(),"weak_consensus":weak || c.inferred,
            "inferred":c.inferred,"temporal_composite":temporal_composite}),
        None => json!({"detected":false,"court":[],"net":null,
            "sampled_frames":total,"supporting_frames":detected.len(),
            "message":"Не удалось уверенно выделить границы площадки на этих кадрах. Выберите более четкий момент видео или отметьте углы вручную"}),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_preserves_static_floor_and_removes_moving_occlusions() {
        // Five video frames; the same white paint is visible in every frame,
        // but a dark player silhouette covers it once at each pixel.
        let frames = [
            [10_u8, 210, 210],
            [210, 10, 210],
            [210, 210, 10],
            [210, 210, 210],
            [210, 210, 210],
        ];
        let flattened: Vec<u8> = frames.into_iter().flatten().collect();
        assert_eq!(
            temporal_median_rgb(&flattened, 3, 5),
            Some(vec![210, 210, 210])
        );
        assert!(temporal_median_rgb(&flattened, 3, 2).is_none());
    }
}
