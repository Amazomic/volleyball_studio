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
use volley_core::court_geometry::CourtTransform;

// The detector provides the two physically different halves and observed
// net area from one source of truth. No artificial net is fabricated.
fn zone_response(corners: [[f64; 2]; 4], net: Option<[[f64; 2]; 2]>) -> Value {
    CourtTransform::new(corners)
        .and_then(|t| t.with_net(net))
        .ok()
        .and_then(|t| t.zones())
        .map_or(Value::Null, |zones| json!(zones))
}

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
    if !(3..=7).contains(&count) || frames.len() != frame_bytes.checked_mul(count)? {
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
    let dimensions = volley_media::probe(&source).map_err(storage::bad_request)?;
    let aspect_ratio = dimensions.width as f64 / dimensions.height as f64;
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
    if strong.is_none() {
        // Only borrow a human-confirmed calibration from the same apparent camera view.
        // The proposed lines remain advisory and are never saved automatically.
        if let Some(mut learned) = output
            .stdout
            .chunks_exact(frame_bytes)
            .filter_map(|pixels| crate::training_samples::match_court(&state, pixels, aspect_ratio))
            .max_by(|a, b| {
                a["confidence"]
                    .as_f64()
                    .unwrap_or(0.0)
                    .total_cmp(&b["confidence"].as_f64().unwrap_or(0.0))
            })
        {
            if let Ok(corners) = serde_json::from_value::<[[f64; 2]; 4]>(learned["court"].clone()) {
                let net = serde_json::from_value::<Option<[[f64; 2]; 2]>>(learned["net"].clone())
                    .unwrap_or(None);
                learned["zones"] = zone_response(corners, net);
            }
            learned["sampled_frames"] = json!(total);
            learned["supporting_frames"] = json!(0);
            learned["review_required"] = json!(true);
            learned["message"] = json!("Найден похожий ранее размеченный ракурс. Проверьте углы и сетку перед сохранением.");
            return Ok(Json(learned));
        }
    }
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
            "zones":zone_response(c.court, c.net),
            "confidence":c.confidence,"sampled_frames":total,
            "supporting_frames":detected.len(),"weak_consensus":weak || c.inferred,
            "inferred":c.inferred,"temporal_composite":temporal_composite}),
        None => json!({"detected":false,"court":[],"net":null,"zones":null,
            "sampled_frames":total,"supporting_frames":detected.len(),
            "message":"Не удалось уверенно выделить границы площадки на этих кадрах. Выберите более четкий момент видео или отметьте углы вручную"}),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zones_are_derived_without_fabricating_unobserved_net() {
        let corners = [[0.1, 0.8], [0.35, 0.2], [0.7, 0.2], [0.95, 0.8]];
        let without = zone_response(corners, None);
        assert!(without["near_half"].is_array());
        assert!(without["far_half"].is_array());
        assert!(without.get("net_surface").is_none());
        let with = zone_response(corners, Some([[0.2, 0.4], [0.8, 0.4]]));
        assert!(with["net_surface"].is_array());
    }

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
