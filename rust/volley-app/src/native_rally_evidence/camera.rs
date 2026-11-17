use super::{ValidatedRun, MAX_EVIDENCE};
use crate::{
    projects,
    storage::{safe_file, BoundedReader},
    AppState,
};
use serde_json::Value;

/// A camera track is derived only from a complete SHA-verified native replay.
/// Low-score, isolated and motionless detections cannot steer the crop.
pub(crate) fn verified_camera_tracks(
    state: &AppState,
    run: &ValidatedRun,
    clips: &[volley_media::montage::SelectedClip],
) -> Result<Vec<Vec<volley_media::montage::CameraPoint>>, String> {
    use volley_media::montage::CameraPoint;
    if clips.is_empty()
        || clips.len() > 80
        || clips.iter().any(|clip| {
            clip.video != run.video
                || clip.start_frame > clip.end_frame
                || clip.end_frame >= run.frames
        })
    {
        return Err("Камера: клипы не принадлежат проверенному CPU-источнику".into());
    }
    let base = projects::project_root(state)
        .join(&run.project_id)
        .join("native_cpu_replays")
        .join(&run.replay_id);
    let path = safe_file(&base, "tracked_frames.jsonl", MAX_EVIDENCE)?;
    let mut rows = BoundedReader::open(&path)?;
    let mut found = vec![Vec::<CameraPoint>::new(); clips.len()];
    for frame in 0..run.frames {
        let bytes = rows.line()?.ok_or("Camera source replay is truncated")?;
        let row: Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("Bad camera replay frame: {e}"))?;
        if row["frame_index"].as_u64() != Some(frame)
            || row["track_ids_provisional"] != true
            || row["confirmed_athlete_identities"] != false
            || row["touches_confirmed"] != false
        {
            return Err("Camera frame has incorrect source identity or order".into());
        }
        let ball = &row["selected_ball"];
        if ball["source_model"] != "vballnet" {
            continue;
        }
        let detection = &ball["detection"];
        let score = detection["score"].as_f64().unwrap_or_default();
        if !score.is_finite()
            || score < 0.55
            || detection["frame_index"].as_u64() != Some(frame)
            || detection["class_name"] != "ball"
        {
            continue;
        }
        let Some(coords) = detection["bbox_xyxy_norm"].as_array() else {
            continue;
        };
        if coords.len() != 4 {
            continue;
        }
        let Some(v) = coords.iter().map(Value::as_f64).collect::<Option<Vec<_>>>() else {
            continue;
        };
        if !v.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            || v[0] >= v[2]
            || v[1] >= v[3]
        {
            continue;
        }
        let (x, y) = ((v[0] + v[2]) / 2.0, (v[1] + v[3]) / 2.0);
        for (clip, track) in clips.iter().zip(found.iter_mut()) {
            if (clip.start_frame..=clip.end_frame).contains(&frame) {
                track.push(CameraPoint {
                    time_sec: (frame - clip.start_frame) as f64 / 30.0,
                    x,
                    y,
                });
            }
        }
    }
    if rows.line()?.is_some() || rows.finish() != run.replay_sha {
        return Err("Камера: исходный видеотрек изменён (SHA256)".into());
    }
    found.into_iter().map(clean_camera_track).collect()
}
fn clean_camera_track(
    raw: Vec<volley_media::montage::CameraPoint>,
) -> Result<Vec<volley_media::montage::CameraPoint>, String> {
    use volley_media::montage::CameraPoint;
    if raw.len() < 3 {
        return Err("Недостаточно уверенных точек VBallNet для слежения за мячом".into());
    }
    // An observation needs a plausible moving neighbour. A fixed false positive
    // and a sudden cross-court jump are both excluded before smoothing.
    let plausible = |a: &CameraPoint, b: &CameraPoint| {
        let delta_t = (b.time_sec - a.time_sec).abs();
        let distance = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
        delta_t > 0.0 && delta_t <= 0.75 && (0.003..=0.40).contains(&distance)
    };
    let mut kept = Vec::new();
    for (i, row) in raw.iter().enumerate() {
        if (i > 0 && plausible(&raw[i - 1], row))
            || (i + 1 < raw.len() && plausible(row, &raw[i + 1]))
        {
            kept.push(row.clone());
        }
    }
    if kept.len() < 3 {
        return Err("Статичная/разорванная траектория не подходит для слежения за мячом".into());
    }
    let first = &kept[0];
    let last = kept.last().unwrap();
    if ((first.x - last.x).powi(2) + (first.y - last.y).powi(2)).sqrt() < 0.012
        && kept
            .iter()
            .all(|p| (p.x - first.x).abs() < 0.025 && (p.y - first.y).abs() < 0.025)
    {
        return Err("Ложная неподвижная точка мяча отклонена".into());
    }
    let mut smooth = Vec::new();
    let mut last_x = kept[0].x;
    let mut last_y = kept[0].y;
    for point in kept {
        if smooth
            .last()
            .is_some_and(|prev: &CameraPoint| point.time_sec - prev.time_sec < 0.14)
        {
            continue;
        }
        let delta_t = smooth.last().map_or(0.15, |prev: &CameraPoint| {
            (point.time_sec - prev.time_sec).max(0.01)
        });
        // Respect maximum crop motion and prevent camera shake.
        let alpha = (delta_t * 2.5).clamp(0.15, 0.65);
        last_x += alpha * (point.x - last_x);
        last_y += alpha * (point.y - last_y);
        smooth.push(CameraPoint {
            time_sec: point.time_sec,
            x: last_x,
            y: last_y,
        });
    }
    if smooth.len() > 100 {
        let stride = (smooth.len() + 99) / 100;
        let last = smooth.last().cloned();
        smooth = smooth.into_iter().step_by(stride).collect();
        if let Some(last) = last {
            if smooth.last().is_none_or(|p| p.time_sec != last.time_sec) {
                smooth.push(last);
            }
        }
    }
    Ok(smooth)
}
