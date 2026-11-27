use serde_json::{json, Value};

const FPS: f64 = 30.0;
const MAX_CLIPS: usize = 80;
const MAX_FRAMES: u64 = 18_000;

fn known_video(input: &Value, video: &Value) -> bool {
    video.as_str().is_some_and(|name| !name.is_empty())
        && input["videos"]
            .as_array()
            .is_some_and(|videos| videos.contains(video))
}

fn bounds(input: &Value) -> Option<Value> {
    let video = &input["video"];
    if !known_video(input, video) {
        return None;
    }
    let duration = input["duration"].as_f64()?;
    let start = input["start"].as_f64()?;
    let end = input["end"].as_f64()?;
    if duration <= 0.0
        || start < 0.0
        || end <= start
        || end > duration
        || duration * FPS > 9_007_199_254_740_991.0
    {
        return None;
    }
    let start_frame = (start * FPS).round() as u64;
    let end_frame = ((end * FPS).round().min((duration * FPS).floor()) as u64).checked_sub(1)?;
    (end_frame >= start_frame)
        .then(|| json!({"video":video,"start_frame":start_frame,"end_frame":end_frame}))
}

fn frames(clip: &Value) -> Option<u64> {
    let start = clip["start_frame"].as_u64()?;
    let end = clip["end_frame"].as_u64()?;
    end.checked_sub(start)?.checked_add(1)
}

fn summary(input: &Value) -> Value {
    let clips = input["clips"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    let total = clips
        .iter()
        .try_fold(0_u64, |sum, clip| sum.checked_add(frames(clip)?));
    let valid = clips.iter().all(|clip| known_video(input, &clip["video"]));
    let can_add = valid
        && clips.len() < MAX_CLIPS
        && known_video(input, &input["next"]["video"])
        && total
            .zip(frames(&input["next"]))
            .and_then(|(a, b)| a.checked_add(b))
            .is_some_and(|n| n <= MAX_FRAMES);
    let can_export =
        valid && clips.len() <= MAX_CLIPS && total.is_some_and(|n| (30..=MAX_FRAMES).contains(&n));
    json!({"frames":total.unwrap_or_default(),"canAdd":can_add,"canExport":can_export})
}

pub(crate) fn call(input: &Value) -> Value {
    match input["kind"].as_str() {
        Some("bounds") => bounds(input).unwrap_or(Value::Null),
        Some("summary") => summary(input),
        _ => Value::Null,
    }
}
