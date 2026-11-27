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

fn favorite_id(value: &Value) -> Option<&str> {
    value.as_str().filter(|id| {
        id.len() == 24
            && id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

fn draft_order(value: &Value) -> Vec<&str> {
    let Some(rows) = value.as_array().filter(|rows| rows.len() <= MAX_CLIPS) else {
        return Vec::new();
    };
    let mut seen = std::collections::BTreeSet::new();
    let mut order = Vec::new();
    for row in rows {
        let Some(id) = favorite_id(row) else {
            return Vec::new();
        };
        if !seen.insert(id) {
            return Vec::new();
        }
        order.push(id);
    }
    order
}

fn draft_read(input: &Value) -> Value {
    let Some(raw) = input["raw"].as_str().filter(|raw| raw.len() <= 2400) else {
        return json!([]);
    };
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        return json!([]);
    };
    json!(draft_order(&value))
}

fn draft_accept(input: &Value) -> Value {
    let mut seen = std::collections::BTreeSet::new();
    let indices: Vec<_> = input["items"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(i, item)| {
            let id = favorite_id(&item["id"])?;
            let start = item["start"].as_f64()?;
            let end = item["end"].as_f64()?;
            if item["video"].as_str().is_none()
                || item["video"] != input["video"]
                || start < 0.0
                || end <= start
                || !seen.insert(id)
            {
                return None;
            }
            Some(i)
        })
        .take(MAX_CLIPS)
        .collect();
    let order: Vec<_> = draft_order(&input["order"])
        .into_iter()
        .filter(|id| seen.contains(id))
        .collect();
    json!({"indices":indices,"order":order})
}

fn draft_chosen(input: &Value) -> Value {
    let items = input["items"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut by_id = std::collections::BTreeMap::new();
    for (i, item) in items.iter().enumerate() {
        if let Some(id) = favorite_id(&item["id"]) {
            by_id.entry(id).or_insert(i);
        }
    }
    let indices: Vec<_> = draft_order(&input["order"])
        .into_iter()
        .filter_map(|id| by_id.get(id).copied())
        .collect();
    let duration: f64 = indices
        .iter()
        .map(|&i| {
            (items[i]["end"].as_f64().unwrap_or_default()
                - items[i]["start"].as_f64().unwrap_or_default())
            .max(0.0)
        })
        .sum();
    json!({"indices":indices,"duration":duration})
}

fn draft_edit(input: &Value) -> Value {
    let mut order = draft_order(&input["order"]);
    let Some(id) = favorite_id(&input["id"]) else {
        return json!(order);
    };
    let position = order.iter().position(|&item| item == id);
    match (input["action"].as_str(), position) {
        (Some("add"), None) if order.len() < MAX_CLIPS => order.push(id),
        (Some("remove"), Some(i)) => {
            order.remove(i);
        }
        (Some("move"), Some(i)) => {
            if let Some(target) = input["index"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n < order.len())
            {
                order.remove(i);
                order.insert(target, id);
            }
        }
        _ => {}
    }
    json!(order)
}

pub(crate) fn call(input: &Value) -> Value {
    match input["kind"].as_str() {
        Some("bounds") => bounds(input).unwrap_or(Value::Null),
        Some("summary") => summary(input),
        Some("draft_read") => draft_read(input),
        Some("draft_accept") => draft_accept(input),
        Some("draft_chosen") => draft_chosen(input),
        Some("draft_edit") => draft_edit(input),
        _ => Value::Null,
    }
}
