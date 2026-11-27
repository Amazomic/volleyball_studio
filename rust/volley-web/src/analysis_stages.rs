use serde_json::{json, Value};

const BLOCK_STAGES: [&str; 3] = ["boxes", "touch_detector", "pose"];

fn stage_id(entry: &Value) -> &str {
    let name = entry
        .as_str()
        .or_else(|| entry["name"].as_str())
        .unwrap_or_default();
    if name.starts_with("Ожидание очереди") {
        return "";
    }
    let id = entry["id"].as_str().unwrap_or_default();
    match id {
        "ground_detector" => return "rallies",
        "touch_types" => return "touches",
        "video" | "ball" | "ball_stabilizer" | "rallies" | "touch_detector" | "players"
        | "boxes" | "pose" | "tracking" | "touches" | "publish" | "identity" | "fusion"
        | "cache" | "material" => return id,
        _ => {}
    }
    let has = |parts: &[&str]| parts.iter().any(|part| name.contains(part));
    if has(&["Проверка исходного видео", "Подготовка исходного видео"])
    {
        "video"
    } else if has(&[
        "ball_stabilizer",
        "Ball Stabilizer",
        "Интерполяция траектории мяча",
    ]) {
        "ball_stabilizer"
    } else if matches!(
        name,
        "Ground Detector" | "ground_detector" | "Rally Detector"
    ) {
        "rallies"
    } else if name == "Touch Detector" {
        "touch_detector"
    } else if matches!(name, "Touch Attribution" | "Pose Geometry") {
        "touches"
    } else if name.contains("Поиск розыгрышей") {
        "rallies"
    } else if name.contains("Связывание рамок игроков") {
        "boxes"
    } else if name
        .split_once("GPU")
        .is_some_and(|(_, rest)| rest.contains("VBallNet") || rest.contains("VballNet"))
    {
        "ball"
    } else if name.contains("RTMPose") && !name.contains("RF-DETR") {
        "pose"
    } else if name.contains("Объединение") {
        "fusion"
    } else if has(&["RF-DETR", "RTMDet", "игровые интервалы", "общий поток"])
    {
        "players"
    } else if name.contains("GPU-анализ завершён") {
        "fusion"
    } else if name.contains("Track ID") {
        "tracking"
    } else if has(&["касания в первичных", "физические касания"])
    {
        "touches"
    } else if name.contains("Публикация розыгрышей") {
        "publish"
    } else if name.contains("автоматическая привязка") {
        "identity"
    } else if name.contains("Монтажная временная разметка") {
        "material"
    } else if has(&[
        "сохранённые версии",
        "Сохранённый GPU-анализ",
        "сохранённый GPU-анализ",
    ]) {
        "cache"
    } else {
        ""
    }
}

fn number(value: &Value) -> Option<f64> {
    value.as_f64().filter(|n| n.is_finite())
}

fn set_field(row: &mut Value, key: &str, value: Option<&Value>) {
    if let Some(fields) = row.as_object_mut() {
        if let Some(value) = value {
            fields.insert(key.to_owned(), value.clone());
        } else {
            fields.remove(key);
        }
    }
}

fn combined(stages: &Value) -> Vec<Value> {
    let mut result: Vec<Value> = Vec::new();
    for item in stages.as_array().into_iter().flatten() {
        if !item.is_object() {
            result.push(item.clone());
            continue;
        }
        let Some(prev) = result.last_mut() else {
            result.push(item.clone());
            continue;
        };
        let ground = prev["id"] == "ground_detector"
            || matches!(
                prev["name"].as_str(),
                Some("Ground Detector" | "ground_detector")
            );
        let pose_geometry =
            |row: &Value| row["id"] == "touch_types" || row["name"] == "Pose Geometry";
        let merge_ground = ground && stage_id(item) == "rallies" && !prev["finished_ms"].is_null();
        let merge_touch = stage_id(prev) == "touches"
            && stage_id(item) == "touches"
            && (pose_geometry(prev) || pose_geometry(item));
        if !merge_ground && !merge_touch {
            result.push(item.clone());
            continue;
        }
        let mut row = item.clone();
        let start = prev
            .get("started_ms")
            .filter(|v| !v.is_null())
            .or_else(|| item.get("started_ms"));
        let sum = number(&prev["duration_ms"])
            .zip(number(&item["duration_ms"]))
            .map(|(a, b)| a + b);
        let duration = if merge_touch {
            start
                .and_then(number)
                .zip(number(&item["finished_ms"]))
                .map(|(a, b)| (b - a).max(0.0))
                .or(sum)
        } else {
            sum
        };
        row["id"] = json!(if merge_ground { "rallies" } else { "touches" });
        if merge_touch {
            row["name"] = json!("Touch Attribution");
        }
        set_field(&mut row, "started_ms", start);
        if let Some(duration) = duration {
            row["duration_ms"] = json!(duration);
        }
        *prev = row;
    }
    result
}

fn has_block(blocks: &Value, id: &str) -> bool {
    BLOCK_STAGES.contains(&id) && number(&blocks[id]["rallies_total"]).is_some_and(|n| n != 0.0)
}

fn touch_duration(entry: &Value, now: Option<f64>) -> Option<f64> {
    let base = number(&entry["active_duration_ms"])?;
    let delta = if entry["state"] == "running" {
        number(&entry["block_started_ms"])
            .zip(now)
            .map_or(0.0, |(start, now)| (now - start).max(0.0))
    } else {
        0.0
    };
    Some(base + delta)
}

fn stage_entry(id: &str, entry: &Value, blocks: &Value, now: Option<f64>) -> Value {
    if id == "players" && entry.is_object() {
        if let Some(finish) = number(&blocks["detector_finished_ms"]) {
            let mut row = entry.clone();
            row["state"] = json!("complete");
            row["finished_ms"] = json!(finish);
            row["duration_ms"] =
                json!(number(&entry["started_ms"]).map(|start| (finish - start).max(0.0)));
            return row;
        }
    }
    if !has_block(blocks, id) {
        return entry.clone();
    }
    let mut row = blocks[id].clone();
    let duration = if id == "touch_detector" {
        touch_duration(&row, now)
    } else {
        number(&row["started_ms"])
            .zip(number(&row["finished_ms"]))
            .map(|(a, b)| (b - a).max(0.0))
    };
    row["duration_ms"] = json!(duration);
    row
}

fn stage_state(id: &str, entry: &Value, blocks: &Value) -> Value {
    if id == "players" && number(&blocks["detector_finished_ms"]).is_some() {
        return json!("complete");
    }
    if has_block(blocks, id) {
        return json!(entry["state"].as_str().filter(|s| !s.is_empty()));
    }
    Value::Null
}

fn history(stages: &Value, blocks: &Value, now: Option<f64>) -> Value {
    if blocks.is_null() {
        return stages.clone();
    }
    let mut entries: Vec<_> = stages
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let id = entry["id"].as_str().unwrap_or_default();
            if matches!(id, "boxes" | "pose") || id == "touch_detector" && has_block(blocks, id) {
                return None;
            }
            Some(stage_entry(id, entry, blocks, now))
        })
        .collect();
    for id in BLOCK_STAGES {
        let mut entry = stage_entry(id, &Value::Null, blocks, now);
        if !entry["started_ms"].is_null() {
            entry["id"] = json!(id);
            entry["name"] = json!(id);
            entries.push(entry);
        }
    }
    entries.sort_by(|a, b| {
        number(&a["started_ms"])
            .unwrap_or_default()
            .total_cmp(&number(&b["started_ms"]).unwrap_or_default())
    });
    json!(entries)
}

fn percent(entry: &Value, running: bool, done: bool) -> Option<f64> {
    if done {
        return Some(100.0);
    }
    if !running {
        return Some(0.0);
    }
    number(&entry["progress_percent"])
        .or_else(|| {
            let processed = number(&entry["frames_processed"])?;
            let total = number(&entry["frames_total"]).filter(|n| *n > 0.0)?;
            Some(processed / total * 100.0)
        })
        .map(|n| n.clamp(0.0, 100.0))
}

fn link(input: &Value) -> &'static str {
    let ready = input["poseReady"] == true;
    let disabled = input["poseEnabled"] == false;
    match input["condition"].as_str() {
        Some("no_pose") if ready => "disabled",
        Some("no_pose") if disabled => "available",
        Some("no_pose") if input["hasRun"] == true => "disabled",
        Some("no_pose") => "optional",
        Some("pose") if ready => "available",
        Some("pose") if disabled => "disabled",
        Some("pose") => "optional",
        _ => "available",
    }
}

pub(crate) fn call(input: &Value) -> Value {
    let id = input["id"].as_str().unwrap_or_default();
    let entry = &input["entry"];
    let blocks = &input["blocks"];
    let now = number(&input["now"]);
    match input["kind"].as_str() {
        Some("id") => json!(stage_id(entry)),
        Some("combine") => json!(combined(&input["stages"])),
        Some("entry") => stage_entry(id, entry, blocks, now),
        Some("state") => stage_state(id, entry, blocks),
        Some("history") => history(&input["stages"], blocks, now),
        Some("touch_duration") => json!(touch_duration(entry, now)),
        Some("percent") => json!(percent(
            entry,
            input["running"] == true,
            input["done"] == true
        )),
        Some("link") => json!(link(input)),
        _ => Value::Null,
    }
}
