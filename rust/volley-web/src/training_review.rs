use serde_json::{json, Value};

const MAX_SAFE_FRAME: u64 = 9_007_199_254_740_991;

fn frame(value: &Value) -> Option<u64> {
    value.as_u64().filter(|n| *n <= MAX_SAFE_FRAME)
}

fn interval(row: &Value) -> Option<(u64, u64)> {
    let start = frame(&row["start"])?;
    let end = frame(&row["end"])?;
    (start <= end).then_some((start, end))
}

struct Group {
    start: u64,
    end: u64,
    indices: Vec<usize>,
}

fn merge(input: &Value) -> Value {
    let mut items: Vec<_> = input["items"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(i, row)| interval(row).map(|(start, end)| (start, end, i)))
        .collect();
    items.sort_by_key(|&(start, end, _)| (start, end));
    let stops: Vec<_> = input["stops"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|stop| {
            let start = stop.get(0)?.as_f64()?;
            let end = stop.get(1)?.as_f64()?;
            Some((start, end))
        })
        .collect();
    let gap = input["gapFrames"].as_f64().unwrap_or_default();
    let mut groups: Vec<Group> = Vec::new();
    for (start, end, index) in items {
        if let Some(previous) = groups.last_mut() {
            let hold = stops
                .iter()
                .any(|&(a, b)| a < start as f64 && b > previous.end as f64);
            let separated = start.saturating_sub(previous.end.saturating_add(1));
            if !hold && (start <= previous.end + 1 || (separated as f64) < gap) {
                previous.end = previous.end.max(end);
                previous.indices.push(index);
                continue;
            }
        }
        groups.push(Group {
            start,
            end,
            indices: vec![index],
        });
    }
    json!(groups
        .into_iter()
        .map(|g| json!({"start":g.start,"end":g.end,"indices":g.indices}))
        .collect::<Vec<_>>())
}

fn seeds(input: &Value) -> Value {
    if input["source"] != "calculated_ball_trajectory_rallies" {
        return json!([]);
    }
    let Some(total) = frame(&input["totalFrames"]) else {
        return json!([]);
    };
    json!(input["intervals"]
        .as_array()
        .into_iter()
        .flatten()
        .take(1024)
        .filter_map(interval)
        .filter(|&(start, end)| start < end && end < total)
        .map(|(start, end)| json!({"start":start,"end":end}))
        .collect::<Vec<_>>())
}

pub(crate) fn call(input: &Value) -> Value {
    match input["kind"].as_str() {
        Some("merge") => merge(input),
        Some("seeds") => seeds(input),
        _ => Value::Null,
    }
}
