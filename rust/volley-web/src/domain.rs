use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
};
const MAX_INPUT: usize = 8 * 1024 * 1024;
thread_local! {
    static INPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static HASHES: RefCell<BTreeMap<u32,Sha256>> = const { RefCell::new(BTreeMap::new()) };
}
#[no_mangle]
pub extern "C" fn volley_data_reserve(len: u32) -> *mut u8 {
    if len as usize > MAX_INPUT {
        return std::ptr::null_mut();
    }
    INPUT.with(|input| {
        let mut input = input.borrow_mut();
        input.resize(len as usize, 0);
        input.as_mut_ptr()
    })
}
#[no_mangle]
pub extern "C" fn volley_data_output_ptr() -> *const u8 {
    OUTPUT.with(|o| o.borrow().as_ptr())
}
#[no_mangle]
pub extern "C" fn volley_data_output_len() -> u32 {
    OUTPUT.with(|o| o.borrow().len() as u32)
}
fn id(value: &Value) -> bool {
    value.as_str().is_some_and(volley_core::contracts::job_id)
}

fn rows<'a>(overview: &'a Value, stage: &str) -> Vec<&'a Value> {
    overview["stages"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|r| r["key"] == stage)
        .and_then(|r| r["items"].as_array())
        .into_iter()
        .flatten()
        .filter(|r| id(&r["id"]))
        .collect()
}
fn add(
    choices: &mut Vec<Value>,
    by: &mut BTreeMap<String, Vec<Value>>,
    video: &str,
    kind: &str,
    row: &Value,
) -> String {
    let value = json!([video, kind, row["id"]]).to_string();
    if !by.contains_key(&value) {
        let digest = Sha256::digest(value.as_bytes());
        let first = [
            "Быстрый",
            "Тихий",
            "Яркий",
            "Смелый",
            "Лёгкий",
            "Лунный",
            "Вольный",
            "Верный",
            "Зоркий",
            "Горный",
            "Синий",
            "Ранний",
            "Тёплый",
            "Свежий",
            "Дальний",
            "Светлый",
        ];
        let second = [
            "сокол",
            "ветер",
            "мяч",
            "тигр",
            "луч",
            "кедр",
            "орёл",
            "парус",
            "берег",
            "ястреб",
            "ручей",
            "маяк",
            "океан",
            "гром",
            "атлас",
            "космос",
        ];
        let display_name = format!(
            "{} {}",
            first[digest[0] as usize % first.len()],
            second[digest[1] as usize % second.len()]
        );
        choices.push(json!({"value":value,"video":video,"kind":kind,"id":row["id"],"model":row["model"],"created_at":row["created_at"],"display_name":display_name}));
        by.insert(value.clone(), Vec::new());
    }
    value
}
fn chain(input: &Value) -> Value {
    let mut choices = Vec::new();
    let mut by = BTreeMap::<String, Vec<Value>>::new();
    let mut run_to = BTreeMap::new();
    for video in input["videos"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        let overview = &input["overviews"][video];
        let fusion = rows(overview, "fusion");
        let gpu = rows(overview, "gpu");
        let replay: BTreeMap<_, _> = rows(overview, "tracking")
            .into_iter()
            .filter(|r| id(&r["source_fusion_id"]))
            .map(|r| {
                (
                    r["id"].as_str().unwrap_or_default(),
                    r["source_fusion_id"].as_str().unwrap_or_default(),
                )
            })
            .collect();
        let mut fusion_keys = BTreeMap::new();
        let mut fused_gpu = BTreeSet::new();
        for row in fusion {
            fusion_keys.insert(
                row["id"].as_str().unwrap_or_default(),
                add(&mut choices, &mut by, video, "fusion", row),
            );
            for key in ["person_job_id", "vball_job_id"] {
                if id(&row[key]) {
                    fused_gpu.insert(row[key].as_str().unwrap_or_default());
                }
            }
        }
        for row in gpu {
            if !fused_gpu.contains(row["id"].as_str().unwrap_or_default()) {
                add(&mut choices, &mut by, video, "gpu", row);
            }
        }
        let mut seen = BTreeSet::new();
        for run in input["runs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|r| r["video"] == video && id(&r["rally_run_id"]))
        {
            let run_id = run["rally_run_id"].as_str().unwrap_or_default();
            if !seen.insert(run_id) {
                continue;
            }
            let source = run["source_replay_id"].as_str().unwrap_or_default();
            let key = replay
                .get(source)
                .and_then(|s| fusion_keys.get(s))
                .cloned()
                .unwrap_or_else(|| {
                    add(
                        &mut choices,
                        &mut by,
                        video,
                        "unlinked",
                        &json!({"id":if source.is_empty(){run_id}else{source}}),
                    )
                });
            if let Some(items) = by.get_mut(&key) {
                items.push(run.clone());
            }
            run_to.insert(run_id.to_owned(), key);
        }
    }
    let sources = choices.clone();
    let mut claimed_sources = BTreeSet::new();
    let mut claimed_runs = BTreeSet::new();
    let mut launches: Vec<_> = input["launches"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|launch| id(&launch["id"]))
        .collect();
    launches.sort_by_key(|launch| launch["created_at"].as_u64().unwrap_or_default());
    for launch in launches {
        let mut members = Vec::new();
        let mut rally_runs = Vec::new();
        let mut seen_videos = BTreeSet::new();
        for member in launch["members"].as_array().into_iter().flatten() {
            let Some(video) = member["video"].as_str() else {
                continue;
            };
            if !seen_videos.insert(video) {
                continue;
            }
            let mut raw = None;
            for job in member["gpu_job_ids"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|job| id(job))
            {
                let key = json!([video, "gpu", job]).to_string();
                if let Some(source) = sources.iter().find(|row| row["value"] == key) {
                    claimed_sources.insert(key);
                    raw = Some(source);
                }
            }
            let physical = json!([video, "fusion", member["analysis_id"]]).to_string();
            let source = id(&member["analysis_id"])
                .then(|| sources.iter().find(|row| row["value"] == physical))
                .flatten();
            let Some(source) = source else {
                if let Some(raw) = raw {
                    members.push(raw.clone());
                }
                continue;
            };
            claimed_sources.insert(physical.clone());
            members.push(source.clone());
            if let Some(run) = by.get(&physical).into_iter().flatten().find(|run| {
                id(&member["postprocessing_id"])
                    && run["rally_run_id"] == member["postprocessing_id"]
            }) {
                claimed_runs.insert(run["rally_run_id"].as_str().unwrap_or_default().to_owned());
                rally_runs.push(run.clone());
            }
        }
        if members.is_empty() || launch["hidden"] == true {
            continue;
        }
        let key = add(
            &mut choices,
            &mut by,
            "",
            "launch",
            &json!({"id":launch["id"],"created_at":launch["created_at"]}),
        );
        if let Some(choice) = choices.iter_mut().find(|choice| choice["value"] == key) {
            choice["video"] = members[0]["video"].clone();
            choice["members"] = json!(members);
            if let Some(name) = launch["display_name"]
                .as_str()
                .filter(|name| !name.is_empty())
            {
                choice["display_name"] = json!(name);
            }
        }
        for run in &rally_runs {
            if let Some(run_id) = run["rally_run_id"].as_str() {
                run_to.insert(run_id.to_owned(), key.clone());
            }
        }
        by.insert(key, rally_runs);
    }
    for source in &claimed_sources {
        let Some(rows) = by.get_mut(source) else {
            continue;
        };
        rows.retain(|row| !claimed_runs.contains(row["rally_run_id"].as_str().unwrap_or_default()));
        if rows.is_empty() {
            by.remove(source);
        }
    }
    choices.retain(|choice| {
        let key = choice["value"].as_str().unwrap_or_default();
        !claimed_sources.contains(key) || by.contains_key(key)
    });
    run_to.retain(|id, key| {
        by.get(key)
            .is_some_and(|rows| rows.iter().any(|row| row["rally_run_id"] == id.as_str()))
    });
    json!({"choices":choices,"byAnalysis":by,"runToAnalysis":run_to})
}

fn montage(input: &Value) -> Value {
    let mut rows: Vec<_> = input.as_array().into_iter().flatten().collect();
    rows.sort_by(|a, b| {
        let start = |v: &Value| {
            v["start"]
                .as_f64()
                .filter(|v| v.is_finite())
                .unwrap_or(f64::INFINITY)
        };
        start(a).total_cmp(&start(b))
    });
    json!(rows.iter().map(|r| &r["id"]).collect::<Vec<_>>())
}

fn creative_catalog(input: &Value) -> Value {
    let video = input["video"].as_str().unwrap_or_default();
    let total = input["total_frames"].as_u64().unwrap_or_default();
    let offset = input["analysis_offset_sec"].as_f64().unwrap_or_default();
    let clip = |start: u64, end: u64| {
        if video.is_empty() || start > end || end >= total {
            return None;
        }
        let mut clip = json!({"video":video,"start_frame":start,"end_frame":end});
        if offset != 0.0 {
            clip["analysis_offset_sec"] = json!(offset);
        }
        Some(clip)
    };
    let marked: BTreeSet<_> = input["candidate_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let mut rows = Vec::new();
    let mut players = BTreeMap::new();
    for rally in input["rallies"].as_array().into_iter().flatten() {
        let Some(start) = rally["start_frame"].as_u64() else {
            continue;
        };
        let Some(end) = rally["end_frame"].as_u64() else {
            continue;
        };
        if let Some(clip) = clip(start, end) {
            let favorite = rally["candidate_id"]
                .as_str()
                .is_some_and(|id| marked.contains(id));
            rows.push(json!({"kind":"rally","clip":clip,"favorite":favorite,"label":"Розыгрыш"}));
        }
    }
    for event in input["touch_annotations"].as_array().into_iter().flatten() {
        if event["touch_attributed"].as_bool() != Some(true) {
            continue;
        }
        let Some(frame) = event["frame_index"].as_u64().filter(|&frame| frame < total) else {
            continue;
        };
        let Some(clip) = clip(
            frame.saturating_sub(30),
            frame.saturating_add(30).min(total - 1),
        ) else {
            continue;
        };
        let reviewed = &event["reviewed_action"];
        let action = (reviewed["action_verified"].as_bool() == Some(true))
            .then(|| reviewed["action_kind"].as_str())
            .flatten()
            .filter(|s| !s.is_empty())
            .or_else(|| event["kind"].as_str().filter(|s| !s.is_empty()))
            .unwrap_or("unknown_touch");
        let player = event["player_id"].as_str().unwrap_or_default();
        let name = event["player_name"].as_str().unwrap_or_default();
        if !player.is_empty() && !name.is_empty() {
            players.insert(player, name);
        }
        let label = match action {
            "attack" => "Атака",
            "block" => "Блок",
            "serve" => "Подача",
            "dig" => "Приём",
            "overhead_pass" => "Верхняя передача",
            "low_touch" => "Касание снизу",
            _ => "Касание",
        };
        let label = if name.is_empty() {
            label.to_owned()
        } else {
            format!("{label} · {name}")
        };
        rows.push(
            json!({"kind":"touch","clip":clip,"action":action,"player":player,"label":label}),
        );
    }
    json!({"rows":rows,"players":players.into_iter().collect::<Vec<_>>()})
}

fn creative_filter(input: &Value) -> Value {
    let action = input["action"].as_str().unwrap_or_default();
    let player = input["player"].as_str().unwrap_or_default();
    let kind = input["kind"].as_str().unwrap_or("rallies");
    let indices: Vec<_> = input["rows"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, row)| {
            let matches = match kind {
                "touches" => {
                    row["kind"] == "touch"
                        && (action.is_empty() || row["action"] == action)
                        && (player.is_empty() || row["player"] == player)
                }
                "rallies" => row["kind"] == "rally",
                "favorites" => row["kind"] == "rally" && row["favorite"].as_bool() == Some(true),
                _ => false,
            };
            matches.then_some(index)
        })
        .collect();
    json!(indices)
}

#[no_mangle]
pub extern "C" fn volley_domain_call(operation: u32) -> i32 {
    OUTPUT.with(|out| out.borrow_mut().clear());
    let input = INPUT.with(|i| serde_json::from_slice::<Value>(&i.borrow()));
    let Ok(input) = input else {
        return 0;
    };
    let value = match operation {
        1 => chain(&input),
        2 => montage(&input),
        3 => creative_catalog(&input),
        4 => creative_filter(&input),
        5 => crate::preview_timeline::next_rally(&input),
        6 => crate::analysis_stages::call(&input),
        _ => return 0,
    };
    match serde_json::to_vec(&value) {
        Ok(bytes) => {
            OUTPUT.with(|o| *o.borrow_mut() = bytes);
            1
        }
        Err(_) => 0,
    }
}
#[no_mangle]
pub extern "C" fn volley_hash_begin() -> u32 {
    HASHES.with(|h| {
        let mut h = h.borrow_mut();
        if h.len() >= 32 {
            return 0;
        }
        let Some(id) = (1..=33).find(|id| !h.contains_key(id)) else {
            return 0;
        };
        h.insert(id, Sha256::new());
        id
    })
}
#[no_mangle]
pub extern "C" fn volley_hash_update(id: u32) -> i32 {
    HASHES.with(|h| {
        let mut h = h.borrow_mut();
        let Some(hash) = h.get_mut(&id) else {
            return 0;
        };
        INPUT.with(|i| hash.update(&*i.borrow()));
        1
    })
}
#[no_mangle]
pub extern "C" fn volley_hash_finish(id: u32) -> i32 {
    OUTPUT.with(|o| o.borrow_mut().clear());
    HASHES.with(|h| {
        let Some(hash) = h.borrow_mut().remove(&id) else {
            return 0;
        };
        OUTPUT.with(|o| *o.borrow_mut() = format!("{:x}", hash.finalize()).into_bytes());
        1
    })
}
#[no_mangle]
pub extern "C" fn volley_hash_abort(id: u32) {
    HASHES.with(|h| h.borrow_mut().remove(&id));
}
#[no_mangle]
pub extern "C" fn volley_evidence_fps(metadata: f64) -> f64 {
    if metadata.is_finite() && metadata > 0.0 && metadata <= 1000.0 {
        metadata
    } else {
        30.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_choice_has_stable_two_word_display_name_and_keeps_full_id() {
        let input = json!({"videos":["match.mp4"],"overviews":{"match.mp4":{"stages":[
            {"key":"fusion","items":[{"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","created_at":1700000000}]}
        ]}},"runs":[]});
        let result = chain(&input);
        let choice = &result["choices"][0];
        let name = choice["display_name"].as_str().unwrap();
        assert_eq!(name.split_whitespace().count(), 2);
        assert_eq!(choice["id"], "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert_eq!(
            choice["value"],
            "[\"match.mp4\",\"fusion\",\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"]"
        );
        assert_eq!(
            choice["display_name"],
            chain(&input)["choices"][0]["display_name"]
        );
    }

    #[test]
    fn project_launch_groups_rallies_across_videos_without_mixing_launches() {
        let first = "11111111111111111111111111111111";
        let second = "22222222222222222222222222222222";
        let a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let c = "cccccccccccccccccccccccccccccccc";
        let ar = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1";
        let br = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb1";
        let cr = "ccccccccccccccccccccccccccccccc1";
        let at = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa2";
        let bt = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2";
        let ct = "ccccccccccccccccccccccccccccccc2";
        let input = json!({
            "videos":["a.mp4","b.mp4"],
            "overviews":{
                "a.mp4":{"stages":[{"key":"fusion","items":[{"id":a},{"id":c}]},
                    {"key":"tracking","items":[{"id":ar,"source_fusion_id":a},{"id":cr,"source_fusion_id":c}]}]},
                "b.mp4":{"stages":[{"key":"fusion","items":[{"id":b}]},
                    {"key":"tracking","items":[{"id":br,"source_fusion_id":b}]}]}
            },
            "runs":[{"video":"a.mp4","rally_run_id":at,"source_replay_id":ar},
                    {"video":"b.mp4","rally_run_id":bt,"source_replay_id":br},
                    {"video":"a.mp4","rally_run_id":ct,"source_replay_id":cr}],
            "launches":[
                {"id":first,"created_at":100,"members":[
                    {"video":"a.mp4","analysis_id":a,"postprocessing_id":at},
                    {"video":"b.mp4","analysis_id":b,"postprocessing_id":bt}]},
                {"id":second,"created_at":200,"members":[
                    {"video":"a.mp4","analysis_id":c,"postprocessing_id":ct},
                    {"video":"b.mp4","analysis_id":b,"postprocessing_id":bt}]}
            ]
        });
        let result = chain(&input);
        let choices = result["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 2);
        let first_key = json!(["", "launch", first]).to_string();
        let second_key = json!(["", "launch", second]).to_string();
        assert_eq!(result["byAnalysis"][&first_key][0]["rally_run_id"], at);
        assert_eq!(result["byAnalysis"][&first_key][1]["rally_run_id"], bt);
        assert_eq!(result["byAnalysis"][&second_key][0]["rally_run_id"], ct);
        assert_eq!(result["byAnalysis"][&second_key][1]["rally_run_id"], bt);
        let mut hidden = input;
        hidden["launches"][0]["hidden"] = json!(true);
        let result = chain(&hidden);
        assert_eq!(result["choices"].as_array().unwrap().len(), 1);
        assert!(result["byAnalysis"][&first_key].is_null());
        assert_eq!(result["byAnalysis"][&second_key][1]["rally_run_id"], bt);
    }

    #[test]
    fn fused_person_and_ball_jobs_do_not_appear_as_separate_analyses() {
        let ball = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let person = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let fusion = "cccccccccccccccccccccccccccccccc";
        let orphan = "dddddddddddddddddddddddddddddddd";
        // Match the metadata returned by studio_analysis::stage_summary.
        let input = json!({"videos":["match.mp4"],"overviews":{"match.mp4":{"stages":[
            {"key":"fusion","items":[{"id":fusion,"person_job_id":person,"vball_job_id":ball}]},
            {"key":"gpu","items":[{"id":ball},{"id":person},{"id":orphan}]}
        ]}},"runs":[]});
        let result = chain(&input);
        let choices = result["choices"].as_array().unwrap();
        assert_eq!(
            choices.len(),
            2,
            "fused model jobs must not be separate GPU analyses"
        );
        assert_eq!(choices[0]["id"], fusion);
        assert_eq!(choices[0]["kind"], "fusion");
        assert_eq!(choices[1]["id"], orphan);
        assert_eq!(choices[1]["kind"], "gpu");
    }
}
