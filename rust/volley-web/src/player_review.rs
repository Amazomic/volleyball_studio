use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

struct Candidate<'a> {
    player_id: &'a str,
    name: &'a str,
    name_rank: u64,
    total: f64,
    supported: usize,
    best: f64,
    track: u64,
    reference: Option<&'a Value>,
    reference_kind: Option<&'a Value>,
}

impl Candidate<'_> {
    fn mean(&self) -> f64 {
        self.total / self.supported as f64
    }

    fn output(&self, group_size: usize) -> Value {
        let mut row = json!({
            "playerId": self.player_id, "name": self.name, "total": self.total,
            "supported": self.supported, "score": self.mean(), "trackId": self.track,
            "groupSize": group_size
        });
        if let Some(reference) = self.reference {
            row["reference"] = reference.clone();
        }
        if let Some(kind) = self.reference_kind {
            row["referenceKind"] = kind.clone();
        }
        row
    }
}

fn candidates(input: &Value) -> Value {
    let roster: BTreeMap<_, _> = input["roster"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|row| {
            Some((
                row["id"].as_str()?,
                (
                    row["name"].as_str()?,
                    row["rank"].as_u64().unwrap_or_default(),
                ),
            ))
        })
        .collect();
    let mut members = BTreeSet::new();
    let mut positions = BTreeMap::new();
    let mut ranked: Vec<Candidate<'_>> = Vec::new();
    for track in input["tracks"].as_array().into_iter().flatten() {
        let Some(id) = track["id"].as_u64() else {
            continue;
        };
        if !members.insert(id) {
            continue;
        }
        let mut matches: BTreeMap<&str, (f64, &Value, usize)> = BTreeMap::new();
        for (index, row) in track["matches"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
        {
            let Some(player) = row["player_id"]
                .as_str()
                .filter(|player| roster.contains_key(player))
            else {
                continue;
            };
            let Some(score) = row["cosine_similarity"]
                .as_f64()
                .filter(|score| *score > 0.0 && *score <= 1.0)
            else {
                continue;
            };
            let best = matches.entry(player).or_insert((score, row, index));
            if score > best.0 {
                *best = (score, row, best.2);
            }
        }
        let mut unique: Vec<_> = matches.into_iter().collect();
        unique.sort_by_key(|(_, (_, _, index))| *index);
        for (player, (score, row, _)) in unique {
            let index = *positions.entry(player).or_insert_with(|| {
                let (name, name_rank) = roster[player];
                ranked.push(Candidate {
                    player_id: player,
                    name,
                    name_rank,
                    total: 0.0,
                    supported: 0,
                    best: 0.0,
                    track: id,
                    reference: None,
                    reference_kind: None,
                });
                ranked.len() - 1
            });
            let current = &mut ranked[index];
            current.total += score;
            current.supported += 1;
            if score > current.best {
                current.best = score;
                current.track = id;
                current.reference = row.get("reference_track_id");
                current.reference_kind = row.get("reference_kind");
            }
        }
    }
    ranked.sort_by(|a, b| {
        b.supported
            .cmp(&a.supported)
            .then_with(|| b.mean().total_cmp(&a.mean()))
            .then_with(|| a.name_rank.cmp(&b.name_rank))
    });
    json!(ranked
        .iter()
        .map(|candidate| candidate.output(members.len()))
        .collect::<Vec<_>>())
}

fn order(input: &Value) -> Value {
    let mut rows: Vec<_> = input["cards"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(index, row)| {
            (
                index,
                row["score"].as_f64().unwrap_or(-1.0),
                row["id"].as_u64().unwrap_or(u64::MAX),
            )
        })
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
    json!(rows.iter().map(|row| row.0).collect::<Vec<_>>())
}

fn track_id(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_str()?.parse().ok())
        .filter(|id| *id > 0 && *id <= 9_007_199_254_740_991)
}

fn ids(value: &Value) -> Vec<u64> {
    let mut seen = BTreeSet::new();
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(track_id)
        .filter(|id| seen.insert(*id))
        .collect()
}

fn visible_ids(input: &Value) -> BTreeSet<u64> {
    let pending: BTreeSet<_> = ids(&input["pending"]).into_iter().collect();
    ids(&input["unassigned"])
        .into_iter()
        .filter(|id| !pending.contains(id))
        .collect()
}

fn visible(input: &Value) -> Value {
    let allowed = visible_ids(input);
    let source = input.get("ids").unwrap_or(&input["unassigned"]);
    json!(source
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(track_id)
        .filter(|id| allowed.contains(id))
        .collect::<Vec<_>>())
}

fn groups(input: &Value) -> Value {
    let allowed = visible_ids(input);
    let rows: Vec<_> = input["groups"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|group| {
            let mut row = group.as_object()?.clone();
            let original = group["ids"].as_array()?;
            let members: Vec<_> = original
                .iter()
                .filter_map(track_id)
                .filter(|id| allowed.contains(id))
                .collect();
            if members.len() < 2 {
                return None;
            }
            row.insert("ids".to_owned(), json!(members));
            row.insert("originalCount".to_owned(), json!(original.len()));
            Some(row)
        })
        .collect();
    json!(rows)
}

fn confirmed_ids(input: &Value, current: &str, updated: &str) -> Vec<u64> {
    let result = &input["result"][updated];
    ids(if result.is_array() {
        result
    } else {
        &input[current]
    })
}

fn edit(input: &Value) -> Value {
    let changed: BTreeSet<_> = ids(&input["ids"]).into_iter().collect();
    let owner = &input["owner"];
    let owner_name = input["ownerName"].as_str().unwrap_or_default();
    let action = input["action"].as_str().unwrap_or_default();
    let unassigned = confirmed_ids(input, "unassigned", "unassigned_track_ids");
    let excluded = confirmed_ids(input, "excluded", "excluded_track_ids");
    let comparison = confirmed_ids(input, "comparison", "comparison_track_ids");
    let view_player = input["viewPlayer"].as_str().filter(|id| !id.is_empty());
    let view_conflicts = input["viewConflicts"] == true;
    let visible = if input["viewExcluded"] == true {
        &excluded
    } else {
        &unassigned
    };
    let allowed: BTreeSet<_> = visible.iter().copied().collect();
    let mut rows = Vec::new();
    for row in input["rows"].as_array().into_iter().flatten() {
        let Some(id) = track_id(&row["track_id"]) else {
            continue;
        };
        let Some(mut fields) = row.as_object().cloned() else {
            continue;
        };
        if changed.contains(&id) {
            match action {
                "assign" => {
                    fields.insert("assigned_player_id".into(), owner.clone());
                    fields.insert("assigned_player_name".into(), json!(owner_name));
                    fields.insert("identity_source".into(), json!("manual"));
                    fields.insert("identity_conflict".into(), json!(false));
                }
                "unassign" => {
                    fields.insert("assigned_player_id".into(), Value::Null);
                    fields.insert("assigned_player_name".into(), Value::Null);
                    fields.insert("identity_source".into(), json!("none"));
                }
                "exclude" | "restore" => {
                    fields.insert("excluded".into(), json!(action == "exclude"));
                }
                _ => {}
            }
        }
        let keep = match view_player {
            Some(player) => {
                fields.get("assigned_player_id").and_then(Value::as_str) == Some(player)
            }
            None => view_conflicts || allowed.contains(&id),
        };
        if keep {
            rows.push(Value::Object(fields));
        }
    }
    let mut total = input["filteredTotal"].as_u64().unwrap_or_default();
    let mut offset = input["offset"].as_u64().unwrap_or_default();
    if view_player.is_none() && !view_conflicts {
        total = visible.len() as u64;
        let limit = input["limit"].as_u64().filter(|n| *n > 0).unwrap_or(24);
        if total == 0 {
            offset = 0;
        } else if offset >= total {
            offset = (total - 1) / limit * limit;
        }
    }
    let mut result = json!({"rows":rows,"unassigned":unassigned,"excluded":excluded,"comparison":comparison,
        "filteredTotal":total,"offset":offset});
    if let Some(mut reasons) = input["reasons"].as_object().cloned() {
        for id in changed {
            reasons.remove(&id.to_string());
        }
        result["reasons"] = json!(reasons);
    }
    result
}

pub(crate) fn call(input: &Value) -> Value {
    match input["kind"].as_str() {
        Some("candidates") => candidates(input),
        Some("order") => order(input),
        Some("visible") => visible(input),
        Some("groups") => groups(input),
        Some("edit") => edit(input),
        _ => Value::Null,
    }
}
