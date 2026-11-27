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

pub(crate) fn call(input: &Value) -> Value {
    match input["kind"].as_str() {
        Some("candidates") => candidates(input),
        Some("order") => order(input),
        _ => Value::Null,
    }
}
