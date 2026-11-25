use super::{for_each_verified_scoped_frame, overlay_players};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};
use volley_core::court_geometry::{CourtSide, CourtTransform, TeamSideVotes};

type Sides = BTreeMap<u64, CourtSide>;
type CacheKey = (PathBuf, String, String, (u64, u64));
type Cache = VecDeque<(CacheKey, Arc<Sides>)>;
static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();

fn rally_sides(
    source: (&Path, &str, u64),
    interval: (u64, u64),
    court: &CourtTransform,
    revision: &str,
) -> Result<Arc<Sides>, String> {
    let (path, sha, total) = source;
    let key = (
        path.to_owned(),
        sha.to_owned(),
        revision.to_owned(),
        interval,
    );
    let cache = CACHE.get_or_init(|| Mutex::new(VecDeque::new()));
    if let Some(sides) = cache
        .lock()
        .map_err(|_| "Team side cache lock")?
        .iter()
        .find(|(cached, _)| cached == &key)
        .map(|(_, sides)| Arc::clone(sides))
    {
        return Ok(sides);
    }
    let mut votes = BTreeMap::<u64, TeamSideVotes>::new();
    for_each_verified_scoped_frame(path, sha, total, &[interval], |item| {
        if let Some((frame, bytes)) = item {
            let row: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
            if row["frame_index"].as_u64() != Some(frame) {
                return Err("Invalid team side source frame".into());
            }
            for player in overlay_players(&row) {
                if player["interpolated"] == true {
                    continue;
                }
                if let (Some(id), Some(bbox)) =
                    (player["id"].as_u64(), super::box_norm(&player["box"]))
                {
                    votes.entry(id).or_default().observe(court, bbox);
                }
            }
        }
        Ok(())
    })?;
    let sides = Arc::new(
        votes
            .into_iter()
            .filter_map(|(id, vote)| Some((id, vote.side()?)))
            .collect(),
    );
    let mut entries = cache.lock().map_err(|_| "Team side cache lock")?;
    if entries.len() >= 128 {
        entries.pop_front();
    }
    entries.push_back((key, Arc::clone(&sides)));
    Ok(sides)
}

pub(super) fn annotate(
    frames: &mut [Value],
    source: (&Path, &str, u64),
    intervals: &[(u64, u64)],
    court: &CourtTransform,
    revision: &str,
) -> Result<(), String> {
    for &(start, end) in intervals {
        if !frames.iter().any(|row| {
            row["frame"]
                .as_u64()
                .is_some_and(|frame| (start..=end).contains(&frame))
        }) {
            continue;
        }
        let sides = rally_sides(source, (start, end), court, revision)?;
        for row in frames.iter_mut().filter(|row| {
            row["frame"]
                .as_u64()
                .is_some_and(|frame| (start..=end).contains(&frame))
        }) {
            if let Some(players) = row["players"].as_array_mut() {
                for player in players {
                    player["team_side"] =
                        json!(player["id"].as_u64().and_then(|id| sides.get(&id)));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::fs;

    #[test]
    fn team_sides_are_constant_within_each_rally_and_ignore_interpolation() {
        let root = std::env::temp_dir().join(format!("volley-team-sides-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("tracked_frames.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..10 {
            let (x, y) = if frame < 5 {
                (0.25, 0.75)
            } else {
                (0.75, 0.25)
            };
            let p = |id, x: f64| {
                json!({"track_id":id,"is_verified_athlete":false,
                "detection":{"bbox_xyxy_norm":[x-0.02,0.3,x+0.02,0.6]}})
            };
            let row = json!({
                "frame_index":frame,"track_ids_provisional":true,
                "confirmed_athlete_identities":false,"touches_confirmed":false,
                "interpolated_track_ids":[3],
                "tracked_players":[p(1,if frame==4 {y} else {x}),p(2,y),p(3,0.35)]
            });
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        fs::write(&path, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let court = CourtTransform::new([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]).unwrap();
        let mut frames = vec![
            json!({"frame":4,"players":[{"id":1},{"id":2},{"id":3}]}),
            json!({"frame":5,"players":[{"id":1},{"id":2},{"id":3}]}),
        ];
        annotate(
            &mut frames,
            (&path, &sha, 10),
            &[(0, 4), (5, 9)],
            &court,
            "test",
        )
        .unwrap();
        assert_eq!(frames[0]["players"][0]["team_side"], "near");
        assert_eq!(frames[0]["players"][1]["team_side"], "far");
        assert!(frames[0]["players"][2]["team_side"].is_null());
        assert_eq!(frames[1]["players"][0]["team_side"], "far");
        assert_eq!(frames[1]["players"][1]["team_side"], "near");
        assert!(frames[1]["players"][2]["team_side"].is_null());
        fs::remove_dir_all(root).unwrap();
    }
}
