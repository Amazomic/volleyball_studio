use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct Span {
    first: u64,
    last: u64,
    observations: u64,
}

fn boundaries(spans: &BTreeMap<u64, Span>, frames: u64) -> Vec<u64> {
    let reliable = spans
        .iter()
        .filter(|(_, span)| span.observations >= 12 && span.last.saturating_sub(span.first) >= 90)
        .map(|(&id, &span)| (id, span))
        .collect::<Vec<_>>();
    let mut candidates = Vec::new();
    for (_, ending) in &reliable {
        let deaths = reliable
            .iter()
            .filter(|(_, span)| {
                span.last >= ending.last.saturating_sub(15)
                    && span.last <= ending.last.saturating_add(15)
            })
            .collect::<Vec<_>>();
        let births = reliable
            .iter()
            .filter(|(_, span)| {
                span.first >= ending.last.saturating_sub(15)
                    && span.first <= ending.last.saturating_add(30)
            })
            .collect::<Vec<_>>();
        if deaths.len() < 3 || births.len() < 2 {
            continue;
        }
        let last_old = deaths.iter().map(|(_, span)| span.last).max().unwrap();
        let first_new = births.iter().map(|(_, span)| span.first).min().unwrap();
        if first_new <= last_old || first_new - last_old > 30 {
            continue;
        }
        let cut = last_old + (first_new - last_old) / 2;
        if cut < 90 || cut.saturating_add(90) >= frames {
            continue;
        }
        let survivors = reliable
            .iter()
            .filter(|(_, span)| {
                span.first.saturating_add(90) < cut && cut.saturating_add(90) < span.last
            })
            .count();
        if survivors == 0 {
            continue;
        }
        candidates.push((cut, deaths.len() + births.len()));
    }
    candidates.sort_by_key(|(cut, _)| *cut);
    let mut selected: Vec<(u64, usize)> = Vec::new();
    for candidate in candidates {
        if let Some(previous) = selected.last_mut() {
            if candidate.0.saturating_sub(previous.0) < 90 {
                if candidate.1 > previous.1 {
                    *previous = candidate;
                }
                continue;
            }
        }
        selected.push(candidate);
    }
    selected.into_iter().map(|(cut, _)| cut).collect()
}

fn remap_id(
    id: u64,
    at: u64,
    later: &BTreeMap<u64, Vec<(u64, u64)>>,
    earlier: &BTreeMap<u64, (u64, u64)>,
) -> u64 {
    if let Some(&(cut, replacement)) = earlier.get(&id) {
        if at < cut {
            return replacement;
        }
    }
    later
        .get(&id)
        .and_then(|versions| {
            versions
                .iter()
                .rev()
                .find(|(cut, _)| at >= *cut)
                .map(|(_, replacement)| *replacement)
        })
        .unwrap_or(id)
}

pub(super) fn split(
    path: &Path,
    expected_sha: &str,
    frames: u64,
) -> Result<(String, u64, u64), String> {
    let mut input = BoundedReader::open(path)?;
    let mut spans = BTreeMap::<u64, Span>::new();
    let mut handoffs = super::identity_handoff::Detector::default();
    for expected_frame in 0..frames {
        let bytes = input.line()?.ok_or("Turnover split source is truncated")?;
        let frame: Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("Turnover source JSON: {e}"))?;
        if frame["frame_index"].as_u64() != Some(expected_frame) {
            return Err("Turnover source frames are out of order".into());
        }
        let rows = frame["tracked_players"]
            .as_array()
            .ok_or("Turnover source tracks missing")?;
        handoffs.observe(expected_frame, rows);
        for row in rows {
            let id = row["track_id"]
                .as_u64()
                .filter(|id| *id > 0)
                .ok_or("Turnover source Track ID missing")?;
            let span = spans.entry(id).or_insert(Span {
                first: expected_frame,
                last: expected_frame,
                observations: 0,
            });
            span.last = expected_frame;
            span.observations += 1;
        }
    }
    if input.line()?.is_some() || input.finish() != expected_sha {
        return Err("Turnover source SHA-256 mismatch".into());
    }
    let cuts = boundaries(&spans, frames);
    let measured_handoffs = handoffs.valid_cuts(
        &spans
            .iter()
            .map(|(&id, s)| (id, (s.first, s.last, s.observations)))
            .collect(),
    );
    if cuts.is_empty() && measured_handoffs.is_empty() {
        return Ok((expected_sha.to_owned(), 0, 0));
    }
    let mut next = spans.keys().next_back().copied().unwrap_or(0);
    let mut replacements = BTreeMap::<u64, Vec<(u64, u64)>>::new();
    for cut in cuts {
        for (&id, span) in &spans {
            let previous = replacements
                .get(&id)
                .and_then(|rows| rows.last())
                .map(|(at, _)| *at)
                .unwrap_or(span.first);
            if previous.saturating_add(90) >= cut || cut.saturating_add(90) >= span.last {
                continue;
            }
            next = next.checked_add(1).ok_or("Track ID space exhausted")?;
            replacements.entry(id).or_default().push((cut, next));
        }
    }
    // Preserve the late ID for the new body, so a user seeing ID 13 at 46s
    // can assign that same ID. The earlier unrelated person receives a new ID.
    // Tracks already segmented at a global roster turnover retain that policy.
    let mut earlier = BTreeMap::<u64, (u64, u64)>::new();
    for candidate in measured_handoffs {
        if replacements.contains_key(&candidate.track) || earlier.contains_key(&candidate.track) {
            continue;
        }
        next = next.checked_add(1).ok_or("Track ID space exhausted")?;
        earlier.insert(candidate.track, (candidate.at, next));
    }
    let handoff_splits = earlier.len() as u64;
    let splits = replacements.values().map(Vec::len).sum::<usize>() as u64 + handoff_splits;
    if splits == 0 {
        return Ok((expected_sha.to_owned(), 0, 0));
    }
    let temp = path.with_extension("turnover.partial");
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp)
        .map_err(|e| format!("Create turnover split snapshot: {e}"))?;
    let mut output = BufWriter::new(file);
    let mut reader = BoundedReader::open(path)?;
    let mut hash = Sha256::new();
    let write_result = (|| -> Result<(), String> {
        for frame_index in 0..frames {
            let bytes = reader
                .line()?
                .ok_or("Turnover source changed during split")?;
            let mut frame: Value = serde_json::from_slice(&bytes)
                .map_err(|e| format!("Turnover frame changed: {e}"))?;
            if frame["frame_index"].as_u64() != Some(frame_index) {
                return Err("Turnover frames changed during split".into());
            }
            for row in frame["tracked_players"]
                .as_array_mut()
                .ok_or("Turnover tracks missing")?
            {
                let id = row["track_id"]
                    .as_u64()
                    .ok_or("Turnover Track ID missing")?;
                let new_id = remap_id(id, frame_index, &replacements, &earlier);
                if new_id != id {
                    row["track_id"] = json!(new_id);
                }
            }
            if let Some(inferred) = frame
                .get_mut("interpolated_track_ids")
                .and_then(Value::as_array_mut)
            {
                for id in inferred {
                    let previous = id.as_u64().ok_or("Invalid turnover inferred ID")?;
                    let new_id = remap_id(previous, frame_index, &replacements, &earlier);
                    if new_id != previous {
                        *id = json!(new_id);
                    }
                }
            }
            let mut serialized = serde_json::to_vec(&frame).map_err(|e| e.to_string())?;
            serialized.push(b'\n');
            if serialized.len() > MAX_ROW_BYTES {
                return Err("Turnover split row exceeds 512 KiB".into());
            }
            output.write_all(&serialized).map_err(|e| e.to_string())?;
            hash.update(&serialized);
        }
        if reader.line()?.is_some() || reader.finish() != expected_sha {
            return Err("Turnover source changed before commit".into());
        }
        output.flush().map_err(|e| e.to_string())?;
        output.get_ref().sync_all().map_err(|e| e.to_string())
    })();
    drop(output);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    fs::rename(&temp, path).map_err(|e| format!("Publish turnover split: {e}"))?;
    Ok((format!("{:x}", hash.finalize()), splits, handoff_splits))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(first: u64, last: u64, observations: u64) -> Span {
        Span {
            first,
            last,
            observations,
        }
    }

    #[test]
    fn two_rallies_separate_surviving_track_from_new_athletes() {
        let tracks = BTreeMap::from([
            (2, span(0, 849, 179)),
            (3, span(0, 750, 211)),
            (4, span(0, 850, 153)),
            (5, span(0, 1730, 249)),
            (7, span(180, 852, 141)),
            (10, span(753, 849, 33)),
            (11, span(860, 1728, 188)),
            (12, span(860, 1053, 24)),
            (13, span(860, 1728, 181)),
        ]);
        assert_eq!(boundaries(&tracks, 1737), vec![856]);
    }

    #[test]
    fn two_new_tracks_at_roster_cut_separate_surviving_identities() {
        let tracks = BTreeMap::from([
            (1, span(0, 850, 171)),
            (2, span(0, 850, 171)),
            (3, span(0, 705, 133)),
            (4, span(0, 1735, 339)),
            (6, span(155, 1735, 285)),
            (7, span(180, 850, 135)),
            (10, span(800, 1170, 46)),
            (11, span(855, 1735, 174)),
            (12, span(855, 1735, 177)),
            (13, span(955, 1735, 157)),
        ]);
        assert_eq!(boundaries(&tracks, 1737), vec![852]);
    }

    #[test]
    fn two_reacquired_tracks_without_three_departures_do_not_cut_roster() {
        let tracks = BTreeMap::from([
            (1, span(0, 850, 171)),
            (2, span(0, 850, 171)),
            (4, span(0, 1735, 339)),
            (11, span(855, 1735, 174)),
            (12, span(855, 1735, 177)),
        ]);
        assert!(boundaries(&tracks, 1737).is_empty());
    }

    #[test]
    fn brief_partial_tracking_loss_does_not_split_survivors() {
        let tracks = BTreeMap::from([
            (1, span(0, 500, 200)),
            (2, span(0, 250, 90)),
            (3, span(255, 500, 90)),
            (4, span(0, 500, 200)),
        ]);
        assert!(boundaries(&tracks, 501).is_empty());
    }

    #[test]
    fn handoff_relabels_only_earlier_person_and_retains_later_id() {
        let root = std::env::temp_dir().join(format!("volley-handoff-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let path = root.join("tracked_frames.jsonl");
        let mut file = BufWriter::new(File::create(&path).unwrap());
        let mut sha = Sha256::new();
        for frame in 0..1650_u64 {
            let mut players = Vec::new();
            let before = (860..=1040).contains(&frame) && ((frame - 860) % 3 == 0 || frame == 930);
            if before || frame == 1041 {
                players.push(json!({"track_id":12,"detection":{
                    "bbox_xyxy_norm":[0.4435,0.4749,0.4876,0.6601]}}));
            }
            if before {
                let box_before = if frame >= 1038 {
                    json!([0.4296, 0.4249, 0.4622, 0.5683])
                } else {
                    json!([0.395, 0.4249, 0.429, 0.5683])
                };
                players.push(json!({"track_id":13,"detection":{
                    "bbox_xyxy_norm":box_before}}));
            }
            if frame >= 1044 && frame <= 1599 && (frame - 1044) % 3 == 0 {
                players.push(json!({"track_id":13,"detection":{
                    "bbox_xyxy_norm":[0.4429,0.4528,0.4793,0.6537]}}));
            }
            let mut row = serde_json::to_vec(&json!({
                "frame_index":frame,"tracked_players":players,
                "raw_player_candidates":[{"source":"original"}]
            }))
            .unwrap();
            row.push(b'\n');
            file.write_all(&row).unwrap();
            sha.update(row);
        }
        file.flush().unwrap();
        drop(file);
        let original = format!("{:x}", sha.finalize());
        let (digest, splits, handoffs) = split(&path, &original, 1650).unwrap();
        assert_eq!((splits, handoffs), (1, 1));
        let bytes = fs::read(&path).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), digest);
        let rows = String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert!(rows[930]["tracked_players"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["track_id"] == 14));
        assert!(rows[930]["tracked_players"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["track_id"] == 12));
        assert!(rows[1044]["tracked_players"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["track_id"] == 13));
        assert!(rows[1389]["tracked_players"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["track_id"] == 13));
        assert_eq!(
            rows[930]["raw_player_candidates"],
            json!([{"source":"original"}])
        );
        assert_eq!(
            rows[1044]["raw_player_candidates"],
            json!([{"source":"original"}])
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn split_preserves_raw_evidence_and_separates_track_ids() {
        let root = std::env::temp_dir().join(format!("volley-turnover-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let path = root.join("tracked_frames.jsonl");
        let mut out = BufWriter::new(File::create(&path).unwrap());
        let mut sha = Sha256::new();
        for frame in 0..500_u64 {
            let mut ids = vec![4];
            if frame < 220 {
                ids.extend([1, 2, 3]);
            }
            if frame >= 226 {
                ids.extend([5, 6, 7]);
            }
            let row = json!({
                "frame_index":frame,
                "tracked_players":ids.iter().map(|id|json!({"track_id":id,"detection":{"score":0.9}})).collect::<Vec<_>>(),
                "raw_player_candidates":[{"source":"raw"}],

            });
            let mut bytes = serde_json::to_vec(&row).unwrap();
            bytes.push(b'\n');
            out.write_all(&bytes).unwrap();
            sha.update(bytes);
        }
        out.flush().unwrap();
        drop(out);
        let original = format!("{:x}", sha.finalize());
        let (digest, count, handoffs) = split(&path, &original, 500).unwrap();
        assert_eq!(count, 1);
        assert_eq!(handoffs, 0);
        assert_ne!(digest, original);
        let rows = fs::read_to_string(&path).unwrap();
        let parsed: Vec<Value> = rows
            .lines()
            .map(|row| serde_json::from_str(row).unwrap())
            .collect();
        assert_eq!(parsed.len(), 500);
        assert!(parsed[210]["tracked_players"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["track_id"] == 4));
        assert!(parsed[250]["tracked_players"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["track_id"] == 8));
        assert!(parsed[250].get("interpolated_track_ids").is_none());
        assert_eq!(
            parsed[250]["raw_player_candidates"],
            json!([{"source":"raw"}])
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(&path).unwrap())),
            digest
        );
        fs::remove_dir_all(root).unwrap();
    }
}
