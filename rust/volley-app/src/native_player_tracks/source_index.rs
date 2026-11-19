use super::*;
#[derive(Clone)]
pub(super) struct TrackIndex {
    path: PathBuf,
    sha: String,
    rally_sha: Option<String>,
    frames: u64,
    size: u64,
    modified: Option<SystemTime>,
    change: Option<(u64, u64, i64, i64)>,
    tracks: BTreeMap<u64, Aggregate>,
}
/// The in-memory map uses tuple keys. JSON object keys must be strings;
/// serialize each measured pair as a canonical "first:second" key.
pub(super) mod pair_geometry_keys {
    use super::{BTreeMap, PairGeometry};
    use serde::de::Error as _;
    use serde::ser::SerializeMap;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(
        pairs: &BTreeMap<(u64, u64), PairGeometry>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(pairs.len()))?;
        for (&(first, second), geometry) in pairs {
            map.serialize_entry(&format!("{first}:{second}"), geometry)?;
        }
        map.end()
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<BTreeMap<(u64, u64), PairGeometry>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let keys = BTreeMap::<String, PairGeometry>::deserialize(deserializer)?;
        let mut pairs = BTreeMap::new();
        for (key, geometry) in keys {
            let (first, second) = key
                .split_once(':')
                .ok_or_else(|| D::Error::custom("Invalid observed pair key"))?;
            let first = first
                .parse::<u64>()
                .map_err(|_| D::Error::custom("Invalid first observed ID"))?;
            let second = second
                .parse::<u64>()
                .map_err(|_| D::Error::custom("Invalid second observed ID"))?;
            if first == 0 || first >= second || format!("{first}:{second}") != key {
                return Err(D::Error::custom("Invalid canonical observed pair"));
            }
            pairs.insert((first, second), geometry);
        }
        Ok(pairs)
    }
}

const INDEX_CACHE_SCHEMA: &str = "volley-native-track-index/1";
const MAX_INDEX_CACHE_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
struct StoredTrackIndex {
    schema: String,
    sha: String,
    rally_sha: Option<String>,
    frames: u64,
    size: u64,
    modified: SystemTime,
    change: (u64, u64, i64, i64),
    tracks: TrackObservations,
}

fn stored_index_path(path: &Path, scope: Option<&str>) -> Option<PathBuf> {
    if path.file_name()?.to_str()? != "tracked_frames.jsonl" {
        return None;
    }
    let replay = path.parent()?;
    let replay_id = replay.file_name()?.to_str()?;
    if !opaque_id(replay_id) || replay.parent()?.file_name()?.to_str()? != "native_cpu_replays" {
        return None;
    }
    let filename = match scope {
        Some(sha) if valid_sha(sha) => format!("rally-{sha}.json"),
        Some(_) => return None,
        None => "full.json".to_owned(),
    };
    Some(
        replay
            .parent()?
            .parent()?
            .join("native_track_indices")
            .join(replay_id)
            .join(filename),
    )
}

fn read_stored_index(
    path: &Path,
    sha: &str,
    frames: u64,
    size: u64,
    modified: Option<SystemTime>,
    change: Option<(u64, u64, i64, i64)>,
    scope: Option<&str>,
) -> Option<TrackObservations> {
    let target = stored_index_path(path, scope)?;
    let changed = change?;
    let modified = modified?;
    let meta = fs::symlink_metadata(&target).ok()?;
    if meta.file_type().is_symlink()
        || !meta.is_file()
        || meta.len() == 0
        || meta.len() > MAX_INDEX_CACHE_BYTES
    {
        return None;
    }
    let bytes = fs::read(&target).ok()?;
    let stored: StoredTrackIndex = serde_json::from_slice(&bytes).ok()?;
    if stored.schema != INDEX_CACHE_SCHEMA
        || stored.sha != sha
        || stored.rally_sha.as_deref() != scope
        || stored.frames != frames
        || stored.size != size
        || stored.modified != modified
        || stored.change != changed
        || stored.tracks.len() > MAX_TRACKS
        || stored.tracks.iter().any(|(&id, info)| {
            id == 0
                || id > i64::MAX as u64
                || info.first > info.last
                || info.last >= frames
                || info.observed == 0
                || info.observed > frames
                || info.best_frame < info.first
                || info.best_frame > info.last
                || info.visible_spans.len() > frames as usize
        })
    {
        return None;
    }
    Some(stored.tracks)
}

type StoredSourceStamp = (u64, Option<SystemTime>, Option<(u64, u64, i64, i64)>);

fn store_index(
    path: &Path,
    sha: &str,
    frames: u64,
    metadata: StoredSourceStamp,
    scope: Option<&str>,
    tracks: &TrackObservations,
) -> Result<(), String> {
    let (size, modified, change) = metadata;
    let (Some(target), Some(modified), Some(change)) =
        (stored_index_path(path, scope), modified, change)
    else {
        return Ok(());
    };
    let cache_dir = target.parent().ok_or("Missing Track ID index folder")?;
    let root = cache_dir.parent().ok_or("Missing Track ID index root")?;
    for folder in [root, cache_dir] {
        match fs::symlink_metadata(folder) {
            Ok(meta) if meta.file_type().is_symlink() || !meta.is_dir() => {
                return Err("Unsafe persistent Track ID index folder".into());
            }
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(folder).map_err(|e| format!("Create Track ID index: {e}"))?;
            }
            Err(err) => return Err(format!("Check Track ID index folder: {err}")),
        }
    }
    let record = StoredTrackIndex {
        schema: INDEX_CACHE_SCHEMA.into(),
        sha: sha.into(),
        rally_sha: scope.map(str::to_owned),
        frames,
        size,
        modified,
        change,
        tracks: tracks.clone(),
    };
    // Large replay indices may exceed the generic 32 MiB settings/sidecar
    // limit. Use a bounded, compact, atomic index-specific writer (128 MiB).
    let bytes =
        serde_json::to_vec(&record).map_err(|e| format!("Serialize Track ID index: {e}"))?;
    if bytes.len() as u64 > MAX_INDEX_CACHE_BYTES {
        return Err(format!(
            "Persistent Track ID index is too large ({} bytes; limit {} bytes)",
            bytes.len(),
            MAX_INDEX_CACHE_BYTES
        ));
    }
    match fs::symlink_metadata(&target) {
        Ok(meta) if meta.file_type().is_symlink() || !meta.is_file() => {
            return Err("Unsafe persistent Track ID index output".into());
        }
        Ok(_) => (),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        Err(e) => return Err(format!("Inspect Track ID index output: {e}")),
    }
    let temp = cache_dir.join(format!(
        ".{}.{}.tmp",
        target.file_name().unwrap_or_default().to_string_lossy(),
        uuid::Uuid::new_v4()
    ));
    let written = (|| -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, &target)
    })();
    if let Err(error) = written {
        let _ = fs::remove_file(&temp);
        return Err(format!("Persist Track ID index: {error}"));
    }
    Ok(())
}

static TRACK_INDEX_GATES: OnceLock<Mutex<BTreeMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();

pub(crate) fn prepare_replay_index(path: &Path, sha: &str, frames: u64) -> Result<(), String> {
    let tracks = index_replay(path, sha, frames)?;
    if let Some(target) = stored_index_path(path, None) {
        // An earlier read may have cached an index in memory while the disk
        // write failed. Ensure persistence for the CPU analysis boundary.
        let meta = fs::metadata(path).map_err(|e| format!("Track index source: {e}"))?;
        let modified = meta.modified().ok();
        let change = volley_media::preparation::file_change_stamp(path, &meta);
        if read_stored_index(path, sha, frames, meta.len(), modified, change, None).is_none() {
            if change.is_none() || modified.is_none() {
                return Err(
                    "Persistent Track ID index requires a reliable source change stamp".into(),
                );
            }
            store_index(
                path,
                sha,
                frames,
                (meta.len(), modified, change),
                None,
                &tracks,
            )?;
            if read_stored_index(path, sha, frames, meta.len(), modified, change, None).is_none() {
                return Err(format!(
                    "Persistent Track ID index failed verification: {}",
                    target.display()
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn prepare_rally_index(
    state: &AppState,
    project_id: &str,
    video: &str,
    replay_id: &str,
    rally_run_id: &str,
) -> Result<(), String> {
    source_for_rallies(state, project_id, video, replay_id, Some(rally_run_id))
        .map_err(|e| format!("Build rally Track ID index: {e:?}"))?;
    Ok(())
}

static TRACK_INDICES: OnceLock<Mutex<Vec<TrackIndex>>> = OnceLock::new();
type TrackObservations = BTreeMap<u64, Aggregate>;
pub(super) fn index_replay(
    path: &Path,
    expected_sha: &str,
    frames: u64,
) -> Result<BTreeMap<u64, Aggregate>, String> {
    index_with_scope(path, expected_sha, frames, None).map(|(all, _)| all)
}

pub(super) fn index_rally_replay(
    path: &Path,
    expected_sha: &str,
    frames: u64,
    rally_sha: &str,
    windows: &crate::native_rally_evidence::RallyWindows,
) -> Result<(TrackObservations, TrackObservations), String> {
    let (selected, full) =
        index_with_scope(path, expected_sha, frames, Some((rally_sha, windows)))?;
    Ok((
        selected,
        full.ok_or("Full source index missing after scoped tracking scan")?,
    ))
}

fn index_with_scope(
    path: &Path,
    expected_sha: &str,
    frames: u64,
    scope: Option<(&str, &crate::native_rally_evidence::RallyWindows)>,
) -> Result<(TrackObservations, Option<TrackObservations>), String> {
    if frames == 0 || frames > 48 * 3600 * 30 || !valid_sha(expected_sha) {
        return Err("Unsupported native track replay length or SHA-256".into());
    }
    if scope.is_some_and(|(sha, _)| !valid_sha(sha)) {
        return Err("Invalid selected rally SHA-256".into());
    }
    let gate = {
        let store = TRACK_INDEX_GATES.get_or_init(|| Mutex::new(BTreeMap::new()));
        let mut paths = store
            .lock()
            .map_err(|_| "Track index gate lock is poisoned")?;
        paths
            .entry(path.to_owned())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    };
    let _gate = gate
        .lock()
        .map_err(|_| "Track index build lock is poisoned")?;
    let meta = fs::metadata(path).map_err(|e| format!("Track replay metadata: {e}"))?;
    let modified = meta.modified().ok();
    let change = volley_media::preparation::file_change_stamp(path, &meta);
    let selected_sha = scope.map(|(sha, _)| sha);
    let store = TRACK_INDICES.get_or_init(|| Mutex::new(Vec::new()));
    let same_file = |item: &TrackIndex| {
        item.path == path
            && item.sha == expected_sha
            && item.frames == frames
            && item.size == meta.len()
            && item.modified == modified
            && item.change == change
    };
    if change.is_some() {
        let cached = store.lock().map_err(|_| "Track index lock is poisoned")?;
        if let Some(hit) = cached
            .iter()
            .find(|item| same_file(item) && item.rally_sha.as_deref() == selected_sha)
        {
            if scope.is_none() {
                return Ok((hit.tracks.clone(), None));
            }
            if let Some(full) = cached
                .iter()
                .find(|item| same_file(item) && item.rally_sha.is_none())
            {
                return Ok((hit.tracks.clone(), Some(full.tracks.clone())));
            }
        }
    }
    let full_disk = read_stored_index(
        path,
        expected_sha,
        frames,
        meta.len(),
        modified,
        change,
        None,
    );
    let scoped_disk = selected_sha.and_then(|sha| {
        read_stored_index(
            path,
            expected_sha,
            frames,
            meta.len(),
            modified,
            change,
            Some(sha),
        )
    });
    if let Some(selected) = if scope.is_none() {
        full_disk.clone()
    } else {
        scoped_disk.clone()
    } {
        if scope.is_none() || full_disk.is_some() {
            let mut cache = store.lock().map_err(|_| "Track index lock is poisoned")?;
            let entry = |rally_sha: Option<&str>, tracks: TrackObservations| TrackIndex {
                path: path.to_owned(),
                sha: expected_sha.to_owned(),
                rally_sha: rally_sha.map(str::to_owned),
                frames,
                size: meta.len(),
                modified,
                change,
                tracks,
            };
            if let Some(full) = &full_disk {
                cache.retain(|item| item.path != path || item.rally_sha.is_some());
                cache.push(entry(None, full.clone()));
            }
            if scope.is_some() {
                cache.retain(|item| item.path != path || item.rally_sha.as_deref() != selected_sha);
                cache.push(entry(selected_sha, selected.clone()));
            }
            while cache.len() > 8 {
                cache.remove(0);
            }
            return Ok((selected, if scope.is_some() { full_disk } else { None }));
        }
    }
    let (tracks, full) = scan_with_scope(
        path,
        expected_sha,
        frames,
        scope.map(|(_, windows)| windows),
    )?;
    let after = fs::metadata(path).map_err(|e| format!("Check indexed replay: {e}"))?;
    if after.len() != meta.len()
        || after.modified().ok() != modified
        || volley_media::preparation::file_change_stamp(path, &after) != change
    {
        return Err("Track replay changed while building the index".into());
    }
    if change.is_some() {
        let _ = store_index(
            path,
            expected_sha,
            frames,
            (meta.len(), modified, change),
            None,
            full.as_ref().unwrap_or(&tracks),
        );
        if let Some(sha) = selected_sha {
            let _ = store_index(
                path,
                expected_sha,
                frames,
                (meta.len(), modified, change),
                Some(sha),
                &tracks,
            );
        }
        let mut cache = store.lock().map_err(|_| "Track index lock is poisoned")?;
        let entry = |rally_sha: Option<&str>, tracks: TrackObservations| TrackIndex {
            path: path.to_owned(),
            sha: expected_sha.to_owned(),
            rally_sha: rally_sha.map(str::to_owned),
            frames,
            size: meta.len(),
            modified,
            change,
            tracks,
        };
        if let Some(ref all) = full {
            cache.retain(|row| row.path != path || row.rally_sha.is_some());
            cache.push(entry(None, all.clone()));
        }
        cache.retain(|row| row.path != path || row.rally_sha.as_deref() != selected_sha);
        cache.push(entry(selected_sha, tracks.clone()));
        while cache.len() > 8 {
            cache.remove(0);
        }
    }
    Ok((tracks, full))
}
fn observe_track(
    tracks: &mut BTreeMap<u64, Aggregate>,
    tracked: &Tracked,
    frame_idx: u64,
) -> Result<(), String> {
    if !tracks.contains_key(&tracked.track_id) && tracks.len() >= MAX_TRACKS {
        return Err("Too many distinct native Track IDs".into());
    }
    let record = tracks.entry(tracked.track_id).or_insert_with(|| Aggregate {
        first: frame_idx,
        last: frame_idx,
        observed: 0,
        best_frame: frame_idx,
        best_score: 0.0,
        best_box: [0.0; 4],
        first_box: tracked.detection.bbox_xyxy_norm,
        last_box: tracked.detection.bbox_xyxy_norm,
        photo_samples: Vec::new(),
        visible_spans: Vec::new(),
        observed_pairs: BTreeMap::new(),
    });
    if let Some((_, end)) = record.visible_spans.last_mut() {
        if *end == frame_idx || end.saturating_add(1) == frame_idx {
            *end = frame_idx;
        } else {
            record.visible_spans.push((frame_idx, frame_idx));
        }
    } else {
        record.visible_spans.push((frame_idx, frame_idx));
    }
    record.last = frame_idx;
    record.last_box = tracked.detection.bbox_xyxy_norm;
    record.observed += 1;
    if record.photo_samples.is_empty() || frame_idx.is_multiple_of(6) {
        let sample = PhotoSample {
            frame: frame_idx,
            bbox: tracked.detection.bbox_xyxy_norm,
        };
        if record.photo_samples.len() < 12 {
            record.photo_samples.push(sample);
        } else {
            let slot = (frame_idx / 6 % 12) as usize;
            record.photo_samples[slot] = sample;
        }
    }
    if tracked.detection.score > record.best_score {
        record.best_score = tracked.detection.score;
        record.best_frame = frame_idx;
        record.best_box = tracked.detection.bbox_xyxy_norm;
    }
    Ok(())
}
fn boxes_co_located(a: [f64; 4], b: [f64; 4]) -> bool {
    let width_a = a[2] - a[0];
    let height_a = a[3] - a[1];
    let width_b = b[2] - b[0];
    let height_b = b[3] - b[1];
    if width_a <= 0.0 || width_b <= 0.0 || height_a <= 0.0 || height_b <= 0.0 {
        return false;
    }
    let area_a = width_a * height_a;
    let area_b = width_b * height_b;
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let iou = intersection / (area_a + area_b - intersection);
    let center_x = ((a[0] + a[2]) - (b[0] + b[2])).abs() * 0.5;
    let center_y = ((a[1] + a[3]) - (b[1] + b[3])).abs() * 0.5;
    iou >= 0.62
        && area_a.min(area_b) / area_a.max(area_b) >= 0.55
        && center_x <= 0.2 * width_a.min(width_b)
        && center_y <= 0.2 * height_a.min(height_b)
}

fn record_observed_pairs(tracks: &mut TrackObservations, frame: u64, boxes: &[(u64, [f64; 4])]) {
    for (position, (first_id, first_box)) in boxes.iter().enumerate() {
        for (second_id, second_box) in boxes.iter().skip(position + 1) {
            let pair = ((*first_id).min(*second_id), (*first_id).max(*second_id));
            let same_position = boxes_co_located(*first_box, *second_box);
            for id in [*first_id, *second_id] {
                let observation = tracks
                    .get_mut(&id)
                    .expect("observed Track ID already indexed")
                    .observed_pairs
                    .entry(pair)
                    .or_default();
                if same_position {
                    observation.same_position = observation.same_position.saturating_add(1);
                } else {
                    observation.distinct_position = observation.distinct_position.saturating_add(1);
                    if observation.first_distinct_frame.is_none() {
                        observation.first_distinct_frame = Some(frame);
                    }
                }
            }
        }
    }
}

fn scan_with_scope(
    path: &Path,
    expected_sha: &str,
    frames: u64,
    windows: Option<&crate::native_rally_evidence::RallyWindows>,
) -> Result<(TrackObservations, Option<TrackObservations>), String> {
    let mut reader = BoundedReader::open(path)?;
    let mut full = BTreeMap::<u64, Aggregate>::new();
    let mut scoped = windows.map(|_| BTreeMap::<u64, Aggregate>::new());
    for frame_idx in 0..frames {
        let bytes = reader
            .line()?
            .ok_or("Native Track ID frame source is truncated")?;
        let frame: Frame = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Invalid native Track ID row: {e}"))?;
        if frame.frame_index != frame_idx
            || !frame.track_ids_provisional
            || frame.confirmed_athlete_identities
            || frame.touches_confirmed
            || frame.tracked_players.len() > 128
        {
            return Err("Native track source is out of order or claims verified identities".into());
        }
        let in_play = windows.is_some_and(|w| w.contains(frame_idx));
        let inferred: BTreeSet<u64> = frame.interpolated_track_ids.iter().copied().collect();
        if inferred.len() != frame.interpolated_track_ids.len()
            || inferred.len() > frame.tracked_players.len()
        {
            return Err("Duplicate or invalid inferred Track ID metadata".into());
        }
        let mut appeared = BTreeSet::new();
        let mut observed_boxes = Vec::new();
        for tracked in &frame.tracked_players {
            if tracked.track_id == 0
                || tracked.track_id > i64::MAX as u64
                || tracked.is_verified_athlete
                || !appeared.insert(tracked.track_id)
                || tracked.detection.frame_index != frame_idx
                || tracked.detection.class_name != "player"
            {
                return Err("Native track has fabricated identity or repeated ID per frame".into());
            }
            tracked
                .detection
                .validate(frame_idx - frame_idx % 9, "players")?;
            // Interpolated positions are visual continuity, not independent
            // observations of an athlete. Including them makes two separate
            // IDs appear co-visible during a gap and fabricates ReID conflicts.
            if inferred.contains(&tracked.track_id) {
                continue;
            }
            observed_boxes.push((tracked.track_id, tracked.detection.bbox_xyxy_norm));
            observe_track(&mut full, tracked, frame_idx)?;
            if in_play {
                if let Some(ref mut selected) = scoped {
                    observe_track(selected, tracked, frame_idx)?;
                }
            }
        }
        record_observed_pairs(&mut full, frame_idx, &observed_boxes);
        if in_play {
            if let Some(ref mut selected) = scoped {
                record_observed_pairs(selected, frame_idx, &observed_boxes);
            }
        }
        if !inferred.is_subset(&appeared) {
            return Err("Interpolated Track ID metadata references a missing player".into());
        }
    }
    if reader.line()?.is_some() {
        return Err("Extra undeclared Track ID frames".into());
    }
    if reader.finish() != expected_sha {
        return Err("Native Track ID SHA-256 mismatch".into());
    }
    Ok(match scoped {
        Some(selected) => (selected, Some(full)),
        None => (full, None),
    })
}

pub(super) fn source(
    state: &AppState,
    project_id: &str,
    video: &str,
    replay_id: &str,
) -> Result<Inputs, ApiError> {
    source_with_verified_tracks(state, project_id, video, replay_id, None)
}

/// A scoped index is already hash-verified from the exact immutable replay.
/// Reuse its full-scope identity set for manual link validation instead of
/// reading the entire video a second time on filesystems without change stamps.
fn source_with_verified_tracks(
    state: &AppState,
    project_id: &str,
    video: &str,
    replay_id: &str,
    verified: Option<(&str, u64, TrackObservations)>,
) -> Result<Inputs, ApiError> {
    storage::project_id(project_id)?;
    storage::video_name(video)?;
    if !opaque_id(replay_id) {
        return Err(storage::bad_request("Invalid native replay ID"));
    }
    let project = projects::existing_project(state, project_id)?;
    if !project["videos"]
        .as_array()
        .is_some_and(|v| v.iter().any(|row| row.as_str() == Some(video)))
    {
        return Err(storage::bad_request(
            "Video does not belong to selected project",
        ));
    }
    let plan =
        preparation::plan_for_run(state, project_id, video, "native_cpu_replays", replay_id)?;
    let prepared = plan
        .ready_path()
        .ok_or_else(|| storage::bad_request("Prepared video missing"))?;
    let (size, mtime) = inference_stream::source_stamp(prepared).map_err(storage::bad_request)?;
    let project_dir = projects::project_root(state).join(project_id);
    safe_folder(&project_dir).map_err(storage::bad_request)?;
    let replay_root = project_dir.join("native_cpu_replays");
    safe_folder(&replay_root).map_err(storage::bad_request)?;
    let replay_dir = replay_root.join(replay_id);
    safe_folder(&replay_dir).map_err(storage::bad_request)?;
    let manifest: Value = serde_json::from_slice(
        &fs::read(
            safe_file(&replay_dir, "manifest.json", MAX_MANIFEST).map_err(storage::bad_request)?,
        )
        .map_err(|e| storage::internal(format!("Read Track ID manifest: {e}")))?,
    )
    .map_err(|e| storage::bad_request(format!("Bad Track ID manifest: {e}")))?;
    if manifest["schema"] != "volley-native-cpu-tracking/1"
        || manifest["state"] != "provisional_tracklets_rebuilt"
        || manifest["project_id"].as_str() != Some(project_id)
        || manifest["video"].as_str() != Some(video)
        || manifest["replay_id"].as_str() != Some(replay_id)
        || manifest["source_identity"] != json!({"bytes":size,"mtime_ns":mtime})
        || manifest["fps"] != 30
        || manifest["file"] != "tracked_frames.jsonl"
        || manifest["track_ids_provisional"] != true
        || manifest["athlete_identities_confirmed"] != false
        || manifest["touches_confirmed"] != false
        || manifest["cpu_processing_complete"] != false
    {
        return Err(storage::bad_request(
            "Track ID replay is not valid for this video",
        ));
    }
    let frames = manifest["frames"]
        .as_u64()
        .filter(|x| *x > 0 && *x <= 48 * 3600 * 30)
        .ok_or_else(|| storage::bad_request("Invalid native Track ID frame count"))?;
    let sha = manifest["sha256"]
        .as_str()
        .filter(|s| valid_sha(s))
        .ok_or_else(|| storage::bad_request("Missing Track ID source SHA-256"))?
        .to_owned();
    let tracks = if let Some((verified_sha, verified_frames, tracks)) = verified {
        if sha != verified_sha || frames != verified_frames {
            return Err(storage::bad_request(
                "Rally-scoped and full-video Track ID source manifests disagree",
            ));
        }
        tracks
    } else {
        index_replay(
            &safe_file(&replay_dir, "tracked_frames.jsonl", MAX_EVIDENCE)
                .map_err(storage::bad_request)?,
            &sha,
            frames,
        )
        .map_err(storage::bad_request)?
    };
    let evidence = EvidenceSource::native(
        project_id,
        video,
        replay_id,
        &sha,
        SourceStamp {
            bytes: size,
            mtime_ns: u64::try_from(mtime)
                .map_err(|_| storage::bad_request("Video timestamp outside supported range"))?,
        },
        FrameRate::Exact {
            numerator: 30,
            denominator: 1,
        },
    )
    .map_err(storage::bad_request)?;
    let known = VerifiedTrackSet::new(evidence.clone(), tracks.keys().copied())
        .map_err(storage::bad_request)?;
    let (catalog, _) = player_catalog::load(state, false)?;
    if project
        .get("player_ids")
        .is_some_and(|value| !value.is_null() && !value.is_array())
    {
        return Err(storage::bad_request(
            "Project roster must contain a player_ids array",
        ));
    }
    let allowed = project["player_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>();
    let mut roster = BTreeMap::new();
    for entry in catalog["players"].as_array().into_iter().flatten() {
        if let (Some(id), Some(name)) = (entry["id"].as_str(), entry["name"].as_str()) {
            if valid_player_id(id) && allowed.contains(id) {
                roster.insert(id.to_owned(), name.to_owned());
            }
        }
    }
    let links_dir = project_dir.join("native_track_assignments");
    if let Ok(meta) = fs::symlink_metadata(&links_dir) {
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err(storage::bad_request("Unsafe player assignment directory"));
        }
    }
    Ok(Inputs {
        evidence,
        known,
        project_id: project_id.into(),
        video: video.into(),
        replay_id: replay_id.into(),
        source_sha: sha,
        tracks,
        roster,
        sidecar: links_dir.join(format!("{replay_id}.json")),
        can_write: state.writes_enabled,
    })
}

/// A rally run is the authoritative scope for displayed/assigned Track IDs.
/// Keep full-frame evidence unchanged for subsequent CPU Touch/Rally reprocessing.
pub(super) fn source_for_rallies(
    state: &AppState,
    project_id: &str,
    video: &str,
    replay_id: &str,
    rally_run_id: Option<&str>,
) -> Result<Inputs, ApiError> {
    let Some(run_id) = rally_run_id else {
        return source(state, project_id, video, replay_id);
    };
    let run = crate::native_rally_evidence::inspect_project(state, project_id, run_id)?;
    if run.replay_id != replay_id || run.video != video {
        return Err(storage::bad_request(
            "Selected rally version does not match CPU Track ID source",
        ));
    }
    let windows =
        crate::native_rally_evidence::verified_track_windows(&run).map_err(storage::bad_request)?;
    let replay_dir = projects::project_root(state)
        .join(project_id)
        .join("native_cpu_replays")
        .join(replay_id);
    let (scoped, full) = index_rally_replay(
        &safe_file(&replay_dir, "tracked_frames.jsonl", MAX_EVIDENCE)
            .map_err(storage::bad_request)?,
        &run.replay_sha,
        run.frames,
        &run.rally_sha,
        &windows,
    )
    .map_err(storage::bad_request)?;
    // The same verified scan collected *both* the rally subset and all IDs.
    let mut ctx = source_with_verified_tracks(
        state,
        project_id,
        video,
        replay_id,
        Some((&run.replay_sha, run.frames, full)),
    )?;
    if run.replay_sha != ctx.source_sha {
        return Err(storage::bad_request("Rally/CPU replay checksum mismatch"));
    }
    ctx.tracks = scoped;
    // Only observations inside detected play count for player review.
    // Avoid admitting someone observed for a few frames during a rally
    // just because they were visible for minutes during an intermission.
    let manifest: Value = serde_json::from_slice(
        &fs::read(
            safe_file(&replay_dir, "manifest.json", MAX_MANIFEST).map_err(storage::bad_request)?,
        )
        .map_err(|e| storage::internal(format!("Read CPU tracking settings: {e}")))?,
    )
    .map_err(|e| storage::bad_request(format!("Malformed CPU tracking settings: {e}")))?;
    let min_observations = manifest["settings"]["min_track_observations"]
        .as_u64()
        .filter(|n| (6..=90).contains(n))
        .unwrap_or(12);
    ctx.tracks.retain(|_, row| row.observed >= min_observations);
    Ok(ctx)
}

#[cfg(test)]
mod rally_scope_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use uuid::Uuid;

    #[test]
    fn exact_rally_intervals_exclude_intermission_ids_and_photo_candidates() {
        let root = std::env::temp_dir().join(format!("volley-rally-index-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("tracked_frames.jsonl");
        let mut raw = Vec::new();
        for frame in 0..48u64 {
            let mut tracks = vec![(7, 0.90)];
            if (2..=5).contains(&frame) {
                tracks.push((9, 0.99)); // spectator visible only before the play
            }
            if (16..=18).contains(&frame) {
                tracks.push((8, 0.91)); // genuine player appearing during play
            }
            if (29..=32).contains(&frame) {
                tracks.push((10, 0.98)); // player visible only between plays
            }
            let players = tracks
                .iter()
                .enumerate()
                .map(|(query, (id, score))| {
                    json!({
                        "track_id":id,"is_verified_athlete":false,
                        "detection":{
                            "frame_index":frame,"class_name":"player","score":score,
                            "bbox_xyxy_norm":[0.15 + 0.15 * (query as f64),0.2,
                                0.25 + 0.15 * (query as f64),0.75],
                            "query_index":query,
                            "touch_probability":null,"head_point_norm":null
                        }
                    })
                })
                .collect::<Vec<_>>();
            let row = json!({
                "frame_index":frame,"track_ids_provisional":true,
                "confirmed_athlete_identities":false,"touches_confirmed":false,
                "tracked_players":players,"selected_ball":null
            });
            raw.extend(serde_json::to_vec(&row).unwrap());
            raw.push(b'\n');
        }
        fs::write(&path, &raw).unwrap();
        let sha = format!("{:x}", Sha256::digest(&raw));
        let (full, secondary) = scan_with_scope(&path, &sha, 48, None).unwrap();
        assert!(secondary.is_none());
        assert_eq!(full.len(), 4);
        let windows = crate::native_rally_evidence::RallyWindows {
            intervals: vec![(12, 20), (40, 44)],
            contexts: std::collections::BTreeMap::new(),
        };
        let (scoped, full_also) = scan_with_scope(&path, &sha, 48, Some(&windows)).unwrap();
        assert_eq!(full_also.unwrap().len(), 4);
        assert_eq!(scoped.len(), 2);
        assert_eq!(scoped.get(&7).unwrap().observed, 14);
        assert_eq!(scoped.get(&7).unwrap().first, 12);
        assert_eq!(scoped.get(&7).unwrap().last, 44);
        assert_eq!(scoped.get(&7).unwrap().best_frame, 12);
        let photos = &scoped.get(&7).unwrap().photo_samples;
        assert!(photos.len() >= 2);
        assert!(photos.iter().all(|photo| windows.contains(photo.frame)));
        assert_ne!(photos[0].frame, photos[1].frame);
        assert_eq!(scoped.get(&8).unwrap().observed, 3);
        assert!(!scoped.contains_key(&9));
        assert!(!scoped.contains_key(&10));
        assert!(scan_with_scope(&path, &"0".repeat(64), 48, Some(&windows)).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod inferred_visibility_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use uuid::Uuid;

    #[test]
    fn inferred_positions_never_create_co_visibility_or_reid_samples() {
        let root = std::env::temp_dir().join(format!("track-observation-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let file = root.join("frames.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..9_u64 {
            let first_actual = [0, 2, 8].contains(&frame);
            let second_actual = [4, 6].contains(&frame);
            let first_inferred = (3..8).contains(&frame);
            let second_inferred = frame == 5;
            let mut players = Vec::new();
            let mut inferred = Vec::new();
            for (id, include, synthetic) in [
                (7, first_actual || first_inferred, first_inferred),
                (8, second_actual || second_inferred, second_inferred),
            ] {
                if !include {
                    continue;
                }
                players.push(json!({
                    "track_id":id,"is_verified_athlete":false,
                    "detection":{
                        "frame_index":frame,"class_name":"player",
                        "score":0.88,"query_index":0,
                        "bbox_xyxy_norm":[0.30,0.30,0.40,0.70]
                    }
                }));
                if synthetic {
                    inferred.push(id);
                }
            }
            let row = json!({
                "frame_index":frame,
                "selected_ball":null,
                "tracked_players":players,
                "interpolated_track_ids":inferred,
                "track_ids_provisional":true,
                "confirmed_athlete_identities":false,
                "touches_confirmed":false
            });
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        fs::write(&file, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let indexed = index_replay(&file, &sha, 9).unwrap();
        assert_eq!(indexed[&7].observed, 3);
        assert_eq!(indexed[&8].observed, 2);
        assert!(!tracks_co_visible(&indexed[&7], &indexed[&8]));
        assert!(indexed[&7]
            .photo_samples
            .iter()
            .all(|p| [0, 2, 8].contains(&p.frame)));
        assert!(indexed[&8]
            .photo_samples
            .iter()
            .all(|p| [4, 6].contains(&p.frame)));
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod spatial_duplicate_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use uuid::Uuid;

    fn observed_pair(second_boxes: &[[f64; 4]], interpolated: bool) -> TrackObservations {
        let root = std::env::temp_dir().join(format!("track-pair-{}", Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let file = root.join("tracked_frames.jsonl");
        let mut bytes = Vec::new();
        for (frame, second) in second_boxes.iter().enumerate() {
            let players = vec![
                json!({
                    "track_id": 7, "is_verified_athlete": false,
                    "detection": {
                        "frame_index":frame, "class_name":"player", "score":0.92, "query_index":0,
                        "bbox_xyxy_norm":[0.2,0.1,0.4,0.7]
                    }
                }),
                json!({
                    "track_id": 8, "is_verified_athlete": false,
                    "detection": {
                        "frame_index":frame, "class_name":"player", "score":0.90, "query_index":0,
                        "bbox_xyxy_norm":second
                    }
                }),
            ];
            let row = json!({
                "frame_index":frame, "tracked_players":players,
                "interpolated_track_ids":if interpolated {vec![8]} else {vec![]},
                "track_ids_provisional":true,
                "confirmed_athlete_identities":false,
                "touches_confirmed":false
            });
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        fs::write(&file, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let indexed = index_replay(&file, &sha, second_boxes.len() as u64).unwrap();
        fs::remove_dir_all(root).unwrap();
        indexed
    }

    #[test]
    fn true_duplicate_player_boxes_can_share_athlete_assignment() {
        let boxes = [[0.21, 0.11, 0.41, 0.71]; 4];
        let indexed = observed_pair(&boxes, false);
        assert_eq!(indexed[&7].observed_pairs[&(7, 8)].same_position, 4);
        assert_eq!(indexed[&7].observed_pairs[&(7, 8)].distinct_position, 0);
        assert!(!tracks_co_visible(&indexed[&7], &indexed[&8]));
    }

    #[test]
    fn spatially_distinct_players_still_conflict() {
        let boxes = [[0.60, 0.1, 0.80, 0.7]; 4];
        let indexed = observed_pair(&boxes, false);
        assert_eq!(indexed[&7].observed_pairs[&(7, 8)].distinct_position, 4);
        assert!(tracks_co_visible(&indexed[&7], &indexed[&8]));
    }

    #[test]
    fn brief_or_inconsistent_overlap_does_not_suppress_identity_conflict() {
        let short = observed_pair(&[[0.2, 0.1, 0.4, 0.7]], false);
        assert!(tracks_co_visible(&short[&7], &short[&8]));
        let mixed = observed_pair(
            &[
                [0.2, 0.1, 0.4, 0.7],
                [0.2, 0.1, 0.4, 0.7],
                [0.6, 0.1, 0.8, 0.7],
            ],
            false,
        );
        assert!(tracks_co_visible(&mixed[&7], &mixed[&8]));
    }

    #[test]
    fn interpolated_boxes_do_not_create_spatial_duplicate_evidence() {
        let boxes = [[0.2, 0.1, 0.4, 0.7]; 4];
        let indexed = observed_pair(&boxes, true);
        assert!(!indexed.contains_key(&8));
        assert!(indexed[&7].observed_pairs.is_empty());
    }
}

#[cfg(test)]
mod persistent_index_tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn multiple_covisible_players_are_persisted_with_json_pair_keys() {
        let root = std::env::temp_dir().join(format!("volley-multiple-tracks-{}", Uuid::new_v4()));
        let replay = root.join("native_cpu_replays").join("a".repeat(32));
        fs::create_dir_all(&replay).unwrap();
        let path = replay.join("tracked_frames.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..4 {
            let tracked = |id: u64, bbox| {
                json!({
                    "track_id":id,
                    "is_verified_athlete":false,
                    "detection":{
                        "frame_index":frame,"class_name":"player","score":0.92,
                        "bbox_xyxy_norm":bbox,"query_index":id
                    }
                })
            };
            let row = json!({
                "frame_index":frame,"selected_ball":null,
                "interpolated_track_ids":[],
                "track_ids_provisional":true,
                "confirmed_athlete_identities":false,
                "touches_confirmed":false,
                "tracked_players":[tracked(7,[0.10,0.15,0.30,0.80]),
                                   tracked(8,[0.60,0.15,0.80,0.80])]
            });
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        fs::write(&path, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        prepare_replay_index(&path, &sha, 4).unwrap();
        let cached = stored_index_path(&path, None).unwrap();
        let on_disk = fs::read_to_string(&cached).unwrap();
        assert!(
            on_disk.contains(r#""7:8""#),
            "Measured pair tuple keys must survive JSON storage"
        );
        TRACK_INDICES.get().unwrap().lock().unwrap().clear();
        let reloaded = index_replay(&path, &sha, 4).unwrap();
        assert_eq!(reloaded[&7].observed_pairs[&(7, 8)].distinct_position, 4);
        assert_eq!(reloaded[&8].observed_pairs[&(7, 8)].distinct_position, 4);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn index_is_saved_reused_and_reverified_when_source_changes() {
        let root = std::env::temp_dir().join(format!("volley-persisted-tracks-{}", Uuid::new_v4()));
        let replay_id = "a".repeat(32);
        let replay = root.join("native_cpu_replays").join(&replay_id);
        fs::create_dir_all(&replay).unwrap();
        let path = replay.join("tracked_frames.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..3u64 {
            let row = json!({
                "frame_index":frame,
                "selected_ball":null,
                "interpolated_track_ids":[],
                "track_ids_provisional":true,
                "confirmed_athlete_identities":false,
                "touches_confirmed":false,
                "tracked_players":[{
                    "track_id":7,
                    "is_verified_athlete":false,
                    "detection":{
                        "frame_index":frame,
                        "class_name":"player",
                        "score":0.91,
                        "bbox_xyxy_norm":[0.2,0.2,0.4,0.7],
                        "query_index":0
                    }
                }]
            });
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        fs::write(&path, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        prepare_replay_index(&path, &sha, 3).unwrap();
        let cache_path = stored_index_path(&path, None).unwrap();
        assert!(cache_path.is_file());
        // Even a warm in-memory cache must re-create a missing durable index.
        fs::remove_file(&cache_path).unwrap();
        prepare_replay_index(&path, &sha, 3).unwrap();
        assert!(cache_path.is_file());
        TRACK_INDICES.get().unwrap().lock().unwrap().clear();
        assert_eq!(index_replay(&path, &sha, 3).unwrap()[&7].observed, 3);
        let rally_sha = "b".repeat(64);
        let windows = crate::native_rally_evidence::RallyWindows {
            intervals: vec![(1, 2)],
            contexts: BTreeMap::new(),
        };
        let (in_play, all) = index_rally_replay(&path, &sha, 3, &rally_sha, &windows).unwrap();
        assert_eq!(in_play[&7].observed, 2);
        assert_eq!(all[&7].observed, 3);
        assert!(stored_index_path(&path, Some(&rally_sha))
            .unwrap()
            .is_file());
        TRACK_INDICES.get().unwrap().lock().unwrap().clear();
        let (reloaded, full) = index_rally_replay(&path, &sha, 3, &rally_sha, &windows).unwrap();
        assert_eq!(reloaded[&7].observed, 2);
        assert_eq!(full[&7].observed, 3);

        fs::write(&cache_path, b"{corrupt").unwrap();
        TRACK_INDICES.get().unwrap().lock().unwrap().clear();
        assert_eq!(index_replay(&path, &sha, 3).unwrap()[&7].observed, 3);

        TRACK_INDICES.get().unwrap().lock().unwrap().clear();
        let tampered = String::from_utf8(bytes).unwrap().replace("0.91", "0.92");
        fs::write(&path, tampered).unwrap();
        assert!(index_replay(&path, &sha, 3).is_err());
        fs::remove_dir_all(&root).unwrap();
    }
}
