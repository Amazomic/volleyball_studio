//! Bounded, source-bound frame overlays from the immutable CPU replay.
//! Indexed byte offsets make seeking inexpensive even in long source videos.
#[cfg(test)]
use crate::native_player_tracks::NativeReviewedOwners;
use crate::{
    inference_stream,
    native_player_tracks::{self},
    native_preview_pose, native_rally_preview, preparation, projects,
    storage::{self, safe_file, safe_folder, ApiResult},
    AppState,
};
use axum::{
    extract::{Query, State},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::SystemTime,
};

const MAX_MANIFEST_BYTES: u64 = 16 * 1024;
mod continuity;
const MAX_REPLAY_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_LINE_BYTES: u64 = 512 * 1024 + 1;
const MAX_WINDOW: u64 = 180;
const STRIDE: u64 = 90;
const MAX_OFFSET_INDEX_BYTES: u64 = 4 * 1024 * 1024;
const OFFSET_INDEX_SCHEMA: &str = "volley-verified-overlay-offsets/1";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayQuery {
    project_id: String,
    rally_run_id: String,
    start_frame: u64,
    count: u64,
    raw: Option<bool>,
    pose: Option<bool>,
    ball_confidence: Option<u8>,
    ravel_ball_confidence: Option<u8>,
    player_confidence: Option<u8>,
    touch_confidence: Option<u8>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayPlayerLabelsQuery {
    project_id: String,
    rally_run_id: String,
}

/// A small, independent roster snapshot keeps a player rename or a reviewed
/// Track ID edit from invalidating immutable frame-window caches.
#[cfg(test)]
fn reviewed_player_names(
    reviewed: &native_player_tracks::NativeReviewedOwners,
) -> std::collections::BTreeMap<String, String> {
    reviewed
        .owners
        .iter()
        .filter_map(|(track, owner)| {
            let name = reviewed.roster.get(owner)?.trim();
            (!name.is_empty()).then(|| (track.to_string(), name.to_owned()))
        })
        .collect()
}

pub async fn player_labels(
    State(state): State<Arc<AppState>>,
    Query(query): Query<OverlayPlayerLabelsQuery>,
) -> ApiResult<Value> {
    let result = tokio::task::spawn_blocking(move || {
        let run =
            native_rally_preview::inspect_project(&state, &query.project_id, &query.rally_run_id)?;
        let (source_sha, assignment_revision, names, manual_track_ids) =
            native_player_tracks::display_player_names(
                &state,
                &run.project_id,
                &run.video,
                &run.replay_id,
            )?;
        if source_sha != run.replay_sha {
            return Err(storage::bad_request(
                "Player links do not belong to the selected CPU replay",
            ));
        }
        Ok::<_, storage::ApiError>(json!({
            "schema":"volley-native-preview-player-labels/1",
            "project_id":run.project_id,
            "video":run.video,
            "rally_run_id":run.run_id,
            "replay_id":run.replay_id,
            "source_replay_sha256":run.replay_sha,
            "assignment_revision":assignment_revision,
            "names":names,
            "manual_track_ids":manual_track_ids,
            "identity_source":"manual_and_automatic"
        }))
    })
    .await
    .map_err(|error| storage::internal(format!("Player labels task failure: {error}")))??;
    Ok(Json(result))
}

#[derive(Clone)]
struct ReplayIndex {
    path: PathBuf,
    sha: String,
    size: u64,
    modified: Option<SystemTime>,
    change: Option<(u64, u64, i64, i64)>,
    offsets: Arc<Vec<u64>>,
    frames: u64,
}
static INDICES: OnceLock<Mutex<Vec<ReplayIndex>>> = OnceLock::new();

/// A persistent *derived* offset table: the replay SHA was checked while the
/// table was created. It is reusable only for the exact unchanged source file,
/// verified by a reliable kernel change stamp (ctime+inode / NTFS USN).
/// Never reuse on filesystems that cannot provide such a stamp.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct VerifiedOffsetIndex {
    schema: String,
    replay_sha256: String,
    source_bytes: u64,
    modified_ns: u64,
    source_change: (u64, u64, i64, i64),
    frames: u64,
    offsets_sha256: String,
    offsets: Vec<u64>,
}
fn offset_index_path(path: &Path) -> PathBuf {
    path.with_file_name(".overlay-verified-offsets-v1.json")
}
fn unix_modified_ns(modified: Option<SystemTime>) -> Option<u64> {
    use std::time::UNIX_EPOCH;
    u64::try_from(modified?.duration_since(UNIX_EPOCH).ok()?.as_nanos()).ok()
}
fn offsets_digest(offsets: &[u64]) -> String {
    let mut digest = Sha256::new();
    for offset in offsets {
        digest.update(offset.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}
fn valid_offsets(offsets: &[u64], frames: u64, bytes: u64) -> bool {
    frames > 0
        && offsets.len() == frames.div_ceil(STRIDE) as usize
        && offsets.first() == Some(&0)
        && offsets.iter().all(|value| *value < bytes)
        && offsets.windows(2).all(|pair| pair[0] < pair[1])
}
fn saved_offset_index(
    path: &Path,
    sha: &str,
    frames: u64,
    meta: &fs::Metadata,
    modified: Option<SystemTime>,
    change: Option<(u64, u64, i64, i64)>,
) -> Option<ReplayIndex> {
    let stamp = change?;
    let modified_ns = unix_modified_ns(modified)?;
    let cache_path = offset_index_path(path);
    let cache_meta = fs::symlink_metadata(&cache_path).ok()?;
    if !cache_meta.is_file()
        || cache_meta.file_type().is_symlink()
        || cache_meta.len() > MAX_OFFSET_INDEX_BYTES
    {
        return None;
    }
    let value: VerifiedOffsetIndex = serde_json::from_slice(&fs::read(cache_path).ok()?).ok()?;
    if value.schema != OFFSET_INDEX_SCHEMA
        || value.replay_sha256 != sha
        || value.source_bytes != meta.len()
        || value.modified_ns != modified_ns
        || value.source_change != stamp
        || value.frames != frames
        || !valid_offsets(&value.offsets, frames, meta.len())
        || value.offsets_sha256 != offsets_digest(&value.offsets)
    {
        return None;
    }
    Some(ReplayIndex {
        path: path.to_owned(),
        sha: sha.to_owned(),
        size: meta.len(),
        modified,
        change,
        offsets: Arc::new(value.offsets),
        frames,
    })
}
fn save_offset_index(entry: &ReplayIndex) {
    let (Some(change), Some(modified_ns)) = (entry.change, unix_modified_ns(entry.modified)) else {
        return;
    };
    if !valid_offsets(&entry.offsets, entry.frames, entry.size) {
        return;
    }
    let cache = VerifiedOffsetIndex {
        schema: OFFSET_INDEX_SCHEMA.into(),
        replay_sha256: entry.sha.clone(),
        source_bytes: entry.size,
        modified_ns,
        source_change: change,
        frames: entry.frames,
        offsets_sha256: offsets_digest(&entry.offsets),
        offsets: entry.offsets.as_ref().clone(),
    };
    // A stale derived sidecar is disposable; on Windows rename() cannot
    // overwrite an existing file. Remove only a regular, non-symlink cache.
    // An unwritable cache must never prevent drawing verified source frames.
    let cache_path = offset_index_path(&entry.path);
    if let Ok(meta) = fs::symlink_metadata(&cache_path) {
        if !meta.is_file() || meta.file_type().is_symlink() {
            return;
        }
        if fs::remove_file(&cache_path).is_err() {
            return;
        }
    }
    if let Ok(value) = serde_json::to_value(cache) {
        let _ = storage::write_json_atomic(&cache_path, &value);
    }
}

fn bounded_line(reader: &mut BufReader<File>) -> Result<Option<Vec<u8>>, String> {
    use std::io::Read;
    let mut bytes = Vec::new();
    let len = reader
        .take(MAX_LINE_BYTES)
        .read_until(b'\n', &mut bytes)
        .map_err(|e| format!("Read overlay row: {e}"))?;
    if len == 0 {
        return Ok(None);
    }
    if bytes.last() != Some(&b'\n') {
        return Err("Oversized or unfinished overlay row".into());
    }
    Ok(Some(bytes))
}

fn index_for(path: &Path, sha: &str, frames: u64) -> Result<ReplayIndex, String> {
    let store = INDICES.get_or_init(|| Mutex::new(Vec::new()));
    let mut cache = store.lock().map_err(|_| "Overlay index lock is poisoned")?;
    let meta = fs::metadata(path).map_err(|e| format!("Overlay metadata: {e}"))?;
    if meta.len() > MAX_REPLAY_BYTES {
        return Err("Overlay replay exceeds size limit".into());
    }
    let modified = meta.modified().ok();
    // mtime alone is unsafe on Windows (clock-tick granularity); the platform
    // change stamp detects same-size edits. Without it, never trust the cache.
    let change = volley_media::preparation::file_change_stamp(path, &meta);
    if change.is_some() {
        if let Some(hit) = cache
            .iter()
            .find(|item| {
                item.path == path
                    && item.sha == sha
                    && item.frames == frames
                    && item.size == meta.len()
                    && item.modified == modified
                    && item.change == change
            })
            .cloned()
        {
            return Ok(hit);
        }
    }
    if let Some(hit) = saved_offset_index(path, sha, frames, &meta, modified, change) {
        cache.retain(|row| row.path != path);
        cache.push(hit.clone());
        if cache.len() > 6 {
            cache.remove(0);
        }
        return Ok(hit);
    }
    // First access validates the entire immutable replay once and stores
    // sparse byte offsets. A new mtime/size/sha/change-stamp invalidates it.
    let mut reader = BufReader::new(File::open(path).map_err(|e| format!("Open replay: {e}"))?);
    let mut digest = Sha256::new();
    let mut offsets = Vec::new();
    let mut byte_offset = 0u64;
    for index in 0..frames {
        if index % STRIDE == 0 {
            offsets.push(byte_offset);
        }
        let row = bounded_line(&mut reader)?.ok_or("Replay has fewer rows than its manifest")?;
        byte_offset += row.len() as u64;
        digest.update(&row);
    }
    if bounded_line(&mut reader)?.is_some() {
        return Err("Replay has extra rows".into());
    }
    if format!("{:x}", digest.finalize()) != sha {
        return Err("Replay SHA-256 does not match selected CPU version".into());
    }
    // Do not publish a verified table if the source changed during hashing.
    let after = fs::metadata(path).map_err(|e| format!("Overlay metadata changed: {e}"))?;
    if after.len() != meta.len()
        || after.modified().ok() != modified
        || volley_media::preparation::file_change_stamp(path, &after) != change
        || byte_offset != meta.len()
    {
        return Err("CPU replay changed during overlay index verification".into());
    }
    let entry = ReplayIndex {
        path: path.to_owned(),
        sha: sha.to_owned(),
        size: meta.len(),
        modified,
        change,
        offsets: Arc::new(offsets),
        frames,
    };
    if change.is_some() {
        cache.retain(|row| row.path != path);
        cache.push(entry.clone());
        if cache.len() > 6 {
            cache.remove(0);
        }
        save_offset_index(&entry);
    }
    Ok(entry)
}

fn box_norm(value: &Value) -> Option<[f64; 4]> {
    let array = value.as_array()?;
    if array.len() != 4 {
        return None;
    }
    let bb = [
        array[0].as_f64()?,
        array[1].as_f64()?,
        array[2].as_f64()?,
        array[3].as_f64()?,
    ];
    if bb
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || bb[2] <= bb[0]
        || bb[3] <= bb[1]
    {
        return None;
    }
    Some(bb)
}

// Rendering must never show a visual-only interpolated ID over a measured
// player. Photo/identity review deliberately excludes interpolated evidence,
// so drawing a competing inferred ID creates apparent identity swaps in video.
fn boxes_clash(a: [f64; 4], b: [f64; 4]) -> bool {
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let area = |v: [f64; 4]| (v[2] - v[0]) * (v[3] - v[1]);
    let iou = intersection / (area(a) + area(b) - intersection).max(1e-12);
    let center_distance = (((a[0] + a[2] - b[0] - b[2]) * 0.5).powi(2)
        + ((a[1] + a[3] - b[1] - b[3]) * 0.5).powi(2))
    .sqrt();
    iou > 0.20 || center_distance < 0.045
}

fn overlay_players(frame: &Value) -> Vec<Value> {
    let inferred = frame["interpolated_track_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .collect::<std::collections::BTreeSet<_>>();
    let players = frame["tracked_players"].as_array();
    let observed_boxes = players
        .into_iter()
        .flatten()
        .filter(|p| {
            p["track_id"]
                .as_u64()
                .is_some_and(|id| !inferred.contains(&id))
        })
        .filter_map(|p| box_norm(&p["detection"]["bbox_xyxy_norm"]))
        .collect::<Vec<_>>();
    players
        .into_iter()
        .flatten()
        .take(128)
        .filter_map(|p| {
            let id = p["track_id"].as_u64().filter(|id| *id > 0)?;
            if p["is_verified_athlete"] != false {
                return None;
            }
            let bb = box_norm(&p["detection"]["bbox_xyxy_norm"])?;
            if inferred.contains(&id)
                && observed_boxes
                    .iter()
                    .any(|measured| boxes_clash(bb, *measured))
            {
                return None;
            }
            let mut player = json!({"id":id,"box":bb,"interpolated":inferred.contains(&id)});
            if p["pose"]
                .as_array()
                .is_some_and(|points| points.len() == 17)
            {
                player["pose"] = p["pose"].clone();
                player["pose_interpolated"] = json!(p["pose_interpolated"] == true);
            }
            Some(player)
        })
        .collect()
}

fn raw_frame(
    frame: &Value,
    ball_threshold: u8,
    ravel_ball_threshold: u8,
    player_threshold: u8,
    touch_threshold: u8,
) -> Value {
    let above = |value: &Value, threshold: u8| {
        value.as_f64().is_some_and(|score| {
            score.is_finite()
                && (0.0..=1.0).contains(&score)
                && score >= f64::from(threshold) / 100.0
        })
    };
    let balls = frame["ball_candidates"]
        .as_array()
        .into_iter()
        .flatten()
        .take(2)
        .filter_map(|item| {
            let detection = &item["detection"];
            let source = item["source_model"].as_str()?;
            if !matches!(source, "ravel" | "vballnet")
                || !above(
                    &detection["score"],
                    if source == "ravel" {
                        ravel_ball_threshold
                    } else {
                        ball_threshold
                    },
                )
            {
                return None;
            }
            let b = box_norm(&detection["bbox_xyxy_norm"])?;
            Some(json!({"point":[(b[0]+b[2])*0.5,(b[1]+b[3])*0.5],"source":source,"score":detection["score"]}))
        })
        .collect::<Vec<_>>();
    let mut players = Vec::new();
    let mut touches = Vec::new();
    for item in frame
        .get("player_candidates")
        .unwrap_or(&frame["raw_player_candidates"])
        .as_array()
        .into_iter()
        .flatten()
        .take(128)
    {
        let Some(b) = box_norm(&item["bbox_xyxy_norm"]) else {
            continue;
        };
        if above(&item["score"], player_threshold) {
            players.push(json!({"box":b,"score":item["score"],"touch_probability":item["touch_probability"],"query_index":item["query_index"]}));
        }
        if above(&item["touch_probability"], touch_threshold) {
            touches.push(json!({"box":b}));
        }
    }
    json!({"balls":balls,"players":players,"touches":touches})
}

#[cfg(test)]
fn repaired_ball_observations(
    samples: &[Option<(&str, [f64; 2])>],
    width: u32,
    height: u32,
    player_boxes: &[Vec<[f64; 4]>],
) -> (Vec<Option<[f64; 2]>>, Vec<bool>) {
    let vballnet = samples
        .iter()
        .map(|sample| {
            sample
                .filter(|(source, _)| *source == "vballnet")
                .map(|(_, point)| point)
        })
        .collect::<Vec<_>>();
    // Render-only, exactly the same false-location guard as CPU Touch.
    // No synthetic point can be used to prove a ball/player contact.
    volley_core::touch_pipeline::repair_vballnet_trajectory_with_players(
        &vballnet,
        player_boxes,
        width,
        height,
    )
}

#[cfg(test)]
fn gated_ball_observations(samples: &[Option<(&str, [f64; 2])>]) -> Vec<Option<[f64; 2]>> {
    repaired_ball_observations(samples, 1280, 720, &vec![Vec::new(); samples.len()]).0
}

fn extract_frames(
    path: &Path,
    sha: &str,
    total: u64,
    start: u64,
    end: u64,
    scope: &native_rally_preview::RallyWindows,
    width: u32,
    height: u32,
    min_ball_score: f64,
    raw_thresholds: Option<(u8, u8, u8, u8)>,
    ball_scope: Option<(&[u64], f64)>,
) -> Result<Vec<Value>, String> {
    let index = index_for(path, sha, total)?;
    let starts = scope
        .intervals
        .iter()
        .map(|(start, _)| *start)
        .filter(|start| *start > 0)
        .fold(vec![0], |mut starts, start| {
            starts.push(start);
            starts
        });
    let (rally_starts, tracking_score) = ball_scope.unwrap_or((&starts, min_ball_score));
    let ball_mask = crate::cpu_retrack::ball_track::read(
        path,
        sha,
        total,
        width,
        height,
        tracking_score,
        rally_starts,
    )?;
    // Teleport islands can last ten frames; ballistic fit needs one extra
    // real observation on each side. Preserve this halo for every page.
    let read_start = start.saturating_sub(30);
    let read_end = end.saturating_add(30).min(total);
    let anchor = read_start / STRIDE * STRIDE;
    let offset = index.offsets[(read_start / STRIDE) as usize];
    let mut file = File::open(path).map_err(|e| format!("Open indexed replay: {e}"))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("Seek replay: {e}"))?;
    let mut reader = BufReader::new(file);
    let mut output = Vec::with_capacity((end - start) as usize);
    for i in anchor..read_end {
        let bytes = bounded_line(&mut reader)?.ok_or("Indexed replay is truncated")?;
        if i < read_start {
            continue;
        }
        let frame: Value =
            serde_json::from_slice(&bytes).map_err(|_| "Malformed indexed overlay frame")?;
        if frame["frame_index"].as_u64() != Some(i)
            || frame["track_ids_provisional"] != true
            || frame["confirmed_athlete_identities"] != false
            || frame["touches_confirmed"] != false
        {
            return Err("Overlay frame provenance mismatch".into());
        }
        let ball = frame["selected_ball"]["detection"]["score"]
            .as_f64()
            .filter(|score| {
                score.is_finite() && *score >= min_ball_score && ball_mask.accepted[i as usize]
            })
            .and_then(|_| box_norm(&frame["selected_ball"]["detection"]["bbox_xyxy_norm"]))
            .map(|bb| json!([(bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5]));
        let in_rally = scope.contains(i);
        let players = if in_rally {
            overlay_players(&frame)
        } else {
            Vec::new()
        };
        let raw = raw_thresholds
            .map(|(ball, ravel, player, touch)| raw_frame(&frame, ball, ravel, player, touch));
        output.push(json!({
            "frame":i,"ball":ball,"ball_source":frame["selected_ball"]["source_model"],
            "raw":raw,"in_rally":in_rally,"players":players,
            "ball_candidates_count":ball_mask.candidate_counts[i as usize],
            "ball_candidate_accepted":ball_mask.accepted[i as usize],
            "ball_candidate_rejected":ball_mask.rejected[i as usize],
            "leading_ball_track_id":ball_mask.track_ids[i as usize],
        }));
    }
    let samples = output
        .iter()
        .map(|row| {
            let source = row["ball_source"].as_str()?;
            let ball = row["ball"].as_array()?;
            Some((source, [ball[0].as_f64()?, ball[1].as_f64()?]))
        })
        .collect::<Vec<_>>();
    let player_boxes = output
        .iter()
        .map(|row| {
            row["players"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|player| box_norm(&player["box"]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let accepted = samples
        .iter()
        .map(|sample| {
            sample
                .filter(|(source, _)| *source == "vballnet")
                .map(|(_, point)| point)
        })
        .collect::<Vec<_>>();
    let track_ids = &ball_mask.track_ids[read_start as usize..read_end as usize];
    let mut gated = accepted.clone();
    let mut synthetic = vec![false; accepted.len()];
    for &(rally_start, rally_end) in &scope.intervals {
        let begin = rally_start.max(read_start);
        let end = rally_end.saturating_add(1).min(read_end);
        if begin >= end {
            continue;
        }
        let begin = (begin - read_start) as usize;
        let end = (end - read_start) as usize;
        let (points, inferred) = volley_core::touch_pipeline::repair_accepted_vballnet_trajectory(
            &accepted[begin..end],
            &player_boxes[begin..end],
            &track_ids[begin..end],
            width,
            height,
        );
        gated[begin..end].copy_from_slice(&points);
        synthetic[begin..end].copy_from_slice(&inferred);
    }
    for ((row, ball), inferred) in output.iter_mut().zip(gated).zip(synthetic) {
        // Reject outliers even if their model confidence is high; distinguish
        // every inferred position from an actual VBallNet detection.
        let origin = if ball.is_none() {
            None
        } else if inferred {
            Some("vballnet_interpolated")
        } else {
            Some("vballnet")
        };
        row["ball_origin"] = json!(origin);
        row["ball_interpolated"] = json!(inferred);
        row["ball"] = json!(ball);
        row.as_object_mut().unwrap().remove("ball_source");
    }
    Ok(output
        .into_iter()
        .filter(|row| {
            row["frame"]
                .as_u64()
                .is_some_and(|frame| frame >= start && frame < end)
        })
        .collect())
}

pub async fn get(
    State(state): State<Arc<AppState>>,
    Query(query): Query<OverlayQuery>,
) -> ApiResult<Value> {
    // inspect_project runs ffprobe and SHA verification; this handler is
    // polled continuously during playback, so keep it off the tokio workers.
    let inspect_state = state.clone();
    let project_id = query.project_id.clone();
    let run_id = query.rally_run_id.clone();
    let run = tokio::task::spawn_blocking(move || {
        native_rally_preview::inspect_project(&inspect_state, &project_id, &run_id)
    })
    .await
    .map_err(|e| storage::internal(format!("Overlay inspect task: {e}")))??;
    if query.count == 0 || query.count > MAX_WINDOW || query.start_frame >= run.frames {
        return Err(storage::bad_request("Overlay frame window is out of range"));
    }
    let thresholds = (
        query.ball_confidence.unwrap_or(50),
        query
            .ravel_ball_confidence
            .or(query.ball_confidence)
            .unwrap_or(50),
        query.player_confidence.unwrap_or(50),
        query.touch_confidence.unwrap_or(50),
    );
    if thresholds.0 > 100 || thresholds.1 > 100 || thresholds.2 > 100 || thresholds.3 > 100 {
        return Err(storage::bad_request("Confidence must be between 0 and 100"));
    }
    let raw_thresholds = query.raw.unwrap_or(false).then_some(thresholds);
    let end = query
        .start_frame
        .saturating_add(query.count)
        .min(run.frames);
    let replay_dir = projects::project_root(&state)
        .join(&query.project_id)
        .join("native_cpu_replays")
        .join(&run.replay_id);
    safe_folder(&replay_dir).map_err(storage::bad_request)?;
    let manifest: Value = serde_json::from_slice(
        &fs::read(
            safe_file(&replay_dir, "manifest.json", MAX_MANIFEST_BYTES)
                .map_err(storage::bad_request)?,
        )
        .map_err(|e| storage::bad_request(format!("Read replay manifest: {e}")))?,
    )
    .map_err(|e| storage::bad_request(format!("Invalid replay manifest: {e}")))?;
    if manifest["schema"] != "volley-native-cpu-tracking/1"
        || manifest["state"] != "provisional_tracklets_rebuilt"
        || manifest["project_id"] != query.project_id
        || manifest["video"] != run.video
        || manifest["replay_id"] != run.replay_id
        || manifest["file"] != "tracked_frames.jsonl"
        || manifest["fps"] != 30
        || manifest["frames"].as_u64() != Some(run.frames)
        || manifest["sha256"].as_str() != Some(run.replay_sha.as_str())
        || manifest["track_ids_provisional"] != true
        || manifest["athlete_identities_confirmed"] != false
        || manifest["touches_confirmed"] != false
        || manifest["cpu_processing_complete"] != false
    {
        return Err(storage::bad_request(
            "Overlay replay does not match postprocessing",
        ));
    }
    let replay_path = safe_file(&replay_dir, "tracked_frames.jsonl", MAX_REPLAY_BYTES)
        .map_err(storage::bad_request)?;
    let replay_sha = run.replay_sha.clone();
    let total = run.frames;
    let interpolate_players = manifest["interpolate_players"].as_bool().unwrap_or(true);
    let interpolate_pose = manifest["interpolate_pose"].as_bool().unwrap_or(true);
    let start = query.start_frame;
    let preview_run = run.clone();
    let fusion_for_pose = if query.pose.unwrap_or(false) {
        Some(
            manifest["source_fusion_id"]
                .as_str()
                .filter(|id| volley_core::contracts::job_id(id))
                .ok_or_else(|| storage::bad_request("CPU replay has no valid GPU fusion for pose"))?
                .to_owned(),
        )
    } else {
        None
    };
    let pose_root = projects::project_root(&state).join(&query.project_id);
    let continuity_state = state.clone();
    let (frames, poses_available) = tokio::task::spawn_blocking(move || {
        let windows = native_rally_preview::verified_track_windows(&preview_run)
            .map_err(|e| format!("Unverified rally scope: {e}"))?;
        let rally_dir = preview_run
            .rally_file
            .parent()
            .ok_or("Missing rally directory")?;
        let saved: Value = serde_json::from_slice(
            &fs::read(safe_file(rally_dir, "manifest.json", MAX_MANIFEST_BYTES)?)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|_| "Invalid rally ball policy manifest")?;
        let ball_scope = crate::cpu_retrack::ball_track::saved_scope(&saved)?;
        let (context_start, context_end) = continuity::context(start, end, &windows.intervals);
        let mut frames = extract_frames(
            &replay_path,
            &replay_sha,
            total,
            context_start,
            context_end,
            &windows,
            preview_run.width,
            preview_run.height,
            preview_run.min_ball_score,
            raw_thresholds,
            ball_scope
                .as_ref()
                .map(|(starts, score)| (starts.as_slice(), *score)),
        )?;
        let reviewed = native_player_tracks::reviewed_owners(
            &continuity_state,
            &preview_run.project_id,
            &preview_run.video,
            &preview_run.replay_id,
        )
        .map_err(|_| "Confirmed player links unavailable for interpolation")?;
        if reviewed.source_sha != preview_run.replay_sha {
            return Err("Confirmed player links belong to another CPU replay".into());
        }
        let poses_available = if let Some(fusion) = fusion_for_pose {
            native_preview_pose::enrich_exact(
                &pose_root,
                &fusion,
                context_start,
                context_end,
                &mut frames,
            )?;
            native_preview_pose::processed_window_policy(
                &pose_root,
                &fusion,
                total,
                context_start,
                context_end,
                &mut frames,
                Some(interpolate_pose),
            )?
        } else {
            false
        };
        continuity::fill(
            &mut frames,
            &windows.intervals,
            &reviewed.owners,
            interpolate_players,
        );
        if poses_available {
            continuity::append_poses(&mut frames);
        }
        frames.retain(|row| row["frame"].as_u64().is_some_and(|f| f >= start && f < end));
        Ok::<_, String>((frames, poses_available))
    })
    .await
    .map_err(|e| storage::internal(format!("Overlay task failure: {e}")))?
    .map_err(storage::bad_request)?;
    Ok(Json(json!({
        "schema":"volley-native-preview-overlay/1",
        "continuity_policy":"cpu_rally_linear_v1",
        "ball_policy":"vballnet_leading_ball_rally_v1",
        "expected_balls":1,
        "fps":30,
        "project_id":run.project_id,"video":run.video,
        "rally_run_id":run.run_id,"replay_sha256":run.replay_sha,
        "width":run.width,"height":run.height,
        "start_frame":start,"end_frame":end,
        "frames":frames,
        "poses_available":poses_available,
        "touches_verified":false,"athlete_identities_verified":false,
    })))
}

/// Source-verified single frame used by the human training library.
/// Reuses the exact same index and pose provenance as the user preview.
pub(crate) async fn training_frame(
    state: Arc<AppState>,
    project_id: &str,
    rally_run_id: &str,
    frame: u64,
) -> Result<Value, storage::ApiError> {
    let Json(overlay) = get(
        State(state),
        Query(OverlayQuery {
            project_id: project_id.to_owned(),
            rally_run_id: rally_run_id.to_owned(),
            start_frame: frame.saturating_sub(6),
            count: 13,
            raw: None,
            pose: Some(true),
            ball_confidence: None,
            ravel_ball_confidence: None,
            player_confidence: None,
            touch_confidence: None,
        }),
    )
    .await?;
    Ok(overlay)
}

/// Independent immutable GPU view: the selected fusion is authoritative, not any CPU replay.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuOverlayQuery {
    project_id: String,
    video: String,
    fusion_id: String,
    start_frame: u64,
    count: u64,
}

fn gpu_raw_frame(frame: &Value, expected: u64) -> Result<Value, String> {
    if frame["frame_index"].as_u64() != Some(expected)
        || frame["player_track_ids_confirmed"] != false
        || frame["touches_confirmed"] != false
    {
        return Err("GPU frame index or evidence provenance mismatch".into());
    }
    let first = expected - expected % 9;
    let player_rows = frame["player_candidates"]
        .as_array()
        .ok_or("Missing GPU player candidates")?;
    let ball_rows = frame["ball_candidates"]
        .as_array()
        .ok_or("Missing GPU ball candidates")?;
    if player_rows.len() > 33 || ball_rows.len() > 2 {
        return Err("GPU frame exceeds model output bounds".into());
    }
    let mut players = Vec::new();
    for row in player_rows {
        let candidate: volley_core::fusion::Candidate =
            serde_json::from_value(row.clone()).map_err(|_| "Invalid GPU player candidate")?;
        candidate.validate(first, "players")?;
        if candidate.frame_index != expected || candidate.class_name != "player" {
            return Err("GPU player candidate is out of frame".into());
        }
        players.push(json!({"box":candidate.bbox_xyxy_norm,
            "score":candidate.score,"touch_probability":candidate.touch_probability,
            "query_index":candidate.query_index}));
    }
    let mut balls = Vec::new();
    for row in ball_rows {
        let source = row["source_model"]
            .as_str()
            .ok_or("Missing GPU ball model")?;
        let model = match source {
            "vballnet" => "ball",
            "ravel" => "players",
            _ => return Err("Unrecognized GPU ball model".into()),
        };
        let candidate: volley_core::fusion::Candidate =
            serde_json::from_value(row["detection"].clone())
                .map_err(|_| "Invalid GPU ball candidate")?;
        candidate.validate(first, model)?;
        if candidate.frame_index != expected || candidate.class_name != "ball" {
            return Err("GPU ball candidate is out of frame".into());
        }
        let [x1, y1, x2, y2] = candidate.bbox_xyxy_norm;
        balls.push(json!({"source":source,"score":candidate.score,
            "point":[(x1+x2)*0.5,(y1+y2)*0.5]}));
    }
    Ok(json!({"balls":balls,"players":players,"poses":[]}))
}

fn extract_gpu_frames(
    path: &Path,
    sha: &str,
    total: u64,
    start: u64,
    end: u64,
) -> Result<Vec<Value>, String> {
    let index = index_for(path, sha, total)?;
    let anchor = start / STRIDE * STRIDE;
    let offset = index.offsets[(start / STRIDE) as usize];
    let mut file = File::open(path).map_err(|e| format!("Open GPU frames: {e}"))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("Seek GPU frames: {e}"))?;
    let mut reader = BufReader::new(file);
    let mut output = Vec::with_capacity((end - start) as usize);
    for i in anchor..end {
        let line = bounded_line(&mut reader)?.ok_or("GPU frame evidence truncated")?;
        if i < start {
            continue;
        }
        let row: Value =
            serde_json::from_slice(&line).map_err(|_| "Malformed GPU frame evidence")?;
        output.push(json!({"frame":i,"raw":gpu_raw_frame(&row,i)?}));
    }
    Ok(output)
}

/// Resolve the player detector from the original immutable GPU evidence,
/// never from the environment of the currently running Studio.
fn gpu_person_detector(project_root: &Path, fusion: &Value) -> &'static str {
    let Some(source) = fusion["sources"]["person_pose"].as_object() else {
        return "unknown";
    };
    let Some(job_id) = source.get("job_id").and_then(Value::as_str) else {
        return "unknown";
    };
    let Some(sha) = source.get("sha256").and_then(Value::as_str) else {
        return "unknown";
    };
    if !volley_core::contracts::job_id(job_id) || !volley_core::contracts::sha256(sha) {
        return "unknown";
    }
    let runs = project_root.join("native_gpu_runs");
    let Ok(()) = safe_folder(&runs) else {
        return "unknown";
    };
    let folder = runs.join(job_id);
    let Ok(()) = safe_folder(&folder) else {
        return "unknown";
    };
    let Ok(file) = safe_file(&folder, "manifest.json", MAX_MANIFEST_BYTES) else {
        return "unknown";
    };
    let Ok(bytes) = fs::read(file) else {
        return "unknown";
    };
    let Ok(model) = serde_json::from_slice::<Value>(&bytes) else {
        return "unknown";
    };
    if model["schema"] != "volley-native-gpu-model-detections/1"
        || model["status"] != "model_raw_detections_complete"
        || model["model"] != "players"
        || model["job_id"] != job_id
        || model["project_id"] != fusion["project_id"]
        || model["video"] != fusion["video"]
        || model["sha256"] != sha
        || model["source_identity"] != fusion["source_identity"]
    {
        return "unknown";
    }
    match model["person_pipeline"].as_str() {
        Some(pipeline) if pipeline.starts_with("rfdetr_medium_") => "rfdetr-medium",
        // Read-only historical evidence remains viewable after retiring
        // its detector; do not label it as a current RF-DETR run.
        Some(pipeline) if pipeline.starts_with("rtmdet_nano_") => "legacy",
        None => "legacy",
        _ => "unknown",
    }
}
fn inspect_gpu_source(
    state: &AppState,
    query: &GpuOverlayQuery,
) -> Result<(PathBuf, String, u64, u32, u32, PathBuf, &'static str), storage::ApiError> {
    storage::project_id(&query.project_id)?;
    storage::video_name(&query.video)?;
    if !volley_core::contracts::job_id(&query.fusion_id) {
        return Err(storage::bad_request("Invalid GPU fusion ID"));
    }
    let project = projects::existing_project(state, &query.project_id)?;
    if !project["videos"].as_array().is_some_and(|videos| {
        videos
            .iter()
            .any(|v| v.as_str() == Some(query.video.as_str()))
    }) {
        return Err(storage::bad_request("GPU video does not belong to project"));
    }
    let plan = preparation::plan_for_run(
        state,
        &query.project_id,
        &query.video,
        "native_gpu_fusions",
        &query.fusion_id,
    )?;
    let prepared = plan
        .ready_path()
        .ok_or_else(|| storage::bad_request("Prepared GPU video is unavailable"))?;
    let (size, mtime) = inference_stream::source_stamp(prepared).map_err(storage::bad_request)?;
    let dimensions = volley_media::probe(prepared).map_err(storage::bad_request)?;
    let root = projects::project_root(state).join(&query.project_id);
    safe_folder(&root).map_err(storage::bad_request)?;
    let fusion_root = root.join("native_gpu_fusions");
    safe_folder(&fusion_root).map_err(storage::bad_request)?;
    let folder = fusion_root.join(&query.fusion_id);
    safe_folder(&folder).map_err(storage::bad_request)?;
    let manifest: Value = serde_json::from_slice(
        &fs::read(
            safe_file(&folder, "manifest.json", MAX_MANIFEST_BYTES)
                .map_err(storage::bad_request)?,
        )
        .map_err(|e| storage::bad_request(format!("Read GPU fusion: {e}")))?,
    )
    .map_err(|_| storage::bad_request("Invalid GPU fusion manifest"))?;
    if manifest["schema"] != "volley-native-gpu-fusion/1"
        || manifest["state"] != "raw_candidates_fused"
        || manifest["project_id"] != query.project_id
        || manifest["fusion_id"] != query.fusion_id
        || manifest["video"] != query.video
        || manifest["fps"] != 30
        || manifest["file"] != "frames.jsonl"
        || manifest["source_identity"] != json!({"bytes":size,"mtime_ns":mtime})
        || manifest["player_track_ids_confirmed"] != false
        || manifest["touches_confirmed"] != false
        || manifest["cpu_processing_complete"] != false
    {
        return Err(storage::bad_request("GPU source provenance mismatch"));
    }
    let total = manifest["emitted_frames"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= 48 * 3600 * 30 && *n % 9 == 0)
        .ok_or_else(|| storage::bad_request("Invalid GPU fusion frame count"))?;
    if manifest["windows"].as_u64() != Some(total / 9) {
        return Err(storage::bad_request("GPU fusion window count mismatch"));
    }
    let sha = manifest["sha256"]
        .as_str()
        .filter(|s| volley_core::contracts::sha256(s))
        .ok_or_else(|| storage::bad_request("Invalid GPU fusion SHA"))?
        .to_owned();
    let path =
        safe_file(&folder, "frames.jsonl", MAX_REPLAY_BYTES).map_err(storage::bad_request)?;
    let person_detector = gpu_person_detector(&root, &manifest);
    Ok((
        path,
        sha,
        total,
        dimensions.width,
        dimensions.height,
        root,
        person_detector,
    ))
}

pub async fn gpu(
    State(state): State<Arc<AppState>>,
    Query(query): Query<GpuOverlayQuery>,
) -> ApiResult<Value> {
    if query.count == 0 || query.count > MAX_WINDOW {
        return Err(storage::bad_request("Invalid GPU overlay window"));
    }
    let (path, sha, total, width, height, root, person_detector) = tokio::task::spawn_blocking({
        let state = state.clone();
        let query = GpuOverlayQuery {
            project_id: query.project_id.clone(),
            video: query.video.clone(),
            fusion_id: query.fusion_id.clone(),
            start_frame: query.start_frame,
            count: query.count,
        };
        move || inspect_gpu_source(&state, &query)
    })
    .await
    .map_err(|e| storage::internal(format!("GPU source inspect task: {e}")))??;
    if query.start_frame >= total {
        return Err(storage::bad_request("GPU overlay frame is beyond source"));
    }
    let start = query.start_frame;
    let end = start.saturating_add(query.count).min(total);
    let fusion = query.fusion_id.clone();
    let response_sha = sha.clone();
    let (frames, pose_run_id) = tokio::task::spawn_blocking(move || {
        let mut frames = extract_gpu_frames(&path, &sha, total, start, end)?;
        let pose_run_id =
            native_preview_pose::raw_window(&root, &fusion, total, start, end, &mut frames)?;
        Ok::<_, String>((frames, pose_run_id))
    })
    .await
    .map_err(|e| storage::internal(format!("GPU overlay task: {e}")))?
    .map_err(storage::bad_request)?;
    Ok(Json(json!({"schema":"volley-native-gpu-overlay/1",
        "project_id":query.project_id,"video":query.video,"fusion_id":query.fusion_id,
        "source_fusion_sha256":response_sha,"fps":30,
        "width":width,"height":height,"total_frames":total,"person_detector":person_detector,
        "start_frame":start,"end_frame":end,"frames":frames,
        "poses_available":pose_run_id.is_some(),"pose_run_id":pose_run_id,
        "touches_verified":false,"athlete_identities_verified":false})))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_person_detector_uses_saved_gpu_model_not_current_environment() {
        let root = std::env::temp_dir().join(format!("volley-gpu-model-{}", uuid::Uuid::new_v4()));
        let job_id = "0123456789abcdef0123456789abcdef";
        let folder = root.join("native_gpu_runs").join(job_id);
        fs::create_dir_all(&folder).unwrap();
        let sha = "a".repeat(64);
        let source = json!({"schema":"volley-native-gpu-model-detections/1",
            "status":"model_raw_detections_complete","model":"players",
            "job_id":job_id,"project_id":"0123456789ab","video":"game.mp4",
            "sha256":sha,"source_identity":{"bytes":123,"mtime_ns":456},
            "person_pipeline":"rfdetr_medium_18250b5a_rtmpose_m_detector_boxes_3hz_v1"});
        let fusion = json!({"project_id":"0123456789ab","video":"game.mp4",
            "sources":{"person_pose":{"job_id":job_id,"sha256":sha}},
            "source_identity":{"bytes":123,"mtime_ns":456}});
        let save = |value: &Value| {
            fs::write(
                folder.join("manifest.json"),
                serde_json::to_vec(value).unwrap(),
            )
            .unwrap()
        };
        save(&source);
        assert_eq!(gpu_person_detector(&root, &fusion), "rfdetr-medium");
        let mut legacy = source.clone();
        legacy["person_pipeline"] = json!("rtmdet_nano_tiled_rtmpose_m_coco17_3hz_v5");
        save(&legacy);
        assert_eq!(gpu_person_detector(&root, &fusion), "legacy");
        legacy.as_object_mut().unwrap().remove("person_pipeline");
        save(&legacy);
        assert_eq!(gpu_person_detector(&root, &fusion), "legacy");
        let mut changed = fusion.clone();
        changed["sources"]["person_pose"]["sha256"] = json!("b".repeat(64));
        assert_eq!(gpu_person_detector(&root, &changed), "unknown");
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn raw_gpu_window_is_independent_of_cpu_and_rejects_changed_sha() {
        let root = std::env::temp_dir().join(format!("volley-raw-gpu-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("frames.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..9u64 {
            let ball = json!({"source_model":"vballnet","detection":{
                "frame_index":frame,"class_name":"ball","score":0.61,
                "bbox_xyxy_norm":[0.1,0.1,0.2,0.2]}});
            let player = json!({"frame_index":frame,"class_name":"player",
                "score":0.72,"bbox_xyxy_norm":[0.2,0.2,0.4,0.8],
                "query_index":5,"touch_probability":0.83});
            let row = json!({"frame_index":frame,
                "player_candidates":[player],"ball_candidates":[ball],
                "player_track_ids_confirmed":false,"touches_confirmed":false});
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        fs::write(&path, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let window = extract_gpu_frames(&path, &sha, 9, 3, 6).unwrap();
        assert_eq!(window.len(), 3);
        assert_eq!(window[0]["frame"], 3);
        assert_eq!(window[0]["raw"]["players"][0]["score"], 0.72);
        assert_eq!(window[0]["raw"]["players"][0]["touch_probability"], 0.83);
        assert_eq!(window[0]["raw"]["balls"][0]["score"], 0.61);
        assert!(window[0]["raw"]["poses"].as_array().unwrap().is_empty());
        assert!(window[0].get("track_id").is_none());
        assert!(extract_gpu_frames(&path, &"0".repeat(64), 9, 3, 6).is_err());
        let mut changed = bytes;
        changed[16] ^= 1;
        fs::write(&path, changed).unwrap();
        assert!(extract_gpu_frames(&path, &sha, 9, 3, 6).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn raw_gpu_fusion_refuses_fabricated_track_provenance() {
        let row = json!({"frame_index":0,"player_track_ids_confirmed":true,
            "touches_confirmed":false,"player_candidates":[],"ball_candidates":[]});
        assert!(gpu_raw_frame(&row, 0).is_err());
        let row = json!({"frame_index":0,"player_track_ids_confirmed":false,
            "touches_confirmed":false,
            "player_candidates":[{"frame_index":0,"class_name":"player",
                "score":0.9,"bbox_xyxy_norm":[0.2,0.2,0.4,0.8],
                "query_index":5,"touch_probability":0.85}],
            "ball_candidates":[]});
        let result = gpu_raw_frame(&row, 0).unwrap();
        assert_eq!(result["players"][0]["touch_probability"], 0.85);
        assert!(
            result["touches"].is_null(),
            "A model contact probability is not a CPU Touch"
        );
    }
    #[test]
    fn persisted_index_restores_and_rejects_corruption_or_changed_source() {
        let root =
            std::env::temp_dir().join(format!("volley-overlay-persist-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("tracked_frames.jsonl");
        let mut content = Vec::new();
        for frame in 0..390u64 {
            content.extend(serde_json::to_vec(&json!({"frame_index":frame})).unwrap());
            content.push(b'\n');
        }
        fs::write(&source, &content).unwrap();
        let sha = format!("{:x}", Sha256::digest(&content));
        let initial = index_for(&source, &sha, 390).unwrap();
        let meta = fs::metadata(&source).unwrap();
        let modified = meta.modified().ok();
        let change = volley_media::preparation::file_change_stamp(&source, &meta);
        let disk = offset_index_path(&source);
        if change.is_some() && unix_modified_ns(modified).is_some() {
            assert!(
                disk.exists(),
                "Verified index should survive server restart"
            );
            let restored = saved_offset_index(&source, &sha, 390, &meta, modified, change).unwrap();
            assert_eq!(*restored.offsets, *initial.offsets);
            let mut tampered: Value = serde_json::from_slice(&fs::read(&disk).unwrap()).unwrap();
            tampered["offsets"][1] = json!(42);
            fs::write(&disk, serde_json::to_vec(&tampered).unwrap()).unwrap();
            assert!(
                saved_offset_index(&source, &sha, 390, &meta, modified, change).is_none(),
                "A corrupted offset table cannot be trusted"
            );
            // Mimic a process restart, so the in-memory verified copy is gone.
            INDICES
                .get()
                .unwrap()
                .lock()
                .unwrap()
                .retain(|row| row.path != source);
            let repaired = index_for(&source, &sha, 390).unwrap();
            assert_eq!(*repaired.offsets, *initial.offsets);
            assert!(
                saved_offset_index(&source, &sha, 390, &meta, modified, change).is_some(),
                "Full source re-verification repairs the derived cache"
            );
        }
        let mut changed = content;
        changed[15] ^= 1;
        fs::write(&source, changed).unwrap();
        // No reused in-memory or on-disk SHA after source mutation.
        assert!(index_for(&source, &sha, 390).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn concurrent_overlay_requests_share_the_verified_index() {
        let root =
            std::env::temp_dir().join(format!("volley-overlay-index-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let input = root.join("tracked_frames.jsonl");
        let data = b"{}\n".repeat(100_000);
        fs::write(&input, &data).unwrap();
        let sha = format!("{:x}", Sha256::digest(&data));
        let barrier = Arc::new(std::sync::Barrier::new(4));
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let path = input.clone();
                let sha = sha.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    index_for(&path, &sha, 100_000).unwrap()
                })
            })
            .collect();
        let indices: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert!(indices
            .iter()
            .all(|index| index.offsets.as_ptr() == indices[0].offsets.as_ptr()));
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn raw_fusion_players_are_not_dependent_on_cpu_replay() {
        let frame = json!({"player_candidates":[{"score":0.6,"touch_probability":0.9,"bbox_xyxy_norm":[0.2,0.2,0.4,0.8]}]});
        let raw = raw_frame(&frame, 50, 50, 60, 80);
        assert_eq!(raw["players"].as_array().unwrap().len(), 1);
        assert_eq!(raw["touches"].as_array().unwrap().len(), 1);
    }
    #[test]
    fn raw_thresholds_filter_ball_player_and_touch_independently() {
        let frame = json!({
            "ball_candidates":[{"source_model":"vballnet","detection":{"score":0.7,"bbox_xyxy_norm":[0.1,0.1,0.2,0.2]}},{"source_model":"ravel","detection":{"score":0.8,"bbox_xyxy_norm":[0.3,0.3,0.4,0.4]}}],
            "raw_player_candidates":[{"score":0.6,"touch_probability":0.9,"bbox_xyxy_norm":[0.2,0.2,0.4,0.8]}]
        });
        let raw = raw_frame(&frame, 80, 90, 70, 80);
        assert_eq!(raw["balls"].as_array().unwrap().len(), 0);
        assert_eq!(raw["players"].as_array().unwrap().len(), 0);
        assert_eq!(raw["touches"].as_array().unwrap().len(), 1);
        let raw = raw_frame(&frame, 70, 90, 60, 95);
        assert_eq!(raw["balls"].as_array().unwrap().len(), 1);
        assert_eq!(raw["players"].as_array().unwrap().len(), 1);
        assert_eq!(raw["touches"].as_array().unwrap().len(), 0);
        let raw = raw_frame(&frame, 90, 80, 60, 95);
        assert_eq!(raw["balls"].as_array().unwrap().len(), 1);
        assert_eq!(raw["balls"][0]["source"], "ravel");
    }
    #[test]
    fn processed_overlay_uses_only_vball_observations() {
        let samples = vec![
            Some(("ravel", [0.1, 0.1])),
            Some(("vballnet", [0.0, 0.2])),
            Some(("vballnet", [0.1, 0.2])),
            Some(("vballnet", [0.2, 0.2])),
            Some(("ravel", [0.3, 0.2])),
            Some(("ravel", [0.9, 0.9])),
            Some(("vballnet", [0.5, 0.2])),
            Some(("ravel", [0.6, 0.2])),
        ];
        let result = gated_ball_observations(&samples);
        let expected = vec![
            None,
            Some([0.0, 0.2]),
            Some([0.1, 0.2]),
            Some([0.2, 0.2]),
            Some([0.3, 0.2]),
            Some([0.4, 0.2]),
            Some([0.5, 0.2]),
            None,
        ];
        for (actual, expected) in result.iter().zip(expected) {
            match (actual, expected) {
                (Some(actual), Some(expected)) => assert!(
                    (actual[0] - expected[0]).abs() < 1e-12
                        && (actual[1] - expected[1]).abs() < 1e-12
                ),
                (None, None) => (),
                _ => panic!("RAVEL cannot become an observed processed ball"),
            }
        }
    }

    #[test]
    fn seeking_inside_remote_ball_streak_keeps_original_leading_ball() {
        let root = std::env::temp_dir().join(format!("leading-ball-page-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("tracked_frames.jsonl");
        let mut bytes = Vec::new();
        for frame in 0..500_u64 {
            let x = if (370..450).contains(&frame) {
                0.85 + (frame - 370) as f64 * 0.001
            } else {
                0.30 + frame as f64 * 0.0005
            };
            let row = json!({"frame_index":frame,"track_ids_provisional":true,
                "confirmed_athlete_identities":false,"touches_confirmed":false,
                "selected_ball":{"source_model":"vballnet","detection":{
                    "score":0.8,"bbox_xyxy_norm":[x,0.4,x+0.01,0.41]}},"tracked_players":[]});
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        fs::write(&path, &bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let scope = native_rally_preview::RallyWindows {
            intervals: vec![(0, 499)],
            contexts: std::collections::BTreeMap::new(),
        };
        let rows = extract_frames(
            &path, &sha, 500, 400, 405, &scope, 1280, 720, 0.35, None, None,
        )
        .unwrap();
        assert!(rows.iter().all(|row| row["ball"].is_null()));
        assert_eq!(fs::read(&path).unwrap(), bytes);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn processed_overlay_interpolates_vballnet_without_reusing_ravel_ball() {
        let root = std::env::temp_dir().join(format!("volley-ball-gap-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("tracked_frames.jsonl");
        let mut data = Vec::new();
        for frame in 0..8u64 {
            let source = if (3..=4).contains(&frame) {
                "ravel"
            } else {
                "vballnet"
            };
            let bbox = match frame {
                0 => [0.0, 0.2, 0.1, 0.3],
                1 => [0.1, 0.2, 0.2, 0.3],
                2 => [0.2, 0.2, 0.3, 0.3],
                5 => [0.5, 0.2, 0.6, 0.3],
                _ => [0.9, 0.8, 1.0, 0.9],
            };
            let ball = (frame <= 5).then(|| {
                json!({
                    "source_model": source,
                    "detection": {"score":0.95,"bbox_xyxy_norm":bbox}
                })
            });
            let row = json!({
                "frame_index":frame,"track_ids_provisional":true,
                "confirmed_athlete_identities":false,"touches_confirmed":false,
                "selected_ball":ball,
                "ball_candidates":ball.iter().cloned().collect::<Vec<_>>(),
                "tracked_players":[]
            });
            data.extend(serde_json::to_vec(&row).unwrap());
            data.push(b'\n');
        }
        fs::write(&path, &data).unwrap();
        let sha = format!("{:x}", Sha256::digest(&data));
        let scope = native_rally_preview::RallyWindows {
            intervals: vec![(0, 7)],
            contexts: std::collections::BTreeMap::new(),
        };
        // A request beginning inside the hole needs the VballNet anchors
        // outside the requested subwindow. The +/-8-frame read halo supplies them.
        let frames = extract_frames(
            &path,
            &sha,
            8,
            3,
            5,
            &scope,
            640,
            360,
            0.35,
            Some((50, 50, 50, 50)),
            None,
        )
        .unwrap();
        assert_eq!(frames.len(), 2);
        for row in &frames {
            assert_eq!(row["ball_interpolated"], true);
            assert_eq!(row["ball_origin"], "vballnet_interpolated");
            assert!(row["ball"][0].as_f64().unwrap() < 0.55);
            assert_eq!(row["raw"]["balls"][0]["source"], "ravel");
        }
        assert!((frames[0]["ball"][0].as_f64().unwrap() - 0.35).abs() < 1e-12);
        assert!((frames[1]["ball"][0].as_f64().unwrap() - 0.45).abs() < 1e-12);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn processed_ball_interpolation_stays_within_saved_rally_boundaries() {
        let root = std::env::temp_dir().join(format!("volley-rally-gap-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("tracked_frames.jsonl");
        let mut data = Vec::new();
        for frame in 0..12 {
            let x = 0.20 + f64::from(frame) * 0.01;
            let ball = (!(4..=5).contains(&frame)).then(|| {
                json!({
                    "source_model":"vballnet",
                    "detection":{"score":0.95,"bbox_xyxy_norm":[x,0.39,x+0.01,0.41]}
                })
            });
            let row = json!({
                "frame_index":frame,"track_ids_provisional":true,
                "confirmed_athlete_identities":false,"touches_confirmed":false,
                "selected_ball":ball,"tracked_players":[]
            });
            data.extend(serde_json::to_vec(&row).unwrap());
            data.push(b'\n');
        }
        fs::write(&path, &data).unwrap();
        let sha = format!("{:x}", Sha256::digest(&data));
        for (intervals, filled) in [
            (vec![(0, 11)], true),
            (vec![(0, 4), (5, 11)], false),
            (vec![(0, 3), (6, 11)], false),
            (vec![], false),
        ] {
            let scope = native_rally_preview::RallyWindows {
                intervals,
                contexts: std::collections::BTreeMap::new(),
            };
            let frames = extract_frames(
                &path,
                &sha,
                12,
                4,
                6,
                &scope,
                1280,
                720,
                0.35,
                None,
                Some((&[0], 0.35)),
            )
            .unwrap();
            for row in frames {
                assert_eq!(row["ball_interpolated"], filled);
                assert_eq!(row["ball"].is_array(), filled);
                if filled {
                    let expected = if row["frame"] == 4 { 0.245 } else { 0.255 };
                    assert!((row["ball"][0].as_f64().unwrap() - expected).abs() < 1e-12);
                }
            }
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn inferred_player_labels_never_cover_a_different_measured_player() {
        let frame = json!({
            "interpolated_track_ids":[2,3],
            "tracked_players":[
                {"track_id":1,"is_verified_athlete":false,
                    "detection":{"bbox_xyxy_norm":[0.40,0.30,0.48,0.64]}},
                {"track_id":2,"is_verified_athlete":false,
                    "detection":{"bbox_xyxy_norm":[0.415,0.32,0.485,0.63]}},
                {"track_id":3,"is_verified_athlete":false,
                    "detection":{"bbox_xyxy_norm":[0.72,0.27,0.79,0.63]}}
            ]
        });
        let players = overlay_players(&frame);
        assert_eq!(players.len(), 2);
        assert_eq!(players[0]["id"], 1, "measured player retains its ID");
        assert_eq!(players[0]["interpolated"], false);
        assert_eq!(
            players[1]["id"], 3,
            "independent visual interpolation remains visible"
        );
        assert_eq!(players[1]["interpolated"], true);
        assert!(
            !players.iter().any(|player| player["id"] == 2),
            "ghost player may not put a second identity over a measured body"
        );
    }

    #[test]
    fn overlays_hide_players_outside_plays_and_keep_inside_player_ids() {
        let root =
            std::env::temp_dir().join(format!("volley-overlay-scope-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let input = root.join("tracked_frames.jsonl");
        let mut data = Vec::new();
        for frame in 0..6u64 {
            let row = json!({
                "frame_index":frame,"track_ids_provisional":true,
                "confirmed_athlete_identities":false,"touches_confirmed":false,
                "selected_ball":null,
                "tracked_players":[{"track_id":7,"is_verified_athlete":false,
                    "detection":{"bbox_xyxy_norm":[0.1,0.1,0.2,0.8]}}]
            });
            data.extend(serde_json::to_vec(&row).unwrap());
            data.push(b'\n');
        }
        fs::write(&input, &data).unwrap();
        let sha = format!("{:x}", Sha256::digest(&data));
        let active = native_rally_preview::RallyWindows {
            intervals: vec![(2, 4)],
            contexts: std::collections::BTreeMap::new(),
        };
        let overlay =
            extract_frames(&input, &sha, 6, 0, 6, &active, 640, 360, 0.35, None, None).unwrap();
        assert_eq!(overlay.len(), 6);
        for (frame, item) in overlay.iter().enumerate() {
            let in_rally = (2..=4).contains(&frame);
            assert_eq!(item["in_rally"], in_rally);
            assert_eq!(
                item["players"].as_array().unwrap().len(),
                if in_rally { 1 } else { 0 }
            );
        }
        assert_eq!(overlay[2]["players"][0]["id"], 7);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn player_labels_include_only_manually_reviewed_owners() {
        use std::collections::{BTreeMap, BTreeSet};
        let reviewed = NativeReviewedOwners {
            source_sha: "a".repeat(64),
            revision: "test".into(),
            owners: BTreeMap::from([
                (7, "player-a".into()),
                (8, "player-b".into()),
                (9, "missing".into()),
            ]),
            excluded: BTreeSet::new(),
            roster: BTreeMap::from([
                ("player-a".into(), "Ильнур".into()),
                ("player-b".into(), " ".into()),
            ]),
        };
        let names = reviewed_player_names(&reviewed);
        assert_eq!(names.get("7").map(String::as_str), Some("Ильнур"));
        assert!(!names.contains_key("8"));
        assert!(!names.contains_key("9"));
        assert_eq!(names.len(), 1);
    }

    #[test]
    fn rejects_invalid_boxes() {
        assert!(box_norm(&json!([0.1, 0.2, 0.9, 0.8])).is_some());
        assert!(box_norm(&json!([0.8, 0.2, 0.1, 0.7])).is_none());
        assert!(box_norm(&json!([-0.1, 0.2, 0.9, 0.7])).is_none());
        assert!(box_norm(&json!([0.1, 0.2, 0.9])).is_none());
    }
}
