//! Versioned manual assignment of *provisional* native CPU Track IDs to
//! athlete registry entries. The raw GPU/CPU evidence is NEVER rewritten.
//! ReID appearance vectors / saved athlete photo templates are not generated
//! by this module; each owner association requires an explicit human action.
use crate::{
    cpu_retrack::opaque_id,
    inference_stream, native_reid_cache, player_catalog, preparation, project_guard, projects,
    storage::{self, safe_file, safe_folder, ApiError, ApiResult, BoundedReader},
    versioned_store::{self, revision},
    AppState,
};
use axum::{
    extract::{Query, State},
    http::HeaderMap,
    Json,
};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::SystemTime,
};
use volley_core::{
    continuity::{group_tracklets, ContinuityConfig, ContinuityReport, PlayerDetection},
    evidence_source::{EvidenceSource, FrameRate, SourceStamp, VerifiedTrackSet},
    fusion::Candidate,
    identity_review::{self, ManualReviewState, ReviewAction, ReviewErrorKind, ReviewedLink},
    motion::{ball_motion_extrema, BallObservation},
    reid::{self, Feature, FeatureMode, TrackInfo},
};

pub(crate) const MAX_TRACKS: usize = 20_000;
const REID_PHOTOS_PER_TRACK: usize = 5;
const PHOTO_SAMPLE_POLICY: &str = "five_observations_v2";
const MAX_MANIFEST: u64 = 16 * 1024;
const MAX_EVIDENCE: u64 = 64 * 1024 * 1024 * 1024;
const MAX_LINKS: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackQuery {
    project_id: String,
    video: String,
    replay_id: String,
    /// Optional exact CPU rally run for player review (not the raw replay).
    #[serde(default)]
    rally_run_id: Option<String>,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    view: Option<String>,
    #[serde(default)]
    view_player_id: Option<String>,
}
fn default_limit() -> usize {
    40
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectTrackReset {
    project_id: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackEdit {
    project_id: String,
    video: String,
    replay_id: String,
    action: String,
    track_ids: Vec<u64>,
    #[serde(default)]
    rally_run_id: Option<String>,
    #[serde(default)]
    player_id: Option<String>,
    expected_revision: String,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
}
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
struct Aggregate {
    first: u64,
    last: u64,
    observed: u64,
    best_frame: u64,
    best_score: f64,
    best_box: [f64; 4],
    #[serde(default)]
    first_box: [f64; 4],
    #[serde(default)]
    last_box: [f64; 4],
    photo_samples: Vec<PhotoSample>,
    visible_spans: Vec<(u64, u64)>,
    #[serde(with = "source_index::pair_geometry_keys")]
    observed_pairs: BTreeMap<(u64, u64), PairGeometry>,
}
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
struct PairGeometry {
    same_position: u32,
    distinct_position: u32,
    // Actual timestamp of the first independently detected pair in separate
    // positions. Never inferred from interpolated or overlapping lifespans.
    first_distinct_frame: Option<u64>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PhotoSample {
    frame: u64,
    bbox: [f64; 4],
}
#[derive(Deserialize)]
struct Tracked {
    track_id: u64,
    detection: Candidate,
    is_verified_athlete: bool,
}
#[derive(Deserialize)]
struct NativeSelectedBall {
    source_model: String,
    detection: Candidate,
}
#[derive(Deserialize)]
struct Frame {
    frame_index: u64,
    selected_ball: Option<NativeSelectedBall>,
    tracked_players: Vec<Tracked>,
    #[serde(default)]
    interpolated_track_ids: Vec<u64>,
    track_ids_provisional: bool,
    confirmed_athlete_identities: bool,
    touches_confirmed: bool,
}
struct Inputs {
    evidence: EvidenceSource,
    known: VerifiedTrackSet,
    project_id: String,
    video: String,
    replay_id: String,
    source_sha: String,
    analysis_offset_sec: f64,
    tracks: BTreeMap<u64, Aggregate>,
    roster: BTreeMap<String, String>,
    sidecar: PathBuf,
    can_write: bool,
}
pub(crate) use volley_core::contracts::player_id as valid_player_id;
pub(crate) use volley_core::contracts::sha256 as valid_sha;
mod source_index;
use source_index::*;
pub(crate) use source_index::{
    prepare_rally_index, prepare_replay_index, replay_rally_track_ids, replay_track_counts,
};
mod assignments;
pub(crate) use assignments::*;

mod trajectory;
pub(crate) use trajectory::*;
mod continuity;
pub(crate) use continuity::*;
mod reid_handlers;
pub(crate) use reid_handlers::*;
mod photo_comparison;
mod photos;
pub(crate) use photos::*;
mod shared_crops;
pub(crate) use shared_crops::{
    plan_shared_reid_crops, SharedReidCrops, DEFAULT_REID_BUFFER_MIB, MAX_REID_BUFFER_MIB,
};
mod embedding_jobs;
pub(crate) use embedding_jobs::*;
mod reference_store;
pub(crate) use reference_store::*;
mod reference_refresh;
pub(crate) use reference_refresh::*;
#[cfg(test)]
mod tests;

fn reset_saved_player_links(doc: &mut Value) -> Result<(usize, bool), String> {
    let manual = doc["assignments"]
        .as_object()
        .ok_or("Invalid manual bindings")?;
    let automatic = doc["automatic_assignments"].as_object();
    for id in manual
        .keys()
        .chain(automatic.into_iter().flat_map(|links| links.keys()))
    {
        id.parse::<u64>().map_err(|_| "Invalid linked ID")?;
    }
    let removed = manual.len() + automatic.map_or(0, serde_json::Map::len);
    let suppressed = doc["automatic_suppressed_ids"]
        .as_array()
        .is_some_and(|ids| !ids.is_empty());
    if removed == 0 && !suppressed {
        return Ok((0, false));
    }
    doc["assignments"] = json!({});
    doc["automatic_assignments"] = json!({});
    doc["automatic_suppressed_ids"] = json!([]);
    Ok((removed, true))
}

// Reset all saved player Track ID bindings for this project, keeping evidence and albums.
pub async fn reset_project(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ProjectTrackReset>,
) -> crate::storage::ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let result = tokio::task::spawn_blocking(move || {
        let _guard = project_guard::write_guard(&state, &req.project_id)?;
        if !state.writes_enabled {
            return Err(storage::conflict("Project binding edits are disabled"));
        }
        storage::project_id(&req.project_id)?;
        let root = projects::project_root(&state).join(&req.project_id);
        safe_folder(&root).map_err(storage::bad_request)?;
        let directory = root.join("native_track_assignments");
        if !directory.exists() {
            return Ok(json!({"reset_tracks":0,"reset_replays":0}));
        }
        safe_folder(&directory).map_err(storage::bad_request)?;
        let mut count = 0usize;
        let mut versions = 0usize;
        for item in fs::read_dir(&directory)
            .map_err(|e| storage::internal(format!("Read bindings directory: {e}")))?
        {
            let item = item.map_err(|e| storage::internal(format!("Read binding item: {e}")))?;
            let path = item.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|e| storage::bad_request(format!("Check binding item: {e}")))?;
            if metadata.file_type().is_symlink()
                || !metadata.is_file()
                || path.extension().and_then(|v| v.to_str()) != Some("json")
            {
                return Err(storage::bad_request("Unsafe Track ID binding entry"));
            }
            let Some(bytes) = versioned_store::read_optional_bounded(&path, MAX_LINKS)
                .map_err(storage::bad_request)?
            else {
                continue;
            };
            let mut doc: Value = serde_json::from_slice(&bytes)
                .map_err(|e| storage::bad_request(format!("Invalid bindings JSON: {e}")))?;
            if doc["schema"] != "volley-native-player-track-links/1"
                || doc["project_id"] != req.project_id
                || doc["replay_id"].as_str() != path.file_stem().and_then(|v| v.to_str())
            {
                return Err(storage::bad_request("Invalid project Track ID bindings"));
            }
            let (removed, changed) =
                reset_saved_player_links(&mut doc).map_err(storage::bad_request)?;
            if !changed {
                continue;
            }
            count += removed;
            versioned_store::write_json_checked(&directory, &path, &doc, MAX_LINKS)
                .map_err(storage::bad_request)?;
            versions += 1;
        }
        Ok(json!({"reset_tracks":count,"reset_replays":versions}))
    })
    .await
    .map_err(|e| storage::internal(format!("Reset project Track ID task: {e}")))??;
    Ok(Json(result))
}
