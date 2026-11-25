use super::*;

const MAX_TRACKS_PER_ATHLETE: usize = 16;
const MAX_REFRESH_TRACKS: usize = 128;
const MAX_NEW_PHOTOS_PER_ATHLETE: usize = 3;
const MAX_DISABLED_PER_ATHLETE: usize = 8;
const NOVEL_COSINE_LIMIT: f32 = 0.92;
const OTHER_ATHLETE_COSINE_LIMIT: f32 = 0.80;
const OTHER_ATHLETE_MARGIN: f32 = 0.06;
const DUPLICATE_COSINE_LIMIT: f32 = 0.995;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefreshReferencesRequest {
    project_id: String,
    video: String,
    replay_id: String,
    #[serde(default)]
    rally_run_id: Option<String>,
    expected_source_sha256: String,
}

#[derive(Clone)]
struct RefreshAlbum {
    player_id: String,
    revision: String,
    photos: Vec<Value>,
}

struct RefreshSnapshot {
    albums: Vec<RefreshAlbum>,
    candidates: Vec<(u64, String)>,
    confirmed_tracks: usize,
    model: String,
}

fn refresh_snapshot(
    state: &AppState,
    req: &RefreshReferencesRequest,
) -> Result<RefreshSnapshot, ApiError> {
    let ctx = source_for_rallies(
        state,
        &req.project_id,
        &req.video,
        &req.replay_id,
        req.rally_run_id.as_deref(),
    )?;
    if ctx.source_sha != req.expected_source_sha256 {
        return Err(storage::conflict(
            "CPU replay changed during reference refresh",
        ));
    }
    let raw = existing(&ctx.sidecar).map_err(storage::bad_request)?;
    let identity = validated(raw.as_deref(), &ctx).map_err(storage::bad_request)?;
    let assigned = identity["assignments"]
        .as_object()
        .ok_or_else(|| storage::bad_request("Invalid reviewed player assignments"))?;
    let mut albums = Vec::new();
    for player_id in ctx.roster.keys() {
        let folder = checked_reference_folder(state, player_id, false)?;
        let raw =
            versioned_store::read_optional_bounded(&folder.join("index.json"), MAX_REFERENCE_INDEX)
                .map_err(storage::bad_request)?;
        let doc = references_doc(raw.as_deref(), player_id)?;
        let photos = doc["photos"]
            .as_array()
            .ok_or_else(|| storage::bad_request("Invalid photo album"))?
            .clone();
        albums.push(RefreshAlbum {
            player_id: player_id.clone(),
            revision: revision(raw.as_deref()),
            photos,
        });
    }
    let mut per_player = BTreeMap::<String, Vec<(u64, u64)>>::new();
    let mut confirmed_tracks = 0;
    for (id, player) in assigned {
        let (Ok(track_id), Some(player_id)) = (id.parse::<u64>(), player.as_str()) else {
            continue;
        };
        let (Some(track), true) = (
            ctx.tracks.get(&track_id),
            ctx.roster.contains_key(player_id),
        ) else {
            continue;
        };
        confirmed_tracks += 1;
        let already_saved = albums
            .iter()
            .find(|album| album.player_id == player_id)
            .is_some_and(|album| {
                album.photos.iter().any(|photo| {
                    photo["track_id"].as_u64() == Some(track_id)
                        && photo["project_id"] == req.project_id
                        && photo["video"] == req.video
                        && photo["replay_id"] == req.replay_id
                        && photo["source_replay_sha256"] == req.expected_source_sha256
                })
            });
        if !already_saved && track.best_box != [0.0; 4] && track.observed > 0 {
            per_player
                .entry(player_id.to_owned())
                .or_default()
                .push((track_id, track.observed));
        }
    }
    let mut candidates = Vec::new();
    for (player_id, mut tracks) in per_player {
        tracks.sort_by_key(|(id, observed)| (std::cmp::Reverse(*observed), *id));
        for (track_id, _) in tracks.into_iter().take(MAX_TRACKS_PER_ATHLETE) {
            candidates.push((track_id, player_id.clone()));
        }
    }
    candidates.truncate(MAX_REFRESH_TRACKS);
    let model = volley_inference::reid_openvino::fingerprint(&state.repository_root)
        .map_err(storage::bad_request)?;
    Ok(RefreshSnapshot {
        albums,
        candidates,
        confirmed_tracks,
        model,
    })
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return -1.0;
    }
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn max_cosine<'a>(target: &[f32], vectors: impl Iterator<Item = &'a Vec<f32>>) -> f32 {
    vectors
        .map(|vector| cosine(target, vector))
        .fold(-1.0, f32::max)
}

#[derive(PartialEq, Eq, Debug)]
enum PhotoDecision {
    Novel,
    Duplicate,
    Ambiguous,
}

fn assess_photo(
    vector: &[f32],
    athlete: &str,
    active: &BTreeMap<String, Vec<Vec<f32>>>,
) -> PhotoDecision {
    let own = active
        .get(athlete)
        .map_or(-1.0, |rows| max_cosine(vector, rows.iter()));
    if own >= NOVEL_COSINE_LIMIT {
        return PhotoDecision::Duplicate;
    }
    let other = active
        .iter()
        .filter(|(player, _)| player.as_str() != athlete)
        .flat_map(|(_, vectors)| vectors.iter());
    let other_score = max_cosine(vector, other);
    if other_score >= OTHER_ATHLETE_COSINE_LIMIT
        && (own < 0.0 || other_score + OTHER_ATHLETE_MARGIN >= own)
    {
        PhotoDecision::Ambiguous
    } else {
        PhotoDecision::Novel
    }
}

async fn infer_missing(
    state: Arc<AppState>,
    model: &str,
    inputs: Vec<(String, Vec<f32>)>,
) -> Result<Vec<(String, Vec<f32>)>, ApiError> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let repo = state.repository_root.clone();
    let library = std::env::var("VOLLEY_OPENVINO_C_LIB")
        .map_err(|_| storage::bad_request("ReID OpenVINO C runtime is not configured"))?;
    let model = model.to_owned();
    tokio::task::spawn_blocking(move || -> Result<Vec<(String, Vec<f32>)>, ApiError> {
        if volley_inference::reid_openvino::fingerprint(&repo).map_err(storage::bad_request)?
            != model
        {
            return Err(storage::conflict("ReID model changed during photo refresh"));
        }
        let refs: Vec<&[f32]> = inputs.iter().map(|(_, input)| input.as_slice()).collect();
        let vectors =
            volley_inference::reid_openvino::infer_batch(&repo, Path::new(&library), &refs)
                .map_err(storage::bad_request)?;
        Ok(inputs
            .into_iter()
            .zip(vectors)
            .map(|((sha, _), vector)| (sha, vector))
            .collect())
    })
    .await
    .map_err(|e| storage::internal(format!("Reference feature worker: {e}")))?
}

async fn album_vectors(
    state: Arc<AppState>,
    album: &RefreshAlbum,
    model: &str,
) -> Result<BTreeMap<String, Vec<f32>>, ApiError> {
    let mut vectors = photo_comparison::read_global_features(&state, &album.player_id, model)?;
    let folder = checked_reference_folder(&state, &album.player_id, false)?;
    let mut pending = Vec::new();
    for photo in &album.photos {
        if photo["reid_active"] == false {
            continue;
        }
        let sha = photo["image_sha256"]
            .as_str()
            .ok_or_else(|| storage::bad_request("Reference SHA missing"))?;
        let id = photo["id"]
            .as_str()
            .ok_or_else(|| storage::bad_request("Reference ID missing"))?;
        let path = folder.join(format!("ref-{id}.jpg"));
        let jpeg = versioned_store::read_optional_bounded(&path, MAX_PHOTO_BYTES as u64)
            .map_err(storage::bad_request)?
            .ok_or_else(|| storage::conflict("Saved reference photo is missing"))?;
        if jpeg.len() < 100
            || !jpeg.starts_with(&[0xff, 0xd8])
            || !jpeg.ends_with(&[0xff, 0xd9])
            || format!("{:x}", Sha256::digest(&jpeg)) != sha
        {
            return Err(storage::conflict("Saved reference photo checksum mismatch"));
        }
        if vectors.contains_key(sha) || pending.iter().any(|(id, _)| id == sha) {
            continue;
        }
        pending.push((sha.to_owned(), photo_comparison::decode(&jpeg).await?));
        if pending.len() >= 24 {
            for (key, vector) in
                infer_missing(state.clone(), model, std::mem::take(&mut pending)).await?
            {
                vectors.insert(key, vector);
            }
        }
    }
    for (key, vector) in infer_missing(state.clone(), model, pending).await? {
        vectors.insert(key, vector);
    }
    photo_comparison::persist_album_features(&state, &album.player_id, model, &vectors)?;
    Ok(vectors)
}

fn same_reference_track(left: &Value, right: &Value) -> bool {
    let Some(track_id) = left["track_id"].as_u64().filter(|id| *id != 0) else {
        return false;
    };
    if right["track_id"].as_u64() != Some(track_id) {
        return false;
    }
    for key in ["project_id", "video", "replay_id", "source_replay_sha256"] {
        if left[key].as_str().is_none_or(str::is_empty) || left[key] != right[key] {
            return false;
        }
    }
    left["rally_run_id"] == right["rally_run_id"]
}

fn redundant_references(
    album: &RefreshAlbum,
    vectors: &BTreeMap<String, Vec<f32>>,
) -> BTreeSet<String> {
    let mut kept: Vec<(&Value, &Vec<f32>)> = Vec::new();
    let mut disabled = BTreeSet::new();
    let active_count = album
        .photos
        .iter()
        .filter(|photo| photo["reid_active"] != false)
        .count();
    if active_count < 5 {
        return disabled;
    }
    for photo in &album.photos {
        if photo["reid_active"] == false {
            continue;
        }
        let (Some(id), Some(sha)) = (photo["id"].as_str(), photo["image_sha256"].as_str()) else {
            continue;
        };
        let Some(vector) = vectors.get(sha) else {
            continue;
        };
        let close_to_same_track = kept.iter().any(|(other_photo, other)| {
            same_reference_track(photo, other_photo)
                && cosine(vector, other) >= DUPLICATE_COSINE_LIMIT
        });
        if close_to_same_track
            && disabled.len() < MAX_DISABLED_PER_ATHLETE
            && active_count.saturating_sub(disabled.len()) > 4
        {
            disabled.insert(id.to_owned());
        } else {
            kept.push((photo, vector));
        }
    }
    disabled
}

fn disable_redundant(
    state: &AppState,
    project: &str,
    album: &RefreshAlbum,
    expected_revision: &str,
    ids: &BTreeSet<String>,
) -> Result<(), ApiError> {
    if ids.is_empty() {
        return Ok(());
    }
    reference_player(state, project, &album.player_id)?;
    let folder = checked_reference_folder(state, &album.player_id, true)?;
    let lock_path = folder.join(".references.lock");
    if fs::symlink_metadata(&lock_path)
        .ok()
        .is_some_and(|metadata| metadata.file_type().is_symlink())
    {
        return Err(storage::bad_request("Unsafe reference album lock"));
    }
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|e| storage::internal(format!("Reference lock open: {e}")))?;
    lock.lock_exclusive()
        .map_err(|e| storage::internal(format!("Reference lock: {e}")))?;
    let index = folder.join("index.json");
    let raw = versioned_store::read_optional_bounded(&index, MAX_REFERENCE_INDEX)
        .map_err(storage::bad_request)?;
    if revision(raw.as_deref()) != expected_revision {
        return Err(storage::conflict(
            "Photo album changed during reference pruning",
        ));
    }
    let mut document = references_doc(raw.as_deref(), &album.player_id)?;
    let photos = document["photos"]
        .as_array_mut()
        .ok_or_else(|| storage::bad_request("Invalid album photo list"))?;
    let mut changed = 0;
    for photo in photos {
        if photo["id"].as_str().is_some_and(|id| ids.contains(id)) && photo["reid_active"] != false
        {
            photo["reid_active"] = json!(false);
            changed += 1;
        }
    }
    if changed != ids.len() {
        return Err(storage::conflict(
            "Reference photo set changed during refresh",
        ));
    }
    versioned_store::write_json_checked(&folder, &index, &document, MAX_REFERENCE_INDEX)
        .map_err(storage::bad_request)?;
    Ok(())
}

pub async fn refresh_reference_features(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<RefreshReferencesRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    if !state.writes_enabled {
        return Err(storage::conflict(
            "Updating athlete references requires writable storage",
        ));
    }
    if !valid_sha(&req.expected_source_sha256)
        || req.rally_run_id.as_deref().is_some_and(|id| !opaque_id(id))
    {
        return Err(storage::bad_request("Invalid CPU replay provenance"));
    }
    let scan_state = state.clone();
    let scan_req = req.clone();
    let plan = tokio::task::spawn_blocking(move || refresh_snapshot(&scan_state, &scan_req))
        .await
        .map_err(|e| storage::internal(format!("Reference scan task: {e}")))??;
    let mut track_vectors = BTreeMap::<u64, Vec<f32>>::new();
    for batch in plan.candidates.chunks(48) {
        let Json(result) = embeddings(
            State(state.clone()),
            headers.clone(),
            Json(EmbeddingRequest {
                project_id: req.project_id.clone(),
                video: req.video.clone(),
                replay_id: req.replay_id.clone(),
                rally_run_id: req.rally_run_id.clone(),
                expected_source_sha256: req.expected_source_sha256.clone(),
                track_ids: batch.iter().map(|(id, _)| *id).collect(),
                transient: false,
                reid_device: Default::default(),
            }),
        )
        .await?;
        if result["model_fingerprint"] != plan.model {
            return Err(storage::conflict(
                "ReID model changed while refreshing reference features",
            ));
        }
        for row in result["features"].as_array().into_iter().flatten() {
            let id = row["track_id"]
                .as_u64()
                .ok_or_else(|| storage::internal("Track embedding ID missing"))?;
            let vector = serde_json::from_value::<Vec<f32>>(row["vector"].clone())
                .map_err(|_| storage::internal("Invalid track embedding"))?;
            if track_vectors.insert(id, vector).is_some() {
                return Err(storage::internal("Duplicate track embedding"));
            }
        }
    }
    if track_vectors.len() != plan.candidates.len() {
        return Err(storage::internal("Incomplete confirmed Track ID features"));
    }
    let mut all_features = BTreeMap::<String, BTreeMap<String, Vec<f32>>>::new();
    {
        let _reid = state
            .reid_limit
            .acquire()
            .await
            .map_err(|e| storage::internal(format!("ReID feature refresh gate: {e}")))?;
        let _preview = state
            .preview_limit
            .acquire()
            .await
            .map_err(|e| storage::internal(format!("Photo decode refresh gate: {e}")))?;
        for album in &plan.albums {
            all_features.insert(
                album.player_id.clone(),
                album_vectors(state.clone(), album, &plan.model).await?,
            );
        }
    }
    let _identity = state
        .identity_limit
        .acquire()
        .await
        .map_err(|e| storage::internal(format!("Photo refresh gate: {e}")))?;
    let recheck_state = state.clone();
    let recheck_req = req.clone();
    let recheck =
        tokio::task::spawn_blocking(move || refresh_snapshot(&recheck_state, &recheck_req))
            .await
            .map_err(|e| storage::internal(format!("Reference recheck task: {e}")))??;
    if recheck.model != plan.model
        || recheck.confirmed_tracks != plan.confirmed_tracks
        || recheck.candidates != plan.candidates
        || recheck
            .albums
            .iter()
            .map(|a| (&a.player_id, &a.revision))
            .collect::<Vec<_>>()
            != plan
                .albums
                .iter()
                .map(|a| (&a.player_id, &a.revision))
                .collect::<Vec<_>>()
    {
        return Err(storage::conflict(
            "Player roster, reviewed Track ID or photos changed during reference refresh",
        ));
    }
    let mut active = BTreeMap::<String, Vec<Vec<f32>>>::new();
    for album in &plan.albums {
        let features = &all_features[&album.player_id];
        let values = album
            .photos
            .iter()
            .filter(|photo| photo["reid_active"] != false)
            .filter_map(|photo| photo["image_sha256"].as_str())
            .filter_map(|sha| features.get(sha).cloned())
            .collect();
        active.insert(album.player_id.clone(), values);
    }
    let mut proposals = Vec::new();
    let mut skipped_ambiguous = 0;
    let mut skipped_duplicate = 0;
    let mut skipped_capacity = 0;
    let mut per_player_added = BTreeMap::<String, usize>::new();
    for (track_id, player_id) in &plan.candidates {
        let vector = &track_vectors[track_id];
        match assess_photo(vector, player_id, &active) {
            PhotoDecision::Duplicate => skipped_duplicate += 1,
            PhotoDecision::Ambiguous => skipped_ambiguous += 1,
            PhotoDecision::Novel => {
                let Some(album) = plan.albums.iter().find(|a| a.player_id == *player_id) else {
                    continue;
                };
                let count = per_player_added.entry(player_id.clone()).or_default();
                if *count >= MAX_NEW_PHOTOS_PER_ATHLETE
                    || album.photos.len() + *count >= MAX_PLAYER_PHOTOS
                {
                    skipped_capacity += 1;
                } else {
                    proposals.push((*track_id, player_id.clone()));
                    active
                        .entry(player_id.clone())
                        .or_default()
                        .push(vector.clone());
                    *count += 1;
                }
            }
        }
    }
    let mut revisions = plan
        .albums
        .iter()
        .map(|a| (a.player_id.clone(), a.revision.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut added = 0;
    for (track_id, player_id) in &proposals {
        let request = ReferenceRequest {
            project_id: req.project_id.clone(),
            player_id: player_id.clone(),
            action: "enroll".into(),
            expected_revision: revisions.get(player_id).cloned(),
            video: Some(req.video.clone()),
            replay_id: Some(req.replay_id.clone()),
            rally_run_id: req.rally_run_id.clone(),
            photo_index: 0,
            track_id: Some(*track_id),
            expected_source_sha256: Some(req.expected_source_sha256.clone()),
            reference_id: None,
        };
        let crop = photo(
            State(state.clone()),
            Query(PhotoQuery {
                project_id: req.project_id.clone(),
                video: req.video.clone(),
                replay_id: req.replay_id.clone(),
                track_id: *track_id,
                rally_run_id: req.rally_run_id.clone(),
                photo_index: 0,
            }),
        )
        .await?;
        let jpeg = axum::body::to_bytes(crop.into_body(), MAX_PHOTO_BYTES)
            .await
            .map_err(|_| storage::bad_request("Selected Track ID JPEG exceeds limit"))?;
        let writer = state.clone();
        let result =
            tokio::task::spawn_blocking(move || edit_references(&writer, &request, Some(&jpeg)))
                .await
                .map_err(|e| storage::internal(format!("Save new reference photo: {e}")))??;
        let new_revision = result["revision"]
            .as_str()
            .ok_or_else(|| storage::internal("Reference revision missing after save"))?;
        revisions.insert(player_id.clone(), new_revision.to_owned());
        added += 1;
    }
    let mut deactivated = 0;
    for album in &plan.albums {
        let ids = redundant_references(album, &all_features[&album.player_id]);
        if ids.is_empty() {
            continue;
        }
        let writer = state.clone();
        let project = req.project_id.clone();
        let original = album.clone();
        let expected = revisions[&album.player_id].clone();
        let changed = ids.len();
        tokio::task::spawn_blocking(move || {
            disable_redundant(&writer, &project, &original, &expected, &ids)
        })
        .await
        .map_err(|e| storage::internal(format!("Deactivate redundant references: {e}")))??;
        deactivated += changed;
    }
    Ok(Json(json!({
        "schema":"volley-native-player-reference-refresh/1",
        "project_id":req.project_id,
        "replay_id":req.replay_id,
        "model_fingerprint":plan.model,
        "confirmed_tracks":plan.confirmed_tracks,
        "examined_tracks":plan.candidates.len(),
        "added":added,
        "deactivated":deactivated,
        "skipped_duplicate":skipped_duplicate,
        "skipped_ambiguous":skipped_ambiguous,
        "skipped_capacity":skipped_capacity,
        "reference_photos_preserved":true,
        "assignments_changed":false,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding(a: f32, b: f32) -> Vec<f32> {
        vec![a, b]
    }

    #[test]
    fn ignores_same_player_duplicate_and_cross_player_collision() {
        let mut active = BTreeMap::new();
        active.insert("alice".to_owned(), vec![embedding(1.0, 0.0)]);
        active.insert("bob".to_owned(), vec![embedding(0.0, 1.0)]);
        assert_eq!(
            assess_photo(&embedding(1.0, 0.0), "alice", &active),
            PhotoDecision::Duplicate,
        );
        assert_eq!(
            assess_photo(&embedding(0.1, 0.995), "alice", &active),
            PhotoDecision::Ambiguous,
        );
        assert_eq!(
            assess_photo(&embedding(0.7, 0.0), "alice", &BTreeMap::new()),
            PhotoDecision::Novel,
        );
    }

    #[test]
    fn redundant_templates_are_deactivated_only_within_same_track_and_without_deleting() {
        let album = RefreshAlbum {
            player_id: "alice".into(),
            revision: "absent".into(),
            photos: (1..=6)
                .map(|id| {
                    json!({
                        "id":format!("{id:024x}"),
                        "image_sha256":format!("{id:064x}"),
                        "track_id":if id < 4 { 7 } else { 8 },
                        "project_id":"project-1",
                        "video":"match-1.mp4",
                        "replay_id":"cpu-replay-1",
                        "source_replay_sha256":"a".repeat(64),
                        "rally_run_id":"rally-1",
                    })
                })
                .collect(),
        };
        let mut vectors = BTreeMap::new();
        for id in 1..=6 {
            vectors.insert(format!("{id:064x}"), embedding(1.0, 0.0));
        }
        let disabled = redundant_references(&album, &vectors);
        assert_eq!(disabled.len(), 2);
        assert!(disabled.contains(&format!("{:024x}", 2)));
        assert!(disabled.contains(&format!("{:024x}", 3)));
        assert!(!disabled.contains(&format!("{:024x}", 4)));
        assert_eq!(album.photos.len(), 6);
    }

    #[test]
    fn same_numeric_track_id_across_videos_is_not_a_duplicate_reference() {
        let videos = ["match-one.mp4", "match-two.mp4", "match-three.mp4"];
        let album = RefreshAlbum {
            player_id: "alice".into(),
            revision: "absent".into(),
            photos: (1..=6)
                .map(|id| {
                    json!({
                        "id":format!("{id:024x}"),
                        "image_sha256":format!("{id:064x}"),
                        "track_id":if id <= 2 { 7 } else { id as u64 + 20 },
                        "project_id":"project-1",
                        "video":if id == 2 { videos[1] } else { videos[0] },
                        "replay_id":"cpu-replay-1",
                        "source_replay_sha256":"a".repeat(64),
                        "rally_run_id":"rally-1",
                    })
                })
                .collect(),
        };
        let vectors = (1..=6)
            .map(|id| (format!("{id:064x}"), embedding(1.0, 0.0)))
            .collect();
        assert!(redundant_references(&album, &vectors).is_empty());
    }

    #[test]
    fn same_numeric_track_id_across_replays_is_not_a_duplicate_reference() {
        let album = RefreshAlbum {
            player_id: "alice".into(),
            revision: "absent".into(),
            photos: (1..=5)
                .map(|id| {
                    json!({
                        "id":format!("{id:024x}"),
                        "image_sha256":format!("{id:064x}"),
                        "track_id":if id <= 2 { 7 } else { id as u64 + 20 },
                        "project_id":"project-1",
                        "video":"match-1.mp4",
                        "replay_id":if id == 2 { "cpu-replay-2" } else { "cpu-replay-1" },
                        "source_replay_sha256":"a".repeat(64),
                        "rally_run_id":"rally-1",
                    })
                })
                .collect(),
        };
        let vectors = (1..=5)
            .map(|id| (format!("{id:064x}"), embedding(1.0, 0.0)))
            .collect();
        assert!(redundant_references(&album, &vectors).is_empty());
    }
}
