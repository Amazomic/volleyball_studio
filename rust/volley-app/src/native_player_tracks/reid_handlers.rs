use super::*;
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeReidQuery {
    pub(super) project_id: String,
    pub(super) video: String,
    pub(super) replay_id: String,
    pub(super) expected_source_sha256: String,
    #[serde(default = "default_reid_threshold")]
    pub(super) threshold_percent: u8,
}
pub(super) fn default_reid_threshold() -> u8 {
    30
}

pub(super) fn native_reid_report(
    ctx: &Inputs,
    doc: &Value,
    cached: &BTreeMap<u64, Vec<f32>>,
    threshold: u8,
) -> Result<Value, String> {
    let mut manual = BTreeMap::<u64, String>::new();
    for (id, player) in doc["assignments"]
        .as_object()
        .ok_or("Invalid assignment registry")?
    {
        manual.insert(
            id.parse::<u64>().map_err(|_| "Invalid assigned track ID")?,
            player
                .as_str()
                .ok_or("Invalid assigned athlete")?
                .to_owned(),
        );
    }
    let excluded = doc["excluded_ids"]
        .as_array()
        .ok_or("Invalid excluded tracks")?
        .iter()
        .map(|v| v.as_u64().ok_or("Invalid excluded ID"))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let reviewed = identity_review::resolve_reviewed(
        ctx.known.ids(),
        &ctx.roster,
        &manual
            .into_iter()
            .map(|(track_id, owner)| ReviewedLink {
                track_id,
                owner,
                reviewed: true,
            })
            .collect::<Vec<_>>(),
        &excluded,
    );
    if !reviewed.conflicts.is_empty() {
        return Err("Conflicting manual ownership blocks ReID suggestions".into());
    }
    let stats = ctx
        .tracks
        .iter()
        .map(|(&id, track)| TrackInfo {
            id: i64::try_from(id).expect("Replay Track ID bounded by verified manifest"),
            first_frame: i64::try_from(track.first).expect("Replay frame bounded by manifest"),
            last_frame: i64::try_from(track.last).expect("Replay frame bounded by manifest"),
            detected_frames: u32::try_from(track.observed).unwrap_or(u32::MAX),
            start_center: [
                (track.first_box[0] + track.first_box[2]) * 0.5,
                (track.first_box[1] + track.first_box[3]) * 0.5,
            ],
            end_center: [
                (track.last_box[0] + track.last_box[2]) * 0.5,
                (track.last_box[1] + track.last_box[3]) * 0.5,
            ],
            fps: 30.0,
        })
        .collect::<Vec<_>>();
    let features = cached
        .iter()
        .filter(|(id, _)| ctx.known.contains_number(**id))
        .map(|(&id, vec)| {
            Ok(Feature {
                id: i64::try_from(id).map_err(|_| "Track ID out of ReID range")?,
                vector: vec.clone(),
            })
        })
        .collect::<Result<Vec<_>, &str>>()?;
    let owners = reviewed
        .owners
        .into_iter()
        .map(|(id, player)| (id as i64, player))
        .collect::<std::collections::HashMap<_, _>>();
    let mut excluded: std::collections::HashSet<i64> = reviewed
        .excluded
        .into_iter()
        .map(|id| id as i64)
        .collect::<std::collections::HashSet<_>>();
    excluded.extend(owners.keys().copied());
    let report = reid::build_groups(
        &features,
        &stats,
        &owners,
        &excluded,
        FeatureMode::OpenVino,
        threshold,
    )?;
    // These are proposals only: the CPU replay, original detector boxes,
    // and immutable pose evidence must not be rewritten by a preview.
    // The full ReID embedding (not skeleton shape or bounding-box geometry)
    // is the identity gate; geometry only rejects physically impossible joins.
    let tracklets = ctx
        .tracks
        .iter()
        .filter(|(&id, row)| {
            id <= i64::MAX as u64
                && !excluded.contains(&(id as i64))
                && cached.contains_key(&id)
                && row.first_box != [0.0; 4]
                && row.last_box != [0.0; 4]
        })
        .filter_map(|(&id, row)| {
            Some(reid::tracklet_stitch::Tracklet {
                id,
                first_frame: row.first,
                last_frame: row.last,
                observations: u32::try_from(row.observed).ok()?,
                first_box: row.first_box,
                last_box: row.last_box,
                reid: cached.get(&id)?.clone(),
                confirmed_owner: doc["assignments"][id.to_string()]
                    .as_str()
                    .map(str::to_owned),
            })
        })
        .take(1024)
        .collect::<Vec<_>>();
    let links = reid::tracklet_stitch::propose_links(
        &tracklets,
        threshold.max(reid::tracklet_stitch::DEFAULT_REID_STITCH_THRESHOLD_PERCENT),
        reid::tracklet_stitch::DEFAULT_STITCH_MAX_GAP_FRAMES,
    )?;
    let mut result = json!(report);
    result["tracklet_links"] = json!(links
        .into_iter()
        .map(|link| json!({
            "previous_id":link.previous_id,
            "next_id":link.next_id,
            "last_observed_frame":link.last_observed_frame,
            "first_resumed_frame":link.first_resumed_frame,
            "last_observed_box":link.last_observed_box,
            "first_resumed_box":link.first_resumed_box,
            "cosine_similarity":link.cosine_similarity,
            "gap_frames":link.first_resumed_frame.saturating_sub(link.last_observed_frame + 1),
            "provenance":"measured_track_boundaries",
            "status":"suggested_uncommitted"
        }))
        .collect::<Vec<_>>());
    result["tracklet_stitch_threshold_percent"] =
        json!(threshold.max(reid::tracklet_stitch::DEFAULT_REID_STITCH_THRESHOLD_PERCENT));
    result["tracklet_links_applied"] = json!(false);
    Ok(result)
}
pub async fn reid_preview(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<NativeReidQuery>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    if !(1..=99).contains(&req.threshold_percent) {
        return Err(storage::bad_request("ReID threshold must be 1..99"));
    }
    let _permit = state
        .reid_limit
        .acquire()
        .await
        .map_err(|e| storage::internal(format!("Native ReID gate: {e}")))?;
    let state = state.clone();
    let response = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        let _guard = project_guard::write_guard(&state, &req.project_id)?;
        let ctx = source(&state, &req.project_id, &req.video, &req.replay_id)?;
        if ctx.source_sha != req.expected_source_sha256 {
            return Err(storage::conflict("Selected ReID replay SHA changed"));
        }
        let before = volley_inference::reid_openvino::fingerprint(&state.repository_root)
            .map_err(storage::bad_request)?;
        let scope = native_reid_cache::CacheScope {
            project_id: req.project_id.clone(),
            video: req.video.clone(),
            replay_id: req.replay_id.clone(),
            replay_sha256: ctx.source_sha.clone(),
            model_fingerprint: before.clone(),
        };
        let cache = native_reid_cache::read(&state.data_root, &scope)
            .map_err(storage::bad_request)?;
        if cache.invalidated {
            return Err(storage::conflict("ReID embeddings are stale; regenerate from current model"));
        }
        if cache.vectors.is_empty() {
            return Err(storage::bad_request("No native embeddings cached; extract Track ID features first"));
        }
        let raw = existing(&ctx.sidecar).map_err(storage::bad_request)?;
        let identity = validated(raw.as_deref(), &ctx).map_err(storage::bad_request)?;
        let report = native_reid_report(&ctx, &identity, &cache.vectors, req.threshold_percent)
            .map_err(storage::bad_request)?;
        if volley_inference::reid_openvino::fingerprint(&state.repository_root)
            .map_err(storage::bad_request)? != before {
            return Err(storage::conflict("ReID model changed while comparing cached features"));
        }
        Ok(json!({
            "schema":"volley-native-reid-groups/1",
            "source_kind":ctx.evidence.source_kind(),
            "project_id":ctx.project_id,
            "video":ctx.video,
            "replay_id":ctx.replay_id,
            "source_replay_sha256":ctx.source_sha,
            "model_fingerprint":before,
            "method":volley_inference::reid_openvino::MODEL_NAME,
            "reviewed_identity_revision":revision(raw.as_deref()),
            "embedding":report,
            "auto_assign_enabled":false,
            "confirmed":false,
            "warning":"Cosine similarity is not identity probability. ReID groups are proposals only."
        }))
    }).await.map_err(|e| storage::internal(format!("Native ReID worker: {e}")))??;
    Ok(Json(response))
}
pub(super) const MAX_REID_REVIEW_BYTES: u64 = 256 * 1024;
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReidReviewRequest {
    pub(crate) project_id: String,
    pub(crate) video: String,
    pub(crate) replay_id: String,
    pub(crate) expected_source_sha256: String,
    #[serde(default)]
    pub(crate) rally_run_id: Option<String>,
    pub(crate) action: String,
    #[serde(default)]
    pub(crate) track_ids: Vec<u64>,
    #[serde(default)]
    pub(crate) expected_revision: Option<String>,
    #[serde(default = "default_reid_threshold")]
    pub(crate) threshold_percent: u8,
    #[serde(default)]
    pub(crate) auto_threshold_percent: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReidReviewDoc {
    pub(super) schema: String,
    pub(super) project_id: String,
    pub(super) video: String,
    pub(super) replay_id: String,
    pub(super) replay_sha256: String,
    pub(super) model_fingerprint: String,
    pub(super) groups: Vec<Vec<u64>>,
    pub(super) suppressed_ids: Vec<u64>,
}
impl ReidReviewDoc {
    pub(super) fn new(ctx: &Inputs, fingerprint: &str) -> Self {
        Self {
            schema: "volley-native-reid-group-review/1".into(),
            project_id: ctx.project_id.clone(),
            video: ctx.video.clone(),
            replay_id: ctx.replay_id.clone(),
            replay_sha256: ctx.source_sha.clone(),
            model_fingerprint: fingerprint.into(),
            groups: Vec::new(),
            suppressed_ids: Vec::new(),
        }
    }
}
pub(super) fn reviewed_group_is_safe(ctx: &Inputs, identities: &Value, ids: &[u64]) -> bool {
    let owners = identities["assignments"].as_object();
    let Some(owners) = owners else { return false };
    let excluded = identities["excluded_ids"].as_array();
    let Some(excluded) = excluded else {
        return false;
    };
    let excluded = excluded
        .iter()
        .filter_map(Value::as_u64)
        .collect::<BTreeSet<_>>();
    let mut members = Vec::with_capacity(ids.len());
    for id in ids {
        let Some(row) = ctx.tracks.get(id) else {
            return false;
        };
        if excluded.contains(id) {
            return false;
        }
        members.push(reid::ManualGroupMember {
            track_id: match i64::try_from(*id) {
                Ok(id) => id,
                Err(_) => return false,
            },
            first_frame: row.first as i64,
            last_frame: row.last as i64,
            confirmed_owner: owners.get(&id.to_string()).and_then(Value::as_str),
        });
    }
    reid::valid_manual_group(&members)
}
pub(super) fn checked_review(
    doc: &ReidReviewDoc,
    ctx: &Inputs,
    fingerprint: &str,
    identities: &Value,
) -> Result<(), String> {
    if doc.schema != "volley-native-reid-group-review/1"
        || doc.project_id != ctx.project_id
        || doc.video != ctx.video
        || doc.replay_id != ctx.replay_id
        || doc.replay_sha256 != ctx.source_sha
        || doc.model_fingerprint != fingerprint
    {
        return Err("Saved ReID manual review belongs to another replay/model".into());
    }
    if doc.groups.len() > 200 || doc.suppressed_ids.len() > MAX_TRACKS {
        return Err("Too many reviewed ReID groups or suppressed IDs".into());
    }
    let mut used = BTreeSet::new();
    for group in &doc.groups {
        if !reviewed_group_is_safe(ctx, identities, group)
            || group.iter().any(|id| !used.insert(*id))
        {
            return Err("Saved ReID group has unsafe, stale or duplicated Track IDs".into());
        }
    }
    let mut suppressed = BTreeSet::new();
    for id in &doc.suppressed_ids {
        if !ctx.known.contains_number(*id) || used.contains(id) || !suppressed.insert(*id) {
            return Err("Invalid saved manually split ReID Track IDs".into());
        }
    }
    Ok(())
}
pub(super) fn modify_review(
    doc: &mut ReidReviewDoc,
    ctx: &Inputs,
    identities: &Value,
    action: &str,
    ids: &[u64],
) -> Result<(), String> {
    if ids.is_empty() || ids.len() > 200 || (action == "merge" && ids.len() < 2) {
        return Err("Choose 1–200 Track IDs (at least 2 to merge)".into());
    }
    let chosen = ids.iter().copied().collect::<BTreeSet<_>>();
    if chosen.len() != ids.len() || ids.iter().any(|id| !ctx.known.contains_number(*id)) {
        return Err("Duplicate or unknown Track ID in manual ReID review".into());
    }
    if action == "merge" && !reviewed_group_is_safe(ctx, identities, ids) {
        return Err("Cannot merge overlapping, excluded or differently confirmed players".into());
    }
    if action != "merge" && action != "split" {
        return Err("Unsupported ReID review action".into());
    }
    let mut updated = doc.clone();
    if action == "split" && chosen.iter().all(|id| updated.suppressed_ids.contains(id)) {
        return Err("Selected IDs were already manually split".into());
    }
    for group in &mut updated.groups {
        group.retain(|id| !chosen.contains(id));
    }
    updated.groups.retain(|group| group.len() >= 2);
    if action == "merge" {
        updated.groups.push(chosen.iter().copied().collect());
        updated.suppressed_ids.retain(|id| !chosen.contains(id));
    } else {
        let mut suppressed = updated
            .suppressed_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        suppressed.extend(chosen);
        updated.suppressed_ids = suppressed.into_iter().collect();
    }
    checked_review(&updated, ctx, &doc.model_fingerprint, identities)?;
    *doc = updated;
    Ok(())
}
pub(super) fn review_path(state: &AppState, ctx: &Inputs) -> Result<PathBuf, ApiError> {
    let project = projects::project_root(state).join(&ctx.project_id);
    safe_folder(&project).map_err(storage::bad_request)?;
    let folder = project.join("native_reid_reviews");
    if let Ok(meta) = fs::symlink_metadata(&folder) {
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err(storage::bad_request("Unsafe manual ReID review folder"));
        }
    }
    Ok(folder.join(format!("{}.json", ctx.replay_id)))
}
pub(super) struct PhotoReference {
    pub(super) player_id: String,
    pub(super) vector: Vec<f32>,
    pub(super) project_id: String,
    pub(super) video: String,
    pub(super) replay_id: String,
    pub(super) track_id: u64,
}

pub(super) fn photo_match_candidates(
    ctx: &Inputs,
    identity: &Value,
    vectors: &BTreeMap<u64, Vec<f32>>,
    references: &[PhotoReference],
) -> Result<Value, String> {
    photo_match_candidates_with_samples(ctx, identity, vectors, &BTreeMap::new(), references)
}

/// A track's highest-scoring crop may belong to someone else after an ID
/// switch. Combine independent measured crops before choosing the owner:
/// a lucky 99% score from one corrupted crop must not monopolize an athlete.
pub(super) fn photo_match_candidates_with_samples(
    ctx: &Inputs,
    identity: &Value,
    vectors: &BTreeMap<u64, Vec<f32>>,
    alternatives: &BTreeMap<u64, Vec<Vec<f32>>>,
    references: &[PhotoReference],
) -> Result<Value, String> {
    let assignments = identity["assignments"]
        .as_object()
        .ok_or("Invalid assignments")?;
    let excluded = identity["excluded_ids"]
        .as_array()
        .ok_or("Invalid excluded IDs")?
        .iter()
        .filter_map(Value::as_u64)
        .collect::<BTreeSet<_>>();
    let mut matches = serde_json::Map::new();
    for (&id, vector) in vectors {
        if assignments.contains_key(&id.to_string()) || excluded.contains(&id) {
            continue;
        }
        let Some(track) = ctx.tracks.get(&id) else {
            continue;
        };
        let mut samples = vec![vector.as_slice()];
        if let Some(others) = alternatives.get(&id).filter(|others| {
            others.len() == 2 && others.iter().all(|other| other.len() == vector.len())
        }) {
            samples.extend(others.iter().map(Vec::as_slice));
        }
        let mut ranked = BTreeMap::<String, (Vec<f64>, u64)>::new();
        for reference in references {
            if !ctx.roster.contains_key(&reference.player_id)
                || reference.vector.len() != vector.len()
                || assignments.iter().any(|(confirmed_id, player)| {
                    player.as_str() == Some(reference.player_id.as_str())
                        && confirmed_id.parse::<u64>().ok().is_some_and(|other| {
                            other != id && measured_conflict_frame(ctx, id, other).is_some()
                        })
                })
            {
                continue;
            }
            if reference.project_id == ctx.project_id
                && reference.video == ctx.video
                && reference.replay_id == ctx.replay_id
            {
                if let Some(other) = ctx.tracks.get(&reference.track_id) {
                    if reference.track_id != id && tracks_co_visible(track, other) {
                        continue;
                    }
                }
            }
            let entry = ranked
                .entry(reference.player_id.clone())
                .or_insert_with(|| (vec![0.0; samples.len()], reference.track_id));
            for (index, sample) in samples.iter().enumerate() {
                let score = sample
                    .iter()
                    .zip(&reference.vector)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>();
                if score.is_finite() && score > entry.0[index] {
                    entry.0[index] = score.min(1.0);
                    entry.1 = reference.track_id;
                }
            }
        }
        let mut rows = ranked
            .into_iter()
            .filter(|(_, (scores, _))| scores.iter().any(|score| *score > 0.0))
            .map(|(player, (scores, reference_track_id))| {
                let score = scores.iter().sum::<f64>() / scores.len() as f64;
                (player, score, scores, reference_track_id)
            })
            .collect::<Vec<_>>();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if !rows.is_empty() {
            matches.insert(
                id.to_string(),
                json!(rows
                    .into_iter()
                    .take(ctx.roster.len())
                    .map(|(player, score, scores, track_id)| json!({
                        "player_id":player,"name":ctx.roster[&player],
                        "cosine_similarity":score,
                        "sample_similarities":scores,
                        "samples_used":samples.len(),
                        "reference_track_id":track_id,
                        "reference_kind":"saved_photo"
                    }))
                    .collect::<Vec<_>>()),
            );
        }
    }
    Ok(Value::Object(matches))
}
/// Rank only against explicitly reviewed athlete Track IDs from the same
/// SHA-bound CPU replay. Cosine is a visual similarity, NOT a probability of
/// identity. Saved photo references have no verified embedding in this cache.
pub(super) fn player_match_candidates(
    ctx: &Inputs,
    identity: &Value,
    vectors: &BTreeMap<u64, Vec<f32>>,
) -> Result<Value, String> {
    let assignments = identity["assignments"]
        .as_object()
        .ok_or("Invalid reviewed athlete assignments")?;
    let excluded = identity["excluded_ids"]
        .as_array()
        .ok_or("Invalid excluded IDs")?
        .iter()
        .filter_map(Value::as_u64)
        .collect::<BTreeSet<_>>();
    let anchors = assignments
        .iter()
        .filter_map(|(id, owner)| {
            let id = id.parse::<u64>().ok()?;
            let owner = owner.as_str()?;
            let vector = vectors.get(&id)?;
            let source = ctx.tracks.get(&id)?;
            (!excluded.contains(&id) && ctx.roster.contains_key(owner))
                .then_some((id, owner, source, vector))
        })
        .collect::<Vec<_>>();
    let mut matches = serde_json::Map::new();
    for (&track_id, vector) in vectors {
        if assignments.contains_key(&track_id.to_string()) || excluded.contains(&track_id) {
            continue;
        }
        let Some(candidate) = ctx.tracks.get(&track_id) else {
            continue;
        };
        let mut nearest = BTreeMap::<String, (f64, u64)>::new();
        for &(anchor_id, player_id, reference, other) in &anchors {
            if tracks_co_visible(candidate, reference) || other.len() != vector.len() {
                continue;
            }
            let cosine = vector
                .iter()
                .zip(other)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>();
            if !cosine.is_finite() || cosine <= 0.0 {
                continue;
            }
            let similarity = cosine.min(1.0);
            let entry = nearest
                .entry(player_id.to_owned())
                .or_insert((similarity, anchor_id));
            if similarity > entry.0 {
                *entry = (similarity, anchor_id);
            }
        }
        let mut ranked = nearest.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0).then_with(|| a.0.cmp(&b.0)));
        if ranked.is_empty() {
            continue;
        }
        let rows = ranked
            .into_iter()
            .take(3)
            .map(|(player_id, (score, reference_id))| {
                json!({
                    "player_id":player_id,
                    "name":ctx.roster[&player_id],
                    "cosine_similarity":(score * 10_000.0).round() / 10_000.0,
                    "reference_track_id":reference_id
                })
            })
            .collect::<Vec<_>>();
        matches.insert(track_id.to_string(), json!(rows));
    }
    Ok(Value::Object(matches))
}

pub(super) fn review_response(
    ctx: &Inputs,
    identity: &Value,
    vectors: &BTreeMap<u64, Vec<f32>>,
    doc: &ReidReviewDoc,
    version: String,
    can_write: bool,
    threshold: u8,
) -> Result<Value, String> {
    checked_review(doc, ctx, &doc.model_fingerprint, identity)?;
    let pinned = doc
        .groups
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    let suppressed = doc.suppressed_ids.iter().copied().collect::<BTreeSet<_>>();
    let mut preview = native_reid_report(ctx, identity, vectors, threshold)?;
    let groups = preview["groups"]
        .as_array_mut()
        .ok_or("ReID groups missing")?;
    groups.retain(|group| {
        group["track_ids"].as_array().is_some_and(|ids| {
            ids.iter().all(|id| {
                id.as_str()
                    .and_then(|s| s.parse::<u64>().ok())
                    .is_some_and(|id| !pinned.contains(&id) && !suppressed.contains(&id))
            })
        })
    });
    let assigned = identity["assignments"]
        .as_object()
        .ok_or("Assignments missing")?;
    for group in &doc.groups {
        let members = group
            .iter()
            .filter(|id| !assigned.contains_key(&id.to_string()))
            .copied()
            .collect::<Vec<_>>();
        if members.len() < 2 {
            continue;
        }
        let group_vectors = members
            .iter()
            .map(|id| vectors.get(id).map(Vec::as_slice))
            .collect::<Option<Vec<_>>>();
        let average = group_vectors.as_deref().and_then(reid::average_similarity);
        let key = members
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let digest = format!("{:x}", Sha256::digest(key.as_bytes()));
        groups.push(json!({
            "id":format!("manual-{}", &digest[..16]),
            "track_ids":members.iter().map(u64::to_string).collect::<Vec<_>>(),
            "similarity":null,
            "average_similarity":average,
            "source":"manual"
        }));
    }
    groups.sort_by(|a, b| {
        b["track_ids"]
            .as_array()
            .map_or(0, Vec::len)
            .cmp(&a["track_ids"].as_array().map_or(0, Vec::len))
            .then_with(|| a["id"].as_str().cmp(&b["id"].as_str()))
    });
    let used = groups
        .iter()
        .flat_map(|group| {
            group["track_ids"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .filter_map(|s| s.parse::<u64>().ok())
        })
        .collect::<BTreeSet<_>>();
    let excluded = identity["excluded_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .collect::<BTreeSet<_>>();
    let unmatched = ctx
        .known
        .ids()
        .iter()
        .filter(|id| !used.contains(id) && !excluded.contains(id))
        .map(u64::to_string)
        .collect::<Vec<_>>();
    preview["unmatched_ids"] = json!(unmatched);
    preview["confirmed"] = json!(false);
    let player_matches = player_match_candidates(ctx, identity, vectors)?;
    Ok(json!({
        "schema":"volley-native-reid-group-review/1",
        "project_id":ctx.project_id,"video":ctx.video,"replay_id":ctx.replay_id,
        "source_replay_sha256":ctx.source_sha,"revision":version,
        "threshold_percent":threshold,"writable":can_write,
        "groups":preview["groups"],"unmatched_ids":preview["unmatched_ids"],
        "tracklet_links":preview["tracklet_links"],
        "tracklet_stitch_threshold_percent":preview["tracklet_stitch_threshold_percent"],
        "tracklet_links_applied":false,
        "player_matches":player_matches,
        "embedded_count":preview["embedded_count"],
        "missing_embedding_count":preview["missing_visual_ids"].as_array().map_or(0, Vec::len),
        "co_visible_veto_count":preview["co_visible_veto_count"],
        "confirmed_owner_veto_count":preview["confirmed_owner_veto_count"],
        "ambiguous_lookalike_veto_count":preview["ambiguous_lookalike_veto_count"],
        "suppressed_ids":doc.suppressed_ids,
        "similarity_type":"cosine_similarity_not_probability",
        "player_match_method":"reviewed_track_to_track_openvino_cosine",
        "auto_assign_enabled":false,"confirmed":false
    }))
}
pub(super) fn scope_review_groups(
    result: &mut Value,
    scope: &BTreeSet<u64>,
    vectors: &BTreeMap<u64, Vec<f32>>,
) -> Result<(), String> {
    let groups = result["groups"]
        .as_array_mut()
        .ok_or("ReID groups missing")?;
    for group in groups.iter_mut() {
        let members = group["track_ids"]
            .as_array()
            .ok_or("ReID group IDs missing")?
            .iter()
            .filter_map(Value::as_str)
            .filter_map(|id| id.parse::<u64>().ok())
            .filter(|id| scope.contains(id))
            .collect::<Vec<_>>();
        let features = members
            .iter()
            .map(|id| vectors.get(id).map(Vec::as_slice))
            .collect::<Option<Vec<_>>>();
        group["track_ids"] = json!(members.iter().map(u64::to_string).collect::<Vec<_>>());
        group["average_similarity"] = json!(features.as_deref().and_then(reid::average_similarity));
    }
    groups.retain(|group| {
        group["track_ids"]
            .as_array()
            .is_some_and(|ids| ids.len() >= 2)
    });
    groups.sort_by(|a, b| {
        b["track_ids"]
            .as_array()
            .map_or(0, Vec::len)
            .cmp(&a["track_ids"].as_array().map_or(0, Vec::len))
            .then_with(|| a["id"].as_str().cmp(&b["id"].as_str()))
    });
    Ok(())
}

pub async fn reid_review(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ReidReviewRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    if matches!(
        req.action.as_str(),
        "auto_assign" | "postprocess_auto_assign"
    ) {
        if req
            .auto_threshold_percent
            .is_some_and(|value| !(1..=99).contains(&value))
        {
            return Err(storage::bad_request(
                "Invalid automatic similarity threshold",
            ));
        }
        let cached_state = state.clone();
        let cached_req = req.clone();
        if let Some(saved) = tokio::task::spawn_blocking(move || {
            save_cached_automatic_photo_links(&cached_state, &cached_req)
        })
        .await
        .map_err(|e| storage::internal(format!("Cached automatic identity worker: {e}")))??
        {
            return Ok(Json(saved));
        }
        let comparison =
            super::photo_comparison::compare(state.clone(), headers, req.clone()).await?;
        let result = tokio::task::spawn_blocking(move || {
            save_automatic_photo_links(&state, &req, &comparison)
        })
        .await
        .map_err(|e| storage::internal(format!("Automatic identity worker: {e}")))??;
        return Ok(Json(result));
    }
    if req.action == "refresh_matches" {
        let comparison =
            super::photo_comparison::compare(state.clone(), headers, req.clone()).await?;
        let result = tokio::task::spawn_blocking(move || {
            save_photo_match_suggestions(&state, &req, &comparison)
        })
        .await
        .map_err(|e| storage::internal(format!("Photo candidate refresh worker: {e}")))??;
        return Ok(Json(result));
    }
    if req.action == "compare_photos" {
        return Ok(Json(
            super::photo_comparison::compare(state, headers, req).await?,
        ));
    }
    if !matches!(req.action.as_str(), "read" | "merge" | "split")
        || !(1..=99).contains(&req.threshold_percent)
    {
        return Err(storage::bad_request(
            "Invalid manual ReID review action or threshold",
        ));
    }
    let _slot = state
        .reid_limit
        .acquire()
        .await
        .map_err(|e| storage::internal(format!("Manual ReID lock slot: {e}")))?;
    let state = state.clone();
    let result = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        let _guard = project_guard::write_guard(&state, &req.project_id)?;
        let ctx = source(&state, &req.project_id, &req.video, &req.replay_id)?;
        if ctx.source_sha != req.expected_source_sha256 {
            return Err(storage::conflict("Native CPU replay SHA changed"));
        }
        let model = volley_inference::reid_openvino::fingerprint(&state.repository_root)
            .map_err(storage::bad_request)?;
        let raw_identity = existing(&ctx.sidecar).map_err(storage::bad_request)?;
        let identity = validated(raw_identity.as_deref(), &ctx).map_err(storage::bad_request)?;
        let path = review_path(&state, &ctx)?;
        let raw = versioned_store::read_optional_bounded(&path, MAX_REID_REVIEW_BYTES)
            .map_err(storage::bad_request)?;
        let mut doc = match raw.as_deref() {
            Some(data) => serde_json::from_slice::<ReidReviewDoc>(data)
                .map_err(|e| storage::bad_request(format!("Invalid saved ReID review: {e}")))?,
            None => ReidReviewDoc::new(&ctx, &model),
        };
        checked_review(&doc, &ctx, &model, &identity).map_err(storage::conflict)?;
        let scope = native_reid_cache::CacheScope {
            project_id: ctx.project_id.clone(),
            video: ctx.video.clone(),
            replay_id: ctx.replay_id.clone(),
            replay_sha256: ctx.source_sha.clone(),
            model_fingerprint: model.clone(),
        };
        let cache =
            native_reid_cache::read(&state.data_root, &scope).map_err(storage::bad_request)?;
        // A missing feature cache is not an empty Track ID source. Always
        // return manual groups and unmatched singleton IDs for read requests.
        // Cached vectors are optional for reading/editing explicit manual groups.
        if cache.invalidated {
            return Err(storage::conflict(
                "ReID embeddings are stale; regenerate current features",
            ));
        }
        if req.action != "read" {
            if !state.writes_enabled {
                return Err(storage::conflict(
                    "Editing groups requires isolated project storage",
                ));
            }
            let expected = req.expected_revision.as_deref().ok_or_else(|| {
                storage::bad_request("Read current group revision before editing")
            })?;
            versioned_store::expect_revision(raw.as_deref(), expected)
                .map_err(storage::conflict)?;
            modify_review(&mut doc, &ctx, &identity, &req.action, &req.track_ids)
                .map_err(storage::conflict)?;
            let folder = path
                .parent()
                .ok_or_else(|| storage::internal("Missing ReID review folder"))?;
            versioned_store::write_json_checked(
                folder,
                &path,
                &serde_json::to_value(&doc)
                    .map_err(|e| storage::internal(format!("Encode ReID groups: {e}")))?,
                MAX_REID_REVIEW_BYTES,
            )
            .map_err(storage::bad_request)?;
        }
        let now = versioned_store::read_optional_bounded(&path, MAX_REID_REVIEW_BYTES)
            .map_err(storage::bad_request)?;
        if volley_inference::reid_openvino::fingerprint(&state.repository_root)
            .map_err(storage::bad_request)?
            != model
        {
            return Err(storage::conflict(
                "OpenVINO model changed during manual ReID review",
            ));
        }
        let mut response = review_response(
            &ctx,
            &identity,
            &cache.vectors,
            &doc,
            revision(now.as_deref()),
            state.writes_enabled,
            req.threshold_percent,
        )
        .map_err(storage::bad_request)?;
        if req.rally_run_id.is_some() {
            let scoped = source_for_rallies(
                &state,
                &req.project_id,
                &req.video,
                &req.replay_id,
                req.rally_run_id.as_deref(),
            )?;
            scope_review_groups(
                &mut response,
                &scoped.tracks.keys().copied().collect(),
                &cache.vectors,
            )
            .map_err(storage::bad_request)?;
        }
        Ok(response)
    })
    .await
    .map_err(|e| storage::internal(format!("Manual ReID task: {e}")))??;
    Ok(Json(result))
}
