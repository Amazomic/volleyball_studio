use super::*;

pub(crate) fn conflicting_frame_tracks(
    owners: &BTreeMap<u64, String>,
    tracks: impl IntoIterator<Item = u64>,
) -> BTreeSet<u64> {
    let mut grouped = BTreeMap::<&str, Vec<u64>>::new();
    for id in tracks {
        if let Some(owner) = owners.get(&id) {
            grouped.entry(owner).or_default().push(id);
        }
    }
    grouped
        .into_values()
        .filter(|ids| ids.len() > 1)
        .flatten()
        .collect()
}

/// An overlap of lifespan ranges is not evidence of co-visibility:
/// the same athlete may have a lost/reacquired track during those ranges.
pub(super) fn tracks_co_visible(a: &Aggregate, b: &Aggregate) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.visible_spans.len() && j < b.visible_spans.len() {
        let (a0, a1) = a.visible_spans[i];
        let (b0, b1) = b.visible_spans[j];
        if a0 <= b1 && b0 <= a1 {
            let same_athlete_duplicate = a.observed_pairs.iter().any(|(pair, evidence)| {
                evidence.same_position >= 2
                    && evidence.distinct_position == 0
                    && b.observed_pairs.get(pair) == Some(evidence)
            });
            return !same_athlete_duplicate;
        }
        if a1 < b1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    false
}
/// This is deliberately based on two *measured* detections in the same
/// video frame, not overlapping Track ID lifetimes or interpolated boxes.
/// Distinct detections represent two bodies and cannot share an athlete.
pub(super) fn measured_conflict_frame(ctx: &Inputs, a: u64, b: u64) -> Option<u64> {
    let pair = (a.min(b), a.max(b));
    let evidence = ctx
        .tracks
        .get(&a)?
        .observed_pairs
        .get(&pair)
        .or_else(|| ctx.tracks.get(&b)?.observed_pairs.get(&pair))?;
    (evidence.distinct_position >= 2)
        .then_some(evidence.first_distinct_frame)
        .flatten()
}

/// Check the newly requested manual links against **both** confirmed and
/// existing automatic owners, before changing a single saved link. Otherwise
/// manually assigning one ID silently evicts another automatic ID with that
/// owner and makes a different gallery card appear unexpectedly.
pub(super) fn new_manual_owner_conflict(
    ctx: &Inputs,
    reviewed: &BTreeMap<u64, String>,
    _automatic: &Value,
    changed: &[u64],
) -> Option<(u64, u64, u64)> {
    let existing = reviewed;
    for &id in changed {
        let owner = reviewed.get(&id)?;
        for (&other, other_owner) in existing {
            if id != other && owner == other_owner {
                if let Some(frame) = measured_conflict_frame(ctx, id, other) {
                    return Some((id, other, frame));
                }
            }
        }
    }
    None
}
pub(super) fn conflicting_automatic_links(
    ctx: &Inputs,
    reviewed: &BTreeMap<u64, String>,
    automatic: &Value,
) -> BTreeSet<u64> {
    automatic
        .as_object()
        .into_iter()
        .flat_map(|map| map.iter())
        .filter_map(|(id, owner)| {
            let id = id.parse::<u64>().ok()?;
            let owner = owner.as_str()?;
            reviewed
                .iter()
                .any(|(other, assigned)| {
                    *other != id
                        && assigned == owner
                        && measured_conflict_frame(ctx, id, *other).is_some()
                })
                .then_some(id)
        })
        .collect()
}
/// A conflict is an actual contradictory assignment, not an unassigned
/// photo suggestion that could not be safely accepted.
fn assigned_visibility_conflicts(
    ctx: &Inputs,
    assignments: &BTreeMap<u64, String>,
) -> BTreeSet<u64> {
    let mut by_owner = BTreeMap::<&str, Vec<u64>>::new();
    for (id, owner) in assignments {
        if ctx.tracks.contains_key(id) {
            by_owner.entry(owner.as_str()).or_default().push(*id);
        }
    }
    let mut conflicts = BTreeSet::new();
    for ids in by_owner.values() {
        for (index, first) in ids.iter().enumerate() {
            for second in ids.iter().skip(index + 1) {
                if measured_conflict_frame(ctx, *first, *second).is_some() {
                    conflicts.insert(*first);
                    conflicts.insert(*second);
                }
            }
        }
    }
    conflicts
}

pub(super) fn choose_automatic_photo_links(
    ctx: &Inputs,
    doc: &Value,
    matches: &Value,
) -> Result<BTreeMap<u64, String>, String> {
    let assigned = doc["assignments"]
        .as_object()
        .ok_or("Invalid assignments")?;
    let excluded: BTreeSet<_> = doc["excluded_ids"]
        .as_array()
        .ok_or("Invalid exclusions")?
        .iter()
        .filter_map(Value::as_u64)
        .collect();
    let suppressed: BTreeSet<_> = doc["automatic_suppressed_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .collect();
    // Evaluate strongest evidence first. An athlete can have many tracklets,
    // but not two co-visible tracklets with distinct positions.
    let mut candidates = Vec::<(f64, u64, String)>::new();
    for (id, rows) in matches.as_object().ok_or("Invalid photo matches")? {
        let track_id = id.parse::<u64>().map_err(|_| "Invalid track ID")?;
        if !ctx.tracks.contains_key(&track_id)
            || assigned.contains_key(id)
            || excluded.contains(&track_id)
            || suppressed.contains(&track_id)
        {
            continue;
        }
        for row in rows.as_array().ok_or("Invalid candidate list")? {
            let Some(owner) = row["player_id"]
                .as_str()
                .filter(|owner| ctx.roster.contains_key(*owner))
            else {
                continue;
            };
            let Some(score) = row["cosine_similarity"]
                .as_f64()
                .filter(|score| score.is_finite() && *score > 0.30 && *score <= 1.0)
            else {
                continue;
            };
            candidates.push((score, track_id, owner.to_owned()));
        }
    }
    // Never fall back to a different athlete just because the strongest
    // match is already represented by a co-visible Track ID. Such fallback
    // mislabeled other people with a weaker (>30%) photo similarity.
    let mut tracks = candidates
        .iter()
        .map(|v| v.1)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let highest = |id| {
        candidates
            .iter()
            .filter(|v| v.1 == id)
            .map(|v| v.0)
            .fold(0.0_f64, f64::max)
    };
    tracks.sort_by(|a, b| {
        highest(*b)
            .total_cmp(&highest(*a))
            .then_with(|| ctx.tracks[b].observed.cmp(&ctx.tracks[a].observed))
            .then_with(|| a.cmp(b))
    });
    let mut chosen = BTreeMap::<u64, String>::new();
    for &id in &tracks {
        let candidate = &ctx.tracks[&id];
        let overlaps = |other_id: u64| {
            ctx.tracks
                .get(&other_id)
                .is_some_and(|other| tracks_co_visible(candidate, other))
        };
        let mut ranked = candidates.iter().filter(|v| v.1 == id).collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.2.cmp(&b.2)));
        let Some((score, _, owner)) = ranked.first() else {
            continue;
        };
        // A near-equal photo match is ambiguous only when that other athlete
        // could actually be the current body. If the alternative has already
        // been identified in an independently observed, co-visible track,
        // it is not a valid contender for this same video frame.
        let occupied_in_frame = |possible_owner: &str| {
            assigned.iter().any(|(other_id, other_owner)| {
                other_owner.as_str() == Some(possible_owner)
                    && other_id.parse::<u64>().ok().is_some_and(overlaps)
            }) || chosen
                .iter()
                .any(|(other_id, other_owner)| other_owner == possible_owner && overlaps(*other_id))
        };
        if ranked
            .iter()
            .skip(1)
            .any(|second| *score - second.0 < 0.04 && !occupied_in_frame(&second.2))
        {
            continue;
        }
        // Reject only *measured co-visible* tracks owned by the same athlete.
        // Sequential reacquisitions remain eligible for that athlete.
        if !occupied_in_frame(owner) {
            chosen.insert(id, (*owner).clone());
        }
    }
    Ok(chosen)
}
/// Re-evaluate persisted provisional links against the *full* observed replay.
/// Older automatic passes could select a weaker, wrong athlete when the best
/// athlete was already occupied. Do not expose such stale names in previews.
pub(super) fn reconcile_automatic_photo_links(ctx: &Inputs, doc: &mut Value) -> Result<(), String> {
    let Some(saved) = doc["automatic_assignments"].as_object() else {
        return Ok(());
    };
    if saved.is_empty() {
        return Ok(());
    }
    // Without the original per-athlete similarities these automatic identity
    // claims cannot be checked, so they are not safe to display as names.
    let Some(matches) = doc["automatic_identity_evidence"]["matches"].as_object() else {
        doc["automatic_assignments"] = json!({});
        return Ok(());
    };
    let preserved_matches = matches
        .iter()
        .filter(|(id, _)| saved.contains_key(*id))
        .map(|(id, rows)| (id.clone(), rows.clone()))
        .collect();
    let expected = choose_automatic_photo_links(ctx, doc, &Value::Object(preserved_matches))?;
    let safe = saved
        .iter()
        .filter_map(|(id, owner)| {
            let id_number = id.parse::<u64>().ok()?;
            let player = owner.as_str()?;
            (expected.get(&id_number).map(String::as_str) == Some(player))
                .then(|| (id.clone(), player.to_owned()))
        })
        .collect::<BTreeMap<_, _>>();
    doc["automatic_assignments"] = json!(safe);
    Ok(())
}

pub(super) fn existing(sidecar: &Path) -> Result<Option<Vec<u8>>, String> {
    versioned_store::read_optional_bounded(sidecar, MAX_LINKS)
}
/// A saved auto-match may survive a reference album *addition*: its original
/// photo evidence still exists, and the new photo only makes a future ReID
/// replay potentially better. Removing/replacing reference evidence invalidates
/// the old auto-match. A whole-album revision check alone incorrectly removed
/// every automatic owner after the first manual photo enrollment.
pub(super) fn reference_evidence_is_current(
    doc: &Value,
    album_revisions: &[(String, String)],
    current_hashes: &BTreeMap<String, BTreeSet<String>>,
) -> bool {
    let evidence = &doc["automatic_identity_evidence"];
    if evidence["album_revisions"] == json!(album_revisions) {
        return true;
    }
    let Ok(previous_revisions) =
        serde_json::from_value::<Vec<(String, String)>>(evidence["album_revisions"].clone())
    else {
        return false;
    };
    let Some(original) = evidence["reference_photo_sha256"].as_object() else {
        return false;
    };
    if previous_revisions.len() != current_hashes.len()
        || original.len() != current_hashes.len()
        || previous_revisions
            .iter()
            .any(|(player, _)| !current_hashes.contains_key(player))
    {
        return false;
    }
    current_hashes.iter().all(|(player, current)| {
        let Some(original) = original.get(player).and_then(Value::as_array) else {
            return false;
        };
        original.iter().all(|sha| {
            sha.as_str()
                .is_some_and(|value| valid_sha(value) && current.contains(value))
        })
    })
}

fn reference_album_snapshot(
    state: &AppState,
    ctx: &Inputs,
) -> Option<(Vec<(String, String)>, BTreeMap<String, BTreeSet<String>>)> {
    let mut revisions = Vec::new();
    let mut hashes = BTreeMap::<String, BTreeSet<String>>::new();
    for player in ctx.roster.keys() {
        let folder = checked_reference_folder(state, player, false).ok()?;
        let raw =
            versioned_store::read_optional_bounded(&folder.join("index.json"), MAX_REFERENCE_INDEX)
                .ok()?;
        let album = references_doc(raw.as_deref(), player).ok()?;
        let mut photos = BTreeSet::new();
        for photo in album["photos"].as_array()? {
            let id = photo["id"].as_str()?;
            let expected = photo["image_sha256"].as_str()?;
            let bytes = versioned_store::read_optional_bounded(
                &folder.join(format!("ref-{id}.jpg")),
                MAX_PHOTO_BYTES as u64,
            )
            .ok()??;
            if format!("{:x}", Sha256::digest(&bytes)) != expected {
                return None;
            }
            photos.insert(expected.to_owned());
        }
        hashes.insert(player.clone(), photos);
        revisions.push((player.clone(), revision(raw.as_deref())));
    }
    Some((revisions, hashes))
}

fn automatic_links_current(state: &AppState, ctx: &Inputs, doc: &Value) -> bool {
    if doc["automatic_assignments"]
        .as_object()
        .is_none_or(|map| map.is_empty())
    {
        return true;
    }
    let Ok(model) = volley_inference::reid_openvino::fingerprint(&state.repository_root) else {
        return false;
    };
    if doc["automatic_identity_evidence"]["model_fingerprint"] != model {
        return false;
    }
    let Some((albums, photo_sha256)) = reference_album_snapshot(state, ctx) else {
        return false;
    };
    reference_evidence_is_current(doc, &albums, &photo_sha256)
}
pub(super) fn base_document(ctx: &Inputs) -> Value {
    json!({
        "schema":"volley-native-player-track-links/1",
        "project_id":ctx.project_id,"video":ctx.video,
        "replay_id":ctx.replay_id,
        "source_replay_sha256":ctx.source_sha,
        "assignments":{},"excluded_ids":[],
        "athlete_identity_claims_manually_reviewed":true,
        "automatic_reid_binding":false,
    })
}
pub(super) fn validated(raw: Option<&[u8]>, ctx: &Inputs) -> Result<Value, String> {
    let Some(bytes) = raw else {
        return Ok(base_document(ctx));
    };
    let doc: Value = serde_json::from_slice(bytes)
        .map_err(|e| format!("Invalid track assignments JSON: {e}"))?;
    if doc["schema"] != "volley-native-player-track-links/1"
        || doc["project_id"] != ctx.project_id
        || doc["video"] != ctx.video
        || doc["replay_id"] != ctx.replay_id
        || doc["source_replay_sha256"] != ctx.source_sha
        || doc["athlete_identity_claims_manually_reviewed"] != true
        || doc["automatic_reid_binding"] != false
    {
        return Err("Manual Track ID links refer to a different replay or schema".into());
    }
    let assignments = doc["assignments"]
        .as_object()
        .ok_or("Invalid assigned Track IDs")?;
    let excluded = doc["excluded_ids"]
        .as_array()
        .ok_or("Invalid excluded Track IDs")?;
    if assignments.len() > MAX_TRACKS || excluded.len() > MAX_TRACKS {
        return Err("Manual player links exceed safety cap".into());
    }
    let mut excluded_set = BTreeSet::new();
    for value in excluded {
        let id = value.as_u64().ok_or("Invalid excluded Track ID")?;
        if !ctx.known.contains_number(id) || !excluded_set.insert(id) {
            return Err("Unknown or duplicate excluded Track ID".into());
        }
    }
    for (key, value) in assignments {
        let id = key
            .parse::<u64>()
            .map_err(|_| "Invalid assigned Track ID")?;
        if id.to_string() != *key || !ctx.known.contains_number(id) || excluded_set.contains(&id) {
            return Err("Stale, excluded or unknown assigned Track ID".into());
        }
        let owner = value.as_str().ok_or("Invalid player ID in Track link")?;
        if !ctx.roster.contains_key(owner) {
            return Err("Linked player is no longer on the project's roster".into());
        }
    }
    if let Some(automatic) = doc.get("automatic_assignments") {
        for (key, value) in automatic.as_object().ok_or("Invalid automatic links")? {
            let id = key
                .parse::<u64>()
                .map_err(|_| "Invalid automatic Track ID")?;
            if id.to_string() != *key
                || !ctx.known.contains_number(id)
                || assignments.contains_key(key)
                || excluded_set.contains(&id)
                || !value
                    .as_str()
                    .is_some_and(|owner| ctx.roster.contains_key(owner))
            {
                return Err("Stale or conflicting automatic Track ID link".into());
            }
        }
    }
    Ok(doc)
}
pub(super) fn page(
    ctx: &Inputs,
    doc: &Value,
    version: &str,
    offset: usize,
    limit: usize,
) -> Result<Value, String> {
    filtered_page(ctx, doc, version, offset, limit, None, None)
}
pub(super) fn filtered_page(
    ctx: &Inputs,
    doc: &Value,
    version: &str,
    offset: usize,
    limit: usize,
    view: Option<&str>,
    player_id: Option<&str>,
) -> Result<Value, String> {
    if limit == 0 || limit > 100 || offset > MAX_TRACKS {
        return Err("Invalid Track ID page size".into());
    }
    if !matches!(
        view,
        None | Some("unassigned") | Some("conflict") | Some("player")
    ) || (view == Some("player") && player_id.is_none_or(|id| !ctx.roster.contains_key(id)))
        || (view != Some("player") && player_id.is_some())
    {
        return Err("Invalid Track ID gallery filter".into());
    }
    let assignments = doc["assignments"]
        .as_object()
        .ok_or("Assignments missing")?;
    let excluded = doc["excluded_ids"]
        .as_array()
        .ok_or("Excluded list missing")?
        .iter()
        .filter_map(Value::as_u64)
        .collect::<BTreeSet<_>>();
    let reviewed_links = assignments
        .iter()
        .map(|(id, owner)| {
            Ok(ReviewedLink {
                track_id: id.parse::<u64>().map_err(|_| "Invalid reviewed ID")?,
                owner: owner
                    .as_str()
                    .ok_or("Invalid reviewed player ID")?
                    .to_owned(),
                reviewed: true,
            })
        })
        .collect::<Result<Vec<_>, &str>>()?;
    let reviewed =
        identity_review::resolve_reviewed(ctx.known.ids(), &ctx.roster, &reviewed_links, &excluded);
    let mut owners = reviewed.owners.clone();
    if let Some(automatic) = doc["automatic_assignments"].as_object() {
        for (id, owner) in automatic {
            if let (Ok(id), Some(owner)) = (id.parse::<u64>(), owner.as_str()) {
                if !excluded.contains(&id) {
                    owners.entry(id).or_insert_with(|| owner.to_owned());
                }
            }
        }
    }
    let actual_conflicts = assigned_visibility_conflicts(ctx, &owners);
    let scoped_assigned = owners
        .keys()
        .filter(|id| ctx.tracks.contains_key(id))
        .count();
    let scoped_excluded = excluded
        .iter()
        .filter(|id| ctx.tracks.contains_key(id))
        .count();
    let matches = |id: &u64| match view {
        Some("unassigned") => !owners.contains_key(id),
        Some("conflict") => actual_conflicts.contains(id),
        Some("player") => owners.get(id).map(String::as_str) == player_id,
        _ => true,
    };
    let filtered_total = ctx.tracks.keys().filter(|id| matches(id)).count();
    let mut player_track_counts = std::collections::BTreeMap::<&str, usize>::new();
    for id in ctx.tracks.keys() {
        if let Some(owner) = owners.get(id) {
            *player_track_counts.entry(owner.as_str()).or_default() += 1;
        }
    }
    let mut rows = Vec::new();
    for (id, info) in ctx
        .tracks
        .iter()
        .filter(|(id, _)| matches(id))
        .skip(offset)
        .take(limit)
    {
        let owner = owners.get(id).map(String::as_str);
        rows.push(json!({
            "track_id":id,"first_frame":info.first,"last_frame":info.last,
            "observations":info.observed,"best_frame":info.best_frame,
            "best_score":info.best_score,"bbox_xyxy_norm":info.best_box,
            "assigned_player_id":owner,
            "assigned_player_name":owner.and_then(|v|ctx.roster.get(v)),
            "excluded":excluded.contains(id),
            "identity_conflict":actual_conflicts.contains(id),
            "identity_source":if reviewed.owners.contains_key(id){"manual"}else if owner.is_some(){"automatic_photo"}else{"none"},
            "track_id_provisional":true,
        }));
    }
    let roster = ctx
        .roster
        .iter()
        .map(|(id, name)| json!({"id":id,"name":name}))
        .collect::<Vec<_>>();
    Ok(json!({
        "schema":"volley-native-player-track-links/1",
        "project_id":ctx.project_id,"video":ctx.video,"replay_id":ctx.replay_id,
        "source_replay_sha256":ctx.source_sha,
        "source_kind":ctx.evidence.source_kind(),
        "evidence_source":ctx.evidence,
        "revision":version,"writable":ctx.can_write,
        "total_tracks":ctx.tracks.len(),"offset":offset,"limit":limit,
        "filtered_total":filtered_total,
        "player_track_counts":player_track_counts,
        "unassigned_track_ids":ctx.tracks.keys().filter(|id| !owners.contains_key(id)).copied().collect::<Vec<_>>(),
        "comparison_track_ids":ctx.tracks.keys().filter(|id| !owners.contains_key(id) && !excluded.contains(id)).copied().collect::<Vec<_>>(),
        "rally_track_ids":ctx.tracks.keys().copied().collect::<Vec<_>>(),
        "assigned_count":scoped_assigned,"excluded_count":scoped_excluded,
        "identity_conflict_track_ids":actual_conflicts,
        "roster":roster,"tracks":rows,
        "automatic_identity_matches":doc["automatic_identity_evidence"]["matches"],
        "athlete_links_require_manual_confirmation":true,
        "automatic_reid_binding":doc["automatic_assignments"].as_object().is_some_and(|map| !map.is_empty()),"photos_auto_registered":false,
    }))
}
pub(crate) struct NativeReviewedOwners {
    pub source_sha: String,
    pub revision: String,
    pub owners: BTreeMap<u64, String>,
    pub excluded: BTreeSet<u64>,
    pub roster: BTreeMap<String, String>,
}
pub(crate) fn reviewed_owners(
    state: &AppState,
    project_id: &str,
    video: &str,
    replay_id: &str,
) -> Result<NativeReviewedOwners, ApiError> {
    let ctx = source(state, project_id, video, replay_id)?;
    let raw = existing(&ctx.sidecar).map_err(storage::bad_request)?;
    let doc = validated(raw.as_deref(), &ctx).map_err(storage::bad_request)?;
    let links = doc["assignments"]
        .as_object()
        .into_iter()
        .flat_map(|map| map.iter())
        .map(|(id, owner)| ReviewedLink {
            track_id: id
                .parse::<u64>()
                .expect("Source-validated numeric Track ID"),
            owner: owner
                .as_str()
                .expect("Source-validated roster owner")
                .to_owned(),
            reviewed: true,
        })
        .collect::<Vec<_>>();
    let excluded = doc["excluded_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .collect::<BTreeSet<_>>();
    let resolved =
        identity_review::resolve_reviewed(ctx.known.ids(), &ctx.roster, &links, &excluded);
    if !resolved.conflicts.is_empty() {
        return Err(storage::conflict(
            "Manual native athlete assignments contain contradictions",
        ));
    }
    Ok(NativeReviewedOwners {
        source_sha: ctx.source_sha,
        revision: revision(raw.as_deref()),
        owners: resolved.owners,
        excluded: resolved.excluded,
        roster: ctx.roster,
    })
}
/// Automatic identity suggestions are scoped to measured observations inside
/// the selected rallies. An unrelated detection during the break between two
/// rallies cannot veto the same athlete's later Track ID. Keep the source SHA
/// and rally run bound to the saved suggestion so every consumer reconciles
/// against the exact same measured frames.
fn automatic_assignment_scope(
    state: &AppState,
    full: &Inputs,
    doc: &Value,
) -> Result<Option<Inputs>, ApiError> {
    let Some(run) = doc["automatic_identity_evidence"]["rally_run_id"].as_str() else {
        return Ok(None);
    };
    let scoped = source_for_rallies(
        state,
        &full.project_id,
        &full.video,
        &full.replay_id,
        Some(run),
    )?;
    if scoped.source_sha != full.source_sha {
        return Err(storage::conflict("Automatic identity rally source changed"));
    }
    Ok(Some(scoped))
}

/// Merge current, evidence-reconciled automatic photo assignments with
/// manual identities for rally-scoped statistics. Never replace manual links.
pub(crate) fn statistics_owners(
    state: &AppState,
    project_id: &str,
    video: &str,
    replay_id: &str,
) -> Result<(NativeReviewedOwners, BTreeSet<u64>), ApiError> {
    let mut owners = reviewed_owners(state, project_id, video, replay_id)?;
    let ctx = source(state, project_id, video, replay_id)?;
    if owners.source_sha != ctx.source_sha {
        return Err(storage::conflict("Track ID and roster source mismatch"));
    }
    let raw = existing(&ctx.sidecar).map_err(storage::bad_request)?;
    let mut doc = validated(raw.as_deref(), &ctx).map_err(storage::bad_request)?;
    if !automatic_links_current(state, &ctx, &doc) {
        return Ok((owners, BTreeSet::new()));
    }
    let assigned_scope = automatic_assignment_scope(state, &ctx, &doc)?;
    reconcile_automatic_photo_links(assigned_scope.as_ref().unwrap_or(&ctx), &mut doc)
        .map_err(storage::bad_request)?;
    let mut automatic = BTreeSet::new();
    for (id, owner) in doc["automatic_assignments"]
        .as_object()
        .into_iter()
        .flatten()
    {
        let (Some(track), Some(person)) = (id.parse::<u64>().ok(), owner.as_str()) else {
            continue;
        };
        if !owners.excluded.contains(&track)
            && !owners.owners.contains_key(&track)
            && ctx.known.ids().contains(&track)
            && owners.roster.contains_key(person)
        {
            owners.owners.insert(track, person.to_owned());
            automatic.insert(track);
        }
    }
    Ok((owners, automatic))
}

pub(crate) fn display_player_names(
    state: &AppState,
    project_id: &str,
    video: &str,
    replay_id: &str,
) -> Result<(String, String, BTreeMap<String, String>, BTreeSet<String>), ApiError> {
    let ctx = source(state, project_id, video, replay_id)?;
    let raw = existing(&ctx.sidecar).map_err(storage::bad_request)?;
    let mut doc = validated(raw.as_deref(), &ctx).map_err(storage::bad_request)?;
    if !automatic_links_current(state, &ctx, &doc) {
        doc["automatic_assignments"] = json!({});
    } else {
        let assigned_scope = automatic_assignment_scope(state, &ctx, &doc)?;
        reconcile_automatic_photo_links(assigned_scope.as_ref().unwrap_or(&ctx), &mut doc)
            .map_err(storage::bad_request)?;
    }
    let manual_ids: BTreeSet<String> = doc["assignments"]
        .as_object()
        .into_iter()
        .flat_map(|map| map.keys().cloned())
        .collect();
    let mut names: BTreeMap<String, String> = doc["assignments"]
        .as_object()
        .into_iter()
        .flat_map(|map| map.iter())
        .filter_map(|(id, owner)| Some((id.clone(), ctx.roster.get(owner.as_str()?)?.to_owned())))
        .collect();
    names.extend(
        doc["automatic_assignments"]
            .as_object()
            .into_iter()
            .flat_map(|map| map.iter())
            .filter_map(|(id, owner)| {
                let name = ctx.roster.get(owner.as_str()?)?;
                Some((id.clone(), name.to_owned()))
            }),
    );
    Ok((ctx.source_sha, revision(raw.as_deref()), names, manual_ids))
}
pub(crate) fn automatic_links_revision(
    state: &AppState,
    project_id: &str,
    video: &str,
    replay_id: &str,
) -> Result<Option<String>, ApiError> {
    let ctx = source(state, project_id, video, replay_id)?;
    let raw = existing(&ctx.sidecar).map_err(storage::bad_request)?;
    let doc = validated(raw.as_deref(), &ctx).map_err(storage::bad_request)?;
    Ok(automatic_links_current(state, &ctx, &doc).then(|| revision(raw.as_deref())))
}
pub(super) fn read_internal(state: &AppState, req: &TrackQuery) -> Result<Value, ApiError> {
    let ctx = source_for_rallies(
        state,
        &req.project_id,
        &req.video,
        &req.replay_id,
        req.rally_run_id.as_deref(),
    )?;
    let bytes = existing(&ctx.sidecar).map_err(storage::bad_request)?;
    let mut doc = validated(bytes.as_deref(), &ctx).map_err(storage::bad_request)?;
    if !automatic_links_current(state, &ctx, &doc) {
        doc["automatic_assignments"] = json!({});
    } else {
        let assigned_scope = automatic_assignment_scope(state, &ctx, &doc)?;
        reconcile_automatic_photo_links(assigned_scope.as_ref().unwrap_or(&ctx), &mut doc)
            .map_err(storage::bad_request)?;
    }
    filtered_page(
        &ctx,
        &doc,
        &revision(bytes.as_deref()),
        req.offset,
        req.limit,
        req.view.as_deref(),
        req.view_player_id.as_deref(),
    )
    .map_err(storage::bad_request)
}
pub(super) fn change(state: &AppState, req: &TrackEdit) -> Result<Value, ApiError> {
    if !state.writes_enabled {
        return Err(storage::conflict(
            "Editing Track ID requires isolated project storage",
        ));
    }
    if !matches!(
        req.action.as_str(),
        "assign" | "unassign" | "exclude" | "restore"
    ) || req.track_ids.is_empty()
        || req.track_ids.len() > 200
        || req.track_ids.iter().collect::<BTreeSet<_>>().len() != req.track_ids.len()
    {
        return Err(storage::bad_request(
            "Choose 1–200 unique IDs and a valid edit action",
        ));
    }
    if !versioned_store::valid_revision(&req.expected_revision) {
        return Err(storage::bad_request("Invalid assignment revision"));
    }
    let ctx = source_for_rallies(
        state,
        &req.project_id,
        &req.video,
        &req.replay_id,
        req.rally_run_id.as_deref(),
    )?;
    if req.track_ids.iter().any(|id| !ctx.tracks.contains_key(id)) {
        return Err(storage::bad_request(
            "Selected Track ID has no observation in the chosen rally version",
        ));
    }
    if req.action == "assign"
        && !req
            .player_id
            .as_ref()
            .is_some_and(|id| valid_player_id(id) && ctx.roster.contains_key(id))
    {
        return Err(storage::bad_request(
            "Selected athlete is not in project roster",
        ));
    }
    if req.action != "assign" && req.player_id.is_some() {
        return Err(storage::bad_request(
            "Player ID is only accepted for an assignment",
        ));
    }
    let before = existing(&ctx.sidecar).map_err(storage::bad_request)?;
    if versioned_store::expect_revision(before.as_deref(), &req.expected_revision).is_err() {
        return Err(storage::conflict(
            "Track bindings changed; refresh the revision",
        ));
    }
    let mut doc = validated(before.as_deref(), &ctx).map_err(storage::bad_request)?;
    let assignments = doc["assignments"]
        .as_object()
        .ok_or_else(|| storage::bad_request("Missing assignment object"))?
        .iter()
        .map(|(key, value)| {
            Ok((
                key.parse::<u64>()
                    .map_err(|_| storage::bad_request("Invalid saved Track ID"))?,
                value
                    .as_str()
                    .ok_or_else(|| storage::bad_request("Invalid saved athlete ID"))?
                    .to_owned(),
            ))
        })
        .collect::<Result<BTreeMap<_, _>, ApiError>>()?;
    let excluded = doc["excluded_ids"]
        .as_array()
        .ok_or_else(|| storage::bad_request("Missing excluded list"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .ok_or_else(|| storage::bad_request("Invalid excluded ID"))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut review = ManualReviewState {
        assignments,
        excluded,
        rejected: BTreeSet::new(),
    };
    review
        .apply(
            &ctx.known,
            &ctx.roster,
            ReviewAction::parse(&req.action).map_err(|e| storage::bad_request(e.message))?,
            &req.track_ids,
            req.player_id.as_deref(),
            false,
        )
        .map_err(|e| match e.kind {
            ReviewErrorKind::Invalid => storage::bad_request(e.message),
            ReviewErrorKind::Conflict => storage::conflict(e.message),
        })?;
    doc["assignments"] = json!(review
        .assignments
        .iter()
        .map(|(id, player)| (id.to_string(), player.clone()))
        .collect::<BTreeMap<_, _>>());
    doc["excluded_ids"] = json!(review.excluded);
    let mut suppressed: BTreeSet<u64> = doc["automatic_suppressed_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .collect();
    for id in &req.track_ids {
        if req.action == "unassign" {
            suppressed.insert(*id);
        }
        if req.action == "assign" || req.action == "restore" {
            suppressed.remove(id);
        }
    }
    doc["automatic_suppressed_ids"] = json!(suppressed);
    // The gallery uses rally-scoped measured positions. Keep conflict
    // resolution in that same timeframe, not at unrelated moments of the
    // replay (especially the gap between rallies).
    let whole_source = source(state, &req.project_id, &req.video, &req.replay_id)?;
    let saved_scope = automatic_assignment_scope(state, &whole_source, &doc)?;
    let conflict_tracks = saved_scope.as_ref().unwrap_or(&ctx);
    let previous_automatic = doc["automatic_assignments"].clone();
    if req.action == "assign" {
        if let Some((id, other, frame)) = new_manual_owner_conflict(
            conflict_tracks,
            &review.assignments,
            &previous_automatic,
            &req.track_ids,
        ) {
            let message = if req.track_ids.contains(&other) {
                format!(
                    "В выбранных Track ID {id} и {other} обнаружены разные люди в кадре {frame}. Нельзя привязать их к одному игроку: исключите ошибочный ID из группы или снимите его выбор."
                )
            } else {
                format!(
                    "ID {id} конфликтует с ранее подтверждённым ID {other}: разные люди в кадре {frame}. Проверьте кадр и при ошибке сначала снимите прежнюю привязку ID {other}."
                )
            };
            return Err(storage::conflict(message));
        }
    }
    let displaced_automatic = if req.action == "assign" {
        conflicting_automatic_links(conflict_tracks, &review.assignments, &previous_automatic)
    } else {
        BTreeSet::new()
    };
    if let Some(automatic) = doc
        .get_mut("automatic_assignments")
        .and_then(Value::as_object_mut)
    {
        for id in req.track_ids.iter().chain(displaced_automatic.iter()) {
            automatic.remove(&id.to_string());
        }
    }
    suppressed.extend(displaced_automatic.iter().copied());
    doc["automatic_suppressed_ids"] = json!(suppressed);
    // Legacy sidecars contain only whole-album revisions. Capture the exact
    // reference hashes before a manual assignment enrolls a new photo; that
    // subsequent append must not unassign every remaining automatic Track ID.
    if doc["automatic_identity_evidence"]["reference_photo_sha256"].is_null()
        && doc["automatic_assignments"]
            .as_object()
            .is_some_and(|links| !links.is_empty())
        && automatic_links_current(state, &conflict_tracks, &doc)
    {
        if let Some((_, hashes)) = reference_album_snapshot(state, &conflict_tracks) {
            doc["automatic_identity_evidence"]["reference_photo_sha256"] = json!(hashes);
        }
    }
    reconcile_automatic_photo_links(conflict_tracks, &mut doc).map_err(storage::bad_request)?;
    if req.action == "assign" {
        for (id, previous_owner) in previous_automatic
            .as_object()
            .into_iter()
            .flat_map(|map| map.iter())
        {
            if req
                .track_ids
                .iter()
                .any(|selected| selected.to_string() == *id)
            {
                continue;
            }
            if !displaced_automatic.contains(&id.parse::<u64>().unwrap_or_default())
                && doc["automatic_assignments"][id] != *previous_owner
            {
                return Err(storage::conflict(format!(
                    "Привязка ID {id} изменила бы другого игрока. Изменения отменены; обновите распознавание или проверьте кадры."
                )));
            }
        }
    }
    if before.is_none()
        || doc != validated(before.as_deref(), &ctx).map_err(storage::bad_request)?
    {
        let parent = ctx
            .sidecar
            .parent()
            .ok_or_else(|| storage::internal("Link directory missing"))?;
        versioned_store::write_json_checked(parent, &ctx.sidecar, &doc, MAX_LINKS)
            .map_err(storage::bad_request)?;
    }
    let new = existing(&ctx.sidecar).map_err(storage::bad_request)?;
    let mut view = validated(new.as_deref(), &ctx).map_err(storage::bad_request)?;
    if !automatic_links_current(state, &ctx, &view) {
        view["automatic_assignments"] = json!({});
    } else {
        reconcile_automatic_photo_links(&conflict_tracks, &mut view)
            .map_err(storage::bad_request)?;
    }
    let mut result = page(
        &ctx,
        &view,
        &revision(new.as_deref()),
        req.offset,
        req.limit,
    )
    .map_err(storage::bad_request)?;
    result["displaced_automatic_track_ids"] = json!(displaced_automatic);
    Ok(result)
}

pub(super) fn refresh_automatic_photo_links(
    ctx: &Inputs,
    doc: &Value,
    matches: &Value,
) -> Result<(Value, BTreeMap<u64, String>), String> {
    let mut updated = doc.clone();
    updated["automatic_assignments"] = json!({});
    let chosen = choose_automatic_photo_links(ctx, &updated, matches)?;
    updated["automatic_assignments"] = json!(chosen
        .iter()
        .map(|(id, player)| (id.to_string(), player.clone()))
        .collect::<BTreeMap<_, _>>());
    Ok((updated, chosen))
}

pub(super) fn save_automatic_photo_links(
    state: &AppState,
    req: &ReidReviewRequest,
    comparison: &Value,
) -> Result<Value, ApiError> {
    persist_photo_comparison(state, req, comparison, true)
}

/// Save the ranked candidates without changing manual or automatic owners.
/// This is used by the gallery's explicit refresh and survives page reloads.
pub(super) fn save_photo_match_suggestions(
    state: &AppState,
    req: &ReidReviewRequest,
    comparison: &Value,
) -> Result<Value, ApiError> {
    persist_photo_comparison(state, req, comparison, false)
}

fn persist_photo_comparison(
    state: &AppState,
    req: &ReidReviewRequest,
    comparison: &Value,
    auto_assign: bool,
) -> Result<Value, ApiError> {
    if !state.writes_enabled {
        return Err(storage::conflict(
            "Automatic links require isolated writable storage",
        ));
    }
    let _guard = if auto_assign {
        project_guard::write_guard_for_postprocessing(state, &req.project_id)?
    } else {
        project_guard::write_guard(state, &req.project_id)?
    };
    let snapshot = super::photo_comparison::snapshot(state, req)?;
    if comparison["identity_revision"] != json!(snapshot.identity_revision)
        || comparison["album_revisions"] != json!(snapshot.albums)
        || comparison["model_fingerprint"] != json!(snapshot.model)
    {
        return Err(storage::conflict(
            "Photo references or model changed during automatic comparison",
        ));
    }
    let ctx = source(state, &req.project_id, &req.video, &req.replay_id)?;
    if ctx.source_sha != req.expected_source_sha256 {
        return Err(storage::conflict("Source replay changed"));
    }
    let before = existing(&ctx.sidecar).map_err(storage::bad_request)?;
    if comparison["identity_revision"].as_str() != Some(revision(before.as_deref()).as_str()) {
        return Err(storage::conflict(
            "Player links changed during automatic comparison",
        ));
    }
    let mut doc = validated(before.as_deref(), &ctx).map_err(storage::bad_request)?;
    let scoped = source_for_rallies(
        state,
        &req.project_id,
        &req.video,
        &req.replay_id,
        req.rally_run_id.as_deref(),
    )?;
    let matches = &comparison["player_matches"];
    let chosen = if auto_assign {
        let (updated, chosen) =
            refresh_automatic_photo_links(&scoped, &doc, matches).map_err(storage::bad_request)?;
        doc = updated;
        chosen
    } else {
        BTreeMap::new()
    };
    let (current_albums, reference_hashes) = reference_album_snapshot(state, &ctx)
        .ok_or_else(|| storage::conflict("Saved player reference photos changed"))?;
    if comparison["album_revisions"] != json!(current_albums) {
        return Err(storage::conflict(
            "Reference photo albums changed during automatic assignment",
        ));
    }
    doc["automatic_identity_evidence"] = json!({
        "threshold":0.30, "matches":matches,
        "rally_run_id":req.rally_run_id,
        "album_revisions":comparison["album_revisions"],
        "reference_photo_sha256":reference_hashes,
        "model_fingerprint":comparison["model_fingerprint"]
    });
    let unassigned: Vec<_> = scoped
        .tracks
        .keys()
        .filter(|id| {
            let key = id.to_string();
            !doc["assignments"]
                .as_object()
                .is_some_and(|map| map.contains_key(&key))
                && !doc["automatic_assignments"]
                    .as_object()
                    .is_some_and(|map| map.contains_key(&key))
        })
        .copied()
        .collect();
    let parent = ctx
        .sidecar
        .parent()
        .ok_or_else(|| storage::internal("Link directory missing"))?;
    versioned_store::write_json_checked(parent, &ctx.sidecar, &doc, MAX_LINKS)
        .map_err(storage::bad_request)?;
    let saved = existing(&ctx.sidecar).map_err(storage::bad_request)?;
    if auto_assign {
        Ok(
            json!({"automatic_count":chosen.len(),"automatic_assignments":chosen,
            "unassigned_track_ids":unassigned,
            "identity_revision":revision(saved.as_deref()),"confirmed":false}),
        )
    } else {
        let mut response = comparison.clone();
        response["identity_revision"] = json!(revision(saved.as_deref()));
        response["saved"] = json!(true);
        Ok(response)
    }
}
pub async fn read(
    State(state): State<Arc<AppState>>,
    Query(req): Query<TrackQuery>,
) -> ApiResult<Value> {
    // Reading a source-SHA-verified, immutable replay and an atomically
    // replaced assignment sidecar does not modify the project. Do not take
    // the exclusive writer lock: parallel gallery/ReID reads must coexist.
    let result = tokio::task::spawn_blocking(move || read_internal(&state, &req))
        .await
        .map_err(|e| storage::internal(format!("Track ID read task: {e}")))??;
    Ok(Json(result))
}
pub async fn edit(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<TrackEdit>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let result = tokio::task::spawn_blocking(move || {
        let _lock = project_guard::write_guard(&state, &req.project_id)?;
        change(&state, &req)
    })
    .await
    .map_err(|e| storage::internal(format!("Track ID edit task: {e}")))??;
    Ok(Json(result))
}
