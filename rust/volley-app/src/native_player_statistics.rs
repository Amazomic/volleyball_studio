//! Project-scoped person statistics over one unique, SHA-verified native CPU
//! source per video, counting only resolved roster athletes inside rallies.
//! Unresolved Track IDs remain diagnostics, not new report players.
use crate::{
    native_player_tracks::{self, NativeReviewedOwners},
    native_rally_evidence::{self, NativeMotion, ValidatedRun},
    project_guard, projects,
    storage::{self, safe_file, ApiError, ApiResult, BoundedReader},
    versioned_store::{self, revision},
    AppState,
};
use axum::{extract::State, http::HeaderMap, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};
const MAX_SOURCE: u64 = 64 * 1024 * 1024 * 1024;
const MAX_STATS: u64 = 8 * 1024 * 1024;
const REPORT_ALGORITHM: &str = "rally-roster-attributed-touches-v5";
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatsSource {
    rally_run_id: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    project_id: String,
    sources: Vec<StatsSource>,
    #[serde(default = "preview")]
    mode: String,
    #[serde(default)]
    expected_revision: Option<String>,
    #[serde(default)]
    expected_source_fingerprint: Option<String>,
}
fn expected_source_matches(current: &Value, expected: &str) -> bool {
    expected.len() == 64
        && expected
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        && current["source_fingerprint"].as_str() == Some(expected)
}
fn preview() -> String {
    "preview".into()
}
/// Same-frame, SHA-verified and pose-supported contact credited by the
/// Report. The estimated_kind is an automatic hypothesis, not a review label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MontageContact {
    pub frame: u64,
    pub track_id: u64,
    pub estimated_kind: Option<String>,
}
#[derive(Default, Debug)]
struct ContactCounts {
    montage_contacts: Vec<MontageContact>,
    per_track: BTreeMap<u64, u64>,
    estimated_actions: BTreeMap<String, u64>,
    estimated_per_track: BTreeMap<u64, BTreeMap<String, u64>>,
    unassigned: u64,
    touches: u64,
    excluded_proposals: u64,
    unverified_proposals: u64,
    generic_touches: u64,
    outside_rally_proposals: u64,
}
impl ContactCounts {
    fn report_candidates(&self) -> u64 {
        self.touches + self.unassigned
    }
}

fn count_initial_serves(
    proposals: &[Value],
    owners: &NativeReviewedOwners,
) -> (BTreeMap<String, u64>, u64) {
    let mut per_player = BTreeMap::<String, u64>::new();
    let mut unassigned = 0u64;
    for proposal in proposals {
        let athlete = proposal["provisional_track_id"]
            .as_u64()
            .filter(|id| !owners.excluded.contains(id))
            .and_then(|id| owners.owners.get(&id))
            .filter(|id| owners.roster.contains_key(*id));
        if let Some(id) = athlete {
            *per_player.entry(id.clone()).or_default() += 1;
        } else {
            unassigned += 1;
        }
    }
    (per_player, unassigned)
}
/// Re-verify the exact video-to-touch and frame-to-Track relationship, not
/// merely that a guessed Track ID appears somewhere in the match.
fn count_native_contacts(
    run: &ValidatedRun,
    replay_file: &Path,
    owners: &NativeReviewedOwners,
    windows: &native_rally_evidence::RallyWindows,
) -> Result<ContactCounts, String> {
    if owners.source_sha != run.replay_sha {
        return Err("Player review source differs from Touch's CPU Track ID snapshot".into());
    }
    let mut source = BoundedReader::open(replay_file)?;
    let mut contacts = BoundedReader::open(&run.touch_file)?;
    let mut next = contacts
        .line()?
        .map(|row| {
            serde_json::from_slice::<NativeMotion>(&row)
                .map_err(|e| format!("Malformed native statistics Touch: {e}"))
        })
        .transpose()?;
    let mut before = None::<u64>;
    let mut total = ContactCounts::default();
    let mut verified_touches = 0_u64;
    let mut last_attributed = vec![BTreeMap::<u64, u64>::new(); windows.intervals.len()];
    for frame in 0..run.frames {
        let raw = source
            .line()?
            .ok_or("Native player-statistics source truncated")?;
        let row: Value = serde_json::from_slice(&raw)
            .map_err(|e| format!("Malformed CPU Track ID source: {e}"))?;
        if row["frame_index"].as_u64() != Some(frame)
            || row["track_ids_provisional"] != true
            || row["confirmed_athlete_identities"] != false
            || row["touches_confirmed"] != false
        {
            return Err("Invalid or non-provisional track source frame".into());
        }
        let players = row["tracked_players"]
            .as_array()
            .ok_or("Missing player Track ID evidence")?;
        if players.len() > 128 {
            return Err("Unbounded player detections per frame".into());
        }
        let mut observed = BTreeSet::new();
        for player in players {
            let id = player["track_id"]
                .as_u64()
                .ok_or("Invalid actual player detection Track ID")?;
            if id == 0
                || id > i64::MAX as u64
                || !observed.insert(id)
                || player["is_verified_athlete"] != false
                || player["detection"]["frame_index"].as_u64() != Some(frame)
                || player["detection"]["class_name"] != "player"
            {
                return Err("Untrusted player detection in Touch source".into());
            }
        }
        if let Some(event) = next.as_ref() {
            let (contact_frame, track) = event.event_key();
            if contact_frame < frame
                || before.is_some_and(|last| contact_frame <= last)
                || !event.valid(run.frames, run.width, run.height)
                || !event.is_physical()
            {
                return Err("Untrusted, duplicate or unphysical Touch candidate".into());
            }
            if contact_frame == frame {
                if row["selected_ball"].is_null() && !event.ball_occluded_at_contact() {
                    return Err("Source Touch frame has no observed selected ball".into());
                }
                if track.is_some_and(|id| !observed.contains(&id)) {
                    return Err("Touch Track ID is absent from the same source frame".into());
                }
                verified_touches += 1;
                let pose_verified =
                    event.body_contact_supported && event.contact_landmark.is_some();
                let interval = windows.intervals.partition_point(|(_, end)| *end < frame);
                let mut outside = BTreeMap::new();
                let final_touch = native_rally_evidence::finalize_touch(
                    event,
                    windows.contains(frame),
                    windows.contexts.get(&frame),
                    if windows.contains(frame) {
                        &mut last_attributed[interval]
                    } else {
                        &mut outside
                    },
                )?;
                if !windows.contains(frame) {
                    total.outside_rally_proposals += 1;
                } else if let Some(id) = track {
                    if native_player_tracks::conflicting_frame_tracks(
                        &owners.owners,
                        observed.iter().copied(),
                    )
                    .contains(&id)
                    {
                        total.unassigned += 1;
                        total.unverified_proposals += 1;
                    } else if !final_touch.attributed {
                        total.unverified_proposals += 1;
                    } else if owners.excluded.contains(&id) {
                        total.excluded_proposals += 1;
                    } else {
                        total.touches += 1;
                        *total.per_track.entry(id).or_default() += 1;
                        if !pose_verified {
                            total.generic_touches += 1;
                        }
                        let estimated_kind = final_touch.kind.map(|kind| kind.as_str().to_owned());
                        total.montage_contacts.push(MontageContact {
                            frame,
                            track_id: id,
                            estimated_kind: estimated_kind.clone(),
                        });
                        if let Some(kind) = estimated_kind {
                            *total.estimated_actions.entry(kind.clone()).or_default() += 1;
                            *total
                                .estimated_per_track
                                .entry(id)
                                .or_default()
                                .entry(kind)
                                .or_default() += 1;
                        }
                    }
                } else {
                    total.unassigned += 1;
                    total.unverified_proposals += 1;
                }
                before = Some(contact_frame);
                next = contacts
                    .line()?
                    .map(|row| {
                        serde_json::from_slice::<NativeMotion>(&row)
                            .map_err(|e| format!("Malformed native Touch: {e}"))
                    })
                    .transpose()?;
            }
        }
    }
    if source.line()?.is_some()
        || source.finish() != run.replay_sha
        || next.is_some()
        || verified_touches != run.expected_touches
        || contacts.finish() != run.touch_sha
    {
        return Err("Native player-statistics inputs changed or have undeclared events".into());
    }
    Ok(total)
}

/// Montage and statistics share a single SHA-verified contact selector.
/// Only contacts credited by the Report (inside a rally, matching player and
/// COCO17 landmark on the exact replay frame) may become player reel cuts.
pub(crate) fn verified_contacts_for_montage(
    state: &AppState,
    run: &ValidatedRun,
    owners: &NativeReviewedOwners,
) -> Result<Vec<MontageContact>, ApiError> {
    let root = projects::project_root(state)
        .join(&run.project_id)
        .join("native_cpu_replays")
        .join(&run.replay_id);
    let replay =
        safe_file(&root, "tracked_frames.jsonl", MAX_SOURCE).map_err(storage::bad_request)?;
    let windows =
        native_rally_evidence::verified_track_windows(run).map_err(storage::bad_request)?;
    let counts =
        count_native_contacts(run, &replay, owners, &windows).map_err(storage::bad_request)?;
    debug_assert_eq!(counts.touches as usize, counts.montage_contacts.len());
    Ok(counts.montage_contacts)
}
fn select_unique_video(seen: &mut BTreeSet<String>, video: &str) -> bool {
    seen.insert(video.to_owned())
}
fn calculate(state: &AppState, request: &Request) -> Result<Value, ApiError> {
    storage::project_id(&request.project_id)?;
    if request.sources.is_empty() || request.sources.len() > 40 {
        return Err(storage::bad_request(
            "Select 1..40 native CPU runs, one per video",
        ));
    }

    let project = projects::existing_project(state, &request.project_id)?;
    let video_count = project["videos"].as_array().map_or(0, Vec::len);
    if request.sources.len() > video_count {
        return Err(storage::bad_request(
            "More CPU runs selected than project videos",
        ));
    }
    let mut selected = BTreeSet::new();
    let mut names = BTreeMap::<String, String>::new();
    let mut results = BTreeMap::<String, u64>::new();
    let mut predicted_counts = BTreeMap::<String, u64>::new();
    let mut predicted_players = BTreeMap::<String, BTreeMap<String, u64>>::new();
    let mut unknown = 0u64;
    let mut contacts = 0u64;
    let mut unverified = 0u64;
    let mut generic = 0u64;
    let mut excluded_proposals = 0u64;
    let mut outside_proposals = 0u64;
    let mut initial_serve_candidates = 0u64;
    let mut unassigned_initial_serve_candidates = 0u64;
    let mut player_initial_serves = BTreeMap::<String, u64>::new();
    let mut automatic_identity_ids = BTreeSet::<String>::new();
    let mut automatically_linked_tracks = 0u64;
    let mut rallies = 0u64;
    let mut sources = Vec::new();
    let mut track_breakdown = Vec::new();
    for input in &request.sources {
        native_rally_evidence::verify_run_or_candidate(
            state,
            &request.project_id,
            &input.rally_run_id,
            None,
        )?;
        let run = native_rally_evidence::inspect_project(
            state,
            &request.project_id,
            &input.rally_run_id,
        )?;
        if !select_unique_video(&mut selected, &run.video) {
            return Err(storage::conflict(
                "Do not double-count multiple CPU revisions of one video",
            ));
        }
        let (owners, auto_ids) = native_player_tracks::statistics_owners(
            state,
            &request.project_id,
            &run.video,
            &run.replay_id,
        )?;
        automatically_linked_tracks += auto_ids.len() as u64;
        for track in &auto_ids {
            if let Some(owner) = owners.owners.get(track) {
                automatic_identity_ids.insert(owner.clone());
            }
        }
        if run.replay_sha != owners.source_sha {
            return Err(storage::conflict(
                "Stats Touch/Track identity source SHA mismatch",
            ));
        }
        for (id, name) in &owners.roster {
            if names
                .insert(id.clone(), name.clone())
                .is_some_and(|old| old != *name)
            {
                return Err(storage::conflict(
                    "Project athlete registry changed during processing",
                ));
            }
        }
        let root = projects::project_root(state)
            .join(&request.project_id)
            .join("native_cpu_replays")
            .join(&run.replay_id);
        let replay =
            safe_file(&root, "tracked_frames.jsonl", MAX_SOURCE).map_err(storage::bad_request)?;
        let (windows, serve_proposals) =
            native_rally_evidence::verified_track_windows_with_serves(&run)
                .map_err(storage::bad_request)?;
        let (credited_serves, unassigned_serves) = count_initial_serves(&serve_proposals, &owners);
        initial_serve_candidates += serve_proposals.len() as u64;
        unassigned_initial_serve_candidates += unassigned_serves;
        for (id, count) in credited_serves {
            *player_initial_serves.entry(id).or_default() += count;
        }
        rallies += run.expected_rallies;
        let subtotal = count_native_contacts(&run, &replay, &owners, &windows)
            .map_err(storage::bad_request)?;
        let report_candidates = subtotal.report_candidates();
        contacts += report_candidates;
        unknown += subtotal.unassigned;
        unverified += subtotal.unverified_proposals;
        generic += subtotal.generic_touches;
        excluded_proposals += subtotal.excluded_proposals;
        outside_proposals += subtotal.outside_rally_proposals;
        let mut run_unassigned = subtotal.unassigned;
        for (tid, count) in subtotal.per_track {
            let athlete = owners.owners.get(&tid);
            if let Some(player_id) = athlete {
                // Report credit requires a real project-roster player, never
                // a synthetic card derived from an unassigned CPU Track ID.
                *results.entry(player_id.clone()).or_default() += count;
                if let Some(actions) = subtotal.estimated_per_track.get(&tid) {
                    let totals = predicted_players.entry(player_id.clone()).or_default();
                    for (kind, count) in actions {
                        *totals.entry(kind.clone()).or_default() += count;
                        *predicted_counts.entry(kind.clone()).or_default() += count;
                    }
                }
            } else {
                unknown += count;
                run_unassigned += count;
            }
            track_breakdown.push(json!({
                "video":run.video,"source_replay_id":run.replay_id,
                "source_touch_id":run.touch_id,"track_id":tid,"touches":count,
                "confirmed_player_id":athlete,
                "manually_reviewed":athlete.is_some() && !auto_ids.contains(&tid),
                "automatic_photo_assignment":auto_ids.contains(&tid),
                "contact_landmark_required":true,
                "estimated_actions":subtotal.estimated_per_track.get(&tid).cloned().unwrap_or_default()
            }));
        }
        let identity_mapping_fingerprint = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&json!({
                    "owners":owners.owners,
                    "excluded":owners.excluded,
                    "automatic_track_ids":auto_ids,
                }))
                .map_err(|e| storage::internal(format!("Player identity fingerprint: {e}")))?
            )
        );
        sources.push(json!({
            "identity_mapping_fingerprint":identity_mapping_fingerprint,
            "video":run.video,"rally_run_id":run.run_id,
            "source_replay_id":run.replay_id,"source_replay_sha256":run.replay_sha,
            "source_touch_id":run.touch_id,"source_touch_sha256":run.touch_sha,
            "source_rally_sha256":run.rally_sha,
            "identity_revision":owners.revision,
            "rally_candidates":run.expected_rallies,
            "initial_serve_candidates":serve_proposals.len(),
            "unassigned_initial_serve_candidates":unassigned_serves,
            "total_touch_candidates":report_candidates,
            "unassigned_touch_candidates":run_unassigned,
            "unverified_touch_proposals":subtotal.unverified_proposals,
            "generic_touch_candidates":subtotal.generic_touches,
            "excluded_touch_proposals":subtotal.excluded_proposals,
            "outside_rally_touch_proposals":subtotal.outside_rally_proposals,
            "estimated_actions":subtotal.estimated_actions
        }));
    }
    sources.sort_by(|a, b| a["video"].as_str().cmp(&b["video"].as_str()));
    track_breakdown.sort_by(|a, b| {
        a["video"]
            .as_str()
            .cmp(&b["video"].as_str())
            .then_with(|| a["track_id"].as_u64().cmp(&b["track_id"].as_u64()))
    });
    let player_touches = names
        .keys()
        .map(|id| results.get(id).copied().unwrap_or(0))
        .collect::<Vec<_>>();
    let (mean_touches, activity) = volley_core::statistics::activity_against_mean(&player_touches);
    let players = names
        .iter()
        .zip(activity)
        .map(|((id, name), activity)| {
            let actions = predicted_players.get(id).cloned().unwrap_or_default();
            let touches = results.get(id).copied().unwrap_or(0);
            let unclassified = touches.saturating_sub(actions.values().sum::<u64>());
            json!({
                "player_id":id,"name":name,"touches":touches,"unclassified_touches":unclassified,
                "initial_serve_candidates":player_initial_serves.get(id).copied().unwrap_or(0),
                "activity":activity,
                "estimated_actions":actions,
                "identity_source":if automatic_identity_ids.contains(id) {
                    "photo_auto_or_reviewed_native_track_ids"
                } else {
                    "manually_reviewed_native_track_ids"
                }
            })
        })
        .collect::<Vec<_>>();
    let source_checksum = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&sources)
                .map_err(|e| storage::internal(format!("Encode source fingerprint: {e}")))?
        )
    );
    let sum: u64 = results.values().sum();
    if sum + unknown != contacts
        || players.len() != names.len()
        || player_initial_serves.values().sum::<u64>() + unassigned_initial_serve_candidates
            != initial_serve_candidates
    {
        return Err(storage::internal(
            "Native stats conservation accounting failed",
        ));
    }
    let has_predictions = !predicted_counts.is_empty();
    Ok(json!({
        "schema":"volley-native-player-statistics/1",
        "algorithm":REPORT_ALGORITHM,
        "project_id":request.project_id,
        "source_fingerprint":source_checksum,
        "sources":sources,"video_count":selected.len(),
        "roster_player_ids":names.keys().collect::<Vec<_>>(),
        "rally_candidates":rallies,
        "initial_serve_candidates":initial_serve_candidates,
        "unassigned_initial_serve_candidates":unassigned_initial_serve_candidates,
        "mean_player_touches":mean_touches,
        "estimated_hits":predicted_counts.get("attack").copied().unwrap_or(0) + predicted_counts.get("serve").copied().unwrap_or(0),
        "total_touch_candidates":contacts,
        "assigned_touch_candidates":sum,
        "unassigned_touch_candidates":unknown,
        "unverified_touch_proposals":unverified,
        "excluded_touch_proposals":excluded_proposals,
        "outside_rally_touch_proposals":outside_proposals,
        "players":players,"raw_tracks":track_breakdown,
        "estimated_actions":predicted_counts,
        "auto_player_identity_enabled":automatically_linked_tracks > 0,
        "automatic_action_classification":has_predictions,
        "automatic_action_classification_verified":false,
        "all_contacts_physically_verified":true,
        "all_contacts_pose_verified":generic==0,
        "fallback_touch_candidates":generic
    }))
}
/// One source of truth for report-montage and Report-tab player totals:
/// only pose-confirmed, source-bound Touch events in verified rally windows.
pub(crate) fn rally_report_counts(
    state: &AppState,
    run: &ValidatedRun,
    owners: &NativeReviewedOwners,
) -> Result<BTreeMap<String, (u64, u64, u64)>, ApiError> {
    let root = projects::project_root(state)
        .join(&run.project_id)
        .join("native_cpu_replays")
        .join(&run.replay_id);
    let replay =
        safe_file(&root, "tracked_frames.jsonl", MAX_SOURCE).map_err(storage::bad_request)?;
    let windows =
        native_rally_evidence::verified_track_windows(run).map_err(storage::bad_request)?;
    let counts =
        count_native_contacts(run, &replay, owners, &windows).map_err(storage::bad_request)?;
    let mut result = BTreeMap::<String, (u64, u64, u64)>::new();
    for (track, touches) in counts.per_track {
        let Some(name) = owners
            .owners
            .get(&track)
            .and_then(|owner| owners.roster.get(owner))
            .cloned()
        else {
            continue;
        };
        let actions = counts.estimated_per_track.get(&track);
        let attacks = actions.and_then(|a| a.get("attack")).copied().unwrap_or(0);
        let blocks = actions.and_then(|a| a.get("block")).copied().unwrap_or(0);
        let entry = result.entry(name).or_default();
        entry.0 += touches;
        entry.1 += attacks;
        entry.2 += blocks;
    }
    Ok(result)
}

/// Exact selected CPU-rally versions and project roster define the report
/// snapshot. Manual changes to Track-ID ownership intentionally do not
/// invalidate it: the explicit "Recalculate statistics" action does that.
fn matches_saved_report(snapshot: &Value, req: &Request, project: &Value) -> bool {
    if snapshot["schema"] != "volley-native-player-statistics/1"
        || snapshot["algorithm"] != REPORT_ALGORITHM
        || snapshot["project_id"].as_str() != Some(req.project_id.as_str())
    {
        return false;
    }
    let selected = req
        .sources
        .iter()
        .map(|source| source.rally_run_id.as_str())
        .collect::<BTreeSet<_>>();
    let Some(sources) = snapshot["sources"].as_array() else {
        return false;
    };
    let cached = sources
        .iter()
        .filter_map(|source| source["rally_run_id"].as_str())
        .collect::<BTreeSet<_>>();
    if selected.len() != req.sources.len() || selected != cached || sources.len() != selected.len()
    {
        return false;
    }
    let roster = project["player_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>();
    let saved_roster = snapshot["roster_player_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>();
    let players = snapshot["players"].as_array();
    roster == saved_roster
        && players.is_some_and(|items| {
            items.len() == roster.len()
                && items.iter().all(|player| {
                    player["player_id"]
                        .as_str()
                        .is_some_and(|id| roster.contains(id))
                })
        })
}

fn run(state: &AppState, req: &Request) -> Result<Value, ApiError> {
    let _guard = project_guard::write_guard(state, &req.project_id)?;
    if !matches!(req.mode.as_str(), "preview" | "open" | "recalculate") {
        return Err(storage::bad_request(
            "Native stats mode must be preview, open or recalculate",
        ));
    }
    if req.mode != "recalculate"
        && (req.expected_source_fingerprint.is_some() || req.expected_revision.is_some())
    {
        return Err(storage::bad_request(
            "Read/open does not accept a write revision",
        ));
    }
    let project_data = projects::existing_project(state, &req.project_id)?;
    let project = projects::project_root(state).join(&req.project_id);
    let file = project.join("native_player_statistics.json");
    let raw =
        versioned_store::read_optional_bounded(&file, MAX_STATS).map_err(storage::bad_request)?;
    let previous_revision = revision(raw.as_deref());
    if req.mode == "open" {
        if let Some(saved) = raw.as_deref() {
            let mut snapshot: Value = serde_json::from_slice(saved).map_err(|e| {
                storage::bad_request(format!("Invalid native statistics snapshot: {e}"))
            })?;
            if matches_saved_report(&snapshot, req, &project_data) {
                snapshot["saved_revision"] = json!(previous_revision);
                snapshot["snapshot_persisted"] = json!(true);
                snapshot["reused_saved_snapshot"] = json!(true);
                return Ok(snapshot);
            }
        }
    }
    if req.mode == "recalculate" {
        let expected = req
            .expected_revision
            .as_deref()
            .ok_or_else(|| storage::bad_request("Expected statistics revision missing"))?;
        versioned_store::expect_revision(raw.as_deref(), expected).map_err(storage::conflict)?;
    }
    if req.mode != "preview" && !state.writes_enabled {
        return Err(storage::conflict(
            "Saving statistics requires isolated project storage",
        ));
    }
    let mut output = calculate(state, req)?;
    if req.mode == "recalculate" {
        if let Some(expected) = req.expected_source_fingerprint.as_deref() {
            if !expected_source_matches(&output, expected) {
                return Err(storage::conflict(
                    "CPU or athlete sources changed since statistics preview",
                ));
            }
        }
    }
    if req.mode != "preview" {
        versioned_store::write_json_checked(&project, &file, &output, MAX_STATS)
            .map_err(storage::bad_request)?;
    }
    let current_revision = if req.mode == "preview" {
        previous_revision
    } else {
        let updated = versioned_store::read_optional_bounded(&file, MAX_STATS)
            .map_err(storage::bad_request)?;
        revision(updated.as_deref())
    };
    output["saved_revision"] = json!(current_revision);
    output["snapshot_persisted"] = json!(req.mode != "preview");
    output["reused_saved_snapshot"] = json!(false);
    Ok(output)
}
pub async fn statistics(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<Request>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let _permit = state
        .statistics_limit
        .acquire()
        .await
        .map_err(|e| storage::internal(format!("Native stats CPU gate: {e}")))?;
    let worker = state.clone();
    let output = tokio::task::spawn_blocking(move || run(&worker, &req))
        .await
        .map_err(|e| storage::internal(format!("Native stats worker: {e}")))??;
    Ok(Json(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use uuid::Uuid;
    fn fixture() -> (
        std::path::PathBuf,
        ValidatedRun,
        std::path::PathBuf,
        NativeReviewedOwners,
    ) {
        let root = std::env::temp_dir().join(format!("native-stats-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let replay = root.join("tracked_frames.jsonl");
        let touches = root.join("touch_candidates.jsonl");
        let mut replay_bytes = Vec::new();
        let mut touch_bytes = Vec::new();
        for frame in 0..80u64 {
            let track = if frame < 40 { 7 } else { 8 };
            let row = json!({
                "frame_index":frame,"track_ids_provisional":true,
                "confirmed_athlete_identities":false,"touches_confirmed":false,
                "selected_ball":{"source_model":"vballnet"},
                "tracked_players":[{
                    "track_id":track,"is_verified_athlete":false,
                    "detection":{"frame_index":frame,"class_name":"player"}
                }]
            });
            replay_bytes.extend(serde_json::to_vec(&row).unwrap());
            replay_bytes.push(b'\n');
            if [10, 20, 42, 50, 60].contains(&frame) {
                let contact = json!({
                    "frame_index":frame,"provisional_track_id":track,
                    "probability":0.96,"ball_x_px":200.0,
                    "ball_y_px":100.0,"motion_frame_index":frame,
                    "direction_change_deg":90.0,"speed_before_px_frame":4.0,
                    "speed_after_px_frame":8.0,"velocity_residual_px_frame":10.0,
                    "technique":"unknown_touch","physical_motion_supported":true,
                    "action_prediction":if frame==20 {
                        json!({
                            "kind":"attack","evidence_score":0.84,
                            "method":"pose_keypoints_motion_v1","depth_ratio":-0.2,
                            "posture_ratio":0.25
                        })
                    } else {Value::Null},
                    "body_contact_supported":true,
                    "contact_landmark":{"joint_indices":[9,9],"point_px":[200.0,100.0],
                      "distance_px":0.0,"normalized_distance":0.0},
                    "athlete_identity_verified":false
                });
                touch_bytes.extend(serde_json::to_vec(&contact).unwrap());
                touch_bytes.push(b'\n');
            }
        }
        fs::write(&replay, &replay_bytes).unwrap();
        fs::write(&touches, &touch_bytes).unwrap();
        let sha = format!("{:x}", Sha256::digest(&replay_bytes));
        let run = ValidatedRun {
            project_id: "abcdef012345".into(),
            video: "match.mp4".into(),
            run_id: "1".repeat(32),
            project_source: root.join("video.mp4"),
            rally_file: root.join("rallies.jsonl"),
            touch_file: touches,
            replay_id: "2".repeat(32),
            replay_sha: sha.clone(),
            touch_id: "3".repeat(32),
            rally_sha: "4".repeat(64),
            touch_sha: format!("{:x}", Sha256::digest(&touch_bytes)),
            frames: 80,
            source_duration_sec: 80.0 / 30.0,
            analysis_range: None,
            width: 640,
            height: 360,
            context: 45,
            expected_rallies: 1,
            expected_touches: 5,
            min_ball_score: 0.35,
        };
        let owner = NativeReviewedOwners {
            source_sha: sha,
            revision: "a".repeat(64),
            owners: BTreeMap::from([(7, "player1".into()), (8, "player2".into())]),
            excluded: BTreeSet::new(),
            roster: BTreeMap::from([
                ("player1".into(), "Alice".into()),
                ("player2".into(), "Bob".into()),
            ]),
        };
        (root, run, replay, owner)
    }
    #[test]
    fn initial_serves_credit_only_linked_roster_players_without_inventing_touches() {
        let (root, _, _, mut owners) = fixture();
        let proposals = vec![
            json!({"provisional_track_id":7}),
            json!({"provisional_track_id":8}),
            json!({"provisional_track_id":null}),
            json!({"provisional_track_id":99}),
        ];
        owners.excluded.insert(8);
        let (players, unknown) = count_initial_serves(&proposals, &owners);
        assert_eq!(players, BTreeMap::from([("player1".to_owned(), 1)]));
        assert_eq!(unknown, 3);
        assert_eq!(
            players.values().sum::<u64>() + unknown,
            proposals.len() as u64
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unassigned_physical_touch_preserves_report_conservation() {
        let (root, mut run, replay, owners) = fixture();
        let mut modified = Vec::new();
        for line in fs::read_to_string(&run.touch_file).unwrap().lines() {
            let mut event: Value = serde_json::from_str(line).unwrap();
            if event["frame_index"] == 20 {
                event["provisional_track_id"] = Value::Null;
                event["body_contact_supported"] = json!(false);
                event["contact_landmark"] = Value::Null;
                event["action_prediction"] = Value::Null;
            }
            modified.extend(serde_json::to_vec(&event).unwrap());
            modified.push(b'\n');
        }
        fs::write(&run.touch_file, &modified).unwrap();
        run.touch_sha = format!("{:x}", Sha256::digest(&modified));
        let counted = count_native_contacts(
            &run,
            &replay,
            &owners,
            &native_rally_evidence::RallyWindows {
                intervals: vec![(0, 79)],
                contexts: BTreeMap::new(),
            },
        )
        .unwrap();
        assert_eq!(counted.touches, 4);
        assert_eq!(counted.unassigned, 1);
        assert_eq!(counted.report_candidates(), 5);
        assert_eq!(
            counted.per_track.values().sum::<u64>() + counted.unassigned,
            counted.report_candidates()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn conflicting_same_frame_does_not_credit_touch_but_later_frame_does() {
        let (root, mut run, replay, mut owners) = fixture();
        let input = fs::read_to_string(&replay).unwrap();
        let mut bytes = Vec::new();
        for line in input.lines() {
            let mut row: Value = serde_json::from_str(line).unwrap();
            if row["frame_index"] == 10 {
                row["tracked_players"].as_array_mut().unwrap().push(json!({
                    "track_id":8,"is_verified_athlete":false,
                    "detection":{"frame_index":10,"class_name":"player"}
                }));
            }
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        fs::write(&replay, &bytes).unwrap();
        run.replay_sha = format!("{:x}", Sha256::digest(&bytes));
        owners.source_sha = run.replay_sha.clone();
        owners.owners.insert(8, "player1".into());
        let counted = count_native_contacts(
            &run,
            &replay,
            &owners,
            &native_rally_evidence::RallyWindows {
                intervals: vec![(0, 79)],
                contexts: BTreeMap::new(),
            },
        )
        .unwrap();
        assert_eq!(counted.unassigned, 1);
        assert!(!counted
            .montage_contacts
            .iter()
            .any(|event| event.frame == 10));
        assert!(counted
            .montage_contacts
            .iter()
            .any(|event| event.frame == 20));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn native_contact_count_credits_only_same_frame_reviewed_owners() {
        let (root, run, replay, mut owners) = fixture();
        let counted = count_native_contacts(
            &run,
            &replay,
            &owners,
            &native_rally_evidence::RallyWindows {
                intervals: vec![(0, 79)],
                contexts: BTreeMap::new(),
            },
        )
        .unwrap();
        assert_eq!(counted.touches, 5);
        assert_eq!(counted.unassigned, 0);
        assert_eq!(counted.per_track.get(&7), Some(&2));
        assert_eq!(counted.per_track.get(&8), Some(&3));
        assert_eq!(
            counted
                .estimated_per_track
                .get(&7)
                .and_then(|actions| actions.get("attack")),
            Some(&1)
        );
        let player_attack_frames = counted
            .montage_contacts
            .iter()
            .filter(|contact| {
                contact.track_id == 7 && contact.estimated_kind.as_deref() == Some("attack")
            })
            .map(|contact| contact.frame)
            .collect::<Vec<_>>();
        assert_eq!(
            player_attack_frames,
            vec![20],
            "Report's attack must provide exactly the montage frame"
        );
        owners.owners.remove(&8);
        let counted = count_native_contacts(
            &run,
            &replay,
            &owners,
            &native_rally_evidence::RallyWindows {
                intervals: vec![(0, 79)],
                contexts: BTreeMap::new(),
            },
        )
        .unwrap();
        assert_eq!(counted.per_track.get(&8), Some(&3)); // observed raw only
        assert!(!owners.owners.contains_key(&8)); // never auto-attaches
        owners.excluded.insert(8);
        let counted = count_native_contacts(
            &run,
            &replay,
            &owners,
            &native_rally_evidence::RallyWindows {
                intervals: vec![(0, 79)],
                contexts: BTreeMap::new(),
            },
        )
        .unwrap();
        assert_eq!(counted.unassigned, 0);
        assert_eq!(counted.excluded_proposals, 3);
        assert_eq!(counted.touches, 2);
        assert_eq!(counted.per_track.len(), 1);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn final_rally_context_is_shared_by_statistics_and_montage() {
        use volley_core::action_prediction::ActionKind;
        let (root, run, replay, owners) = fixture();
        let scope = native_rally_evidence::RallyWindows {
            intervals: vec![(0, 79)],
            contexts: BTreeMap::from([
                (10, (Some(7), true, None)),
                (20, (Some(7), false, Some(ActionKind::Serve))),
            ]),
        };
        let counts = count_native_contacts(&run, &replay, &owners, &scope).unwrap();
        assert_eq!(counts.touches, 4);
        assert_eq!(counts.estimated_actions.get("serve"), Some(&1));
        assert!(counts
            .montage_contacts
            .iter()
            .any(|event| event.frame == 20 && event.estimated_kind.as_deref() == Some("serve")));
        assert!(!counts
            .montage_contacts
            .iter()
            .any(|event| event.frame == 10));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stats_exclude_touches_between_rallies_without_skipping_source_validation() {
        let (root, run, replay, owners) = fixture();
        let scope = native_rally_evidence::RallyWindows {
            intervals: vec![(8, 22), (48, 53)],
            contexts: BTreeMap::new(),
        };
        let counts = count_native_contacts(&run, &replay, &owners, &scope).unwrap();
        // Source contains five physically valid Touch events; only three
        // are inside detected-play windows: 10, 20 and 50.
        assert_eq!(counts.touches, 3);
        assert_eq!(counts.per_track.get(&7), Some(&2));
        assert_eq!(counts.per_track.get(&8), Some(&1));
        assert_eq!(counts.unassigned, 0);
        assert_eq!(counts.per_track.values().sum::<u64>(), 3);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn nearest_track_touch_is_counted_without_unjustified_action_kind() {
        let (root, mut run, replay, owners) = fixture();
        let mut modified = Vec::new();
        for line in fs::read_to_string(&run.touch_file).unwrap().lines() {
            let mut row: Value = serde_json::from_str(line).unwrap();
            if row["frame_index"] == 20 {
                row["body_contact_supported"] = json!(false);
                row.as_object_mut().unwrap().remove("contact_landmark");
            }
            if row["frame_index"] == 50 {
                row["provisional_track_id"] = Value::Null;
                row["body_contact_supported"] = json!(false);
                row.as_object_mut().unwrap().remove("contact_landmark");
            }
            modified.extend(serde_json::to_vec(&row).unwrap());
            modified.push(b'\n');
        }
        fs::write(&run.touch_file, &modified).unwrap();
        run.touch_sha = format!("{:x}", Sha256::digest(&modified));
        let counts = count_native_contacts(
            &run,
            &replay,
            &owners,
            &native_rally_evidence::RallyWindows {
                intervals: vec![(0, 79)],
                contexts: BTreeMap::new(),
            },
        )
        .unwrap();
        assert_eq!(counts.unverified_proposals, 1);
        assert_eq!(counts.touches, 4);
        assert_eq!(counts.per_track.get(&7), Some(&2));
        assert_eq!(counts.per_track.get(&8), Some(&2));
        assert_eq!(counts.unassigned, 1);
        assert!(!counts.estimated_actions.contains_key("attack"));
        assert_eq!(
            counts
                .montage_contacts
                .iter()
                .find(|c| c.frame == 20)
                .unwrap()
                .estimated_kind,
            None
        );
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn native_stats_reject_missing_track_in_touch_frame_even_with_new_sha() {
        let (root, mut run, replay, mut owners) = fixture();
        let raw = fs::read(&replay).unwrap();
        let mut modified = Vec::new();
        for line in raw.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
            let mut row: Value = serde_json::from_slice(line).unwrap();
            if row["frame_index"] == 42 {
                row["tracked_players"][0]["track_id"] = json!(9);
            }
            modified.extend(serde_json::to_vec(&row).unwrap());
            modified.push(b'\n');
        }
        fs::write(&replay, &modified).unwrap();
        run.replay_sha = format!("{:x}", Sha256::digest(&modified));
        owners.source_sha = run.replay_sha.clone();
        let failure = count_native_contacts(
            &run,
            &replay,
            &owners,
            &native_rally_evidence::RallyWindows {
                intervals: vec![(0, 79)],
                contexts: BTreeMap::new(),
            },
        )
        .unwrap_err();
        assert!(
            failure.contains("absent"),
            "Incorrect source gate: {failure}"
        );
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn stats_recalculate_checks_expected_source_fingerprint() {
        assert!(expected_source_matches(
            &json!({"source_fingerprint":"f".repeat(64)}),
            &"f".repeat(64),
        ));
        assert!(!expected_source_matches(
            &json!({"source_fingerprint":"e".repeat(64)}),
            &"f".repeat(64),
        ));
        assert!(!expected_source_matches(
            &json!({"source_fingerprint":"f".repeat(64)}),
            "invalid",
        ));
        let request: Request = serde_json::from_value(json!({
            "project_id":"abcdef012345",
            "sources":[{"rally_run_id":"1".repeat(32)}],
            "mode":"recalculate",
            "expected_revision":"absent",
            "expected_source_fingerprint":"f".repeat(64)
        }))
        .unwrap();
        assert_eq!(
            request.expected_source_fingerprint.as_deref(),
            Some("f".repeat(64).as_str())
        );
    }
    #[test]
    fn report_snapshot_reuse_depends_on_roster_and_cpu_version_not_manual_identity() {
        let project = json!({"player_ids":["player1","player2"]});
        let source = "a".repeat(32);
        let request: Request = serde_json::from_value(json!({
            "project_id":"abcdef012345",
            "sources":[{"rally_run_id":source}],
            "mode":"open"
        }))
        .unwrap();
        let snapshot = json!({
            "schema":"volley-native-player-statistics/1",
            "algorithm":REPORT_ALGORITHM,
            "project_id":"abcdef012345",
            "sources":[{"rally_run_id":source, "identity_revision":"old"}],
            "roster_player_ids":["player1","player2"],
            "players":[{"player_id":"player1"},{"player_id":"player2"}]
        });
        assert!(matches_saved_report(&snapshot, &request, &project));
        let mut reassigned = snapshot.clone();
        reassigned["sources"][0]["identity_revision"] = json!("new");
        assert!(matches_saved_report(&reassigned, &request, &project));
        let new_source: Request = serde_json::from_value(json!({
            "project_id":"abcdef012345",
            "sources":[{"rally_run_id":"b".repeat(32)}],"mode":"open"
        }))
        .unwrap();
        assert!(!matches_saved_report(&snapshot, &new_source, &project));
        assert!(!matches_saved_report(
            &snapshot,
            &request,
            &json!({"player_ids":["player1","player3"]})
        ));
        reassigned["players"][0]["player_id"] = json!("track:movie:999");
        assert!(!matches_saved_report(&reassigned, &request, &project));
    }

    #[test]
    fn report_only_includes_roster_athlete_credits() {
        let (root, run, replay, mut owners) = fixture();
        owners.owners.remove(&8);
        let counts = count_native_contacts(
            &run,
            &replay,
            &owners,
            &native_rally_evidence::RallyWindows {
                intervals: vec![(8, 22), (48, 53)],
                contexts: BTreeMap::new(),
            },
        )
        .unwrap();
        let credited = counts
            .per_track
            .iter()
            .filter(|(track, _)| owners.owners.contains_key(track))
            .map(|(_, count)| count)
            .sum::<u64>();
        let unknown = counts
            .per_track
            .iter()
            .filter(|(track, _)| !owners.owners.contains_key(track))
            .map(|(_, count)| count)
            .sum::<u64>();
        assert_eq!((credited, unknown), (2, 1));
        assert_eq!(credited + unknown, counts.touches);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn native_stats_reject_double_counting_same_video_cpu_revisions() {
        let mut seen = BTreeSet::new();
        assert!(select_unique_video(&mut seen, "match.mp4"));
        assert!(!select_unique_video(&mut seen, "match.mp4"));
        assert!(select_unique_video(&mut seen, "second.mp4"));
        assert_eq!(seen.len(), 2);
    }
}
