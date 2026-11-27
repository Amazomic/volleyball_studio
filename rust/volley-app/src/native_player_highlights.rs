//! Player reels from the exact SHA-verified contacts credited by Report.
//! Action prediction is an estimate; a saved human action review may override
//! its label. No unlabeled tracks or contacts outside rallies produce clips.
use crate::{
    montage, native_action_review, native_player_statistics, native_player_tracks,
    native_rally_evidence::{self, ValidatedRun},
    project_guard, projects,
    storage::{self, ApiError, ApiResult},
    AppState,
};
use axum::{extract::State, http::HeaderMap, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::Arc,
};
use volley_media::montage::{SelectedClip, VideoLayout};

const MAX_RUNS: usize = 40;
const MAX_CLIPS: usize = 80;
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    project_id: String,
    player_id: String,
    #[serde(default = "default_action_kind")]
    action_kind: String,
    #[serde(default = "default_clip_context")]
    clip_context_sec: f64,
    #[serde(default)]
    target_duration_sec: Option<f64>,
    rally_run_ids: Vec<String>,
    #[serde(default)]
    preferred_rally_run_id: Option<String>,
    #[serde(default = "preview")]
    operation: String,
    #[serde(default = "auto")]
    selection_mode: String,
    #[serde(default)]
    manual_rally_keys: Vec<String>,
    #[serde(default = "default_max")]
    max_clips: usize,
    #[serde(default)]
    expected_source_fingerprint: Option<String>,
    #[serde(default)]
    encoder: Option<String>,
    #[serde(default)]
    layout: VideoLayout,
    #[serde(default = "default_zoom")]
    camera_zoom: f64,
    #[serde(default)]
    audio_file: Option<String>,
    #[serde(default = "default_volume")]
    audio_volume: f64,
    #[serde(default)]
    include_recognition_data: bool,
}
fn default_zoom() -> f64 {
    1.25
}
fn default_action_kind() -> String {
    "attack".into()
}
fn default_clip_context() -> f64 {
    1.0
}
fn preview() -> String {
    "preview".into()
}
fn auto() -> String {
    "auto".into()
}
fn default_volume() -> f64 {
    40.0
}
fn default_max() -> usize {
    12
}

#[derive(Debug, Clone)]
struct Candidate {
    key: String,
    candidate_id: String,
    rally_run_id: String,
    video: String,
    start: u64,
    end: u64,
    verified_attacks: u64,
    initial_serves: u64,
    human_reviewed: u64,
    frames: Vec<u64>,
}
impl Candidate {
    fn view(&self) -> Value {
        json!({
            "key":self.key,"candidate_id":self.candidate_id,
            "rally_run_id":self.rally_run_id,"video":self.video,
            "detected_start_frame":self.start,"detected_end_frame":self.end,
            "verified_player_actions":self.verified_attacks,
            "initial_serve_candidates":self.initial_serves,
            "matched_player_events":self.frames.len(),
            "human_reviewed_actions":self.human_reviewed,
            "action_label_source":if self.initial_serves > 0 {
                "report_trajectory_serve_candidate_or_pose_supported_contact"
            } else {
                "pose_supported_prediction_or_explicit_review"
            },
            "athlete_link_source":"reviewed_or_auto_linked_roster_player"
        })
    }
}
fn select_candidates(
    candidates: &[Candidate],
    mode: &str,
    manual: &[String],
    max: usize,
) -> Result<Vec<Candidate>, String> {
    if max == 0 || max > MAX_CLIPS || !matches!(mode, "auto" | "manual") {
        return Err("Choose auto or manual highlights, 1..80 clips".into());
    }
    if mode == "auto" {
        if !manual.is_empty() {
            return Err("Automatic highlights cannot accept manual IDs".into());
        }
        return Ok(candidates.iter().take(max).cloned().collect());
    }
    if manual.is_empty()
        || manual.len() > max
        || manual.iter().collect::<HashSet<_>>().len() != manual.len()
    {
        return Err("Manual highlights require 1..max distinct source-scoped rally keys".into());
    }
    let by_key = candidates
        .iter()
        .map(|c| (c.key.as_str(), c))
        .collect::<HashMap<_, _>>();
    manual
        .iter()
        .map(|key| {
            by_key
                .get(key.as_str())
                .map(|c| (*c).clone())
                .ok_or_else(|| format!("Unverified or stale athlete attack rally: {key}"))
        })
        .collect()
}
/// A manual label changes the predicted type only for the exact, previously
/// validated contact frame and track. Other contacts retain Report predictions.
fn select_player_actions(
    verified: &[native_player_statistics::MontageContact],
    owners: &native_player_tracks::NativeReviewedOwners,
    player_id: &str,
    kind: &str,
    reviewed: &BTreeMap<(u64, u64), String>,
) -> Vec<(u64, bool)> {
    verified
        .iter()
        .filter_map(|event| {
            if owners.excluded.contains(&event.track_id)
                || owners.owners.get(&event.track_id).map(String::as_str) != Some(player_id)
            {
                return None;
            }
            let manual = reviewed.get(&(event.frame, event.track_id));
            let actual_kind = manual
                .map(String::as_str)
                .or(event.estimated_kind.as_deref())
                .unwrap_or("unknown_touch");
            (kind == "all" || kind == actual_kind).then_some((event.frame, manual.is_some()))
        })
        .collect()
}
fn reviewed_action_labels(response: &Value) -> Result<BTreeMap<(u64, u64), String>, String> {
    let actions = response["reviewed_actions"]
        .as_array()
        .ok_or("Missing reviewed action entries")?;
    let mut labels = BTreeMap::new();
    for action in actions {
        if action["action_verified"] != true
            || action["motion_supported"] != true
            || action["action_source"] != "explicit_human_review"
            || action["athlete_identity_verified"] != false
        {
            return Err("Untrusted manual action review".into());
        }
        let (Some(frame), Some(track), Some(kind)) = (
            action["frame_index"].as_u64(),
            action["provisional_track_id"].as_u64(),
            action["action_kind"].as_str(),
        ) else {
            continue;
        };
        if !matches!(
            kind,
            "attack" | "block" | "serve" | "dig" | "overhead_pass" | "low_touch"
        ) {
            return Err("Unsupported manual action label".into());
        }
        if labels.insert((frame, track), kind.to_string()).is_some() {
            return Err("Duplicate manual action label".into());
        }
    }
    Ok(labels)
}
fn select_initial_serve_frames(
    proposals: &[Value],
    owners: &native_player_tracks::NativeReviewedOwners,
    player_id: &str,
) -> Vec<u64> {
    proposals
        .iter()
        .filter(|proposal| {
            native_player_statistics::initial_serve_owner(proposal, owners)
                .is_some_and(|owner| owner == player_id)
        })
        .filter_map(|proposal| {
            proposal["trajectory_start_frame"]
                .as_u64()
                .or_else(|| proposal["rally_start_frame"].as_u64())
        })
        .collect()
}

fn rally_for_frame(intervals: &[(String, u64, u64)], frame: u64) -> Option<usize> {
    let index = intervals.partition_point(|(_, start, _)| *start <= frame);
    index.checked_sub(1).filter(|i| frame <= intervals[*i].2)
}

fn append_initial_serves(
    intervals: &[(String, u64, u64)],
    frames: &[u64],
    per_rally: &mut [Vec<u64>],
    per_rally_serves: &mut [u64],
) -> Result<(), String> {
    for &frame in frames {
        let index = rally_for_frame(intervals, frame)
            .ok_or("Report initial serve falls outside montage rallies")?;
        if !per_rally[index].contains(&frame) {
            per_rally[index].push(frame);
            per_rally_serves[index] += 1;
        }
    }
    Ok(())
}

struct Plan {
    response: Value,
    clips: Vec<SelectedClip>,
    camera_tracks: Option<Vec<Vec<volley_media::montage::CameraPoint>>>,
}
fn plan(state: &AppState, request: &Request) -> Result<Plan, ApiError> {
    storage::project_id(&request.project_id)?;
    if request.player_id.is_empty()
        || request.player_id.len() > 128
        || request.rally_run_ids.is_empty()
        || request.rally_run_ids.len() > MAX_RUNS
        || request.rally_run_ids.iter().collect::<HashSet<_>>().len() != request.rally_run_ids.len()
        || request.max_clips == 0
        || request.max_clips > MAX_CLIPS
        || !matches!(request.selection_mode.as_str(), "auto" | "manual")
        || !matches!(
            request.action_kind.as_str(),
            "all"
                | "attack"
                | "block"
                | "serve"
                | "dig"
                | "overhead_pass"
                | "low_touch"
                | "unknown_touch"
        )
        || !request.clip_context_sec.is_finite()
        || !(0.2..=3.0).contains(&request.clip_context_sec)
        || request
            .target_duration_sec
            .is_some_and(|x| !x.is_finite() || !(1.0..=600.0).contains(&x))
        || !request.camera_zoom.is_finite()
        || !(1.0..=2.5).contains(&request.camera_zoom)
    {
        return Err(storage::bad_request(
            "Invalid player, native CPU runs or highlight settings",
        ));
    }
    if !request.audio_volume.is_finite() || !(0.0..=100.0).contains(&request.audio_volume) {
        return Err(storage::bad_request("Soundtrack volume must be 0..100"));
    }
    if !matches!(request.operation.as_str(), "preview" | "start") {
        return Err(storage::bad_request("Supported operations: preview/start"));
    }
    let expected = request.expected_source_fingerprint.as_deref();
    if (request.operation == "start"
        && !expected.is_some_and(|v| {
            v.len() == 64
                && v.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        }))
        || (request.operation == "preview" && expected.is_some())
    {
        return Err(storage::bad_request(
            "Start requires a valid preview SHA fingerprint",
        ));
    }
    let project = projects::existing_project(state, &request.project_id)?;
    if request.rally_run_ids.len() > project["videos"].as_array().map_or(0, Vec::len) {
        return Err(storage::bad_request(
            "Multiple CPU versions of the same video cannot be combined",
        ));
    }
    let mut videos = BTreeSet::new();
    let mut runs = BTreeMap::<String, ValidatedRun>::new();
    let mut sources = Vec::new();
    let mut candidates = Vec::<Candidate>::new();
    let mut athlete_name = None::<String>;
    // Sorted source IDs make fingerprint deterministic regardless of UI order.
    let mut source_ids = request.rally_run_ids.clone();
    source_ids.sort();
    for run_id in &source_ids {
        let run = native_rally_evidence::inspect_project(state, &request.project_id, run_id)?;
        if !videos.insert(run.video.clone()) {
            return Err(storage::conflict(
                "More than one CPU version selected for the same video",
            ));
        }
        // The legacy and native run adapters MUST NOT be mixed by ID guessing.
        let (owners, automatic_track_ids) = native_player_tracks::statistics_owners(
            state,
            &request.project_id,
            &run.video,
            &run.replay_id,
        )?;
        if owners.source_sha != run.replay_sha {
            return Err(storage::conflict(
                "Athlete assignments belong to another replay SHA",
            ));
        }
        let name = owners
            .roster
            .get(&request.player_id)
            .ok_or_else(|| storage::bad_request("Player is not in selected project roster"))?;
        if athlete_name.as_ref().is_some_and(|old| old != name) {
            return Err(storage::conflict(
                "Athlete catalog changed between selected runs",
            ));
        }
        athlete_name = Some(name.clone());
        // Exactly the Report-tab selector: same-frame Track ID, ball,
        // pose contact landmark, SHA checks, and a verified rally interval.
        let report_contacts =
            native_player_statistics::verified_contacts_for_montage(state, &run, &owners)?;
        let reviewed = native_action_review::verified_actions(state, &run)?;
        if reviewed["schema"] != "volley-native-action-review/1"
            || reviewed["source_replay_id"] != run.replay_id
            || reviewed["source_touch_id"] != run.touch_id
            || reviewed["source_touch_sha256"] != run.touch_sha
            || reviewed["source_rally_sha256"] != run.rally_sha
            || reviewed["automatic_game_action_classification"] != false
        {
            return Err(storage::conflict(
                "Action labels have a mismatched source/version",
            ));
        }
        let (windows, serve_proposals) =
            native_rally_evidence::verified_track_windows_with_serves(&run)
                .map_err(storage::bad_request)?;
        let intervals =
            native_rally_evidence::verified_rally_intervals(&run).map_err(storage::bad_request)?;
        if windows.intervals
            != intervals
                .iter()
                .map(|(_, start, end)| (*start, *end))
                .collect::<Vec<_>>()
        {
            return Err(storage::conflict(
                "Report and montage rally intervals differ",
            ));
        }
        let manual_labels = reviewed_action_labels(&reviewed).map_err(storage::conflict)?;
        let mut per_rally = vec![Vec::<u64>::new(); intervals.len()];
        let mut per_rally_manual = vec![0_u64; intervals.len()];
        let mut per_rally_serves = vec![0_u64; intervals.len()];
        for (frame, manually_reviewed) in select_player_actions(
            &report_contacts,
            &owners,
            &request.player_id,
            &request.action_kind,
            &manual_labels,
        ) {
            if let Some(index) = rally_for_frame(&intervals, frame) {
                per_rally[index].push(frame);
                per_rally_manual[index] += u64::from(manually_reviewed);
            }
        }
        let physical_counts = per_rally
            .iter()
            .map(|frames| frames.len() as u64)
            .collect::<Vec<_>>();
        if matches!(request.action_kind.as_str(), "serve" | "all") {
            let serve_frames =
                select_initial_serve_frames(&serve_proposals, &owners, &request.player_id);
            append_initial_serves(
                &intervals,
                &serve_frames,
                &mut per_rally,
                &mut per_rally_serves,
            )
            .map_err(storage::conflict)?;
        }
        for (i, (id, start, end)) in intervals.into_iter().enumerate() {
            per_rally[i].sort_unstable();
            per_rally[i].dedup();
            if per_rally[i].is_empty() {
                continue;
            }
            candidates.push(Candidate {
                key: format!("{}:{id}", run.run_id),
                candidate_id: id,
                rally_run_id: run.run_id.clone(),
                video: run.video.clone(),
                start,
                end,
                verified_attacks: physical_counts[i],
                initial_serves: per_rally_serves[i],
                human_reviewed: per_rally_manual[i],
                frames: per_rally[i].clone(),
            });
        }
        sources.push(json!({
            "video":run.video,"rally_run_id":run.run_id,
            "replay_sha256":run.replay_sha,
            "touch_sha256":run.touch_sha,"rally_sha256":run.rally_sha,
            "identity_revision":owners.revision,
            "resolved_owner_assignments":owners.owners,
            "excluded_track_ids":owners.excluded,
            "automatic_track_ids":automatic_track_ids,
            "action_revision":reviewed["revision"],
            "initial_serve_sha256":format!("{:x}", Sha256::digest(
                serde_json::to_vec(&serve_proposals)
                    .map_err(|e| storage::internal(format!("Encode serve proposals: {e}")))?
            )),
            "identity_source":"report_roster_reviewed_and_reconciled_auto",
            "action_source":"report_pose_contacts_and_trajectory_serve_candidates"
        }));
        runs.insert(run.run_id.clone(), run);
    }
    candidates.sort_by(|a, b| {
        b.frames
            .len()
            .cmp(&a.frames.len())
            .then_with(|| a.video.cmp(&b.video))
            .then_with(|| a.start.cmp(&b.start))
            .then_with(|| a.key.cmp(&b.key))
    });
    let fingerprint = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&json!({
                "project_id":request.project_id,"player_id":request.player_id,
                "action_kind":request.action_kind,"clip_context_sec":request.clip_context_sec,
                "sources":sources,
            }))
            .map_err(|e| storage::internal(format!("Encode highlights fingerprint: {e}")))?
        )
    );
    if let Some(expected) = expected {
        if expected != fingerprint {
            return Err(storage::conflict(
                "Reviewed player, actions or CPU sources changed since preview",
            ));
        }
    }
    let selected = select_candidates(
        &candidates,
        &request.selection_mode,
        &request.manual_rally_keys,
        request.max_clips,
    )
    .map_err(storage::bad_request)?;
    // No UI frame bounds or FFmpeg source paths are accepted. The shared
    // original-source clip selector reconstructs intervals from full SHA.
    let mut by_run = BTreeMap::<String, Vec<String>>::new();
    for item in &selected {
        by_run
            .entry(item.rally_run_id.clone())
            .or_default()
            .push(item.candidate_id.clone());
    }
    let mut checked = BTreeMap::<String, SelectedClip>::new();
    for (id, keys) in by_run {
        let run = runs
            .get(&id)
            .ok_or_else(|| storage::internal("Validated run absent"))?;
        let clips = native_rally_evidence::verified_manual_clips(run, keys.clone())
            .map_err(storage::bad_request)?;
        for (candidate, clip) in keys.into_iter().zip(clips) {
            checked.insert(format!("{id}:{candidate}"), clip);
        }
    }
    // Build source-authenticated short windows for EACH reviewed contact.
    // Never substitute an entire rally for an athlete contact reel.
    let context_before = (request.clip_context_sec * 30.0).round() as u64;
    let context_after = ((request.clip_context_sec + 0.35) * 30.0).round() as u64;
    let mut clips = Vec::new();
    for entry in &selected {
        let source = checked
            .remove(&entry.key)
            .ok_or_else(|| storage::internal("Missing verified source interval"))?;
        let mut frames = entry.frames.clone();
        frames.sort_unstable();
        let mut windows = Vec::<(u64, u64)>::new();
        for frame in frames {
            let start = frame.saturating_sub(context_before).max(source.start_frame);
            let end = frame.saturating_add(context_after).min(source.end_frame);
            if let Some(last) = windows.last_mut() {
                if start <= last.1.saturating_add(1) {
                    last.1 = last.1.max(end);
                    continue;
                }
            }
            windows.push((start, end));
        }
        for (start, end) in windows {
            let mut clip = SelectedClip {
                video: source.video.clone(),
                start_frame: start,
                end_frame: end,
                analysis_offset_sec: source.analysis_offset_sec,
            };
            if clips.len() == MAX_CLIPS {
                break;
            }
            // A very short final fragment cannot be rendered usefully.
            if clip.end_frame.saturating_sub(clip.start_frame) < 10 {
                clip.end_frame = (clip.start_frame + 10).min(source.end_frame);
            }
            clips.push(clip);
        }
        if clips.len() == MAX_CLIPS {
            break;
        }
    }
    let camera_tracks = if request.layout == VideoLayout::VerticalFollow {
        let mut by_source = vec![Vec::new(); clips.len()];
        for run in runs.values() {
            let indices: Vec<usize> = clips
                .iter()
                .enumerate()
                .filter_map(|(index, clip)| (clip.video == run.video).then_some(index))
                .collect();
            if indices.is_empty() {
                continue;
            }
            let subset = indices
                .iter()
                .map(|i| clips[*i].clone())
                .collect::<Vec<_>>();
            let calculated = native_rally_evidence::verified_camera_tracks(state, run, &subset)
                .map_err(storage::bad_request)?;
            for (index, track) in indices.into_iter().zip(calculated) {
                by_source[index] = track;
            }
        }
        Some(by_source)
    } else {
        None
    };
    let attack_total = selected.iter().map(|c| c.verified_attacks).sum::<u64>();
    let serve_total = selected.iter().map(|c| c.initial_serves).sum::<u64>();
    let matched_total = selected.iter().map(|c| c.frames.len() as u64).sum::<u64>();
    let reviewed_total = selected.iter().map(|c| c.human_reviewed).sum::<u64>();
    let view = json!({
        "schema":"volley-native-reviewed-player-highlights/1",
        "project_id":request.project_id,
        "player_id":request.player_id,"player_name":athlete_name,
        "selection_mode":request.selection_mode,
        "action_kind":request.action_kind,
        "clip_context_sec":request.clip_context_sec,
        "requested_duration_sec":request.target_duration_sec,
        "sources":sources,"source_fingerprint":fingerprint,
        "candidates":candidates.iter().take(80).map(Candidate::view).collect::<Vec<_>>(),
        "total_eligible_rallies":candidates.len(),
        "selected":selected.iter().map(Candidate::view).collect::<Vec<_>>(),
        "selected_verified_attacks":attack_total,
        "selected_verified_player_contacts":attack_total,
        "selected_initial_serve_candidates":serve_total,
        "selected_matching_player_events":matched_total,
        "selected_manually_reviewed_actions":reviewed_total,
        "selected_clips":clips.len(),
        "short_contact_segments_only":true,
        "reviewed_actions_only":false,
        "manually_linked_athlete_only":false,
        "automatic_action_classification":true,
        "action_classification_verified":false,
        "render_source":"original_recording",
        "cancellation_route":"/api/montage/native/cancel",
    });
    Ok(Plan {
        response: view,
        clips,
        camera_tracks,
    })
}
pub async fn highlights(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<Request>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let _permit = state
        .statistics_limit
        .acquire()
        .await
        .map_err(|e| storage::internal(format!("Highlights CPU gate: {e}")))?;
    let operation = request.operation.clone();
    let project_id = request.project_id.clone();
    let encoder = request.encoder.clone();
    let layout = request.layout;
    let audio_file = request.audio_file.clone();
    let audio_volume = request.audio_volume;
    let target_duration_sec = request.target_duration_sec;
    let camera_zoom = request.camera_zoom;
    let include_recognition_data = request.include_recognition_data;
    let preferred_rally_run_id = request.preferred_rally_run_id.clone();
    let state_copy = state.clone();
    let planned = tokio::task::spawn_blocking(move || {
        let _guard = project_guard::write_guard(&state_copy, &request.project_id)?;
        plan(&state_copy, &request)
    })
    .await
    .map_err(|e| storage::internal(format!("Highlights planning task: {e}")))??;
    drop(_permit);
    if operation == "preview" {
        return Ok(Json(planned.response));
    }
    if planned.clips.is_empty() {
        return Err(storage::bad_request(
            "Для выбранного игрока и типа действий нет подходящих событий внутри розыгрышей",
        ));
    }
    let evidence = json!({
        "schema":"volley-native-verified-player-highlights/1",
        "player_id":planned.response["player_id"],
        "player_name":planned.response["player_name"],
        "source_fingerprint":planned.response["source_fingerprint"],
        "sources":planned.response["sources"],
        "selected_rallies":planned.response["selected"],
        "selection_mode":planned.response["selection_mode"],
        "estimated_attack_count":if planned.response["action_kind"]=="attack" {planned.response["selected_verified_attacks"].clone()} else {json!(0)},
        "pose_verified_contact_count":planned.response["selected_verified_player_contacts"],
        "initial_serve_candidate_count":planned.response["selected_initial_serve_candidates"],
        "manually_reviewed_action_count":planned.response["selected_manually_reviewed_actions"],
        "action_kind":planned.response["action_kind"],
        "human_review_only":false,
        "automatic_action_classification_verified":false,
        "source_video":"original_recording",
    });
    let runs = planned.response["sources"]
        .as_array()
        .ok_or_else(|| storage::bad_request("Missing highlight sources"))?
        .iter()
        .filter_map(|source| source["rally_run_id"].as_str().map(str::to_owned))
        .collect();
    let materials =
        crate::montage_material::material_bindings(state.clone(), project_id.clone(), runs).await?;
    let Json(job) = montage::start_native_montage(
        State(state),
        headers,
        Json(montage::MontageRequest {
            materials,
            preferred_rally_run_id,
            project_id,
            clips: planned.clips,
            audio_file,
            audio_volume,
            encoder,
            layout,
            target_duration_sec,
            camera_zoom,
            include_recognition_data,
            camera_tracks: planned.camera_tracks,
            verified_highlights: Some(evidence),
            report_outro: None,
        }),
    )
    .await?;
    let mut response = planned.response;
    response["operation"] = json!("start");
    response["job_id"] = job["job_id"].clone();
    response["status"] = job["status"].clone();
    Ok(Json(response))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn candidate(run: &str, key: &str, count: u64) -> Candidate {
        Candidate {
            key: format!("{run}:{key}"),
            candidate_id: key.into(),
            rally_run_id: run.into(),
            video: format!("{run}.mp4"),
            start: 100,
            end: 180,
            verified_attacks: count,
            initial_serves: 0,
            human_reviewed: 0,
            frames: vec![120],
        }
    }
    #[test]
    fn report_predicted_attacks_are_available_without_manual_review() {
        let mut owners = native_player_tracks::NativeReviewedOwners {
            source_sha: "a".repeat(64),
            revision: "absent".into(),
            owners: BTreeMap::from([(7, "player-a".into()), (8, "player-b".into())]),
            excluded: BTreeSet::new(),
            roster: BTreeMap::from([
                ("player-a".into(), "Alice".into()),
                ("player-b".into(), "Bob".into()),
            ]),
        };
        // This is the already physically validated Report selector output.
        let report_contacts = vec![
            native_player_statistics::MontageContact {
                frame: 20,
                track_id: 7,
                estimated_kind: Some("attack".into()),
            },
            native_player_statistics::MontageContact {
                frame: 23,
                track_id: 7,
                estimated_kind: Some("dig".into()),
            },
            native_player_statistics::MontageContact {
                frame: 27,
                track_id: 8,
                estimated_kind: Some("attack".into()),
            },
            native_player_statistics::MontageContact {
                frame: 30,
                track_id: 7,
                estimated_kind: None,
            },
        ];
        let no_review = BTreeMap::new();
        assert_eq!(
            select_player_actions(&report_contacts, &owners, "player-a", "attack", &no_review),
            vec![(20, false)],
            "The attack shown in Report must appear in the player's reel"
        );
        assert_eq!(
            select_player_actions(&report_contacts, &owners, "player-a", "all", &no_review),
            vec![(20, false), (23, false), (30, false)]
        );
        assert!(
            select_player_actions(&report_contacts, &owners, "player-b", "dig", &no_review)
                .is_empty()
        );

        // Human correction overrides the estimate of this very contact.
        let reviewed = json!({"reviewed_actions":[{
            "action_kind":"block","action_verified":true,
            "motion_supported":true,"action_source":"explicit_human_review",
            "athlete_identity_verified":false,
            "frame_index":20,"provisional_track_id":7
        }]});
        let labels = reviewed_action_labels(&reviewed).unwrap();
        assert!(
            select_player_actions(&report_contacts, &owners, "player-a", "attack", &labels)
                .is_empty()
        );
        assert_eq!(
            select_player_actions(&report_contacts, &owners, "player-a", "block", &labels),
            vec![(20, true)]
        );
        // A manual label alone never creates an unobserved contact.
        let fake = BTreeMap::from([((29, 7), "attack".to_string())]);
        assert_eq!(
            select_player_actions(&report_contacts, &owners, "player-a", "attack", &fake),
            vec![(20, false)]
        );

        // Auto-IDs are just as valid as manual IDs once reconciled in stats_owners.
        owners.owners.insert(9, "player-a".into());
        let inferred = vec![native_player_statistics::MontageContact {
            frame: 44,
            track_id: 9,
            estimated_kind: Some("attack".into()),
        }];
        assert_eq!(
            select_player_actions(&inferred, &owners, "player-a", "attack", &no_review),
            vec![(44, false)]
        );
        owners.excluded.insert(9);
        assert!(
            select_player_actions(&inferred, &owners, "player-a", "attack", &no_review).is_empty()
        );

        let mut spoof = reviewed.clone();
        spoof["reviewed_actions"][0]["action_source"] = json!("automatic_reid");
        assert!(reviewed_action_labels(&spoof).is_err());
    }
    #[test]
    fn report_initial_serves_are_available_without_physical_contacts() {
        let mut owners = native_player_tracks::NativeReviewedOwners {
            source_sha: "a".repeat(64),
            revision: "stable".into(),
            owners: BTreeMap::from([
                (7, "player-a".into()),
                (8, "player-b".into()),
                (9, "player-a".into()),
            ]),
            excluded: BTreeSet::from([9]),
            roster: BTreeMap::from([
                ("player-a".into(), "Alice".into()),
                ("player-b".into(), "Bob".into()),
            ]),
        };
        let proposals = vec![
            json!({"rally_start_frame":10,"trajectory_start_frame":12,"provisional_track_id":7}),
            json!({"rally_start_frame":30,"trajectory_start_frame":31,"provisional_track_id":8}),
            json!({"rally_start_frame":30,"trajectory_start_frame":32,"provisional_track_id":null}),
            json!({"rally_start_frame":30,"trajectory_start_frame":33,"provisional_track_id":9}),
        ];
        let intervals = vec![("first".into(), 10, 20), ("second".into(), 30, 40)];
        let selected = select_initial_serve_frames(&proposals, &owners, "player-a");
        assert_eq!(selected, vec![12]);
        let mut contacts = vec![Vec::new(), Vec::new()];
        let mut counts = vec![0, 0];
        append_initial_serves(&intervals, &selected, &mut contacts, &mut counts).unwrap();
        assert_eq!(contacts, vec![vec![12], Vec::<u64>::new()]);
        assert_eq!(counts, vec![1, 0]);
        append_initial_serves(&intervals, &selected, &mut contacts, &mut counts).unwrap();
        assert_eq!(counts, vec![1, 0]);
        assert_eq!(contacts, vec![vec![12], Vec::<u64>::new()]);
        assert!(append_initial_serves(&intervals, &[21], &mut contacts, &mut counts).is_err());
        assert_eq!(
            select_initial_serve_frames(&proposals, &owners, "player-b"),
            vec![31]
        );
        owners.excluded.insert(7);
        assert!(select_initial_serve_frames(&proposals, &owners, "player-a").is_empty());
    }

    #[test]
    fn selected_contacts_are_restricted_to_verified_rallies() {
        let intervals = vec![("rally-1".into(), 10, 20), ("rally-2".into(), 30, 40)];
        assert_eq!(rally_for_frame(&intervals, 10), Some(0));
        assert_eq!(rally_for_frame(&intervals, 20), Some(0));
        assert_eq!(rally_for_frame(&intervals, 21), None);
        assert_eq!(rally_for_frame(&intervals, 30), Some(1));
        assert_eq!(rally_for_frame(&intervals, 41), None);
    }
    #[test]
    fn automatic_top_attacks_and_source_scoped_manual_overrides() {
        let mut rows = vec![
            candidate("runA", "a", 3),
            candidate("runB", "b", 2),
            candidate("runA", "c", 1),
        ];
        assert_eq!(
            select_candidates(&rows, "auto", &[], 2)
                .unwrap()
                .into_iter()
                .map(|r| r.key)
                .collect::<Vec<_>>(),
            vec!["runA:a", "runB:b"]
        );
        assert_eq!(
            select_candidates(&rows, "manual", &["runA:c".into(), "runA:a".into()], 2)
                .unwrap()
                .into_iter()
                .map(|r| r.key)
                .collect::<Vec<_>>(),
            vec!["runA:c", "runA:a"]
        );
        assert!(
            select_candidates(&rows, "manual", &["runA:a".into(), "runA:a".into()], 2).is_err()
        );
        assert!(select_candidates(&rows, "manual", &["runC:a".into()], 2).is_err());
        assert!(select_candidates(&rows, "auto", &["runA:a".into()], 2).is_err());
        assert!(select_candidates(&rows, "manual", &[], 2).is_err());
        assert!(select_candidates(&rows, "auto", &[], 0).is_err());
        rows[0].key = "runB:a".into();
        assert!(select_candidates(&rows, "manual", &["runA:a".into()], 2).is_err());
    }
}
