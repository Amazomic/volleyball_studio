use crate::{
    native_rally_evidence::{generate, inspect_project, read_manifest, Choice, ValidatedRun},
    project_guard, projects,
    storage::{self, safe_file, safe_folder, ApiError},
    AppState,
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, sync::Arc};

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct MaterialBinding {
    pub(crate) project_id: String,
    pub(crate) rally_run_id: String,
    pub(crate) revision: String,
    pub(crate) video: String,
    pub(crate) original_source: Value,
    pub(crate) intervals: Vec<(u64, u64)>,
}

pub(crate) fn montage_material(state: &AppState, run: &ValidatedRun) -> Result<Value, ApiError> {
    let mut timeline = generate(run, Choice::Timeline)
        .map_err(storage::bad_request)?
        .value;
    let manifest = read_manifest(
        run.rally_file
            .parent()
            .ok_or_else(|| storage::bad_request("Rally folder missing"))?,
        "Rally",
    )
    .map_err(storage::bad_request)?;
    // A montage replays the immutable CPU run, not the latest editable court.
    // The geometry revision recorded in the run remains authoritative.
    // Later manual court edits must not invalidate saved original-video clips.
    let geometry_available = !manifest["court_revision"].is_null();
    let (owners, automatic) = crate::native_player_tracks::statistics_owners(
        state,
        &run.project_id,
        &run.video,
        &run.replay_id,
    )?;
    let reviewed_actions = crate::native_action_review::verified_actions(state, run)?;
    if owners.source_sha != run.replay_sha {
        return Err(storage::conflict("Timeline identity source changed"));
    }
    let annotations = timeline["touch_annotations"]
        .as_array_mut()
        .ok_or_else(|| storage::bad_request("Missing final contacts"))?;
    for contact in annotations.iter_mut() {
        let track = contact["provisional_track_id"].as_u64();
        let owner = track.and_then(|id| owners.owners.get(&id));
        contact["reviewed_action"] = reviewed_actions["reviewed_actions"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|action| {
                action["frame_index"] == contact["frame_index"]
                    && action["provisional_track_id"] == contact["provisional_track_id"]
            })
            .cloned()
            .unwrap_or(Value::Null);
        contact["player_id"] = json!(owner);
        contact["player_name"] = json!(owner.and_then(|id| owners.roster.get(id)));
        contact["athlete_identity_verified"] =
            json!(owner.is_some() && track.is_some_and(|id| !automatic.contains(&id)));
        contact["automatic_identity"] = json!(track.is_some_and(|id| automatic.contains(&id)));
    }
    let attributed = annotations
        .iter()
        .filter(|row| row["touch_attributed"] == true)
        .collect::<Vec<_>>();
    let unknown = attributed
        .iter()
        .filter(|row| row["kind"].is_null())
        .count();
    let unidentified = attributed
        .iter()
        .filter(|row| row["player_id"].is_null())
        .count();
    let clips = timeline["rallies"]
        .as_array()
        .ok_or_else(|| storage::bad_request("Missing final rallies"))?
        .iter()
        .map(|row| {
            let start = row["start_frame"]
                .as_u64()
                .unwrap_or(0)
                .saturating_sub(run.context);
            let end = row["end_frame"]
                .as_u64()
                .unwrap_or(start)
                .saturating_add(run.context)
                .min(run.frames.saturating_sub(1));
            json!({"candidate_id":row["candidate_id"],"video":run.video,
                "start_frame":start,"end_frame":end,"analysis_offset_sec":0.0,
                "start_sec":start as f64 / 30.0,"end_sec":(end+1) as f64 / 30.0,
                "render_source":"original_recording"})
        })
        .collect::<Vec<_>>();
    let original = volley_media::preparation::source_provenance(&run.project_source)
        .map_err(storage::bad_request)?;
    let dependency = json!({"source_replay_id":run.replay_id,"source_replay_sha256":run.replay_sha,
        "source_touch_id":run.touch_id,"source_touch_sha256":run.touch_sha,
        "rally_run_id":run.run_id,"source_rallies_sha256":run.rally_sha,
        "geometry_revision":manifest["court_revision"],"identity_revision":owners.revision,
        "action_revision":reviewed_actions["revision"],
        "owners":owners.owners,"automatic_ids":automatic,"roster":owners.roster,
        "original_source":original,"analysis_original_verified":!manifest["original_source"].is_null()});
    timeline["material_schema"] = json!("volley-montage-material/1");
    timeline["dependencies"] = dependency;
    timeline["montage_plan"] = json!(clips);
    timeline["readiness"] = json!({"manual_montage_ready":!clips.is_empty(),
        "personalized_selection_available":!owners.owners.is_empty(),
        "unidentified_contacts":unidentified,"unknown_action_contacts":unknown,
        "geometry_available":geometry_available,"authoritative_game_events":false,
        "original_verified_against_analysis":!manifest["original_source"].is_null()});
    finalize_montage_material(timeline)
}

// JSON numbers must be normalized through the same deserialize/serialize
// cycle as the persisted sidecar. Otherwise a float can change representation
// when the file is loaded, invalidating its own content digest at start time.
fn finalize_montage_material(mut timeline: Value) -> Result<Value, ApiError> {
    for _ in 0..8 {
        let bytes = serde_json::to_vec(&timeline)
            .map_err(|e| storage::internal(format!("Timeline normalization: {e}")))?;
        let normalized: Value = serde_json::from_slice(&bytes)
            .map_err(|e| storage::internal(format!("Timeline normalization: {e}")))?;
        if normalized == timeline {
            let revision = crate::versioned_store::revision(Some(&bytes));
            timeline["material_revision"] = json!(revision);
            return Ok(timeline);
        }
        timeline = normalized;
    }
    Err(storage::internal(
        "Montage material numbers did not stabilize",
    ))
}

pub(crate) fn publish_material(
    state: &AppState,
    project: &str,
    rally: &str,
) -> Result<Value, ApiError> {
    let _guard = project_guard::write_guard_for_postprocessing(state, project)?;
    let run = inspect_project(state, project, rally)?;
    let material = montage_material(state, &run)?;
    let revision = material["material_revision"]
        .as_str()
        .ok_or_else(|| storage::bad_request("Missing timeline revision"))?;
    let parent = run
        .rally_file
        .parent()
        .ok_or_else(|| storage::bad_request("Rally folder missing"))?
        .join("montage_timelines");
    if parent.exists() {
        safe_folder(&parent).map_err(storage::bad_request)?;
    }
    let path = parent.join(format!("{revision}.json"));
    if let Some(bytes) = crate::versioned_store::read_optional_bounded(&path, 128 * 1024 * 1024)
        .map_err(storage::bad_request)?
    {
        let existing: Value = serde_json::from_slice(&bytes)
            .map_err(|_| storage::bad_request("Corrupt montage timeline"))?;
        if existing != material {
            return Err(storage::conflict("Immutable montage timeline differs"));
        }
    } else {
        crate::versioned_store::write_json_checked(&parent, &path, &material, 128 * 1024 * 1024)
            .map_err(storage::internal)?;
    }
    Ok(
        json!({"schema":"volley-montage-material/1","revision":revision,
        "rally_run_id":rally,"readiness":material["readiness"]}),
    )
}

pub(crate) async fn material_bindings(
    state: Arc<AppState>,
    project: String,
    runs: Vec<String>,
) -> Result<Vec<MaterialBinding>, ApiError> {
    tokio::task::spawn_blocking(move || {
        let mut bindings = Vec::new();
        for run in runs.into_iter().collect::<BTreeSet<_>>() {
            let published = publish_material(&state, &project, &run)?;
            let revision = published["revision"]
                .as_str()
                .ok_or_else(|| storage::bad_request("Material revision missing"))?;
            let root = projects::project_root(&state)
                .join(&project)
                .join("native_cpu_rallies")
                .join(&run)
                .join("montage_timelines");
            safe_folder(&root).map_err(storage::bad_request)?;
            let path = safe_file(&root, &format!("{revision}.json"), 128 * 1024 * 1024)
                .map_err(storage::bad_request)?;
            let material: Value = serde_json::from_slice(
                &fs::read(path)
                    .map_err(|e| storage::internal(format!("Read montage material: {e}")))?,
            )
            .map_err(|_| storage::bad_request("Invalid montage material"))?;
            let mut content = material.clone();
            content
                .as_object_mut()
                .ok_or_else(|| storage::bad_request("Invalid material object"))?
                .remove("material_revision");
            let encoded = serde_json::to_vec(&content)
                .map_err(|e| storage::internal(format!("Material digest: {e}")))?;
            if crate::versioned_store::revision(Some(&encoded)) != revision {
                return Err(storage::conflict("Saved montage material digest differs"));
            }
            let intervals = material["montage_plan"]
                .as_array()
                .ok_or_else(|| storage::bad_request("Missing montage intervals"))?
                .iter()
                .map(|clip| {
                    Ok((
                        clip["start_frame"]
                            .as_u64()
                            .ok_or_else(|| storage::bad_request("Invalid montage start"))?,
                        clip["end_frame"]
                            .as_u64()
                            .ok_or_else(|| storage::bad_request("Invalid montage end"))?,
                    ))
                })
                .collect::<Result<Vec<_>, ApiError>>()?;
            bindings.push(MaterialBinding {
                project_id: project.clone(),
                rally_run_id: run,
                revision: revision.to_owned(),
                video: material["video"].as_str().unwrap_or_default().to_owned(),
                original_source: material["dependencies"]["original_source"].clone(),
                intervals,
            });
        }
        Ok(bindings)
    })
    .await
    .map_err(|e| storage::internal(format!("Montage material worker: {e}")))?
}

pub(crate) fn validate_material_binding(
    state: &AppState,
    binding: &MaterialBinding,
) -> Result<(), ApiError> {
    let run = inspect_project(state, &binding.project_id, &binding.rally_run_id)?;
    let current = montage_material(state, &run)?;
    if current["material_revision"] != binding.revision
        || current["dependencies"]["original_source"] != binding.original_source
    {
        return Err(storage::conflict(
            "Montage material changed; review the selection again",
        ));
    }
    volley_media::preparation::verify_source_provenance(
        &run.project_source,
        &binding.original_source,
    )
    .map_err(storage::conflict)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn montage_material_digest_survives_pretty_sidecar_roundtrip() {
        let source = json!({
            "montage_plan": [{"start_sec": 0.12345678901234567, "end_sec": 12.345678901234567}],
            "dependencies": {"original_source": {"sha256": "abc"}},
        });
        let finalized = finalize_montage_material(source).unwrap();
        let bytes = serde_json::to_vec_pretty(&finalized).unwrap();
        let mut restored: Value = serde_json::from_slice(&bytes).unwrap();
        let revision = restored["material_revision"].as_str().unwrap().to_owned();
        restored
            .as_object_mut()
            .unwrap()
            .remove("material_revision");
        let encoded = serde_json::to_vec(&restored).unwrap();
        assert_eq!(crate::versioned_store::revision(Some(&encoded)), revision);
    }
}
