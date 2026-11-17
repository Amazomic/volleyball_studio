//! Source-verified game report: rank long, eventful rallies and append a
//! human-review-only athlete statistics slate. No Track ID guesses or proxy pixels.
use crate::{
    montage, native_action_review, native_player_tracks, native_rally_evidence, project_guard,
    projects,
    storage::{self, ApiError, ApiResult},
    AppState,
};
use axum::{extract::State, http::HeaderMap, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};
use volley_media::montage::{SelectedClip, VideoLayout};

const MAX_RUNS: usize = 40;
const MAX_CLIPS: usize = 80;
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReportRequest {
    project_id: String,
    rally_run_ids: Vec<String>,
    #[serde(default = "default_operation")]
    operation: String,
    #[serde(default)]
    expected_source_fingerprint: Option<String>,
    #[serde(default = "default_duration")]
    target_duration_sec: f64,
    #[serde(default)]
    audio_file: Option<String>,
    #[serde(default = "default_volume")]
    audio_volume: f64,
    #[serde(default)]
    encoder: Option<String>,
    #[serde(default)]
    layout: VideoLayout,
    #[serde(default = "default_zoom")]
    camera_zoom: f64,
}
fn default_operation() -> String {
    "preview".into()
}
fn default_duration() -> f64 {
    60.0
}
fn default_zoom() -> f64 {
    1.25
}
fn default_volume() -> f64 {
    40.0
}

struct ReportPlan {
    clips: Vec<SelectedClip>,
    outro: Value,
    preview: Value,
    camera_tracks: Option<Vec<Vec<volley_media::montage::CameraPoint>>>,
}
#[derive(Clone)]
struct Ranked {
    run_id: String,
    video: String,
    candidate_id: String,
    start: u64,
    end: u64,
    touches: u64,
    coverage: f64,
}
fn report_plan(state: &AppState, req: &ReportRequest) -> Result<ReportPlan, ApiError> {
    storage::project_id(&req.project_id)?;
    let _project = projects::existing_project(state, &req.project_id)?;
    if req.rally_run_ids.is_empty()
        || req.rally_run_ids.len() > MAX_RUNS
        || req.rally_run_ids.iter().collect::<HashSet<_>>().len() != req.rally_run_ids.len()
        || !matches!(req.operation.as_str(), "preview" | "start")
        || !req.target_duration_sec.is_finite()
        || !(8.0..=600.0).contains(&req.target_duration_sec)
    {
        return Err(storage::bad_request(
            "Выберите CPU-версии и длину отчёта 8–600 секунд",
        ));
    }
    let mut sources = Vec::new();
    let mut videos = HashSet::new();
    let mut runs = BTreeMap::new();
    let mut ranked = Vec::<Ranked>::new();
    let mut stats = BTreeMap::<String, (u64, u64, u64)>::new();
    let mut ids = req.rally_run_ids.clone();
    ids.sort();
    for id in &ids {
        let run = native_rally_evidence::inspect_project(state, &req.project_id, id)?;
        if !videos.insert(run.video.clone()) {
            return Err(storage::bad_request(
                "Нельзя смешивать CPU-версии одного видео",
            ));
        }
        let raw = native_rally_evidence::verified_quick_rally_candidates(&run)
            .map_err(storage::bad_request)?;
        for (candidate_id, start, end, touches, coverage) in raw {
            let seconds = (end - start + 1) as f64 / 30.0;
            if seconds < 3.0 || touches < 2 || !coverage.is_finite() {
                continue;
            }
            ranked.push(Ranked {
                run_id: id.clone(),
                video: run.video.clone(),
                candidate_id,
                start,
                end,
                touches,
                coverage,
            });
        }
        let (owners, _auto_ids) = native_player_tracks::statistics_owners(
            state,
            &req.project_id,
            &run.video,
            &run.replay_id,
        )?;
        if owners.source_sha != run.replay_sha {
            return Err(storage::conflict(
                "Привязки игроков принадлежат другой GPU-версии",
            ));
        }
        let reviewed = native_action_review::verified_actions(state, &run)?;
        if reviewed["source_rally_sha256"] != run.rally_sha
            || reviewed["source_touch_sha256"] != run.touch_sha
        {
            return Err(storage::conflict(
                "Метки действий принадлежат другой CPU-версии",
            ));
        }
        // Exactly the same SHA-verified, rally-scoped body-contact counts
        // as the Report tab. Unreviewed identity stays an explicit Track ID.
        for (name, (touches, attacks, blocks)) in
            crate::native_player_statistics::rally_report_counts(state, &run, &owners)?
        {
            let entry = stats.entry(name).or_default();
            entry.0 += touches;
            entry.1 += attacks;
            entry.2 += blocks;
        }
        sources.push(json!({
            "video":run.video, "rally_run_id":run.run_id,
            "replay_sha":run.replay_sha, "touch_sha":run.touch_sha,
            "rally_sha":run.rally_sha,
            "identity_revision":owners.revision,
            "resolved_identity_mapping":owners.owners,
            "action_revision":reviewed["revision"],
        }));
        runs.insert(id.clone(), run);
    }
    ranked.sort_by(|a, b| {
        let score = |item: &Ranked| {
            let duration = (item.end - item.start + 1) as f64 / 30.0;
            duration.min(45.0) * 0.65 + item.touches.min(15) as f64 * 3.0 + item.coverage * 5.0
        };
        score(b)
            .total_cmp(&score(a))
            .then_with(|| b.touches.cmp(&a.touches))
            .then_with(|| a.start.cmp(&b.start))
    });
    let mut chosen = Vec::new();
    let mut length = 0.0;
    let budget = req.target_duration_sec - 5.0;
    for row in ranked {
        let seconds = (row.end - row.start + 1) as f64 / 30.0;
        if length >= budget || chosen.len() == MAX_CLIPS {
            break;
        }
        if length + seconds > budget && budget - length < 2.0 {
            continue;
        }
        length += seconds.min(budget - length);
        chosen.push(row);
    }
    if chosen.is_empty() {
        return Err(storage::bad_request(
            "Нет подходящих проверенных розыгрышей для отчёта",
        ));
    }
    let mut clips = Vec::new();
    for row in &chosen {
        let run = runs
            .get(&row.run_id)
            .ok_or_else(|| storage::internal("Missing CPU source"))?;
        let Some(clip) =
            native_rally_evidence::verified_manual_clips(run, vec![row.candidate_id.clone()])
                .map_err(storage::bad_request)?
                .into_iter()
                .next()
        else {
            return Err(storage::conflict("Report rally source disappeared"));
        };
        clips.push(clip);
    }
    let camera_tracks = if req.layout == VideoLayout::VerticalFollow {
        let mut by_clip = vec![Vec::new(); clips.len()];
        for run in runs.values() {
            let indices: Vec<usize> = clips
                .iter()
                .enumerate()
                .filter_map(|(i, clip)| (clip.video == run.video).then_some(i))
                .collect();
            if indices.is_empty() {
                continue;
            }
            let subset = indices
                .iter()
                .map(|i| clips[*i].clone())
                .collect::<Vec<_>>();
            let derived = native_rally_evidence::verified_camera_tracks(state, run, &subset)
                .map_err(storage::bad_request)?;
            for (i, track) in indices.into_iter().zip(derived) {
                by_clip[i] = track;
            }
        }
        Some(by_clip)
    } else {
        None
    };
    let mut lines = vec![
        "СТАТИСТИКА МАТЧА".to_owned(),
        format!("Розыгрышей в ролике: {}", chosen.len()),
        "Оценочные действия внутри розыгрышей:".to_owned(),
    ];
    if stats.is_empty() {
        lines.push("Не обнаружены касания с подтверждённой точкой тела".into());
    } else {
        let mut rows = stats.iter().collect::<Vec<_>>();
        rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then_with(|| a.0.cmp(b.0)));
        for (name, (total, attacks, blocks)) in rows.into_iter().take(9) {
            let clean = name.replace(['\n', '\r'], " ");
            let name: String = clean.chars().take(26).collect();
            lines.push(format!(
                "{name}: {total} кас., {attacks} атак, {blocks} блоков"
            ));
        }
    }
    lines.push("Подтверждены контакты тела; неподписанные игроки обозначены Track ID".into());
    let source_fingerprint = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&json!({
                "project_id":req.project_id,
                "target_duration_sec":req.target_duration_sec,
                "sources":sources,
            }))
            .map_err(|e| storage::internal(format!("Report fingerprint: {e}")))?
        )
    );
    if req.operation == "start"
        && req.expected_source_fingerprint.as_deref() != Some(&source_fingerprint)
    {
        return Err(storage::conflict(
            "CPU-версии, привязки или ручная разметка изменились. Повторите предпросмотр",
        ));
    }
    let outro = json!({
        "schema":"volley-native-reviewed-report/1",
        "lines":lines,"sources":sources,"source_fingerprint":source_fingerprint,
        "statistics_source":"verified_rally_pose_contacts_and_reviewed_track_links",
    });
    let preview = json!({
        "schema":"volley-native-game-report/1",
        "project_id":req.project_id,"source_fingerprint":source_fingerprint,
        "selected_rallies":chosen.len(),
        "selected_content_sec":length,
        "requested_duration_sec":req.target_duration_sec,
        "selected":chosen.iter().map(|row|json!({
            "video":row.video,"start_frame":row.start,"end_frame":row.end,
            "touches":row.touches,"ball_coverage":row.coverage
        })).collect::<Vec<_>>(),
        "statistics":lines,
        "statistics_reviewed_only":false
    });
    Ok(ReportPlan {
        clips,
        outro,
        preview,
        camera_tracks,
    })
}
pub async fn report(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ReportRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let operation = req.operation.clone();
    let project_id = req.project_id.clone();
    let audio_file = req.audio_file.clone();
    let audio_volume = req.audio_volume;
    let encoder = req.encoder.clone();
    let layout = req.layout;
    let target = req.target_duration_sec;
    let camera_zoom = req.camera_zoom;
    let worker = state.clone();
    let plan = tokio::task::spawn_blocking(move || {
        let _guard = project_guard::write_guard(&worker, &req.project_id)?;
        report_plan(&worker, &req)
    })
    .await
    .map_err(|e| storage::internal(format!("Report planning: {e}")))??;
    if operation == "preview" {
        return Ok(Json(plan.preview));
    }
    let runs = plan.outro["sources"]
        .as_array()
        .ok_or_else(|| storage::bad_request("Missing report sources"))?
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
            project_id,
            clips: plan.clips,
            audio_file,
            audio_volume,
            encoder,
            layout,
            target_duration_sec: Some(target),
            camera_zoom,
            camera_tracks: plan.camera_tracks,
            verified_highlights: None,
            report_outro: Some(plan.outro),
        }),
    )
    .await?;
    let mut response = plan.preview;
    response["status"] = job["status"].clone();
    response["job_id"] = job["job_id"].clone();
    Ok(Json(response))
}
