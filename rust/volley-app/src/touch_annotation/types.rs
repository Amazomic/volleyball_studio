use super::*;
use crate::native_preview_pose;
use std::collections::{BTreeMap, BTreeSet};
use volley_core::{
    action_prediction::{self, CourtContext, PositionEvidence},
    pose_classification::{self, PoseEvidence},
    touch_pipeline::{observed_pose_motion, PoseFrame, PosePlayer},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ClassifyRequest {
    project_id: String,
    rally_run_id: String,
    source_replay_sha256: String,
    rally: crate::rally_training_comparison::TouchRallyFilter,
    frames: Vec<u64>,
}

fn pose_window(
    stored: &BTreeMap<u64, Value>,
    frame: u64,
    total: u64,
) -> Option<(u64, Vec<Value>, [f64; 4], f64)> {
    let lo = frame.saturating_sub(9);
    let hi = frame.saturating_add(9).min(total.saturating_sub(1));
    let row = stored.get(&frame)?;
    let selected = row.get("selected_ball")?;
    if selected["source_model"] != "vballnet" {
        return None;
    }
    let ball = selected["detection"]["bbox_xyxy_norm"]
        .as_array()
        .filter(|a| a.len() == 4)?
        .iter()
        .map(Value::as_f64)
        .collect::<Option<Vec<_>>>()?;
    let bbox: [f64; 4] = ball.try_into().ok()?;
    if bbox
        .iter()
        .any(|n| !n.is_finite() || !(0.0..=1.0).contains(n))
        || bbox[0] >= bbox[2]
        || bbox[1] >= bbox[3]
    {
        return None;
    }
    let score = selected["detection"]["score"].as_f64()?;
    if !(0.0..=1.0).contains(&score) {
        return None;
    }
    let mut rows = Vec::with_capacity((hi - lo + 1) as usize);
    for at in lo..=hi {
        let frame_data = stored.get(&at)?;
        let players = frame_data["tracked_players"]
            .as_array()?
            .iter()
            .filter_map(|p| {
                let id = p["track_id"].as_u64()?;
                let bbox = p["detection"]["bbox_xyxy_norm"].as_array()?;
                (id > 0 && bbox.len() == 4 && p["is_verified_athlete"] == false)
                    .then(|| json!({"id":id,"box":bbox,"score":p["detection"]["score"]}))
            })
            .collect::<Vec<_>>();
        rows.push(json!({"frame":at,"players":players}));
    }
    Some((lo, rows, bbox, score))
}

fn classify(
    rows: &[Value],
    frame: u64,
    ball_box: [f64; 4],
    ball_score: f64,
    width: u32,
    height: u32,
) -> (Option<&'static str>, Option<u64>, Option<f64>) {
    let (w, h) = (width as f64, height as f64);
    let ball_x = (ball_box[0] + ball_box[2]) * 0.5;
    let ball_y = (ball_box[1] + ball_box[3]) * 0.5;
    let radius = ((ball_box[2] - ball_box[0]) * w + (ball_box[3] - ball_box[1]) * h) * 0.25;
    let poses = rows
        .iter()
        .filter_map(|row| {
            let at = row["frame"].as_u64()?;
            let players = row["players"]
                .as_array()?
                .iter()
                .filter_map(|player| {
                    let id = player["id"].as_u64()?;
                    let points =
                        serde_json::from_value::<[[f64; 3]; 17]>(player.get("pose")?.clone())
                            .ok()?;
                    Some(PosePlayer {
                        track_id: id,
                        keypoints: points,
                        interpolated: player["pose_interpolated"] == true,
                    })
                })
                .collect();
            Some(PoseFrame {
                frame_index: at,
                players,
            })
        })
        .collect::<Vec<_>>();
    let Some(current) = rows.iter().find(|row| row["frame"].as_u64() == Some(frame)) else {
        return (None, None, None);
    };
    let mut hypotheses = Vec::new();
    for player in current["players"].as_array().into_iter().flatten() {
        let Some(id) = player["id"].as_u64() else {
            continue;
        };
        let Some(bounds) = player["box"]
            .as_array()
            .filter(|a| a.len() == 4)
            .and_then(|a| a.iter().map(Value::as_f64).collect::<Option<Vec<_>>>())
            .and_then(|a| <[f64; 4]>::try_from(a).ok())
        else {
            continue;
        };
        if bounds
            .iter()
            .any(|n| !n.is_finite() || !(0.0..=1.0).contains(n))
            || bounds[0] >= bounds[2]
            || bounds[1] >= bounds[3]
        {
            continue;
        }
        let Some(track_score) = player["score"]
            .as_f64()
            .filter(|v| v.is_finite() && (0.35..=1.0).contains(v))
        else {
            continue;
        };
        let Some(points) = poses
            .iter()
            .find(|p| p.frame_index == frame)
            .and_then(|p| p.players.iter().find(|p| p.track_id == id))
        else {
            continue;
        };
        let evidence = PoseEvidence {
            keypoints: points.keypoints,
            ball_x_norm: ball_x,
            ball_y_norm: ball_y,
            video_width: w,
            video_height: h,
            player_bbox_norm: bounds,
        };
        let Some(distance) = pose_classification::contact_distance(&evidence, 0.35) else {
            continue;
        };
        if distance > 0.30 {
            continue;
        }
        let geometry = action_prediction::predict(PositionEvidence {
            ball_y_px: ball_y * h,
            ball_radius_px: radius,
            player_bbox_px: [bounds[0] * w, bounds[1] * h, bounds[2] * w, bounds[3] * h],
            standing_height_px: (bounds[3] - bounds[1]) * h,
            touch_score: ball_score,
            track_score,
            context: CourtContext::Unavailable,
        });
        let Some(geometry) = geometry else { continue };
        let Some((before, after)) = observed_pose_motion(&poses, frame, id) else {
            continue;
        };
        let Some(predicted) =
            pose_classification::refine_with_motion(&evidence, &geometry, &before, &after)
        else {
            continue;
        };
        hypotheses.push((id, distance, predicted));
    }
    hypotheses.sort_by(|a, b| a.1.total_cmp(&b.1));
    let Some((id, nearest, prediction)) = hypotheses.first() else {
        return (None, None, None);
    };
    if hypotheses
        .get(1)
        .is_some_and(|(_, other, _)| other - nearest < 0.10)
    {
        return (None, None, None);
    }
    (
        Some(prediction.kind.as_str()),
        Some(*id),
        Some(prediction.evidence_score),
    )
}

pub async fn calculate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ClassifyRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    let _permit = state
        .touch_limit
        .acquire()
        .await
        .map_err(|e| storage::internal(format!("Touch type gate: {e}")))?;
    let state = state.clone();
    let result = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        if req.frames.len() > 4096 || req.frames.windows(2).any(|w| w[0] >= w[1]) {
            return Err(storage::bad_request(
                "Touch frames must be sorted, unique and bounded",
            ));
        }
        let run = inspect_project(&state, &req.project_id, &req.rally_run_id)?;
        if run.replay_sha != req.source_replay_sha256 {
            return Err(storage::conflict("Ball candidate replay changed"));
        }
        let intervals =
            crate::rally_training_comparison::touch_rally_scope(&state, &run, &req.rally)?;
        let mut library = lock(|| read(&state))?;
        let scores = trajectory::scan_scoped_trajectory(&run, &intervals)?;
        trajectory::include_legacy_reviewed_labels(
            &mut library,
            &run.project_id,
            &run.video,
            &source_sha(&run)?,
            &run.replay_sha,
            &scores,
        );
        let profile = trajectory::trajectory_profile(&library);
        let expected = trajectory::predict(&scores, profile, run.frames)
            .into_iter()
            .filter(|f| intervals.iter().any(|&(a, b)| a <= *f && *f <= b))
            .collect::<Vec<_>>();
        if expected != req.frames {
            return Err(storage::conflict(
                "Touch candidates changed: recalculate ball touches",
            ));
        }
        let project = run
            .rally_file
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .ok_or_else(|| storage::bad_request("Missing project source root"))?;
        let replay = project.join("native_cpu_replays").join(&run.replay_id);
        let manifest: Value = serde_json::from_slice(
            &std::fs::read(
                storage::safe_file(&replay, "manifest.json", 16384)
                    .map_err(storage::bad_request)?,
            )
            .map_err(|e| storage::bad_request(e.to_string()))?,
        )
        .map_err(|e| storage::bad_request(format!("Invalid replay manifest: {e}")))?;
        let fusion = manifest["source_fusion_id"]
            .as_str()
            .filter(|f| volley_core::contracts::job_id(f))
            .ok_or_else(|| storage::bad_request("Missing verified source fusion"))?;
        if manifest["replay_id"] != run.replay_id
            || manifest["project_id"] != run.project_id
            || manifest["video"] != run.video
            || manifest["sha256"] != run.replay_sha
        {
            return Err(storage::conflict("CPU replay and RTMPose source differ"));
        }
        if native_preview_pose::source_info(project, fusion)
            .map_err(storage::bad_request)?
            .is_none()
        {
            return Err(storage::bad_request(
                "Для определения типа касания необходим сохранённый RTMPose",
            ));
        }
        let mut needed = BTreeSet::new();
        for &frame in &expected {
            for at in frame.saturating_sub(9)..=frame.saturating_add(9).min(run.frames - 1) {
                needed.insert(at);
            }
        }
        let path = storage::safe_file(&replay, "tracked_frames.jsonl", 64 * 1024 * 1024 * 1024)
            .map_err(storage::bad_request)?;
        let mut reader = storage::BoundedReader::open(&path).map_err(storage::bad_request)?;
        let mut stored = BTreeMap::new();
        for at in 0..run.frames {
            let line = reader
                .line()
                .map_err(storage::bad_request)?
                .ok_or_else(|| storage::bad_request("Truncated replay"))?;
            if !needed.contains(&at) {
                continue;
            }
            let row: Value =
                serde_json::from_slice(&line).map_err(|e| storage::bad_request(e.to_string()))?;
            if row["frame_index"] != at
                || row["track_ids_provisional"] != true
                || row["touches_confirmed"] != false
                || row["confirmed_athlete_identities"] != false
            {
                return Err(storage::conflict(
                    "Untrusted replay frame for type classification",
                ));
            }
            stored.insert(at, row);
        }
        if reader.line().map_err(storage::bad_request)?.is_some()
            || reader.finish() != run.replay_sha
        {
            return Err(storage::conflict(
                "Replay SHA changed while classifying touches",
            ));
        }
        let mut touches = Vec::with_capacity(expected.len());
        for frame in expected {
            let mut kind = None;
            let mut track = None;
            let mut score = None;
            if let Some((lo, mut rows, ball, ball_score)) = pose_window(&stored, frame, run.frames)
            {
                let end = lo + rows.len() as u64;
                let pose_ready = native_preview_pose::enrich_continuous_policy(
                    project,
                    fusion,
                    lo,
                    end,
                    &mut rows,
                    Some(true),
                )
                .map_err(storage::bad_request)?;
                if pose_ready {
                    (kind, track, score) =
                        classify(&rows, frame, ball, ball_score, run.width, run.height);
                }
            }
            touches.push(json!({"frame":frame,"kind":kind,"track_id":track,"score":score}));
        }
        let classified = touches.iter().filter(|v| v["kind"].is_string()).count();
        Ok(json!({"schema":"volley-trajectory-touch-types/1",
            "project_id":run.project_id,"video":run.video,"rally_run_id":run.run_id,
            "source_replay_sha256":run.replay_sha,
            "total_frames":run.frames,
            "method":"vballnet_candidates_rtmpose_motion_v1",
            "classified":classified,"total":touches.len(),"touches":touches,
            "gpu_rerun":false}))
    })
    .await
    .map_err(|e| storage::internal(format!("Touch type worker: {e}")))??;
    Ok(Json(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_pose_or_ball_never_produces_type() {
        let row = json!({"frame":10,"players":[{"id":2,"box":[0.2,0.2,0.6,0.8]}]});
        assert_eq!(
            classify(&[row], 10, [0.3, 0.3, 0.34, 0.34], 0.8, 1920, 1080).0,
            None
        );
        assert!(pose_window(&BTreeMap::new(), 10, 100).is_none());
    }

    #[test]
    fn observed_pose_motion_classifies_only_measured_candidate() {
        let mut joints = [[0.5, 0.5, 0.0]; 17];
        joints[5] = [0.43, 0.44, 0.95];
        joints[6] = [0.57, 0.44, 0.95];
        joints[7] = [0.41, 0.31, 0.96];
        joints[8] = [0.59, 0.31, 0.96];
        joints[9] = [0.49, 0.22, 0.96];
        joints[10] = [0.52, 0.22, 0.96];
        let mut before = joints;
        before[9][1] = 0.31;
        before[10][1] = 0.31;
        let mut after = joints;
        after[9][1] = 0.18;
        after[10][1] = 0.18;
        let row = |frame, points, interpolated| {
            json!({"frame":frame,"players":[{
                "id":11,"box":[0.35,0.17,0.65,0.90],"score":0.9,
                "pose":points,"pose_interpolated":interpolated
            }]})
        };
        let rows = vec![
            row(9, before, false),
            row(10, joints, true),
            row(11, after, false),
        ];
        let (kind, track, score) = classify(&rows, 10, [0.48, 0.18, 0.52, 0.22], 0.9, 640, 360);
        assert_eq!(kind, Some("overhead_pass"));
        assert_eq!(track, Some(11));
        assert!(score.is_some_and(|s| s > 0.0));
        let motionless = vec![
            row(9, joints, false),
            row(10, joints, true),
            row(11, joints, false),
        ];
        assert_eq!(
            classify(&motionless, 10, [0.48, 0.18, 0.52, 0.22], 0.9, 640, 360).0,
            None
        );
    }
}
