use super::*;
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallTrajectoryRequest {
    pub(super) project_id: String,
    pub(super) video: String,
    pub(super) replay_id: String,
    pub(super) expected_source_sha256: String,
    pub(super) start_frame: u64,
    #[serde(default = "trajectory_count")]
    pub(super) count: u64,
}
pub(super) fn trajectory_count() -> u64 {
    300
}
pub(super) fn extract_native_ball_trajectory(
    path: &Path,
    expected_frames: u64,
    expected_sha: &str,
    start: u64,
    count: u64,
    width: u32,
    height: u32,
) -> Result<(Vec<BallObservation>, Option<String>), String> {
    if count == 0
        || count > 1200
        || start
            .checked_add(count)
            .is_none_or(|end| end > expected_frames)
        || !(64..=16384).contains(&width)
        || !(64..=16384).contains(&height)
    {
        return Err("Selected-ball diagnostic window is out of bounds".into());
    }
    let mut reader = BoundedReader::open(path)?;
    let mut observations = Vec::new();
    let mut source_model = None::<String>;
    for frame_index in 0..expected_frames {
        let bytes = reader
            .line()?
            .ok_or("Native ball trajectory input truncated")?;
        let row: Frame = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Invalid saved native frame: {e}"))?;
        if row.frame_index != frame_index
            || !row.track_ids_provisional
            || row.confirmed_athlete_identities
            || row.touches_confirmed
        {
            return Err("Ball trajectory source lacks provisional native status".into());
        }
        if frame_index < start || frame_index >= start + count {
            continue;
        }
        let Some(ball) = row.selected_ball else {
            continue;
        };
        if ball.source_model != "vballnet"
            || ball.detection.class_name != "ball"
            || ball.detection.frame_index != frame_index
        {
            return Err("Saved ball trajectory contains unsupported detection".into());
        }
        ball.detection
            .validate(frame_index - frame_index % 9, "ball")?;
        let bbox = ball.detection.bbox_xyxy_norm;
        if source_model
            .as_ref()
            .is_some_and(|s| s != &ball.source_model)
        {
            return Err(
                "Selected-ball source switched in motion window; select a narrower window".into(),
            );
        }
        source_model = Some(ball.source_model);
        observations.push(BallObservation {
            frame: i64::try_from(frame_index).map_err(|_| "Frame index overflow")?,
            x: (bbox[0] + bbox[2]) * width as f64 * 0.5,
            y: (bbox[1] + bbox[3]) * height as f64 * 0.5,
        });
    }
    if reader.line()?.is_some() || reader.finish() != expected_sha {
        return Err("Saved native selected-ball replay SHA-256 has changed".into());
    }
    Ok((observations, source_model))
}
pub async fn ball_trajectory(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<BallTrajectoryRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    if req.count == 0 || req.count > 1200 {
        return Err(storage::bad_request(
            "Ball trajectory preview covers 1..1200 frames",
        ));
    }
    let _permit = state
        .touch_limit
        .acquire()
        .await
        .map_err(|e| storage::internal(format!("Ball CPU gate: {e}")))?;
    let worker = state.clone();
    let response = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        let _guard = project_guard::write_guard(&worker, &req.project_id)?;
        let ctx = source(&worker, &req.project_id, &req.video, &req.replay_id)?;
        if ctx.source_sha != req.expected_source_sha256 {
            return Err(storage::conflict("Selected-ball replay source SHA changed"));
        }
        let project = projects::project_root(&worker)
            .join(&req.project_id)
            .join("native_cpu_replays")
            .join(&req.replay_id);
        let manifest =
            safe_file(&project, "manifest.json", MAX_MANIFEST).map_err(storage::bad_request)?;
        let header: Value = serde_json::from_slice(
            &fs::read(manifest)
                .map_err(|e| storage::internal(format!("Read ball manifest: {e}")))?,
        )
        .map_err(|e| storage::bad_request(format!("Bad native ball manifest: {e}")))?;
        let frames = header["frames"]
            .as_u64()
            .ok_or_else(|| storage::bad_request("Native ball manifest lacks frame count"))?;
        let plan = preparation::plan_for(&worker, &req.project_id, &req.video)?;
        let prepared = plan.ready_path().ok_or_else(|| {
            storage::bad_request("Prepared video unavailable for trajectory dimensions")
        })?;
        let dimensions = volley_media::probe(prepared).map_err(storage::bad_request)?;
        let input = safe_file(&project, "tracked_frames.jsonl", MAX_EVIDENCE)
            .map_err(storage::bad_request)?;
        let (samples, source_model) = extract_native_ball_trajectory(
            &input,
            frames,
            &ctx.source_sha,
            req.start_frame,
            req.count,
            dimensions.width,
            dimensions.height,
        )
        .map_err(storage::bad_request)?;
        let extrema = ball_motion_extrema(&samples, 8).map_err(storage::bad_request)?;
        let trajectory_breaks = volley_core::motion::trajectory_breaks(&samples);
        Ok(json!({
            "schema":"volley-native-ball-extrema/1",
            "project_id":req.project_id,"video":req.video,"replay_id":req.replay_id,
            "source_replay_sha256":ctx.source_sha,
            "selected_ball_source":source_model,
            "window_start_frame":req.start_frame,"window_frames":req.count,
            "observations":samples.len(),"motion_extrema":extrema,
            "trajectory_breaks":trajectory_breaks,
            "trajectory_detector_version":volley_core::touch_pipeline::TRAJECTORY_DETECTOR_VERSION,
            "method":"gravity_compensated_rust_ball_motion",
            "touches_confirmed":false,"athlete_identities_verified":false,
            "gpu_inference_rerun":false
        }))
    })
    .await
    .map_err(|e| storage::internal(format!("Ball trajectory worker: {e}")))??;
    Ok(Json(response))
}

pub(crate) fn full_video_ball_suggestions(
    run: &crate::native_rally_evidence::ValidatedRun,
) -> Result<(Vec<(u64, f64)>, usize), String> {
    full_video_ball_suggestions_scoped(run, &[(0, run.frames.saturating_sub(1))])
}

pub(crate) fn full_video_ball_suggestions_scoped(
    run: &crate::native_rally_evidence::ValidatedRun,
    scope: &[(u64, u64)],
) -> Result<(Vec<(u64, f64)>, usize), String> {
    full_video_ball_suggestions_impl(run, scope, None)
}

pub(crate) fn full_video_ball_impacts_scoped(
    run: &crate::native_rally_evidence::ValidatedRun,
    scope: &[(u64, u64)],
    config: volley_core::motion::DirectionBreakConfig,
) -> Result<(Vec<(u64, f64)>, usize), String> {
    if !config.valid() {
        return Err("Invalid ball direction change parameters".into());
    }
    full_video_ball_suggestions_impl(run, scope, Some(config))
}

fn full_video_ball_suggestions_impl(
    run: &crate::native_rally_evidence::ValidatedRun,
    scope: &[(u64, u64)],
    impact: Option<volley_core::motion::DirectionBreakConfig>,
) -> Result<(Vec<(u64, f64)>, usize), String> {
    if scope
        .iter()
        .any(|&(start, end)| start > end || end >= run.frames)
        || scope.windows(2).any(|pair| pair[0].1 >= pair[1].0)
    {
        return Err("Invalid ball rally search intervals".into());
    }
    if run.frames == 0
        || run.frames > 48 * 3600 * 30
        || !(64..=16384).contains(&run.width)
        || !(64..=16384).contains(&run.height)
    {
        return Err("Invalid saved trajectory video dimensions".into());
    }
    let project = run
        .rally_file
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .ok_or("Invalid saved rally directory")?;
    let replay_dir = project.join("native_cpu_replays").join(&run.replay_id);
    let input = safe_file(&replay_dir, "tracked_frames.jsonl", MAX_EVIDENCE)?;
    let mut reader = BoundedReader::open(&input)?;
    let mut points = Vec::<BallObservation>::new();
    let mut candidates = Vec::<(u64, f64)>::new();
    let mut observations = 0usize;
    let mut scope_index = 0usize;
    fn emit(
        points: &[BallObservation],
        candidates: &mut Vec<(u64, f64)>,
        impact: Option<volley_core::motion::DirectionBreakConfig>,
    ) -> Result<(), String> {
        if points.len() < 9 {
            return Ok(());
        }
        let breaks = if let Some(config) = impact {
            volley_core::motion::trajectory_direction_breaks(points, config)
        } else {
            volley_core::motion::trajectory_breaks(points)
        };
        for b in breaks {
            if b.frame >= 0 {
                let score = if impact.is_some() {
                    b.motion.angle / (1.0 + b.fit_error)
                } else {
                    b.motion.residual / (1.0 + b.fit_error)
                };
                candidates.push((b.frame as u64, score));
            }
        }
        if impact.is_none() {
            for b in ball_motion_extrema(points, 5)? {
                candidates.push((b.frame_index, b.velocity_residual_px_frame * 0.4));
            }
        }
        Ok(())
    }
    for frame in 0..run.frames {
        let bytes = reader.line()?.ok_or("Saved ball replay is truncated")?;
        let row: Frame = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Invalid ball replay frame: {e}"))?;
        if row.frame_index != frame
            || !row.track_ids_provisional
            || row.confirmed_athlete_identities
            || row.touches_confirmed
        {
            return Err("Ball source frame state differs from saved replay".into());
        }
        while scope.get(scope_index).is_some_and(|&(_, end)| end < frame) {
            emit(&points, &mut candidates, impact)?;
            points.clear();
            scope_index += 1;
        }
        if !scope
            .get(scope_index)
            .is_some_and(|&(start, end)| start <= frame && frame <= end)
        {
            continue;
        }
        let Some(ball) = row.selected_ball else {
            if !points.is_empty() && frame as i64 - points.last().unwrap().frame > 4 {
                emit(&points, &mut candidates, impact)?;
                points.clear();
            }
            continue;
        };
        if ball.source_model != "vballnet"
            || ball.detection.class_name != "ball"
            || ball.detection.frame_index != frame
        {
            return Err("Unsupported saved ball detection".into());
        }
        ball.detection.validate(frame - frame % 9, "ball")?;
        if ball.detection.score < run.min_ball_score {
            continue;
        }
        let b = ball.detection.bbox_xyxy_norm;
        if points.last().is_some_and(|p| frame as i64 - p.frame > 4) {
            emit(&points, &mut candidates, impact)?;
            points.clear();
        }
        points.push(BallObservation {
            frame: frame as i64,
            x: (b[0] + b[2]) * run.width as f64 * 0.5,
            y: (b[1] + b[3]) * run.height as f64 * 0.5,
        });
        observations += 1;
        if points.len() >= 900 {
            emit(&points, &mut candidates, impact)?;
            points = points.split_off(points.len() - 24);
        }
    }
    emit(&points, &mut candidates, impact)?;
    if reader.line()?.is_some() || reader.finish() != run.replay_sha {
        return Err("Ball replay SHA-256 differs from saved source".into());
    }
    candidates.sort_by_key(|(f, _)| *f);
    // Local maxima from the two existing estimators refer to the same impact.
    let mut dedup = Vec::<(u64, f64)>::new();
    for p in candidates {
        if let Some(last) = dedup
            .last_mut()
            .filter(|q| q.0.abs_diff(p.0) <= impact.map_or(5, |c| c.min_separation_frames as u64))
        {
            if p.1 > last.1 {
                *last = p;
            }
        } else {
            dedup.push(p);
        }
    }
    if dedup.len() > 32768 {
        return Err("Too many ball trajectory events".into());
    }
    Ok((dedup, observations))
}

#[cfg(test)]
mod scored_trajectory_tests {
    use super::*;

    #[test]
    fn scans_saved_replay_and_rejects_changed_sha() {
        let root =
            std::env::temp_dir().join(format!("volley-scored-ball-{}", uuid::Uuid::new_v4()));
        let replay_id = "a".repeat(24);
        let rally_id = "b".repeat(24);
        let dir = root.join("native_cpu_replays").join(&replay_id);
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes = Vec::new();
        for frame in 0_u64..50 {
            let x = if frame <= 25 {
                0.15 + frame as f64 * 0.01
            } else {
                0.40 - (frame - 25) as f64 * 0.01
            };
            let ball = Candidate {
                frame_index: frame,
                class_name: "ball".into(),
                score: 0.95,
                bbox_xyxy_norm: [x, 0.40, x + 0.01, 0.42],
                query_index: None,
                touch_probability: None,
                head_point_norm: None,
            };
            let row = json!({
                "frame_index":frame,"tracked_players":[],
                "selected_ball":{"source_model":"vballnet","detection":ball},
                "track_ids_provisional":true,
                "confirmed_athlete_identities":false,"touches_confirmed":false
            });
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        let replay = dir.join("tracked_frames.jsonl");
        std::fs::write(&replay, &bytes).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let run = crate::native_rally_evidence::ValidatedRun {
            project_id: "project".into(),
            video: "match.mp4".into(),
            run_id: rally_id.clone(),
            project_source: root.join("match.mp4"),
            rally_file: root
                .join("native_cpu_rallies")
                .join(&rally_id)
                .join("rally_candidates.jsonl"),
            touch_file: root.join("touch.jsonl"),
            replay_id,
            replay_sha: digest,
            touch_id: "c".repeat(24),
            rally_sha: "d".repeat(64),
            touch_sha: "e".repeat(64),
            frames: 50,
            source_duration_sec: 50.0 / 30.0,
            analysis_range: None,
            width: 640,
            height: 360,
            context: 0,
            expected_rallies: 0,
            expected_touches: 0,
            min_ball_score: 0.35,
        };
        let (events, observed) = full_video_ball_suggestions(&run).unwrap();
        assert_eq!(observed, 50);
        assert!(!events.is_empty());
        assert!(events.iter().all(|(f, score)| *f < 50 && score.is_finite()));
        let (scoped, observed) = full_video_ball_suggestions_scoped(&run, &[(20, 30)]).unwrap();
        assert_eq!(observed, 11);
        assert!(scoped.iter().all(|(frame, _)| (20..=30).contains(frame)));
        let (impacts, _) = full_video_ball_impacts_scoped(
            &run,
            &[(0, 49)],
            volley_core::motion::DirectionBreakConfig::default(),
        )
        .unwrap();
        assert_eq!(impacts.len(), 1, "{impacts:?}");
        assert!(impacts[0].0.abs_diff(25) <= 2);
        let (strong_turn, _) = full_video_ball_impacts_scoped(
            &run,
            &[(0, 49)],
            volley_core::motion::DirectionBreakConfig {
                min_turn_angle_deg: 170.0,
                ..volley_core::motion::DirectionBreakConfig::default()
            },
        )
        .unwrap();
        assert_eq!(strong_turn.len(), 1);
        let (no_rallies, observed) = full_video_ball_suggestions_scoped(&run, &[]).unwrap();
        assert_eq!(observed, 0);
        assert!(no_rallies.is_empty());
        assert!(full_video_ball_suggestions_scoped(&run, &[(20, 30), (25, 40)]).is_err());
        let mut corrupted = run;
        corrupted.replay_sha = "0".repeat(64);
        assert!(full_video_ball_suggestions(&corrupted).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
