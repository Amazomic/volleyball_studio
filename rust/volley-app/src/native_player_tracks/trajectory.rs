use super::*;
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallTrajectoryRequest {
    pub(super) project_id: String,
    pub(super) video: String,
    pub(super) replay_id: String,
    #[serde(default)]
    pub(super) rally_run_id: Option<String>,
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
        if req.start_frame.checked_add(req.count).is_none_or(|end| end>frames) {
            return Err(storage::bad_request("Selected ball window is out of range"));
        }
        let plan = preparation::plan_for(&worker, &req.project_id, &req.video)?;
        let prepared = plan.ready_path().ok_or_else(|| {
            storage::bad_request("Prepared video unavailable for trajectory dimensions")
        })?;
        let dimensions = volley_media::probe(prepared).map_err(storage::bad_request)?;
        let input = safe_file(&project, "tracked_frames.jsonl", MAX_EVIDENCE)
            .map_err(storage::bad_request)?;
        let canonical = if let Some(rally_id) = req.rally_run_id.as_deref() {
            let run=crate::native_rally_evidence::inspect_project(
                &worker,&req.project_id,rally_id,
            )?;
            if run.replay_id != req.replay_id || run.replay_sha != ctx.source_sha
                || run.video != req.video || run.frames != frames
            {
                return Err(storage::conflict("Selected ball trajectory and postprocessing differ"));
            }
            let touch_dir=run.touch_file.parent()
                .ok_or_else(||storage::bad_request("Missing selected Touch manifest"))?;
            let manifest:Value=serde_json::from_slice(
                &fs::read(safe_file(touch_dir,"manifest.json",MAX_MANIFEST)
                    .map_err(storage::bad_request)?)
                    .map_err(|e|storage::bad_request(format!("Read Touch lineage: {e}")))?,
            ).map_err(|e|storage::bad_request(format!("Parse Touch lineage: {e}")))?;
            if manifest.get("ball_stabilizer").is_some() {
                Some(crate::cpu_touches::verified_ball_trajectory(
                    &projects::project_root(&worker).join(&req.project_id),
                    &manifest["ball_stabilizer"],&req.project_id,&req.video,
                    &req.replay_id,&ctx.source_sha,frames,
                ).map_err(storage::bad_request)?)
            } else { None }
        } else { None };
        let (samples, source_model) = if let Some((path,sha))=canonical.as_ref() {
            let mut reader=BoundedReader::open(path).map_err(storage::bad_request)?;
            let mut points=Vec::new();
            for frame in 0..frames {
                let bytes=reader.line().map_err(storage::bad_request)?
                    .ok_or_else(||storage::bad_request("Truncated stabilized trajectory"))?;
                let row:crate::cpu_touches::CanonicalBallFrame=serde_json::from_slice(&bytes)
                    .map_err(|e|storage::bad_request(format!("Invalid stabilized frame: {e}")))?;
                row.validate(frame).map_err(storage::bad_request)?;
                if frame>=req.start_frame && frame<req.start_frame.saturating_add(req.count) {
                    if let Some([x,y])=row.ball {
                        points.push(BallObservation{
                            frame:frame as i64,x:x*dimensions.width as f64,
                            y:y*dimensions.height as f64,
                        });
                    }
                }
            }
            if reader.line().map_err(storage::bad_request)?.is_some()
                || reader.finish()!=*sha
            {
                return Err(storage::bad_request("Canonical ball extrema source SHA mismatch"));
            }
            (points,Some("vballnet_stabilized".to_owned()))
        } else {
            extract_native_ball_trajectory(
                &input,frames,&ctx.source_sha,req.start_frame,req.count,
                dimensions.width,dimensions.height,
            ).map_err(storage::bad_request)?
        };
        let extrema = ball_motion_extrema(&samples, 8).map_err(storage::bad_request)?;
        let trajectory_breaks = volley_core::motion::trajectory_breaks(&samples);
        Ok(json!({
            "schema":"volley-native-ball-extrema/1",
            "project_id":req.project_id,"video":req.video,"replay_id":req.replay_id,
            "source_replay_sha256":ctx.source_sha,
            "selected_ball_source":source_model,
            "ball_trajectory_source":if canonical.is_some(){"ball_stabilizer_file_v1"}else{"legacy_cpu_replay_v1"},
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
    let result = full_video_ball_suggestions_impl(run, scope, None, None)?;
    Ok((result.events, result.observed))
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub(crate) struct BallReconstructionStats {
    pub raw_observations: usize,
    pub processed_frames: usize,
    pub source_rows_read: usize,
    pub rejected_outliers: usize,
    pub interpolated_frames: usize,
}

pub(crate) struct BallImpactScan {
    pub events: Vec<(u64, f64)>,
    pub ground_contacts: Vec<volley_core::motion::GroundContact>,
    pub observed: usize,
    pub reconstruction: BallReconstructionStats,
}

#[cfg(test)]
pub(crate) fn full_video_ball_impacts_with_reconstruction(
    run: &crate::native_rally_evidence::ValidatedRun,
    scope: &[(u64, u64)],
    config: volley_core::motion::DirectionBreakConfig,
) -> Result<BallImpactScan, String> {
    if !config.valid() {
        return Err("Invalid ball direction change parameters".into());
    }
    full_video_ball_suggestions_impl(run, scope, Some(config), None)
}

pub(crate) fn full_video_ball_impacts_with_court(
    run: &crate::native_rally_evidence::ValidatedRun,
    scope: &[(u64, u64)],
    config: volley_core::motion::DirectionBreakConfig,
    court: Option<&volley_core::court_geometry::CourtTransform>,
) -> Result<BallImpactScan, String> {
    if !config.valid() {
        return Err("Invalid ball direction change parameters".into());
    }
    full_video_ball_suggestions_impl(run, scope, Some(config), court)
}

fn full_video_ball_suggestions_impl(
    run: &crate::native_rally_evidence::ValidatedRun,
    scope: &[(u64, u64)],
    impact: Option<volley_core::motion::DirectionBreakConfig>,
    court: Option<&volley_core::court_geometry::CourtTransform>,
) -> Result<BallImpactScan, String> {
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
    let touch_dir = run
        .touch_file
        .parent()
        .ok_or("Missing Touch source directory")?;
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(safe_file(touch_dir, "manifest.json", 16 * 1024)?)
            .map_err(|e| format!("Read Touch trajectory lineage: {e}"))?,
    )
    .map_err(|e| format!("Invalid Touch trajectory lineage: {e}"))?;
    let canonical_source = if manifest.get("ball_stabilizer").is_some() {
        Some(crate::cpu_touches::verified_ball_trajectory(
            project,
            &manifest["ball_stabilizer"],
            &run.project_id,
            &run.video,
            &run.replay_id,
            &run.replay_sha,
            run.frames,
        )?)
    } else {
        None
    };
    let (input, expected_sha) = if let Some((path, sha)) = &canonical_source {
        (path.clone(), sha.clone())
    } else {
        let replay_dir = project.join("native_cpu_replays").join(&run.replay_id);
        (
            safe_file(&replay_dir, "tracked_frames.jsonl", MAX_EVIDENCE)?,
            run.replay_sha.clone(),
        )
    };
    let canonical = canonical_source.is_some();
    let mut points = Vec::<BallObservation>::new();
    let mut candidates = Vec::<(u64, f64)>::new();
    let mut ground_contacts = Vec::<volley_core::motion::GroundContact>::new();
    let mut observations = 0usize;
    let mut reconstruction = BallReconstructionStats::default();
    let mut synthetic = std::collections::HashSet::<i64>::new();
    let max_observation_gap = if canonical && impact.is_some() { 10 } else { 4 };
    let mut scope_index = 0usize;
    fn emit(
        points: &[BallObservation],
        events: (
            &mut Vec<(u64, f64)>,
            &mut Vec<volley_core::motion::GroundContact>,
        ),
        court: Option<(&volley_core::court_geometry::CourtTransform, u64)>,
        options: (Option<volley_core::motion::DirectionBreakConfig>, u32, u32),
        reconstruction: &mut BallReconstructionStats,
        canonical: bool,
        synthetic: &std::collections::HashSet<i64>,
    ) -> Result<(), String> {
        let (impact, width, height) = options;
        let prepared = if !canonical {
            impact
                .map(|config| -> Result<_, String> {
                    reconstruction.raw_observations += points.len();
                    let restored = volley_core::ball_interpolation::reconstruct_short_ball_motion(
                        points,
                        width,
                        height,
                        config.max_join_error_px,
                    )?;
                    reconstruction.rejected_outliers += restored.rejected_frames.len();
                    reconstruction.interpolated_frames += restored.interpolated_frames.len();
                    Ok(restored)
                })
                .transpose()?
        } else {
            None
        };
        let prepared_points = prepared
            .as_ref()
            .map_or(points, |restored| restored.observations.as_slice());
        if prepared_points.len() < 9 {
            return Ok(());
        }
        let breaks = if let Some(config) = impact {
            volley_core::motion::trajectory_direction_breaks(prepared_points, config)
        } else {
            volley_core::motion::trajectory_breaks(points)
        };
        for b in breaks {
            if b.frame >= 0
                && ((impact.is_some() && !b.contact_observed)
                    || (!synthetic.contains(&b.frame)
                        && !prepared.as_ref().is_some_and(|reconstructed| {
                            reconstructed
                                .interpolated_frames
                                .binary_search(&b.frame)
                                .is_ok()
                        })))
            {
                let score = if impact.is_some() {
                    b.motion.angle / (1.0 + b.fit_error)
                } else {
                    b.motion.residual / (1.0 + b.fit_error)
                };
                if let Some(contact) = court.and_then(|(court, rally_end)| {
                    volley_core::motion::classify_ground_contact(
                        prepared_points,
                        &b,
                        court,
                        width,
                        height,
                        rally_end,
                    )
                }) {
                    events.1.push(contact);
                } else {
                    events.0.push((b.frame as u64, score));
                }
            }
        }
        if impact.is_none() {
            for b in ball_motion_extrema(points, 5)? {
                if !synthetic.contains(&(b.frame_index as i64)) {
                    events
                        .0
                        .push((b.frame_index, b.velocity_residual_px_frame * 0.4));
                }
            }
        }
        Ok(())
    }
    let source_rows_read = crate::native_preview_overlay::for_each_verified_scoped_frame(
        &input,
        &expected_sha,
        run.frames,
        scope,
        |entry| {
            let Some((frame, bytes)) = entry else {
                emit(
                    &points,
                    (&mut candidates, &mut ground_contacts),
                    court.zip(scope.get(scope_index).map(|interval| interval.1)),
                    (impact, run.width, run.height),
                    &mut reconstruction,
                    canonical,
                    &synthetic,
                )?;
                points.clear();
                synthetic.clear();
                scope_index += 1;
                return Ok(());
            };
            reconstruction.processed_frames += 1;
            let (position, inferred) = if canonical {
                let row: crate::cpu_touches::CanonicalBallFrame = serde_json::from_slice(bytes)
                    .map_err(|e| format!("Invalid saved stabilized ball frame: {e}"))?;
                row.validate(frame)?;
                reconstruction.rejected_outliers += usize::from(row.rejected);
                reconstruction.interpolated_frames += usize::from(row.interpolated);
                (
                    row.ball.map(|[x, y]| BallObservation {
                        frame: frame as i64,
                        x: x * run.width as f64,
                        y: y * run.height as f64,
                    }),
                    row.interpolated,
                )
            } else {
                let row: Frame = serde_json::from_slice(bytes)
                    .map_err(|e| format!("Invalid ball replay frame: {e}"))?;
                if row.frame_index != frame
                    || !row.track_ids_provisional
                    || row.confirmed_athlete_identities
                    || row.touches_confirmed
                {
                    return Err("Ball source frame state differs from saved replay".into());
                }
                let candidate = match row.selected_ball {
                    Some(ball) => {
                        if ball.source_model != "vballnet"
                            || ball.detection.class_name != "ball"
                            || ball.detection.frame_index != frame
                        {
                            return Err("Unsupported saved ball detection".into());
                        }
                        ball.detection.validate(frame - frame % 9, "ball")?;
                        (ball.detection.score >= run.min_ball_score).then(|| {
                            let b = ball.detection.bbox_xyxy_norm;
                            BallObservation {
                                frame: frame as i64,
                                x: (b[0] + b[2]) * run.width as f64 * 0.5,
                                y: (b[1] + b[3]) * run.height as f64 * 0.5,
                            }
                        })
                    }
                    None => None,
                };
                (candidate, false)
            };
            if inferred {
                synthetic.insert(frame as i64);
            }
            let Some(ball) = position else {
                if !points.is_empty()
                    && frame as i64 - points.last().unwrap().frame > max_observation_gap
                {
                    emit(
                        &points,
                        (&mut candidates, &mut ground_contacts),
                        court.zip(scope.get(scope_index).map(|interval| interval.1)),
                        (impact, run.width, run.height),
                        &mut reconstruction,
                        canonical,
                        &synthetic,
                    )?;
                    points.clear();
                }
                return Ok(());
            };
            if points
                .last()
                .is_some_and(|p| frame as i64 - p.frame > max_observation_gap)
            {
                emit(
                    &points,
                    (&mut candidates, &mut ground_contacts),
                    court.zip(scope.get(scope_index).map(|interval| interval.1)),
                    (impact, run.width, run.height),
                    &mut reconstruction,
                    canonical,
                    &synthetic,
                )?;
                points.clear();
            }
            points.push(ball);
            observations += 1;
            if points.len() >= 900 {
                emit(
                    &points,
                    (&mut candidates, &mut ground_contacts),
                    court.zip(scope.get(scope_index).map(|interval| interval.1)),
                    (impact, run.width, run.height),
                    &mut reconstruction,
                    canonical,
                    &synthetic,
                )?;
                points = points.split_off(points.len() - 24);
            }
            Ok(())
        },
    )?;
    reconstruction.source_rows_read = source_rows_read;
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
    ground_contacts.sort_by_key(|contact| contact.frame);
    ground_contacts.dedup_by(|a, b| a.frame.abs_diff(b.frame) <= 2);
    dedup.retain(|(frame, _)| {
        let index =
            ground_contacts.partition_point(|contact| contact.frame.saturating_add(2) < *frame);
        ground_contacts
            .get(index)
            .is_none_or(|contact| frame.abs_diff(contact.frame) > 2)
    });
    if dedup.len() > 32768 || ground_contacts.len() > 32768 {
        return Err("Too many ball trajectory events".into());
    }
    reconstruction.raw_observations = observations;
    Ok(BallImpactScan {
        events: dedup,
        ground_contacts,
        observed: observations,
        reconstruction,
    })
}

#[cfg(test)]
mod scored_trajectory_tests {
    use super::*;

    #[test]
    fn canonical_stabilizer_gap_survives_debug_trajectory_scan() {
        let root = std::env::temp_dir().join(format!("volley-debug-gap-{}", uuid::Uuid::new_v4()));
        let replay_id = "a".repeat(24);
        let touch_id = "b".repeat(24);
        let rally_id = "c".repeat(24);
        let artifact_id = "d".repeat(32);
        let replay_sha = "e".repeat(64);
        let frames = 70_u64;
        let settings = json!({"max_gap_frames":12,"max_step_fraction":0.12});
        let stabilizer = root.join("native_cpu_ball_stabilizer").join(&artifact_id);
        let touch_dir = root.join("native_cpu_touch_runs").join(&touch_id);
        std::fs::create_dir_all(&stabilizer).unwrap();
        std::fs::create_dir_all(&touch_dir).unwrap();
        let mut bytes = Vec::new();
        let mut observed = 0;
        for frame in 0..frames {
            let ball = if (20..=48).contains(&frame) && !(30..=37).contains(&frame) {
                observed += 1;
                let t = frame as f64 - 29.0;
                let y = if frame <= 29 {
                    820.0 + 48.0 * t + 0.15 * t * t
                } else {
                    790.0 - 42.0 * t + 0.20 * t * t
                };
                Some([(1035.0 + 2.5 * t) / 1920.0, y / 1080.0])
            } else {
                None
            };
            let row = json!({
                "frame":frame,"ball":ball,"interpolated":false,
                "rejected":(30..=37).contains(&frame) || frame == 4
            });
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        let sha = format!("{:x}", Sha256::digest(&bytes));
        std::fs::write(stabilizer.join("trajectory.jsonl"), &bytes).unwrap();
        let summary = json!({
            "schema":"volley-ball-stabilizer/1","version":"ball_stabilizer_v6",
            "source_replay_id":replay_id,"source_replay_sha256":replay_sha,
            "artifact_id":artifact_id,"trajectory_file":"trajectory.jsonl",
            "trajectory_sha256":sha,"frames":frames,"settings":settings,
            "confidence_frames":observed,"observed_frames":observed,
            "rejected_frames":9,"interpolated_frames":0
        });
        let manifest = json!({
            "schema":"volley-native-stabilized-ball-trajectory/1","state":"complete",
            "project_id":"test","video":"match.mp4","artifact_id":artifact_id,
            "source_replay_id":replay_id,"source_replay_sha256":replay_sha,
            "frames":frames,"algorithm_version":"ball_stabilizer_v6",
            "settings":settings,"file":"trajectory.jsonl","sha256":sha,
            "confidence_frames":observed,"observed_frames":observed,
            "rejected_frames":9,"interpolated_frames":0
        });
        std::fs::write(
            stabilizer.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        std::fs::write(
            touch_dir.join("manifest.json"),
            serde_json::to_vec(&json!({"ball_stabilizer":summary})).unwrap(),
        )
        .unwrap();
        let run = crate::native_rally_evidence::ValidatedRun {
            project_id: "test".into(),
            video: "match.mp4".into(),
            run_id: rally_id.clone(),
            project_source: root.join("match.mp4"),
            rally_file: root
                .join("native_cpu_rallies")
                .join(&rally_id)
                .join("rally_candidates.jsonl"),
            touch_file: touch_dir.join("touch_candidates.jsonl"),
            replay_id,
            replay_sha,
            touch_id,
            rally_sha: "f".repeat(64),
            touch_sha: "f".repeat(64),
            frames,
            source_duration_sec: frames as f64 / 30.0,
            analysis_range: None,
            width: 1920,
            height: 1080,
            context: 0,
            expected_rallies: 0,
            expected_touches: 0,
            min_ball_score: 0.35,
        };
        let result = full_video_ball_impacts_with_reconstruction(
            &run,
            &[(19, 48)],
            volley_core::motion::DirectionBreakConfig::default(),
        )
        .unwrap();
        assert!(
            result
                .events
                .iter()
                .any(|(frame, _)| (28..=31).contains(frame)),
            "Saved canonical gap must yield one debug candidate: {:?}",
            result.events
        );
        assert_eq!(result.reconstruction.rejected_outliers, 8);
        assert_eq!(result.reconstruction.processed_frames, 30);
        let court = volley_core::court_geometry::CourtTransform::new([
            [0.1, 0.2],
            [0.9, 0.2],
            [0.9, 0.8],
            [0.1, 0.8],
        ])
        .unwrap();
        let classified = full_video_ball_impacts_with_court(
            &run,
            &[(19, 48)],
            volley_core::motion::DirectionBreakConfig::default(),
            Some(&court),
        )
        .unwrap();
        assert!(
            classified
                .ground_contacts
                .iter()
                .any(|contact| (28..=31).contains(&contact.frame)),
            "Court bounce should not remain a player touch: {:?}",
            classified.ground_contacts
        );
        assert!(
            !classified
                .events
                .iter()
                .any(|(frame, _)| (28..=31).contains(frame)),
            "Ground bounce leaked back into player touches: {:?}",
            classified.events
        );
        std::fs::remove_dir_all(root).unwrap();
    }
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
            let x = if frame == 30 {
                0.95
            } else if frame <= 25 {
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
                "selected_ball":if frame==31 { Value::Null } else {
                    json!({"source_model":"vballnet","detection":ball})
                },
                "track_ids_provisional":true,
                "confirmed_athlete_identities":false,"touches_confirmed":false
            });
            bytes.extend(serde_json::to_vec(&row).unwrap());
            bytes.push(b'\n');
        }
        let replay = dir.join("tracked_frames.jsonl");
        std::fs::write(&replay, &bytes).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let touch_dir = root.join("native_cpu_touch_runs").join("c".repeat(24));
        std::fs::create_dir_all(&touch_dir).unwrap();
        std::fs::write(touch_dir.join("manifest.json"), b"{}").unwrap();
        let run = crate::native_rally_evidence::ValidatedRun {
            project_id: "project".into(),
            video: "match.mp4".into(),
            run_id: rally_id.clone(),
            project_source: root.join("match.mp4"),
            rally_file: root
                .join("native_cpu_rallies")
                .join(&rally_id)
                .join("rally_candidates.jsonl"),
            touch_file: touch_dir.join("touch_candidates.jsonl"),
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
        assert_eq!(observed, 49);
        assert!(!events.is_empty());
        assert!(events.iter().all(|(f, score)| *f < 50 && score.is_finite()));
        let (scoped, observed) = full_video_ball_suggestions_scoped(&run, &[(20, 30)]).unwrap();
        assert_eq!(observed, 11);
        assert!(scoped.iter().all(|(frame, _)| (20..=30).contains(frame)));
        let impacts = full_video_ball_impacts_with_reconstruction(
            &run,
            &[(0, 49)],
            volley_core::motion::DirectionBreakConfig::default(),
        )
        .unwrap();
        assert_eq!(impacts.events.len(), 1);
        assert!(impacts.events[0].0.abs_diff(25) <= 2);
        assert_eq!(impacts.reconstruction.rejected_outliers, 1);
        assert!(impacts.reconstruction.interpolated_frames >= 1);
        let strong_turn = full_video_ball_impacts_with_reconstruction(
            &run,
            &[(0, 49)],
            volley_core::motion::DirectionBreakConfig {
                min_turn_angle_deg: 170.0,
                ..volley_core::motion::DirectionBreakConfig::default()
            },
        )
        .unwrap();
        assert_eq!(strong_turn.events.len(), 1);
        let (no_rallies, observed) = full_video_ball_suggestions_scoped(&run, &[]).unwrap();
        assert_eq!(observed, 0);
        assert!(no_rallies.is_empty());
        assert!(full_video_ball_suggestions_scoped(&run, &[(20, 30), (25, 40)]).is_err());
        let mut corrupted = run;
        corrupted.replay_sha = "0".repeat(64);
        assert!(full_video_ball_suggestions(&corrupted).is_err());
        let mut damaged_outside_scope = b"not-json\n".to_vec();
        damaged_outside_scope
            .extend_from_slice(&bytes[bytes.iter().position(|&byte| byte == b'\n').unwrap() + 1..]);
        std::fs::write(&replay, &damaged_outside_scope).unwrap();
        corrupted.replay_sha = format!("{:x}", Sha256::digest(&damaged_outside_scope));
        let only_rally = full_video_ball_impacts_with_reconstruction(
            &corrupted,
            &[(20, 30)],
            volley_core::motion::DirectionBreakConfig::default(),
        )
        .unwrap();
        assert_eq!(only_rally.reconstruction.processed_frames, 11);
        assert_eq!(only_rally.reconstruction.source_rows_read, 31);
        assert_eq!(only_rally.observed, 11);
        assert!(only_rally
            .events
            .iter()
            .all(|(frame, _)| (20..=30).contains(frame)));
        std::fs::remove_dir_all(root).unwrap();
    }
}
