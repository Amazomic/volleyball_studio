//! Saved-trajectory only detector calibration from explicit manual labels.
//! Correction records are never substituted for model predictions.
use super::*;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CalculateRequest {
    project_id: String,
    rally_run_id: String,
    rally: crate::rally_training_comparison::TouchRallyFilter,
    #[serde(default)]
    impact: volley_core::motion::DirectionBreakConfig,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TrajectoryCandidate {
    pub frame: u64,
    pub score: f64,
}
#[derive(Clone, Copy, Serialize)]
pub(super) struct TrajectoryProfile {
    min_score: f64,
    shift_frames: i64,
    training_videos: usize,
    detector_version: &'static str,
}
pub(super) fn predict(
    candidates: &[TrajectoryCandidate],
    profile: TrajectoryProfile,
    total: u64,
) -> Vec<u64> {
    let mut frames = candidates
        .iter()
        .filter(|c| c.score >= profile.min_score)
        .filter_map(|c| c.frame.checked_add_signed(profile.shift_frames))
        .filter(|&frame| frame < total)
        .collect::<Vec<_>>();
    frames.sort_unstable();
    frames.dedup();
    frames
}
// Linear-time threshold search, with exact O(n*m) matching reserved for
// final per-video diagnostic metrics. Arrays are sorted by frame.
fn quick_score(expected: &[u64], predicted: &[u64]) -> (usize, usize, usize, u64) {
    let (mut i, mut j, mut hits, mut distance) = (0, 0, 0, 0);
    while i < expected.len() && j < predicted.len() {
        let left = expected[i];
        let right = predicted[j];
        if left.saturating_add(1) < right {
            i += 1;
            continue;
        }
        if right.saturating_add(1) < left {
            j += 1;
            continue;
        }
        hits += 1;
        distance += left.abs_diff(right);
        i += 1;
        j += 1;
    }
    (
        hits,
        predicted.len() - hits,
        expected.len() - hits,
        distance,
    )
}
pub(super) fn trajectory_profile(lib: &Library) -> TrajectoryProfile {
    let examples = lib
        .videos
        .iter()
        .filter(|v| !v.trajectory_replay_sha.is_empty() && v.total_frames > 0)
        .collect::<Vec<_>>();
    let mut best = TrajectoryProfile {
        min_score: 8.0,
        shift_frames: 0,
        training_videos: examples.len(),
        detector_version: "calibrated_ball_motion_v1",
    };
    if examples.is_empty() {
        return best;
    }
    let mut quality = -1.0_f64;
    let mut error = f64::INFINITY;
    let (mut least_extra, mut least_missed) = (usize::MAX, usize::MAX);
    for min_score in [
        2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 13.0, 17.0, 22.0, 30.0, 40.0, 60.0, 100.0, 10000.1,
    ] {
        for shift in -2_i64..=2 {
            let profile = TrajectoryProfile {
                min_score,
                shift_frames: shift,
                ..best
            };
            let (mut hits, mut extra, mut missed, mut total_error) = (0, 0, 0, 0.0);
            for v in &examples {
                let actual = predict(&v.trajectory_candidates, profile, v.total_frames);
                let mut expected = v.touches.iter().map(|t| t.frame).collect::<Vec<_>>();
                expected.sort_unstable();
                let (m, fp, fn_count, difference) = quick_score(&expected, &actual);
                hits += m;
                extra += fp;
                missed += fn_count;
                total_error += difference as f64 * 1000.0 / 30.0;
            }
            let denom = 2 * hits + extra + missed;
            let f1 = if denom == 0 {
                0.0
            } else {
                2.0 * hits as f64 / denom as f64
            };
            let delta = if hits == 0 {
                f64::INFINITY
            } else {
                total_error / hits as f64
            };
            let tie = (f1 - quality).abs() < 1e-10;
            let improvement = extra < least_extra
                || (extra == least_extra && missed < least_missed)
                || (extra == least_extra
                    && missed == least_missed
                    && (delta < error - 1e-6
                        || ((delta - error).abs() < 1e-6
                            && shift.abs() < best.shift_frames.abs())));
            if f1 > quality + 1e-10 || (tie && improvement) {
                least_extra = extra;
                least_missed = missed;
                quality = f1;
                error = delta;
                best = profile;
            }
        }
    }
    best
}
#[cfg(test)]
pub(super) fn include_legacy_reviewed_labels(
    lib: &mut Library,
    project_id: &str,
    video: &str,
    source_sha: &str,
    replay_sha: &str,
    scores: &[TrajectoryCandidate],
) -> bool {
    if let Some(labels) = lib.videos.iter_mut().find(|labels| {
        labels.project_id == project_id
            && labels.video == video
            && labels.source_sha256 == source_sha
            && labels.reviewed
            && labels.trajectory_replay_sha != replay_sha
    }) {
        labels.trajectory_replay_sha = replay_sha.to_owned();
        labels.trajectory_candidates = scores.to_vec();
        return true;
    }
    false
}
pub(super) fn scan_trajectory(run: &ValidatedRun) -> Result<Vec<TrajectoryCandidate>, ApiError> {
    let (events, _count) = crate::native_player_tracks::full_video_ball_suggestions(run)
        .map_err(storage::bad_request)?;
    Ok(events
        .into_iter()
        .map(|(frame, score)| TrajectoryCandidate {
            frame,
            score: score.clamp(0.0, 10000.0),
        })
        .collect())
}
pub(super) fn scan_collision_trajectory(
    run: &ValidatedRun,
    intervals: &[(u64, u64)],
    impact: volley_core::motion::DirectionBreakConfig,
) -> Result<
    (
        Vec<TrajectoryCandidate>,
        crate::native_player_tracks::BallReconstructionStats,
    ),
    ApiError,
> {
    if !impact.valid() {
        return Err(storage::bad_request("Invalid collision detector settings"));
    }
    let scan = crate::native_player_tracks::full_video_ball_impacts_with_reconstruction(
        run, intervals, impact,
    )
    .map_err(storage::bad_request)?;
    let candidates = scan
        .events
        .into_iter()
        .map(|(frame, score)| TrajectoryCandidate {
            frame,
            score: score.clamp(0.0, 10000.0),
        })
        .collect();
    Ok((candidates, scan.reconstruction))
}

pub(super) fn collision_frames(scores: &[TrajectoryCandidate], total: u64) -> Vec<u64> {
    let mut frames = scores
        .iter()
        .map(|score| score.frame)
        .filter(|frame| *frame < total)
        .collect::<Vec<_>>();
    frames.sort_unstable();
    frames.dedup();
    frames
}

fn inside_rally(frame: u64, intervals: &[(u64, u64)]) -> bool {
    let i = intervals.partition_point(|&(_, end)| end < frame);
    intervals
        .get(i)
        .is_some_and(|&(start, end)| start <= frame && frame <= end)
}

pub async fn calculate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<CalculateRequest>,
) -> ApiResult<Value> {
    storage::validate_origin(&headers)?;
    if !req.impact.valid() {
        return Err(storage::bad_request("Invalid collision detector settings"));
    }
    let _permit = state
        .touch_limit
        .acquire()
        .await
        .map_err(|e| storage::internal(format!("Trajectory calculation gate: {e}")))?;
    let worker = state.clone();
    let output = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        let state = worker;
        let run = inspect_project(&state, &req.project_id, &req.rally_run_id)?;
        let intervals = crate::rally_training_comparison::touch_rally_scope(
            &state, &run, &req.rally,
        )?;
        let source = source_sha(&run)?;
        let lib = lock(|| read(&state))?;
        let labels = lib
            .videos
            .iter()
            .find(|v| v.project_id == run.project_id && v.video == run.video);
        if labels.is_some_and(|v| v.source_sha256 != source) {
            return Err(storage::conflict("Source video changed since annotation"));
        }
        let truth = labels.map(|v| {
            let mut frames = v
                .touches
                .iter()
                .map(|t| t.frame)
                .filter(|&frame| inside_rally(frame, &intervals))
                .collect::<Vec<_>>();
            frames.sort_unstable();
            frames
        });
        let (scores, preprocessing) = scan_collision_trajectory(&run, &intervals, req.impact)?;
        let selected = collision_frames(&scores, run.frames);
        let calibration = TrajectoryProfile {
            min_score: 0.0,
            shift_frames: 0,
            training_videos: 0,
            detector_version: "touch_detector_v4",
        };
        let proposals = scores
            .iter()
            .filter(|score| selected.binary_search(&score.frame).is_ok())
            .map(|score| json!({"frame":score.frame,"score":score.score}))
            .collect::<Vec<_>>();
        let metrics = truth.map(|frames| accuracy(&frames, &selected, 1, 30.0));
        Ok(json!({"schema":"volley-trajectory-touch-proposals/1",
            "project_id":run.project_id,"video":run.video,"rally_run_id":run.run_id,
            "source_replay_sha256":run.replay_sha,"total_frames":run.frames,
            "proposals":proposals,"candidate_count":scores.len(),
            "rally_intervals":intervals.iter().map(|&(start,end)| json!({"start":start,"end":end})).collect::<Vec<_>>(),
            "rally_count":intervals.len(),
            "calibration":calibration,"impact_settings":req.impact,
            "ball_motion_interpolation":{"stage":"ball_motion_interpolation_v1","result":preprocessing},
            "metrics_on_labeled_video":metrics,
            "gpu_rerun":false,"player_pose_required":false,
            "method":"touch_detector_v4"}))
    })
    .await
    .map_err(|e| storage::internal(format!("Trajectory worker: {e}")))??;
    Ok(Json(output))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn labeled(touches: &[u64], candidates: Vec<TrajectoryCandidate>) -> Library {
        Library {
            schema: SCHEMA.into(),
            videos: vec![VideoLabels {
                project_id: "test".into(),
                video: "match.mp4".into(),
                source_sha256: "a".repeat(64),
                rally_run_id: "0".repeat(24),
                total_frames: 200,
                revision: 2,
                reviewed: false,
                quality: None,
                baseline_touches: vec![],
                baseline_rallies: vec![],
                baseline_touch_sha: String::new(),
                baseline_rally_sha: String::new(),
                trajectory_replay_sha: "b".repeat(64),
                trajectory_candidates: candidates,
                touches: touches
                    .iter()
                    .map(|&frame| Mark {
                        id: Uuid::new_v4().to_string(),
                        frame,
                    })
                    .collect(),
                rallies: vec![],
            }],
        }
    }
    #[test]
    fn search_bounds_exclude_contacts_outside_ball_rallies() {
        let ranges = [(10, 20), (40, 50)];
        assert!(!inside_rally(9, &ranges));
        assert!(inside_rally(10, &ranges));
        assert!(inside_rally(20, &ranges));
        assert!(!inside_rally(21, &ranges));
        assert!(inside_rally(50, &ranges));
        assert!(!inside_rally(51, &ranges));
        assert!(!inside_rally(20, &[]));
        let profile = TrajectoryProfile {
            min_score: 8.0,
            shift_frames: 2,
            training_videos: 0,
            detector_version: "test",
        };
        let proposed = predict(&[c(8, 9.0), c(25, 9.0), c(48, 9.0)], profile, 100)
            .into_iter()
            .filter(|&frame| inside_rally(frame, &ranges))
            .collect::<Vec<_>>();
        assert_eq!(proposed, vec![10, 50]);
    }

    #[test]
    fn existing_reviewed_video_calibrates_without_resaving_annotations() {
        let mut lib = labeled(&[102], vec![]);
        lib.videos[0].reviewed = true;
        lib.videos[0].trajectory_replay_sha.clear();
        let original = serde_json::to_value(&lib.videos[0]).unwrap();
        let samples = vec![c(100, 9.0), c(140, 3.0)];
        assert!(!include_legacy_reviewed_labels(
            &mut lib,
            "test",
            "match.mp4",
            &"0".repeat(64),
            &"b".repeat(64),
            &samples,
        ));
        assert!(!include_legacy_reviewed_labels(
            &mut lib,
            "test",
            "other.mp4",
            &"a".repeat(64),
            &"b".repeat(64),
            &samples,
        ));
        assert_eq!(lib.videos[0].trajectory_replay_sha, "");
        assert!(include_legacy_reviewed_labels(
            &mut lib,
            "test",
            "match.mp4",
            &"a".repeat(64),
            &"b".repeat(64),
            &samples,
        ));
        let profile = trajectory_profile(&lib);
        assert_eq!(profile.training_videos, 1);
        assert_eq!(profile.shift_frames, 2);
        assert_eq!(
            predict(&lib.videos[0].trajectory_candidates, profile, 200),
            vec![102]
        );
        assert_ne!(serde_json::to_value(&lib.videos[0]).unwrap(), original);
        let mut unreviewed = labeled(&[102], vec![]);
        unreviewed.videos[0].trajectory_replay_sha.clear();
        assert!(!include_legacy_reviewed_labels(
            &mut unreviewed,
            "test",
            "match.mp4",
            &"a".repeat(64),
            &"b".repeat(64),
            &samples,
        ));
        assert_eq!(trajectory_profile(&unreviewed).training_videos, 0);
    }
    #[test]
    fn saved_manual_timing_calibrates_recalculation() {
        let lib = labeled(&[102], vec![c(100, 9.0), c(140, 3.0)]);
        let model = trajectory_profile(&lib);
        assert_eq!(model.training_videos, 1);
        assert_eq!(model.shift_frames, 2);
        assert_eq!(
            predict(&lib.videos[0].trajectory_candidates, model, 200),
            vec![102]
        );
    }
    #[test]
    fn saved_negative_labels_suppress_false_positives() {
        let lib = labeled(&[], vec![c(10, 7.0), c(60, 13.0)]);
        let model = trajectory_profile(&lib);
        assert!(predict(&lib.videos[0].trajectory_candidates, model, 200).is_empty());
    }
    fn c(frame: u64, score: f64) -> TrajectoryCandidate {
        TrajectoryCandidate { frame, score }
    }
    #[test]
    fn shift_and_cutoff_change_predictions_without_copying_labels() {
        let samples = vec![c(100, 9.0), c(115, 2.0), c(140, 11.0)];
        let profile = TrajectoryProfile {
            min_score: 8.0,
            shift_frames: -1,
            training_videos: 1,
            detector_version: "test",
        };
        assert_eq!(predict(&samples, profile, 200), vec![99, 139]);
    }
}
