use super::*;

pub(crate) const FILE: &str = "ground_detector.json";
const LIMIT: usize = 30_000;

pub(crate) struct PrimaryTouchDetection {
    pub(crate) breaks: Vec<volley_core::motion::TrajectoryBreak>,
    pub(crate) contacts: Vec<volley_core::motion::GroundContact>,
    pub(crate) summary: Value,
}

pub(crate) fn candidates(
    motion: &volley_core::ball_interpolation::StabilizedBallMotion,
    impacts: &[volley_core::motion::TrajectoryBreak],
    intervals: &[(u64, u64)],
    court: Option<&CourtTransform>,
    width: u32,
    height: u32,
    cancel: &AtomicBool,
) -> Result<Vec<volley_core::motion::GroundContact>, String> {
    let Some(court) = court else {
        return Ok(Vec::new());
    };
    let mut result = Vec::new();
    for impact in impacts {
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let Ok(frame) = u64::try_from(impact.frame) else {
            continue;
        };
        let index = intervals.partition_point(|&(_, end)| end < frame);
        let Some(&(start, end)) = intervals.get(index) else {
            continue;
        };
        if frame < start {
            continue;
        }
        let first = impact.frame.saturating_sub(32).max(0);
        let last = impact
            .frame
            .saturating_add(48)
            .min(motion.points.len() as i64 - 1);
        let first = first
            .min(*impact.before_frames.first().unwrap_or(&first))
            .max(0);
        let last = last
            .max(*impact.after_frames.last().unwrap_or(&last))
            .min(motion.points.len() as i64 - 1);
        let points = (first..=last)
            .filter_map(|frame| {
                let [x, y] = motion.points.get(usize::try_from(frame).ok()?)?.as_ref()?;
                Some(volley_core::motion::BallObservation {
                    frame,
                    x: x * f64::from(width),
                    y: y * f64::from(height),
                })
            })
            .collect::<Vec<_>>();
        let observed = points
            .iter()
            .copied()
            .filter(|point| {
                !motion
                    .interpolated
                    .get(point.frame as usize)
                    .copied()
                    .unwrap_or(true)
            })
            .collect::<Vec<_>>();
        if let Some(contact) = volley_core::motion::classify_ground_contact_with_observed(
            &points, &observed, impact, court, width, height, end,
        ) {
            if result.len() >= LIMIT {
                return Err("Ground candidate limit exceeded".into());
            }
            result.push(contact);
        }
    }
    let mut used_frames = result
        .iter()
        .map(|c| c.frame)
        .collect::<std::collections::BTreeSet<_>>();
    for &(start, end) in intervals {
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let first = (start as usize).saturating_sub(12).min(motion.points.len());
        let last = (end as usize).saturating_add(31).min(motion.points.len());
        let points = motion.points[first..last]
            .iter()
            .enumerate()
            .filter_map(|(offset, point)| {
                let frame = first + offset;
                let [x, y] = (*point)?;
                Some(volley_core::motion::BallObservation {
                    frame: frame as i64,
                    x: x * f64::from(width),
                    y: y * f64::from(height),
                })
            })
            .collect::<Vec<_>>();
        let observed = points
            .iter()
            .copied()
            .filter(|point| {
                !motion
                    .interpolated
                    .get(point.frame as usize)
                    .copied()
                    .unwrap_or(true)
            })
            .collect::<Vec<_>>();
        let terminal =
            volley_core::motion::terminal_ground_contacts(&points, court, width, height, end)
                .into_iter()
                .filter(|contact| {
                    let frame = contact.frame as usize;
                    let measured = |i: usize| {
                        motion.points.get(i).is_some_and(Option::is_some)
                            && !motion.interpolated.get(i).copied().unwrap_or(true)
                    };
                    let observed_impact = measured(frame)
                        && (frame.saturating_sub(4)..frame)
                            .filter(|&i| measured(i))
                            .count()
                            >= 2;
                    let bracketed_impact = !measured(frame)
                        && (frame.saturating_sub(5)..frame)
                            .filter(|&i| measured(i))
                            .count()
                            >= 2
                        && (frame + 1..=frame + 4).filter(|&i| measured(i)).count() >= 2
                        && (frame.saturating_sub(3)..frame).any(measured)
                        && (frame + 1..=frame + 3).any(measured);
                    (observed_impact || bracketed_impact)
                        && (frame + 1..=frame + 14).filter(|&i| measured(i)).count() >= 2
                });
        for contact in
            volley_core::motion::rolling_ground_contacts(&observed, court, width, height, end)
                .into_iter()
                .chain(terminal)
                .chain(volley_core::motion::terminal_loss_ground_contacts(
                    &observed, court, width, height, end,
                ))
        {
            if !(start..=end).contains(&contact.frame) {
                continue;
            }
            if result.len() >= LIMIT {
                return Err("Ground candidate limit exceeded".into());
            }
            if used_frames
                .range(contact.frame.saturating_sub(6)..=contact.frame.saturating_add(6))
                .next()
                .is_none()
            {
                used_frames.insert(contact.frame);
                result.push(contact);
            }
        }
    }
    result.sort_unstable_by_key(|contact| contact.frame);
    Ok(result)
}

pub(crate) fn publish(
    folder: &Path,
    contacts: &[volley_core::motion::GroundContact],
    trajectory_sha: &str,
    dimensions: (u32, u32),
    frames: u64,
    court: Option<&video_edits::CourtSnapshot>,
) -> Result<Value, String> {
    let mut summary = json!({
        "schema":"volley-ground-detector/1",
        "version":volley_core::motion::GROUND_DETECTOR_VERSION,
        "source_trajectory_sha256":trajectory_sha,
        "court_revision":court.map(|c| c.revision.as_str()),
        "calibrated":court.is_some_and(|c| c.transform.is_some()),
        "width":dimensions.0,"height":dimensions.1,"frames":frames,
        "candidate_count":contacts.len(),"file":FILE
    });
    let path = folder.join(FILE);
    if path.exists() {
        return Err("Ground Detector artifact already exists".into());
    }
    storage::write_json_atomic(&path, &json!({"summary":summary,"contacts":contacts}))
        .map_err(|e| format!("Save Ground Detector: {e:?}"))?;
    summary["sha256"] = json!(format!(
        "{:x}",
        Sha256::digest(fs::read(&path).map_err(|e| e.to_string())?)
    ));
    Ok(summary)
}

pub(crate) fn prepare_touch_detection(
    state: &AppState,
    req: &TouchRequest,
    stabilized: &StabilizedReplay,
    scope: &PrimaryRallyScope,
    cancel: &AtomicBool,
) -> Result<PrimaryTouchDetection, String> {
    let plan = preparation::plan_for_run(
        state,
        &req.project_id,
        &req.video,
        "native_cpu_replays",
        &req.replay_id,
    )
    .map_err(|e| format!("{e:?}"))?;
    let path = plan
        .ready_path()
        .ok_or("Prepared video unavailable for Ground Detector")?;
    let dim = volley_media::probe(path)?;
    let dimensions = (dim.width, dim.height);
    let court = video_edits::court_snapshot(state, &req.project_id, &req.video)
        .map_err(|e| format!("{e:?}"))?;
    let replay_folder = stabilized
        .project_root
        .join("native_cpu_replays")
        .join(&req.replay_id);
    let replay: Value = serde_json::from_slice(
        &fs::read(safe_file(
            &replay_folder,
            "manifest.json",
            MAX_MANIFEST_BYTES,
        )?)
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mask_folder = stabilized
        .project_root
        .join("native_gpu_runs")
        .join(&scope.ball_job_id);
    let bytes = fs::read(safe_file(
        &mask_folder,
        "rally_mask.json",
        PRIMARY_TOUCH_BREAKS_MAX_BYTES,
    )?)
    .map_err(|e| e.to_string())?;
    if scope.sha256 != format!("{:x}", Sha256::digest(&bytes))
        || replay["player_frame_scope_sha256"] != scope.sha256
        || replay["player_frame_scope_ball_job_id"] != scope.ball_job_id
        || replay["sha256"] != stabilized.source_sha256
    {
        return Err("Ground Detector source lineage mismatch".into());
    }
    let mask: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let saved_breaks = read_primary_touch_breaks(
        &stabilized.project_root,
        &replay,
        dim.width,
        dim.height,
        stabilized.frames,
    )?;
    let primary_breaks_reused = saved_breaks.is_some();
    let mut breaks = match saved_breaks {
        Some(breaks) => breaks,
        None => trajectory_breaks(&stabilized.motion, dimensions, cancel)?,
    };
    breaks.retain(|impact| {
        impact.frame >= 0
            && impact.frame < stabilized.frames as i64
            && impact
                .before_frames
                .iter()
                .chain(&impact.after_frames)
                .all(|&f| f >= 0 && f < stabilized.frames as i64)
    });
    let primary = &mask["mask"]["ground_detector"];
    let revision = json!(court.as_ref().map(|c| c.revision.as_str()));
    let reusable = mask["mask"]["ball_stabilizer"]["algorithm"]
        == volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION
        && primary["version"] == volley_core::motion::GROUND_DETECTOR_VERSION
        && primary["court_revision"] == revision
        && primary["source_trajectory_sha256"]
            == mask["mask"]["ball_stabilizer"]["trajectory_sha256"]
        && primary["width"] == dim.width
        && primary["height"] == dim.height
        && primary["frames"] == scope.source_frames
        && primary["calibrated"] == court.as_ref().is_some_and(|c| c.transform.is_some());
    let intervals =
        align_primary_intervals(&scope.intervals, scope.source_frames, stabilized.frames)?;
    let mut contacts = if reusable {
        let bytes = fs::read(safe_file(
            &mask_folder,
            FILE,
            PRIMARY_TOUCH_BREAKS_MAX_BYTES,
        )?)
        .map_err(|e| e.to_string())?;
        if primary["sha256"] != format!("{:x}", Sha256::digest(&bytes)) {
            return Err("Ground Detector SHA mismatch".into());
        }
        let mut data: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let mut expected = primary.clone();
        expected
            .as_object_mut()
            .ok_or("Invalid Ground Detector summary")?
            .remove("sha256");
        if data["summary"] != expected {
            return Err("Ground Detector summary mismatch".into());
        }
        let rows: Vec<volley_core::motion::GroundContact> =
            serde_json::from_value(data["contacts"].take()).map_err(|e| e.to_string())?;
        if rows.len() > LIMIT
            || primary["candidate_count"] != rows.len()
            || rows.windows(2).any(|w| w[0].frame >= w[1].frame)
            || rows.iter().any(|c| {
                c.frame >= scope.source_frames
                    || !matches!(
                        c.landing.as_str(),
                        "line_uncertain" | "out" | "near_half" | "far_half"
                    )
                    || c.position
                        .iter()
                        .any(|v| !v.is_finite() || !(-0.5..=1.5).contains(v))
            })
        {
            return Err("Invalid Ground Detector candidates".into());
        }
        rows
    } else {
        candidates(
            &stabilized.motion,
            &breaks,
            &intervals,
            court.as_ref().and_then(|c| c.transform.as_ref()),
            dim.width,
            dim.height,
            cancel,
        )?
    };
    contacts.retain(|c| c.frame < stabilized.frames);
    let folder = stabilized
        .project_root
        .join("native_cpu_ball_stabilizer")
        .join(&stabilized.artifact_id);
    let mut summary = publish(
        &folder,
        &contacts,
        &stabilized.trajectory_sha256,
        dimensions,
        stabilized.frames,
        court.as_ref(),
    )?;
    summary["primary_reused"] = json!(reusable);
    summary["primary_breaks_reused"] = json!(primary_breaks_reused);
    contacts.retain(|contact| {
        let index = intervals.partition_point(|&(_, end)| end < contact.frame);
        let Some(&(start, end)) = intervals.get(index) else {
            return false;
        };
        if contact.frame < start {
            return false;
        }
        end.saturating_sub(contact.frame) <= 45
            || (matches!(contact.landing.as_str(), "line_uncertain" | "out")
                && breaks
                    .binary_search_by_key(&(contact.frame as i64), |impact| impact.frame)
                    .is_ok_and(|i| breaks[i].contact_observed))
    });
    summary["scoped_candidate_count"] = json!(contacts.len());
    breaks.retain(|impact| {
        let frame = impact.frame as u64;
        let index = intervals.partition_point(|&(_, end)| end < frame);
        intervals
            .get(index)
            .is_some_and(|&(start, _)| frame >= start)
    });
    Ok(PrimaryTouchDetection {
        breaks,
        contacts,
        summary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stabilized_trajectory_detects_floor_before_player_analysis() {
        let court = CourtTransform::new([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]).unwrap();
        let motion = volley_core::ball_interpolation::StabilizedBallMotion {
            points: [46., 55., 65., 75., 103., 97., 91., 85., 79.]
                .into_iter()
                .map(|y| Some([0.1, y / 200.]))
                .collect(),
            interpolated: vec![false; 9],
            rejected_frames: vec![],
            confidence_frames: 9,
            observed_frames: 9,
        };
        let impact = volley_core::motion::TrajectoryBreak {
            frame: 4,
            position: [100., 103.],
            motion: Default::default(),
            fit_error: 0.5,
            before_frames: vec![0, 1, 2, 3],
            after_frames: vec![5, 6, 7, 8],
            contact_observed: true,
        };
        let cancel = AtomicBool::new(false);
        let events = candidates(
            &motion,
            std::slice::from_ref(&impact),
            &[(0, 8)],
            Some(&court),
            1000,
            200,
            &cancel,
        )
        .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].landing, "line_uncertain");
        assert_eq!(events[0].frame, 4);
        assert!(candidates(
            &motion,
            std::slice::from_ref(&impact),
            &[(0, 8)],
            None,
            1000,
            200,
            &cancel
        )
        .unwrap()
        .is_empty());
        cancel.store(true, Ordering::Release);
        assert!(candidates(
            &motion,
            &[impact],
            &[(0, 8)],
            Some(&court),
            1000,
            200,
            &cancel
        )
        .is_err());
    }

    #[test]
    fn early_in_court_player_like_rebound_is_not_a_ground_boundary() {
        let court = CourtTransform::new([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]).unwrap();
        let mut motion = volley_core::ball_interpolation::StabilizedBallMotion {
            points: [46., 55., 65., 75., 103., 97., 91., 85., 79.]
                .into_iter()
                .map(|y| Some([0.3, y / 200.]))
                .collect(),
            interpolated: vec![false; 9],
            rejected_frames: vec![],
            confidence_frames: 9,
            observed_frames: 9,
        };
        motion.points.resize(100, None);
        motion.interpolated.resize(100, false);
        let impact = volley_core::motion::TrajectoryBreak {
            frame: 4,
            position: [300., 103.],
            motion: Default::default(),
            fit_error: 0.5,
            before_frames: vec![0, 1, 2, 3],
            after_frames: vec![5, 6, 7, 8],
            contact_observed: true,
        };
        let cancel = AtomicBool::new(false);
        let events = candidates(
            &motion,
            std::slice::from_ref(&impact),
            &[(0, 99)],
            Some(&court),
            1000,
            200,
            &cancel,
        )
        .unwrap();
        assert!(
            events.is_empty(),
            "early in-court rebound must not end a rally: {events:?}"
        );
        let near_end = candidates(
            &motion,
            std::slice::from_ref(&impact),
            &[(0, 49)],
            Some(&court),
            1000,
            200,
            &cancel,
        )
        .unwrap();
        assert_eq!(near_end.len(), 1);
        assert_eq!(near_end[0].frame, 4);
        assert_eq!(near_end[0].landing, "near_half");
        let just_too_early = candidates(
            &motion,
            std::slice::from_ref(&impact),
            &[(0, 50)],
            Some(&court),
            1000,
            200,
            &cancel,
        )
        .unwrap();
        assert!(just_too_early.is_empty());
        let outside_scope = candidates(
            &motion,
            std::slice::from_ref(&impact),
            &[(5, 99)],
            Some(&court),
            1000,
            200,
            &cancel,
        )
        .unwrap();
        assert!(outside_scope.is_empty());
        let mut line_impact = impact;
        line_impact.position[0] = 100.;
        for point in motion.points.iter_mut().flatten() {
            point[0] = 0.1;
        }
        let line = candidates(
            &motion,
            std::slice::from_ref(&line_impact),
            &[(0, 99)],
            Some(&court),
            1000,
            200,
            &cancel,
        )
        .unwrap();
        assert_eq!(line[0].landing, "line_uncertain");
        line_impact.contact_observed = false;
        let unobserved_line = candidates(
            &motion,
            &[line_impact],
            &[(0, 99)],
            Some(&court),
            1000,
            200,
            &cancel,
        )
        .unwrap();
        assert!(unobserved_line.is_empty());
    }

    #[test]
    fn rolling_ball_ends_rally_even_without_a_touch_break() {
        let court = CourtTransform::new([[0.06, 1.02], [0.34, 0.294], [0.60, 0.288], [0.89, 1.03]])
            .unwrap();
        let mut motion = volley_core::ball_interpolation::StabilizedBallMotion {
            points: vec![None; 240],
            interpolated: vec![false; 240],
            rejected_frames: vec![],
            confidence_frames: 10,
            observed_frames: 10,
        };
        for (frame, x, y) in [
            (40, 0.35, 0.25),
            (41, 0.35, 0.26),
            (42, 0.35, 0.27),
            (43, 0.35, 0.28),
            (44, 0.35, 0.295),
            (56, 0.358, 0.267),
            (57, 0.358, 0.267),
            (58, 0.360, 0.264),
            (63, 0.362, 0.243),
            (64, 0.368, 0.240),
        ] {
            motion.points[frame] = Some([x, y]);
        }
        let cancel = AtomicBool::new(false);
        let contacts =
            candidates(&motion, &[], &[(0, 239)], Some(&court), 3840, 2160, &cancel).unwrap();
        assert_eq!(contacts.len(), 1);
        assert!((44..=46).contains(&contacts[0].frame));
        assert!(contacts[0].position[1] > 0.29);
        assert!(
            candidates(&motion, &[], &[(0, 239)], None, 3840, 2160, &cancel)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn weak_terminal_rebound_requires_measured_brackets() {
        let court =
            CourtTransform::new([[0.06, 1.02], [0.34, 0.29], [0.60, 0.29], [0.89, 1.03]]).unwrap();
        let mut motion = volley_core::ball_interpolation::StabilizedBallMotion {
            points: vec![None; 60],
            interpolated: vec![false; 60],
            rejected_frames: vec![],
            confidence_frames: 20,
            observed_frames: 16,
        };
        let coords = [
            (0, 0.416, 0.350),
            (2, 0.421, 0.361),
            (3, 0.422, 0.365),
            (4, 0.427, 0.369),
            (5, 0.428, 0.375),
            (6, 0.430, 0.379),
            (7, 0.432, 0.382),
            (8, 0.433, 0.389),
            (9, 0.435, 0.399),
            (10, 0.437, 0.403),
            (11, 0.439, 0.420),
            (12, 0.441, 0.424),
            (13, 0.443, 0.426),
            (14, 0.445, 0.428),
            (15, 0.447, 0.428),
            (16, 0.449, 0.426),
            (17, 0.451, 0.424),
            (18, 0.453, 0.420),
            (19, 0.457, 0.417),
            (50, 0.5, 0.3),
        ];
        for (frame, x, y) in coords {
            motion.points[frame] = Some([x, y]);
        }
        motion.interpolated[13..=16].fill(true);
        let cancel = AtomicBool::new(false);
        let run = |m: &volley_core::ball_interpolation::StabilizedBallMotion| {
            candidates(m, &[], &[(0, 35)], Some(&court), 3840, 2160, &cancel).unwrap()
        };
        let hits = run(&motion);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].frame, 15);
        motion.interpolated[17..=19].fill(true);
        assert!(run(&motion).is_empty());
        motion.interpolated[17..=19].fill(false);
        motion.interpolated[10..=12].fill(true);
        assert!(run(&motion).is_empty());
    }
    #[test]
    fn short_roll_uses_interpolated_descent_but_requires_measured_tail() {
        let court = CourtTransform::new([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]).unwrap();
        let mut motion = volley_core::ball_interpolation::StabilizedBallMotion {
            points: vec![None; 300],
            interpolated: vec![false; 300],
            rejected_frames: vec![],
            confidence_frames: 8,
            observed_frames: 7,
        };
        for frame in 100..=104 {
            motion.points[frame] = Some([0.51, 0.52 + (frame - 100) as f64 * 0.03]);
        }
        motion.interpolated[102] = true;
        motion.points[105] = Some([0.508, 0.644]);
        motion.points[107] = Some([0.505, 0.644]);
        motion.points[170] = Some([0.4, 0.3]);
        let cancel = AtomicBool::new(false);
        let run = |motion: &volley_core::ball_interpolation::StabilizedBallMotion| {
            candidates(motion, &[], &[(0, 239)], Some(&court), 1000, 1000, &cancel).unwrap()
        };
        let contacts = run(&motion);
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].frame, 104);
        motion.interpolated[105] = true;
        motion.interpolated[107] = true;
        assert!(run(&motion).is_empty());
        motion.interpolated[105] = false;
        motion.interpolated[107] = false;
        motion.points[110] = Some([0.505, 0.58]);
        assert!(run(&motion).is_empty());
    }

    #[test]
    fn visible_pickup_after_sand_settle_keeps_original_ground_frame() {
        let court = CourtTransform::new([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]).unwrap();
        let mut motion = volley_core::ball_interpolation::StabilizedBallMotion {
            points: vec![None; 100],
            interpolated: vec![false; 100],
            rejected_frames: vec![],
            confidence_frames: 41,
            observed_frames: 37,
        };
        for frame in 0..=4 {
            motion.points[frame] = Some([0.51, 0.52 + frame as f64 * 0.03]);
        }
        for frame in 5..=24 {
            motion.points[frame] = Some([0.51 + (frame - 4) as f64 * 0.0003, 0.64]);
        }
        for frame in 25..=40 {
            motion.points[frame] = Some([0.516, 0.64 - (frame - 24) as f64 * 0.02]);
        }
        motion.interpolated[9..13].fill(true);
        let cancel = AtomicBool::new(false);
        let run = |motion: &volley_core::ball_interpolation::StabilizedBallMotion| {
            candidates(motion, &[], &[(0, 99)], Some(&court), 1000, 1000, &cancel).unwrap()
        };
        let contacts = run(&motion);
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].frame, 4);
        motion.interpolated[5..=18].fill(true);
        assert!(run(&motion).is_empty());
    }

    #[test]
    fn stage_persists_separate_floor_evidence_and_checksum() {
        let folder = std::env::temp_dir().join(format!("ground-stage-{}", Uuid::new_v4()));
        fs::create_dir(&folder).unwrap();
        let contacts = vec![
            volley_core::motion::GroundContact {
                frame: 1412,
                position: [0.28, 0.64],
                landing: "line_uncertain".into(),
            },
            volley_core::motion::GroundContact {
                frame: 1433,
                position: [0.21, 0.67],
                landing: "out".into(),
            },
        ];
        let summary = publish(
            &folder,
            &contacts,
            &"a".repeat(64),
            (1920, 1080),
            1737,
            None,
        )
        .unwrap();
        let bytes = fs::read(folder.join(FILE)).unwrap();
        let artifact: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(summary["sha256"], format!("{:x}", Sha256::digest(&bytes)));
        assert_eq!(summary["source_trajectory_sha256"], "a".repeat(64));
        assert_eq!(summary["candidate_count"], 2);
        let restored: Vec<volley_core::motion::GroundContact> =
            serde_json::from_value(artifact["contacts"].clone()).unwrap();
        assert_eq!(restored, contacts);
        assert!(publish(
            &folder,
            &contacts,
            &"b".repeat(64),
            (1920, 1080),
            1737,
            None
        )
        .is_err());
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn distant_out_rebound_uses_stabilized_context_outside_fit_anchors() {
        let court =
            CourtTransform::new([[0.1, 0.9], [0.35, 0.28], [0.6, 0.28], [0.9, 0.9]]).unwrap();
        let mut motion = volley_core::ball_interpolation::StabilizedBallMotion {
            points: vec![None; 100],
            interpolated: vec![false; 100],
            rejected_frames: vec![],
            confidence_frames: 60,
            observed_frames: 60,
        };
        for frame in 2..=64 {
            let y = match frame {
                2..=14 => 0.14 + (frame - 2) as f64 * 0.012,
                15..=20 => 0.284 + (frame - 14) as f64 * 0.0015,
                21 => 0.285,
                22..=35 => 0.278 - (frame - 22) as f64 * 0.0045,
                36..=56 => 0.2195 + (frame - 35) as f64 * 0.0034,
                _ => 0.279,
            };
            motion.points[frame] = Some([0.69 - frame.saturating_sub(20) as f64 * 0.0005, y]);
        }
        motion.interpolated[20] = true;
        motion.interpolated[21] = true;
        let impact = volley_core::motion::TrajectoryBreak {
            frame: 20,
            position: [690.0, 293.0],
            motion: Default::default(),
            fit_error: 0.5,
            before_frames: vec![16, 17, 18, 19],
            after_frames: vec![22, 23, 24, 25],
            contact_observed: false,
        };
        let cancel = AtomicBool::new(false);
        let contacts = candidates(
            &motion,
            &[impact],
            &[(0, 99)],
            Some(&court),
            1000,
            1000,
            &cancel,
        )
        .unwrap();
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].frame, 20);
        assert_eq!(contacts[0].landing, "out");
    }
}
