use crate::{
    native_rally_evidence::{generate, inspect_project, Choice, ValidatedRun},
    projects,
    storage::{self, safe_file, ApiError, ApiResult},
    touch_annotation, AppState,
};
use axum::{
    extract::{Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, sync::Arc};
use volley_core::rally_reconstruction::{preliminary_rally_intervals, RallySettings};

const MAX_MANIFEST: u64 = 16 * 1024;
const MAX_TRAJECTORY_DATA: u64 = 16 * 1024 * 1024;
const MAX_COMPARE_INTERVALS: usize = 4096;
const DEFAULT_MIN_TRAJECTORY_SECONDS: f64 = 1.5;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompareQuery {
    project_id: String,
    rally_run_id: String,
    min_impacts: Option<u64>,
    context_frames: Option<u64>,
    min_duration_frames: Option<u64>,
    max_touch_gap_frames: Option<u64>,
    simulate: Option<bool>,
    calibrate: Option<bool>,
    mode: Option<String>,
    min_trajectory_seconds: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TouchRallyFilter {
    mode: String,
    min_trajectory_seconds: Option<f64>,
    min_impacts: Option<u64>,
    context_frames: Option<u64>,
    min_duration_frames: Option<u64>,
    max_touch_gap_frames: Option<u64>,
    merge_gap_seconds: f64,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
struct Interval {
    start: u64,
    end: u64,
}

impl Interval {
    fn new(start: u64, end: u64, frames: u64) -> Result<Self, ApiError> {
        if start >= end || end >= frames {
            return Err(storage::bad_request("Invalid training rally interval"));
        }
        Ok(Self { start, end })
    }
}

fn intervals(value: &Value, frames: u64) -> Result<Vec<Interval>, ApiError> {
    let rows = value
        .as_array()
        .ok_or_else(|| storage::bad_request("Invalid saved rally intervals"))?;
    if rows.len() > MAX_COMPARE_INTERVALS {
        return Err(storage::bad_request("Too many rally intervals to compare"));
    }
    let mut output = Vec::with_capacity(rows.len());
    for row in rows {
        let start = row[0]
            .as_u64()
            .ok_or_else(|| storage::bad_request("Invalid saved rally start"))?;
        let end = row[1]
            .as_u64()
            .ok_or_else(|| storage::bad_request("Invalid saved rally end"))?;
        output.push(Interval::new(start, end, frames)?);
    }
    output.sort_by_key(|v| (v.start, v.end));
    Ok(output)
}

fn cpu_intervals(run: &ValidatedRun) -> Result<(Vec<Interval>, Vec<u64>), ApiError> {
    let timeline = generate(run, Choice::Timeline)
        .map_err(storage::bad_request)?
        .value;
    let rows = timeline["rallies"]
        .as_array()
        .ok_or_else(|| storage::bad_request("CPU rally timeline is missing"))?;
    if rows.len() > MAX_COMPARE_INTERVALS {
        return Err(storage::bad_request("Too many CPU rallies to compare"));
    }
    let mut result = rows
        .iter()
        .map(|r| {
            Interval::new(
                r["start_frame"]
                    .as_u64()
                    .ok_or_else(|| storage::bad_request("CPU rally has no start"))?,
                r["end_frame"]
                    .as_u64()
                    .ok_or_else(|| storage::bad_request("CPU rally has no end"))?,
                run.frames,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    result.sort_by_key(|v| (v.start, v.end));
    let mut touches = timeline["touch_annotations"]
        .as_array()
        .ok_or_else(|| storage::bad_request("CPU touch timeline is missing"))?
        .iter()
        .filter(|touch| touch["touch_attributed"] == true || touch["contact_verified"] == true)
        .filter_map(|touch| touch["frame_index"].as_u64())
        .filter(|&frame| frame < run.frames)
        .collect::<Vec<_>>();
    touches.sort_unstable();
    touches.dedup();
    Ok((result, touches))
}

fn overlap(a: Interval, b: Interval) -> f64 {
    if a.end < b.start || b.end < a.start {
        return 0.0;
    }
    let intersection = a.end.min(b.end) - a.start.max(b.start) + 1;
    let union = a.end - a.start + b.end - b.start + 2 - intersection;
    intersection as f64 / union as f64
}

type IntervalMatching = (Value, Vec<Option<(usize, f64)>>, Vec<bool>);

fn match_intervals(baseline: &[Interval], proposed: &[Interval]) -> IntervalMatching {
    let mut pairs = vec![None; baseline.len()];
    let mut used = vec![false; proposed.len()];
    let mut boundary_sum = 0u64;
    for (i, &a) in baseline.iter().enumerate() {
        let mut best: Option<(usize, f64, u64)> = None;
        for (j, &b) in proposed.iter().enumerate() {
            if used[j] {
                continue;
            }
            let iou = overlap(a, b);
            if iou < 0.5 {
                continue;
            }
            let displacement = a.start.abs_diff(b.start) + a.end.abs_diff(b.end);
            if best.is_none_or(|(_, score, delta)| {
                iou > score || (iou == score && displacement < delta)
            }) {
                best = Some((j, iou, displacement));
            }
        }
        if let Some((j, iou, displacement)) = best {
            used[j] = true;
            pairs[i] = Some((j, iou));
            boundary_sum += displacement;
        }
    }
    let matched = used.iter().filter(|&&v| v).count();
    let extra = proposed.len() - matched;
    let missed = baseline.len() - matched;
    let precision = if proposed.is_empty() {
        if baseline.is_empty() {
            1.0
        } else {
            0.0
        }
    } else {
        matched as f64 / proposed.len() as f64
    };
    let recall = if baseline.is_empty() {
        if proposed.is_empty() {
            1.0
        } else {
            0.0
        }
    } else {
        matched as f64 / baseline.len() as f64
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    let mean_boundary_error_ms = if matched == 0 {
        Value::Null
    } else {
        json!(boundary_sum as f64 * 1000.0 / (60.0 * matched as f64))
    };
    (
        json!({"matched":matched,"extra":extra,"missed":missed,
            "precision":precision,"recall":recall,"f1":f1,
            "mean_boundary_error_ms":mean_boundary_error_ms,
            "iou_threshold":0.5}),
        pairs,
        used,
    )
}

fn differences(
    cpu: &[Interval],
    trajectory: &[Interval],
    pairs: &[Option<(usize, f64)>],
    matched: &[bool],
    reviewed: &[Interval],
) -> Vec<Value> {
    let mut result = Vec::with_capacity(cpu.len() + trajectory.len());
    for (index, &interval) in cpu.iter().enumerate() {
        match pairs[index] {
            Some((other, iou)) => {
                let candidate = trajectory[other];
                result.push(json!({"kind":if iou >= 0.85 {"agreed"} else {"shifted"},
                    "cpu":interval,"trajectory":candidate,
                    "iou":iou,"seek_frame":interval.start.min(candidate.start)}));
            }
            None => result.push(json!({"kind":"cpu_only","cpu":interval,
                "trajectory":null,"iou":null,"seek_frame":interval.start})),
        }
    }
    for (index, &interval) in trajectory.iter().enumerate() {
        if !matched[index] {
            result.push(json!({"kind":"trajectory_only","cpu":null,
                "trajectory":interval,"iou":null,"seek_frame":interval.start}));
        }
    }
    for &interval in reviewed {
        if !cpu.iter().any(|&c| overlap(c, interval) >= 0.5)
            && !trajectory.iter().any(|&c| overlap(c, interval) >= 0.5)
        {
            result.push(json!({"kind":"both_missed","cpu":null,"trajectory":null,
                "reference":interval,"iou":null,"seek_frame":interval.start}));
        }
    }
    result.sort_by_key(|row| row["seek_frame"].as_u64().unwrap_or(u64::MAX));
    result
}

type SavedImpacts = (Vec<u64>, RallySettings, &'static str);

fn saved_impacts(
    state: &AppState,
    run: &ValidatedRun,
    simulate: bool,
) -> Result<Option<SavedImpacts>, ApiError> {
    let root = projects::project_root(state)
        .join(&run.project_id)
        .join("native_cpu_touch_runs")
        .join(&run.touch_id);
    let manifest_path =
        safe_file(&root, "manifest.json", MAX_MANIFEST).map_err(storage::bad_request)?;
    let manifest: Value = serde_json::from_slice(
        &fs::read(manifest_path)
            .map_err(|e| storage::internal(format!("Read touch manifest: {e}")))?,
    )
    .map_err(|_| storage::bad_request("Invalid saved touch manifest"))?;
    if manifest["touch_id"] != run.touch_id
        || manifest["source_replay_id"] != run.replay_id
        || manifest["source_replay_sha256"] != run.replay_sha
        || manifest["sha256"] != run.touch_sha
    {
        return Err(storage::conflict(
            "Touch run lineage changed during comparison",
        ));
    }
    if manifest.get("preliminary_rally_file").is_none() {
        if !simulate {
            return Ok(None);
        }
        let starts: Vec<u64> = match manifest.get("leading_ball_rally_starts") {
            Some(value) => serde_json::from_value(value.clone())
                .map_err(|_| storage::bad_request("Invalid saved leading-ball boundaries"))?,
            None => vec![0],
        };
        let settings = RallySettings {
            min_ball_score: run.min_ball_score,
            ..RallySettings::default()
        };
        settings.validate().map_err(storage::bad_request)?;
        let replay_root = projects::project_root(state)
            .join(&run.project_id)
            .join("native_cpu_replays")
            .join(&run.replay_id);
        let replay = safe_file(
            &replay_root,
            "tracked_frames.jsonl",
            64 * 1024 * 1024 * 1024,
        )
        .map_err(storage::bad_request)?;
        let mask = crate::cpu_retrack::ball_track::read(
            &replay,
            &run.replay_sha,
            run.frames,
            run.width,
            run.height,
            settings.min_ball_score,
            &starts,
        )
        .map_err(storage::bad_request)?;
        let computed = crate::cpu_retrack::ball_track::preliminary_rally_scope(
            &replay,
            &run.replay_sha,
            run.frames,
            (run.width, run.height),
            settings.min_ball_score,
            &mask.accepted,
            settings,
        )
        .map_err(storage::bad_request)?;
        return Ok(Some((
            computed.impacts,
            settings,
            "recalculated_saved_cpu_replay",
        )));
    }
    if manifest["preliminary_rally_file"] != "preliminary_rallies.json" {
        return Err(storage::bad_request("Unexpected preliminary rally file"));
    }
    let expected = manifest["preliminary_rally_sha256"]
        .as_str()
        .filter(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()))
        .ok_or_else(|| storage::bad_request("Preliminary rally SHA missing"))?;
    let path = safe_file(&root, "preliminary_rallies.json", MAX_TRAJECTORY_DATA)
        .map_err(storage::bad_request)?;
    let bytes =
        fs::read(path).map_err(|e| storage::internal(format!("Read saved trajectory: {e}")))?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err(storage::conflict("Saved trajectory checksum changed"));
    }
    let data: Value = serde_json::from_slice(&bytes)
        .map_err(|_| storage::bad_request("Invalid trajectory rally data"))?;
    if data["schema"] != "volley-native-preliminary-ball-rallies/1"
        || data["source_replay_id"] != run.replay_id
        || data["source_replay_sha256"] != run.replay_sha
        || data["frames"].as_u64() != Some(run.frames)
    {
        return Err(storage::conflict(
            "Saved trajectory belongs to a different CPU source",
        ));
    }
    let impacts: Vec<u64> = serde_json::from_value(data["trajectory_impacts"].clone())
        .map_err(|_| storage::bad_request("Invalid saved ball impacts"))?;
    if impacts.iter().any(|&f| f >= run.frames)
        || impacts.windows(2).any(|pair| pair[0] >= pair[1])
        || impacts.len() > run.frames as usize
    {
        return Err(storage::bad_request("Trajectory impacts are inconsistent"));
    }
    let settings: RallySettings = serde_json::from_value(data["settings"].clone())
        .map_err(|_| storage::bad_request("Invalid saved trajectory settings"))?;
    settings.validate().map_err(storage::bad_request)?;
    let saved = intervals(&data["rally_intervals"], run.frames)?;
    let recalculated = preliminary_rally_intervals(&impacts, run.frames, settings)
        .into_iter()
        .map(|(start, end)| Interval::new(start, end, run.frames))
        .collect::<Result<Vec<_>, _>>()?;
    if saved != recalculated {
        return Err(storage::conflict(
            "Saved trajectory intervals do not match the detector",
        ));
    }
    Ok(Some((impacts, settings, "saved_cpu_trajectory_impacts")))
}

fn impacts_within_rallies(impacts: &[u64], rallies: &[Interval]) -> Vec<u64> {
    let mut index = 0usize;
    impacts
        .iter()
        .copied()
        .filter(|&frame| {
            while index < rallies.len() && rallies[index].end < frame {
                index += 1;
            }
            rallies
                .get(index)
                .is_some_and(|rally| rally.start <= frame && frame <= rally.end)
        })
        .collect()
}

fn fit_trajectory_settings(
    impacts: &[u64],
    frames: u64,
    settings: RallySettings,
    reviewed: &[Interval],
) -> Option<(RallySettings, Value, Value)> {
    if reviewed.is_empty()
        || impacts.is_empty()
        || reviewed.len() > 256
        || impacts.len() > 8192
        || reviewed.len().saturating_mul(impacts.len()) > 250_000
        || frames < 2
    {
        return None;
    }
    let evaluate = |candidate: RallySettings| {
        let predicted = preliminary_rally_intervals(impacts, frames, candidate)
            .into_iter()
            .map(|(start, end)| Interval { start, end })
            .collect::<Vec<_>>();
        match_intervals(reviewed, &predicted).0
    };
    let baseline = evaluate(settings);
    let mut best = settings;
    let mut best_metrics = baseline.clone();
    let error_count = |metric: &Value| {
        metric["extra"]
            .as_u64()
            .unwrap_or(u64::MAX)
            .saturating_add(metric["missed"].as_u64().unwrap_or(u64::MAX))
    };
    let boundary_error = |metric: &Value| {
        metric["mean_boundary_error_ms"]
            .as_f64()
            .unwrap_or(f64::INFINITY)
    };
    for min_impacts in [1, 2, 3, 4, 5] {
        for context in [0, 15, 30, 45, 60, 90] {
            for duration in [15, 30, 45, 60, 90, 120] {
                for gap in [90, 105, 120, 180, 240, 360] {
                    let candidate = RallySettings {
                        min_touch_candidates: min_impacts,
                        suggested_context_frames: context,
                        min_rally_frames: duration,
                        max_without_touch_frames: gap,
                        ..settings
                    };
                    let score = evaluate(candidate);
                    let f1 = score["f1"].as_f64().unwrap_or(0.0);
                    let current_f1 = best_metrics["f1"].as_f64().unwrap_or(0.0);
                    let tie = (f1 - current_f1).abs() < 1e-9;
                    if f1 > current_f1 + 1e-9
                        || (tie
                            && (error_count(&score) < error_count(&best_metrics)
                                || (error_count(&score) == error_count(&best_metrics)
                                    && boundary_error(&score) + 1e-6
                                        < boundary_error(&best_metrics))))
                    {
                        best = candidate;
                        best_metrics = score;
                    }
                }
            }
        }
    }
    Some((best, baseline, best_metrics))
}

fn compare(state: &AppState, run: &ValidatedRun, query: &CompareQuery) -> Result<Value, ApiError> {
    let (cpu, cpu_touches) = cpu_intervals(run)?;
    let truth = touch_annotation::comparison_truth(state, run)?;
    let reference = truth
        .as_ref()
        .map(|(items, _, _)| {
            let mut sorted = items
                .iter()
                .map(|&(start, end)| Interval::new(start, end, run.frames))
                .collect::<Result<Vec<_>, _>>()?;
            sorted.sort_by_key(|v| (v.start, v.end));
            Ok(sorted)
        })
        .transpose()?;
    let reviewed = truth.as_ref().is_some_and(|(_, _, reviewed)| *reviewed);
    let calibration_requested = query.calibrate.unwrap_or(false);
    let mode = query.mode.as_deref().unwrap_or("impacts");
    if !matches!(mode, "impacts" | "continuous") {
        return Err(storage::bad_request("Unknown training rally algorithm"));
    }
    if (mode == "continuous"
        && (calibration_requested
            || query.min_impacts.is_some()
            || query.context_frames.is_some()
            || query.min_duration_frames.is_some()
            || query.max_touch_gap_frames.is_some()))
        || (mode == "impacts" && query.min_trajectory_seconds.is_some())
    {
        return Err(storage::bad_request(
            "Training rally parameters do not match selected algorithm",
        ));
    }
    let saved = if mode == "impacts" {
        saved_impacts(
            state,
            run,
            query.simulate.unwrap_or(false) || calibration_requested,
        )?
    } else {
        None
    };
    let (trajectory, impacts, parameters, status, source, calibration) = if mode == "continuous" {
        let seconds = query
            .min_trajectory_seconds
            .unwrap_or(DEFAULT_MIN_TRAJECTORY_SECONDS);
        if !seconds.is_finite() || !(0.1..=120.0).contains(&seconds) {
            return Err(storage::bad_request(
                "Continuous ball minimum must be 0.1–120 seconds",
            ));
        }
        let replay_root = projects::project_root(state)
            .join(&run.project_id)
            .join("native_cpu_replays")
            .join(&run.replay_id);
        let replay = safe_file(
            &replay_root,
            "tracked_frames.jsonl",
            64 * 1024 * 1024 * 1024,
        )
        .map_err(storage::bad_request)?;
        let segments = crate::cpu_retrack::ball_track::continuous_ball_intervals(
            &replay,
            &run.replay_sha,
            run.frames,
            run.min_ball_score,
            seconds,
        )
        .map_err(storage::bad_request)?
        .into_iter()
        .map(|(start, end)| Interval::new(start, end, run.frames))
        .collect::<Result<Vec<_>, _>>()?;
        (
            segments,
            Vec::new(),
            json!({"min_trajectory_seconds":seconds,"min_ball_score":run.min_ball_score}),
            "ready",
            "saved_cpu_replay_continuous_ball",
            None,
        )
    } else if let Some((impacts, mut settings, source)) = saved {
        if let Some(n) = query.min_impacts {
            settings.min_touch_candidates = n;
        }
        if let Some(n) = query.context_frames {
            settings.suggested_context_frames = n;
        }
        if let Some(n) = query.min_duration_frames {
            settings.min_rally_frames = n;
        }
        if let Some(n) = query.max_touch_gap_frames {
            settings.max_without_touch_frames = n;
        }
        settings.validate().map_err(storage::bad_request)?;
        let calibration = if calibration_requested {
            if !reviewed {
                return Err(storage::bad_request(
                    "Confirm the rally annotations before calibrating trajectory parameters",
                ));
            }
            let reviewed_intervals = reference.as_deref().unwrap_or(&[]);
            let (fitted, baseline, optimized) =
                    fit_trajectory_settings(&impacts, run.frames, settings, reviewed_intervals)
                        .ok_or_else(|| storage::bad_request(
                            "Calibration requires approved rallies and a bounded number of saved trajectory impacts",
                        ))?;
            settings = fitted;
            Some(json!({"method":"bounded_rally_parameter_sweep_v1",
                    "training_videos":1,"in_sample_only":true,
                    "baseline":baseline,"fitted":optimized,
                    "source_replay_sha256":run.replay_sha,
                    "published_cpu_unchanged":true}))
        } else {
            None
        };
        let intervals = preliminary_rally_intervals(&impacts, run.frames, settings)
            .into_iter()
            .map(|(start, end)| Interval::new(start, end, run.frames))
            .collect::<Result<Vec<_>, _>>()?;
        (
            intervals,
            impacts,
            json!({"min_impacts":settings.min_touch_candidates,
                "context_frames":settings.suggested_context_frames,
                "min_duration_frames":settings.min_rally_frames,
                "max_touch_gap_frames":settings.max_without_touch_frames}),
            "ready",
            source,
            calibration,
        )
    } else {
        if calibration_requested {
            return Err(storage::bad_request(
                "No saved ball trajectory is available for rally calibration",
            ));
        }
        (
            Vec::new(),
            Vec::new(),
            Value::Null,
            "unavailable",
            "missing_saved_trajectory",
            None,
        )
    };
    let simulated_touches = impacts_within_rallies(&impacts, &trajectory);
    let unassigned_touches = impacts
        .iter()
        .copied()
        .filter(|frame| simulated_touches.binary_search(frame).is_err())
        .collect::<Vec<_>>();
    let cpu_vs_simulated_touches = (status == "ready" && mode == "impacts")
        .then(|| touch_annotation::compare_touch_frames(&cpu_touches, &simulated_touches));
    let reference_touches = truth.as_ref().map(|(_, touches, _)| touches.as_slice());
    let touch_reference = if reviewed {
        reference_touches.map(|touches| {
            json!({
                "cpu":touch_annotation::compare_touch_frames(touches, &cpu_touches),
                "trajectory":if status == "ready" && mode == "impacts" {
                    Some(touch_annotation::compare_touch_frames(touches, &simulated_touches))
                } else {None}
            })
        })
    } else {
        None
    };
    let (cpu_trajectory, pairs, used) = match_intervals(&cpu, &trajectory);
    let (cpu_reference, trajectory_reference) = if reviewed {
        if let Some(ref marks) = reference {
            (
                Some(match_intervals(marks, &cpu).0),
                (status == "ready").then(|| match_intervals(marks, &trajectory).0),
            )
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };
    Ok(json!({"schema":"volley-rally-training-comparison/1",
        "project_id":run.project_id,"video":run.video,
        "rally_run_id":run.run_id,"source_replay_sha256":run.replay_sha,
        "source_touch_sha256":run.touch_sha,"source_rally_sha256":run.rally_sha,
        "trajectory_status":status,"mode":mode,
        "trajectory_detector_version":if mode == "continuous" {
            "coherent_vballnet_motion_v2"
        } else { "ball_down_up_v1" },
        "trajectory_impacts":impacts,"parameters":parameters,
        "simulated_touch_candidates":simulated_touches,
        "unassigned_touch_candidates":unassigned_touches,
        "cpu_touch_candidates":cpu_touches,
        "touch_metrics":{"cpu_trajectory":cpu_vs_simulated_touches,
                         "reference":touch_reference},
        "calibration":calibration,
        "simulation_touch_status":if mode == "continuous" {
            "not_calculated"
        } else { "unverified_ball_direction_change" },
        "cpu_rallies":cpu,"trajectory_rallies":trajectory,
        "reference_rallies":reference,"reference_reviewed":reviewed,
        "metrics":{"cpu_trajectory":if status=="ready" {Some(cpu_trajectory)} else {None},
            "cpu_reference":cpu_reference,"trajectory_reference":trajectory_reference},
        "differences":if status=="ready" {differences(&cpu,&trajectory,&pairs,&used,
            if reviewed {reference.as_deref().unwrap_or(&[])} else {&[]})}
            else {Vec::new()},
        "model_retrained":false,
        "source":source,
        "published_cpu_unchanged":true}))
}

fn merge_touch_scope(mut ranges: Vec<Interval>, gap_seconds: f64) -> Vec<Interval> {
    ranges.sort_by_key(|r| (r.start, r.end));
    let gap_frames = gap_seconds * 30.0;
    let mut merged: Vec<Interval> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(previous) = merged.last_mut() {
            if range.start <= previous.end.saturating_add(1)
                || ((range.start - previous.end - 1) as f64) < gap_frames
            {
                previous.end = previous.end.max(range.end);
                continue;
            }
        }
        merged.push(range);
    }
    merged
}

pub(crate) fn touch_rally_scope(
    state: &AppState,
    run: &ValidatedRun,
    filter: &TouchRallyFilter,
) -> Result<Vec<(u64, u64)>, ApiError> {
    if !filter.merge_gap_seconds.is_finite() || !(0.0..=60.0).contains(&filter.merge_gap_seconds) {
        return Err(storage::bad_request("Invalid training rally merge gap"));
    }
    let query = CompareQuery {
        project_id: run.project_id.clone(),
        rally_run_id: run.run_id.clone(),
        mode: Some(filter.mode.clone()),
        min_trajectory_seconds: filter.min_trajectory_seconds,
        min_impacts: filter.min_impacts,
        context_frames: filter.context_frames,
        min_duration_frames: filter.min_duration_frames,
        max_touch_gap_frames: filter.max_touch_gap_frames,
        simulate: Some(true),
        calibrate: Some(false),
    };
    let result = compare(state, run, &query)?;
    if result["trajectory_status"] != "ready" {
        return Err(storage::bad_request(
            "Ball rally detection is unavailable for touch training",
        ));
    }
    let rows = result["trajectory_rallies"]
        .as_array()
        .ok_or_else(|| storage::bad_request("Missing ball rally intervals"))?;
    if rows.len() > MAX_COMPARE_INTERVALS {
        return Err(storage::bad_request("Too many ball rally intervals"));
    }
    let ranges = rows
        .iter()
        .map(|row| {
            Interval::new(
                row["start"]
                    .as_u64()
                    .ok_or_else(|| storage::bad_request("Invalid ball rally start"))?,
                row["end"]
                    .as_u64()
                    .ok_or_else(|| storage::bad_request("Invalid ball rally end"))?,
                run.frames,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(merge_touch_scope(ranges, filter.merge_gap_seconds)
        .into_iter()
        .map(|row| (row.start, row.end))
        .collect())
}

pub async fn get(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CompareQuery>,
) -> ApiResult<Value> {
    let result = tokio::task::spawn_blocking(move || {
        let run = inspect_project(&state, &query.project_id, &query.rally_run_id)?;
        compare(&state, &run, &query)
    })
    .await
    .map_err(|e| storage::internal(format!("Training comparison task: {e}")))??;
    Ok(Json(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ball_touch_scope_uses_only_merged_ball_intervals() {
        let ranges = vec![
            Interval { start: 0, end: 50 },
            Interval {
                start: 100,
                end: 150,
            },
            Interval {
                start: 300,
                end: 350,
            },
        ];
        assert_eq!(
            merge_touch_scope(ranges.clone(), 4.0),
            vec![
                Interval { start: 0, end: 150 },
                Interval {
                    start: 300,
                    end: 350
                },
            ]
        );
        assert_eq!(merge_touch_scope(ranges, 0.0).len(), 3);
        assert!(merge_touch_scope(Vec::new(), 4.0).is_empty());
    }

    #[test]
    fn training_touch_scope_requires_explicit_rally_parameters() {
        let good = json!({
            "mode":"continuous",
            "min_trajectory_seconds":1.5,
            "merge_gap_seconds":4.0
        });
        assert!(serde_json::from_value::<TouchRallyFilter>(good).is_ok());
        assert!(serde_json::from_value::<TouchRallyFilter>(json!({
            "mode":"continuous","merge_gap_seconds":4.0,"unknown":true
        }))
        .is_err());
    }

    #[test]
    fn training_continuity_settings_are_distinct_from_impact_settings() {
        let parsed: CompareQuery = serde_json::from_value(json!({
            "project_id":"p","rally_run_id":"r","mode":"continuous",
            "min_trajectory_seconds":3.5,"simulate":true
        }))
        .unwrap();
        assert_eq!(parsed.mode.as_deref(), Some("continuous"));
        assert_eq!(parsed.min_trajectory_seconds, Some(3.5));
        assert!(parsed.min_impacts.is_none());
    }

    #[test]
    fn one_to_one_matching_distinguishes_extra_and_missing_rallies() {
        let reference = [
            Interval { start: 10, end: 50 },
            Interval {
                start: 100,
                end: 160,
            },
        ];
        let predicted = [
            Interval { start: 10, end: 50 },
            Interval { start: 11, end: 49 },
            Interval {
                start: 210,
                end: 260,
            },
        ];
        let (score, pairs, used) = match_intervals(&reference, &predicted);
        assert_eq!(score["matched"], 1);
        assert_eq!(score["extra"], 2);
        assert_eq!(score["missed"], 1);
        assert_eq!(pairs.iter().flatten().count(), 1);
        assert_eq!(used.iter().filter(|&&x| x).count(), 1);
        assert_eq!(
            differences(&reference, &predicted, &pairs, &used, &[]).len(),
            4
        );
    }

    #[test]
    fn reviewed_rally_missed_by_both_detectors_is_navigable() {
        let cpu = [Interval { start: 10, end: 50 }];
        let trajectory = [Interval { start: 12, end: 52 }];
        let reviewed = [Interval {
            start: 220,
            end: 260,
        }];
        let (_, pairs, used) = match_intervals(&cpu, &trajectory);
        let rows = differences(&cpu, &trajectory, &pairs, &used, &reviewed);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["kind"], "both_missed");
        assert_eq!(rows[1]["seek_frame"], 220);
    }

    #[test]
    fn tuning_reuses_saved_impacts_without_touch_inference() {
        let impacts = [100, 200, 300];
        let mut settings = RallySettings {
            min_rally_frames: 15,
            suggested_context_frames: 0,
            ..RallySettings::default()
        };
        assert_eq!(
            preliminary_rally_intervals(&impacts, 500, settings).len(),
            1
        );
        settings.max_without_touch_frames = 90;
        assert!(preliminary_rally_intervals(&impacts, 500, settings).is_empty());
        settings.max_without_touch_frames = 120;
        settings.min_touch_candidates = 4;
        assert!(preliminary_rally_intervals(&impacts, 500, settings).is_empty());
    }

    #[test]
    fn simulated_touches_only_follow_valid_rally_intervals() {
        let impacts = [10, 20, 30, 100, 110, 170];
        let segments = [
            Interval { start: 8, end: 35 },
            Interval {
                start: 105,
                end: 125,
            },
        ];
        assert_eq!(
            impacts_within_rallies(&impacts, &segments),
            vec![10, 20, 30, 110]
        );
        assert!(impacts_within_rallies(&impacts, &[]).is_empty());
    }

    #[test]
    fn calibration_improves_reviewed_rally_boundaries_without_changing_impacts() {
        let impacts = [100, 120, 200, 220, 500, 520, 600, 620];
        let reviewed = [
            Interval {
                start: 100,
                end: 220,
            },
            Interval {
                start: 500,
                end: 620,
            },
        ];
        let defaults = RallySettings::default();
        let original = impacts;
        let (recommended, baseline, calibrated) =
            fit_trajectory_settings(&impacts, 900, defaults, &reviewed).unwrap();
        assert_eq!(impacts, original);
        assert_eq!(recommended.suggested_context_frames, 0);
        assert_eq!(calibrated["matched"], 2);
        assert_eq!(calibrated["missed"], 0);
        assert_eq!(calibrated["extra"], 0);
        assert!(calibrated["f1"].as_f64().unwrap() >= baseline["f1"].as_f64().unwrap());
        assert!(
            calibrated["mean_boundary_error_ms"].as_f64().unwrap()
                < baseline["mean_boundary_error_ms"].as_f64().unwrap()
        );
    }

    #[test]
    fn calibration_requires_approved_nonempty_bounded_reference() {
        let impacts = [100, 120, 200, 220];
        let defaults = RallySettings::default();
        assert!(fit_trajectory_settings(&impacts, 500, defaults, &[]).is_none());
        let too_many = vec![
            Interval {
                start: 100,
                end: 200
            };
            257
        ];
        assert!(fit_trajectory_settings(&impacts, 500, defaults, &too_many).is_none());
        let good = [Interval {
            start: 100,
            end: 220,
        }];
        assert!(fit_trajectory_settings(&vec![10; 8193], 500, defaults, &good).is_none());
        assert!(fit_trajectory_settings(&[], 500, defaults, &good).is_none());
        let expensive = vec![Interval { start: 10, end: 40 }; 256];
        assert!(fit_trajectory_settings(&vec![10; 1000], 2000, defaults, &expensive).is_none());
    }

    #[test]
    fn interval_overlap_is_frame_accurate() {
        let a = Interval { start: 0, end: 29 };
        assert_eq!(overlap(a, a), 1.0);
        assert_eq!(overlap(a, Interval { start: 30, end: 45 }), 0.0);
        assert!(overlap(a, Interval { start: 15, end: 44 }) < 0.5);
    }
}
