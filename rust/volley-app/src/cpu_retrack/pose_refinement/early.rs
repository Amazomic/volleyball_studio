use super::*;
use crate::cpu_retrack::{
    replay_frames_with_pose_options, FusedFrame, ReplayOptions, ReplayPhase, ReplayTotals,
};
use crate::rally_handoff::{CompletedRally, RallyWindow};
use std::{
    fs::File,
    path::PathBuf,
    sync::{mpsc, Arc},
    thread,
    time::Duration,
};
use volley_core::{
    ball_interpolation::StabilizedBallMotion, fusion::fuse_window, motion::TrajectoryBreak,
    tracker::TrackingOptions,
};
use volley_inference::decode::RawDetection;

const QUEUE_CAPACITY: usize = 2;

#[derive(Debug)]
pub(crate) struct CompletedRallies {
    scratch: Scratch,
    track_sha: String,
    pose_sha: String,
    windows: Vec<RallyWindow>,
    totals: ReplayTotals,
    track_count: u64,
    groups: usize,
    sampled_frames: u64,
}

impl CompletedRallies {
    pub(crate) fn group_count(&self) -> usize {
        self.groups
    }
}

pub(crate) struct EarlyPoseOptions {
    pub tracking: TrackingOptions,
    pub interpolate_players: bool,
    pub threads: usize,
    pub cancelled: Arc<AtomicBool>,
}

pub(crate) struct EarlyPoseWorker {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<Result<CompletedRallies, String>>>,
}

struct WorkerInput {
    options: EarlyPoseOptions,
    original: PathBuf,
    repository: PathBuf,
    range: Option<AnalysisRange>,
    decoded_frames: u64,
    motion: StabilizedBallMotion,
    rallies: Vec<(u64, u64)>,
    dimensions: (u32, u32),
    memory_mb: u64,
}

type Packet = CompletedRally<RawDetection>;

impl EarlyPoseWorker {
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn start(
        options: EarlyPoseOptions,
        original: PathBuf,
        repository: PathBuf,
        range: Option<AnalysisRange>,
        decoded_frames: u64,
        motion: StabilizedBallMotion,
        rallies: Vec<(u64, u64)>,
        dimensions: (u32, u32),
        memory_mb: u64,
    ) -> Result<(Self, mpsc::SyncSender<Packet>), String> {
        let input = WorkerInput {
            options,
            original,
            repository,
            range,
            decoded_frames,
            motion,
            rallies,
            dimensions,
            memory_mb,
        };
        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let handle = thread::Builder::new()
            .name("volley-early-rally-pose".into())
            .spawn(move || run(input, receiver, &worker_stop))
            .map_err(|e| format!("Spawn rally pose worker: {e}"))?;
        Ok((
            Self {
                stop,
                handle: Some(handle),
            },
            sender,
        ))
    }

    pub(crate) fn finish(mut self) -> Result<CompletedRallies, String> {
        self.handle
            .take()
            .ok_or("Missing rally pose worker")?
            .join()
            .map_err(|_| "Rally pose worker panicked".to_owned())?
    }
}

impl Drop for EarlyPoseWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Debug)]
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!("volley-rally-pose-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&path).map_err(|e| e.to_string())?;
        Ok(Self(path))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct LocalPlan {
    window: RallyWindow,
    scope: PlayerFrameScope,
    reader: BoundedReader,
    schedule: PoseSchedule,
    previous_crops: BTreeMap<u64, [f64; 4]>,
    digest: String,
    totals: ReplayTotals,
    track_count: u64,
    _scratch: Scratch,
}

fn build_local_plan(
    packet: Packet,
    options: &EarlyPoseOptions,
    stop: &AtomicBool,
    global_frames: u64,
    impacts: &[TrajectoryBreak],
    serve_points: &BTreeMap<u64, [f64; 2]>,
    dimensions: (u32, u32),
) -> Result<LocalPlan, String> {
    let window = packet.window;
    let frames = window.right - window.left + 1;
    if frames > 1_000_000 || window.right >= global_frames {
        return Err("Rally pose window exceeds frame bounds".into());
    }
    let scratch = Scratch::new()?;
    let fused = scratch.0.join("fused.jsonl");
    let tracked = scratch.0.join("tracked.jsonl");
    let mut buckets = BTreeMap::<u64, Vec<Candidate>>::new();
    for detection in packet.observations {
        if stop.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        if !(window.left..=window.right).contains(&detection.frame_index) {
            return Err("Rally detection outside its handoff window".into());
        }
        let mut candidate: Candidate = serde_json::from_slice(
            &serde_json::to_vec(&serde_json::json!(&detection)).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        candidate.frame_index -= window.left;
        buckets
            .entry(candidate.frame_index / 9 * 9)
            .or_default()
            .push(candidate);
    }
    let mut writer = BufWriter::new(File::create(&fused).map_err(|e| e.to_string())?);
    let mut hash = Sha256::new();
    for first in (0..frames).step_by(9) {
        if stop.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        for frame in fuse_window(
            first,
            buckets.remove(&first).unwrap_or_default(),
            Vec::new(),
        )? {
            if frame.frame_index >= frames {
                break;
            }
            let mut bytes = serde_json::to_vec(&frame).map_err(|e| e.to_string())?;
            bytes.push(b'\n');
            writer.write_all(&bytes).map_err(|e| e.to_string())?;
            hash.update(&bytes);
        }
    }
    writer.flush().map_err(|e| e.to_string())?;
    drop(writer);
    let scope = PlayerFrameScope {
        selected: vec![(0, frames - 1)],
        rallies: vec![(window.start - window.left, window.end - window.left)],
        ball_job_id: String::new(),
        mask_sha256: String::new(),
    };
    let (digest, totals, track_count) = replay_frames_with_pose_options(
        &fused,
        &tracked,
        frames,
        options.tracking,
        &format!("{:x}", hash.finalize()),
        None,
        ReplayOptions {
            phase: ReplayPhase::StablePlan,
            interpolate_players: options.interpolate_players,
            scope: Some(&scope),
            cancel: Some(stop),
        },
    )?;
    let mut reader = BoundedReader::open(&tracked)?;
    let mut schedule = PoseSchedule::default();
    for frame in 0..frames {
        if stop.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let row = read_crop_plan_row(&mut reader, frame, &scope)?;
        schedule_frame(
            &mut schedule,
            &row,
            window.left + frame,
            global_frames,
            impacts,
            serve_points,
            (f64::from(dimensions.0), f64::from(dimensions.1)),
        )?;
    }
    if reader.line()?.is_some() || reader.finish() != digest {
        return Err("Early crop plan changed before pose inference".into());
    }
    Ok(LocalPlan {
        window,
        scope,
        reader: BoundedReader::open(&tracked)?,
        schedule,
        previous_crops: BTreeMap::new(),
        digest,
        totals,
        track_count,
        _scratch: scratch,
    })
}

fn receive(
    receiver: &mpsc::Receiver<Packet>,
    stop: &AtomicBool,
    cancelled: &AtomicBool,
) -> Result<Option<Packet>, String> {
    loop {
        if stop.load(Ordering::Acquire) || cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        match receiver.recv_timeout(Duration::from_millis(50)) {
            Ok(packet) => return Ok(Some(packet)),
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(None),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
}

fn bump(value: &mut Value, offset: u64) -> Result<(), String> {
    if let Some(n) = value.as_u64() {
        *value = json!(n.checked_add(offset).ok_or("Rally index overflow")?);
    } else if !value.is_null() {
        return Err("Invalid rally index".into());
    }
    Ok(())
}

fn rebase(value: &mut Value, frame_offset: u64, id_offset: u64) -> Result<(), String> {
    match value {
        Value::Object(fields) => {
            for (key, child) in fields {
                match key.as_str() {
                    "frame" | "frame_index" | "rally_roster_window" | "left" => {
                        bump(child, frame_offset)?
                    }
                    "track_id"
                    | "closest_previous_track_id"
                    | "occluding_track_id"
                    | "occluder" => bump(child, id_offset)?,
                    "interpolated_track_ids" | "occluding_track_ids" => {
                        for id in child.as_array_mut().ok_or("Invalid rally ID list")? {
                            bump(id, id_offset)?;
                        }
                    }
                    _ => rebase(child, frame_offset, id_offset)?,
                }
            }
        }
        Value::Array(values) => {
            for child in values {
                rebase(child, frame_offset, id_offset)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn write_row(writer: &mut BufWriter<File>, hash: &mut Sha256, row: &Value) -> Result<(), String> {
    let mut bytes = serde_json::to_vec(row).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    if bytes.len() > MAX_ROW_BYTES {
        return Err("Rally output row exceeds limit".into());
    }
    writer.write_all(&bytes).map_err(|e| e.to_string())?;
    hash.update(bytes);
    Ok(())
}

fn flush(writer: &mut BufWriter<File>) -> Result<(), String> {
    writer.flush().map_err(|e| e.to_string())?;
    writer.get_ref().sync_all().map_err(|e| e.to_string())
}

fn add_totals(total: &mut ReplayTotals, block: &ReplayTotals) {
    macro_rules! sum {
        ($($field:ident),*) => { $(total.$field += block.$field;)* };
    }
    sum!(
        frames,
        detections,
        raw_player_detections,
        rejected,
        ambiguous,
        frames_with_ball,
        tentative_tracks,
        unconfirmed_tracks,
        short_track_detections,
        static_tracks,
        static_track_detections,
        duplicate_track_ids_merged,
        identity_turnover_splits,
        identity_handoff_splits,
        pose_detections,
        unassigned_rally_boxes,
        combined_pose_crops,
        interpolated_player_boxes,
        processed_player_frames,
        bytetrack_low_confidence_matches,
        bytetrack_lost_low_confidence_matches
    );
}

fn run(
    input: WorkerInput,
    receiver: mpsc::Receiver<Packet>,
    stop: &AtomicBool,
) -> Result<CompletedRallies, String> {
    let global_frames = input.decoded_frames / 9 * 9;
    let impacts = crate::cpu_touches::trajectory_breaks(&input.motion, input.dimensions, stop)?;
    let serve_points = initial_serve_pose_points(&input.motion, &input.rallies);
    let make_plan = |packet| {
        build_local_plan(
            packet,
            &input.options,
            stop,
            global_frames,
            &impacts,
            &serve_points,
            input.dimensions,
        )
    };
    let mut plan = Some(make_plan(
        receive(&receiver, stop, &input.options.cancelled)?.ok_or("Missing first rally packet")?,
    )?);
    let scratch = Scratch::new()?;
    let mut tracks =
        BufWriter::new(File::create(scratch.0.join("tracks.jsonl")).map_err(|e| e.to_string())?);
    let mut poses =
        BufWriter::new(File::create(scratch.0.join("poses.jsonl")).map_err(|e| e.to_string())?);
    let mut track_hash = Sha256::new();
    let mut pose_hash = Sha256::new();
    let mut windows = Vec::new();
    let mut totals = ReplayTotals::default();
    let mut track_count = 0;
    let mut groups = 0;
    let mut sampled_frames = 0;
    let mut engine = None;
    let mut done = false;
    let outcome = decode_original_frames_in_range(
        &input.original,
        input.decoded_frames,
        input.range,
        stop,
        |index, pixels, width, height| {
            if input.options.cancelled.load(Ordering::Acquire) || stop.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }
            if plan.is_none() {
                plan = receive(&receiver, stop, &input.options.cancelled)?
                    .map(make_plan)
                    .transpose()?;
            }
            let Some(active) = plan.as_mut() else {
                done = true;
                return Err("rally intervals complete".into());
            };
            if index < active.window.left {
                return Ok(());
            }
            if index > active.window.right {
                return Err("Rally windows out of order".into());
            }
            let mut row = read_crop_plan_row(
                &mut active.reader,
                index - active.window.left,
                &active.scope,
            )?;
            let players = row["tracked_players"]
                .as_array()
                .ok_or("Missing rally tracks")?;
            let crops = frame_crops(
                players,
                index,
                &mut active.previous_crops,
                |frame, track| active.schedule.measure(frame, track),
            )?;
            sampled_frames += u64::from(!crops.is_empty());
            let mut measured = Vec::new();
            for group in crops.chunks(8) {
                if engine.is_none() {
                    engine = Some(PoseEngine::load(
                        &input.repository,
                        Device::Cuda,
                        input.options.threads,
                        input.memory_mb,
                    )?);
                }
                let boxes = group.iter().map(|(_, bbox, _)| *bbox).collect::<Vec<_>>();
                let visibility = group
                    .iter()
                    .map(|(player, _, _)| pose_visibility(&players[*player]))
                    .collect::<Vec<_>>();
                let joints = estimate_group(
                    engine.as_mut().ok_or("Missing rally RTMPose")?,
                    pixels,
                    (width, height),
                    &boxes,
                    &visibility,
                )?;
                groups += 1;
                for ((player, bbox, origin), joints) in group.iter().zip(joints) {
                    if let Some(joints) = joints {
                        measured.push(json!({
                            "track_id": players[*player]["track_id"],
                            "box": bbox, "keypoints": joints,
                            "crop_origin": if *origin == "interpolated_track_box" { "interpolated_box" } else { "observed" },
                        }));
                    }
                }
            }
            let mut pose = json!({"frame":index - active.window.left,"players":measured});
            rebase(&mut pose, active.window.left, totals.tentative_tracks)?;
            rebase(&mut row, active.window.left, totals.tentative_tracks)?;
            write_row(&mut tracks, &mut track_hash, &row)?;
            write_row(&mut poses, &mut pose_hash, &pose)?;
            if index == active.window.right {
                let mut finished = plan.take().ok_or("Missing completed rally")?;
                if finished.reader.line()?.is_some() || finished.reader.finish() != finished.digest
                {
                    return Err("Rally crop plan changed".into());
                }
                windows.push(finished.window);
                add_totals(&mut totals, &finished.totals);
                track_count += finished.track_count;
            }
            Ok(())
        },
    );
    if !done {
        outcome?;
        if plan.is_some() || receive(&receiver, stop, &input.options.cancelled)?.is_some() {
            return Err("Rally pipeline ended with unfinished windows".into());
        }
    }
    flush(&mut tracks)?;
    flush(&mut poses)?;
    drop(tracks);
    drop(poses);
    Ok(CompletedRallies {
        scratch,
        track_sha: format!("{:x}", track_hash.finalize()),
        pose_sha: format!("{:x}", pose_hash.finalize()),
        windows,
        totals,
        track_count,
        groups,
        sampled_frames,
    })
}

impl CompletedRallies {
    pub(in crate::cpu_retrack) fn assemble(
        self,
        source: &Path,
        expected_sha: &str,
        output: &Path,
        frames: u64,
        scope: &PlayerFrameScope,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(String, ReplayTotals, u64, String, u64), String> {
        let expected = crate::rally_handoff::rally_windows(&scope.selected, &scope.rallies)?;
        if self.windows != expected || self.totals.processed_player_frames != scope.frame_count() {
            return Err("Completed rally coverage differs from verified selection".into());
        }
        let mut input = BoundedReader::open(source)?;
        let mut tracked = BoundedReader::open(&self.scratch.0.join("tracks.jsonl"))?;
        let mut measured = BoundedReader::open(&self.scratch.0.join("poses.jsonl"))?;
        let mut output_writer = BufWriter::new(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(output)
                .map_err(|e| e.to_string())?,
        );
        let mut pose_writer = BufWriter::new(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(output.with_file_name("poses.jsonl"))
                .map_err(|e| e.to_string())?,
        );
        let mut output_sha = Sha256::new();
        let mut pose_sha = Sha256::new();
        let mut totals = self.totals;
        totals.frames = frames;
        totals.frames_with_ball = 0;
        for frame in 0..frames {
            if cancelled.is_some_and(|c| c.load(Ordering::Acquire)) {
                return Err("cancelled".into());
            }
            let fused: FusedFrame =
                serde_json::from_slice(&input.line()?.ok_or("Fusion truncated")?)
                    .map_err(|e| e.to_string())?;
            if fused.frame_index != frame
                || fused.player_track_ids_confirmed
                || fused.touches_confirmed
                || fused.ball_candidates.len() > 2
            {
                return Err("Invalid fusion during rally assembly".into());
            }
            let raw = json!(fused.player_candidates);
            let (mut row, pose) = if scope.contains(frame) {
                let row: Value =
                    serde_json::from_slice(&tracked.line()?.ok_or("Rally tracking truncated")?)
                        .map_err(|e| e.to_string())?;
                let pose: Value =
                    serde_json::from_slice(&measured.line()?.ok_or("Rally poses truncated")?)
                        .map_err(|e| e.to_string())?;
                if row["frame_index"] != frame
                    || pose["frame"] != frame
                    || row["raw_player_candidates"] != raw
                {
                    return Err(format!(
                        "Rally output differs from immutable RF-DETR data at frame {frame}"
                    ));
                }
                (row, pose)
            } else {
                if !fused.player_candidates.is_empty() {
                    return Err("Player detections outside rally scope".into());
                }
                (
                    json!({
                        "frame_index":frame, "tracked_players":[], "raw_player_candidates":raw,
                        "ambiguous_detections":0, "new_track_diagnostics":[],
                        "track_ids_provisional":true, "confirmed_athlete_identities":false,
                        "touches_confirmed":false,
                    }),
                    json!({"frame":frame,"players":[]}),
                )
            };
            let selected = fused
                .selected_ball
                .filter(|b| b["source_model"] == "vballnet");
            totals.frames_with_ball += u64::from(selected.is_some());
            row["ball_candidates"] = json!(fused.ball_candidates);
            row["selected_ball"] = json!(selected);
            write_row(&mut output_writer, &mut output_sha, &row)?;
            write_row(&mut pose_writer, &mut pose_sha, &pose)?;
        }
        if input.line()?.is_some()
            || input.finish() != expected_sha
            || tracked.line()?.is_some()
            || tracked.finish() != self.track_sha
            || measured.line()?.is_some()
            || measured.finish() != self.pose_sha
        {
            return Err("Rally assembly checksum or frame count mismatch".into());
        }
        flush(&mut output_writer)?;
        flush(&mut pose_writer)?;
        Ok((
            format!("{:x}", output_sha.finalize()),
            totals,
            self.track_count,
            format!("{:x}", pose_sha.finalize()),
            self.sampled_frames,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembly_uses_completed_rows_and_rejects_missing_or_changed_evidence() {
        let scope = PlayerFrameScope {
            selected: vec![(1, 1)],
            rallies: vec![(1, 1)],
            ball_job_id: String::new(),
            mask_sha256: String::new(),
        };
        let directory = Scratch::new().unwrap();
        let fused = directory.0.join("fusion.jsonl");
        let mut input = Vec::new();
        for frame in 0..3 {
            serde_json::to_writer(&mut input, &json!({
                "frame_index":frame, "player_candidates":[], "ball_candidates":[],
                "selected_ball":null, "player_track_ids_confirmed":false, "touches_confirmed":false
            })).unwrap();
            input.push(b'\n');
        }
        fs::write(&fused, &input).unwrap();
        let input_sha = format!("{:x}", Sha256::digest(&input));
        let completed = |changed: bool| {
            let scratch = Scratch::new().unwrap();
            let row = json!({
                "frame_index":1, "tracked_players":[], "raw_player_candidates":
                    if changed {json!([{"frame_index":1}])} else {json!([])},
                "ball_candidates":[], "selected_ball":null, "ambiguous_detections":0,
                "new_track_diagnostics":[], "track_ids_provisional":true,
                "confirmed_athlete_identities":false, "touches_confirmed":false,
                "rally_roster_window":1
            });
            let mut track_writer =
                BufWriter::new(File::create(scratch.0.join("tracks.jsonl")).unwrap());
            let mut pose_writer =
                BufWriter::new(File::create(scratch.0.join("poses.jsonl")).unwrap());
            let mut track_hash = Sha256::new();
            let mut pose_hash = Sha256::new();
            write_row(&mut track_writer, &mut track_hash, &row).unwrap();
            write_row(
                &mut pose_writer,
                &mut pose_hash,
                &json!({"frame":1,"players":[]}),
            )
            .unwrap();
            flush(&mut track_writer).unwrap();
            flush(&mut pose_writer).unwrap();
            CompletedRallies {
                scratch,
                track_sha: format!("{:x}", track_hash.finalize()),
                pose_sha: format!("{:x}", pose_hash.finalize()),
                windows: crate::rally_handoff::rally_windows(&scope.selected, &scope.rallies)
                    .unwrap(),
                totals: ReplayTotals {
                    frames: 1,
                    processed_player_frames: 1,
                    ..Default::default()
                },
                track_count: 0,
                groups: 0,
                sampled_frames: 0,
            }
        };
        let output = directory.0.join("tracked.jsonl");
        let (sha, totals, _, pose_sha, _) = completed(false)
            .assemble(&fused, &input_sha, &output, 3, &scope, None)
            .unwrap();
        let bytes = fs::read(&output).unwrap();
        assert_eq!(sha, format!("{:x}", Sha256::digest(&bytes)));
        assert_eq!(totals.frames, 3);
        assert_eq!(totals.processed_player_frames, 1);
        let rows = String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert!(rows[0]["rally_roster_window"].is_null());
        assert_eq!(rows[1]["rally_roster_window"], 1);
        assert!(rows[2]["rally_roster_window"].is_null());
        assert_eq!(
            pose_sha,
            format!(
                "{:x}",
                Sha256::digest(fs::read(directory.0.join("poses.jsonl")).unwrap())
            )
        );
        let bad = directory.0.join("bad");
        fs::create_dir(&bad).unwrap();
        assert!(completed(true)
            .assemble(
                &fused,
                &input_sha,
                &bad.join("tracked.jsonl"),
                3,
                &scope,
                None
            )
            .is_err());
    }

    #[test]
    fn rebase_only_frame_and_identity_fields() {
        let mut row = json!({"frame_index":3,"rally_roster_window":0,"tracked_players":[
            {"track_id":1,"detection":{"frame_index":3,"query_index":5,"bbox_xyxy_norm":[0.1,0.2,0.3,0.4]},
            "occluding_track_ids":[2],"occlusion_bridge":{"left":1,"occluder":2}}]});
        rebase(&mut row, 100, 20).unwrap();
        assert_eq!(row["frame_index"], 103);
        assert_eq!(row["rally_roster_window"], 100);
        assert_eq!(row["tracked_players"][0]["track_id"], 21);
        assert_eq!(row["tracked_players"][0]["detection"]["frame_index"], 103);
        assert_eq!(row["tracked_players"][0]["detection"]["query_index"], 5);
        assert_eq!(
            row["tracked_players"][0]["occluding_track_ids"],
            json!([22])
        );
        assert_eq!(
            row["tracked_players"][0]["occlusion_bridge"],
            json!({"left":101,"occluder":22})
        );
    }
}
