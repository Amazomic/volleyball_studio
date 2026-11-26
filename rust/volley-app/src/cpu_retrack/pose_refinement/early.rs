use super::*;
use crate::cpu_retrack::{
    replay_frames_with_pose_options, FusedFrame, ReplayOptions, ReplayPhase, ReplayTotals,
};
use crate::rally_handoff::{update_progress, CompletedRally, RallyStageProgress, RallyWindow};
use std::{
    fs::File,
    path::PathBuf,
    sync::{mpsc, Arc, Mutex},
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
    pub sampling: PoseSampling,
    pub tracking: TrackingOptions,
    pub court: Option<crate::video_edits::CourtSnapshot>,
    pub interpolate_players: bool,
    pub threads: usize,
    pub cancelled: Arc<AtomicBool>,
    pub progress: Arc<Mutex<RallyStageProgress>>,
    pub reid_crops: Option<crate::native_player_tracks::SharedReidCrops>,
    pub checkpoint_root: Option<PathBuf>,
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
    selected: Vec<(u64, u64)>,
    motion: StabilizedBallMotion,
    rallies: Vec<(u64, u64)>,
    impacts: Vec<TrajectoryBreak>,
    dimensions: (u32, u32),
    memory_mb: u64,
}

type Packet = CompletedRally<RawDetection>;

pub(crate) struct RallyWorkerSender {
    sender: mpsc::SyncSender<Packet>,
    error: Arc<Mutex<Option<String>>>,
}

impl RallyWorkerSender {
    pub(crate) fn send(&self, packet: Packet) -> Result<(), String> {
        self.sender.send(packet).map_err(|_| {
            self.error
                .lock()
                .ok()
                .and_then(|error| error.clone())
                .unwrap_or_else(|| {
                    "Rally processing worker stopped without a recorded cause".into()
                })
        })
    }
}

fn run_with_worker_error<T>(
    error: &Mutex<Option<String>>,
    run: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|panic| {
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            Err(format!("Rally processing worker panicked: {message}"))
        });
    if let Err(cause) = &result {
        *error.lock().unwrap_or_else(|poison| poison.into_inner()) = Some(cause.clone());
    }
    result
}

impl EarlyPoseWorker {
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn start(
        options: EarlyPoseOptions,
        original: PathBuf,
        repository: PathBuf,
        range: Option<AnalysisRange>,
        decoded_frames: u64,
        selected: Vec<(u64, u64)>,
        motion: StabilizedBallMotion,
        rallies: Vec<(u64, u64)>,
        impacts: Vec<TrajectoryBreak>,
        dimensions: (u32, u32),
        memory_mb: u64,
    ) -> Result<(Self, RallyWorkerSender), String> {
        let input = WorkerInput {
            options,
            original,
            repository,
            range,
            decoded_frames,
            selected,
            motion,
            rallies,
            impacts,
            dimensions,
            memory_mb,
        };
        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let error = Arc::new(Mutex::new(None));
        let worker_error = error.clone();
        let handle = thread::Builder::new()
            .name("volley-early-rally-pose".into())
            .spawn(move || {
                run_with_worker_error(&worker_error, || run(input, &receiver, &worker_stop))
            })
            .map_err(|e| format!("Spawn rally pose worker: {e}"))?;
        Ok((
            Self {
                stop,
                handle: Some(handle),
            },
            RallyWorkerSender { sender, error },
        ))
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.handle
            .as_ref()
            .is_none_or(thread::JoinHandle::is_finished)
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
    reid_plan: BTreeMap<u64, Vec<[f64; 4]>>,
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
            court_filter: options
                .court
                .as_ref()
                .and_then(|c| c.transform.as_ref())
                .map(|court| crate::cpu_retrack::track_stitch::CourtFilter {
                    court,
                    scope: &scope,
                    serve_points: Some((serve_points, window.left)),
                }),
            interpolate_players: options.interpolate_players,
            scope: Some(&scope),
            cancel: Some(stop),
        },
    )?;
    update_progress(&options.progress, |p| {
        p.boxes_finished(frames, totals.tentative_tracks)
    })?;
    let mut reader = BoundedReader::open(&tracked)?;
    let mut schedule = PoseSchedule::new(options.sampling);
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
    let reid_plan = if options.reid_crops.is_some() {
        crate::native_player_tracks::plan_shared_reid_crops(
            &tracked,
            &digest,
            frames,
            (window.start - window.left, window.end - window.left),
            window.left,
        )
        .unwrap_or_else(|error| {
            eprintln!("Shared ReID crop plan unavailable: {error}");
            BTreeMap::new()
        })
    } else {
        BTreeMap::new()
    };
    Ok(LocalPlan {
        window,
        scope,
        reader: BoundedReader::open(&tracked)?,
        schedule,
        previous_crops: BTreeMap::new(),
        digest,
        totals,
        track_count,
        reid_plan,
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
        outside_tracks,
        outside_track_detections,
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

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RallyReceipt {
    schema: String,
    window: RallyWindow,
    packet_sha: String,
    id_offset: u64,
    totals: ReplayTotals,
    track_count: u64,
    groups: usize,
    sampled_frames: u64,
    track_sha: String,
    pose_sha: String,
    reid_plan: BTreeMap<u64, Vec<[f64; 4]>>,
}

struct RallyStart {
    packet_sha: String,
    track_offset: u64,
    pose_offset: u64,
    groups: usize,
    sampled_frames: u64,
    reid_plan: BTreeMap<u64, Vec<[f64; 4]>>,
}

fn packet_sha(packet: &Packet) -> Result<String, String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&packet.observations).map_err(|e| e.to_string())?)
    ))
}

fn rally_directory(root: &Path, window: RallyWindow) -> PathBuf {
    root.join(format!("rally-{}-{}", window.left, window.right))
}

fn copy_checkpoint_segment(source: &Path, target: &Path, offset: u64) -> Result<String, String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut source = File::open(source).map_err(|e| e.to_string())?;
    let bytes = source
        .metadata()
        .map_err(|e| e.to_string())?
        .len()
        .checked_sub(offset)
        .ok_or("Invalid rally checkpoint offset")?;
    source
        .seek(SeekFrom::Start(offset))
        .map_err(|e| e.to_string())?;
    let mut source = source.take(bytes);
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)
        .map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let count = source.read(&mut buf).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        output.write_all(&buf[..count]).map_err(|e| e.to_string())?;
        hash.update(&buf[..count]);
    }
    if source.limit() != 0 {
        return Err("Rally checkpoint segment was truncated".into());
    }
    output.sync_all().map_err(|e| e.to_string())?;
    Ok(format!("{:x}", hash.finalize()))
}

fn save_rally_checkpoint(
    root: &Path,
    scratch: &Scratch,
    start: &RallyStart,
    mut receipt: RallyReceipt,
) -> Result<(), String> {
    crate::storage::safe_folder(root)?;
    let target = rally_directory(root, receipt.window);
    let tmp = root.join(format!(".rally-{}.tmp", uuid::Uuid::new_v4()));
    fs::create_dir(&tmp).map_err(|e| e.to_string())?;
    let saved = (|| {
        receipt.track_sha = copy_checkpoint_segment(
            &scratch.0.join("tracks.jsonl"),
            &tmp.join("tracks.jsonl"),
            start.track_offset,
        )?;
        receipt.pose_sha = copy_checkpoint_segment(
            &scratch.0.join("poses.jsonl"),
            &tmp.join("poses.jsonl"),
            start.pose_offset,
        )?;
        let payload = serde_json::to_string(&receipt).map_err(|e| e.to_string())?;
        crate::storage::write_json_atomic(
            &tmp.join("receipt.json"),
            &json!({
                "payload":payload,
                "sha256":format!("{:x}", Sha256::digest(payload.as_bytes())),
            }),
        )
        .map_err(|e| format!("{e:?}"))?;
        if target.exists() {
            return Err("Rally checkpoint already committed".into());
        }
        fs::rename(&tmp, &target).map_err(|e| format!("Publish rally checkpoint: {e}"))
    })();
    if saved.is_err() {
        let _ = fs::remove_dir_all(&tmp);
    }
    saved
}

fn load_rally_checkpoint(
    root: &Path,
    window: RallyWindow,
    sha: &str,
    id_offset: u64,
) -> Result<Option<(RallyReceipt, PathBuf)>, String> {
    let folder = rally_directory(root, window);
    match fs::symlink_metadata(&folder) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.to_string()),
        Ok(_) => crate::storage::safe_folder(&folder)?,
    }
    let envelope = crate::storage::read_json(&folder.join("receipt.json"))
        .map_err(|e| format!("{e:?}"))?
        .ok_or("Missing rally checkpoint receipt")?;
    let payload = envelope["payload"]
        .as_str()
        .ok_or("Missing rally checkpoint payload")?;
    if envelope["sha256"] != format!("{:x}", Sha256::digest(payload.as_bytes())) {
        return Err("Rally checkpoint receipt checksum mismatch".into());
    }
    let receipt: RallyReceipt = serde_json::from_str(payload).map_err(|e| e.to_string())?;
    let frames = window.right - window.left + 1;
    if receipt.schema != "volley-phase1-rally/1"
        || receipt.window != window
        || receipt.packet_sha != sha
        || receipt.id_offset != id_offset
        || receipt.totals.processed_player_frames != frames
        || receipt.sampled_frames > frames
        || receipt.reid_plan.iter().any(|(frame, boxes)| {
            !(window.left..=window.right).contains(frame)
                || boxes.iter().flatten().any(|n| !n.is_finite())
        })
    {
        return Err(
            "Rally checkpoint does not match current observations or track identity offset".into(),
        );
    }
    Ok(Some((receipt, folder)))
}

fn replay_checkpoint_rows(
    folder: &Path,
    name: &str,
    expected_sha: &str,
    window: RallyWindow,
    frame_key: &str,
    writer: &mut BufWriter<File>,
    hash: &mut Sha256,
) -> Result<(), String> {
    let path = crate::storage::safe_file(folder, name, 64 * 1024 * 1024 * 1024)?;
    let mut reader = BoundedReader::open(&path)?;
    for frame in window.left..=window.right {
        let bytes = reader.line()?.ok_or("Truncated rally checkpoint")?;
        let row: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if row[frame_key].as_u64() != Some(frame) {
            return Err("Rally checkpoint frame order mismatch".into());
        }
        writer.write_all(&bytes).map_err(|e| e.to_string())?;
        hash.update(&bytes);
    }
    if reader.line()?.is_some() || reader.finish() != expected_sha {
        return Err("Rally checkpoint output checksum or coverage mismatch".into());
    }
    Ok(())
}

fn run(
    input: WorkerInput,
    receiver: &mpsc::Receiver<Packet>,
    stop: &AtomicBool,
) -> Result<CompletedRallies, String> {
    let global_frames = input.decoded_frames / 9 * 9;
    let serve_points = initial_serve_pose_points(&input.motion, &input.rallies);
    let make_plan = |packet| {
        update_progress(&input.options.progress, RallyStageProgress::boxes_started)?;
        build_local_plan(
            packet,
            &input.options,
            stop,
            global_frames,
            &input.impacts,
            &serve_points,
            input.dimensions,
        )
    };
    let mut plan: Option<LocalPlan> = None;
    let mut block_start: Option<RallyStart> = None;
    let mut recovered_reid = BTreeMap::<u64, Vec<[f64; 4]>>::new();
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
    let shared_reid = input.options.reid_crops.as_ref().filter(|buffer| {
        buffer
            .bind(&input.original, input.range.map_or(0.0, |r| r.start_sec))
            .is_ok()
    });
    let outcome = volley_media::raw_decoder::decode_original_frames_selected_in_range(
        &input.original,
        input.decoded_frames,
        input.range,
        &input.selected,
        stop,
        |index, pixels, width, height| {
            if input.options.cancelled.load(Ordering::Acquire) || stop.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }
            while plan.is_none() && !done {
                let Some(packet) = receive(receiver, stop, &input.options.cancelled)? else {
                    if windows.is_empty() {
                        return Err("Missing first rally packet".into());
                    }
                    done = true;
                    break;
                };
                if let Some(root) = input.options.checkpoint_root.as_deref() {
                    let sha = packet_sha(&packet)?;
                    let sha = if input.options.sampling == PoseSampling::default() {
                        sha
                    } else {
                        format!(
                            "{:x}",
                            Sha256::digest(format!(
                                "{}:{}:{}",
                                sha,
                                input.options.sampling.touch_stride,
                                input.options.sampling.background_stride
                            ))
                        )
                    };
                    if let Some((receipt, folder)) =
                        load_rally_checkpoint(root, packet.window, &sha, totals.tentative_tracks)?
                    {
                        replay_checkpoint_rows(
                            &folder,
                            "tracks.jsonl",
                            &receipt.track_sha,
                            receipt.window,
                            "frame_index",
                            &mut tracks,
                            &mut track_hash,
                        )?;
                        replay_checkpoint_rows(
                            &folder,
                            "poses.jsonl",
                            &receipt.pose_sha,
                            receipt.window,
                            "frame",
                            &mut poses,
                            &mut pose_hash,
                        )?;
                        windows.push(receipt.window);
                        add_totals(&mut totals, &receipt.totals);
                        track_count += receipt.track_count;
                        groups += receipt.groups;
                        sampled_frames += receipt.sampled_frames;
                        if shared_reid.is_some() {
                            recovered_reid.extend(receipt.reid_plan);
                        }
                        update_progress(&input.options.progress, |p| {
                            p.rally_recovered(
                                receipt.window.right - receipt.window.left + 1,
                                receipt.totals.tentative_tracks,
                                receipt.sampled_frames,
                            )
                        })?;
                        continue;
                    }
                    let active = make_plan(packet)?;
                    tracks.flush().map_err(|e| e.to_string())?;
                    poses.flush().map_err(|e| e.to_string())?;
                    block_start = Some(RallyStart {
                        packet_sha: sha,
                        track_offset: tracks
                            .get_ref()
                            .metadata()
                            .map_err(|e| e.to_string())?
                            .len(),
                        pose_offset: poses.get_ref().metadata().map_err(|e| e.to_string())?.len(),
                        groups,
                        sampled_frames,
                        reid_plan: active.reid_plan.clone(),
                    });
                    plan = Some(active);
                } else {
                    plan = Some(make_plan(packet)?);
                }
            }
            if let Some(shared_reid) = shared_reid {
                if let Some(boxes) = recovered_reid.remove(&index) {
                    if let Err(error) = shared_reid.capture(index, pixels, width, height, &boxes) {
                        eprintln!("Shared recovered ReID crop unavailable: {error}");
                    }
                }
            }
            let Some(active) = plan.as_mut() else {
                if !recovered_reid.is_empty() {
                    return Ok(());
                }
                return Err("rally intervals complete".into());
            };
            if index < active.window.left {
                return Ok(());
            }
            if index > active.window.right {
                return Err("Rally windows out of order".into());
            }
            if index == active.window.left {
                update_progress(&input.options.progress, RallyStageProgress::pose_started)?;
            }
            let mut row = read_crop_plan_row(
                &mut active.reader,
                index - active.window.left,
                &active.scope,
            )?;
            if let Some(shared_reid) = shared_reid {
                if let Some(boxes) = active.reid_plan.remove(&index) {
                    if let Err(error) = shared_reid.capture(index, pixels, width, height, &boxes) {
                        eprintln!("Shared ReID crop unavailable: {error}");
                    }
                }
            }
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
            update_progress(&input.options.progress, |p| p.pose_frame(!crops.is_empty()))?;
            if index == active.window.right {
                let mut finished = plan.take().ok_or("Missing completed rally")?;
                if finished.reader.line()?.is_some() || finished.reader.finish() != finished.digest
                {
                    return Err("Rally crop plan changed".into());
                }
                if let (Some(root), Some(start)) =
                    (input.options.checkpoint_root.as_deref(), block_start.take())
                {
                    flush(&mut tracks)?;
                    flush(&mut poses)?;
                    save_rally_checkpoint(
                        root,
                        &scratch,
                        &start,
                        RallyReceipt {
                            schema: "volley-phase1-rally/1".into(),
                            window: finished.window,
                            packet_sha: start.packet_sha.clone(),
                            id_offset: totals.tentative_tracks,
                            totals: finished.totals.clone(),
                            track_count: finished.track_count,
                            groups: groups - start.groups,
                            sampled_frames: sampled_frames - start.sampled_frames,
                            track_sha: String::new(),
                            pose_sha: String::new(),
                            reid_plan: start.reid_plan.clone(),
                        },
                    )?;
                }
                windows.push(finished.window);
                add_totals(&mut totals, &finished.totals);
                track_count += finished.track_count;
                update_progress(&input.options.progress, RallyStageProgress::pose_finished)?;
            }
            Ok(())
        },
    );
    if !done || !recovered_reid.is_empty() {
        outcome?;
        if plan.is_some()
            || !recovered_reid.is_empty()
            || receive(receiver, stop, &input.options.cancelled)?.is_some()
        {
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
    fn disconnected_worker_preserves_original_error_and_panic() {
        for panic in [false, true] {
            let (sender, receiver) = mpsc::sync_channel(1);
            let error = Arc::new(Mutex::new(None));
            let producer = RallyWorkerSender {
                sender,
                error: error.clone(),
            };
            let worker = thread::spawn(move || {
                let result = run_with_worker_error::<()>(&error, || {
                    if panic {
                        panic!("decoder panic");
                    }
                    Err("Original RGB frame 516: FFmpeg filter initialization failed".into())
                });
                drop(receiver);
                result
            });
            let result = worker.join().unwrap().unwrap_err();
            let packet = CompletedRally {
                window: RallyWindow {
                    left: 0,
                    right: 8,
                    start: 0,
                    end: 8,
                },
                observations: vec![],
            };
            assert_eq!(producer.send(packet).unwrap_err(), result);
            assert!(result.contains(if panic {
                "decoder panic"
            } else {
                "FFmpeg filter initialization failed"
            }));
        }
    }

    #[test]
    fn rally_checkpoint_recovers_only_committed_rows_and_checks_identity() {
        let root = Scratch::new().unwrap();
        let scratch = Scratch::new().unwrap();
        let window = RallyWindow {
            left: 2,
            right: 3,
            start: 2,
            end: 3,
        };
        let prefix = b"{\"frame_index\":1}\n";
        let tracks = b"{\"frame_index\":2}\n{\"frame_index\":3}\n";
        let poses = b"{\"frame\":2}\n{\"frame\":3}\n";
        fs::write(
            scratch.0.join("tracks.jsonl"),
            [prefix.as_slice(), tracks.as_slice()].concat(),
        )
        .unwrap();
        fs::write(scratch.0.join("poses.jsonl"), poses).unwrap();
        assert!(load_rally_checkpoint(&root.0, window, "packet", 10)
            .unwrap()
            .is_none());
        let start = RallyStart {
            packet_sha: "packet".into(),
            track_offset: prefix.len() as u64,
            pose_offset: 0,
            groups: 0,
            sampled_frames: 0,
            reid_plan: BTreeMap::new(),
        };
        save_rally_checkpoint(
            &root.0,
            &scratch,
            &start,
            RallyReceipt {
                schema: "volley-phase1-rally/1".into(),
                window,
                packet_sha: "packet".into(),
                id_offset: 10,
                totals: ReplayTotals {
                    processed_player_frames: 2,
                    ..Default::default()
                },
                track_count: 1,
                groups: 1,
                sampled_frames: 2,
                track_sha: String::new(),
                pose_sha: String::new(),
                reid_plan: BTreeMap::new(),
            },
        )
        .unwrap();
        let (receipt, folder) = load_rally_checkpoint(&root.0, window, "packet", 10)
            .unwrap()
            .unwrap();
        assert!(load_rally_checkpoint(&root.0, window, "different", 10).is_err());
        assert!(load_rally_checkpoint(&root.0, window, "packet", 11).is_err());
        let output = scratch.0.join("restored.jsonl");
        let mut writer = BufWriter::new(File::create(&output).unwrap());
        replay_checkpoint_rows(
            &folder,
            "tracks.jsonl",
            &receipt.track_sha,
            window,
            "frame_index",
            &mut writer,
            &mut Sha256::new(),
        )
        .unwrap();
        writer.flush().unwrap();
        assert_eq!(fs::read(output).unwrap(), tracks);
        fs::write(
            folder.join("tracks.jsonl"),
            b"{\"frame_index\":2,\"changed\":true}\n{\"frame_index\":3}\n",
        )
        .unwrap();
        assert!(replay_checkpoint_rows(
            &folder,
            "tracks.jsonl",
            &receipt.track_sha,
            window,
            "frame_index",
            &mut writer,
            &mut Sha256::new()
        )
        .is_err());
    }

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
