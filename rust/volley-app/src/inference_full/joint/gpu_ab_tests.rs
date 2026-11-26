use super::*;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::AtomicBool,
    time::Instant,
};

struct IsolatedGpuFixture(PathBuf);
impl Drop for IsolatedGpuFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn state(repo: PathBuf, data_root: PathBuf) -> AppState {
    AppState {
        repository_root: repo,
        data_root,
        writes_enabled: true,
        montage: crate::montage::MontageManager::new(),
        preparation: crate::preparation::PreparationManager::new(),
        preview_limit: tokio::sync::Semaphore::new(1),
        touch_limit: tokio::sync::Semaphore::new(1),
        continuity_limit: tokio::sync::Semaphore::new(1),
        reid_limit: tokio::sync::Semaphore::new(1),
        identity_limit: tokio::sync::Semaphore::new(1),
        statistics_limit: tokio::sync::Semaphore::new(1),
        marks_limit: tokio::sync::Semaphore::new(1),
        quick_reel: crate::quick_reel::QuickReelManager::new(),
        inference_limit: tokio::sync::Semaphore::new(1),
        full_inference: crate::inference_full::FullInferenceManager::new(),
    }
}

fn manifest(root: &Path, category: &str, id: &str) -> Result<Value, String> {
    let file = root
        .join("projects")
        .join("012345abcdef")
        .join(category)
        .join(id)
        .join("manifest.json");
    serde_json::from_slice(&fs::read(&file).map_err(|e| format!("Read {}: {e}", file.display()))?)
        .map_err(|e| format!("Parse manifest: {e}"))
}

fn model_hash(root: &Path, id: &str) -> Result<String, String> {
    manifest(root, "native_gpu_runs", id)?["sha256"]
        .as_str()
        .map(str::to_owned)
        .ok_or("Missing model sha".into())
}

fn fusion_pair(state: &AppState, video: &str, ball: &str, people: &str) -> Result<Value, String> {
    crate::inference_fusion::do_fusion(
        state,
        &crate::inference_fusion::FuseRequest {
            project_id: "012345abcdef".to_owned(),
            video: video.to_owned(),
            person_job_id: people.to_owned(),
            vball_job_id: ball.to_owned(),
        },
    )
    .map_err(|e| format!("Fusion failed: {e:?}"))
}

fn pose_samples(state: &AppState, pose: &Value) -> Result<Vec<(u64, usize)>, String> {
    let id = pose["pose_run_id"].as_str().ok_or("Missing pose run ID")?;
    let file = state
        .data_root
        .join("projects/012345abcdef/native_gpu_pose_runs")
        .join(id)
        .join("poses.jsonl");
    let bytes = fs::read(file).map_err(|e| format!("Read pose evidence: {e}"))?;
    bytes
        .split(|b| *b == b'\n')
        .filter(|row| !row.is_empty())
        .map(|row| {
            let v: Value = serde_json::from_slice(row).map_err(|e| e.to_string())?;
            Ok((
                v["frame"].as_u64().ok_or("Pose frame missing")?,
                v["players"]
                    .as_array()
                    .ok_or("Pose candidates missing")?
                    .len(),
            ))
        })
        .collect()
}

fn replay_touches_and_rallies(
    app: &AppState,
    video: &str,
    fusion: &Value,
) -> Result<(Value, Value, Value), String> {
    let fusion_id = fusion["fusion_id"]
        .as_str()
        .ok_or("Missing fusion id for CPU A/B")?;
    let replay = crate::cpu_retrack::process(
        app,
        &crate::cpu_retrack::ReplayRequest {
            project_id: "012345abcdef".to_owned(),
            video: video.to_owned(),
            fusion_id: fusion_id.to_owned(),
            settings: None,
            interpolate_players: true,
            interpolate_pose: true,
        },
    )
    .map_err(|e| format!("CPU Track ID A/B: {e:?}"))?;
    let replay_id = replay["replay_id"]
        .as_str()
        .ok_or("Missing CPU Replay ID")?;
    let touch = crate::cpu_touches::process_discovery(
        app,
        &crate::cpu_touches::TouchRequest {
            project_id: "012345abcdef".to_owned(),
            video: video.to_owned(),
            replay_id: replay_id.to_owned(),
            settings: None,
            pose_classification: true,
        },
    )
    .map_err(|e| format!("CPU Touch A/B: {e:?}"))?;
    let touch_id = touch["touch_id"].as_str().ok_or("Missing CPU Touch ID")?;
    let rallies = crate::cpu_rallies::process(
        app,
        &crate::cpu_rallies::RallyRequest {
            project_id: "012345abcdef".to_owned(),
            video: video.to_owned(),
            replay_id: replay_id.to_owned(),
            touch_id: touch_id.to_owned(),
            settings: None,
        },
    )
    .map_err(|e| format!("CPU Rally A/B: {e:?}"))?;
    Ok((replay, touch, rallies))
}

/// Run explicitly, never in routine CI:
/// VOLLEY_AB_SOURCE=/path/to/uu/original.mp4
/// VOLLEY_AB_MODEL_ROOT=/path/to/live/read-only/models/repository
/// VOLLEY_PERSON_DETECTOR=rfdetr-medium
/// VOLLEY_SHARED_PRIMARY_BACKEND=auto
/// ORT_DYLIB_PATH=/path/to/libonnxruntime.so
///
/// The match video is opened read-only; FFmpeg creates an isolated six-second
/// keyframe-aligned sample in /tmp; all GPU evidence is disposable.
#[test]
#[ignore = "requires real user's uu video and local GPU/ONNX Runtime; isolated read-only source"]
fn real_uu_match_gpu_ab_legacy_vs_shared() -> Result<(), String> {
    let original = PathBuf::from(
        std::env::var("VOLLEY_AB_SOURCE").map_err(|_| "Set VOLLEY_AB_SOURCE explicitly")?,
    );
    let repository = PathBuf::from(
        std::env::var("VOLLEY_AB_MODEL_ROOT").map_err(|_| "Set VOLLEY_AB_MODEL_ROOT explicitly")?,
    );
    if !original.is_file() || !repository.join("models/rf-detr-medium.onnx").is_file() {
        return Err("Original match or read-only model directory missing".into());
    }
    if std::env::var("VOLLEY_PERSON_DETECTOR").as_deref() != Ok("rfdetr-medium") {
        return Err("A/B GPU test requires RF-DETR Medium model selection".into());
    }
    let base = std::env::temp_dir().join(format!("volley-uu-gpu-ab-{}", Uuid::new_v4().simple()));
    fs::create_dir_all(&base).map_err(|e| e.to_string())?;
    let _cleanup = IsolatedGpuFixture(base.clone());
    let data = base.join("sandbox");
    let uploads = data.join("uploads/project_sources");
    let project = data.join("projects/012345abcdef");
    fs::create_dir_all(&uploads).map_err(|e| e.to_string())?;
    fs::create_dir_all(&project).map_err(|e| e.to_string())?;
    let video = "uu-six-seconds.mp4";
    let source = uploads.join(video);
    let full_match = std::env::var("VOLLEY_AB_FULL_MATCH").as_deref() == Ok("1");
    let duration = if full_match {
        fs::copy(&original, &source)
            .map_err(|e| format!("Copy isolated full-match fixture: {e}"))?;
        println!("AB FULL MATCH: source copy in private /tmp, user original read-only");
        58.166667
    } else {
        let extraction = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-ss",
                "28",
                "-i",
            ])
            .arg(&original)
            .args([
                "-t",
                "6",
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-c:v",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                "-y",
            ])
            .arg(&source)
            .status()
            .map_err(|e| format!("Sample ffmpeg: {e}"))?;
        if !extraction.success() {
            return Err(format!("Cannot extract the user's uu match: {extraction}"));
        }
        6.0
    };
    crate::storage::write_json_atomic(
        &data.join("projects/projects.json"),
        &json!({"schema":1,"projects":[{"id":"012345abcdef",
            "name":"GPU A/B isolated uu","videos":[video]}]}),
    )
    .map_err(|e| format!("Seed isolated project: {e:?}"))?;
    let source_info = volley_media::preparation::source_provenance(&source)?;
    let identity = crate::inference_stream::source_stamp(&source)?;
    let app = state(repository, data);
    let abort = AtomicBool::new(false);
    use sha2::{Digest, Sha256};
    use volley_media::raw_decoder::{
        decode_full_video, decode_full_video_shared_with_backend, DecodePreference, ModelPixels,
    };
    let mut cpu_rgb_hashes = Vec::new();
    let mut cpu_gray_hashes = Vec::new();
    decode_full_video(&source, ModelPixels::PlayersRgb, 9, &abort, |window| {
        cpu_rgb_hashes.push(Sha256::digest(&window.data).to_vec());
        Ok(())
    })?;
    decode_full_video(&source, ModelPixels::BallGray, 9, &abort, |window| {
        cpu_gray_hashes.push(Sha256::digest(&window.data).to_vec());
        Ok(())
    })?;
    let mut rgb_mismatch = Vec::new();
    let mut gray_mismatch = Vec::new();
    let mut n_windows = 0usize;
    let pixel_summary = decode_full_video_shared_with_backend(
        &source,
        9,
        DecodePreference::Auto,
        &abort,
        |window| {
            if cpu_rgb_hashes.get(n_windows) != Some(&Sha256::digest(&window.rgb).to_vec()) {
                rgb_mismatch.push(window.first_frame);
            }
            if cpu_gray_hashes.get(n_windows) != Some(&Sha256::digest(&window.gray).to_vec()) {
                gray_mismatch.push(window.first_frame);
            }
            n_windows += 1;
            Ok(())
        },
    )?;
    println!(
        "AB PIXEL: backend={:?} windows={} rgb_mismatch={:?} gray_mismatch={:?}",
        pixel_summary.backend, n_windows, rgb_mismatch, gray_mismatch
    );
    assert!(
        rgb_mismatch.is_empty() && gray_mismatch.is_empty(),
        "NVDEC and CPU must produce identical model input pixels"
    );
    let memory = 6144u64;
    let threads = 4usize;
    let stride = 10usize;
    let config = |model| Config {
        model,
        device: Device::Cuda,
        gpu_device_id: 0,
        gpu_memory_limit_mb: memory,
        cpu_threads: threads,
    };
    let ball = super::new_context(
        &app,
        "012345abcdef",
        video,
        &source,
        &source_info,
        duration,
        None,
        config(ModelKind::Ball),
        volley_inference::person::DetectorVariant::RfDetrMedium,
        1,
    )?;
    let people = super::new_context(
        &app,
        "012345abcdef",
        video,
        &source,
        &source_info,
        duration,
        None,
        config(ModelKind::Players),
        volley_inference::person::DetectorVariant::RfDetrMedium,
        stride,
    )?;
    let start = Instant::now();
    let old_ball = super::run_model(&ball)?;
    let old_people = super::run_model(&people)?;
    assert_eq!(old_ball.decoded_frames, old_people.decoded_frames);
    let old_fusion = fusion_pair(&app, video, &ball.id, &people.id)?;
    let baseline_pose = crate::pose_inference::process(
        &app,
        "012345abcdef",
        video,
        old_fusion["fusion_id"]
            .as_str()
            .ok_or("Old fusion ID missing")?,
        memory,
        threads,
        &abort,
        |_, _, _, _| Ok(()),
    )?;
    let old_time = start.elapsed().as_secs_f64();
    println!(
        "AB LEGACY seconds={old_time:.3} frames={} windows={} people={} pose_samples={}",
        old_ball.decoded_frames,
        old_ball.windows,
        old_fusion["player_proposals"],
        baseline_pose["sampled_frames"]
    );

    let start = Instant::now();
    let joint = super::run_joint_pair(
        &app,
        "012345abcdef",
        video,
        &source,
        duration,
        None,
        stride,
        true,
        volley_inference::person::DetectorVariant::RfDetrMedium,
        memory,
        threads,
        super::PrimaryRallySettings::default(),
        Default::default(),
        None,
        &abort,
        |_, _, _, _| Ok(()),
    )?;
    let shared_fusion = fusion_pair(&app, video, &joint.ball_job_id, &joint.player_job_id)?;
    let shared_pose = joint
        .pending_pose
        .ok_or("Joint GPU failed to produce pose evidence")?
        .publish(&shared_fusion, &abort)?;
    let shared_time = start.elapsed().as_secs_f64();
    println!("AB SHARED seconds={shared_time:.3} backend={:?} frames={} windows={} people={} pose_samples={}",
        joint.backend,joint.decoded_frames,joint.windows,
        shared_fusion["player_proposals"],shared_pose["sampled_frames"]);
    println!(
        "AB SPEEDUP x{:.3}; source={} bytes; mtime_ns={}",
        old_time / shared_time,
        identity.0,
        identity.1
    );
    println!(
        "AB BALL sha_old={} sha_new={}",
        model_hash(&app.data_root, &ball.id)?,
        model_hash(&app.data_root, &joint.ball_job_id)?
    );
    println!(
        "AB PLAYERS sha_old={} sha_new={}",
        model_hash(&app.data_root, &people.id)?,
        model_hash(&app.data_root, &joint.player_job_id)?
    );

    if old_fusion["emitted_frames"] != shared_fusion["emitted_frames"]
        || old_fusion["decoded_frames"] != shared_fusion["decoded_frames"]
    {
        return Err("A/B fusion frame sequence differs".into());
    }
    if old_fusion["source_identity"] != shared_fusion["source_identity"]
        || shared_fusion["source_identity"] != json!({"bytes":identity.0,"mtime_ns":identity.1})
    {
        return Err("A/B source identity differs".into());
    }
    if shared_pose["source_identity"] != shared_fusion["source_identity"]
        || shared_pose["original_source"] != shared_fusion["original_source"]
    {
        return Err("Shared pose metadata do not match fusion identity".into());
    }
    let old_samples = pose_samples(&app, &baseline_pose)?;
    let new_samples = pose_samples(&app, &shared_pose)?;
    println!(
        "AB POSE sampled={}; first={:?}; last={:?}",
        new_samples.len(),
        new_samples.first(),
        new_samples.last()
    );
    if old_samples != new_samples {
        let mismatch = old_samples.iter().zip(&new_samples).find(|(a, b)| a != b);
        return Err(format!(
            "A/B pose indices/counts differ: old={} new={} first={mismatch:?}",
            old_samples.len(),
            new_samples.len()
        ));
    }
    if model_hash(&app.data_root, &ball.id)? != model_hash(&app.data_root, &joint.ball_job_id)? {
        return Err("A/B ball detector raw evidence SHA differs".into());
    }
    if model_hash(&app.data_root, &people.id)? != model_hash(&app.data_root, &joint.player_job_id)?
    {
        return Err("A/B player detector raw evidence SHA differs".into());
    }

    if baseline_pose["sha256"] != shared_pose["sha256"] {
        let read_rows = |id: &str| -> Result<Vec<Value>, String> {
            let file = app
                .data_root
                .join("projects/012345abcdef/native_gpu_pose_runs")
                .join(id)
                .join("poses.jsonl");
            fs::read_to_string(file)
                .map_err(|e| e.to_string())?
                .lines()
                .map(|row| serde_json::from_str::<Value>(row).map_err(|e| e.to_string()))
                .collect()
        };
        let old_rows = read_rows(baseline_pose["pose_run_id"].as_str().ok_or("old pose id")?)?;
        let new_rows = read_rows(shared_pose["pose_run_id"].as_str().ok_or("new pose id")?)?;
        let mut mismatched_rows = 0usize;
        let mut mismatched_boxes = 0usize;
        let mut mismatched_kpts = 0usize;
        let mut max_abs = 0f64;
        let mut max_xy = 0f64;
        let mut max_confidence = 0f64;
        let mut first_difference = None;
        for (a, b) in old_rows.iter().zip(&new_rows) {
            if a == b {
                continue;
            }
            mismatched_rows += 1;
            if first_difference.is_none() {
                first_difference = Some((a["frame"].clone(), a.clone(), b.clone()));
            }
            let a_people = a["players"].as_array().ok_or("Old pose players missing")?;
            let b_people = b["players"].as_array().ok_or("New pose players missing")?;
            for (ap, bp) in a_people.iter().zip(b_people) {
                if ap["box"] != bp["box"] {
                    mismatched_boxes += 1;
                }
                let aa = ap["keypoints"].as_array().ok_or("Missing old keypoints")?;
                let bb = bp["keypoints"].as_array().ok_or("Missing new keypoints")?;
                for (kp1, kp2) in aa.iter().zip(bb) {
                    if kp1 != kp2 {
                        mismatched_kpts += 1;
                    }
                    for (index, (n1, n2)) in kp1
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(kp2.as_array().unwrap())
                        .enumerate()
                    {
                        if let (Some(x), Some(y)) = (n1.as_f64(), n2.as_f64()) {
                            let delta = (x - y).abs();
                            max_abs = max_abs.max(delta);
                            if index == 2 {
                                max_confidence = max_confidence.max(delta);
                            } else {
                                max_xy = max_xy.max(delta);
                            }
                        }
                    }
                }
            }
        }
        println!("AB POSE mismatch rows={} boxes={} keypoints={} max_abs={:.12} max_xy={:.12} max_confidence={:.12} total_rows={}",
            mismatched_rows,mismatched_boxes,mismatched_kpts,max_abs,max_xy,max_confidence,old_rows.len());
        if let Some((frame, a, b)) = first_difference {
            println!(
                "AB FIRST mismatch frame={} old_box={:?} new_box={:?}",
                frame, a["players"][0]["box"], b["players"][0]["box"]
            );
            println!(
                "AB FIRST mismatch old_point={:?} new_point={:?}",
                a["players"][0]["keypoints"][0], b["players"][0]["keypoints"][0]
            );
        }
        if old_rows.len() != new_rows.len()
            || mismatched_boxes != 0
            || max_xy > 0.00001
            || max_confidence > 0.001
        {
            return Err(format!(
                "A/B pose numeric guard exceeded: xy={max_xy}, confidence={max_confidence}"
            ));
        }
        println!("AB POSE numerically equivalent: coordinate tolerance <= 0.00001, confidence <= 0.001; evidence SHA differs");
    }
    let cpu_old = replay_touches_and_rallies(&app, video, &old_fusion)?;
    let cpu_new = replay_touches_and_rallies(&app, video, &shared_fusion)?;
    let fields = [
        ("track_frames", &cpu_old.0["frames"], &cpu_new.0["frames"]),
        (
            "tracks",
            &cpu_old.0["provisional_track_count"],
            &cpu_new.0["provisional_track_count"],
        ),
        (
            "pose_measured",
            &cpu_old.0["pose_measured_track_proposals"],
            &cpu_new.0["pose_measured_track_proposals"],
        ),
        (
            "touch_candidates",
            &cpu_old.1["touch_candidates"],
            &cpu_new.1["touch_candidates"],
        ),
        (
            "touch_motion",
            &cpu_old.1["motion_supported_proposals_before_dedup"],
            &cpu_new.1["motion_supported_proposals_before_dedup"],
        ),
        (
            "rally_candidates",
            &cpu_old.2["rally_candidates"],
            &cpu_new.2["rally_candidates"],
        ),
        (
            "rally_ingested_touches",
            &cpu_old.2["touch_candidates_ingested"],
            &cpu_new.2["touch_candidates_ingested"],
        ),
    ];
    for (label, previous, current) in fields {
        println!("AB CPU {label}: legacy={} shared={}", previous, current);
        if previous != current {
            return Err(format!(
                "CPU A/B mismatch in {label}: legacy={previous}, shared={current}"
            ));
        }
    }
    println!(
        "AB CPU touch_evidence_sha: legacy={} shared={}",
        cpu_old.1["sha256"], cpu_new.1["sha256"]
    );
    if cpu_old.1["sha256"] != cpu_new.1["sha256"] {
        return Err("CPU A/B touch proposal JSONL differs".into());
    }
    let rally_rows = |run: &Value| -> Result<Vec<Value>, String> {
        let id = run["rally_run_id"].as_str().ok_or("Missing rally ID")?;
        let file = app
            .data_root
            .join("projects/012345abcdef/native_cpu_rallies")
            .join(id)
            .join("rally_candidates.jsonl");
        let mut rows = Vec::new();
        for line in fs::read_to_string(&file)
            .map_err(|e| format!("Read rally candidates {}: {e}", file.display()))?
            .lines()
        {
            let mut row: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
            if let Some(object) = row.as_object_mut() {
                // candidate_id is SHA(replay_id:touch_id:start:end), and
                // thus intentionally differs between immutable runs.
                object.remove("candidate_id");
            }
            rows.push(row);
        }
        Ok(rows)
    };
    let old_rallies = rally_rows(&cpu_old.2)?;
    let new_rallies = rally_rows(&cpu_new.2)?;
    println!(
        "AB CPU rally rows: old={} new={} raw_sha_equal={}",
        old_rallies.len(),
        new_rallies.len(),
        cpu_old.2["sha256"] == cpu_new.2["sha256"]
    );
    if old_rallies != new_rallies {
        let differing = old_rallies
            .iter()
            .zip(&new_rallies)
            .position(|(a, b)| a != b);
        return Err(format!(
            "CPU A/B rally temporal/content mismatch at {differing:?}; candidate_id excluded"
        ));
    }
    println!(
        "AB PASS: exact input pixels and ball/person evidence; pose within strict numerical tolerance; CPU tracking, touches and rallies agree"
    );
    Ok(())
}

type PoseBenchmarkRun = (f64, u64, Value, String, String, Vec<Value>, Vec<Value>);

#[test]
#[ignore = "requires CUDA models, isolated sample and explicit benchmark inputs"]
fn original_pose_single_decode_benchmark() -> Result<(), String> {
    let original = PathBuf::from(std::env::var("VOLLEY_AB_SOURCE").map_err(|e| e.to_string())?);
    let repo = PathBuf::from(std::env::var("VOLLEY_AB_MODEL_ROOT").map_err(|e| e.to_string())?);
    let base = std::env::temp_dir().join(format!("volley-pose-perf-{}", Uuid::new_v4().simple()));
    fs::create_dir_all(&base).map_err(|e| e.to_string())?;
    let _cleanup = IsolatedGpuFixture(base.clone());
    let data = base.join("sandbox");
    let uploads = data.join("uploads/project_sources");
    fs::create_dir_all(&uploads).map_err(|e| e.to_string())?;
    fs::create_dir_all(data.join("projects/012345abcdef")).map_err(|e| e.to_string())?;
    let video = "bench.mp4";
    let source = uploads.join(video);
    let sample_start = std::env::var("VOLLEY_AB_SAMPLE_START_SEC").unwrap_or_else(|_| "28".into());
    let sample_duration =
        std::env::var("VOLLEY_AB_SAMPLE_DURATION_SEC").unwrap_or_else(|_| "4".into());
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-ss"])
        .arg(&sample_start)
        .arg("-i")
        .arg(&original)
        .arg("-t")
        .arg(&sample_duration)
        .args(["-map", "0:v:0", "-an", "-c:v", "copy", "-y"])
        .arg(&source)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("Cannot create isolated benchmark sample".into());
    }
    crate::storage::write_json_atomic(
        &data.join("projects/projects.json"),
        &json!({"schema":1,"projects":[{"id":"012345abcdef","name":"Pose perf","videos":[video]}]}),
    )
    .map_err(|e| format!("Create benchmark project: {e:?}"))?;
    let app = state(repo, data);
    let abort = AtomicBool::new(false);
    let info = volley_media::probe(&source)?;
    let run = |shared_pose: bool| -> Result<PoseBenchmarkRun, String> {
        let started = Instant::now();
        let pair = super::run_joint_pair(
            &app,
            "012345abcdef",
            video,
            &source,
            info.duration_sec,
            None,
            1,
            shared_pose,
            volley_inference::person::DetectorVariant::RfDetrMedium,
            0,
            4,
            super::PrimaryRallySettings::default(),
            Default::default(),
            None,
            &abort,
            |_, _, _, _| Ok(()),
        )?;
        let gpu_sec = started.elapsed().as_secs_f64();
        if std::env::var("VOLLEY_ZERO_COPY_PRIMARY").as_deref() == Ok("1") && !pair.zero_copy {
            return Err(format!(
                "GPU A/B requires nvdec_zero_copy, fallback={:?}",
                pair.fallback_reason
            ));
        }
        let ball_sha = model_hash(&app.data_root, &pair.ball_job_id)?;
        let person_sha = model_hash(&app.data_root, &pair.player_job_id)?;
        let resolved = crate::projects::canonical_source(&app, video)
            .map_err(|error| format!("Benchmark video lookup: {error:?}"))?;
        let recorded = manifest(&app.data_root, "native_gpu_runs", &pair.ball_job_id)?
            ["original_source"]
            .clone();
        let canonical_proof = volley_media::preparation::source_provenance(&resolved)?;
        if recorded != canonical_proof {
            return Err(format!(
                "GPU A/B original differs: requested={} resolved={} recorded={} actual={}",
                source.display(),
                resolved.display(),
                recorded,
                canonical_proof
            ));
        }
        let fusion = fusion_pair(&app, video, &pair.ball_job_id, &pair.player_job_id)?;
        let pose = if shared_pose {
            pair.pending_pose
                .ok_or("Missing shared original pose")?
                .publish(&fusion, &abort)?
        } else {
            crate::pose_inference::process(
                &app,
                "012345abcdef",
                video,
                fusion["fusion_id"].as_str().ok_or("Missing fusion id")?,
                0,
                4,
                &abort,
                |_, _, _, _| Ok(()),
            )?
        };
        let elapsed = started.elapsed().as_secs_f64();
        let pose_id = pose["pose_run_id"].as_str().ok_or("Missing pose id")?;
        let file = app
            .data_root
            .join("projects/012345abcdef/native_gpu_pose_runs")
            .join(pose_id)
            .join("poses.jsonl");
        let content = fs::read_to_string(&file).map_err(|e| e.to_string())?;
        let rows = content
            .lines()
            .map(|line| serde_json::from_str(line).map_err(|e| e.to_string()))
            .collect::<Result<Vec<Value>, String>>()?;
        let ball_rows = fs::read_to_string(
            app.data_root
                .join("projects/012345abcdef/native_gpu_runs")
                .join(&pair.ball_job_id)
                .join("detections.jsonl"),
        )
        .map_err(|e| e.to_string())?
        .lines()
        .map(|line| serde_json::from_str(line).map_err(|e| e.to_string()))
        .collect::<Result<Vec<Value>, String>>()?;
        println!("POSE_BENCH {} parallel={} frames={} gpu_seconds={gpu_sec:.3} total_seconds={elapsed:.3} poses={} samples={} pose_sha={} ball_sha={} person_sha={}",
            if shared_pose {"single"} else {"baseline"},
            std::env::var("VOLLEY_PARALLEL_POSE").unwrap_or_else(|_| "default".into()),
            pair.decoded_frames,pose["player_poses"],pose["sampled_frames"],pose["sha256"],ball_sha,person_sha);
        Ok((
            elapsed,
            pair.decoded_frames,
            pose,
            ball_sha,
            person_sha,
            rows,
            ball_rows,
        ))
    };
    match std::env::var("VOLLEY_AB_VARIANT").as_deref() {
        Ok(
            "compare"
            | "reverse"
            | "pipeline"
            | "pipeline_reverse"
            | "pipeline_repeat_inline"
            | "pipeline_warmed"
            | "pipeline_warmed_reverse",
        ) => {
            let (baseline, shared) = match std::env::var("VOLLEY_AB_VARIANT").as_deref() {
                Ok("pipeline_warmed") | Ok("pipeline_warmed_reverse") => {
                    let reverse = std::env::var("VOLLEY_AB_VARIANT").as_deref()
                        == Ok("pipeline_warmed_reverse");
                    std::env::set_var("VOLLEY_PARALLEL_POSE", "0");
                    let _ = run(true)?;
                    std::env::set_var("VOLLEY_PARALLEL_POSE", "1");
                    let _ = run(true)?;
                    if reverse {
                        std::env::set_var("VOLLEY_PARALLEL_POSE", "1");
                        let shared = run(true)?;
                        std::env::set_var("VOLLEY_PARALLEL_POSE", "0");
                        (run(true)?, shared)
                    } else {
                        std::env::set_var("VOLLEY_PARALLEL_POSE", "0");
                        let baseline = run(true)?;
                        std::env::set_var("VOLLEY_PARALLEL_POSE", "1");
                        (baseline, run(true)?)
                    }
                }
                Ok("pipeline_repeat_inline") => {
                    std::env::set_var("VOLLEY_PARALLEL_POSE", "0");
                    (run(true)?, run(true)?)
                }
                Ok("pipeline") => {
                    std::env::set_var("VOLLEY_PARALLEL_POSE", "0");
                    let baseline = run(true)?;
                    std::env::set_var("VOLLEY_PARALLEL_POSE", "1");
                    (baseline, run(true)?)
                }
                Ok("pipeline_reverse") => {
                    std::env::set_var("VOLLEY_PARALLEL_POSE", "1");
                    let shared = run(true)?;
                    std::env::set_var("VOLLEY_PARALLEL_POSE", "0");
                    (run(true)?, shared)
                }
                Ok("reverse") => {
                    let shared = run(true)?;
                    (run(false)?, shared)
                }
                _ => (run(false)?, run(true)?),
            };
            if baseline.3 != shared.3 {
                let first = baseline
                    .6
                    .iter()
                    .zip(&shared.6)
                    .enumerate()
                    .find(|(_, (old, new))| old != new);
                let changed = baseline
                    .6
                    .iter()
                    .zip(&shared.6)
                    .filter(|(old, new)| old != new)
                    .count();
                eprintln!(
                    "BALL_AB mismatch: rows_old={} rows_new={} changed={} first={:?}",
                    baseline.6.len(),
                    shared.6.len(),
                    changed,
                    first.map(|(index, (old, new))| (index, old, new))
                );
            }
            if baseline.1 != shared.1 || baseline.3 != shared.3 || baseline.4 != shared.4 {
                return Err("GPU detection frame counts or SHA changed with native pose".into());
            }
            if matches!(
                std::env::var("VOLLEY_AB_VARIANT").as_deref(),
                Ok("pipeline" | "pipeline_reverse" | "pipeline_warmed" | "pipeline_warmed_reverse")
            ) && baseline.2["player_poses"].as_u64().unwrap_or(0) == 0
            {
                return Err(
                    "Parallel pose comparison requires actual players and pose crops".into(),
                );
            }
            if baseline.2["sampled_frames"] != shared.2["sampled_frames"]
                || baseline.2["player_poses"] != shared.2["player_poses"]
            {
                return Err("Pose sample or player count differs from reference".into());
            }
            if baseline.5.len() != shared.5.len() {
                return Err("Pose row counts differ".into());
            }
            let mut row_mismatches = 0usize;
            let mut box_mismatches = 0usize;
            let mut keypoint_mismatches = 0usize;
            let mut maximum_xy = 0.0f64;
            let mut maximum_score = 0.0f64;
            for (old, new) in baseline.5.iter().zip(&shared.5) {
                if old == new {
                    continue;
                }
                row_mismatches += 1;
                if old["frame"] != new["frame"] {
                    return Err("Pose frame indices differ".into());
                }
                let old_people = old["players"].as_array().ok_or("Bad reference pose row")?;
                let new_people = new["players"].as_array().ok_or("Bad streamed pose row")?;
                if old_people.len() != new_people.len() {
                    return Err("Pose player counts differ".into());
                }
                for (a, b) in old_people.iter().zip(new_people) {
                    if a["box"] != b["box"] || a["crop_origin"] != b["crop_origin"] {
                        box_mismatches += 1;
                    }
                    let old_points = a["keypoints"]
                        .as_array()
                        .ok_or("Missing reference keypoints")?;
                    let new_points = b["keypoints"]
                        .as_array()
                        .ok_or("Missing streamed keypoints")?;
                    if old_points.len() != new_points.len() {
                        return Err("Keypoint counts differ".into());
                    }
                    for (oa, nb) in old_points.iter().zip(new_points) {
                        if oa != nb {
                            keypoint_mismatches += 1;
                        }
                        let aa = oa.as_array().ok_or("Invalid reference keypoint")?;
                        let bb = nb.as_array().ok_or("Invalid streamed keypoint")?;
                        for index in 0..3 {
                            let delta = (aa[index].as_f64().ok_or("Invalid reference value")?
                                - bb[index].as_f64().ok_or("Invalid streamed value")?)
                            .abs();
                            if index == 2 {
                                maximum_score = maximum_score.max(delta);
                            } else {
                                maximum_xy = maximum_xy.max(delta);
                            }
                        }
                    }
                }
            }
            println!("POSE_PARITY rows={} box_mismatches={} keypoint_mismatches={} max_xy={maximum_xy:.12} max_score={maximum_score:.12}",
                row_mismatches, box_mismatches, keypoint_mismatches);
            println!(
                "POSE_SPEEDUP old={:.3}s new={:.3}s speedup={:.3}x reduction={:.2}%",
                baseline.0,
                shared.0,
                baseline.0 / shared.0,
                100.0 * (1.0 - shared.0 / baseline.0)
            );
            if box_mismatches != 0 || maximum_xy > 0.00001 || maximum_score > 0.001 {
                return Err("Pose parity exceeded source-coordinate tolerances".into());
            }
        }
        Ok("single") => {
            run(true)?;
        }
        _ => {
            run(false)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires local CUDA ONNX models and synthetic source; isolated GPU pipeline"]
fn primary_rally_prefilter_cuda_integration_smoke() -> Result<(), String> {
    let original = PathBuf::from(std::env::var("VOLLEY_AB_SOURCE").map_err(|e| e.to_string())?);
    let repository =
        PathBuf::from(std::env::var("VOLLEY_AB_MODEL_ROOT").map_err(|e| e.to_string())?);
    if !original.is_file() || !repository.join("models/rf-detr-medium.onnx").is_file() {
        return Err("GPU smoke source or model unavailable".into());
    }
    let base = std::env::temp_dir().join(format!(
        "volley-primary-mask-smoke-{}",
        Uuid::new_v4().simple()
    ));
    fs::create_dir_all(&base).map_err(|e| e.to_string())?;
    let _cleanup = IsolatedGpuFixture(base.clone());
    let data = base.join("sandbox");
    let uploads = data.join("uploads/project_sources");
    fs::create_dir_all(&uploads).map_err(|e| e.to_string())?;
    fs::create_dir_all(data.join("projects/012345abcdef")).map_err(|e| e.to_string())?;
    let video = "synthetic.mp4";
    let source = uploads.join(video);
    let start = std::env::var("VOLLEY_AB_SAMPLE_START_SEC").unwrap_or_else(|_| "28".into());
    let seconds = std::env::var("VOLLEY_AB_SAMPLE_DURATION_SEC").unwrap_or_else(|_| "6".into());
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-ss"])
        .arg(start)
        .arg("-i")
        .arg(&original)
        .arg("-t")
        .arg(seconds)
        .args(["-map", "0:v:0", "-an", "-c:v", "copy", "-y"])
        .arg(&source)
        .status()
        .map_err(|e| e.to_string())?;
    if !output.success() {
        return Err("Could not create isolated GPU smoke sample".into());
    }
    crate::storage::write_json_atomic(
        &data.join("projects/projects.json"),
        &json!({"schema":1,"projects":[{"id":"012345abcdef",
            "name":"Primary GPU rally mask smoke","videos":[video]}]}),
    )
    .map_err(|e| format!("Seed isolated GPU smoke project: {e:?}"))?;
    let app = state(repository, data);
    let abort = AtomicBool::new(false);
    let info = volley_media::probe(&source)?;
    let analysis_range = std::env::var("VOLLEY_AB_ANALYSIS_TRIM_START")
        .ok()
        .map(|value| {
            let start_sec = value
                .parse::<f64>()
                .map_err(|_| "Invalid GPU smoke trim start")?;
            let interval = volley_media::raw_decoder::AnalysisRange {
                start_sec,
                end_sec: info.duration_sec,
            };
            interval.validate(info.duration_sec)?;
            Ok::<_, String>(interval)
        })
        .transpose()?;
    let analysis_duration =
        analysis_range.map_or(info.duration_sec, |value| value.end_sec - value.start_sec);
    let person_stride = std::env::var("VOLLEY_AB_PERSON_STRIDE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    let detector = volley_inference::person::DetectorVariant::from_env()?;
    let started = Instant::now();
    let pair = super::run_joint_pair(
        &app,
        "012345abcdef",
        video,
        &source,
        analysis_duration,
        analysis_range,
        person_stride,
        true,
        detector,
        0,
        4,
        super::PrimaryRallySettings::default(),
        Default::default(),
        None,
        &abort,
        |_, _, _, _| Ok(()),
    )?;
    let evidence = manifest(&app.data_root, "native_gpu_runs", &pair.player_job_id)?;
    if !evidence["person_pipeline"]
        .as_str()
        .is_some_and(|pipeline| pipeline.starts_with(&detector.id().replace('-', "_")))
    {
        return Err("Player manifest does not match selected RF-DETR variant".into());
    }
    if std::env::var("VOLLEY_ZERO_COPY_PRIMARY").as_deref() == Ok("1") {
        if !pair.zero_copy || pair.backend != DecodeBackend::Nvdec {
            return Err(format!(
                "Zero-copy was requested but primary scan fell back: {:?}",
                pair.fallback_reason
            ));
        }
        for job in [&pair.ball_job_id, &pair.player_job_id] {
            let evidence = manifest(&app.data_root, "native_gpu_runs", job)?;
            if evidence["decode_backend"] != "nvdec_zero_copy"
                || evidence["decode_pipeline"] != "gpu_resident_nvdec_primary_v1"
            {
                return Err("Primary GPU evidence was not produced by CUDA zero-copy".into());
            }
        }
    }
    let mask = pair
        .rally_filter
        .as_ref()
        .ok_or("Primary GPU scan did not publish a rally mask")?;
    if mask["schema"] != "volley-primary-rally-mask/1" {
        return Err("Primary GPU mask schema mismatch".into());
    }
    println!(
        "RALLY MASK: ball_frames={} detected={} uncertain={} fallback={}",
        mask["observed_ball_frames"],
        mask["detected"].as_array().map_or(0, Vec::len),
        mask["uncertain"].as_array().map_or(0, Vec::len),
        mask["fallback_reason"]
    );
    let selected = mask["selected_frames"]
        .as_u64()
        .ok_or("Primary GPU mask has no selected frame count")?;
    if selected == 0 || selected > pair.decoded_frames {
        return Err("Primary GPU mask selected-frame count is out of bounds".into());
    }
    if std::env::var("VOLLEY_AB_EXPECT_SPARSE").as_deref() == Ok("1") {
        let mask = pair
            .rally_filter
            .as_ref()
            .ok_or("Sparse GPU check requires a rally filter")?;
        if selected >= pair.decoded_frames || mask["selected"].as_array().map_or(0, Vec::len) < 2 {
            return Err("GPU did not skip separate non-game intervals".into());
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "PRIMARY GPU MASK: zero_copy={} person_stride={} frames={} selected={} windows={} elapsed={elapsed:.3}s",
        pair.zero_copy, person_stride, pair.decoded_frames, selected, pair.windows
    );
    let ball_sha = model_hash(&app.data_root, &pair.ball_job_id)?;
    println!("PRIMARY GPU BALL SHA256: {ball_sha}");
    if let Ok(expected) = std::env::var("VOLLEY_AB_EXPECT_BALL_SHA256") {
        if ball_sha != expected {
            return Err("Filtered VBallNet evidence differs from the reference run".into());
        }
    }
    let fusion = fusion_pair(&app, video, &pair.ball_job_id, &pair.player_job_id)?;
    let pose = pair
        .pending_pose
        .ok_or("GPU smoke did not produce RTMPose evidence")?
        .publish(&fusion, &abort)?;
    if pose["sampled_frames"].as_u64()
        != Some((pair.windows as u64 * 9).div_ceil(person_stride as u64))
    {
        return Err("GPU smoke pose sampling disagrees with source timeline".into());
    }
    let (replay, touches, rallies) = replay_touches_and_rallies(&app, video, &fusion)?;
    println!(
        "PRIMARY GPU -> CPU: zero_copy={} frames={} players={} poses={} replay={} touches={} rallies={}",
        pair.zero_copy,
        pair.decoded_frames,
        fusion["player_proposals"],
        pose["player_poses"],
        replay["frames"],
        touches["touch_candidates"],
        rallies["rallies"]
    );
    Ok(())
}

#[test]
#[ignore = "requires Windows CUDA/ORT and repository example video"]
fn phase1_checkpoint_cuda_resume() -> Result<(), String> {
    let repository =
        PathBuf::from(std::env::var("VOLLEY_AB_MODEL_ROOT").map_err(|e| e.to_string())?);
    let original = PathBuf::from(std::env::var("VOLLEY_AB_SOURCE").map_err(|e| e.to_string())?);
    let base = std::env::temp_dir().join(format!("volley-resume-gpu-{}", Uuid::new_v4().simple()));
    fs::create_dir(&base).map_err(|e| e.to_string())?;
    let _cleanup = IsolatedGpuFixture(base.clone());
    let data = base.join("sandbox");
    let uploads = data.join("uploads/project_sources");
    let project = data.join("projects/012345abcdef");
    let checkpoints = project.join("native_studio_pipelines");
    fs::create_dir_all(&uploads).map_err(|e| e.to_string())?;
    fs::create_dir_all(&checkpoints).map_err(|e| e.to_string())?;
    let video = "resume-example.mp4";
    let source = uploads.join(video);
    let status = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(original)
        .args([
            "-vf", "fps=30", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y",
        ])
        .arg(&source)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("Cannot create GPU example fixture".into());
    }
    crate::storage::write_json_atomic(&data.join("projects/projects.json"),
        &json!({"schema":1,"projects":[{"id":"012345abcdef","name":"Resume smoke","videos":[video]}]}))
        .map_err(|e| format!("{e:?}"))?;
    let duration = volley_media::probe(&source)?.duration_sec;
    let app = state(repository, data);
    let run = |interrupt: bool| -> Result<(Result<JointPair, String>, Value), String> {
        let cancel = Arc::new(AtomicBool::new(false));
        let progress = Arc::new(StdMutex::new(RallyStageProgress::default()));
        let result = run_joint_pair(
            &app,
            "012345abcdef",
            video,
            &source,
            duration,
            None,
            3,
            false,
            volley_inference::person::DetectorVariant::RfDetrMedium,
            0,
            4,
            PrimaryRallySettings {
                min_trajectory_seconds: 1.0,
                ..Default::default()
            },
            Default::default(),
            Some(EarlyPoseOptions {
                independent: None,
                court: None,
                sampling: Default::default(),
                tracking: Default::default(),
                interpolate_players: true,
                threads: 4,
                cancelled: cancel.clone(),
                progress: progress.clone(),
                reid_crops: None,
                checkpoint_root: Some(checkpoints.clone()),
            }),
            &cancel,
            |_, _, _, _| {
                let snapshot = serde_json::to_value(&*progress.lock().map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
                if interrupt && snapshot["pose"]["rallies_completed"].as_u64().unwrap_or(0) > 0 {
                    cancel.store(true, Ordering::Release);
                    return Err("cancelled".into());
                }
                Ok(())
            },
        );
        let snapshot = serde_json::to_value(&*progress.lock().map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        Ok((result, snapshot))
    };
    let (first, before) = run(true)?;
    println!("Interrupted: {before}");
    if first.err().as_deref() != Some("cancelled")
        || before["pose"]["rallies_completed"].as_u64().unwrap_or(0) == 0
    {
        return Err("GPU example did not reach a committed rally before interruption".into());
    }
    let (resumed, after) = run(false)?;
    let resumed = resumed?;
    println!("Resumed: {after}");
    if after["rallies_recovered"].as_u64().unwrap_or(0)
        < before["pose"]["rallies_completed"].as_u64().unwrap_or(0)
        || after["pose"]["rallies_completed"] != after["pose"]["rallies_total"]
        || resumed.early_poses.is_none()
    {
        return Err("Resume did not restore all committed rallies".into());
    }
    let _fusion = fusion_pair(&app, video, &resumed.ball_job_id, &resumed.player_job_id)?;
    Ok(())
}
