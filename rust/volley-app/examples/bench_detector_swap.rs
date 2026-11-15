//! Offline detector-swap experiment, not part of running Studio:
//! SAM 3.1 masks -> person boxes -> RTMPose -> existing Rust tracker.
//! SAM's native IDs are deliberately NOT used for association here.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
    process::{Command, Stdio},
    time::Instant,
};
use volley_core::{
    fusion::Candidate,
    tracker::{ProvisionalTracker, TrackingOptions},
};
use volley_inference::{person::skeleton_box, pose::PoseEngine, Device};

#[derive(Deserialize)]
struct SamFrame {
    frame: u64,
    tracks: Vec<SamTrack>,
}
#[derive(Deserialize)]
struct SamTrack {
    id: u64,
    bbox: [f64; 4],
}
fn fingerprint(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hash = Sha256::new();
    let mut chunk = [0u8; 65536];
    loop {
        let len = reader.read(&mut chunk)?;
        if len == 0 {
            break;
        }
        hash.update(&chunk[..len]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let repo = args
        .next()
        .ok_or("need source repo directory with model weights")?;
    let video = args.next().ok_or("need exact sample MP4")?;
    let sam_path = args.next().ok_or("need SAM native track output JSONL")?;
    let output = args.next().ok_or("need output JSONL")?;
    let checksum = fingerprint(Path::new(&video))?;
    let upstream = Path::new(&sam_path).with_extension("stats.json");
    let mut upstream_seconds = None;
    if upstream.is_file() {
        let meta: serde_json::Value = serde_json::from_reader(File::open(&upstream)?)?;
        if meta["video_sha256"].as_str() != Some(checksum.as_str()) {
            return Err("SAM JSONL metadata video hash differs from current video".into());
        }
        upstream_seconds = meta["inference_seconds"].as_f64();
    }
    let mut pose = PoseEngine::load(Path::new(&repo), Device::Cuda, 8, 0)?;
    let mut tracker = ProvisionalTracker::new(TrackingOptions::default())?;
    let mut sam_frames = BufReader::new(File::open(&sam_path)?).lines();
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            &video,
            "-map",
            "0:v:0",
            "-vf",
            "fps=30000/1001,scale=640:360",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let mut reader = BufReader::new(ffmpeg.stdout.take().ok_or("no ffmpeg stdout")?);
    let mut writer = BufWriter::new(File::create(&output)?);
    let mut count = 0u64;
    let mut proposals = 0u64;
    let mut pose_errors = 0u64;
    let mut missing_skeleton = 0u64;
    let started = Instant::now();
    loop {
        let mut rgb = vec![0; 640 * 360 * 3];
        match reader.read_exact(&mut rgb) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let json = sam_frames
            .next()
            .ok_or("SAM output ends before video ends")??;
        let detected: SamFrame = serde_json::from_str(&json)?;
        if detected.frame != count {
            return Err(format!("SAM frame index mismatch: {} != {count}", detected.frame).into());
        }
        let mut candidates = Vec::new();
        for (index, item) in detected.tracks.into_iter().enumerate() {
            let [x1, y1, x2, y2] = item.bbox;
            if !item
                .bbox
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                || x1 >= x2
                || y1 >= y2
            {
                continue;
            }
            proposals += 1;
            let observed_pose = pose.infer(&rgb, item.bbox);
            let (head, score) = match observed_pose {
                Ok(points) => {
                    let head = (points[0].score >= 0.32)
                        .then_some([points[0].x as f64, points[0].y as f64]);
                    match skeleton_box(&points) {
                        Some((_, pose_score)) => (head, f64::from(pose_score).max(0.30)),
                        None => {
                            missing_skeleton += 1;
                            (head, 0.50)
                        }
                    }
                }
                Err(_) => {
                    pose_errors += 1;
                    (None, 0.50)
                }
            };
            // Source ID deliberately ignored: use identical Rust tracker as baseline.
            let _source_sam_id = item.id;
            candidates.push(Candidate {
                frame_index: count,
                class_name: "player".to_owned(),
                score: score.clamp(0.0, 1.0),
                bbox_xyxy_norm: item.bbox,
                head_point_norm: head,
                query_index: Some(index),
                touch_probability: None,
            });
        }
        let tracked = tracker.process_frame(count, &candidates)?;
        let tracks = tracked
            .tracks
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "id": entry.track_id, "bbox": entry.detection.bbox_xyxy_norm,
                })
            })
            .collect::<Vec<_>>();
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({
                "frame": count, "tracks": tracks,
            }),
        )?;
        writeln!(writer)?;
        count += 1;
    }
    writer.flush()?;
    drop(reader);
    let exit = ffmpeg.wait()?;
    if !exit.success() || sam_frames.next().is_some() || count == 0 {
        return Err("video decoder error or SAM/MP4 frame mismatch".into());
    }
    let pose_seconds = started.elapsed().as_secs_f64();
    let total_seconds = upstream_seconds.map(|secs| secs + pose_seconds);
    let report = serde_json::json!({
        "model": "SAM3.1-proposals+RTMPose+ProvisionalTracker",
        "frames": count, "video_sha256": checksum, "video": video,
        "source_sam_jsonl": sam_path,
        "sam_proposals": proposals,
        "pose_without_confident_skeleton": missing_skeleton,
        "pose_errors": pose_errors,
        "pose_stage_seconds": pose_seconds,
        "sam_stage_seconds": upstream_seconds,
        "inference_seconds": total_seconds,
        "fps": total_seconds.map(|t| count as f64 / t),
        "note": "Full pipeline FPS only available with actual SAM time metadata",
    });
    std::fs::write(
        Path::new(&output).with_extension("stats.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report:#}");
    Ok(())
}
