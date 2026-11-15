//! One-shot *isolated benchmark* of the existing RTMDet+RTMPose+Rust tracker.
//! Raw RGB24 640x360, all frames, same input clip as SAM 3.1.
//! NOT a new Studio runtime.
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    process::{Command, Stdio},
    time::Instant,
};
use volley_core::{
    fusion::Candidate,
    tracker::{ProvisionalTracker, TrackingOptions},
};
use volley_inference::{person::PersonEngine, Device};

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
    let repo = args.next().ok_or("need repo directory")?;
    let input = args.next().ok_or("need sample video MP4")?;
    let output = args.next().ok_or("need .jsonl output")?;
    let mut engine = PersonEngine::load(Path::new(&repo), Device::Cuda, 8, 0)?;
    let mut tracker = ProvisionalTracker::new(TrackingOptions::default())?;
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            &input,
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
    let mut pipe = BufReader::new(ffmpeg.stdout.take().ok_or("no ffmpeg stdout")?);
    let mut writer = BufWriter::new(File::create(&output)?);
    let mut next_frame = 0u64;
    let mut total_boxes = 0u64;
    let bytes = 640 * 360 * 3;
    let mut frames_batch = Vec::with_capacity(bytes * 9);
    let started = Instant::now();
    loop {
        let mut frame = vec![0; bytes];
        match pipe.read_exact(&mut frame) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        frames_batch.extend_from_slice(&frame);
        if frames_batch.len() < bytes * 9 {
            continue;
        }
        engine.detect_frames(&frames_batch, |_, detected| {
            let candidates = detected
                .into_iter()
                .enumerate()
                .map(|(i, p)| Candidate {
                    frame_index: next_frame,
                    class_name: "player".into(),
                    score: f64::from(p.score),
                    bbox_xyxy_norm: p.bbox,
                    query_index: Some(i),
                    head_point_norm: p.head,
                    touch_probability: None,
                })
                .collect::<Vec<_>>();
            let tracked = tracker.process_frame(next_frame, &candidates)?;
            total_boxes += tracked.tracks.len() as u64;
            let tracks = tracked
                .tracks
                .iter()
                .map(|track| {
                    serde_json::json!({
                        "id": track.track_id,
                        "bbox": track.detection.bbox_xyxy_norm,
                    })
                })
                .collect::<Vec<_>>();
            serde_json::to_writer(
                &mut writer,
                &serde_json::json!({
                    "frame": next_frame,
                    "tracks": tracks,
                }),
            )
            .map_err(|e| e.to_string())?;
            writeln!(writer).map_err(|e| e.to_string())?;
            next_frame += 1;
            Ok(())
        })?;
        frames_batch.clear();
    }
    if !frames_batch.is_empty() {
        engine.detect_frames(&frames_batch, |_, detected| {
            let candidates = detected
                .into_iter()
                .enumerate()
                .map(|(i, p)| Candidate {
                    frame_index: next_frame,
                    class_name: "player".into(),
                    score: f64::from(p.score),
                    bbox_xyxy_norm: p.bbox,
                    query_index: Some(i),
                    head_point_norm: p.head,
                    touch_probability: None,
                })
                .collect::<Vec<_>>();
            let tracked = tracker.process_frame(next_frame, &candidates)?;
            total_boxes += tracked.tracks.len() as u64;
            let tracks = tracked
                .tracks
                .iter()
                .map(|track| {
                    serde_json::json!({
                        "id": track.track_id,
                        "bbox": track.detection.bbox_xyxy_norm,
                    })
                })
                .collect::<Vec<_>>();
            serde_json::to_writer(
                &mut writer,
                &serde_json::json!({
                    "frame": next_frame, "tracks": tracks,
                }),
            )
            .map_err(|e| e.to_string())?;
            writeln!(writer).map_err(|e| e.to_string())?;
            next_frame += 1;
            Ok(())
        })?;
    }
    writer.flush()?;
    drop(pipe);
    let exit = ffmpeg.wait()?;
    if !exit.success() {
        return Err(format!("ffmpeg exit: {exit}").into());
    }
    if next_frame == 0 {
        return Err("no video frames".into());
    }
    let duration = started.elapsed().as_secs_f64();
    let report = serde_json::json!({
        "model": "RTMDet-Nano+RTMPose+ProvisionalTracker",
        "frames": next_frame,
        "total_observed_boxes": total_boxes,
        "video": input,
        "video_sha256": fingerprint(Path::new(&input))?,
        "fps": (next_frame as f64 / duration),
        "inference_seconds": duration,
        "note": "Identical 640x360 @29.97fps input to SAM experiment",
    });
    let metadata = Path::new(&output).with_extension("stats.json");
    std::fs::write(metadata, serde_json::to_vec_pretty(&report)?)?;
    println!("{report:#}");
    Ok(())
}
