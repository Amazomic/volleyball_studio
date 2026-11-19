//! Hardware/software pixel parity and lifecycle regression tests.
use super::*;
#[cfg(unix)]
use crate::raw_decoder::{
    decode_full_video as reference_software_video,
    decode_original_frames as reference_software_original,
};
use sha2::{Digest, Sha256};
#[cfg(windows)]
use std::io::Cursor;

#[cfg(windows)]
fn reference_software_video(
    source: &Path,
    pixels: ModelPixels,
    stride: usize,
    cancelled: &AtomicBool,
    mut consume: impl FnMut(RawWindow) -> Result<(), String>,
) -> Result<DecodeSummary, String> {
    let (width, height, channels, format) = pixels.dimensions();
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(source)
        .args([
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &format!("fps=30,scale={width}:{height}:flags=bilinear,format={format}"),
            "-pix_fmt",
            format,
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "Software reference decoder: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let mut reader = FrameWindows::new(
        Cursor::new(output.stdout),
        width * height * channels,
        stride,
    )
    .map_err(|error| error.to_string())?;
    let mut windows = 0;
    while let Some(window) = reader.next_window().map_err(|error| error.to_string())? {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        consume(window)?;
        windows += 1;
    }
    Ok(DecodeSummary {
        first_frame: 0,
        windows,
        decoded_frames: reader.decoded_frames(),
        stride,
    })
}

#[cfg(windows)]
fn reference_software_original(
    source: &Path,
    expected_frames: u64,
    cancelled: &AtomicBool,
    mut consume: impl FnMut(u64, &[u8], usize, usize) -> Result<(), String>,
) -> Result<(), String> {
    let info = crate::probe(source)?;
    let (width, height) = (info.width as usize, info.height as usize);
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(source)
        .args([
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            "fps=30,format=rgb24",
            "-pix_fmt",
            "rgb24",
            "-frames:v",
            &expected_frames.to_string(),
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "Software original reference: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let frame_bytes = width * height * 3;
    if output.stdout.len() != expected_frames as usize * frame_bytes {
        return Err("Software original frame count mismatch".into());
    }
    for (index, frame) in output.stdout.chunks_exact(frame_bytes).enumerate() {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        consume(index as u64, frame, width, height)?;
    }
    Ok(())
}

#[test]
fn shared_cpu_stream_matches_two_separate_decoders_pixel_for_pixel() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return;
    }
    let temp = SocketDir::new().unwrap();
    let source = temp.0.join("synthetic.mkv");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=60:duration=1.0",
            "-an",
            "-c:v",
            "mpeg4",
            "-y",
        ])
        .arg(&source)
        .status()
        .unwrap();
    assert!(status.success());
    let cancel = AtomicBool::new(false);
    for stride in [1, 3, 9] {
        let mut gray = Vec::new();
        let mut rgb = Vec::new();
        let first =
            reference_software_video(&source, ModelPixels::BallGray, stride, &cancel, |w| {
                gray.push((w.first_frame, Sha256::digest(&w.data).to_vec()));
                Ok(())
            })
            .unwrap();
        let second =
            reference_software_video(&source, ModelPixels::PlayersRgb, stride, &cancel, |w| {
                rgb.push((w.first_frame, Sha256::digest(&w.data).to_vec()));
                Ok(())
            })
            .unwrap();
        assert_eq!(first.decoded_frames, second.decoded_frames);
        let mut count = 0;
        let shared = decode_full_video_shared(&source, stride, &cancel, |batch| {
            assert_eq!(batch.first_frame, gray[count].0);
            assert_eq!(batch.first_frame, rgb[count].0);
            assert_eq!(Sha256::digest(&batch.gray).to_vec(), gray[count].1);
            assert_eq!(Sha256::digest(&batch.rgb).to_vec(), rgb[count].1);
            count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(shared.windows, first.windows);
        assert_eq!(shared.decoded_frames, first.decoded_frames);
        assert_eq!(count, shared.windows);
    }
    fs::remove_file(&source).unwrap();
}

#[test]
fn final_partial_rgb_frames_are_supplied_only_as_pose_lookahead() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return;
    }
    let temp = SocketDir::new().unwrap();
    let source = temp.0.join("tail.mkv");
    assert!(Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30:duration=0.73",
            "-an",
            "-c:v",
            "mpeg4",
            "-y",
        ])
        .arg(&source)
        .status()
        .unwrap()
        .success());
    let frame_bytes = 640 * 360 * 3;
    let raw = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&source)
        .args([
            "-map",
            "0:v:0",
            "-vf",
            "fps=30,scale=640:360:flags=bilinear,format=rgb24",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .output()
        .unwrap();
    assert!(raw.status.success());
    assert_eq!(raw.stdout.len() % frame_bytes, 0);
    let frames = (raw.stdout.len() / frame_bytes) as u64;
    let windows = std::cell::Cell::new(0usize);
    let mut received = 0usize;
    let cancelled = AtomicBool::new(false);
    let result = decode_full_video_shared_with_backend_and_tail(
        &source,
        9,
        DecodePreference::Software,
        0,
        &cancelled,
        |batch| {
            assert_eq!(batch.first_frame, windows.get() as u64 * 9);
            windows.set(windows.get() + 1);
            Ok(())
        },
        |tail, decoded| {
            assert_eq!(decoded, frames);
            assert_eq!(tail, &raw.stdout[windows.get() * 9 * frame_bytes..]);
            assert!(tail.len() < 9 * frame_bytes);
            received += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(received, 1);
    assert_eq!(result.decode.windows, windows.get());
    assert_eq!(result.decode.decoded_frames, frames);
    fs::remove_file(source).unwrap();
}
#[test]
fn cancellation_after_first_shared_window_reaps_decoder() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return;
    }
    let temp = SocketDir::new().unwrap();
    let source = temp.0.join("short.mkv");
    assert!(Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=30:duration=1.0",
            "-an",
            "-c:v",
            "mpeg4",
            "-y",
        ])
        .arg(&source)
        .status()
        .unwrap()
        .success());
    let cancel = AtomicBool::new(false);
    let started = Instant::now();
    let result = decode_full_video_shared(&source, 3, &cancel, |_| {
        cancel.store(true, Ordering::Release);
        Ok(())
    });
    assert!(result.unwrap_err().contains("cancelled"));
    assert!(started.elapsed() < Duration::from_secs(2));
    fs::remove_file(&source).unwrap();
}
#[test]
fn auto_h264_backend_preserves_all_model_input_bytes() {
    if Command::new("ffmpeg").arg("-version").output().is_err()
        || Command::new("ffprobe").arg("-version").output().is_err()
    {
        return;
    }
    let temp = SocketDir::new().unwrap();
    let source = temp.0.join("auto-h264.mp4");
    let generated = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=60:duration=0.7",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-y",
        ])
        .arg(&source)
        .status()
        .unwrap();
    assert!(generated.success());
    let cancelled = AtomicBool::new(false);
    let mut software = Vec::new();
    let sw = decode_full_video_shared(&source, 9, &cancelled, |batch| {
        software.push((
            batch.first_frame,
            Sha256::digest(&batch.gray).to_vec(),
            Sha256::digest(&batch.rgb).to_vec(),
        ));
        Ok(())
    })
    .unwrap();
    let mut current = 0usize;
    let choice = decode_full_video_shared_with_backend(
        &source,
        9,
        DecodePreference::Auto,
        &cancelled,
        |batch| {
            assert_eq!(batch.first_frame, software[current].0);
            assert_eq!(Sha256::digest(&batch.gray).to_vec(), software[current].1);
            assert_eq!(Sha256::digest(&batch.rgb).to_vec(), software[current].2);
            current += 1;
            Ok(())
        },
    )
    .unwrap();
    eprintln!(
        "Auto backend: {:?}, fallback: {:?}",
        choice.backend, choice.fallback_reason
    );
    assert_eq!(choice.decode.windows, sw.windows);
    assert_eq!(choice.decode.decoded_frames, sw.decoded_frames);
    assert_eq!(current, sw.windows);
    if choice.backend == DecodeBackend::Software {
        assert!(choice.fallback_reason.is_some());
    } else {
        assert_eq!(choice.backend, DecodeBackend::Nvdec);
        assert!(choice.fallback_reason.is_none());
    }
    fs::remove_file(source).unwrap();
}

#[test]
fn uncertified_codec_never_uses_nvdec_even_when_forced() {
    if Command::new("ffmpeg").arg("-version").output().is_err()
        || Command::new("ffprobe").arg("-version").output().is_err()
    {
        return;
    }
    let temp = SocketDir::new().unwrap();
    let source = temp.0.join("unsupported.mkv");
    assert!(Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30:duration=0.5",
            "-c:v",
            "mpeg4",
            "-y",
        ])
        .arg(&source)
        .status()
        .unwrap()
        .success());
    let cancelled = AtomicBool::new(false);
    let selected = select_decode_backend(&source, DecodePreference::Auto, &cancelled).unwrap();
    assert_eq!(selected.backend, DecodeBackend::Software);
    assert!(selected.fallback_reason.unwrap().contains("not certified"));
    let mut called = false;
    let rejected = decode_full_video_shared_with_backend(
        &source,
        1,
        DecodePreference::Nvdec,
        &cancelled,
        |_| {
            called = true;
            Ok(())
        },
    );
    assert!(rejected.unwrap_err().contains("not certified"));
    assert!(!called);
    fs::remove_file(source).unwrap();
}

#[test]
fn preflight_does_not_fake_cuda_on_small_hevc_frames() {
    if Command::new("ffmpeg").arg("-version").output().is_err()
        || Command::new("ffprobe").arg("-version").output().is_err()
    {
        return;
    }
    let temp = SocketDir::new().unwrap();
    let source = temp.0.join("small-hevc.mkv");
    let generated = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30:duration=0.5",
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-x265-params",
            "pools=1:frame-threads=1:log-level=error",
            "-pix_fmt",
            "yuv420p",
            "-y",
        ])
        .arg(&source)
        .status();
    if !generated.is_ok_and(|s| s.success()) {
        return; // Optional external encoder not installed on this host.
    }
    let cancelled = AtomicBool::new(false);
    let selection = select_decode_backend(&source, DecodePreference::Auto, &cancelled).unwrap();
    assert_eq!(selection.backend, DecodeBackend::Software);
    assert!(selection.fallback_reason.unwrap().contains("geometry"));
    assert!(select_decode_backend(&source, DecodePreference::Nvdec, &cancelled).is_err());
    fs::remove_file(source).unwrap();
}
#[test]
fn hevc_8bit_cuda_parity_and_hevc_10bit_conservative_fallback() {
    if Command::new("ffmpeg").arg("-version").output().is_err()
        || Command::new("ffprobe").arg("-version").output().is_err()
    {
        return;
    }
    let temp = SocketDir::new().unwrap();
    let cancelled = AtomicBool::new(false);
    for (name, format) in [("hevc-8", "yuv420p"), ("hevc-10", "yuv420p10le")] {
        let source = temp.0.join(format!("{name}.mkv"));
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "2",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x180:rate=30:duration=0.7",
                "-c:v",
                "libx265",
                "-preset",
                "ultrafast",
                "-x265-params",
                "pools=1:frame-threads=1:log-level=error",
                "-pix_fmt",
                format,
                "-y",
            ])
            .arg(&source)
            .status();
        if !status.is_ok_and(|s| s.success()) {
            return; // Hardware parity is conditional on local encoder availability.
        }
        let mut previous = Vec::new();
        let software = decode_full_video_shared(&source, 9, &cancelled, |batch| {
            previous.push((
                batch.first_frame,
                Sha256::digest(&batch.gray).to_vec(),
                Sha256::digest(&batch.rgb).to_vec(),
            ));
            Ok(())
        })
        .unwrap();
        let mut position = 0;
        let result = decode_full_video_shared_with_backend(
            &source,
            9,
            DecodePreference::Auto,
            &cancelled,
            |batch| {
                assert_eq!(batch.first_frame, previous[position].0);
                assert_eq!(Sha256::digest(&batch.gray).to_vec(), previous[position].1);
                assert_eq!(Sha256::digest(&batch.rgb).to_vec(), previous[position].2);
                position += 1;
                Ok(())
            },
        )
        .unwrap();
        eprintln!(
            "{name}: backend={:?}, fallback={:?}",
            result.backend, result.fallback_reason
        );
        assert_eq!(result.decode.windows, software.windows);
        assert_eq!(result.decode.decoded_frames, software.decoded_frames);
        assert_eq!(position, software.windows);
        if name == "hevc-10" {
            assert_eq!(result.backend, DecodeBackend::Software);
            assert!(result.fallback_reason.unwrap().contains("not certified"));
        }
        fs::remove_file(&source).unwrap();
    }
}

#[test]
#[ignore = "requires NVIDIA GPU and H264/HEVC encoders"]
fn rotated_nvdec_matches_software_pixels() {
    let temp = SocketDir::new().unwrap();
    let cancelled = AtomicBool::new(false);
    for (codec, size) in [("libx264", "320x180"), ("libx265", "3840x2160")] {
        let original = temp.0.join("original.mp4");
        assert!(Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc2=size={size}:rate=30:duration=0.4"),
                "-c:v",
                codec,
                "-preset",
                "ultrafast",
                "-threads",
                "2",
            ])
            .args(if codec == "libx265" {
                vec!["-x265-params", "pools=1:frame-threads=1:log-level=error"]
            } else {
                vec![]
            })
            .args(["-pix_fmt", "yuv420p", "-y"])
            .arg(&original)
            .status()
            .unwrap()
            .success());
        for rotation in ["-90", "90", "180", "0"] {
            let rotated = temp.0.join("rotated.mp4");
            assert!(Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-display_rotation:v:0",
                    rotation,
                    "-i"
                ])
                .arg(&original)
                .args(["-c", "copy", "-y"])
                .arg(&rotated)
                .status()
                .unwrap()
                .success());
            let mut expected = Vec::new();
            let software = decode_full_video_shared(&rotated, 9, &cancelled, |batch| {
                expected.push((batch.first_frame, batch.gray, batch.rgb));
                Ok(())
            })
            .unwrap();
            let mut position = 0;
            let hardware = decode_full_video_shared_with_backend(
                &rotated,
                9,
                DecodePreference::Nvdec,
                &cancelled,
                |batch| {
                    assert_eq!(batch.first_frame, expected[position].0);
                    assert_eq!(
                        Sha256::digest(&batch.gray),
                        Sha256::digest(&expected[position].1),
                        "{codec} rotation={rotation} gray"
                    );
                    assert_eq!(
                        Sha256::digest(&batch.rgb),
                        Sha256::digest(&expected[position].2),
                        "{codec} rotation={rotation} rgb"
                    );
                    position += 1;
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(hardware.backend, DecodeBackend::Nvdec);
            assert_eq!(hardware.decode.decoded_frames, software.decoded_frames);
            assert_eq!(position, expected.len());
            for pixels in [ModelPixels::BallGray, ModelPixels::PlayersRgb] {
                let mut count = 0;
                let decoded = crate::raw_decoder::decode_full_video(
                    &rotated,
                    pixels,
                    9,
                    &cancelled,
                    |window| {
                        let bytes = if matches!(pixels, ModelPixels::BallGray) {
                            &expected[count].1
                        } else {
                            &expected[count].2
                        };
                        assert_eq!(window.first_frame, expected[count].0);
                        assert_eq!(Sha256::digest(&window.data), Sha256::digest(bytes));
                        count += 1;
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(decoded.decoded_frames, software.decoded_frames);
                assert_eq!(count, expected.len());
                crate::raw_decoder::decode_windows(
                    &rotated,
                    pixels,
                    0,
                    1,
                    9,
                    &cancelled,
                    |window| {
                        let bytes = if matches!(pixels, ModelPixels::BallGray) {
                            &expected[0].1
                        } else {
                            &expected[0].2
                        };
                        assert_eq!(Sha256::digest(&window.data), Sha256::digest(bytes));
                        Ok(())
                    },
                )
                .unwrap();
            }
            eprintln!("{codec} {size} rotation={rotation}: NVDEC pixel parity passed");
            fs::remove_file(rotated).unwrap();
        }
        fs::remove_file(original).unwrap();
    }
}

#[test]
fn nvdec_preview_clears_display_rotation_before_manual_pixel_rotation() {
    let decision = DecodeSelection {
        backend: DecodeBackend::Nvdec,
        fallback_reason: None,
        rotation_filter: "transpose=clock,",
    };
    let command =
        shared_decode_command_with_preview(Path::new("rotated.mp4"), &decision, 2, true, false);
    let args: Vec<_> = command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let rotation = args
        .iter()
        .position(|arg| arg == "-display_rotation:v:0")
        .unwrap();
    let input = args.iter().position(|arg| arg == "-i").unwrap();
    assert_eq!(args[rotation + 1], "0");
    assert!(args.iter().any(|arg| arg == "-noautorotate"));
    assert!(rotation < input);
    let filter = &args[args
        .iter()
        .position(|arg| arg == "-filter_complex")
        .unwrap()
        + 1];
    assert!(filter.contains("hwdownload,format=nv12,transpose=clock,split=2"));
}

#[test]
fn shared_command_honors_requested_decoder_and_filter_threads() {
    let decision = DecodeSelection {
        backend: DecodeBackend::Software,
        fallback_reason: None,
        rotation_filter: "",
    };
    for threads in [0, 1, 32] {
        let command = shared_decode_command(Path::new("input.mp4"), &decision, threads);
        let args: Vec<_> = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        for option in ["-threads", "-filter_complex_threads"] {
            let position = args.iter().position(|arg| arg == option).unwrap();
            assert_eq!(args[position + 1], threads.to_string());
        }
        assert!(
            args.iter().position(|arg| arg == "-threads").unwrap()
                < args.iter().position(|arg| arg == "-i").unwrap()
        );
    }
}

#[test]
#[ignore = "requires NVIDIA GPU and HEVC encoder"]
fn hevc_nvdec_high_cpu_thread_budget_preserves_complete_windows() {
    let temp = SocketDir::new().unwrap();
    let source = temp.0.join("hevc-4k.mp4");
    assert!(Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=3840x2160:rate=30:duration=0.4",
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-x265-params",
            "pools=1:frame-threads=1:log-level=error",
            "-pix_fmt",
            "yuv420p",
            "-y",
        ])
        .arg(&source)
        .status()
        .unwrap()
        .success());
    let cancelled = AtomicBool::new(false);
    let mut expected = Vec::new();
    let software = decode_full_video_shared(&source, 9, &cancelled, |batch| {
        expected.push((Sha256::digest(batch.gray), Sha256::digest(batch.rgb)));
        Ok(())
    })
    .unwrap();
    for threads in [32, 128] {
        let mut position = 0;
        let hardware = decode_full_video_shared_with_backend_and_tail(
            &source,
            9,
            DecodePreference::Nvdec,
            threads,
            &cancelled,
            |batch| {
                assert_eq!(Sha256::digest(batch.gray), expected[position].0);
                assert_eq!(Sha256::digest(batch.rgb), expected[position].1);
                position += 1;
                Ok(())
            },
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(hardware.backend, DecodeBackend::Nvdec);
        assert_eq!(hardware.decode.decoded_frames, software.decoded_frames);
        assert_eq!(position, expected.len());
    }
    fs::remove_file(source).unwrap();
}

fn preview_source(root: &Path, name: &str, rate: u32, duration: &str, audio: bool) -> PathBuf {
    let source = root.join(name);
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc2=size=160x90:rate={rate}:duration={duration}"),
    ]);
    if audio {
        cmd.args([
            "-f",
            "lavfi",
            "-i",
            &format!("sine=frequency=440:duration={duration}"),
            "-c:a",
            "aac",
        ]);
    }
    assert!(cmd
        .args(["-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-y"])
        .arg(&source)
        .status()
        .unwrap()
        .success());
    source
}

#[test]
fn shared_preview_preserves_model_pixels_native_fps_audio_and_per_video_isolation() {
    use crate::preparation::primary_preview::PrimaryPreviewPlan;
    let temp = SocketDir::new().unwrap();
    let cancel = AtomicBool::new(false);
    let first = preview_source(&temp.0, "first.mp4", 60, "0.73", true);
    let second = preview_source(&temp.0, "second.mp4", 25, "1.16", false);
    let mut outputs = Vec::new();
    for (source, name, native_fps, audio) in [
        (&first, "first.mp4", 60.0, true),
        (&second, "second.mp4", 25.0, false),
    ] {
        let plan = PrimaryPreviewPlan::new(source, &temp.0, "project", name).unwrap();
        let other = PrimaryPreviewPlan::new(source, &temp.0, "other-project", name).unwrap();
        assert_ne!(plan.output, other.output);
        assert!(plan.ready_path().is_none());
        let mut measured = Vec::new();
        let baseline = decode_full_video_shared(source, 9, &cancel, |batch| {
            measured.push((
                batch.first_frame,
                Sha256::digest(&batch.gray).to_vec(),
                Sha256::digest(&batch.rgb).to_vec(),
            ));
            Ok(())
        })
        .unwrap();
        let write = plan.begin().unwrap();
        let mut count = 0;
        let mut tail_count = 0;
        let summary = decode_full_video_shared_with_preview(
            source,
            9,
            DecodePreference::Software,
            2,
            &cancel,
            Some(PrimaryPreviewOutput {
                path: &write.path,
                hardware: false,
            }),
            |batch| {
                assert_eq!(batch.first_frame, measured[count].0);
                assert_eq!(Sha256::digest(&batch.gray).to_vec(), measured[count].1);
                assert_eq!(Sha256::digest(&batch.rgb).to_vec(), measured[count].2);
                count += 1;
                Ok(())
            },
            |_, frames| {
                tail_count = frames;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(baseline.decoded_frames, summary.decode.decoded_frames);
        assert_eq!(tail_count, baseline.decoded_frames);
        assert_eq!(count, measured.len());
        assert!(plan.ready_path().is_none());
        write.publish(&cancel).unwrap();
        assert_eq!(plan.ready_path(), Some(plan.output.as_path()));
        let probe = crate::ffprobe(&plan.output).unwrap();
        let info = crate::inspect_probe(&probe).unwrap();
        assert!((info.fps.unwrap() - native_fps).abs() < 0.1);
        assert_eq!((info.width, info.height), (160, 90));
        assert_eq!(
            probe["streams"]
                .as_array()
                .unwrap()
                .iter()
                .any(|stream| stream["codec_type"] == "audio"),
            audio
        );
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(&plan.manifest).unwrap()).unwrap();
        assert_eq!(receipt["video"], name);
        assert_eq!(receipt["timeline"]["frame_timing"], "native_pts");
        outputs.push(plan.output.clone());
    }
    assert_ne!(outputs[0], outputs[1]);
    assert_ne!(
        fs::read(&outputs[0]).unwrap(),
        fs::read(&outputs[1]).unwrap()
    );
    assert!((crate::probe(&outputs[0]).unwrap().duration_sec - 0.73).abs() < 0.1);
    assert!((crate::probe(&outputs[1]).unwrap().duration_sec - 1.16).abs() < 0.1);
    fs::remove_dir_all(&temp.0).unwrap();
}

#[test]
fn shared_preview_failure_or_cancellation_never_publishes_or_leaves_partial() {
    use crate::preparation::primary_preview::PrimaryPreviewPlan;
    let temp = SocketDir::new().unwrap();
    let source = preview_source(&temp.0, "failure.mp4", 30, "1.0", false);
    let plan = PrimaryPreviewPlan::new(&source, &temp.0, "project", "failure.mp4").unwrap();
    for cancelled in [false, true] {
        let write = plan.begin().unwrap();
        let path = write.path.clone();
        let cancel = AtomicBool::new(cancelled);
        let result = decode_full_video_shared_with_preview(
            &source,
            9,
            DecodePreference::Software,
            2,
            &cancel,
            Some(PrimaryPreviewOutput {
                path: &path,
                hardware: false,
            }),
            |_| Err("synthetic model failure".into()),
            |_, _| Ok(()),
        );
        assert!(result.is_err());
        drop(write);
        assert!(!path.exists());
        assert!(!path.parent().unwrap().exists());
        assert!(plan.ready_path().is_none());
        assert!(!plan.manifest.exists());
    }
    fs::remove_dir_all(&temp.0).unwrap();
}

#[test]
#[ignore = "requires NVIDIA NVDEC/NVENC"]
fn shared_preview_nvdec_nvenc_preserves_rotation_audio_and_native_fps() {
    use crate::preparation::primary_preview::PrimaryPreviewPlan;
    let temp = SocketDir::new().unwrap();
    let base = temp.0.join("hevc.mp4");
    assert!(Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=60:duration=0.73",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=0.73",
            "-c:v",
            "libx265",
            "-x265-params",
            "log-level=error:pools=1",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-y"
        ])
        .arg(&base)
        .status()
        .unwrap()
        .success());
    let cancel = AtomicBool::new(false);
    for rotation in [0, 90, -90, 180] {
        let name = format!("rotated-{rotation}.mp4");
        let source = temp.0.join(&name);
        assert!(Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&base)
            .args([
                "-c",
                "copy",
                "-metadata:s:v:0",
                &format!("rotate={rotation}"),
                "-y"
            ])
            .arg(&source)
            .status()
            .unwrap()
            .success());
        let plan = PrimaryPreviewPlan::new(&source, &temp.0, "project", &name).unwrap();
        let write = plan.begin().unwrap();
        let mut baseline = Vec::new();
        decode_full_video_shared_with_backend_and_tail(
            &source,
            9,
            DecodePreference::Nvdec,
            2,
            &cancel,
            |batch| {
                baseline.push((
                    Sha256::digest(&batch.gray).to_vec(),
                    Sha256::digest(&batch.rgb).to_vec(),
                ));
                Ok(())
            },
            |_, _| Ok(()),
        )
        .unwrap();
        let mut index = 0;
        let result = decode_full_video_shared_with_preview(
            &source,
            9,
            DecodePreference::Nvdec,
            2,
            &cancel,
            Some(PrimaryPreviewOutput {
                path: &write.path,
                hardware: true,
            }),
            |batch| {
                assert_eq!(
                    (
                        Sha256::digest(&batch.gray).to_vec(),
                        Sha256::digest(&batch.rgb).to_vec()
                    ),
                    baseline[index]
                );
                index += 1;
                Ok(())
            },
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(result.backend, DecodeBackend::Nvdec);
        assert_eq!(index, baseline.len());
        write.publish(&cancel).unwrap();
        let info = crate::probe(plan.ready_path().unwrap()).unwrap();
        assert_eq!(info.rotation, 0);
        assert!((info.fps.unwrap() - 60.0).abs() < 0.1);
        let original = crate::probe(&source).unwrap();
        assert_eq!((info.width, info.height), (original.width, original.height));
        let probe = crate::ffprobe(&plan.output).unwrap();
        assert!(probe["streams"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["codec_name"] == "aac"));
    }
    fs::remove_dir_all(&temp.0).unwrap();
}

#[test]
fn primary_preview_cancel_before_publish_cleans_encoded_file_and_source_change_invalidates_cache() {
    use crate::preparation::primary_preview::PrimaryPreviewPlan;
    let temp = SocketDir::new().unwrap();
    let source = preview_source(&temp.0, "source.mp4", 30, "0.6", false);
    let plan = PrimaryPreviewPlan::new(&source, &temp.0, "project", "source.mp4").unwrap();
    let cancel = AtomicBool::new(false);
    let write = plan.begin().unwrap();
    let scratch = write.path.clone();
    decode_full_video_shared_with_preview(
        &source,
        9,
        DecodePreference::Software,
        2,
        &cancel,
        Some(PrimaryPreviewOutput {
            path: &write.path,
            hardware: false,
        }),
        |_| Ok(()),
        |_, _| Ok(()),
    )
    .unwrap();
    cancel.store(true, Ordering::Release);
    assert_eq!(write.publish(&cancel).unwrap_err(), "cancelled");
    assert!(!scratch.exists());
    assert!(plan.ready_path().is_none());
    cancel.store(false, Ordering::Release);
    let write = plan.begin().unwrap();
    decode_full_video_shared_with_preview(
        &source,
        9,
        DecodePreference::Software,
        2,
        &cancel,
        Some(PrimaryPreviewOutput {
            path: &write.path,
            hardware: false,
        }),
        |_| Ok(()),
        |_, _| Ok(()),
    )
    .unwrap();
    write.publish(&cancel).unwrap();
    assert!(plan.ready_path().is_some());
    use std::io::Write;
    fs::OpenOptions::new()
        .append(true)
        .open(&source)
        .unwrap()
        .write_all(b"source changed")
        .unwrap();
    assert!(plan.ready_path().is_none());
    fs::remove_dir_all(&temp.0).unwrap();
}

#[test]
fn native_pose_source_has_original_decoder_pixels_and_synchronized_frames() {
    let temp = SocketDir::new().unwrap();
    let base = temp.0.join("native-base.mp4");
    assert!(Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=60:duration=1.17",
            "-c:v",
            "mpeg4",
            "-y"
        ])
        .arg(&base)
        .status()
        .unwrap()
        .success());
    for rotation in [0, 90] {
        let source = temp.0.join(format!("native-{rotation}.mp4"));
        assert!(Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&base)
            .args([
                "-c",
                "copy",
                "-metadata:s:v:0",
                &format!("rotate={rotation}"),
                "-y"
            ])
            .arg(&source)
            .status()
            .unwrap()
            .success());
        let cancel = AtomicBool::new(false);
        let native_info = crate::probe(&source).unwrap();
        let mut classic = Vec::new();
        let summary = decode_full_video_shared_with_backend_and_tail(
            &source,
            9,
            DecodePreference::Software,
            2,
            &cancel,
            |_| Ok(()),
            |_, _| Ok(()),
        )
        .unwrap();
        reference_software_original(
            &source,
            summary.decode.decoded_frames,
            &cancel,
            |index, frame, width, height| {
                assert_eq!(
                    (width, height),
                    (native_info.width as usize, native_info.height as usize)
                );
                classic.push((index, Sha256::digest(frame).to_vec()));
                Ok(())
            },
        )
        .unwrap();
        let mut checked = 0;
        let native =
            decode_full_video_shared_with_native_and_tail(
                &source,
                9,
                DecodePreference::Software,
                2,
                &cancel,
                true,
                |batch| {
                    let original = batch.native.expect("Native branch must be populated");
                    assert_eq!(
                        (original.width, original.height),
                        (native_info.width as usize, native_info.height as usize)
                    );
                    let frame_bytes = original.width * original.height * 3;
                    assert_eq!(original.rgb.len(), frame_bytes * 9);
                    for offset in 0..9 {
                        let frame = batch.first_frame + offset;
                        let pixels = &original.rgb
                            [offset as usize * frame_bytes..(offset as usize + 1) * frame_bytes];
                        assert_eq!(classic[frame as usize].0, frame);
                        assert_eq!(Sha256::digest(pixels).to_vec(), classic[frame as usize].1,
                        "Original-resolution pixels changed at frame {frame}, rotation {rotation}");
                        checked += 1;
                    }
                    Ok(())
                },
                |_, _| Ok(()),
            )
            .unwrap();
        assert_eq!(native.decode.decoded_frames, summary.decode.decoded_frames);
        assert_eq!(checked, native.decode.windows * 9);
    }
}

#[cfg(windows)]
#[test]
fn windows_nvdec_matches_previous_independent_h264_model_and_pose_frames() {
    let temp = SocketDir::new().unwrap();
    let source = temp.0.join("windows-nvdec-parity.mp4");
    assert!(Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=60:duration=1.17",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
        ])
        .arg(&source)
        .status()
        .unwrap()
        .success());
    let cancelled = AtomicBool::new(false);
    let backend = select_decode_backend(&source, DecodePreference::Auto, &cancelled).unwrap();
    if backend.backend != DecodeBackend::Nvdec {
        eprintln!("NVDEC unavailable; hardware parity requires a supported NVIDIA GPU");
        return;
    }
    let mut old_ball = Vec::new();
    let mut old_people = Vec::new();
    let old_ball_receipt = crate::raw_decoder::decode_full_video(
        &source,
        ModelPixels::BallGray,
        9,
        &cancelled,
        |window| {
            old_ball.push(Sha256::digest(&window.data).to_vec());
            Ok(())
        },
    )
    .unwrap();
    let old_people_receipt = crate::raw_decoder::decode_full_video(
        &source,
        ModelPixels::PlayersRgb,
        9,
        &cancelled,
        |window| {
            old_people.push(Sha256::digest(&window.data).to_vec());
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(
        old_ball_receipt.decoded_frames,
        old_people_receipt.decoded_frames
    );
    let mut old_original = Vec::new();
    crate::raw_decoder::decode_original_frames(
        &source,
        old_ball_receipt.decoded_frames,
        &cancelled,
        |_, rgb, _, _| {
            old_original.push(Sha256::digest(rgb).to_vec());
            Ok(())
        },
    )
    .unwrap();
    let mut windows = 0;
    let summary = decode_full_video_shared_with_native_and_tail(
        &source,
        9,
        DecodePreference::Auto,
        0,
        &cancelled,
        true,
        |batch| {
            assert_eq!(batch.first_frame, windows as u64 * 9);
            assert_eq!(Sha256::digest(&batch.gray).to_vec(), old_ball[windows]);
            assert_eq!(Sha256::digest(&batch.rgb).to_vec(), old_people[windows]);
            let original = batch.native.expect("Original pixels are required");
            let frame_bytes = original.width * original.height * 3;
            for offset in 0..9 {
                let start = offset * frame_bytes;
                let old_frame = windows * 9 + offset;
                assert_eq!(
                    Sha256::digest(&original.rgb[start..start + frame_bytes]).to_vec(),
                    old_original[old_frame],
                    "RTMPose original RGB differs at frame {old_frame}"
                );
            }
            windows += 1;
            Ok(())
        },
        |_, _| Ok(()),
    )
    .unwrap();
    assert_eq!(summary.backend, DecodeBackend::Nvdec);
    assert_eq!(summary.decode.windows, old_ball_receipt.windows);
    assert_eq!(
        summary.decode.decoded_frames,
        old_ball_receipt.decoded_frames
    );
    assert_eq!(windows, old_ball.len());
    fs::remove_file(source).unwrap();
}
