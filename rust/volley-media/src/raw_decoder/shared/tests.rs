//! Hardware/software pixel parity and lifecycle regression tests.
use super::*;
use crate::raw_decoder::decode_full_video;
use sha2::{Digest, Sha256};

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
        let first = decode_full_video(&source, ModelPixels::BallGray, stride, &cancel, |w| {
            gray.push((w.first_frame, Sha256::digest(&w.data).to_vec()));
            Ok(())
        })
        .unwrap();
        let second = decode_full_video(&source, ModelPixels::PlayersRgb, stride, &cancel, |w| {
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
