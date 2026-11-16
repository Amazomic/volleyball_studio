//! One FFmpeg decode, with the existing CPU pixel filters on two synchronous
//! streams. Linux UNIX sockets carry frames without full-video raw caches.
//! No model observations or cache manifests are published by this module.
mod backend;
use super::{stderr_tail, DecodeSummary, ManagedChild, ModelPixels};
use crate::frame_windows::{FrameWindows, RawWindow, MAX_FULL_VIDEO_STRIDE};
pub use backend::{select_decode_backend, DecodeBackend, DecodePreference, DecodeSelection};
use serde::Serialize;
use std::{
    fs::{self, DirBuilder},
    io::{self, Read},
    os::unix::{
        fs::DirBuilderExt,
        net::{UnixListener, UnixStream},
    },
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const POLL: Duration = Duration::from_millis(100);
const INIT_LIMIT: Duration = Duration::from_secs(30);
const IDLE_LIMIT: Duration = Duration::from_secs(120);

/// A synchronized 30-FPS window; both inputs use the canonical FFmpeg CPU
/// scaling/format filters and have identical source-frame indices.
#[derive(Debug)]
pub struct PrimaryFrameBatch {
    pub first_frame: u64,
    pub gray: Vec<u8>,
    pub rgb: Vec<u8>,
}

/// Immutable summary returned after both outputs reach verified EOF.
#[derive(Clone, Debug, Serialize)]
pub struct PrimaryDecodeSummary {
    pub decode: DecodeSummary,
    pub backend: DecodeBackend,
    pub fallback_reason: Option<String>,
}

struct SocketDir(PathBuf);

impl SocketDir {
    fn new() -> Result<Self, String> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        for attempt in 0..8 {
            let path = std::env::temp_dir().join(format!(
                "volley-primary-{}-{nanos}-{attempt}",
                std::process::id()
            ));
            match DirBuilder::new().mode(0o700).create(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(format!("Create private decode directory: {error}")),
            }
        }
        Err("Could not allocate an isolated decoder socket directory".into())
    }
}

impl Drop for SocketDir {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.0.join("gray.sock"));
        let _ = fs::remove_file(self.0.join("rgb.sock"));
        let _ = fs::remove_dir(&self.0);
    }
}

fn accept(
    listener: &UnixListener,
    child: &mut ManagedChild,
    cancelled: &AtomicBool,
) -> Result<UnixStream, String> {
    let started = Instant::now();
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        match listener.accept() {
            Ok((stream, _)) => {
                stream
                    .set_read_timeout(Some(POLL))
                    .map_err(|error| format!("Set decoder read timeout: {error}"))?;
                return Ok(stream);
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if let Some(status) = child
                    .0
                    .try_wait()
                    .map_err(|error| format!("Check FFmpeg startup: {error}"))?
                {
                    return Err(format!(
                        "Shared FFmpeg exited before opening both streams: {status}"
                    ));
                }
                if started.elapsed() >= INIT_LIMIT {
                    return Err("Shared FFmpeg did not open the output stream in time".into());
                }
                thread::sleep(POLL);
            }
            Err(error) => return Err(format!("Accept FFmpeg output stream: {error}")),
        }
    }
}

struct CancellableSocket<'a> {
    stream: UnixStream,
    cancelled: &'a AtomicBool,
}

impl Read for CancellableSocket<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let started = Instant::now();
        loop {
            if self.cancelled.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "cancelled",
                ));
            }
            match self.stream.read(buffer) {
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) =>
                {
                    if started.elapsed() >= IDLE_LIMIT {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "Shared FFmpeg stopped producing frames",
                        ));
                    }
                }
                other => return other,
            }
        }
    }
}

enum WindowEvent {
    Data(RawWindow),
    Eof(u64, Vec<u8>),
    Failed(String),
}

fn drain(
    stream: UnixStream,
    frame_bytes: usize,
    stride: usize,
    cancelled: &AtomicBool,
    sender: SyncSender<WindowEvent>,
) {
    let source = CancellableSocket { stream, cancelled };
    let mut windows = match FrameWindows::new(source, frame_bytes, stride) {
        Ok(reader) => reader,
        Err(error) => {
            let _ = sender.send(WindowEvent::Failed(error.to_string()));
            return;
        }
    };
    loop {
        let (event, done) = match windows.next_window() {
            Ok(Some(window)) => (WindowEvent::Data(window), false),
            Ok(None) => (
                WindowEvent::Eof(
                    windows.decoded_frames(),
                    windows.trailing_frame_bytes().unwrap_or_default(),
                ),
                true,
            ),
            Err(error) => (WindowEvent::Failed(error.to_string()), true),
        };
        if sender.send(event).is_err() || done {
            break;
        }
    }
}

fn receive(
    receiver: &Receiver<WindowEvent>,
    cancelled: &AtomicBool,
) -> Result<WindowEvent, String> {
    let started = Instant::now();
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        match receiver.recv_timeout(POLL) {
            Ok(WindowEvent::Failed(error)) => return Err(error),
            Ok(event) => return Ok(event),
            Err(RecvTimeoutError::Disconnected) => {
                return Err("Shared decode reader ended without an EOF receipt".into());
            }
            Err(RecvTimeoutError::Timeout) if started.elapsed() >= IDLE_LIMIT => {
                return Err("Shared decode reader stalled".into());
            }
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

/// A single FFmpeg process applies fps=30 before splitting into the canonical
/// gray 512x288 and RGB 640x360 CPU filters. Every 9-frame window is joined
/// at the same global frame index; mismatches fail rather than silently drift.
/// Channels are bounded to two windows per output, regardless of video length.
///
/// Use the legacy software backend explicitly for parity tests and diagnostics.
pub fn decode_full_video_shared(
    source: &Path,
    stride: usize,
    cancelled: &AtomicBool,
    consume: impl FnMut(PrimaryFrameBatch) -> Result<(), String>,
) -> Result<DecodeSummary, String> {
    decode_full_video_shared_impl(
        source,
        stride,
        cancelled,
        DecodeBackend::Software,
        consume,
        |_, _| Ok(()),
    )
}

/// Backend decision is made before creating any model evidence; an NVDEC
/// failure after this point is terminal, never an in-run software fallback.
pub fn decode_full_video_shared_with_backend(
    source: &Path,
    stride: usize,
    preference: DecodePreference,
    cancelled: &AtomicBool,
    consume: impl FnMut(PrimaryFrameBatch) -> Result<(), String>,
) -> Result<PrimaryDecodeSummary, String> {
    let decision = select_decode_backend(source, preference, cancelled)?;
    let decode = decode_full_video_shared_impl(
        source,
        stride,
        cancelled,
        decision.backend,
        consume,
        |_, _| Ok(()),
    )?;
    Ok(PrimaryDecodeSummary {
        decode,
        backend: decision.backend,
        fallback_reason: decision.fallback_reason,
    })
}

/// Like decode_full_video_shared_with_backend, but also supplies the final
/// partial 0..8 RGB frames as bounded pose lookahead after the complete windows
/// have been read. Tail frames are NEVER model-detection windows.
pub fn decode_full_video_shared_with_backend_and_tail(
    source: &Path,
    stride: usize,
    preference: DecodePreference,
    cancelled: &AtomicBool,
    consume: impl FnMut(PrimaryFrameBatch) -> Result<(), String>,
    on_tail: impl FnMut(&[u8], u64) -> Result<(), String>,
) -> Result<PrimaryDecodeSummary, String> {
    if stride != 9 {
        return Err("Pose lookahead requires disjoint nine-frame model windows".into());
    }
    let decision = select_decode_backend(source, preference, cancelled)?;
    let decode = decode_full_video_shared_impl(
        source,
        stride,
        cancelled,
        decision.backend,
        consume,
        on_tail,
    )?;
    Ok(PrimaryDecodeSummary {
        decode,
        backend: decision.backend,
        fallback_reason: decision.fallback_reason,
    })
}

fn decode_full_video_shared_impl(
    source: &Path,
    stride: usize,
    cancelled: &AtomicBool,
    backend: DecodeBackend,
    mut consume: impl FnMut(PrimaryFrameBatch) -> Result<(), String>,
    mut on_tail: impl FnMut(&[u8], u64) -> Result<(), String>,
) -> Result<DecodeSummary, String> {
    if !(1..=MAX_FULL_VIDEO_STRIDE).contains(&stride) {
        return Err("Shared decode stride must be between 1 and 30".into());
    }
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    if !source.is_file() {
        return Err("Original source video is not a file".into());
    }

    let sockets = SocketDir::new()?;
    let gray_path = sockets.0.join("gray.sock");
    let rgb_path = sockets.0.join("rgb.sock");
    let gray_listener = UnixListener::bind(&gray_path)
        .map_err(|error| format!("Bind gray decode stream: {error}"))?;
    let rgb_listener = UnixListener::bind(&rgb_path)
        .map_err(|error| format!("Bind RGB decode stream: {error}"))?;
    gray_listener
        .set_nonblocking(true)
        .map_err(|error| format!("Gray socket: {error}"))?;
    rgb_listener
        .set_nonblocking(true)
        .map_err(|error| format!("RGB socket: {error}"))?;

    // Hardware frames are explicitly downloaded in NV12 format before the
    // old CPU fps/scale/format chain. Requiring CUDA output prevents FFmpeg
    // from silently decoding unsupported sources on CPU.
    let filter_prefix = if backend == DecodeBackend::Nvdec {
        "[0:v]hwdownload,format=nv12,fps=30,split=2[ball][players];"
    } else {
        "[0:v]fps=30,split=2[ball][players];"
    };
    let filters = format!(
        "{filter_prefix}[ball]scale=512:288:flags=bilinear,format=gray[g];\
[players]scale=640:360:flags=bilinear,format=rgb24[r]"
    );
    let mut command = Command::new("ffmpeg");
    command.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        "4",
    ]);
    if backend == DecodeBackend::Nvdec {
        command.args(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]);
    }
    command
        .arg("-i")
        .arg(source)
        .args([
            "-filter_complex",
            &filters,
            "-map",
            "[g]",
            "-an",
            "-sn",
            "-dn",
        ])
        .args(["-f", "rawvideo", "-pix_fmt", "gray"])
        .arg(format!("unix://{}", gray_path.display()))
        .args(["-map", "[r]", "-an", "-sn", "-dn"])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg(format!("unix://{}", rgb_path.display()))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut child = ManagedChild(
        command
            .spawn()
            .map_err(|error| format!("Launch shared FFmpeg: {error}"))?,
    );
    let error_reader = stderr_tail(
        child
            .0
            .stderr
            .take()
            .ok_or("Shared FFmpeg stderr missing")?,
    );

    let result = (|| -> Result<DecodeSummary, String> {
        let gray = accept(&gray_listener, &mut child, cancelled)?;
        let rgb = accept(&rgb_listener, &mut child, cancelled)?;
        let (gw, gh, gc, _) = ModelPixels::BallGray.dimensions();
        let (rw, rh, rc, _) = ModelPixels::PlayersRgb.dimensions();
        thread::scope(|scope| {
            let (gray_tx, gray_rx) = mpsc::sync_channel(2);
            let (rgb_tx, rgb_rx) = mpsc::sync_channel(2);
            scope.spawn(|| drain(gray, gw * gh * gc, stride, cancelled, gray_tx));
            scope.spawn(|| drain(rgb, rw * rh * rc, stride, cancelled, rgb_tx));
            let outcome = (|| -> Result<DecodeSummary, String> {
                let mut windows = 0usize;
                let decoded_frames;
                loop {
                    let ball = receive(&gray_rx, cancelled)?;
                    let people = receive(&rgb_rx, cancelled)?;
                    match (ball, people) {
                        (WindowEvent::Data(gray), WindowEvent::Data(rgb)) => {
                            let expected = windows as u64 * stride as u64;
                            if gray.first_frame != expected || rgb.first_frame != expected {
                                return Err(
                                    "Shared FFmpeg outputs have different frame indices".into()
                                );
                            }
                            consume(PrimaryFrameBatch {
                                first_frame: expected,
                                gray: gray.data,
                                rgb: rgb.data,
                            })?;
                            windows += 1;
                        }
                        (
                            WindowEvent::Eof(gray_frames, gray_tail),
                            WindowEvent::Eof(rgb_frames, rgb_tail),
                        ) => {
                            if gray_frames != rgb_frames {
                                return Err(
                                    "Shared FFmpeg outputs have different frame counts".into()
                                );
                            }
                            if stride == 9 {
                                let expected_tail = gray_frames
                                    .checked_sub(windows as u64 * 9)
                                    .filter(|len| *len < 9)
                                    .ok_or("Shared decoder reported an impossible tail length")?
                                    as usize;
                                if gray_tail.len() != expected_tail * gw * gh * gc
                                    || rgb_tail.len() != expected_tail * rw * rh * rc
                                {
                                    return Err(
                                        "Shared decode RGB/gray tail lengths disagree".into()
                                    );
                                }
                                if cancelled.load(Ordering::Acquire) {
                                    return Err("cancelled".into());
                                }
                                on_tail(&rgb_tail, gray_frames)?;
                            }
                            decoded_frames = gray_frames;
                            break;
                        }
                        _ => {
                            return Err(
                                "Shared FFmpeg streams did not end at the same window".into()
                            )
                        }
                    }
                }
                if cancelled.load(Ordering::Acquire) {
                    return Err("cancelled".into());
                }
                if windows == 0 {
                    return Err("Video does not contain a complete nine-frame model window".into());
                }
                let status = child
                    .0
                    .wait()
                    .map_err(|error| format!("Wait shared FFmpeg: {error}"))?;
                if !status.success() {
                    return Err(format!("Shared FFmpeg exited with {status}"));
                }
                Ok(DecodeSummary {
                    first_frame: 0,
                    windows,
                    decoded_frames,
                    stride,
                })
            })();
            if outcome.is_err() {
                // Kill before reader joins; joining a blocked decoder would hang.
                let _ = child.0.kill();
                let _ = child.0.wait();
            }
            drop(gray_rx);
            drop(rgb_rx);
            outcome
        })
    })();

    if result.is_err() {
        let _ = child.0.kill();
        let _ = child.0.wait();
    }
    let stderr = error_reader
        .join()
        .unwrap_or_else(|_| "Shared FFmpeg stderr reader panicked".into());
    result.map_err(|error| {
        if stderr.trim().is_empty() {
            error
        } else {
            format!(
                "{error}: {}",
                stderr.trim().chars().take(500).collect::<String>()
            )
        }
    })
}

#[cfg(test)]
mod tests;
