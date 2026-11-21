//! One FFmpeg decode, with the existing CPU pixel filters on two synchronous
//! streams. Local Unix sockets or Windows loopback TCP avoid raw-video caches.
//! No model observations or cache manifests are published by this module.
mod backend;
use super::{stderr_tail, DecodeSummary, ManagedChild, ModelPixels};
use crate::frame_windows::{FrameWindows, RawWindow, MAX_FULL_VIDEO_STRIDE};
pub use backend::{select_decode_backend, DecodeBackend, DecodePreference, DecodeSelection};
use serde::Serialize;
use std::{
    fs::{self, DirBuilder},
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender},
        Arc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(windows)]
use std::net::{TcpListener as DecodeListener, TcpStream as DecodeStream};
#[cfg(unix)]
use std::os::unix::{
    fs::DirBuilderExt,
    net::{UnixListener as DecodeListener, UnixStream as DecodeStream},
};

const POLL: Duration = Duration::from_millis(100);
const INIT_LIMIT: Duration = Duration::from_secs(30);
const IDLE_LIMIT: Duration = Duration::from_secs(120);
pub const MAX_NATIVE_RGB_FRAME_BYTES: usize = crate::frame_windows::MAX_FRAME_BYTES;

/// A synchronized 30-FPS window; both inputs use the canonical FFmpeg CPU
/// scaling/format filters and have identical source-frame indices.
#[derive(Debug)]
pub struct PrimaryFrameBatch {
    pub first_frame: u64,
    pub gray: Vec<u8>,
    pub rgb: Vec<u8>,
    pub native: Option<NativeRgbBatch>,
}

#[derive(Debug)]
pub struct NativeRgbBatch {
    pub rgb: Arc<Vec<u8>>,
    pub width: usize,
    pub height: usize,
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
            let mut directory = DirBuilder::new();
            directory.recursive(false);
            #[cfg(unix)]
            directory.mode(0o700);
            match directory.create(&path) {
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
        #[cfg(unix)]
        let _ = fs::remove_file(self.0.join("gray.sock"));
        #[cfg(unix)]
        let _ = fs::remove_file(self.0.join("rgb.sock"));
        #[cfg(unix)]
        let _ = fs::remove_file(self.0.join("native.sock"));
        let _ = fs::remove_dir(&self.0);
    }
}

#[cfg(unix)]
fn bind_decode_stream(path: &Path) -> Result<(DecodeListener, String), String> {
    let listener = DecodeListener::bind(path)
        .map_err(|error| format!("Bind FFmpeg output stream: {error}"))?;
    Ok((listener, format!("unix://{}", path.display())))
}

#[cfg(windows)]
fn bind_decode_stream(_path: &Path) -> Result<(DecodeListener, String), String> {
    let listener = DecodeListener::bind(("127.0.0.1", 0))
        .map_err(|error| format!("Bind loopback FFmpeg stream: {error}"))?;
    let address = listener
        .local_addr()
        .map_err(|error| format!("Resolve FFmpeg loopback address: {error}"))?;
    Ok((listener, format!("tcp://{address}")))
}

fn accept(
    listener: &DecodeListener,
    child: &mut ManagedChild,
    cancelled: &AtomicBool,
) -> Result<DecodeStream, String> {
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
    stream: DecodeStream,
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
    stream: DecodeStream,
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
        &DecodeSelection {
            backend: DecodeBackend::Software,
            fallback_reason: None,
            rotation_filter: "",
        },
        0,
        None,
        None,
        None,
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
        &decision,
        0,
        None,
        None,
        None,
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
    cpu_threads: usize,
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
        &decision,
        cpu_threads,
        None,
        None,
        None,
        consume,
        on_tail,
    )?;
    Ok(PrimaryDecodeSummary {
        decode,
        backend: decision.backend,
        fallback_reason: decision.fallback_reason,
    })
}

#[expect(clippy::too_many_arguments)]
pub fn decode_full_video_shared_with_native_and_tail(
    source: &Path,
    stride: usize,
    preference: DecodePreference,
    cpu_threads: usize,
    cancelled: &AtomicBool,
    native: bool,
    consume: impl FnMut(PrimaryFrameBatch) -> Result<(), String>,
    on_tail: impl FnMut(&[u8], u64) -> Result<(), String>,
) -> Result<PrimaryDecodeSummary, String> {
    decode_full_video_shared_with_native_and_tail_in_range(
        source,
        stride,
        preference,
        cpu_threads,
        None,
        cancelled,
        native,
        consume,
        on_tail,
    )
}

#[expect(clippy::too_many_arguments)]
pub fn decode_full_video_shared_with_native_and_tail_in_range(
    source: &Path,
    stride: usize,
    preference: DecodePreference,
    cpu_threads: usize,
    range: Option<super::AnalysisRange>,
    cancelled: &AtomicBool,
    native: bool,
    consume: impl FnMut(PrimaryFrameBatch) -> Result<(), String>,
    on_tail: impl FnMut(&[u8], u64) -> Result<(), String>,
) -> Result<PrimaryDecodeSummary, String> {
    if stride != 9 {
        return Err("Original pose stream requires disjoint nine-frame model windows".into());
    }
    let dimensions = if native {
        let info = crate::probe(source)?;
        let (width, height) = (info.width as usize, info.height as usize);
        let bytes = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .filter(|bytes| *bytes <= MAX_NATIVE_RGB_FRAME_BYTES)
            .ok_or("Native pose frame exceeds 32 MiB streaming limit")?;
        if bytes == 0 {
            return Err("Invalid native pose frame dimensions".into());
        }
        Some((width, height))
    } else {
        None
    };
    let decision = select_decode_backend(source, preference, cancelled)?;
    let decode = decode_full_video_shared_impl(
        source,
        stride,
        cancelled,
        &decision,
        cpu_threads,
        None,
        dimensions,
        range,
        consume,
        on_tail,
    )?;
    Ok(PrimaryDecodeSummary {
        decode,
        backend: decision.backend,
        fallback_reason: decision.fallback_reason,
    })
}

#[derive(Clone, Copy)]
pub struct PrimaryPreviewOutput<'a> {
    pub path: &'a Path,
    pub hardware: bool,
}

#[expect(clippy::too_many_arguments)]
pub fn decode_full_video_shared_with_preview(
    source: &Path,
    stride: usize,
    preference: DecodePreference,
    cpu_threads: usize,
    cancelled: &AtomicBool,
    preview: Option<PrimaryPreviewOutput<'_>>,
    consume: impl FnMut(PrimaryFrameBatch) -> Result<(), String>,
    on_tail: impl FnMut(&[u8], u64) -> Result<(), String>,
) -> Result<PrimaryDecodeSummary, String> {
    if stride != 9 {
        return Err("Preview requires disjoint nine-frame model windows".into());
    }
    let decision = select_decode_backend(source, preference, cancelled)?;
    let decode = decode_full_video_shared_impl(
        source,
        stride,
        cancelled,
        &decision,
        cpu_threads,
        preview,
        None,
        None,
        consume,
        on_tail,
    )?;
    Ok(PrimaryDecodeSummary {
        decode,
        backend: decision.backend,
        fallback_reason: decision.fallback_reason,
    })
}

#[cfg(unix)]
pub(super) fn auto_decode_input(
    source: &Path,
    cancelled: &AtomicBool,
) -> Result<(Command, String), String> {
    let decision = select_decode_backend(source, DecodePreference::Auto, cancelled)?;
    eprintln!(
        "Video decoder: {:?}; fallback={}",
        decision.backend,
        decision.fallback_reason.as_deref().unwrap_or("none")
    );
    Ok(decode_input_command(source, &decision, 0, None))
}

#[cfg(unix)]
pub(super) fn auto_decode_input_in_range(
    source: &Path,
    cancelled: &AtomicBool,
    range: Option<super::AnalysisRange>,
) -> Result<(Command, String), String> {
    let decision = select_decode_backend(source, DecodePreference::Auto, cancelled)?;
    Ok(decode_input_command(source, &decision, 0, range))
}

fn decode_input_command(
    source: &Path,
    decision: &DecodeSelection,
    cpu_threads: usize,
    range: Option<super::AnalysisRange>,
) -> (Command, String) {
    let mut command = Command::new("ffmpeg");
    let decoder_threads = if decision.backend == DecodeBackend::Nvdec {
        1
    } else {
        cpu_threads
    };
    command.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        &decoder_threads.to_string(),
    ]);
    let prefix = if decision.backend == DecodeBackend::Nvdec {
        // FFmpeg copies the original display matrix into the encoded MP4 even
        // after we manually transpose NVDEC frames. Clear it on the INPUT;
        // the explicit rotation_filter below still rotates the actual pixels.
        command.args([
            "-noautorotate",
            "-display_rotation:v:0",
            "0",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
        ]);
        format!("hwdownload,format=nv12,{}", decision.rotation_filter)
    } else {
        String::new()
    };
    if let Some(range) = range {
        range.seek_input(&mut command);
    }
    command.arg("-i").arg(source);
    (command, prefix)
}

fn shared_decode_command(
    source: &Path,
    decision: &DecodeSelection,
    cpu_threads: usize,
    range: Option<super::AnalysisRange>,
) -> Command {
    shared_decode_command_with_preview(source, decision, cpu_threads, false, false, range)
}

fn shared_decode_command_with_preview(
    source: &Path,
    decision: &DecodeSelection,
    cpu_threads: usize,
    preview: bool,
    native: bool,
    range: Option<super::AnalysisRange>,
) -> Command {
    let (mut command, prefix) = decode_input_command(source, decision, cpu_threads, range);
    let input = if preview {
        format!("[0:v]{prefix}split=2[model][preview];[model]")
    } else {
        format!("[0:v]{prefix}")
    };
    let mut filters = if native {
        format!("{input}fps=30,split=3[ball][players][original];[ball]scale=512:288:flags=bilinear,format=gray[g];[players]scale=640:360:flags=bilinear,format=rgb24[r];[original]format=rgb24[n]")
    } else {
        format!("{input}fps=30,split=2[ball][players];[ball]scale=512:288:flags=bilinear,format=gray[g];[players]scale=640:360:flags=bilinear,format=rgb24[r]")
    };
    if preview {
        filters.push_str(";[preview]scale=w='min(1280,iw)':h='min(1280,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1,format=yuv420p[p]");
    }
    command.args([
        "-filter_complex_threads",
        &cpu_threads.to_string(),
        "-filter_complex",
        &filters,
    ]);
    command
}

#[expect(clippy::too_many_arguments)]
fn decode_full_video_shared_impl(
    source: &Path,
    stride: usize,
    cancelled: &AtomicBool,
    decision: &DecodeSelection,
    cpu_threads: usize,
    preview: Option<PrimaryPreviewOutput<'_>>,
    native: Option<(usize, usize)>,
    range: Option<super::AnalysisRange>,
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

    if native.is_some() && preview.is_some() {
        return Err("Native source and encoded preview cannot share a decoder".into());
    }
    let sockets = SocketDir::new()?;
    let gray_path = sockets.0.join("gray.sock");
    let rgb_path = sockets.0.join("rgb.sock");
    let (gray_listener, gray_endpoint) = bind_decode_stream(&gray_path)
        .map_err(|error| format!("Bind gray decode stream: {error}"))?;
    let (rgb_listener, rgb_endpoint) = bind_decode_stream(&rgb_path)
        .map_err(|error| format!("Bind RGB decode stream: {error}"))?;
    let native_listener = if native.is_some() {
        Some(
            bind_decode_stream(&sockets.0.join("native.sock"))
                .map_err(|error| format!("Bind native RGB stream: {error}"))?,
        )
    } else {
        None
    };
    gray_listener
        .set_nonblocking(true)
        .map_err(|error| format!("Gray socket: {error}"))?;
    rgb_listener
        .set_nonblocking(true)
        .map_err(|error| format!("RGB socket: {error}"))?;
    if let Some((listener, _)) = native_listener.as_ref() {
        listener
            .set_nonblocking(true)
            .map_err(|error| format!("Native RGB socket: {error}"))?;
    }

    let mut command = if preview.is_some() {
        shared_decode_command_with_preview(source, decision, cpu_threads, true, false, range)
    } else {
        if native.is_some() {
            shared_decode_command_with_preview(source, decision, cpu_threads, false, true, range)
        } else {
            shared_decode_command(source, decision, cpu_threads, range)
        }
    };
    command
        .args(["-map", "[g]", "-an", "-sn", "-dn"])
        .args(["-f", "rawvideo", "-pix_fmt", "gray"])
        .arg(&gray_endpoint)
        .args(["-map", "[r]", "-an", "-sn", "-dn"])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg(&rgb_endpoint);
    if let Some((_, native_endpoint)) = native_listener.as_ref() {
        command
            .args([
                "-map", "[n]", "-an", "-sn", "-dn", "-f", "rawvideo", "-pix_fmt", "rgb24",
            ])
            .arg(native_endpoint);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    if let Some(preview) = preview {
        command.args([
            "-map", "[p]", "-map", "0:a:0?", "-sn", "-dn", "-c:a", "aac", "-b:a", "128k",
        ]);
        if preview.hardware {
            command.args([
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-rc",
                "vbr",
                "-cq",
                "23",
                "-b:v",
                "0",
            ]);
        } else {
            command.args([
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2",
            ]);
        }
        command
            .args([
                "-fps_mode",
                "passthrough",
                "-force_key_frames",
                "expr:gte(t,n_forced*1)",
                "-map_metadata",
                "-1",
                "-metadata:s:v:0",
                "rotate=0",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                "-y",
            ])
            .arg(preview.path);
    }
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
        let native_stream = native_listener
            .as_ref()
            .map(|(listener, _)| accept(listener, &mut child, cancelled))
            .transpose()?;
        let (gw, gh, gc, _) = ModelPixels::BallGray.dimensions();
        let (rw, rh, rc, _) = ModelPixels::PlayersRgb.dimensions();
        thread::scope(|scope| {
            let (gray_tx, gray_rx) = mpsc::sync_channel(2);
            let (rgb_tx, rgb_rx) = mpsc::sync_channel(2);
            scope.spawn(|| drain(gray, gw * gh * gc, stride, cancelled, gray_tx));
            scope.spawn(|| drain(rgb, rw * rh * rc, stride, cancelled, rgb_tx));
            let native_rx = native_stream.map(|stream| {
                let (tx, rx) = mpsc::sync_channel(1);
                let (width, height) = native.expect("Native dimensions accompany socket");
                scope.spawn(move || drain(stream, width * height * 3, stride, cancelled, tx));
                rx
            });
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
                            let original = if let Some(receiver) = native_rx.as_ref() {
                                let (width, height) = native.ok_or("Missing native dimensions")?;
                                match receive(receiver, cancelled)? {
                                    WindowEvent::Data(window) if window.first_frame == expected => {
                                        Some(NativeRgbBatch {
                                            rgb: Arc::new(window.data),
                                            width,
                                            height,
                                        })
                                    }
                                    _ => {
                                        return Err(
                                            "Native stream window disagrees with detector timeline"
                                                .into(),
                                        )
                                    }
                                }
                            } else {
                                None
                            };
                            consume(PrimaryFrameBatch {
                                first_frame: expected,
                                gray: gray.data,
                                rgb: rgb.data,
                                native: original,
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
                            if let Some(receiver) = native_rx.as_ref() {
                                let (width, height) = native.ok_or("Missing native dimensions")?;
                                let expected_tail = gray_frames
                                    .checked_sub(windows as u64 * stride as u64)
                                    .filter(|count| *count < stride as u64)
                                    .ok_or("Invalid native stream tail")?
                                    as usize;
                                match receive(receiver, cancelled)? {
                                    WindowEvent::Eof(frames, tail)
                                        if frames == gray_frames
                                            && tail.len() == expected_tail * width * height * 3 => {
                                    }
                                    _ => {
                                        return Err(
                                            "Native stream EOF differs from detector timeline"
                                                .into(),
                                        )
                                    }
                                }
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
                let finishing = Instant::now();
                let status = loop {
                    if cancelled.load(Ordering::Acquire) {
                        return Err("cancelled".into());
                    }
                    if finishing.elapsed() > IDLE_LIMIT {
                        return Err("Shared FFmpeg finalization timed out".into());
                    }
                    if let Some(status) = child
                        .0
                        .try_wait()
                        .map_err(|error| format!("Wait shared FFmpeg: {error}"))?
                    {
                        break status;
                    }
                    thread::sleep(POLL);
                };
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
            drop(native_rx);
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
