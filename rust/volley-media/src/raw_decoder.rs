//! One bounded FFmpeg raw-video stream feeding overlapped 9-frame Rust
//! windows. Never seeks for every window or writes an analysis cache.
//! Input paths must be validated by the calling project/preparation layer.
#[cfg(any(unix, windows))]
mod shared;
#[cfg(any(unix, windows))]
pub use shared::{
    decode_full_video_shared, decode_full_video_shared_with_backend,
    decode_full_video_shared_with_backend_and_tail, decode_full_video_shared_with_native_and_tail,
    decode_full_video_shared_with_native_and_tail_in_range, decode_full_video_shared_with_preview,
    select_decode_backend, DecodeBackend, DecodePreference, DecodeSelection, PrimaryDecodeSummary,
    PrimaryFrameBatch, PrimaryPreviewOutput, MAX_NATIVE_RGB_FRAME_BYTES,
};

use crate::frame_windows::{FrameWindows, RawWindow, CLIP_FRAMES, MAX_FRAMES};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    io::{self, Read},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, RecvTimeoutError},
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisRange {
    pub start_sec: f64,
    pub end_sec: f64,
}
impl AnalysisRange {
    pub fn validate(self, duration: f64) -> Result<(), String> {
        if !duration.is_finite()
            || !self.start_sec.is_finite()
            || !self.end_sec.is_finite()
            || self.start_sec < 0.0
            || self.end_sec <= self.start_sec
            || self.end_sec > duration + 0.5
            || self.end_sec - self.start_sec < 9.0 / 30.0
        {
            return Err("Invalid per-video analysis interval".into());
        }
        Ok(())
    }

    pub(crate) fn seek_input(self, command: &mut Command) {
        command.args([
            "-ss",
            &format!("{:.6}", self.start_sec),
            "-t",
            &format!("{:.6}", self.end_sec - self.start_sec),
        ]);
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ModelPixels {
    BallGray,
    PlayersRgb,
}
impl ModelPixels {
    pub fn dimensions(self) -> (usize, usize, usize, &'static str) {
        match self {
            Self::BallGray => (512, 288, 1, "gray"),
            Self::PlayersRgb => (640, 360, 3, "rgb24"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeSummary {
    pub first_frame: u64,
    pub windows: usize,
    pub decoded_frames: u64,
    pub stride: usize,
}

struct ManagedChild(Child);
impl Drop for ManagedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

const TAIL_BYTES: usize = 4096;

/// A hung FFmpeg decoder must not pin the inference worker on a blocking pipe
/// read forever. A scoped thread drains the pipe into bounded chunks while
/// the consumer waits with a timeout: cancellation interrupts within 200 ms
/// and a fully silent decoder fails after the idle limit. Dropping the reader
/// disconnects the channel, which also stops the drain thread.
struct GuardedPipe<'a> {
    receiver: Receiver<io::Result<Vec<u8>>>,
    cancelled: &'a AtomicBool,
    idle_limit: Duration,
    pending: Vec<u8>,
    offset: usize,
}
impl Read for GuardedPipe<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut last_activity = Instant::now();
        loop {
            if self.cancelled.load(Ordering::Acquire) {
                // FrameWindows retries Interrupted (EINTR) but cancellation is terminal.
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "cancelled",
                ));
            }
            if self.offset < self.pending.len() {
                let count = (self.pending.len() - self.offset).min(buf.len());
                buf[..count].copy_from_slice(&self.pending[self.offset..self.offset + count]);
                self.offset += count;
                if self.offset == self.pending.len() {
                    self.pending.clear();
                    self.offset = 0;
                }
                return Ok(count);
            }
            match self.receiver.recv_timeout(Duration::from_millis(200)) {
                Ok(Ok(chunk)) => {
                    if chunk.is_empty() {
                        return Ok(0);
                    }
                    self.pending = chunk;
                    self.offset = 0;
                    last_activity = Instant::now();
                }
                Ok(Err(error)) => return Err(error),
                Err(RecvTimeoutError::Timeout) => {
                    if last_activity.elapsed() >= self.idle_limit {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "FFmpeg decoder produced no data within the idle limit",
                        ));
                    }
                }
                Err(RecvTimeoutError::Disconnected) => return Ok(0),
            }
        }
    }
}
const DECODER_IDLE_LIMIT: Duration = Duration::from_secs(120);

fn stderr_tail(mut error_pipe: impl Read + Send + 'static) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut tail = VecDeque::with_capacity(TAIL_BYTES);
        let mut chunk = [0_u8; 1024];
        while let Ok(count) = error_pipe.read(&mut chunk) {
            if count == 0 {
                break;
            }
            for &byte in &chunk[..count] {
                if tail.len() == TAIL_BYTES {
                    tail.pop_front();
                }
                tail.push_back(byte);
            }
        }
        String::from_utf8_lossy(&tail.into_iter().collect::<Vec<_>>()).into_owned()
    })
}

fn checked_total_frames(first: u64, windows: usize, stride: usize) -> Result<usize, String> {
    if !(1..=120).contains(&windows) || !(1..=CLIP_FRAMES).contains(&stride) {
        return Err("Select 1..120 windows with a temporal stride of 1..9".into());
    }
    let total = CLIP_FRAMES
        .checked_add(
            (windows - 1)
                .checked_mul(stride)
                .ok_or("Window count overflow")?,
        )
        .ok_or("Window count overflow")?;
    let last = first
        .checked_add(total as u64)
        .ok_or("Frame index overflow")?;
    if last > MAX_FRAMES {
        return Err("Requested frames exceed the maximum 48-hour 30fps source".into());
    }
    Ok(total)
}

fn auto_decode_input(source: &Path, cancelled: &AtomicBool) -> Result<(Command, String), String> {
    auto_decode_input_in_range(source, cancelled, None)
}

fn auto_decode_input_in_range(
    source: &Path,
    cancelled: &AtomicBool,
    range: Option<AnalysisRange>,
) -> Result<(Command, String), String> {
    #[cfg(unix)]
    {
        shared::auto_decode_input_in_range(source, cancelled, range)
    }
    #[cfg(not(unix))]
    {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let mut command = Command::new("ffmpeg");
        command.args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "0",
            "-hwaccel",
            "auto",
        ]);
        if let Some(range) = range {
            range.seek_input(&mut command);
        }
        command.arg("-i").arg(source);
        Ok((command, String::new()))
    }
}

/// The callback is invoked synchronously per 9-frame window, allowing one
/// persistent ONNX session to consume a stream with bounded memory. A failed
/// callback or cancellation kills and reaps FFmpeg; no partial result is valid.
pub fn decode_windows(
    source: &Path,
    pixels: ModelPixels,
    first_frame: u64,
    windows: usize,
    stride: usize,
    cancelled: &AtomicBool,
    mut consume: impl FnMut(RawWindow) -> Result<(), String>,
) -> Result<DecodeSummary, String> {
    let total_frames = checked_total_frames(first_frame, windows, stride)?;
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    if !source.is_file() {
        return Err("Prepared source video is not a file".into());
    }
    let (w, h, channels, format) = pixels.dimensions();
    let bytes_per_frame = w * h * channels;
    let seek = first_frame as f64 / 30.0;
    let (mut ff, prefix) = auto_decode_input(source, cancelled)?;
    ff.args(["-ss", &format!("{seek:.6}")])
        .args(["-map", "0:v:0", "-an", "-sn", "-dn"])
        .args([
            "-vf",
            &format!("{prefix}fps=30,scale={w}:{h}:flags=bilinear,format={format}"),
        ])
        .args(["-frames:v", &total_frames.to_string()])
        .args(["-pix_fmt", format, "-f", "rawvideo", "pipe:1"]);
    ff.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = ManagedChild(
        ff.spawn()
            .map_err(|e| format!("FFmpeg launch failed: {e}"))?,
    );
    let mut stdout = child.0.stdout.take().ok_or("FFmpeg stdout missing")?;
    let stderr = child.0.stderr.take().ok_or("FFmpeg stderr missing")?;
    let error_reader = stderr_tail(stderr);
    let result = thread::scope(|scope| {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<io::Result<Vec<u8>>>(8);
        scope.spawn(move || {
            let mut chunk = vec![0_u8; 64 * 1024];
            loop {
                match stdout.read(&mut chunk) {
                    Ok(0) => {
                        let _ = sender.send(Ok(Vec::new()));
                        break;
                    }
                    Ok(count) => {
                        if sender.send(Ok(chunk[..count].to_vec())).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                        break;
                    }
                }
            }
        });
        let guarded = GuardedPipe {
            receiver,
            cancelled,
            idle_limit: DECODER_IDLE_LIMIT,
            pending: Vec::new(),
            offset: 0,
        };
        let outcome = (|| -> Result<DecodeSummary, String> {
            let mut reader = FrameWindows::new(guarded, bytes_per_frame, stride)
                .map_err(|e| format!("Raw frame reader: {e}"))?;
            let mut produced = 0;
            while produced < windows {
                if cancelled.load(Ordering::Acquire) {
                    return Err("cancelled".into());
                }
                let Some(window) = reader
                    .next_window()
                    .map_err(|e| format!("Decode raw frame: {e}"))?
                else {
                    return Err(
                        "Decoder ended before the requested number of complete windows".into(),
                    );
                };
                if window.first_frame != produced as u64 * stride as u64 {
                    return Err("Decoder frame sequence is inconsistent".into());
                }
                consume(RawWindow {
                    first_frame: first_frame + window.first_frame,
                    data: window.data,
                })?;
                produced += 1;
            }
            // FFmpeg has a finite frame limit; its normal EOF and exit code are
            // verified before any result is published.
            let status = child
                .0
                .wait()
                .map_err(|e| format!("Wait for FFmpeg: {e}"))?;
            if !status.success() {
                return Err(format!("FFmpeg returned {status}"));
            }
            Ok(DecodeSummary {
                first_frame,
                windows: produced,
                decoded_frames: reader.decoded_frames(),
                stride,
            })
        })();
        if outcome.is_err() {
            // Kill before the scope joins the drain thread: a hung child would
            // otherwise keep the pipe open and block the join forever.
            let _ = child.0.kill();
            let _ = child.0.wait();
        }
        outcome
    });
    let errors = error_reader
        .join()
        .unwrap_or_else(|_| "FFmpeg stderr thread panicked".into());
    result.map_err(|e| {
        if errors.trim().is_empty() {
            e
        } else {
            format!(
                "{e}: {}",
                errors.trim().chars().take(500).collect::<String>()
            )
        }
    })
}

/// Stream an entire prepared video without loading it into RAM or asking
/// FFmpeg to seek for each window. Only whole nine-frame clips are emitted;
/// any final 1..8 frames that cannot form a full window are left unused.
/// The result is valid only after the FFmpeg process exits successfully.
pub fn decode_full_video(
    source: &Path,
    pixels: ModelPixels,
    stride: usize,
    cancelled: &AtomicBool,
    consume: impl FnMut(RawWindow) -> Result<(), String>,
) -> Result<DecodeSummary, String> {
    decode_full_video_in_range(source, pixels, stride, None, cancelled, consume)
}

pub fn decode_full_video_in_range(
    source: &Path,
    pixels: ModelPixels,
    stride: usize,
    range: Option<AnalysisRange>,
    cancelled: &AtomicBool,
    mut consume: impl FnMut(RawWindow) -> Result<(), String>,
) -> Result<DecodeSummary, String> {
    if !(1..=crate::frame_windows::MAX_FULL_VIDEO_STRIDE).contains(&stride) {
        return Err("Full-video decode stride must be between 1 and 30".into());
    }
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    if !source.is_file() {
        return Err("Prepared source video is not a file".into());
    }
    let (w, h, channels, format) = pixels.dimensions();
    if let Some(range) = range {
        range.validate(crate::probe(source)?.duration_sec)?;
    }
    let (mut ff, prefix) = auto_decode_input_in_range(source, cancelled, range)?;
    ff.args(["-map", "0:v:0", "-an", "-sn", "-dn"])
        .args([
            "-vf",
            &format!("{prefix}fps=30,scale={w}:{h}:flags=bilinear,format={format}"),
        ])
        .args(["-pix_fmt", format, "-f", "rawvideo", "pipe:1"]);
    ff.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = ManagedChild(
        ff.spawn()
            .map_err(|e| format!("FFmpeg full-video launch failed: {e}"))?,
    );
    let mut stdout = child.0.stdout.take().ok_or("FFmpeg stdout missing")?;
    let stderr = child.0.stderr.take().ok_or("FFmpeg stderr missing")?;
    let errors_reader = stderr_tail(stderr);
    let result = thread::scope(|scope| {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<io::Result<Vec<u8>>>(8);
        scope.spawn(move || {
            let mut chunk = vec![0_u8; 64 * 1024];
            loop {
                match stdout.read(&mut chunk) {
                    Ok(0) => {
                        let _ = sender.send(Ok(Vec::new()));
                        break;
                    }
                    Ok(count) => {
                        if sender.send(Ok(chunk[..count].to_vec())).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                        break;
                    }
                }
            }
        });
        let guarded = GuardedPipe {
            receiver,
            cancelled,
            idle_limit: DECODER_IDLE_LIMIT,
            pending: Vec::new(),
            offset: 0,
        };
        let outcome = (|| -> Result<DecodeSummary, String> {
            let mut reader = FrameWindows::new(guarded, w * h * channels, stride)
                .map_err(|e| format!("Video frame reader: {e}"))?;
            let mut produced = 0usize;
            loop {
                if cancelled.load(Ordering::Acquire) {
                    return Err("cancelled".into());
                }
                let Some(window) = reader
                    .next_window()
                    .map_err(|e| format!("Video raw-frame decoding: {e}"))?
                else {
                    break;
                };
                if window.first_frame != (produced as u64) * stride as u64 {
                    return Err("Full-video frame sequence is not monotonic".into());
                }
                consume(window)?;
                produced += 1;
            }
            if cancelled.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }
            if produced == 0 {
                return Err("Video does not contain a complete nine-frame model window".into());
            }
            let status = child.0.wait().map_err(|e| format!("Wait FFmpeg: {e}"))?;
            if !status.success() {
                return Err(format!("FFmpeg exited with {status}"));
            }
            Ok(DecodeSummary {
                first_frame: 0,
                windows: produced,
                decoded_frames: reader.decoded_frames(),
                stride,
            })
        })();
        if outcome.is_err() {
            // Kill before the scope joins the drain thread: a hung child would
            // otherwise keep the pipe open and block the join forever.
            let _ = child.0.kill();
            let _ = child.0.wait();
        }
        outcome
    });
    let reported = errors_reader
        .join()
        .unwrap_or_else(|_| "FFmpeg stderr reader panicked".into());
    result.map_err(|e| {
        if reported.trim().is_empty() {
            e
        } else {
            format!(
                "{e}: {}",
                reported.trim().chars().take(500).collect::<String>()
            )
        }
    })
}

const RGB_HEARTBEAT_FRAMES: u64 = 300;

fn balanced_ranges_expression(ranges: &[(u64, u64)]) -> String {
    match ranges {
        [] => "0".into(),
        [(left, right)] => format!("between(n,{left},{right})"),
        _ => {
            let (left, right) = ranges.split_at(ranges.len() / 2);
            format!(
                "({}+{})",
                balanced_ranges_expression(left),
                balanced_ranges_expression(right)
            )
        }
    }
}

fn balanced_selection_expression(ranges: &[(u64, u64)]) -> String {
    let expression = balanced_ranges_expression(ranges);
    match ranges.last() {
        Some(&(_, last)) => {
            format!("({expression})+lte(n,{last})*not(mod(n,{RGB_HEARTBEAT_FRAMES}))")
        }
        None => expression,
    }
}

fn rgb_range_contains(ranges: &[(u64, u64)], index: u64) -> bool {
    let i = ranges.partition_point(|&(left, _)| left <= index);
    i > 0 && index <= ranges[i - 1].1
}

fn rgb_heartbeat(ranges: &[(u64, u64)], index: u64) -> bool {
    index.is_multiple_of(RGB_HEARTBEAT_FRAMES)
        && ranges.last().is_some_and(|&(_, last)| index <= last)
}

fn selected_rgb_output_count(ranges: &[(u64, u64)]) -> u64 {
    let selected: u64 = ranges.iter().map(|&(left, right)| right - left + 1).sum();
    let extra = ranges.last().map_or(0, |&(_, last)| {
        (0..=last)
            .step_by(RGB_HEARTBEAT_FRAMES as usize)
            .filter(|&index| !rgb_range_contains(ranges, index))
            .count() as u64
    });
    selected + extra
}

const RGB_SEEK_GAP_FRAMES: u64 = 900;
const RGB_SEEK_PREROLL_FRAMES: u64 = 60;

struct RgbSeekWindow<'a> {
    seek_frame: u64,
    ranges: &'a [(u64, u64)],
}

fn plan_rgb_seek_windows(selected: &[(u64, u64)]) -> Vec<RgbSeekWindow<'_>> {
    let mut windows = Vec::new();
    let mut first = 0;
    for i in 1..=selected.len() {
        if i == selected.len()
            || i - first == 256
            || selected[i].0.saturating_sub(selected[i - 1].1) >= RGB_SEEK_GAP_FRAMES
        {
            let start = selected[first].0;
            let seek_frame = if first == 0 && start < RGB_SEEK_GAP_FRAMES {
                0
            } else {
                start.saturating_sub(RGB_SEEK_PREROLL_FRAMES) / 30 * 30
            };
            windows.push(RgbSeekWindow {
                seek_frame,
                ranges: &selected[first..i],
            });
            first = i;
        }
    }
    windows
}

fn supports_rgb_seek(payload: &serde_json::Value, info: &crate::VideoInfo) -> bool {
    let Some(video) = payload["streams"].as_array().and_then(|streams| {
        streams
            .iter()
            .find(|stream| stream["codec_type"] == "video")
    }) else {
        return false;
    };
    matches!(info.codec.as_str(), "h264" | "hevc")
        && info.fps.is_some_and(|fps| (15.0..=120.0).contains(&fps))
        && payload["format"]["format_name"]
            .as_str()
            .is_some_and(|name| {
                name.split(',')
                    .any(|format| format == "mov" || format == "mp4")
            })
        && video["time_base"].as_str().is_some_and(|base| {
            base.strip_prefix("1/")
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|denominator| denominator > 0)
        })
}

fn read_first_rgb_pts(stderr: impl Read) -> Result<Option<i64>, String> {
    use std::io::BufRead;
    let mut reader = io::BufReader::new(stderr);
    let mut line = Vec::with_capacity(1024);
    let mut pts = None;
    loop {
        let bytes = reader.fill_buf().map_err(|error| error.to_string())?;
        if bytes.is_empty() {
            break;
        }
        let count = bytes
            .iter()
            .position(|&byte| byte == b'\n')
            .map_or(bytes.len(), |index| index + 1);
        if pts.is_none() {
            let remaining = 1024usize.saturating_sub(line.len());
            line.extend_from_slice(&bytes[..count.min(remaining)]);
            if bytes[count - 1] == b'\n' {
                let text = String::from_utf8_lossy(&line);
                if text.contains("Parsed_showinfo") && text.contains("n:") {
                    pts = text.split_once("pts:").and_then(|(_, value)| {
                        value.split_whitespace().next()?.parse::<i64>().ok()
                    });
                }
                line.clear();
            }
        }
        reader.consume(count);
    }
    Ok(pts)
}

fn original_rgb_first_pts(
    source: &Path,
    range: Option<AnalysisRange>,
    cancelled: &AtomicBool,
) -> Result<i64, String> {
    let (mut ff, prefix) = auto_decode_input_in_range(source, cancelled, range)?;
    ff.args([
        "-loglevel",
        "info",
        "-nostats",
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
    ])
    .args(["-filter_threads", "2"])
    .args(["-vf", &format!("{prefix}fps=30,select='eq(n,0)',showinfo")])
    .args(["-frames:v", "1", "-f", "null", "-"])
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::piped());
    let mut child = ManagedChild(ff.spawn().map_err(|error| error.to_string())?);
    let stderr = child
        .0
        .stderr
        .take()
        .ok_or("Missing RGB timestamp probe stderr")?;
    let reader = thread::spawn(move || read_first_rgb_pts(stderr));
    let start = Instant::now();
    let status = loop {
        if cancelled.load(Ordering::Acquire) || start.elapsed() >= DECODER_IDLE_LIMIT {
            let _ = child.0.kill();
            let _ = child.0.wait();
            let _ = reader.join();
            return Err("Original RGB timestamp probe cancelled or timed out".into());
        }
        match child.0.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => thread::sleep(Duration::from_millis(25)),
            Err(error) => {
                let _ = child.0.kill();
                let _ = child.0.wait();
                let _ = reader.join();
                return Err(error.to_string());
            }
        }
    };
    let pts = reader
        .join()
        .map_err(|_| "RGB timestamp probe panicked")??;
    if !status.success() {
        return Err(format!("Original RGB timestamp probe exited with {status}"));
    }
    pts.ok_or_else(|| "Missing first normalized RGB timestamp".into())
}

fn configure_original_rgb_output(
    ff: &mut Command,
    prefix: &str,
    expected_frames: u64,
    selection: Option<&str>,
    pts_shift: Option<i64>,
) {
    let filter = match (selection, pts_shift) {
        (Some(expression), Some(shift)) => {
            let expression = expression.replace("n,", "pts,");
            format!("{prefix}fps=30,setpts=PTS-({shift}),select='if(isnan(prev_selected_pts),eq(pts,0),{expression})',format=rgb24")
        }
        (Some(expression), None) => format!("{prefix}fps=30,select='{expression}',format=rgb24"),
        (None, _) => format!("{prefix}fps=30,format=rgb24"),
    };
    if selection.is_some() {
        ff.args(["-fps_mode", "passthrough"]);
    }
    ff.args(["-map", "0:v:0", "-an", "-sn", "-dn"])
        .args(["-filter_threads", "2"])
        .args(["-vf", &filter])
        .args(["-frames:v", &expected_frames.to_string()])
        .args(["-threads:v", "1"])
        .args(["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]);
}

pub fn decode_original_frames(
    source: &Path,
    expected_frames: u64,
    cancelled: &AtomicBool,
    consume: impl FnMut(u64, &[u8], usize, usize) -> Result<(), String>,
) -> Result<(), String> {
    decode_original_frames_in_range(source, expected_frames, None, cancelled, consume)
}

pub fn decode_original_frames_in_range(
    source: &Path,
    expected_frames: u64,
    range: Option<AnalysisRange>,
    cancelled: &AtomicBool,
    consume: impl FnMut(u64, &[u8], usize, usize) -> Result<(), String>,
) -> Result<(), String> {
    decode_original_frames_impl(source, expected_frames, range, None, cancelled, consume)
}

pub fn decode_original_frames_selected_in_range(
    source: &Path,
    expected_frames: u64,
    range: Option<AnalysisRange>,
    selected: &[(u64, u64)],
    cancelled: &AtomicBool,
    consume: impl FnMut(u64, &[u8], usize, usize) -> Result<(), String>,
) -> Result<(), String> {
    if selected.is_empty()
        || selected
            .iter()
            .any(|&(left, right)| left > right || right >= expected_frames)
        || selected.windows(2).any(|pair| pair[0].1 >= pair[1].0)
    {
        return Err("Invalid original RGB selection".into());
    }
    decode_original_frames_impl(
        source,
        expected_frames,
        range,
        Some(selected),
        cancelled,
        consume,
    )
}

fn decode_original_frames_impl(
    source: &Path,
    expected_frames: u64,
    range: Option<AnalysisRange>,
    selected: Option<&[(u64, u64)]>,
    cancelled: &AtomicBool,
    mut consume: impl FnMut(u64, &[u8], usize, usize) -> Result<(), String>,
) -> Result<(), String> {
    if expected_frames == 0 || expected_frames > MAX_FRAMES {
        return Err("Invalid original RGB frame count".into());
    }
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let payload = crate::ffprobe(source)?;
    let info = crate::inspect_probe(&payload)?;
    if let Some(range) = range {
        range.validate(info.duration_sec)?;
    }
    if let Some(selected) = selected.filter(|_| supports_rgb_seek(&payload, &info)) {
        let windows = plan_rgb_seek_windows(selected);
        if windows.iter().any(|window| window.seek_frame != 0) {
            if let Ok(origin) = original_rgb_first_pts(source, range, cancelled) {
                for window in windows {
                    if cancelled.load(Ordering::Acquire) {
                        return Err("cancelled".into());
                    }
                    let last = window.ranges.last().ok_or("Missing RGB seek ranges")?.1;
                    if window.seek_frame == 0 {
                        decode_original_rgb_stream(
                            source,
                            last + 1,
                            range,
                            Some(window.ranges),
                            cancelled,
                            &info,
                            None,
                            &mut consume,
                        )?;
                        continue;
                    }
                    let first = window.ranges[0].0;
                    let local = window
                        .ranges
                        .iter()
                        .map(|&(left, right)| (left - first, right - first))
                        .collect::<Vec<_>>();
                    let start = range.map_or(0.0, |item| item.start_sec);
                    let end = range.map_or(info.duration_sec + 0.1, |item| item.end_sec);
                    let window_range = AnalysisRange {
                        start_sec: start + window.seek_frame as f64 / 30.0,
                        end_sec: end.min(
                            start
                                + ((last + RGB_SEEK_PREROLL_FRAMES + 1) as f64 + origin as f64)
                                    / 30.0,
                        ),
                    };
                    let shift = origin + (first - window.seek_frame) as i64;
                    let mut delivered = false;
                    let result = decode_original_rgb_stream(
                        source,
                        last - first + 1,
                        Some(window_range),
                        Some(&local),
                        cancelled,
                        &info,
                        Some(shift),
                        |index, pixels, width, height| {
                            delivered = true;
                            consume(first + index, pixels, width, height)
                        },
                    );
                    if let Err(error) = result {
                        if delivered || cancelled.load(Ordering::Acquire) {
                            return Err(error);
                        }
                        eprintln!("RGB seek window {first}..{last} unavailable: {error}; using sequential decode");
                        decode_original_rgb_stream(
                            source,
                            last + 1,
                            range,
                            Some(window.ranges),
                            cancelled,
                            &info,
                            None,
                            &mut consume,
                        )?;
                    }
                }
                return Ok(());
            }
        }
    }
    decode_original_rgb_stream(
        source,
        expected_frames,
        range,
        selected,
        cancelled,
        &info,
        None,
        consume,
    )
}

#[expect(clippy::too_many_arguments)]
fn decode_original_rgb_stream(
    source: &Path,
    expected_frames: u64,
    range: Option<AnalysisRange>,
    selected: Option<&[(u64, u64)]>,
    cancelled: &AtomicBool,
    info: &crate::VideoInfo,
    pts_shift: Option<i64>,
    mut consume: impl FnMut(u64, &[u8], usize, usize) -> Result<(), String>,
) -> Result<(), String> {
    let (width, height) = (info.width as usize, info.height as usize);
    if width < 2 || height < 2 || width > 8192 || height > 8192 {
        return Err("Original RGB dimensions exceed the bounded decoder".into());
    }
    let frame_bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("Original RGB frame size overflow")?;
    if let Some(range) = range {
        range.validate(info.duration_sec)?;
    }
    let (mut ff, prefix) = auto_decode_input_in_range(source, cancelled, range)?;
    let expression = selected
        .filter(|ranges| ranges.len() <= 256)
        .map(balanced_selection_expression);
    let output_count = if expression.is_some() {
        selected_rgb_output_count(selected.unwrap_or_default())
    } else {
        expected_frames
    };
    configure_original_rgb_output(
        &mut ff,
        &prefix,
        output_count,
        expression.as_deref(),
        pts_shift,
    );
    ff.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = ManagedChild(
        ff.spawn()
            .map_err(|error| format!("Original RGB FFmpeg launch failed: {error}"))?,
    );
    let mut stdout = child.0.stdout.take().ok_or("Original RGB stdout missing")?;
    let stderr = child.0.stderr.take().ok_or("Original RGB stderr missing")?;
    let errors_reader = stderr_tail(stderr);
    let result = thread::scope(|scope| {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<io::Result<Vec<u8>>>(8);
        scope.spawn(move || {
            let mut chunk = vec![0_u8; 64 * 1024];
            loop {
                match stdout.read(&mut chunk) {
                    Ok(0) => {
                        let _ = sender.send(Ok(Vec::new()));
                        break;
                    }
                    Ok(count) => {
                        if sender.send(Ok(chunk[..count].to_vec())).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                        break;
                    }
                }
            }
        });
        let mut pipe = GuardedPipe {
            receiver,
            cancelled,
            idle_limit: DECODER_IDLE_LIMIT,
            pending: Vec::new(),
            offset: 0,
        };
        let outcome = (|| -> Result<(), String> {
            let mut frame = vec![0_u8; frame_bytes];
            for index in 0..expected_frames {
                if cancelled.load(Ordering::Acquire) {
                    return Err("cancelled".into());
                }
                let included = selected.is_none_or(|ranges| rgb_range_contains(ranges, index));
                if expression.is_some()
                    && !included
                    && !selected.is_some_and(|ranges| rgb_heartbeat(ranges, index))
                {
                    continue;
                }
                pipe.read_exact(&mut frame)
                    .map_err(|error| format!("Original RGB frame {index}: {error}"))?;
                if included {
                    consume(index, &frame, width, height)?;
                }
            }
            let mut extra = [0_u8; 1];
            if pipe.read(&mut extra).map_err(|error| error.to_string())? != 0 {
                return Err("Original RGB produced more frames than the GPU evidence".into());
            }
            if cancelled.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }
            let status = child.0.wait().map_err(|error| error.to_string())?;
            if !status.success() {
                return Err(format!("Original RGB FFmpeg exited with {status}"));
            }
            Ok(())
        })();
        if outcome.is_err() {
            let _ = child.0.kill();
            let _ = child.0.wait();
        }
        outcome
    });
    let errors = errors_reader
        .join()
        .unwrap_or_else(|_| "Original RGB FFmpeg stderr reader panicked".into());
    result.map_err(|error| {
        if errors.trim().is_empty() {
            error
        } else {
            format!(
                "{error}: {}",
                errors.trim().chars().take(500).collect::<String>()
            )
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, process::Command};

    #[test]
    fn bounded_source_decode_reads_only_saved_interval_and_keeps_local_indices() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let root = std::env::temp_dir().join(format!(
            "volley-trim-decode-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let source = root.join("input.mkv");
        let output = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=160x90:r=30:d=2",
                "-f",
                "lavfi",
                "-i",
                "color=c=white:s=160x90:r=30:d=2",
                "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map",
                "[v]",
                "-c:v",
                "mpeg4",
                "-y",
            ])
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{:?}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cancel = AtomicBool::new(false);
        let clip_one = AnalysisRange {
            start_sec: 0.5,
            end_sec: 1.5,
        };
        let clip_two = AnalysisRange {
            start_sec: 2.5,
            end_sec: 3.5,
        };
        let mut samples = Vec::new();
        for clip in [clip_one, clip_two] {
            let mut first = None;
            let report = decode_full_video_in_range(
                &source,
                ModelPixels::BallGray,
                9,
                Some(clip),
                &cancel,
                |window| {
                    assert_eq!(window.first_frame % 9, 0);
                    if first.is_none() {
                        first = Some(window.data[0]);
                    }
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(report.decoded_frames, 30);
            assert_eq!(report.windows, 3);
            samples.push(first.unwrap());
        }
        assert!(samples[0] < 30, "{:?}", samples);
        assert!(samples[1] > 200, "{:?}", samples);
        let mut shared_windows = 0;
        let shared = decode_full_video_shared_with_native_and_tail_in_range(
            &source,
            9,
            DecodePreference::Software,
            0,
            Some(clip_two),
            &cancel,
            false,
            |batch| {
                if shared_windows == 0 {
                    assert!(batch.gray[0] > 200);
                }
                assert_eq!(batch.first_frame, shared_windows * 9);
                shared_windows += 1;
                Ok(())
            },
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(shared_windows, 3);
        assert_eq!(shared.decode.decoded_frames, 30);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn selected_original_rgb_preserves_exact_pixels_and_global_frame_indices() {
        let root = std::env::temp_dir().join(format!(
            "volley-select-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let source = root.join("input.mp4");
        let output = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=25:duration=4",
                "-c:v",
                "libx264",
                "-y",
            ])
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cancel = AtomicBool::new(false);
        let range = Some(AnalysisRange {
            start_sec: 0.37,
            end_sec: 2.37,
        });
        let selected = [(3, 8), (21, 25), (44, 58)];
        let mut expected = Vec::new();
        decode_original_frames_in_range(&source, 60, range, &cancel, |index, pixels, _, _| {
            if selected
                .iter()
                .any(|&(left, right)| (left..=right).contains(&index))
            {
                expected.push((index, pixels.to_vec()));
            }
            Ok(())
        })
        .unwrap();
        let mut actual = Vec::new();
        decode_original_frames_selected_in_range(
            &source,
            60,
            range,
            &selected,
            &cancel,
            |index, pixels, _, _| {
                actual.push((index, pixels.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(actual.len(), 26);
        assert_eq!(actual, expected);
        assert!(decode_original_frames_selected_in_range(
            &source,
            60,
            range,
            &[(5, 8), (7, 9)],
            &cancel,
            |_, _, _, _| Ok(())
        )
        .is_err());
        assert!(decode_original_frames_selected_in_range(
            &source,
            60,
            range,
            &[(59, 60)],
            &cancel,
            |_, _, _, _| Ok(())
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn many_rally_intervals_decode_without_expression_depth_failure() {
        let root = std::env::temp_dir().join(format!("volley-select-many-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let source = root.join("input.mp4");
        let output = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=30:duration=18",
                "-c:v",
                "libx264",
                "-y",
            ])
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let range = Some(AnalysisRange {
            start_sec: 0.37,
            end_sec: 17.37,
        });
        let selected: Vec<(u64, u64)> = (0..119).map(|i| (i * 4, i * 4 + 1)).collect();
        let cancel = AtomicBool::new(false);
        let mut expected = Vec::new();
        decode_original_frames_in_range(&source, 510, range, &cancel, |index, pixels, _, _| {
            if selected
                .iter()
                .any(|&(left, right)| (left..=right).contains(&index))
            {
                expected.push((index, pixels.to_vec()));
            }
            Ok(())
        })
        .unwrap();
        let mut actual = Vec::new();
        let result = decode_original_frames_selected_in_range(
            &source,
            510,
            range,
            &selected,
            &cancel,
            |index, pixels, _, _| {
                actual.push((index, pixels.to_vec()));
                Ok(())
            },
        );
        fs::remove_dir_all(root).unwrap();
        result.unwrap();
        assert_eq!(actual.len(), 238);
        assert!(
            actual == expected,
            "Selected RGB pixels and source frame indices differ"
        );
    }

    #[test]
    fn sparse_original_rgb_filter_keeps_pipe_alive_between_rallies() {
        let root = std::env::temp_dir().join(format!("volley-sparse-rgb-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let source = root.join("input.mp4");
        let created = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=30:duration=45",
                "-c:v",
                "libx264",
                "-y",
            ])
            .arg(&source)
            .output()
            .unwrap();
        assert!(created.status.success());
        let selected = [(1, 2), (1199, 1200)];
        let expression = balanced_selection_expression(&selected);
        let mut command = Command::new("ffmpeg");
        command
            .args(["-hide_banner", "-loglevel", "error", "-i"])
            .arg(&source);
        configure_original_rgb_output(&mut command, "", 1350, Some(&expression), None);
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cancel = AtomicBool::new(false);
        let mut expected = Vec::new();
        decode_original_frames_in_range(&source, 1350, None, &cancel, |index, pixels, _, _| {
            if selected
                .iter()
                .any(|&(left, right)| (left..=right).contains(&index))
            {
                expected.push((index, pixels.to_vec()));
            }
            Ok(())
        })
        .unwrap();
        let mut actual = Vec::new();
        decode_original_frames_selected_in_range(
            &source,
            1350,
            None,
            &selected,
            &cancel,
            |index, pixels, _, _| {
                actual.push((index, pixels.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        fs::remove_dir_all(root).unwrap();
        assert_eq!(output.stdout.len() / (160 * 90 * 3), 8);
        assert_eq!(actual, expected);
    }

    #[test]
    fn original_rgb_sparse_seek_preserves_vfr_pixels_and_trimmed_indices() {
        let root = std::env::temp_dir().join(format!("volley-rgb-seek-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("input.mp4");
        let created = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=25:duration=78",
                "-vf",
                "select='not(eq(mod(n,17),0))'",
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-g",
                "75",
                "-y",
            ])
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            created.status.success(),
            "{}",
            String::from_utf8_lossy(&created.stderr)
        );
        let range = Some(AnalysisRange {
            start_sec: 0.37,
            end_sec: 76.37,
        });
        let selected = [(1, 15), (22, 27), (1199, 1212), (2240, 2250)];
        let payload = crate::ffprobe(&source).unwrap();
        assert!(supports_rgb_seek(
            &payload,
            &crate::inspect_probe(&payload).unwrap()
        ));
        assert!(original_rgb_first_pts(&source, range, &AtomicBool::new(false)).is_ok());
        let windows = plan_rgb_seek_windows(&selected);
        assert_eq!(windows.len(), 3);
        assert!(windows[1].seek_frame > 1000);
        assert!(windows[2].seek_frame > 2000);
        let cancel = AtomicBool::new(false);
        let mut expected = Vec::new();
        decode_original_frames_in_range(&source, 2280, range, &cancel, |index, pixels, _, _| {
            if rgb_range_contains(&selected, index) {
                expected.push((index, pixels.to_vec()));
            }
            Ok(())
        })
        .unwrap();
        let mut actual = Vec::new();
        decode_original_frames_selected_in_range(
            &source,
            2280,
            range,
            &selected,
            &cancel,
            |index, pixels, _, _| {
                actual.push((index, pixels.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        fs::remove_dir_all(root).unwrap();
        assert_eq!(actual.len(), 46);
        assert!(
            actual == expected,
            "Seek changed original RGB pixels or source indices"
        );
    }

    #[test]
    fn original_rgb_seek_into_timestamp_gap_falls_back_without_wrong_frames() {
        let root = std::env::temp_dir().join(format!("volley-rgb-seek-gap-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("input.mp4");
        let created = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=30:duration=45",
                "-vf",
                "select='not(between(n,900,1199))'",
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-g",
                "60",
                "-y",
            ])
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            created.status.success(),
            "{}",
            String::from_utf8_lossy(&created.stderr)
        );
        let selected = [(1050, 1053), (1220, 1223)];
        let cancel = AtomicBool::new(false);
        let mut expected = Vec::new();
        decode_original_frames_in_range(&source, 1350, None, &cancel, |index, pixels, _, _| {
            if rgb_range_contains(&selected, index) {
                expected.push((index, pixels.to_vec()));
            }
            Ok(())
        })
        .unwrap();
        let mut actual = Vec::new();
        decode_original_frames_selected_in_range(
            &source,
            1350,
            None,
            &selected,
            &cancel,
            |index, pixels, _, _| {
                actual.push((index, pixels.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        fs::remove_dir_all(root).unwrap();
        assert_eq!(actual.len(), 8);
        assert!(
            actual == expected,
            "Timestamp gap changed the frame mapping"
        );
    }

    #[test]
    fn original_rgb_hevc_fractional_rate_seek_keeps_last_selected_frames() {
        let root =
            std::env::temp_dir().join(format!("volley-rgb-seek-hevc-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("input.mp4");
        let created = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=30000/1001:duration=66",
                "-c:v",
                "libx265",
                "-preset",
                "ultrafast",
                "-x265-params",
                "pools=1:frame-threads=1:log-level=error:keyint=90",
                "-y",
            ])
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            created.status.success(),
            "{}",
            String::from_utf8_lossy(&created.stderr)
        );
        let selected = [(1001, 1003), (1950, 1955)];
        let range = Some(AnalysisRange {
            start_sec: 0.19,
            end_sec: 65.39,
        });
        let cancel = AtomicBool::new(false);
        let payload = crate::ffprobe(&source).unwrap();
        let info = crate::inspect_probe(&payload).unwrap();
        assert!(supports_rgb_seek(&payload, &info));
        let _origin = original_rgb_first_pts(&source, range, &cancel).unwrap();
        let mut expected = Vec::new();
        decode_original_rgb_stream(
            &source,
            1956,
            range,
            Some(&selected),
            &cancel,
            &info,
            None,
            |index, pixels, _, _| {
                expected.push((index, pixels.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        let mut actual = Vec::new();
        decode_original_frames_selected_in_range(
            &source,
            1956,
            range,
            &selected,
            &cancel,
            |index, pixels, _, _| {
                actual.push((index, pixels.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        fs::remove_dir_all(root).unwrap();
        assert_eq!(actual.len(), 9);
        assert!(
            actual == expected,
            "HEVC seek changed the source pixels or frame indices"
        );
    }

    #[test]
    fn original_rgb_seek_accounts_for_video_start_after_audio() {
        let root =
            std::env::temp_dir().join(format!("volley-rgb-seek-offset-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("input.mp4");
        let created = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=30:duration=66",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=8000:cl=mono",
                "-filter_complex",
                "[0:v]setpts=PTS+3/TB[v]",
                "-map",
                "[v]",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "aac",
                "-t",
                "69",
                "-fps_mode",
                "passthrough",
                "-y",
            ])
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            created.status.success(),
            "{}",
            String::from_utf8_lossy(&created.stderr)
        );
        let selected = [(1000, 1008), (1950, 1958)];
        let cancel = AtomicBool::new(false);
        assert_eq!(original_rgb_first_pts(&source, None, &cancel).unwrap(), 90);
        let info = crate::probe(&source).unwrap();
        let mut expected = Vec::new();
        decode_original_rgb_stream(
            &source,
            1959,
            None,
            Some(&selected),
            &cancel,
            &info,
            None,
            |index, pixels, _, _| {
                expected.push((index, pixels.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        let mut actual = Vec::new();
        decode_original_frames_selected_in_range(
            &source,
            1959,
            None,
            &selected,
            &cancel,
            |index, pixels, _, _| {
                actual.push((index, pixels.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        fs::remove_dir_all(root).unwrap();
        assert_eq!(actual.len(), 18);
        assert!(
            actual == expected,
            "Source stream start offset changed seek frames"
        );
    }

    #[test]
    fn invalid_ranges_fail_closed_before_decode() {
        let duration = 3600.0;
        for invalid in [
            AnalysisRange {
                start_sec: -1.0,
                end_sec: 20.0,
            },
            AnalysisRange {
                start_sec: 20.0,
                end_sec: 20.0,
            },
            AnalysisRange {
                start_sec: 20.0,
                end_sec: 3602.0,
            },
            AnalysisRange {
                start_sec: f64::NAN,
                end_sec: 22.0,
            },
        ] {
            assert!(invalid.validate(duration).is_err());
        }
        assert!(AnalysisRange {
            start_sec: 601.0,
            end_sec: 780.0
        }
        .validate(duration)
        .is_ok());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn original_rgb_4k_output_completes_with_two_gib_address_space() {
        let dir = std::env::temp_dir().join(format!("volley-rgb-memory-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("input.mp4");
        let created = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=3840x2160:rate=30:duration=2",
                "-c:v",
                "mpeg4",
                "-threads",
                "2",
                "-y",
            ])
            .arg(&source)
            .status()
            .unwrap();
        assert!(created.success());
        let mut command = Command::new("prlimit");
        command
            .args([
                "--as=2147483648",
                "--",
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-threads",
                "1",
                "-i",
            ])
            .arg(&source);
        configure_original_rgb_output(&mut command, "", 60, None, None);
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let errors = stderr_tail(child.stderr.take().unwrap());
        let bytes = io::copy(&mut child.stdout.take().unwrap(), &mut io::sink()).unwrap();
        let status = child.wait().unwrap();
        let errors = errors.join().unwrap();
        assert!(status.success(), "{status}: {errors}");
        assert_eq!(bytes, 60 * 3840 * 2160 * 3);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn original_rgb_stream_preserves_native_resolution_and_frame_order() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("volley-original-rgb-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("input.mp4");
        let created = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=192x108:rate=60:duration=0.6",
                "-an",
                "-c:v",
                "mpeg4",
                "-y",
            ])
            .arg(&source)
            .status()
            .unwrap();
        assert!(created.success());
        let cancelled = AtomicBool::new(false);
        let mut indices = Vec::new();
        decode_original_frames(&source, 18, &cancelled, |index, frame, w, h| {
            assert_eq!((w, h), (192, 108));
            assert_eq!(frame.len(), 192 * 108 * 3);
            indices.push(index);
            Ok(())
        })
        .unwrap();
        assert_eq!(indices, (0..18).collect::<Vec<_>>());
        assert!(decode_original_frames(&source, 19, &cancelled, |_, _, _, _| Ok(())).is_err());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn full_video_stream_covers_complete_windows_only_and_can_cancel() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let root =
            std::env::temp_dir().join(format!("volley-full-raw-windows-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("synthetic.mp4");
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=60:duration=0.6",
                "-an",
                "-c:v",
                "mpeg4",
                "-y",
            ])
            .arg(&source)
            .status()
            .unwrap();
        assert!(status.success());
        let cancelled = AtomicBool::new(false);
        let mut frame_ids = Vec::new();
        let summary = decode_full_video(&source, ModelPixels::BallGray, 4, &cancelled, |win| {
            assert_eq!(win.data.len(), 9 * 512 * 288);
            frame_ids.push(win.first_frame);
            Ok(())
        })
        .unwrap();
        assert_eq!(frame_ids, vec![0, 4, 8]);
        assert_eq!(summary.windows, 3);
        assert_eq!(summary.decoded_frames, 18);
        // A separate invocation fails instead of inventing a complete
        // full-video result from an interrupted callback.
        let aborted = AtomicBool::new(false);
        let result = decode_full_video(&source, ModelPixels::BallGray, 9, &aborted, |_| {
            aborted.store(true, Ordering::Release);
            Ok(())
        });
        assert!(result.unwrap_err().contains("cancelled"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn bounded_window_schedule_checks_range() {
        assert_eq!(checked_total_frames(0, 3, 9).unwrap(), 27);
        assert_eq!(checked_total_frames(100, 3, 3).unwrap(), 15);
        assert!(checked_total_frames(0, 121, 9).is_err());
        assert!(checked_total_frames(MAX_FRAMES, 1, 1).is_err());
        assert!(checked_total_frames(0, 2, 0).is_err());
    }
    #[test]
    fn guarded_pipe_times_out_when_the_decoder_is_silent() {
        let (_sender, receiver) = std::sync::mpsc::sync_channel(8);
        let cancelled = AtomicBool::new(false);
        let mut pipe = GuardedPipe {
            receiver,
            cancelled: &cancelled,
            idle_limit: Duration::from_millis(50),
            pending: Vec::new(),
            offset: 0,
        };
        let mut buffer = [0_u8; 16];
        let error = pipe.read(&mut buffer).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    }
    #[test]
    fn guarded_pipe_interrupts_a_blocked_read_on_cancel() {
        let (_sender, receiver) = std::sync::mpsc::sync_channel(8);
        let cancelled = AtomicBool::new(true);
        let mut pipe = GuardedPipe {
            receiver,
            cancelled: &cancelled,
            idle_limit: Duration::from_secs(120),
            pending: Vec::new(),
            offset: 0,
        };
        let mut buffer = [0_u8; 16];
        let error = pipe.read(&mut buffer).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::ConnectionAborted);
    }
    #[test]
    fn guarded_pipe_preserves_bytes_across_chunk_and_buffer_boundaries() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(8);
        let cancelled = AtomicBool::new(false);
        let mut pipe = GuardedPipe {
            receiver,
            cancelled: &cancelled,
            idle_limit: Duration::from_secs(120),
            pending: Vec::new(),
            offset: 0,
        };
        sender.send(Ok(vec![1, 2, 3, 4, 5])).unwrap();
        sender.send(Ok(Vec::new())).unwrap();
        let mut out = Vec::new();
        let mut buffer = [0_u8; 2];
        loop {
            let count = pipe.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            out.extend_from_slice(&buffer[..count]);
        }
        assert_eq!(out, vec![1, 2, 3, 4, 5]);
    }
    #[test]
    fn ffmpeg_streams_one_process_across_ordered_windows() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let root =
            std::env::temp_dir().join(format!("volley-ffmpeg-windows-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("synthetic.mp4");
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30:duration=0.6",
                "-an",
                "-c:v",
                "mpeg4",
                "-y",
            ])
            .arg(&source)
            .status()
            .unwrap();
        assert!(status.success());
        let cancelled = AtomicBool::new(false);
        let mut starts = Vec::new();
        let summary = decode_windows(
            &source,
            ModelPixels::BallGray,
            0,
            3,
            3,
            &cancelled,
            |window| {
                starts.push(window.first_frame);
                assert_eq!(window.data.len(), 512 * 288 * 9);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(starts, vec![0, 3, 6]);
        assert_eq!(summary.decoded_frames, 15);
        let cancelled = AtomicBool::new(true);
        assert!(
            decode_windows(&source, ModelPixels::BallGray, 0, 1, 1, &cancelled, |_| Ok(
                ()
            ))
            .is_err()
        );
        let _ = fs::remove_dir_all(root);
    }
    #[test]
    fn cancellation_during_partial_raw_frame_unblocks_the_window_reader() {
        use std::sync::{mpsc, Arc};
        let cancelled = Arc::new(AtomicBool::new(false));
        let reader_cancelled = Arc::clone(&cancelled);
        let (sender, receiver) = mpsc::sync_channel(2);
        let (done_sender, done_receiver) = mpsc::channel();
        let reader = thread::spawn(move || {
            let pipe = GuardedPipe {
                receiver,
                cancelled: &reader_cancelled,
                idle_limit: Duration::from_secs(120),
                pending: Vec::new(),
                offset: 0,
            };
            let mut windows = FrameWindows::new(pipe, 4, 1).unwrap();
            let result = windows.next_window().unwrap_err().kind();
            done_sender.send(result).unwrap();
        });
        // Half a frame has arrived; the pipe waits for its remainder.
        sender.send(Ok(vec![1_u8, 2_u8])).unwrap();
        thread::sleep(Duration::from_millis(30));
        cancelled.store(true, Ordering::Release);
        assert_eq!(
            done_receiver.recv_timeout(Duration::from_secs(2)).unwrap(),
            io::ErrorKind::ConnectionAborted
        );
        reader.join().unwrap();
    }
}
