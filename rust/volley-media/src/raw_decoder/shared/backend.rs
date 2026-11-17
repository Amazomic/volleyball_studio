//! Conservative hardware selection for the common primary stream.
//! Probe is bounded and cancel-aware; failing CUDA never triggers an implicit
//! mid-stream switch. The software path retains the original filter contract.
use super::super::ManagedChild;
use serde::Serialize;
use serde_json::Value;
use std::{
    io::Read,
    path::Path,
    process::{Command, ExitStatus, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

const PROBE_TIMEOUT: Duration = Duration::from_secs(12);
const PROBE_POLL: Duration = Duration::from_millis(50);
const MAX_PROBE_OUTPUT: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeBackend {
    Software,
    Nvdec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePreference {
    /// Prefer verified CUDA/NVDEC support, otherwise select software before
    /// the full decoding attempt begins.
    Auto,
    /// Never attempt CUDA.
    Software,
    /// Require NVDEC; reject unsupported sources without falling back.
    Nvdec,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeSelection {
    pub backend: DecodeBackend,
    pub fallback_reason: Option<String>,
    #[serde(skip)]
    pub(super) rotation_filter: &'static str,
}

fn bounded_output(
    mut command: Command,
    cancelled: &AtomicBool,
    label: &str,
) -> Result<(ExitStatus, String, String), String> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = ManagedChild(
        command
            .spawn()
            .map_err(|error| format!("Launch {label}: {error}"))?,
    );
    let started = Instant::now();
    let status = loop {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        match child.0.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() >= PROBE_TIMEOUT => {
                return Err(format!("{label} exceeded the probe time limit"));
            }
            Ok(None) => thread::sleep(PROBE_POLL),
            Err(error) => return Err(format!("Wait for {label}: {error}")),
        }
    };
    let mut out = Vec::new();
    let mut err = Vec::new();
    child
        .0
        .stdout
        .take()
        .ok_or("Probe stdout missing")?
        .take((MAX_PROBE_OUTPUT + 1) as u64)
        .read_to_end(&mut out)
        .map_err(|error| format!("{label} stdout: {error}"))?;
    child
        .0
        .stderr
        .take()
        .ok_or("Probe stderr missing")?
        .take((MAX_PROBE_OUTPUT + 1) as u64)
        .read_to_end(&mut err)
        .map_err(|error| format!("{label} stderr: {error}"))?;
    if out.len() > MAX_PROBE_OUTPUT || err.len() > MAX_PROBE_OUTPUT {
        return Err(format!("{label} output exceeded its limit"));
    }
    Ok((
        status,
        String::from_utf8_lossy(&out).into_owned(),
        String::from_utf8_lossy(&err).into_owned(),
    ))
}

fn inspect_video(source: &Path, cancelled: &AtomicBool) -> Result<&'static str, String> {
    let mut probe = Command::new("ffprobe");
    probe.args([
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,profile,pix_fmt,width,height,color_transfer:stream_tags=rotate:stream_side_data=rotation",
        "-of", "json",
    ]);
    probe.arg(source);
    let (status, output, errors) = bounded_output(probe, cancelled, "FFprobe")?;
    if !status.success() {
        return Err(format!("FFprobe failed: {}", errors.trim()));
    }
    let document: Value = serde_json::from_str(&output)
        .map_err(|error| format!("Invalid FFprobe metadata: {error}"))?;
    let stream = document["streams"]
        .as_array()
        .and_then(|streams| streams.first())
        .ok_or("FFprobe did not return a video stream")?;
    let codec = stream["codec_name"].as_str().unwrap_or("");
    let format = stream["pix_fmt"].as_str().unwrap_or("");
    if !matches!((codec, format), ("h264" | "hevc", "yuv420p")) {
        return Err(format!(
            "NVDEC not certified for codec={codec} pixel_format={format}; using software"
        ));
    }
    if stream["height"].as_u64().unwrap_or(0) < 144 || stream["width"].as_u64().unwrap_or(0) < 144 {
        return Err("NVDEC source geometry is below the certified dimensions".into());
    }
    let transfer = stream["color_transfer"].as_str().unwrap_or("");
    if matches!(transfer, "smpte2084" | "arib-std-b67") {
        return Err("NVDEC HDR transfer is not yet certified for parity".into());
    }
    rotation_filter(stream)
}

fn rotation_filter(stream: &Value) -> Result<&'static str, String> {
    let rotation = stream["side_data_list"]
        .as_array()
        .and_then(|sides| sides.iter().find_map(|side| side.get("rotation")))
        .cloned()
        .or_else(|| stream["tags"].get("rotate").cloned());
    let rotation = match rotation {
        None => 0.0,
        Some(Value::String(value)) => value
            .parse::<f64>()
            .map_err(|_| "Invalid video rotation metadata")?,
        Some(value) => value.as_f64().ok_or("Invalid video rotation metadata")?,
    };
    match rotation.rem_euclid(360.0) {
        0.0 => Ok(""),
        90.0 => Ok("transpose=cclock,"),
        180.0 => Ok("hflip,vflip,"),
        270.0 => Ok("transpose=clock,"),
        _ => Err("NVDEC non-orthogonal rotation is not certified for parity".into()),
    }
}

/// Preflight requests CUDA output explicitly, so FFmpeg cannot silently use a
/// software decoder and make the backend label untrustworthy.
fn preflight_nvdec(
    source: &Path,
    rotation_filter: &str,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let mut command = Command::new("ffmpeg");
    command.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-noautorotate",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
    ]);
    command.arg(source).args([
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        &format!("hwdownload,format=nv12,{rotation_filter}null"),
        "-frames:v",
        "2",
        "-f",
        "null",
        "-",
    ]);
    let (status, _, stderr) = bounded_output(command, cancelled, "NVDEC preflight")?;
    if !status.success() {
        return Err(format!(
            "NVDEC failed to decode this source: {}",
            stderr.trim().chars().take(500).collect::<String>()
        ));
    }
    Ok(())
}

pub fn select_decode_backend(
    source: &Path,
    preference: DecodePreference,
    cancelled: &AtomicBool,
) -> Result<DecodeSelection, String> {
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    if !source.is_file() {
        return Err("Original source video is not a file".into());
    }
    if preference == DecodePreference::Software {
        return Ok(DecodeSelection {
            backend: DecodeBackend::Software,
            fallback_reason: None,
            rotation_filter: "",
        });
    }
    match inspect_video(source, cancelled).and_then(|rotation| {
        preflight_nvdec(source, rotation, cancelled)?;
        Ok(rotation)
    }) {
        Ok(rotation_filter) => Ok(DecodeSelection {
            backend: DecodeBackend::Nvdec,
            fallback_reason: None,
            rotation_filter,
        }),
        Err(error) if error == "cancelled" => Err(error),
        Err(error) if preference == DecodePreference::Auto => Ok(DecodeSelection {
            backend: DecodeBackend::Software,
            fallback_reason: Some(error),
            rotation_filter: "",
        }),
        Err(error) => Err(error),
    }
}
