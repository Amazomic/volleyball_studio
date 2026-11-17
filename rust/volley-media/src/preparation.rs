//! Shared content-addressed prepared media, compatible with Python schema 2.
//! Pure planning logic; Rust job orchestration owns FFmpeg execution/cancellation.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{analysis_ready_probe, ffprobe, inspect_probe, VideoInfo};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreparedIdentity {
    pub source: String,
    pub size: u64,
    pub mtime_ns: u128,
    pub width: u32,
    pub fps: u32,
    pub schema: u8,
}

#[derive(Clone, Debug)]
pub struct PreparationPlan {
    pub source: PathBuf,
    pub output: PathBuf,
    pub manifest: PathBuf,
    pub identity: PreparedIdentity,
    pub width: u32,
    pub duration_sec: f64,
    pub already_analysis_ready: bool,
    pub original_preview_safe: bool,
    pub original_source: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparationReuse {
    Original,
    Cached,
}

// Default Python JSON ensure_ascii=True is part of schema 2 hashing.
// serde_json leaves Unicode unescaped, so convert it to ASCII here.
fn python_ascii_string(input: &str) -> String {
    let encoded = serde_json::to_string(input).expect("String is always valid JSON");
    let mut output = String::new();
    for ch in encoded.chars() {
        let code = ch as u32;
        if code > 0xffff {
            let scalar = code - 0x10000;
            let high = 0xd800 + (scalar >> 10);
            let low = 0xdc00 + (scalar & 0x3ff);
            output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
        } else if code >= 128 {
            output.push_str(&format!("\\u{code:04x}"));
        } else {
            output.push(ch);
        }
    }
    output
}

pub fn python_fingerprint_json(item: &PreparedIdentity) -> String {
    // Sorted Python dict keys: fps, mtime_ns, schema, size, source, width.
    format!(
        r#"{{"fps": {}, "mtime_ns": {}, "schema": {}, "size": {}, "source": {}, "width": {}}}"#,
        item.fps,
        item.mtime_ns,
        item.schema,
        item.size,
        python_ascii_string(&item.source),
        item.width
    )
}

pub fn cache_id(item: &PreparedIdentity) -> String {
    let hash = Sha256::digest(python_fingerprint_json(item).as_bytes());
    format!("{hash:x}")[..28].to_owned()
}

/// Streaming SHA-256: a one-megabyte buffer bounds additional RAM even for
/// multi-hour matches. Refuse links and nonregular files before reading.
pub fn prepared_file_sha256(path: &Path) -> io::Result<(u64, String)> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unsafe prepared video",
        ));
    }
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut total = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        total = total.checked_add(read as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Prepared MP4 length overflow")
        })?;
    }
    if total != metadata.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Prepared MP4 changed while hashing",
        ));
    }
    Ok((total, format!("{:x}", digest.finalize())))
}

type SourceProof = (u64, SystemTime, (u64, u64, i64, i64), Value);
static SOURCES: OnceLock<Mutex<HashMap<PathBuf, SourceProof>>> = OnceLock::new();

pub fn source_provenance(path: &Path) -> Result<Value, String> {
    let meta = fs::symlink_metadata(path).map_err(|e| format!("Original metadata: {e}"))?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err("Unsafe original video".into());
    }
    let modified = meta
        .modified()
        .map_err(|e| format!("Original timestamp: {e}"))?;
    let change = file_change_stamp(path, &meta);
    let cache = SOURCES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(stamp) = change {
        let entries = cache.lock().map_err(|_| "Original proof lock poisoned")?;
        if let Some((size, time, previous, proof)) = entries.get(path) {
            if *size == meta.len() && *time == modified && *previous == stamp {
                return Ok(proof.clone());
            }
        }
    }
    let (size, hash) = prepared_file_sha256(path).map_err(|e| format!("Original SHA-256: {e}"))?;
    let probe = ffprobe(path)?;
    let info = inspect_probe(&probe)?;
    let after = fs::symlink_metadata(path).map_err(|e| format!("Original metadata: {e}"))?;
    if after.len() != size
        || after.modified().ok() != Some(modified)
        || file_change_stamp(path, &after) != change
    {
        return Err("Original video changed during verification".into());
    }
    let video = probe["streams"]
        .as_array()
        .and_then(|rows| rows.iter().find(|row| row["codec_type"] == "video"))
        .ok_or("Missing original video stream")?;
    let proof = serde_json::json!({
        "schema":"volley-original-source/1","sha256":hash,"size_bytes":size,
        "video":info,"native_avg_frame_rate":video["avg_frame_rate"],
        "native_time_base":video["time_base"],"native_start_time":video["start_time"],
        "timeline":{"analysis_fps":30,"origin":"original_relative_seconds",
            "frame_time":"frame_index / 30","source_offset_sec":0.0}
    });
    if let Some(stamp) = change {
        let mut entries = cache.lock().map_err(|_| "Original proof lock poisoned")?;
        if entries.len() >= 32 {
            entries.clear();
        }
        entries.insert(path.to_path_buf(), (size, modified, stamp, proof.clone()));
    }
    Ok(proof)
}

pub fn verify_source_provenance(path: &Path, expected: &Value) -> Result<(), String> {
    if expected.is_null() {
        return Ok(());
    }
    if expected["schema"] != "volley-original-source/1" || source_provenance(path)? != *expected {
        return Err("Original video differs from the analysis source".into());
    }
    Ok(())
}

// An analysis records the full SHA above. Save that verified proof OUTSIDE the
// original-video directory so a fresh Studio process can reuse it. A receipt
// without a reliable kernel change stamp is never trusted.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalSourceReceipt {
    schema: String,
    path: String,
    size: u64,
    modified_ns: u128,
    change: (u64, u64, i64, i64),
    proof: Value,
}

fn original_receipt_file(path: &Path, root: &Path) -> Option<PathBuf> {
    let name = path.to_str()?;
    let key = format!("{:x}", Sha256::digest(name.as_bytes()));
    Some(
        root.join("verified-original-sources-v1")
            .join(format!("{key}.json")),
    )
}

fn receipt_matches(
    receipt: &OriginalSourceReceipt,
    path: &Path,
    meta: &fs::Metadata,
    change: (u64, u64, i64, i64),
) -> bool {
    let modified_ns = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|time| time.as_nanos());
    let digest = receipt.proof["sha256"].as_str().unwrap_or("");
    receipt.schema == "volley-verified-original-source/1"
        && path.to_str() == Some(receipt.path.as_str())
        && receipt.size == meta.len()
        && Some(receipt.modified_ns) == modified_ns
        && receipt.change == change
        && receipt.proof["schema"] == "volley-original-source/1"
        && receipt.proof["size_bytes"].as_u64() == Some(meta.len())
        && digest.len() == 64
        && digest
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// SHA-verified content receipt, persisted in the isolated data root. A cached
/// proof is rejected if inode/USN/ctime, length, or mtime changed; without a
/// reliable OS change stamp we fall back to the complete SHA on every access.
pub fn source_provenance_cached(path: &Path, cache_root: &Path) -> Result<Value, String> {
    type Slot = std::sync::Weak<Mutex<()>>;
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Slot>>> = OnceLock::new();
    let lock = {
        let mut slots = LOCKS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|_| "Original source locks poisoned")?;
        slots.retain(|_, slot| slot.strong_count() > 0);
        if let Some(existing) = slots.get(path).and_then(Slot::upgrade) {
            existing
        } else {
            let new = Arc::new(Mutex::new(()));
            slots.insert(path.to_owned(), Arc::downgrade(&new));
            new
        }
    };
    let _guard = lock.lock().map_err(|_| "Original source lock poisoned")?;
    let meta = fs::symlink_metadata(path).map_err(|e| format!("Original metadata: {e}"))?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err("Unsafe original video".into());
    }
    let change = file_change_stamp(path, &meta);
    let receipt_file = original_receipt_file(path, cache_root);
    if let (Some(stamp), Some(file)) = (change, receipt_file.as_ref()) {
        let safe = file
            .parent()
            .and_then(|parent| fs::symlink_metadata(parent).ok())
            .is_some_and(|m| m.is_dir() && !m.file_type().is_symlink());
        let file_meta = fs::symlink_metadata(file).ok();
        if safe
            && file_meta
                .is_some_and(|m| m.is_file() && !m.file_type().is_symlink() && m.len() <= 16 * 1024)
        {
            if let Ok(bytes) = fs::read(file) {
                if let Ok(receipt) = serde_json::from_slice::<OriginalSourceReceipt>(&bytes) {
                    if receipt_matches(&receipt, path, &meta, stamp)
                        && fs::symlink_metadata(path).ok().is_some_and(|after| {
                            after.is_file()
                                && !after.file_type().is_symlink()
                                && after.len() == meta.len()
                                && after.modified().ok() == meta.modified().ok()
                                && file_change_stamp(path, &after) == Some(stamp)
                        })
                    {
                        let cache = SOURCES.get_or_init(|| Mutex::new(HashMap::new()));
                        let mut entries =
                            cache.lock().map_err(|_| "Original proof lock poisoned")?;
                        if entries.len() >= 32 {
                            entries.clear();
                        }
                        entries.insert(
                            path.to_owned(),
                            (
                                meta.len(),
                                meta.modified().map_err(|e| e.to_string())?,
                                stamp,
                                receipt.proof.clone(),
                            ),
                        );
                        return Ok(receipt.proof);
                    }
                }
            }
        }
    }
    // First access for an existing analysis hashes once, then publishes a
    // receipt. A changed source cannot inherit an earlier receipt.
    let proof = source_provenance(path)?;
    if let (Some(stamp), Some(file)) = (change, receipt_file) {
        if let Ok(after) = fs::symlink_metadata(path) {
            if after.is_file()
                && !after.file_type().is_symlink()
                && after.len() == meta.len()
                && after.modified().ok() == meta.modified().ok()
                && file_change_stamp(path, &after) == Some(stamp)
            {
                let parent = file.parent().ok_or("Invalid receipt path")?;
                if fs::create_dir_all(parent).is_ok()
                    && fs::symlink_metadata(parent)
                        .ok()
                        .is_some_and(|m| m.is_dir() && !m.file_type().is_symlink())
                {
                    let receipt = OriginalSourceReceipt {
                        schema: "volley-verified-original-source/1".to_owned(),
                        path: path.to_string_lossy().into_owned(),
                        size: meta.len(),
                        modified_ns: meta
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                            .map(|t| t.as_nanos())
                            .unwrap_or(0),
                        change: stamp,
                        proof: proof.clone(),
                    };
                    if let Ok(bytes) = serde_json::to_vec(&receipt) {
                        if bytes.len() <= 16 * 1024 {
                            let suffix = format!(
                                "{}.{}.tmp",
                                std::process::id(),
                                SOURCE_RECEIPT_SERIAL
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                            );
                            let temporary = file.with_extension(suffix);
                            if fs::OpenOptions::new()
                                .write(true)
                                .create_new(true)
                                .open(&temporary)
                                .and_then(|mut output| {
                                    use std::io::Write;
                                    output.write_all(&bytes)?;
                                    output.sync_all()
                                })
                                .is_ok()
                            {
                                let _ = fs::rename(&temporary, &file);
                            }
                            let _ = fs::remove_file(temporary);
                        }
                    }
                }
            }
        }
    }
    Ok(proof)
}
static SOURCE_RECEIPT_SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// A finished prepared MP4 can be gigabytes. Verify its receipt once per
// process/file stamp, not once per API request. The receipt is still read from
// the manifest every time, and changes in size, mtime, the expected SHA or the
// platform change stamp (Unix ctime+inode, Windows USN+file index) invalidate
// this in-memory proof. Without a reliable change stamp the shortcut is never
// used and the file is re-hashed. A per-path lock deduplicates simultaneous
// requests for the same file without blocking unrelated prepared media.
#[derive(Clone)]
struct VerifiedPrepared {
    size: u64,
    modified: SystemTime,
    expected_sha: String,
    file_change: Option<(u64, u64, i64, i64)>,
}
/// Reliable per-file change identity for in-memory caches of immutable media
/// evidence: Unix ctime+inode, Windows volume serial+file index+USN. `None`
/// means no trustworthy stamp exists and callers must re-verify content on
/// every access instead of trusting a cached result.
#[cfg(unix)]
pub fn file_change_stamp(_path: &Path, metadata: &fs::Metadata) -> Option<(u64, u64, i64, i64)> {
    use std::os::unix::fs::MetadataExt;
    Some((
        metadata.dev(),
        metadata.ino(),
        metadata.ctime(),
        metadata.ctime_nsec(),
    ))
}
// Windows mtime alone is unsafe: NTFS timestamps advance with the system clock
// tick (~1-15 ms), so two same-size writes can share an identical mtime. The
// USN change number increments on every file modification regardless of the
// clock, and the volume serial + file index detect file replacement. If the
// USN cannot be read (non-NTFS volume), return None so the in-memory shortcut
// is never trusted and the content is re-hashed on every check.
#[cfg(windows)]
pub fn file_change_stamp(path: &Path, _metadata: &fs::Metadata) -> Option<(u64, u64, i64, i64)> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION, FILE_READ_ATTRIBUTES,
        FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
    };
    use windows_sys::Win32::System::Ioctl::FSCTL_READ_FILE_USN_DATA;
    use windows_sys::Win32::System::IO::DeviceIoControl;

    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    // SAFETY: `wide` is a NUL-terminated path; the handle is closed on every
    // exit path and the output buffers outlive the calls.
    unsafe {
        let handle = CreateFileW(
            wide.as_ptr(),
            FILE_READ_ATTRIBUTES,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            std::ptr::null(),
            OPEN_EXISTING,
            0,
            std::ptr::null_mut(),
        );
        if handle == INVALID_HANDLE_VALUE {
            return None;
        }
        let mut info: BY_HANDLE_FILE_INFORMATION = std::mem::zeroed();
        let info_ok = GetFileInformationByHandle(handle, &mut info);
        let mut buffer = [0u8; 1024];
        let mut returned = 0u32;
        let ok = DeviceIoControl(
            handle,
            FSCTL_READ_FILE_USN_DATA,
            std::ptr::null(),
            0,
            buffer.as_mut_ptr().cast(),
            buffer.len() as u32,
            &mut returned,
            std::ptr::null_mut(),
        );
        CloseHandle(handle);
        if info_ok == 0 || ok == 0 || returned < 64 {
            return None;
        }
        let major = u16::from_le_bytes([buffer[4], buffer[5]]);
        let usn_offset = match major {
            2 => 24,
            3 | 4 => 40,
            _ => return None,
        };
        if (returned as usize) < usn_offset + 8 {
            return None;
        }
        let usn = i64::from_le_bytes(buffer[usn_offset..usn_offset + 8].try_into().ok()?);
        let index = ((info.nFileIndexHigh as u64) << 32) | info.nFileIndexLow as u64;
        Some((info.dwVolumeSerialNumber as u64, index, usn, 0))
    }
}
#[cfg(not(any(unix, windows)))]
pub fn file_change_stamp(_path: &Path, _metadata: &fs::Metadata) -> Option<(u64, u64, i64, i64)> {
    None
}
type VerifiedSlot = Arc<Mutex<Option<VerifiedPrepared>>>;
static VERIFIED_PREPARED: OnceLock<Mutex<HashMap<PathBuf, VerifiedSlot>>> = OnceLock::new();

fn prepared_receipt_matches(path: &Path, size: u64, expected_sha: &str) -> bool {
    let Ok(before) = fs::symlink_metadata(path) else {
        return false;
    };
    if before.file_type().is_symlink() || !before.is_file() || before.len() != size {
        return false;
    }
    let Ok(modified) = before.modified() else {
        return false;
    };
    let file_change = file_change_stamp(path, &before);
    let Ok(mut slots) = VERIFIED_PREPARED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    else {
        return false;
    };
    let slot = if let Some(existing) = slots.get(path) {
        existing.clone()
    } else {
        // Bound metadata even for programs that process many videos.
        if slots.len() >= 24 {
            if let Some(oldest) = slots.keys().next().cloned() {
                slots.remove(&oldest);
            }
        }
        let created = Arc::new(Mutex::new(None));
        slots.insert(path.to_owned(), created.clone());
        created
    };
    drop(slots);
    let Ok(mut verified) = slot.lock() else {
        return false;
    };
    if verified.as_ref().is_some_and(|v| {
        v.size == size
            && v.modified == modified
            && v.expected_sha == expected_sha
            && v.file_change.is_some()
            && v.file_change == file_change
    }) {
        return true;
    }
    *verified = None;
    // On first access (or a changed file), prove content against the original
    // preparation SHA. Never cache failures or a file changed mid-hash.
    let Ok((count, digest)) = prepared_file_sha256(path) else {
        return false;
    };
    let Ok(after) = fs::symlink_metadata(path) else {
        return false;
    };
    if count != size
        || digest != expected_sha
        || after.file_type().is_symlink()
        || !after.is_file()
        || after.len() != size
        || after.modified().ok() != Some(modified)
        || file_change_stamp(path, &after) != file_change
    {
        return false;
    }
    *verified = Some(VerifiedPrepared {
        size,
        modified,
        expected_sha: expected_sha.to_owned(),
        file_change,
    });
    true
}

impl PreparationPlan {
    pub fn read_original(mut self) -> Self {
        self.already_analysis_ready = true;
        self
    }

    pub fn bind_source(mut self) -> Result<Self, String> {
        let proof = source_provenance(&self.source)?;
        let base = cache_id(&self.identity);
        let hash = proof["sha256"].as_str().ok_or("Missing original digest")?;
        let parent = self.output.parent().ok_or("Missing prepared parent")?;
        self.manifest = parent.join(format!("{base}-{hash}.json"));
        self.output = parent.join(format!("{base}-{hash}.mp4"));
        self.original_source = Some(proof);
        Ok(self)
    }

    pub fn new(source: &Path, shared_root: &Path, width: Option<u32>) -> Result<Self, String> {
        let source = source
            .canonicalize()
            .map_err(|error| format!("Source video inaccessible: {error}"))?;
        if !source.is_file() {
            return Err("Source video is not a file".into());
        }
        let probed: Value = ffprobe(&source)?;
        let info: VideoInfo = inspect_probe(&probed)?;
        let selected_width = width.unwrap_or(info.auto_width);
        if selected_width == 0 || !selected_width.is_multiple_of(2) || selected_width > 1920 {
            return Err("Неправильная ширина нормализации".into());
        }
        let metadata =
            fs::metadata(&source).map_err(|error| format!("Cannot stat video source: {error}"))?;
        let mtime_ns = metadata
            .modified()
            .map_err(|error| format!("Cannot read source modification: {error}"))?
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("Source modification time predates Unix epoch: {error}"))?
            .as_nanos();

        let identity = PreparedIdentity {
            source: source.to_string_lossy().into_owned(),
            size: metadata.len(),
            mtime_ns,
            width: selected_width,
            fps: 30,
            schema: 2,
        };
        let id = cache_id(&identity);
        let root = shared_root.join("prepared");
        let output = root.join(format!("{id}.mp4"));
        let manifest = root.join(format!("{id}.json"));
        let already_analysis_ready = analysis_ready_probe(&probed, &source, selected_width);
        let original_preview_safe = analysis_ready_probe(&probed, &source, info.coded_width);
        Ok(Self {
            source,
            output,
            manifest,
            identity,
            width: selected_width,
            duration_sec: info.duration_sec,
            already_analysis_ready,
            original_preview_safe,
            original_source: None,
        })
    }

    pub fn reused(&self) -> Option<PreparationReuse> {
        if self.already_analysis_ready {
            // An original file cannot be reused after its recorded source
            // stamp changed, even when a caller holds an earlier plan.
            let source_meta = fs::metadata(&self.source).ok()?;
            let stamp = source_meta
                .modified()
                .ok()?
                .duration_since(UNIX_EPOCH)
                .ok()?
                .as_nanos();
            return (source_meta.is_file()
                && source_meta.len() == self.identity.size
                && stamp == self.identity.mtime_ns)
                .then_some(PreparationReuse::Original);
        }
        let manifest_info = fs::symlink_metadata(&self.manifest).ok()?;
        if manifest_info.file_type().is_symlink()
            || !manifest_info.is_file()
            || manifest_info.len() > 64 * 1024
        {
            return None;
        }
        let raw = fs::read(&self.manifest).ok()?;
        let value = serde_json::from_slice::<Value>(&raw).ok()?;
        if value["status"].as_str() != Some("ready")
            || value.get("identity") != serde_json::to_value(&self.identity).ok().as_ref()
        {
            return None;
        }
        // Old ready manifests (Python schema 2 or early Rust without a
        // content receipt) are NOT trusted; rebuild rather than silently
        // accepting a same-size corrupted or replaced video.
        if let Some(expected) = &self.original_source {
            if &value["original_source"] != expected
                || verify_source_provenance(&self.source, expected).is_err()
            {
                return None;
            }
        }
        let declared_size = value["cache_size_bytes"].as_u64()?;
        let declared_hash = value["cache_sha256"].as_str()?;
        if declared_hash.len() != 64
            || !declared_hash
                .bytes()
                .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase())
            || declared_size == 0
        {
            return None;
        }
        prepared_receipt_matches(&self.output, declared_size, declared_hash)
            .then_some(PreparationReuse::Cached)
    }
    pub fn ready_path(&self) -> Option<&Path> {
        match self.reused()? {
            PreparationReuse::Original => Some(&self.source),
            PreparationReuse::Cached => Some(&self.output),
        }
    }

    pub fn ffmpeg_arguments(&self, partial: &Path) -> Vec<String> {
        self.ffmpeg_arguments_with_hardware(partial, true)
    }

    pub fn ffmpeg_arguments_with_hardware(&self, partial: &Path, hardware: bool) -> Vec<String> {
        let mut args: Vec<String> = ["-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
            .into_iter()
            .map(String::from)
            .collect();
        if hardware {
            args.extend(["-hwaccel".into(), "auto".into()]);
        }
        args.extend([
            "-i".into(),
            self.source.to_string_lossy().into_owned(),
            "-vf".into(),
            format!("fps=30,scale={}:-2,setsar=1", self.width),
            "-map".into(),
            "0:v:0".into(),
            "-map".into(),
            "0:a:0?".into(),
            "-c:a".into(),
            "aac".into(),
            "-b:a".into(),
            "128k".into(),
        ]);
        if hardware {
            args.extend(crate::montage::encode_flags(true));
        } else {
            args.extend(
                [
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
                    "-threads", "4",
                ]
                .into_iter()
                .map(String::from),
            );
        }
        args.extend([
            "-movflags".into(),
            "+faststart".into(),
            "-progress".into(),
            "pipe:1".into(),
            "-nostats".into(),
            partial.to_string_lossy().into_owned(),
        ]);
        args
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn original_provenance_hashes_content_and_maps_analysis_to_relative_time() {
        let root = std::env::temp_dir().join(format!("volley-provenance-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("source.mp4");
        let output = std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:r=60000/1001",
                "-t",
                "0.3",
                "-c:v",
                "libx264",
                "-threads",
                "1",
            ])
            .arg(&source)
            .output()
            .unwrap();
        assert!(output.status.success());
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&source)
            .unwrap()
            .write_all(&[0, 0, 0, 9, b'f', b'r', b'e', b'e', 0])
            .unwrap();
        let stamp = std::fs::metadata(&source).unwrap().modified().unwrap();
        let legacy_plan = super::PreparationPlan::new(&source, &root, None).unwrap();
        let bound_before = legacy_plan.clone().bind_source().unwrap();
        assert_ne!(legacy_plan.output, bound_before.output);
        let first = super::source_provenance_cached(&source, &root).unwrap();
        assert_eq!(
            first["sha256"],
            super::prepared_file_sha256(&source).unwrap().1
        );
        if super::file_change_stamp(&source, &std::fs::metadata(&source).unwrap()).is_some() {
            let receipt = super::original_receipt_file(&source, &root).unwrap();
            assert!(
                receipt.is_file(),
                "verified proof should survive a process restart"
            );
            super::SOURCES
                .get()
                .unwrap()
                .lock()
                .unwrap()
                .remove(&source);
            assert_eq!(
                super::source_provenance_cached(&source, &root).unwrap(),
                first,
                "a cold process reuses the durable proof for an unchanged file"
            );
        }
        assert_eq!(first["timeline"]["analysis_fps"], 30);
        assert_eq!(first["timeline"]["origin"], "original_relative_seconds");
        let mut bytes = std::fs::read(&source).unwrap();
        *bytes.last_mut().unwrap() = 1;
        std::fs::write(&source, bytes).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&source)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(stamp))
            .unwrap();
        let changed_plan = super::PreparationPlan::new(&source, &root, None).unwrap();
        assert_eq!(legacy_plan.output, changed_plan.output);
        assert_ne!(
            bound_before.output,
            changed_plan.bind_source().unwrap().output
        );
        assert_ne!(
            first["sha256"],
            super::source_provenance_cached(&source, &root).unwrap()["sha256"]
        );
        std::fs::remove_dir_all(root).unwrap();
    }
    use super::{cache_id, python_fingerprint_json, PreparedIdentity};

    #[test]
    fn schema2_fingerprint_has_python_key_order_and_ascii_escapes() {
        let identity = PreparedIdentity {
            source: "/media/мяч⚽.mp4".into(),
            size: 123,
            mtime_ns: 1730000000000000000,
            width: 1280,
            fps: 30,
            schema: 2,
        };
        assert_eq!(
            python_fingerprint_json(&identity),
            r#"{"fps": 30, "mtime_ns": 1730000000000000000, "schema": 2, "size": 123, "source": "/media/\u043c\u044f\u0447\u26bd.mp4", "width": 1280}"#
        );
        assert_eq!(cache_id(&identity).len(), 28);
    }

    #[test]
    fn sha_cache_detects_same_size_corruption_missing_receipt_and_symlink() {
        use super::{prepared_file_sha256, PreparationPlan, PreparationReuse};
        let root = std::env::temp_dir().join(format!("volley-cache-83-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let input = root.join("original.mp4");
        let cached = root.join("prepared.mp4");
        let receipt = root.join("prepared.json");
        std::fs::write(&input, b"original unchanged bytes").unwrap();
        let meta = std::fs::metadata(&input).unwrap();
        let stamp = meta
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let identity = PreparedIdentity {
            source: input.to_string_lossy().to_string(),
            size: meta.len(),
            mtime_ns: stamp,
            width: 640,
            fps: 30,
            schema: 2,
        };
        let plan = PreparationPlan {
            source: input.clone(),
            output: cached.clone(),
            manifest: receipt.clone(),
            identity: identity.clone(),
            width: 640,
            duration_sec: 1.0,
            already_analysis_ready: false,
            original_preview_safe: false,
            original_source: None,
        };
        std::fs::write(&cached, b"this-is-prepared-video").unwrap();
        let (size, digest) = prepared_file_sha256(&cached).unwrap();
        let entry = serde_json::json!({"status":"ready","identity":identity,
            "cache_size_bytes":size,"cache_sha256":digest,
            "cache_verification":"sha256-stream-v1"});
        std::fs::write(&receipt, serde_json::to_vec(&entry).unwrap()).unwrap();
        assert_eq!(plan.reused(), Some(PreparationReuse::Cached));
        assert_eq!(plan.ready_path(), Some(cached.as_path()));
        // The same-length bitflip used to pass all existing cache reuse checks.
        // Repeat it: on Windows two writes can land in one system clock tick
        // and share an mtime, so a single attempt only caught the bug by luck.
        for _ in 0..64 {
            let mut corrupted = std::fs::read(&cached).unwrap();
            corrupted[0] ^= 1;
            std::fs::write(&cached, corrupted).unwrap();
            assert_eq!(
                plan.reused(),
                None,
                "Same-size tampering must invalidate cache"
            );
            std::fs::write(&cached, b"this-is-prepared-video").unwrap();
            assert_eq!(plan.reused(), Some(PreparationReuse::Cached));
        }
        // A same-size edit cannot defeat the receipt cache by restoring mtime.
        // Linux ctime and inode are also part of the cached file identity.
        #[cfg(unix)]
        {
            let original_mtime = std::fs::metadata(&cached).unwrap().modified().unwrap();
            std::fs::write(&cached, b"this-is-prepared-videX").unwrap();
            std::fs::File::options()
                .write(true)
                .open(&cached)
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(original_mtime))
                .unwrap();
            assert_eq!(
                plan.reused(),
                None,
                "Restoring mtime must not hide a same-size edit"
            );
            std::fs::write(&cached, b"this-is-prepared-video").unwrap();
            assert_eq!(plan.reused(), Some(PreparationReuse::Cached));
        }
        let mut old = entry.clone();
        old.as_object_mut().unwrap().remove("cache_sha256");
        std::fs::write(&receipt, serde_json::to_vec(&old).unwrap()).unwrap();
        assert_eq!(
            plan.reused(),
            None,
            "Unsigned old cache entries require rebuilding"
        );
        let mut wrong = entry.clone();
        wrong["cache_sha256"] = serde_json::json!("x".repeat(64));
        std::fs::write(&receipt, serde_json::to_vec(&wrong).unwrap()).unwrap();
        assert_eq!(plan.reused(), None);
        std::fs::write(&receipt, serde_json::to_vec(&entry).unwrap()).unwrap();
        #[cfg(unix)]
        {
            let out = root.join("external.mp4");
            std::fs::write(&out, b"external protected").unwrap();
            std::fs::remove_file(&cached).unwrap();
            std::os::unix::fs::symlink(&out, &cached).unwrap();
            assert!(prepared_file_sha256(&cached).is_err());
            assert_eq!(
                plan.reused(),
                None,
                "Symlinked cache output cannot be reused"
            );
            std::fs::remove_file(&cached).unwrap();
            std::fs::write(&cached, b"this-is-prepared-video").unwrap();
            std::fs::remove_file(&receipt).unwrap();
            std::os::unix::fs::symlink(&out, &receipt).unwrap();
            assert_eq!(
                plan.reused(),
                None,
                "Symlinked manifest cannot authorize cache"
            );
            assert_eq!(std::fs::read(&out).unwrap(), b"external protected");
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unchanged_original_can_be_reused_but_modified_source_cannot() {
        use super::{PreparationPlan, PreparationReuse};
        let root = std::env::temp_dir().join(format!("volley-source-83-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("match.mp4");
        std::fs::write(&source, b"real analysis ready example").unwrap();
        let info = std::fs::metadata(&source).unwrap();
        let stamp = info
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let plan = PreparationPlan {
            source: source.clone(),
            output: root.join("cached.mp4"),
            manifest: root.join("cached.json"),
            identity: PreparedIdentity {
                source: source.to_string_lossy().to_string(),
                size: info.len(),
                mtime_ns: stamp,
                width: 640,
                fps: 30,
                schema: 2,
            },
            width: 640,
            duration_sec: 1.0,
            already_analysis_ready: true,
            original_preview_safe: true,
            original_source: None,
        };
        assert_eq!(plan.reused(), Some(PreparationReuse::Original));
        std::fs::write(&source, b"modified different size").unwrap();
        assert_eq!(plan.reused(), None);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn stable_id_for_same_source_fingerprint() {
        let fp = PreparedIdentity {
            source: "/tmp/a.mp4".into(),
            size: 3,
            mtime_ns: 100,
            width: 720,
            fps: 30,
            schema: 2,
        };
        assert_eq!(cache_id(&fp), cache_id(&fp));
        let mut other = fp.clone();
        other.width = 1280;
        assert_ne!(cache_id(&fp), cache_id(&other));
    }
    #[test]
    fn preparation_default_requests_gpu_encoding() {
        let plan = super::PreparationPlan {
            source: "source.mp4".into(),
            output: "prepared.mp4".into(),
            manifest: "prepared.json".into(),
            identity: super::PreparedIdentity {
                source: "source.mp4".into(),
                size: 1,
                mtime_ns: 1,
                width: 640,
                fps: 30,
                schema: 2,
            },
            width: 640,
            duration_sec: 1.0,
            already_analysis_ready: false,
            original_preview_safe: false,
            original_source: None,
        };
        let args = plan.ffmpeg_arguments(std::path::Path::new("partial.mp4"));
        let encoder = args.iter().position(|arg| arg == "-c:v").unwrap();
        assert_eq!(args[encoder + 1], "h264_nvenc");
    }
}

pub mod primary_preview;
