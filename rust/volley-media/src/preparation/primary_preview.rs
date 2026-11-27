use super::{prepared_receipt_matches, source_provenance};
use crate::{ffprobe, inspect_probe};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

pub const PROFILE: &str = "h264-native-pts-fit1280-v1";

#[derive(Clone, Debug)]
pub struct PrimaryPreviewPlan {
    pub output: PathBuf,
    pub manifest: PathBuf,
    pub id: String,
    source: PathBuf,
    proof: Value,
    project: String,
    video: String,
}

pub fn scope_root(root: &Path, project: &str, video: &str) -> PathBuf {
    let key = serde_json::to_vec(&(project, video)).unwrap_or_default();
    root.join("primary-playback-v1")
        .join(format!("{:x}", Sha256::digest(key)))
}

impl PrimaryPreviewPlan {
    pub fn new(source: &Path, root: &Path, project: &str, video: &str) -> Result<Self, String> {
        let proof = source_provenance(source)?;
        let hash = proof["sha256"].as_str().ok_or("Missing source SHA")?;
        let id = format!("{:x}", Sha256::digest(format!("{hash}:{PROFILE}")));
        let folder = scope_root(root, project, video).join(&id);
        Ok(Self {
            output: folder.join("preview.mp4"),
            manifest: folder.join("manifest.json"),
            id,
            source: source.to_path_buf(),
            proof,
            project: project.into(),
            video: video.into(),
        })
    }

    pub fn ready_path(&self) -> Option<&Path> {
        self.ready_media().map(|(path, _)| path)
    }

    pub fn ready_media(&self) -> Option<(&Path, String)> {
        if source_provenance(&self.source).ok()? != self.proof {
            return None;
        }
        let meta = fs::symlink_metadata(&self.manifest).ok()?;
        if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 64 * 1024 {
            return None;
        }
        let receipt: Value = serde_json::from_slice(&fs::read(&self.manifest).ok()?).ok()?;
        if receipt["schema"] != "volley-primary-playback/1"
            || receipt["state"] != "ready"
            || receipt["profile"] != PROFILE
            || receipt["project_id"] != self.project
            || receipt["video"] != self.video
            || receipt["preview_id"] != self.id
            || receipt["original_source"] != self.proof
        {
            return None;
        }
        let size = receipt["cache_size_bytes"].as_u64()?;
        let hash = receipt["cache_sha256"].as_str()?;
        if size == 0 || hash.len() != 64 || !hash.bytes().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        prepared_receipt_matches(&self.output, size, hash)
            .then(|| (self.output.as_path(), hash.to_owned()))
    }

    pub fn begin(&self) -> Result<PrimaryPreviewWrite, String> {
        let folder = self.output.parent().ok_or("Missing preview folder")?;
        fs::create_dir_all(folder).map_err(|e| e.to_string())?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        let scratch = folder.join(format!(".partial-{}-{nonce}", std::process::id()));
        fs::create_dir(&scratch).map_err(|e| e.to_string())?;
        Ok(PrimaryPreviewWrite {
            plan: self.clone(),
            path: scratch.join("preview.mp4"),
            scratch,
        })
    }
}

pub struct PrimaryPreviewWrite {
    plan: PrimaryPreviewPlan,
    scratch: PathBuf,
    pub path: PathBuf,
}

impl PrimaryPreviewWrite {
    pub fn publish(self, cancelled: &AtomicBool) -> Result<(), String> {
        check_cancelled(cancelled)?;
        if source_provenance(&self.plan.source)? != self.plan.proof {
            return Err("Preview source changed during analysis".into());
        }
        let info = inspect_probe(&ffprobe(&self.path)?)?;
        let expected = &self.plan.proof["video"];
        let duration = expected["duration_sec"]
            .as_f64()
            .ok_or("Missing source duration")?;
        let ratio = expected["width"].as_f64().ok_or("Missing source width")?
            / expected["height"].as_f64().ok_or("Missing source height")?;
        let duration_delta = (info.duration_sec - duration).abs();
        let aspect_delta = ((info.width as f64 / info.height as f64) / ratio - 1.0).abs();
        if info.codec != "h264"
            || info.width > 1280
            || info.height > 1280
            || info.rotation != 0
            || duration_delta > 0.25
            || aspect_delta > 0.02
        {
            return Err(format!(
                "Invalid primary preview dimensions, rotation or timeline: codec={}, size={}x{}, rotation={}, duration={:.3}s (source={:.3}s, delta={:.3}s), aspect_delta={:.4}",
                info.codec, info.width, info.height, info.rotation,
                info.duration_sec, duration, duration_delta, aspect_delta,
            ));
        }
        check_cancelled(cancelled)?;
        let before = fs::metadata(&self.path).map_err(|e| e.to_string())?;
        let mut file = fs::File::open(&self.path).map_err(|e| e.to_string())?;
        let mut digest = Sha256::new();
        let mut bytes = vec![0u8; 1024 * 1024];
        let mut size = 0u64;
        loop {
            check_cancelled(cancelled)?;
            let count = file.read(&mut bytes).map_err(|e| e.to_string())?;
            if count == 0 {
                break;
            }
            digest.update(&bytes[..count]);
            size += count as u64;
        }
        let after = fs::metadata(&self.path).map_err(|e| e.to_string())?;
        if size != before.len()
            || after.len() != size
            || before.modified().ok() != after.modified().ok()
        {
            return Err("Primary preview changed during verification".into());
        }
        let hash = format!("{:x}", digest.finalize());
        let receipt = json!({
            "schema":"volley-primary-playback/1","state":"ready","profile":PROFILE,
            "project_id":self.plan.project,"video":self.plan.video,"preview_id":self.plan.id,
            "original_source":self.plan.proof,"cache_size_bytes":size,"cache_sha256":hash,
            "timeline":{"origin":"original_relative_seconds","frame_timing":"native_pts"},
            "video_info":info
        });
        let manifest = self.scratch.join("manifest.json");
        fs::write(
            &manifest,
            serde_json::to_vec(&receipt).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if self.plan.ready_path().is_some() {
            return Ok(());
        }
        check_cancelled(cancelled)?;
        fs::rename(&self.path, &self.plan.output).map_err(|e| format!("Publish preview: {e}"))?;
        fs::rename(&manifest, &self.plan.manifest)
            .map_err(|e| format!("Publish preview receipt: {e}"))?;
        Ok(())
    }
}

pub fn encode_separately(plan: &PrimaryPreviewPlan, cancelled: &AtomicBool) -> Result<(), String> {
    check_cancelled(cancelled)?;
    if plan.ready_path().is_some() {
        return Ok(());
    }
    let write = plan.begin()?;
    let filter = "scale=w='min(1280,iw)':h='min(1280,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1,format=yuv420p";
    for hardware in [true, false] {
        check_cancelled(cancelled)?;
        let mut command = Command::new("ffmpeg");
        command
            .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
            .arg(&plan.source)
            .args([
                "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn", "-vf", filter, "-c:a", "aac",
                "-b:a", "128k",
            ]);
        if hardware {
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
            .arg(&write.path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let mut child = command
            .spawn()
            .map_err(|e| format!("Launch independent preview encoding: {e}"))?;
        let status = loop {
            if cancelled.load(Ordering::Acquire) {
                let _ = child.kill();
                let _ = child.wait();
                return Err("cancelled".into());
            }
            if let Some(status) = child
                .try_wait()
                .map_err(|e| format!("Wait independent preview encoder: {e}"))?
            {
                break status;
            }
            thread::sleep(Duration::from_millis(50));
        };
        if status.success() {
            return write.publish(cancelled);
        }
    }
    Err("Independent preview encoding failed with NVENC and libx264".into())
}

impl Drop for PrimaryPreviewWrite {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
        let _ = fs::remove_file(self.scratch.join("manifest.json"));
        let _ = fs::remove_dir(&self.scratch);
    }
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<(), String> {
    if cancelled.load(Ordering::Acquire) {
        Err("cancelled".into())
    } else {
        Ok(())
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::raw_decoder::{decode_full_video_shared_with_backend_and_tail, DecodePreference};

    #[test]
    fn independent_preview_preserves_rotated_native_frame_timing() {
        let root = std::env::temp_dir().join(format!(
            "volley-preview-rotation-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        fs::create_dir_all(&root).unwrap();
        let base = root.join("base.mp4");
        assert!(Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x180:rate=60:duration=0.8",
                "-c:v",
                "libx264",
                "-y",
            ])
            .arg(&base)
            .status()
            .unwrap()
            .success());
        let rotated = root.join("rotated.mp4");
        assert!(Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&base)
            .args(["-c", "copy", "-metadata:s:v:0", "rotate=90", "-y"])
            .arg(&rotated)
            .status()
            .unwrap()
            .success());
        let plan = PrimaryPreviewPlan::new(&rotated, &root, "p", "rotated.mp4").unwrap();
        encode_separately(&plan, &AtomicBool::new(false)).unwrap();
        let encoded = crate::probe(plan.ready_path().unwrap()).unwrap();
        let original = crate::probe(&rotated).unwrap();
        assert_eq!(encoded.rotation, 0);
        assert_eq!(
            (encoded.width, encoded.height),
            (original.width, original.height)
        );
        assert!((encoded.fps.unwrap() - 60.0).abs() < 0.1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn primary_inference_finishes_without_encoding_and_preview_remains_available() {
        let root = std::env::temp_dir().join(format!(
            "volley-separate-preview-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source.mp4");
        assert!(Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x180:rate=25:duration=1.16",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1.16",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                "-y",
            ])
            .arg(&source)
            .status()
            .unwrap()
            .success());
        let plan = PrimaryPreviewPlan::new(&source, &root, "p", "source.mp4").unwrap();
        let cancelled = AtomicBool::new(false);
        let mut windows = 0;
        decode_full_video_shared_with_backend_and_tail(
            &source,
            9,
            DecodePreference::Software,
            2,
            &cancelled,
            |_| {
                windows += 1;
                Ok(())
            },
            |_, _| Ok(()),
        )
        .unwrap();
        assert!(windows > 0);
        assert!(plan.ready_path().is_none());
        cancelled.store(true, Ordering::Release);
        assert_eq!(
            encode_separately(&plan, &cancelled).unwrap_err(),
            "cancelled"
        );
        assert!(plan.ready_path().is_none());
        cancelled.store(false, Ordering::Release);
        encode_separately(&plan, &cancelled).unwrap();
        assert_eq!(plan.ready_path(), Some(plan.output.as_path()));
        let probe = crate::ffprobe(&plan.output).unwrap();
        let info = crate::inspect_probe(&probe).unwrap();
        assert!((info.fps.unwrap() - 25.0).abs() < 0.1);
        assert_eq!((info.width, info.height), (320, 180));
        assert!(probe["streams"]
            .as_array()
            .unwrap()
            .iter()
            .any(|stream| stream["codec_type"] == "audio"));
        encode_separately(&plan, &cancelled).unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
