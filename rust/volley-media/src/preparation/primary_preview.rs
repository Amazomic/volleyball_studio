use super::{prepared_receipt_matches, source_provenance};
use crate::{ffprobe, inspect_probe};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{SystemTime, UNIX_EPOCH},
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
        prepared_receipt_matches(&self.output, size, hash).then_some(self.output.as_path())
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
        if info.codec != "h264"
            || info.width > 1280
            || info.height > 1280
            || info.rotation != 0
            || (info.duration_sec - duration).abs() > 0.25
            || ((info.width as f64 / info.height as f64) / ratio - 1.0).abs() > 0.02
        {
            return Err("Invalid primary preview dimensions, rotation or timeline".into());
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
