//! Verified RTMPose measurements for CPU provisional association.
//! The immutable GPU fusion still supplies balls and raw detector proposals.
//! Pose-derived tracker proposals are versioned CPU evidence, not GPU detections.
use super::{
    safe_file, storage, MAX_INPUT_BYTES, MAX_MANIFEST_BYTES, POSE_REFINEMENT_POLICY,
    TRACK_ASSOCIATION_VERSION,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};
use volley_core::{fusion::Candidate, tracker::PoseEvidence};

const MAX_ROW: usize = 512 * 1024;
const MAX_POSES: usize = 128;

pub(super) struct Snapshot {
    pub id: String,
    pub sha: String,
    pub stride: usize,
    pub folder: PathBuf,
}

pub(super) fn locate_refined(
    root: &Path,
    fusion_id: &str,
    fusion_sha: &str,
    mask_sha: Option<&str>,
    crop_profile: Option<(&volley_core::tracker::TrackingOptions, Option<usize>, bool)>,
) -> Result<Option<Snapshot>, String> {
    let parent = root.join("native_cpu_replays");
    match fs::symlink_metadata(&parent) {
        Ok(_) => storage::safe_folder(&parent)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    let mut best: Option<Snapshot> = None;
    for (index, entry) in fs::read_dir(&parent)
        .map_err(|e| e.to_string())?
        .enumerate()
    {
        if index >= 10_000 {
            return Err("Too many CPU pose evidence versions".into());
        }
        let entry = entry.map_err(|e| e.to_string())?;
        let id = entry.file_name().to_string_lossy().into_owned();
        if !super::opaque_id(&id) {
            continue;
        }
        let folder = entry.path();
        if storage::safe_folder(&folder).is_err() {
            continue;
        }
        let Ok(file) = safe_file(&folder, "manifest.json", MAX_MANIFEST_BYTES) else {
            continue;
        };
        let Ok(bytes) = fs::read(file) else { continue };
        let Ok(record) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        if record["schema"] != "volley-native-cpu-tracking/1"
            || record["replay_id"] != id
            || record["source_fusion_id"] != fusion_id
            || record["source_fusion_sha256"] != fusion_sha
            || record["player_frame_scope_sha256"].as_str() != mask_sha
            || record["track_association_version"] != TRACK_ASSOCIATION_VERSION
            || record["pose_refinement_policy"] != POSE_REFINEMENT_POLICY
            || record["refined_pose_model_sha256"] != volley_inference::pose::MODEL_SHA256
            || record["refined_pose_source_replay_id"] != id
            || !record["pose_refinement_input_sha256"]
                .as_str()
                .is_some_and(volley_core::contracts::sha256)
        {
            continue;
        }
        let Some(sha) = record["refined_pose_sha256"]
            .as_str()
            .filter(|s| volley_core::contracts::sha256(s))
        else {
            continue;
        };
        let Some(stride) = record["refined_pose_stride"]
            .as_u64()
            .filter(|s| (1..=30).contains(s))
        else {
            continue;
        };
        if safe_file(&folder, "poses.jsonl", MAX_INPUT_BYTES).is_err() {
            continue;
        }
        let stride = stride as usize;
        if let Some((settings, requested_stride, interpolate)) = crop_profile {
            let Ok(stored) = serde_json::from_value::<volley_core::tracker::TrackingOptions>(
                record["settings"].clone(),
            ) else {
                continue;
            };
            if requested_stride.is_some_and(|requested| stride != requested)
                || record["interpolate_players"] != interpolate
                || serde_json::to_value(stored).ok() != serde_json::to_value(settings).ok()
            {
                continue;
            }
        }
        if best
            .as_ref()
            .is_none_or(|old| (stride, &id) < (old.stride, &old.id))
        {
            best = Some(Snapshot {
                id,
                sha: sha.to_owned(),
                stride,
                folder,
            });
        }
    }
    Ok(best)
}

type PoseFrameProjection = (Vec<Candidate>, Vec<Option<PoseEvidence>>, u64);

pub(super) struct PoseRows {
    reader: BufReader<File>,
    sha: Sha256,
    expected_sha: String,
    stride: u64,
    next: Option<Value>,
    consumed: u64,
    pose_rows: u64,
}

impl PoseRows {
    pub fn open(snapshot: &Snapshot) -> Result<Self, String> {
        let path = safe_file(&snapshot.folder, "poses.jsonl", 64 * 1024 * 1024 * 1024)?;
        let mut rows = Self {
            reader: BufReader::new(File::open(path).map_err(|e| e.to_string())?),
            sha: Sha256::new(),
            expected_sha: snapshot.sha.clone(),
            stride: snapshot.stride as u64,
            next: None,
            consumed: 0,
            pose_rows: 0,
        };
        rows.next = rows.read_one()?;
        Ok(rows)
    }

    fn read_one(&mut self) -> Result<Option<Value>, String> {
        let mut line = Vec::new();
        let len = self
            .reader
            .read_until(b'\n', &mut line)
            .map_err(|e| format!("Read RTMPose evidence: {e}"))?;
        if len == 0 {
            return Ok(None);
        }
        if len > MAX_ROW || line.last() != Some(&b'\n') {
            return Err("RTMPose source row too large or truncated".into());
        }
        self.sha.update(&line);
        let value: Value =
            serde_json::from_slice(&line).map_err(|e| format!("Decode RTMPose source row: {e}"))?;
        if value["frame"].as_u64() != Some(self.consumed * self.stride)
            || !value["players"].is_array()
            || value["players"]
                .as_array()
                .is_some_and(|v| v.len() > MAX_POSES)
        {
            return Err("RTMPose frames not ordered at the saved sampling stride".into());
        }
        self.consumed += 1;
        Ok(Some(value))
    }

    /// All returned keypoints were measured by RTMPose on the actual video
    /// frame, even when the independently saved crop plan inferred the box.
    pub fn frame(
        &mut self,
        frame: u64,
        _detector: &[Candidate],
    ) -> Result<PoseFrameProjection, String> {
        let sample = self.take_sample(frame)?;
        let Some(sample) = sample else {
            return Ok((Vec::new(), Vec::new(), 0));
        };
        let mut candidates = Vec::new();
        let mut landmarks = Vec::new();
        for pose in sample["players"]
            .as_array()
            .ok_or("Missing RTMPose players")?
        {
            match pose["crop_origin"].as_str() {
                Some("observed" | "interpolated_box") => {}
                None if pose["crop_origin"].is_null() => {}
                _ => return Err("Unknown RTMPose crop origin".into()),
            }
            let Some(bbox) = bbox(pose["box"].as_array()) else {
                return Err("Invalid RTMPose crop box".into());
            };
            let joints = pose["keypoints"]
                .as_array()
                .ok_or("Missing RTMPose keypoints")?;
            if joints.len() != 17 {
                return Err("RTMPose must provide 17 joints per person".into());
            }
            let mut points = [[0.0; 3]; 17];
            for (i, joint) in joints.iter().enumerate() {
                let row = joint
                    .as_array()
                    .filter(|parts| parts.len() == 3)
                    .ok_or("Invalid RTMPose joint shape")?;
                for (dst, src) in points[i].iter_mut().zip(row) {
                    *dst = src.as_f64().ok_or("Invalid RTMPose joint coordinate")?;
                }
            }
            let evidence = PoseEvidence { joints: points };
            if !evidence.valid() {
                continue; // Never promote an uncertain crop to a new Track ID.
            }
            let Some(estimated) = joint_bbox(&evidence) else {
                continue;
            };
            if !evidence.inside_crop(bbox, 0.04) {
                continue; // Bad crop did not actually detect its own person.
            }
            // Two crop proposals can produce the same skeleton; these are one
            // person, not independent confirmation observations.

            let quality = points
                .iter()
                .filter(|p| p[2] >= 0.32)
                .map(|p| p[2])
                .sum::<f64>()
                / points.iter().filter(|p| p[2] >= 0.32).count() as f64;
            let head = (points[0][2] >= 0.32).then_some([points[0][0], points[0][1]]);
            candidates.push(Candidate {
                frame_index: frame,
                class_name: "player".into(),
                score: quality.max(0.35),
                bbox_xyxy_norm: estimated,
                query_index: Some(candidates.len()),
                touch_probability: None,
                head_point_norm: head,
            });
            landmarks.push(Some(evidence));
        }
        let pose_count = candidates.len() as u64;
        self.pose_rows += pose_count;
        Ok((candidates, landmarks, pose_count))
    }

    pub(super) fn take_sample(&mut self, frame: u64) -> Result<Option<Value>, String> {
        let sample = match self.next.as_ref().and_then(|row| row["frame"].as_u64()) {
            Some(at) if at < frame => {
                return Err("Pose source stream fell behind CPU replay".into())
            }
            Some(at) if at == frame => {
                let row = self.next.take().ok_or("Missing indexed pose source")?;
                self.next = self.read_one()?;
                Some(row)
            }
            _ => None,
        };
        Ok(sample)
    }
    pub fn skip_frame(&mut self, frame: u64) -> Result<(), String> {
        match self.next.as_ref().and_then(|row| row["frame"].as_u64()) {
            Some(at) if at < frame => {
                return Err("Pose source stream fell behind CPU replay".into())
            }
            Some(at) if at == frame => {
                self.next = self.read_one()?;
            }
            _ => (),
        }
        Ok(())
    }

    pub fn finish(mut self, count: u64) -> Result<u64, String> {
        while self.next.is_some() {
            if self
                .next
                .as_ref()
                .and_then(|v| v["frame"].as_u64())
                .is_some_and(|f| f >= count)
            {
                return Err("Pose measurements exceed fusion frame count".into());
            }
            self.next = self.read_one()?;
        }
        if format!("{:x}", self.sha.finalize()) != self.expected_sha {
            return Err("RTMPose source SHA changed during CPU tracking".into());
        }
        Ok(self.pose_rows)
    }
}

fn bbox(row: Option<&Vec<Value>>) -> Option<[f64; 4]> {
    let row = row?;
    if row.len() != 4 {
        return None;
    }
    let mut box4 = [0.0; 4];
    for (dst, v) in box4.iter_mut().zip(row) {
        *dst = v.as_f64()?;
        if !dst.is_finite() || *dst < 0.0 || *dst > 1.0 {
            return None;
        }
    }
    (box4[0] < box4[2] && box4[1] < box4[3]).then_some(box4)
}
fn joint_bbox(pose: &PoseEvidence) -> Option<[f64; 4]> {
    let valid = pose
        .joints
        .iter()
        .filter(|p| p[2] >= 0.32)
        .collect::<Vec<_>>();
    if valid.len() < 3 {
        return None;
    }
    let min_x = valid.iter().map(|p| p[0]).fold(1.0, f64::min);
    let max_x = valid.iter().map(|p| p[0]).fold(0.0, f64::max);
    let min_y = valid.iter().map(|p| p[1]).fold(1.0, f64::min);
    let max_y = valid.iter().map(|p| p[1]).fold(0.0, f64::max);
    let w = max_x - min_x;
    let h = max_y - min_y;
    if w < 0.003 || h < 0.008 {
        return None;
    }
    Some([
        (min_x - w * 0.06).max(0.0),
        (min_y - h * 0.06).max(0.0),
        (max_x + w * 0.06).min(1.0),
        (max_y + h * 0.06).min(1.0),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn pose(x: f64, crop: &str) -> Value {
        let joints = (0..17)
            .map(|i| {
                json!([
                    x + (i % 3) as f64 * 0.014,
                    0.2 + (i / 3) as f64 * 0.055,
                    0.94
                ])
            })
            .collect::<Vec<_>>();
        json!({"crop_origin":crop,"box":[x-0.02,0.15,x+0.10,0.75],"keypoints":joints})
    }
    #[test]
    fn measured_pose_survives_missing_detector_and_marks_crop_source() {
        let dir = std::env::temp_dir().join(format!("volley-pose-input-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes =
            serde_json::to_vec(&json!({"frame":0,"players":[pose(0.3,"interpolated_box")]}))
                .unwrap();
        bytes.push(b'\n');
        std::fs::write(dir.join("poses.jsonl"), &bytes).unwrap();
        let snapshot = Snapshot {
            id: "id".into(),
            sha: format!("{:x}", Sha256::digest(&bytes)),
            stride: 3,
            folder: dir.clone(),
        };
        let mut input = PoseRows::open(&snapshot).unwrap();
        let unrelated = Candidate {
            frame_index: 0,
            class_name: "player".into(),
            score: 0.95,
            bbox_xyxy_norm: [0.75, 0.20, 0.86, 0.80],
            query_index: Some(0),
            touch_probability: None,
            head_point_norm: None,
        };
        let (candidates, joints, n) = input.frame(0, std::slice::from_ref(&unrelated)).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(n, 1);
        assert!(candidates[0].head_point_norm.is_some());
        assert!(joints[0].as_ref().unwrap().valid());
        let mut next_unrelated = unrelated;
        next_unrelated.frame_index = 1;
        assert!(input.frame(1, &[next_unrelated]).unwrap().0.is_empty());
        input.finish(3).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn missing_joint_evidence_never_invents_track() {
        let bad = json!({"crop_origin":"interpolated_box","box":[0.2,0.2,0.4,0.8],
                        "keypoints":vec![json!([0.3,0.4,0.0]);17]});
        let dir = std::env::temp_dir().join(format!("volley-pose-input-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes = serde_json::to_vec(&json!({"frame":0,"players":[bad]})).unwrap();
        bytes.push(b'\n');
        std::fs::write(dir.join("poses.jsonl"), &bytes).unwrap();
        let snapshot = Snapshot {
            id: "id".into(),
            sha: format!("{:x}", Sha256::digest(&bytes)),
            stride: 3,
            folder: dir.clone(),
        };
        let mut input = PoseRows::open(&snapshot).unwrap();
        let (candidates, _, n) = input.frame(0, &[]).unwrap();
        assert!(candidates.is_empty());
        assert_eq!(n, 0);
        input.finish(3).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }
}
