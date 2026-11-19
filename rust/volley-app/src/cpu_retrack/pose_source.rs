//! Verified RTMPose measurements for CPU provisional association.
//! The immutable GPU fusion still supplies balls and raw detector proposals.
//! Pose-derived tracker proposals are versioned CPU evidence, not GPU detections.
use super::safe_file;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};
use volley_core::{bbox_iou, fusion::Candidate, tracker::PoseEvidence};

const MAX_ROW: usize = 64 * 1024;
const MAX_POSES: usize = 12;

pub(super) struct Snapshot {
    pub id: String,
    pub sha: String,
    pub stride: usize,
    pub folder: PathBuf,
}

pub(super) fn locate(root: &std::path::Path, fusion_id: &str) -> Result<Option<Snapshot>, String> {
    let Some((id, sha, stride)) = crate::native_preview_pose::source_info(root, fusion_id)? else {
        return Ok(None);
    };
    Ok(Some(Snapshot {
        folder: root.join("native_gpu_pose_runs").join(&id),
        id,
        sha,
        stride,
    }))
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
    recent_pose_crops: Vec<([f64; 4], u64)>,
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
            recent_pose_crops: Vec::new(),
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
        detector: &[Candidate],
    ) -> Result<PoseFrameProjection, String> {
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
        self.recent_pose_crops
            .retain(|(_, observed)| frame.saturating_sub(*observed) <= 12);
        let Some(sample) = sample else {
            // Detector samples can fall between adjacent RTMPose samples. A
            // second detector crop of the *same* skeleton must not mint a new
            // provisional ID just because its bounding box is taller.
            let raw = detector
                .iter()
                .filter(|candidate| {
                    !self
                        .recent_pose_crops
                        .iter()
                        .any(|(crop, _)| bbox_iou(*crop, candidate.bbox_xyxy_norm) > 0.23)
                })
                .cloned()
                .collect::<Vec<_>>();
            return Ok((raw, vec![], 0));
        };
        let mut candidates = Vec::new();
        let mut landmarks = Vec::new();
        let mut crops = Vec::new();
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
            let torso = evidence
                .torso()
                .ok_or("RTMPose has no valid torso anchor")?;
            if torso[0] < bbox[0] - 0.04
                || torso[0] > bbox[2] + 0.04
                || torso[1] < bbox[1] - 0.04
                || torso[1] > bbox[3] + 0.04
            {
                continue; // Bad crop did not actually detect its own person.
            }
            // Two crop proposals can produce the same skeleton; these are one
            // person, not independent confirmation observations.
            if candidates.iter().zip(&landmarks).any(
                |(c, previous): (&Candidate, &Option<PoseEvidence>)| {
                    previous
                        .as_ref()
                        .and_then(PoseEvidence::torso)
                        .zip(evidence.torso())
                        .is_some_and(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]) < 0.012)
                        && previous.as_ref().is_some_and(|prior| {
                            let mut mean = 0.0;
                            let mut n = 0;
                            for i in [5, 6, 7, 8, 11, 12, 13, 14] {
                                let (p, q) = (prior.joints[i], evidence.joints[i]);
                                if p[2] >= 0.5 && q[2] >= 0.5 {
                                    mean += (p[0] - q[0]).hypot(p[1] - q[1]);
                                    n += 1;
                                }
                            }
                            n >= 4 && mean / (n as f64) < 0.012
                        })
                        && bbox_iou(c.bbox_xyxy_norm, estimated) > 0.35
                },
            ) {
                continue;
            }
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
            crops.push(bbox);
            self.recent_pose_crops.push((bbox, frame));
        }
        let pose_count = candidates.len() as u64;
        for raw in detector {
            if candidates.len() >= 128 {
                break;
            }
            if crops
                .iter()
                .any(|&crop| bbox_iou(crop, raw.bbox_xyxy_norm) > 0.25)
                || candidates
                    .iter()
                    .any(|pose| bbox_iou(pose.bbox_xyxy_norm, raw.bbox_xyxy_norm) > 0.28)
            {
                continue;
            }
            let mut candidate = raw.clone();
            candidate.query_index = Some(candidates.len());
            candidates.push(candidate);
            landmarks.push(None);
        }
        self.pose_rows += pose_count;
        Ok((candidates, landmarks, pose_count))
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
    if valid.len() < 6 {
        return None;
    }
    let min_x = valid.iter().map(|p| p[0]).fold(1.0, f64::min);
    let max_x = valid.iter().map(|p| p[0]).fold(0.0, f64::max);
    let min_y = valid.iter().map(|p| p[1]).fold(1.0, f64::min);
    let max_y = valid.iter().map(|p| p[1]).fold(0.0, f64::max);
    let w = max_x - min_x;
    let h = max_y - min_y;
    if w < 0.012 || h < 0.055 {
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
        let (candidates, joints, n) = input.frame(0, &[]).unwrap();
        assert_eq!(n, 1);
        assert!(candidates[0].head_point_norm.is_some());
        assert!(joints[0].as_ref().unwrap().valid());
        assert!(input.frame(1, &[]).unwrap().0.is_empty());
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
