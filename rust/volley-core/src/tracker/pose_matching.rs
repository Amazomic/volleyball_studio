//! Lightweight measured COCO-17 anchors for provisional track association.
//! These are verified RTMPose measurements, not extrapolated joints or ReID.
#[derive(Clone, Copy, Debug)]
pub struct PoseEvidence {
    pub joints: [[f64; 3]; 17],
}

impl PoseEvidence {
    pub fn valid(&self) -> bool {
        self.joints.iter().all(|p| {
            p.iter().all(|v| v.is_finite())
                && p[0] >= 0.0
                && p[0] <= 1.0
                && p[1] >= 0.0
                && p[1] <= 1.0
                && p[2] >= 0.0
                && p[2] <= 1.0
        }) && self.joints.iter().filter(|p| p[2] >= 0.32).count() >= 6
            && [5, 6, 11, 12]
                .iter()
                .filter(|&&i| self.joints[i][2] >= 0.32)
                .count()
                >= 2
    }

    pub fn torso(&self) -> Option<[f64; 2]> {
        let mut total = [0.0, 0.0];
        let mut count = 0.0;
        for i in [5, 6, 11, 12] {
            let joint = self.joints[i];
            if joint[2] >= 0.32 {
                total[0] += joint[0];
                total[1] += joint[1];
                count += 1.0;
            }
        }
        (count >= 2.0).then(|| [total[0] / count, total[1] / count])
    }

    /// A body anchor makes crop-size changes less influential. Relative limb
    /// positions are supporting evidence, never a biometric identity claim.
    pub fn similarity(&self, next: &Self, elapsed_frames: u64) -> Option<f64> {
        let a = self.torso()?;
        let b = next.torso()?;
        let shift = (a[0] - b[0]).hypot(a[1] - b[1]);
        let movement_limit = (0.030 + elapsed_frames as f64 * 0.005).min(0.32);
        if shift > movement_limit {
            return None;
        }
        let mut pose_distance = 0.0;
        let mut matches = 0.0;
        for i in [5, 6, 7, 8, 11, 12, 13, 14] {
            let (p, q) = (self.joints[i], next.joints[i]);
            if p[2] >= 0.32 && q[2] >= 0.32 {
                let dx = (p[0] - a[0]) - (q[0] - b[0]);
                let dy = (p[1] - a[1]) - (q[1] - b[1]);
                pose_distance += (dx * dx + dy * dy).sqrt();
                matches += 1.0;
            }
        }
        let shape = if matches >= 2.0 {
            (1.0 - pose_distance / matches / 0.13).clamp(0.0, 1.0)
        } else {
            0.5
        };
        Some(0.72 * (1.0 - shift / movement_limit) + 0.28 * shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(x: f64) -> PoseEvidence {
        let mut joints = [[0.0, 0.0, 0.9]; 17];
        for (i, joint) in joints.iter_mut().enumerate() {
            *joint = [
                x + (i % 3) as f64 * 0.01,
                0.25 + (i / 3) as f64 * 0.035,
                0.9,
            ];
        }
        PoseEvidence { joints }
    }
    #[test]
    fn torso_motion_is_strong_and_teleportation_is_rejected() {
        let a = sample(0.25);
        assert!(a.valid());
        assert!(a.similarity(&sample(0.26), 3).unwrap() > 0.65);
        assert!(a.similarity(&sample(0.80), 3).is_none());
    }
    #[test]
    fn missing_torso_and_invalid_points_cannot_link() {
        let mut a = sample(0.25);
        a.joints[5][2] = 0.0;
        a.joints[6][2] = 0.0;
        a.joints[11][2] = 0.0;
        assert!(!a.valid());
        a.joints[11][2] = f64::NAN;
        assert!(!a.valid());
    }
}
