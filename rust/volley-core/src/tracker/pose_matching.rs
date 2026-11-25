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
        }) && (self.joints[5..].iter().filter(|p| p[2] >= 0.5).count() >= 3
            || (self.joints.iter().filter(|p| p[2] >= 0.32).count() >= 6
                && self.joints[5..].iter().filter(|p| p[2] >= 0.32).count() >= 2))
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

    pub fn inside_crop(&self, bbox: [f64; 4], tolerance: f64) -> bool {
        let reliable = self.joints.iter().filter(|p| p[2] >= 0.32);
        let total = reliable.clone().count();
        let inside = reliable
            .filter(|p| {
                p[0] >= bbox[0] - tolerance
                    && p[0] <= bbox[2] + tolerance
                    && p[1] >= bbox[1] - tolerance
                    && p[1] <= bbox[3] + tolerance
            })
            .count();
        inside >= 3 && inside * 3 >= total * 2
    }

    pub fn same_observation(&self, other: &Self) -> bool {
        let mut total = 0;
        let mut near = 0;
        let mut body = 0;
        for (i, (p, q)) in self.joints.iter().zip(other.joints.iter()).enumerate() {
            if p[2].min(q[2]) < 0.5 {
                continue;
            }
            total += 1;
            if (p[0] - q[0]).hypot(p[1] - q[1]) <= 0.012 {
                near += 1;
                body += usize::from(i >= 5);
            }
        }
        near >= 3 && body >= 2 && near * 5 >= total * 4
    }

    pub fn distinct_observation(&self, other: &Self) -> bool {
        let mut total = 0;
        let mut far = 0;
        for (p, q) in self.joints.iter().zip(other.joints.iter()) {
            if p[2].min(q[2]) < 0.5 {
                continue;
            }
            total += 1;
            far += usize::from((p[0] - q[0]).hypot(p[1] - q[1]) > 0.025);
        }
        total >= 3 && far * 3 >= total * 2
    }

    pub fn shared_joint_distance(&self, next: &Self) -> Option<f64> {
        let mut distances = self
            .joints
            .iter()
            .zip(next.joints.iter())
            .filter(|(p, q)| p[2].min(q[2]) >= 0.32)
            .map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1]))
            .collect::<Vec<_>>();
        if distances.len() < 3 {
            return None;
        }
        distances.sort_by(f64::total_cmp);
        Some(distances[distances.len() / 2])
    }

    pub fn similarity(&self, next: &Self, elapsed_frames: u64) -> Option<f64> {
        let shift = self.shared_joint_distance(next)?;
        let movement_limit = (0.030 + elapsed_frames as f64 * 0.005).min(0.32);
        if shift > movement_limit {
            return None;
        }
        let mut offsets = self
            .joints
            .iter()
            .zip(next.joints.iter())
            .filter(|(p, q)| p[2].min(q[2]) >= 0.32)
            .map(|(p, q)| [q[0] - p[0], q[1] - p[1]])
            .collect::<Vec<_>>();
        let mut ys = offsets.iter().map(|p| p[1]).collect::<Vec<_>>();
        offsets.sort_by(|a, b| a[0].total_cmp(&b[0]));
        ys.sort_by(f64::total_cmp);
        let middle = offsets.len() / 2;
        let (dx, dy) = (offsets[middle][0], ys[middle]);
        let shape = (1.0
            - offsets
                .iter()
                .map(|p| (p[0] - dx).hypot(p[1] - dy))
                .sum::<f64>()
                / offsets.len() as f64
                / 0.13)
            .clamp(0.0, 1.0);
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
    fn matching_visible_leg_does_not_require_common_torso() {
        let mut a = sample(0.25);
        let mut b = sample(0.251);
        for i in [5, 6, 11, 12] {
            a.joints[i][2] = 0.0;
            b.joints[i][2] = 0.0;
        }
        assert!(a.valid());
        assert!(a.similarity(&b, 1).unwrap() > 0.85);
    }
    #[test]
    fn torso_motion_is_strong_and_teleportation_is_rejected() {
        let a = sample(0.25);
        assert!(a.valid());
        assert!(a.similarity(&sample(0.26), 3).unwrap() > 0.65);
        assert!(a.similarity(&sample(0.80), 3).is_none());
    }
    #[test]
    fn invalid_points_cannot_link() {
        let mut a = sample(0.25);
        a.joints[5][2] = 0.0;
        a.joints[6][2] = 0.0;
        a.joints[11][2] = 0.0;
        assert!(a.valid());
        a.joints[11][2] = f64::NAN;
        assert!(!a.valid());
    }
}
