//! Conservative CPU Track ID reconstruction from player detector/RTMPose boxes.
//! IDs are provisional tracklets, NOT verified athlete identities.
//! Pairing is globally optimal and one-to-one; crossings retain living IDs
//! while ambiguous geometry is reported, not misrepresented as confirmed identity.
use crate::{bbox_iou, fusion::Candidate};
use serde::{Deserialize, Serialize};
mod association;
pub use association::assign as assign_track_scores;
pub mod bytetrack;
mod confirmation;
mod pose_matching;
pub use confirmation::{TrackConfirmation, TrackConfirmationSummary};
pub use pose_matching::PoseEvidence;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrackingOptions {
    /// Score threshold for observed detector/pose boxes, replayable on CPU.
    pub min_player_score: f64,
    pub nms_iou: f64,
    /// Maximum disappearance in frames (90 frames = 3 seconds at the 30 FPS CPU replay rate).
    pub max_gap_frames: u64,
    #[serde(default = "default_lost_low_confidence_max_gap_frames")]
    pub lost_low_confidence_max_gap_frames: u64,
    #[serde(default = "default_lost_low_confidence_min_iou")]
    pub lost_low_confidence_min_iou: f64,
    pub min_assignment_score: f64,
    /// Mark close competing matches as ambiguous without creating extra IDs.
    pub ambiguity_margin: f64,
    /// Minimum number of frames with actual observations before publishing an ID.
    /// Short unconfirmed tracklets remain only in immutable GPU evidence.
    #[serde(default = "default_min_track_observations")]
    pub min_track_observations: u64,
}
const fn default_min_track_observations() -> u64 {
    12
}
const fn default_lost_low_confidence_max_gap_frames() -> u64 {
    30
}
const fn default_lost_low_confidence_min_iou() -> f64 {
    0.30
}

impl Default for TrackingOptions {
    fn default() -> Self {
        Self {
            min_player_score: 0.28,
            nms_iou: 0.72,
            max_gap_frames: 90,
            lost_low_confidence_max_gap_frames: default_lost_low_confidence_max_gap_frames(),
            lost_low_confidence_min_iou: default_lost_low_confidence_min_iou(),
            min_assignment_score: 0.43,
            ambiguity_margin: 0.11,
            min_track_observations: default_min_track_observations(),
        }
    }
}
impl TrackingOptions {
    pub fn validate(mut self) -> Result<Self, String> {
        if !self.min_player_score.is_finite()
            || !(0.05..=1.0).contains(&self.min_player_score)
            || !self.nms_iou.is_finite()
            || !(0.6..=0.99).contains(&self.nms_iou)
            || !(1..=180).contains(&self.max_gap_frames)
            || self.lost_low_confidence_max_gap_frames > 90
            || !self.lost_low_confidence_min_iou.is_finite()
            || !(0.20..=0.85).contains(&self.lost_low_confidence_min_iou)
            || !self.min_assignment_score.is_finite()
            || !(0.2..=1.0).contains(&self.min_assignment_score)
            || !self.ambiguity_margin.is_finite()
            || !(0.0..=0.5).contains(&self.ambiguity_margin)
            || !(6..=90).contains(&self.min_track_observations)
        {
            return Err("Invalid CPU tracker options".into());
        }
        // Accept older saved profiles, but never expire a provisional track
        // before the minimum three-second grace period after its last detection.
        self.max_gap_frames = self.max_gap_frames.max(90);
        Ok(self)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TrackedPlayer {
    pub track_id: u64,
    pub detection: Candidate,
    pub is_verified_athlete: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct NewTrackDiagnostic {
    pub track_id: u64,
    pub reason: &'static str,
    pub closest_previous_track_id: Option<u64>,
    pub closest_predicted_iou: Option<f64>,
    pub frames_since_last_seen: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrackedFrame {
    pub frame_index: u64,
    pub tracks: Vec<TrackedPlayer>,
    pub retained_boxes: usize,
    pub rejected_boxes: usize,
    pub ambiguous_detections: usize,
    pub new_track_diagnostics: Vec<NewTrackDiagnostic>,
}
fn center(b: [f64; 4]) -> [f64; 2] {
    [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]
}
fn height(b: [f64; 4]) -> f64 {
    b[3] - b[1]
}
/// Same head and overlapping body mean two crops of one person, even if the
/// full/partial bounding boxes differ enough to evade ordinary box NMS.
/// Requiring a head on both proposals prevents merging close teammates
/// whose upper-body locations are merely adjacent.
pub fn same_head_body_duplicate(a: &Candidate, b: &Candidate) -> bool {
    let (Some(ha), Some(hb)) = (a.head_point_norm, b.head_point_norm) else {
        return false;
    };
    let (ba, bb) = (a.bbox_xyxy_norm, b.bbox_xyxy_norm);
    let head_dx = (ha[0] - hb[0]).abs();
    let head_dy = (ha[1] - hb[1]).abs();
    let shared_width = (ba[2].min(bb[2]) - ba[0].max(bb[0])).max(0.0);
    let narrow = (ba[2] - ba[0]).min(bb[2] - bb[0]);
    let overlap = bbox_iou(ba, bb);
    let common_y = (ba[3].min(bb[3]) - ba[1].max(bb[1])).max(0.0);
    let short_height = height(ba).min(height(bb));
    (narrow > 0.012
        && head_dx < 0.012
        && head_dy < 0.018
        && shared_width > narrow * 0.55
        && overlap > 0.30)
        // Two RTMPose crops of one head can disagree about whether legs
        // are visible. The IoU then falls below 0.30 despite almost
        // identical heads and substantial torso overlap (mm, 25:54.2).
        || (narrow > 0.012
            && head_dx < 0.008
            && head_dy < 0.009
            && shared_width > narrow * 0.78
            && common_y > short_height * 0.40
            && overlap > 0.14)
}
/// A lower-confidence partial skeleton can sit inside a fuller same-player
/// detection while its inferred nose drifts due to net occlusion. Require
/// substantial *small-box containment*, compatible heads, and a clear
/// score advantage; do not suppress neighboring complete players.
pub fn same_partial_body_duplicate(strong: &Candidate, weak: &Candidate) -> bool {
    let (Some(a_head), Some(b_head)) = (strong.head_point_norm, weak.head_point_norm) else {
        return false;
    };
    if strong.score - weak.score < 0.14 {
        return false;
    }
    let (a, b) = (strong.bbox_xyxy_norm, weak.bbox_xyxy_norm);
    let (aw, ah) = (a[2] - a[0], height(a));
    let (bw, bh) = (b[2] - b[0], height(b));
    if aw < 0.025
        || bw < 0.025
        || ah < 0.12
        || bh < 0.12
        || ah.max(bh) / ah.min(bh) > 1.7
        || (a_head[0] - b_head[0]).abs() > 0.045
        || (a_head[1] - b_head[1]).abs() > 0.065
    {
        return false;
    }
    let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let vertical = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let smaller_area = (aw * ah).min(bw * bh);
    width * vertical >= 0.60 * smaller_area && (center(a)[0] - center(b)[0]).abs() < 0.035
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn older_saved_tracking_profile_gets_default_confirmation() {
        let legacy = serde_json::json!({
            "min_player_score":0.3,"nms_iou":0.88,"max_gap_frames":12,
            "min_assignment_score":0.48,"ambiguity_margin":0.11
        });
        let settings: TrackingOptions = serde_json::from_value(legacy).unwrap();
        let settings = settings.validate().unwrap();
        assert_eq!(settings.max_gap_frames, 90);
        assert_eq!(settings.min_track_observations, 12);
        assert_eq!(settings.lost_low_confidence_max_gap_frames, 30);
        assert_eq!(settings.lost_low_confidence_min_iou, 0.30);
        let bad = TrackingOptions {
            min_track_observations: 5,
            ..settings
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn weak_lost_recovery_settings_are_validated() {
        assert!(TrackingOptions {
            lost_low_confidence_max_gap_frames: 91,
            ..TrackingOptions::default()
        }
        .validate()
        .is_err());
        assert!(TrackingOptions {
            lost_low_confidence_min_iou: 0.1,
            ..TrackingOptions::default()
        }
        .validate()
        .is_err());
        assert!(TrackingOptions {
            lost_low_confidence_max_gap_frames: 0,
            ..TrackingOptions::default()
        }
        .validate()
        .is_ok());
    }
}
