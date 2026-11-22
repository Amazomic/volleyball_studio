use super::super::RawDetection;
use serde::{Deserialize, Serialize};

const CONTEXT_FRAMES: u64 = 30;
const MIN_DETECTIONS: usize = 45;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PrimaryRallySettings {
    pub min_trajectory_seconds: f64,
    pub merge_gap_seconds: f64,
}
impl Default for PrimaryRallySettings {
    fn default() -> Self {
        Self {
            min_trajectory_seconds: crate::cpu_retrack::ball_track::CONTINUOUS_MIN_SECONDS,
            merge_gap_seconds: crate::cpu_retrack::ball_track::CONTINUOUS_MERGE_SECONDS,
        }
    }
}
impl PrimaryRallySettings {
    pub fn validate(self) -> Result<(), String> {
        if !self.min_trajectory_seconds.is_finite()
            || !(0.1..=120.0).contains(&self.min_trajectory_seconds)
        {
            return Err(
                "Минимальная непрерывная траектория должна быть от 0,1 до 120 секунд".into(),
            );
        }
        if !self.merge_gap_seconds.is_finite() || !(0.0..=60.0).contains(&self.merge_gap_seconds) {
            return Err("Объединение розыгрышей должно быть от 0 до 60 секунд".into());
        }
        Ok(())
    }
}

fn merge_rally_intervals(intervals: &[(u64, u64)], gap_seconds: f64) -> Vec<(u64, u64)> {
    crate::cpu_retrack::ball_track::merge_continuous_intervals(intervals, gap_seconds)
        .expect("validated continuous rally merge gap")
}

#[derive(Debug, Serialize)]
pub(super) struct RallyMask {
    schema: &'static str,
    source_frame_count: u64,
    min_trajectory_seconds: f64,
    merge_gap_seconds: f64,
    context_seconds: f64,
    observed_ball_frames: usize,
    pub detected: Vec<(u64, u64)>,
    pub merged: Vec<(u64, u64)>,
    pub selected: Vec<(u64, u64)>,
    pub uncertain: Vec<(u64, u64)>,
    pub fallback_reason: Option<&'static str>,
}

impl RallyMask {
    pub fn from_ball_samples(
        samples: &[RawDetection],
        frames: u64,
        settings: PrimaryRallySettings,
    ) -> Result<Self, String> {
        settings.validate()?;
        let len = usize::try_from(frames).map_err(|_| "Rally mask frame count overflow")?;
        if len == 0 || len > 48 * 3600 * 30 {
            return Err("Rally mask video duration out of range".into());
        }
        let mut points = vec![None; len];
        let mut scores = vec![0.35f32; len];
        for sample in samples.iter().filter(|sample| sample.class_name == "ball") {
            let Ok(frame) = usize::try_from(sample.frame_index) else {
                continue;
            };
            if frame >= len || !sample.score.is_finite() || sample.score < scores[frame] {
                continue;
            }
            let rect = sample.bbox_xyxy_norm;
            if rect
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                || rect[0] >= rect[2]
                || rect[1] >= rect[3]
            {
                continue;
            }
            points[frame] = Some([
                f64::from((rect[0] + rect[2]) * 0.5),
                f64::from((rect[1] + rect[3]) * 0.5),
            ]);
            scores[frame] = sample.score;
        }
        let observed_ball_frames = points.iter().filter(|point| point.is_some()).count();
        let detected = crate::cpu_retrack::ball_track::continuous_intervals_from_positions(
            &points,
            settings.min_trajectory_seconds,
        )?;
        let merged = merge_rally_intervals(&detected, settings.merge_gap_seconds);
        let mut selected = Vec::<(u64, u64)>::new();
        for &(start, end) in &merged {
            let start = start.saturating_sub(CONTEXT_FRAMES);
            let end = end.saturating_add(CONTEXT_FRAMES).min(frames - 1);
            if let Some(last) = selected.last_mut() {
                if start <= last.1.saturating_add(1) {
                    last.1 = last.1.max(end);
                    continue;
                }
            }
            selected.push((start, end));
        }
        let short_motion =
            crate::cpu_retrack::ball_track::continuous_intervals_from_positions(&points, 0.5)?;
        let uncertain = short_motion
            .into_iter()
            .filter(|&(start, end)| {
                let covered = selected
                    .iter()
                    .map(|&(left, right)| {
                        let overlap_start = start.max(left);
                        let overlap_end = end.min(right);
                        if overlap_start > overlap_end {
                            0
                        } else {
                            overlap_end - overlap_start + 1
                        }
                    })
                    .sum::<u64>();
                covered.saturating_mul(2) < end - start + 1
            })
            .collect::<Vec<_>>();
        if !uncertain.is_empty() {
            for &(start, end) in &uncertain {
                selected.push((
                    start.saturating_sub(CONTEXT_FRAMES),
                    end.saturating_add(CONTEXT_FRAMES).min(frames - 1),
                ));
            }
            selected.sort_unstable_by_key(|&(start, end)| (start, end));
            let mut normalized = Vec::<(u64, u64)>::new();
            for (start, end) in selected {
                if let Some(last) = normalized.last_mut() {
                    if start <= last.1.saturating_add(1) {
                        last.1 = last.1.max(end);
                        continue;
                    }
                }
                normalized.push((start, end));
            }
            selected = normalized;
        }
        let fallback_reason = if observed_ball_frames < MIN_DETECTIONS {
            Some("insufficient_ball_observations")
        } else if detected.is_empty() {
            Some("no_reliable_moving_ball_intervals")
        } else {
            None
        };
        if fallback_reason.is_some() {
            selected.clear();
            selected.push((0, frames - 1));
        }
        Ok(Self {
            schema: "volley-primary-rally-mask/1",
            source_frame_count: frames,
            min_trajectory_seconds: settings.min_trajectory_seconds,
            merge_gap_seconds: settings.merge_gap_seconds,
            context_seconds: CONTEXT_FRAMES as f64 / 30.0,
            observed_ball_frames,
            detected,
            merged,
            selected,
            uncertain,
            fallback_reason,
        })
    }

    pub fn includes(&self, frame: u64) -> bool {
        let index = self.selected.partition_point(|&(start, _)| start <= frame);
        index > 0 && frame <= self.selected[index - 1].1
    }

    pub fn sampled_frames(&self, stride: usize) -> u64 {
        let period = stride as u64;
        self.selected
            .iter()
            .map(|&(first, last)| (last / period + 1) - (first.saturating_add(period - 1) / period))
            .sum()
    }

    pub fn included_frames(&self) -> u64 {
        self.selected
            .iter()
            .map(|&(start, end)| end - start + 1)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moving_samples(start: u64, end: u64) -> Vec<RawDetection> {
        (start..=end)
            .map(|frame| {
                let x = (0.2 + (frame - start) as f32 * 0.003).min(0.9);
                RawDetection {
                    frame_index: frame,
                    class_name: "ball",
                    score: 0.9,
                    bbox_xyxy_norm: [x - 0.005, 0.4, x + 0.005, 0.42],
                    query_index: None,
                    touch_probability: None,
                    head_point_norm: None,
                }
            })
            .collect()
    }

    #[test]
    fn missing_ball_falls_back_to_full_analysis() {
        let mask = RallyMask::from_ball_samples(&[], 600, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.fallback_reason, Some("insufficient_ball_observations"));
        assert_eq!(mask.included_frames(), 600);
        assert!(mask.includes(599));
        assert!(!mask.includes(600));
    }

    #[test]
    fn moving_ball_selects_padded_intervals_without_changing_detected_bounds() {
        let samples = moving_samples(80, 180);
        let mask =
            RallyMask::from_ball_samples(&samples, 900, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.fallback_reason, None);
        assert_eq!(mask.detected.len(), 1);
        let (start, end) = mask.detected[0];
        assert!(start >= 80 && end <= 180);
        assert!(mask.includes(start));
        assert!(mask.includes(end));
        assert!(mask.includes(start.saturating_sub(20)));
        assert!(!mask.includes(899));
        assert!(!mask.includes(start.saturating_sub(70)));
    }

    #[test]
    fn separate_play_intervals_leave_long_pauses_outside_gpu_person_scope() {
        let mut samples = moving_samples(60, 150);
        samples.extend(moving_samples(450, 540));
        let mask =
            RallyMask::from_ball_samples(&samples, 900, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.detected.len(), 2);
        assert_eq!(mask.merged.len(), 2);
        assert!(mask.includes(100));
        assert!(mask.includes(500));
        assert!(!mask.includes(300));
        assert!(mask.included_frames() < 400);
        let sampled = (0..900)
            .filter(|frame| frame % 3 == 0 && mask.includes(*frame))
            .count() as u64;
        assert_eq!(mask.sampled_frames(3), sampled);
        assert_eq!(mask.context_seconds, 1.0);
    }

    #[test]
    fn merging_is_strictly_less_than_four_seconds() {
        assert_eq!(
            merge_rally_intervals(&[(30, 100), (221, 300)], 4.0),
            vec![(30, 100), (221, 300)]
        );
        assert_eq!(
            merge_rally_intervals(&[(30, 100), (220, 300)], 4.0),
            vec![(30, 300)]
        );
        assert_eq!(
            merge_rally_intervals(&[(30, 100), (101, 200)], 4.0),
            vec![(30, 200)]
        );
    }

    #[test]
    fn primary_mask_uses_its_own_thresholds_and_saves_them() {
        let samples = moving_samples(60, 155);
        let defaults =
            RallyMask::from_ball_samples(&samples, 450, PrimaryRallySettings::default()).unwrap();
        assert_eq!(defaults.detected.len(), 1);
        let tuned = PrimaryRallySettings {
            min_trajectory_seconds: 4.0,
            merge_gap_seconds: 1.0,
        };
        let mask = RallyMask::from_ball_samples(&samples, 450, tuned).unwrap();
        assert!(mask.detected.is_empty());
        assert_eq!(
            mask.fallback_reason,
            Some("no_reliable_moving_ball_intervals")
        );
        let serialized = serde_json::to_value(&mask).unwrap();
        assert_eq!(serialized["min_trajectory_seconds"], 4.0);
        assert_eq!(serialized["merge_gap_seconds"], 1.0);
        assert_eq!(
            merge_rally_intervals(&[(30, 100), (175, 245)], 1.0).len(),
            2,
        );
        assert_eq!(
            merge_rally_intervals(&[(30, 100), (175, 245)], 4.0).len(),
            1,
        );
        for seconds in [f64::NAN, f64::INFINITY, 0.0, 121.0] {
            assert!(PrimaryRallySettings {
                min_trajectory_seconds: seconds,
                ..PrimaryRallySettings::default()
            }
            .validate()
            .is_err());
        }
    }

    #[test]
    fn short_ambiguous_flight_is_scanned_but_not_confirmed_as_rally() {
        let mut samples = moving_samples(70, 180);
        samples.extend(moving_samples(580, 605));
        let mask =
            RallyMask::from_ball_samples(&samples, 1000, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.detected.len(), 1);
        assert!(mask
            .uncertain
            .iter()
            .any(|&(start, end)| start <= 590 && end >= 590));
        assert!(mask.includes(590));
        assert!(!mask.includes(940));
    }

    #[test]
    fn stationary_ball_does_not_hide_people_analysis() {
        let samples = (0..450)
            .map(|frame| RawDetection {
                frame_index: frame,
                class_name: "ball",
                score: 0.96,
                bbox_xyxy_norm: [0.35, 0.45, 0.36, 0.46],
                query_index: None,
                touch_probability: None,
                head_point_norm: None,
            })
            .collect::<Vec<_>>();
        let mask =
            RallyMask::from_ball_samples(&samples, 450, PrimaryRallySettings::default()).unwrap();
        assert_eq!(
            mask.fallback_reason,
            Some("no_reliable_moving_ball_intervals")
        );
        assert_eq!(mask.included_frames(), 450);
    }
}
