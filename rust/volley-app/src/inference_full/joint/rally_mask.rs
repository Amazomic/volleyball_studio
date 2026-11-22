use super::super::RawDetection;
use serde::{Deserialize, Serialize};

const CONTEXT_FRAMES: u64 = 30;
const MIN_DETECTIONS: usize = 45;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PrimaryRallySettings {
    pub min_trajectory_seconds: f64,
    pub merge_gap_seconds: f64,
    pub stationary_ball_ms: u32,
}
impl Default for PrimaryRallySettings {
    fn default() -> Self {
        Self {
            min_trajectory_seconds: crate::cpu_retrack::ball_track::CONTINUOUS_MIN_SECONDS,
            merge_gap_seconds: crate::cpu_retrack::ball_track::CONTINUOUS_MERGE_SECONDS,
            stationary_ball_ms: crate::cpu_retrack::ball_track::DEFAULT_STATIONARY_BALL_MS,
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
        crate::cpu_retrack::ball_track::stationary_ball_intervals(&[], self.stationary_ball_ms)?;
        Ok(())
    }
}

#[cfg(test)]
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
    min_rally_seconds: f64,
    stationary_ball_ms: u32,
    stationary_intervals: Vec<(u64, u64)>,
    unobserved_intervals: Vec<(u64, u64)>,
    movement_filter_version: &'static str,
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
        let motion = crate::cpu_retrack::ball_track::analyze_moving_ball(
            &points,
            settings.min_trajectory_seconds,
            settings.merge_gap_seconds,
            settings.stationary_ball_ms,
        )?;
        let detected = motion.detected;
        let merged = motion.merged;
        for &(start, end) in &motion.stationary {
            points[start as usize..=end as usize].fill(None);
        }
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
        } else if merged.is_empty() {
            Some("no_reliable_moving_ball_intervals")
        } else {
            None
        };
        if fallback_reason.is_some() {
            selected.clear();
        }
        Ok(Self {
            schema: "volley-primary-rally-mask/1",
            source_frame_count: frames,
            min_trajectory_seconds: settings.min_trajectory_seconds,
            merge_gap_seconds: settings.merge_gap_seconds,
            min_rally_seconds: crate::cpu_retrack::ball_track::MIN_RALLY_FRAMES as f64 / 30.0,
            stationary_ball_ms: settings.stationary_ball_ms,
            stationary_intervals: motion.stationary,
            unobserved_intervals: motion.unobserved,
            movement_filter_version: "stationary_ball_min_rally_5s_v3",
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
    fn missing_ball_does_not_select_full_video() {
        let mask = RallyMask::from_ball_samples(&[], 600, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.fallback_reason, Some("insufficient_ball_observations"));
        assert_eq!(mask.included_frames(), 0);
        assert!(!mask.includes(599));
        assert!(!mask.includes(600));
    }

    #[test]
    fn moving_ball_selects_padded_intervals_without_changing_detected_bounds() {
        let samples = moving_samples(80, 270);
        let mask =
            RallyMask::from_ball_samples(&samples, 900, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.fallback_reason, None);
        assert_eq!(mask.detected.len(), 1);
        let (start, end) = mask.detected[0];
        assert!(start >= 80 && end <= 270);
        assert!(mask.includes(start));
        assert!(mask.includes(end));
        assert!(mask.includes(start.saturating_sub(20)));
        assert!(!mask.includes(899));
        assert!(!mask.includes(start.saturating_sub(70)));
    }

    #[test]
    fn separate_play_intervals_leave_long_pauses_outside_gpu_person_scope() {
        let mut samples = moving_samples(60, 250);
        samples.extend(moving_samples(450, 660));
        let mask =
            RallyMask::from_ball_samples(&samples, 900, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.detected.len(), 2);
        assert_eq!(mask.merged.len(), 2);
        assert!(mask.includes(100));
        assert!(mask.includes(500));
        assert!(!mask.includes(300));
        assert!(mask.included_frames() < 600);
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
        let samples = moving_samples(60, 235);
        let defaults =
            RallyMask::from_ball_samples(&samples, 450, PrimaryRallySettings::default()).unwrap();
        assert_eq!(defaults.detected.len(), 1);
        let tuned = PrimaryRallySettings {
            min_trajectory_seconds: 7.0,
            merge_gap_seconds: 1.0,
            ..PrimaryRallySettings::default()
        };
        let mask = RallyMask::from_ball_samples(&samples, 450, tuned).unwrap();
        assert!(mask.detected.is_empty());
        assert_eq!(
            mask.fallback_reason,
            Some("no_reliable_moving_ball_intervals")
        );
        let serialized = serde_json::to_value(&mask).unwrap();
        assert_eq!(serialized["min_trajectory_seconds"], 7.0);
        assert_eq!(serialized["merge_gap_seconds"], 1.0);
        assert_eq!(serialized["stationary_ball_ms"], 1000);
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
        let mut samples = moving_samples(70, 270);
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
    fn long_unobserved_gap_splits_rallies_instead_of_adding_a_false_continuation() {
        let mut samples = moving_samples(120, 305);
        samples.extend(moving_samples(415, 610));
        let mask =
            RallyMask::from_ball_samples(&samples, 800, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.detected.len(), 2, "{:?}", mask.detected);
        assert_eq!(mask.merged.len(), 2, "{:?}", mask.merged);
        assert!(mask.merged[0].1 < 360 && mask.merged[1].0 > 360);
        assert!(mask
            .unobserved_intervals
            .iter()
            .any(|&(a, b)| a <= 306 && b >= 414));
        let allowed = RallyMask::from_ball_samples(
            &samples,
            800,
            PrimaryRallySettings {
                stationary_ball_ms: 4000,
                ..PrimaryRallySettings::default()
            },
        )
        .unwrap();
        assert_eq!(allowed.merged.len(), 1);
    }

    #[test]
    fn short_occlusion_can_still_merge_segments_of_one_rally() {
        let mut samples = moving_samples(120, 305);
        samples.extend(moving_samples(350, 550));
        let mask =
            RallyMask::from_ball_samples(&samples, 800, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.detected.len(), 2, "{:?}", mask.detected);
        assert_eq!(mask.merged.len(), 1, "{:?}", mask.merged);
        assert!(!mask
            .unobserved_intervals
            .iter()
            .any(|&(start, end)| start <= 349 && end >= 306));
    }

    #[test]
    fn a_held_ball_splits_rallies_despite_the_four_second_merge_gap() {
        let mut samples = moving_samples(100, 285);
        samples.extend((286..=332).map(|frame| RawDetection {
            frame_index: frame,
            class_name: "ball",
            score: 0.95,
            bbox_xyxy_norm: [
                0.49 + (frame % 3) as f32 * 0.002,
                0.4,
                0.50 + (frame % 3) as f32 * 0.002,
                0.42,
            ],
            query_index: None,
            touch_probability: None,
            head_point_norm: None,
        }));
        samples.extend(moving_samples(334, 560));
        let mask =
            RallyMask::from_ball_samples(&samples, 680, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.detected.len(), 2, "{:?}", mask.detected);
        assert_eq!(mask.merged.len(), 2, "{:?}", mask.merged);
        assert!(mask.merged[0].1 < 332 && mask.merged[1].0 > 332);
        assert!(mask
            .stationary_intervals
            .iter()
            .any(|&(start, end)| start <= 305 && end >= 325));
    }

    #[test]
    fn pre_serve_hand_hold_is_not_part_of_the_rally() {
        let mut samples = moving_samples(20, 52);
        samples.extend((53..=235).map(|frame| RawDetection {
            frame_index: frame,
            class_name: "ball",
            score: 0.96,
            bbox_xyxy_norm: [0.35, 0.48, 0.36, 0.49],
            query_index: None,
            touch_probability: None,
            head_point_norm: None,
        }));
        samples.extend(moving_samples(236, 435));
        let mask =
            RallyMask::from_ball_samples(&samples, 510, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.merged.len(), 1, "{:?}", mask.merged);
        assert!(mask.merged[0].0 >= 230, "{:?}", mask.merged);
    }

    #[test]
    fn brief_ball_apex_does_not_end_a_rally() {
        let mut samples = moving_samples(80, 178);
        samples.extend((179..=188).map(|frame| RawDetection {
            frame_index: frame,
            class_name: "ball",
            score: 0.93,
            bbox_xyxy_norm: [0.495, 0.4, 0.505, 0.42],
            query_index: None,
            touch_probability: None,
            head_point_norm: None,
        }));
        let mut second = moving_samples(189, 290);
        for item in &mut second {
            let offset = (item.frame_index - 189) as f32 * 0.003;
            item.bbox_xyxy_norm = [0.495 + offset, 0.4, 0.505 + offset, 0.42];
        }
        samples.extend(second);
        let mask =
            RallyMask::from_ball_samples(&samples, 380, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.merged.len(), 1, "{:?}", mask.merged);
        assert!(!mask
            .stationary_intervals
            .iter()
            .any(|&(start, end)| start <= 188 && end >= 179));
    }

    #[test]
    fn default_stillness_is_versioned_and_tunable() {
        let mut points = vec![None; 650];
        for (i, slot) in points.iter_mut().enumerate().take(220).skip(30) {
            *slot = Some([0.18 + (i - 30) as f64 * 0.003, 0.4]);
        }
        for (i, slot) in points.iter_mut().enumerate().take(250).skip(220) {
            if i % 11 != 0 {
                *slot = Some([0.56 + (i % 4) as f64 * 0.002, 0.41]);
            }
        }
        for (i, slot) in points.iter_mut().enumerate().take(480).skip(251) {
            *slot = Some([0.12 + (i - 251) as f64 * 0.003, 0.4]);
        }
        let strict =
            crate::cpu_retrack::ball_track::analyze_moving_ball(&points, 1.5, 4.0, 1000).unwrap();
        let sensitive =
            crate::cpu_retrack::ball_track::analyze_moving_ball(&points, 1.5, 4.0, 600).unwrap();
        assert!(sensitive
            .stationary
            .iter()
            .any(|&(s, e)| s <= 225 && e >= 243));
        assert_eq!(sensitive.merged.len(), 2, "{:?}", sensitive.merged);
        assert!(strict.stationary.is_empty());
        for bad in [0, 199, 10_001] {
            assert!(PrimaryRallySettings {
                stationary_ball_ms: bad,
                ..PrimaryRallySettings::default()
            }
            .validate()
            .is_err());
        }
    }

    #[test]
    fn short_ball_pass_at_54_seconds_is_not_a_rally() {
        let mut samples = moving_samples(1620, 1710);
        samples.extend(moving_samples(1890, 2100));
        let mask =
            RallyMask::from_ball_samples(&samples, 2400, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.fallback_reason, None);
        assert_eq!(mask.merged.len(), 1, "{:?}", mask.merged);
        assert!(mask.merged[0].0 >= 1800, "{:?}", mask.merged);
        assert_eq!(
            serde_json::to_value(&mask).unwrap()["min_rally_seconds"],
            5.0
        );
    }

    #[test]
    #[ignore = "read-only saved VBallNet evidence: set VOLLEY_RALLY_RAW, VOLLEY_RALLY_FRAMES, VOLLEY_RALLY_SPLIT_FRAMES"]
    fn readonly_saved_ball_evidence_does_not_bridge_confirmed_pause() {
        use std::io::BufRead;
        let path = std::env::var("VOLLEY_RALLY_RAW").unwrap();
        let frames: u64 = std::env::var("VOLLEY_RALLY_FRAMES")
            .unwrap()
            .parse()
            .unwrap();
        let bounds = std::env::var("VOLLEY_RALLY_SPLIT_FRAMES").unwrap();
        let bounds: Vec<u64> = bounds.split(',').map(|v| v.parse().unwrap()).collect();
        assert_eq!(bounds.len(), 2);
        assert!(bounds[0] < bounds[1]);
        let input = std::io::BufReader::new(std::fs::File::open(path).unwrap());
        let mut samples = Vec::new();
        for line in input.lines() {
            let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            for item in row["raw_detections"].as_array().unwrap() {
                if item["class_name"] != "ball" {
                    continue;
                }
                samples.push(RawDetection {
                    frame_index: item["frame_index"].as_u64().unwrap(),
                    class_name: "ball",
                    score: item["score"].as_f64().unwrap() as f32,
                    bbox_xyxy_norm: serde_json::from_value(item["bbox_xyxy_norm"].clone()).unwrap(),
                    query_index: None,
                    touch_probability: None,
                    head_point_norm: None,
                });
            }
        }
        let mask = RallyMask::from_ball_samples(&samples, frames, PrimaryRallySettings::default())
            .unwrap();
        eprintln!(
            "saved-readonly: ball={} detected={} merged={} stationary={} unobserved={}",
            mask.observed_ball_frames,
            mask.detected.len(),
            mask.merged.len(),
            mask.stationary_intervals.len(),
            mask.unobserved_intervals.len(),
        );
        assert!(mask
            .detected
            .iter()
            .any(|&(start, end)| start <= bounds[0] && bounds[0] <= end));
        assert!(mask
            .detected
            .iter()
            .any(|&(start, end)| start <= bounds[1] && bounds[1] <= end));
        assert!(
            mask.merged
                .iter()
                .all(|&(start, end)| start > bounds[0] || end < bounds[1]),
            "unobserved pause merged into a rally: {:?}",
            mask.merged
                .iter()
                .filter(|&&(start, end)| start <= bounds[1] && end >= bounds[0])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn two_short_moving_segments_form_a_rally_when_merged_over_five_seconds() {
        let mut samples = moving_samples(100, 200);
        samples.extend(moving_samples(230, 340));
        let mask =
            RallyMask::from_ball_samples(&samples, 470, PrimaryRallySettings::default()).unwrap();
        assert_eq!(mask.merged.len(), 1, "{:?}", mask.merged);
        assert!(mask.merged[0].1 - mask.merged[0].0 + 1 >= 150);
    }

    #[test]
    fn stationary_ball_does_not_trigger_full_video_people_analysis() {
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
        assert_eq!(mask.included_frames(), 0);
    }
}
