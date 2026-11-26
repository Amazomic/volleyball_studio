use super::super::RawDetection;
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicBool;
use volley_core::ball_interpolation::{
    stabilize_ball_motion, BallStabilizerSettings, ScoredBallSample, StabilizedBallMotion,
    BALL_STABILIZER_ALGORITHM_VERSION,
};
use volley_core::{court_geometry::CourtTransform, motion::GroundContact};

#[derive(Debug, Serialize)]
struct RallyBallStabilizer {
    algorithm: &'static str,
    settings: BallStabilizerSettings,
}

const CONTEXT_FRAMES: u64 = 30;
const MIN_DETECTIONS: usize = 45;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PrimaryRallySettings {
    pub min_trajectory_seconds: f64,
    pub merge_gap_seconds: f64,
    pub stationary_ball_ms: u32,
    pub padding_before_ms: u32,
    pub padding_after_ms: u32,
}
impl Default for PrimaryRallySettings {
    fn default() -> Self {
        Self {
            min_trajectory_seconds: crate::cpu_retrack::ball_track::CONTINUOUS_MIN_SECONDS,
            merge_gap_seconds: crate::cpu_retrack::ball_track::CONTINUOUS_MERGE_SECONDS,
            stationary_ball_ms: crate::cpu_retrack::ball_track::DEFAULT_STATIONARY_BALL_MS,
            padding_before_ms: 500,
            padding_after_ms: 500,
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
        if self.padding_before_ms > 60_000 || self.padding_after_ms > 60_000 {
            return Err("Добавление времени к розыгрышу должно быть от 0 до 60000 мс".into());
        }
        crate::cpu_retrack::ball_track::stationary_ball_intervals(&[], self.stationary_ball_ms)?;
        Ok(())
    }
}

fn pad_rally_intervals(
    intervals: &[(u64, u64)],
    frames: u64,
    before_ms: u32,
    after_ms: u32,
) -> Vec<(u64, u64)> {
    let to_frames = |ms: u32| (u64::from(ms) * 30 + 500) / 1000;
    let before = to_frames(before_ms);
    let after = to_frames(after_ms);
    let mut padded: Vec<_> = intervals
        .iter()
        .map(|&(start, end)| {
            (
                start.saturating_sub(before),
                end.saturating_add(after).min(frames.saturating_sub(1)),
            )
        })
        .collect();
    for i in 1..padded.len() {
        padded[i - 1].1 = padded[i - 1].1.min(intervals[i].0 - 1);
        padded[i].0 = padded[i].0.max(intervals[i - 1].1 + 1);
        if padded[i - 1].1 >= padded[i].0 {
            let left = intervals[i - 1].1;
            let right = intervals[i].0;
            let split = (left + (right - left - 1) / 2)
                .clamp(padded[i].0.max(left), padded[i - 1].1.min(right - 1));
            padded[i - 1].1 = split;
            padded[i].0 = split + 1;
        }
    }
    padded
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
    padding_before_ms: u32,
    padding_after_ms: u32,
    pub(super) rally_intervals: Vec<(u64, u64)>,
    stationary_intervals: Vec<(u64, u64)>,
    unobserved_intervals: Vec<(u64, u64)>,
    movement_filter_version: &'static str,
    context_seconds: f64,
    observed_ball_frames: usize,
    ball_stabilizer: RallyBallStabilizer,
    pub detected: Vec<(u64, u64)>,
    pub merged: Vec<(u64, u64)>,
    pub selected: Vec<(u64, u64)>,
    pub uncertain: Vec<(u64, u64)>,
    pub fallback_reason: Option<&'static str>,
}

impl RallyMask {
    pub(super) fn restrict_to_confirmed(&mut self) {
        let mut selected: Vec<(u64, u64)> = Vec::new();
        for (&(start, end), &(padded_start, padded_end)) in
            self.merged.iter().zip(&self.rally_intervals)
        {
            let left = start.saturating_sub(CONTEXT_FRAMES).min(padded_start);
            let right = end
                .saturating_add(CONTEXT_FRAMES)
                .min(self.source_frame_count - 1)
                .max(padded_end);
            if let Some(last) = selected.last_mut() {
                if left <= last.1.saturating_add(1) {
                    last.1 = last.1.max(right);
                    continue;
                }
            }
            selected.push((left, right));
        }
        self.selected = selected;
    }
    #[cfg(test)]
    pub fn from_ball_samples(
        samples: &[RawDetection],
        frames: u64,
        settings: PrimaryRallySettings,
    ) -> Result<Self, String> {
        Self::with_ball_stabilizer(
            samples,
            frames,
            settings,
            BallStabilizerSettings::default(),
            |_, _| Ok(Vec::new()),
        )
        .map(|(mask, _)| mask)
    }

    #[cfg(test)]
    fn with_ball_stabilizer(
        samples: &[RawDetection],
        frames: u64,
        settings: PrimaryRallySettings,
        stabilizer_settings: BallStabilizerSettings,
        ground_frames: impl FnOnce(&StabilizedBallMotion, &[(u64, u64)]) -> Result<Vec<u64>, String>,
    ) -> Result<(Self, StabilizedBallMotion), String> {
        let trajectory = Self::stabilize(samples, frames, stabilizer_settings)?;
        let mask =
            Self::from_stabilized(&trajectory, settings, stabilizer_settings, ground_frames)?;
        Ok((mask, trajectory))
    }

    pub(super) fn stabilize(
        samples: &[RawDetection],
        frames: u64,
        stabilizer_settings: BallStabilizerSettings,
    ) -> Result<StabilizedBallMotion, String> {
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
        let samples = points
            .into_iter()
            .zip(scores)
            .enumerate()
            .filter_map(|(frame, (point, confidence))| {
                Some(ScoredBallSample {
                    frame,
                    point: point?,
                    confidence,
                })
            })
            .collect::<Vec<_>>();
        Ok(stabilize_ball_motion(&samples, len, stabilizer_settings)?)
    }

    pub(super) fn detect(
        trajectory: &StabilizedBallMotion,
        settings: PrimaryRallySettings,
        stabilizer_settings: BallStabilizerSettings,
        court: Option<&CourtTransform>,
        dimensions: (u32, u32),
        cancelled: &AtomicBool,
    ) -> Result<(Self, Vec<GroundContact>), String> {
        let mut contacts = Vec::new();
        let mask = Self::from_stabilized(
            trajectory,
            settings,
            stabilizer_settings,
            |motion, intervals| {
                if court.is_some() {
                    let breaks =
                        crate::cpu_touches::trajectory_breaks(motion, dimensions, cancelled)?;
                    contacts = crate::cpu_touches::ground_stage::candidates(
                        motion,
                        &breaks,
                        intervals,
                        court,
                        dimensions.0,
                        dimensions.1,
                        cancelled,
                    )?;
                }
                Ok(contacts.iter().map(|contact| contact.frame).collect())
            },
        )?;
        Ok((mask, contacts))
    }

    fn from_stabilized(
        trajectory: &StabilizedBallMotion,
        settings: PrimaryRallySettings,
        stabilizer_settings: BallStabilizerSettings,
        ground_candidates: impl FnOnce(&StabilizedBallMotion, &[(u64, u64)]) -> Result<Vec<u64>, String>,
    ) -> Result<Self, String> {
        settings.validate()?;
        let frames = trajectory.points.len() as u64;
        let observed_ball_frames = trajectory.observed_frames;
        let mut points = trajectory.points.clone();
        let motion = crate::cpu_retrack::ball_track::analyze_moving_ball(
            &points,
            settings.min_trajectory_seconds,
            settings.merge_gap_seconds,
            settings.stationary_ball_ms,
        )?;
        let mut ground_frames = ground_candidates(trajectory, &motion.merged)?;
        ground_frames.sort_unstable();
        ground_frames.dedup();
        let detected = motion.detected;
        let mut grounded_intervals = Vec::new();
        let mut short_grounded = Vec::new();
        let minimum = crate::cpu_retrack::ball_track::MIN_RALLY_FRAMES;
        let preroll = u64::from(
            settings
                .padding_before_ms
                .min(PrimaryRallySettings::default().padding_before_ms),
        ) * 30
            / 1000;
        for (mut start, end) in motion.merged {
            let initial_start = start;
            loop {
                let index = ground_frames.partition_point(|&frame| frame < start);
                let Some(&ground) = ground_frames.get(index).filter(|&&frame| frame <= end) else {
                    grounded_intervals.push((start, end));
                    break;
                };
                let length = ground - start + 1;
                if start == initial_start
                    && length < minimum
                    && length + start.min(preroll) >= minimum
                {
                    short_grounded.push((start, ground));
                }
                grounded_intervals.push((start, ground));
                let next = detected.partition_point(|&(left, _)| left <= ground);
                let Some(&(next_start, _)) = detected.get(next).filter(|&&(left, _)| left <= end)
                else {
                    break;
                };
                start = next_start;
            }
        }
        let mut merged = crate::cpu_retrack::ball_track::filter_short_rallies(grounded_intervals);
        merged.extend(short_grounded);
        merged.sort_unstable();
        let padded = pad_rally_intervals(
            &merged,
            frames,
            settings.padding_before_ms,
            settings.padding_after_ms,
        );
        let merged = merged
            .into_iter()
            .zip(padded)
            .filter_map(|((start, end), (padded_start, _))| {
                (end - padded_start + 1 >= minimum).then_some((start, end))
            })
            .collect::<Vec<_>>();
        let rally_intervals = pad_rally_intervals(
            &merged,
            frames,
            settings.padding_before_ms,
            settings.padding_after_ms,
        );
        for &(start, end) in &motion.stationary {
            points[start as usize..=end as usize].fill(None);
        }
        let mut selected = Vec::<(u64, u64)>::new();
        for (&(start, end), &(padded_start, padded_end)) in merged.iter().zip(&rally_intervals) {
            let start = start.saturating_sub(CONTEXT_FRAMES).min(padded_start);
            let end = end
                .saturating_add(CONTEXT_FRAMES)
                .min(frames - 1)
                .max(padded_end);
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
            padding_before_ms: settings.padding_before_ms,
            padding_after_ms: settings.padding_after_ms,
            rally_intervals,
            stationary_intervals: motion.stationary,
            unobserved_intervals: motion.unobserved,
            movement_filter_version: crate::cpu_retrack::ball_track::MOVEMENT_FILTER_VERSION,
            context_seconds: CONTEXT_FRAMES as f64 / 30.0,
            observed_ball_frames,
            ball_stabilizer: RallyBallStabilizer {
                algorithm: BALL_STABILIZER_ALGORITHM_VERSION,
                settings: stabilizer_settings,
            },
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
    fn rally_padding_is_independent_and_clipped_without_merging() {
        assert_eq!(
            pad_rally_intervals(&[(80, 240), (500, 710)], 900, 500, 500),
            vec![(65, 255), (485, 725)]
        );
        assert_eq!(
            pad_rally_intervals(&[(4, 100), (110, 199)], 200, 1000, 500),
            vec![(0, 104), (105, 199)]
        );
        assert_eq!(
            pad_rally_intervals(&[(4, 100), (110, 199)], 200, 0, 500),
            vec![(4, 109), (110, 199)]
        );
        assert_eq!(pad_rally_intervals(&[(4, 100)], 200, 0, 0), vec![(4, 100)]);
        let mask = RallyMask::from_ball_samples(
            &moving_samples(80, 270),
            900,
            PrimaryRallySettings::default(),
        )
        .unwrap();
        assert_eq!(
            mask.rally_intervals,
            pad_rally_intervals(&mask.merged, 900, 500, 500)
        );
    }

    #[test]
    fn rally_detector_owns_ground_detection_and_preserves_movement_without_court() {
        let samples = moving_samples(80, 330);
        let settings = PrimaryRallySettings::default();
        let stabilizer = BallStabilizerSettings::default();
        let trajectory = RallyMask::stabilize(&samples, 900, stabilizer).unwrap();
        let baseline = RallyMask::from_ball_samples(&samples, 900, settings).unwrap();
        let cancel = AtomicBool::new(false);
        let (mask, contacts) = RallyMask::detect(
            &trajectory,
            settings,
            stabilizer,
            None,
            (1920, 1080),
            &cancel,
        )
        .unwrap();
        assert!(contacts.is_empty());
        assert_eq!(
            serde_json::to_value(mask).unwrap(),
            serde_json::to_value(baseline).unwrap()
        );
        let court = CourtTransform::new([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]).unwrap();
        let cancelled = AtomicBool::new(true);
        assert_eq!(
            RallyMask::detect(
                &trajectory,
                settings,
                stabilizer,
                Some(&court),
                (1920, 1080),
                &cancelled
            )
            .unwrap_err(),
            "cancelled"
        );
    }

    #[test]
    fn rolling_contact_closes_primary_rally_at_impact_before_padding() {
        let mut points = vec![None; 400];
        for (frame, point) in points.iter_mut().enumerate().take(260).skip(60) {
            let travel = if frame < 160 { frame - 60 } else { 259 - frame };
            *point = Some([
                0.35 + travel as f64 * 0.003,
                0.20 + (frame as f64 - 250.0).max(0.0) * 0.005,
            ]);
        }
        for (frame, x, y) in [
            (260, 0.35, 0.25),
            (261, 0.35, 0.26),
            (262, 0.35, 0.27),
            (263, 0.35, 0.28),
            (264, 0.35, 0.295),
            (276, 0.358, 0.267),
            (277, 0.358, 0.267),
            (278, 0.360, 0.264),
            (283, 0.362, 0.243),
            (284, 0.368, 0.240),
        ] {
            points[frame] = Some([x, y]);
        }
        for (frame, point) in points.iter_mut().enumerate().take(340).skip(285) {
            *point = Some([0.368 + (frame - 284) as f64 * 0.004, 0.240]);
        }
        let trajectory = StabilizedBallMotion {
            observed_frames: points.iter().filter(|p| p.is_some()).count(),
            confidence_frames: points.iter().filter(|p| p.is_some()).count(),
            interpolated: vec![false; points.len()],
            rejected_frames: vec![],
            points,
        };
        let court = CourtTransform::new([[0.06, 1.02], [0.34, 0.294], [0.60, 0.288], [0.89, 1.03]])
            .unwrap();
        let (mask, contacts) = RallyMask::detect(
            &trajectory,
            PrimaryRallySettings::default(),
            BallStabilizerSettings::default(),
            Some(&court),
            (3840, 2160),
            &AtomicBool::new(false),
        )
        .unwrap();
        let impact = contacts
            .iter()
            .find(|c| (264..=266).contains(&c.frame))
            .unwrap_or_else(|| {
                panic!("rolling ground contact; contacts={contacts:?}, mask={mask:?}")
            });
        assert_eq!(mask.merged.len(), 1);
        assert_eq!(mask.merged[0].1, impact.frame);
        assert_eq!(mask.rally_intervals[0].1, impact.frame + 15);
    }

    #[test]
    fn recovered_high_flight_remains_subject_to_ground_boundary() {
        let mut points = vec![None; 330];
        for (frame, point) in points.iter_mut().enumerate().take(200).skip(20) {
            let y = if frame < 185 {
                0.4
            } else {
                0.4 - (frame - 184) as f64 * 0.026
            };
            *point = Some([0.2 + (frame - 20) as f64 * 0.002, y]);
        }
        for (frame, point) in points.iter_mut().enumerate().take(244).skip(230) {
            *point = Some([
                0.59 + (frame - 230) as f64 * 0.001,
                0.02 + (frame - 230) as f64 * 0.02,
            ]);
        }
        let trajectory = StabilizedBallMotion {
            observed_frames: points.iter().filter(|p| p.is_some()).count(),
            confidence_frames: points.iter().filter(|p| p.is_some()).count(),
            interpolated: vec![false; points.len()],
            rejected_frames: vec![],
            points,
        };
        let mask = RallyMask::from_stabilized(
            &trajectory,
            PrimaryRallySettings::default(),
            BallStabilizerSettings::default(),
            |_, preliminary| {
                assert_eq!(preliminary, &[(20, 243)]);
                Ok(vec![239])
            },
        )
        .unwrap();
        assert_eq!(mask.merged, vec![(20, 239)]);
        assert_eq!(mask.rally_intervals, vec![(5, 254)]);
        assert!(mask.includes(239));
    }

    #[test]
    fn first_ground_contact_ends_rally_without_reopening_on_bounce() {
        let mut samples = moving_samples(80, 330);
        samples.extend(moving_samples(500, 710));
        let baseline =
            RallyMask::from_ball_samples(&samples, 900, PrimaryRallySettings::default()).unwrap();
        let (mask, _) = RallyMask::with_ball_stabilizer(
            &samples,
            900,
            PrimaryRallySettings::default(),
            BallStabilizerSettings::default(),
            |_, preliminary| {
                assert_eq!(preliminary, baseline.merged.as_slice());
                Ok(vec![240, 255])
            },
        )
        .unwrap();
        assert_eq!(mask.merged, vec![(80, 240), (500, 710)]);
    }

    #[test]
    fn ground_closed_rally_can_keep_short_deficit_from_configured_preroll() {
        let samples = moving_samples(80, 330);
        for (before_ms, expected) in [(500, vec![(80, 224)]), (0, vec![])] {
            let settings = PrimaryRallySettings {
                padding_before_ms: before_ms,
                ..PrimaryRallySettings::default()
            };
            let (mask, _) = RallyMask::with_ball_stabilizer(
                &samples,
                900,
                settings,
                BallStabilizerSettings::default(),
                |_, preliminary| {
                    assert_eq!(preliminary, &[(80, 321)]);
                    Ok(vec![224])
                },
            )
            .unwrap();
            assert_eq!(mask.merged, expected);
            if before_ms != 0 {
                assert_eq!(mask.rally_intervals, vec![(65, 239)]);
            }
        }
        let (mask, _) = RallyMask::with_ball_stabilizer(
            &samples,
            900,
            PrimaryRallySettings {
                padding_before_ms: 60_000,
                ..PrimaryRallySettings::default()
            },
            BallStabilizerSettings::default(),
            |_, _| Ok(vec![180]),
        )
        .unwrap();
        assert!(mask.merged.is_empty());
    }

    #[test]
    fn floor_contact_prevents_merging_distinct_flights_across_short_gap() {
        let mut samples = moving_samples(80, 280);
        samples.extend(moving_samples(315, 520));
        let baseline =
            RallyMask::from_ball_samples(&samples, 900, PrimaryRallySettings::default()).unwrap();
        assert_eq!(baseline.merged.len(), 1);
        let (mask, _) = RallyMask::with_ball_stabilizer(
            &samples,
            900,
            PrimaryRallySettings::default(),
            BallStabilizerSettings::default(),
            |_, preliminary| {
                assert_eq!(preliminary, baseline.merged.as_slice());
                Ok(vec![255, 270])
            },
        )
        .unwrap();
        assert_eq!(mask.merged, vec![(80, 255), (315, 520)]);
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
    fn primary_rally_mask_records_the_stabilizer_used_to_select_player_frames() {
        let mask = RallyMask::from_ball_samples(
            &moving_samples(80, 270),
            900,
            PrimaryRallySettings::default(),
        )
        .unwrap();
        let saved = serde_json::to_value(mask).unwrap();
        assert_eq!(
            saved["ball_stabilizer"]["algorithm"],
            volley_core::ball_interpolation::BALL_STABILIZER_ALGORITHM_VERSION
        );
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
