use std::collections::VecDeque;

const DETECTOR_WINDOW: u64 = 9;
const MAX_RALLY_OBSERVATIONS: usize = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RallyWindow {
    pub left: u64,
    pub right: u64,
    pub start: u64,
    pub end: u64,
}

pub(crate) fn rally_windows(
    selected: &[(u64, u64)],
    rallies: &[(u64, u64)],
) -> Result<Vec<RallyWindow>, String> {
    for intervals in [selected, rallies] {
        if intervals.iter().any(|&(start, end)| start > end)
            || intervals.windows(2).any(|pair| pair[0].1 >= pair[1].0)
        {
            return Err("Invalid rally work intervals".into());
        }
    }
    let mut windows = Vec::new();
    let mut next = 0;
    for &(selected_start, selected_end) in selected {
        let first = next;
        while let Some(&(start, end)) = rallies.get(next) {
            if start > selected_end {
                break;
            }
            if start < selected_start || end > selected_end {
                return Err("Confirmed rally is outside its selected interval".into());
            }
            next += 1;
        }
        let group = &rallies[first..next];
        for (i, &(start, end)) in group.iter().enumerate() {
            let midpoint = |a: u64, b: u64| a + (b - a) / 2;
            let left = if i == 0 {
                selected_start
            } else {
                midpoint(group[i - 1].1, start) + 1
            };
            let right = if i + 1 == group.len() {
                selected_end
            } else {
                midpoint(end, group[i + 1].0)
            };
            windows.push(RallyWindow {
                left,
                right,
                start,
                end,
            });
        }
    }
    if next != rallies.len() {
        return Err("Confirmed rally is outside the selected intervals".into());
    }
    Ok(windows)
}

#[derive(Debug)]
pub(crate) struct CompletedRally<T> {
    pub window: RallyWindow,
    pub observations: Vec<T>,
}

pub(crate) struct RallyHandoff<T> {
    pending: VecDeque<RallyWindow>,
    observations: Vec<T>,
    next_frame: u64,
    frames: u64,
}

impl<T: Clone> RallyHandoff<T> {
    pub(crate) fn new(windows: Vec<RallyWindow>, frames: u64) -> Result<Self, String> {
        if frames == 0
            || !frames.is_multiple_of(DETECTOR_WINDOW)
            || windows.iter().any(|w| {
                !(w.left <= w.start && w.start <= w.end && w.end <= w.right && w.right < frames)
            })
            || windows.windows(2).any(|pair| pair[0].right >= pair[1].left)
        {
            return Err("Invalid detector handoff timeline".into());
        }
        Ok(Self {
            pending: windows.into(),
            observations: Vec::new(),
            next_frame: 0,
            frames,
        })
    }

    pub(crate) fn push(
        &mut self,
        first: u64,
        detections: &[T],
        frame_index: impl Fn(&T) -> u64,
    ) -> Result<Vec<CompletedRally<T>>, String> {
        let end = first
            .checked_add(DETECTOR_WINDOW)
            .ok_or("Detector window overflow")?;
        if first != self.next_frame
            || end > self.frames
            || detections
                .iter()
                .any(|row| !(first..end).contains(&frame_index(row)))
        {
            return Err("Detector handoff frame ordering is invalid".into());
        }
        let mut ready = Vec::new();
        while let Some(window) = self.pending.front().copied() {
            if window.left >= end {
                break;
            }
            let matching = detections
                .iter()
                .filter(|row| (window.left..=window.right).contains(&frame_index(row)));
            for row in matching {
                if self.observations.len() == MAX_RALLY_OBSERVATIONS {
                    return Err("Detector rally handoff exceeds bounded observation limit".into());
                }
                self.observations.push(row.clone());
            }
            if window.right >= end {
                break;
            }
            self.pending.pop_front();
            ready.push(CompletedRally {
                window,
                observations: std::mem::take(&mut self.observations),
            });
        }
        self.next_frame = end;
        Ok(ready)
    }

    pub(crate) fn finish(self) -> Result<(), String> {
        if self.next_frame != self.frames
            || !self.pending.is_empty()
            || !self.observations.is_empty()
        {
            return Err("Detector handoff ended before all rally windows completed".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roster_windows_share_context_without_overlapping() {
        assert_eq!(
            rally_windows(&[(10, 100), (120, 150)], &[(20, 40), (60, 90), (130, 140)]).unwrap(),
            vec![
                RallyWindow {
                    left: 10,
                    right: 50,
                    start: 20,
                    end: 40
                },
                RallyWindow {
                    left: 51,
                    right: 100,
                    start: 60,
                    end: 90
                },
                RallyWindow {
                    left: 120,
                    right: 150,
                    start: 130,
                    end: 140
                },
            ]
        );
    }

    #[test]
    fn emits_completed_intervals_before_detector_end_and_keeps_absolute_frames() {
        let windows = rally_windows(&[(1, 16), (21, 35)], &[(3, 12), (23, 33)]).unwrap();
        let mut handoff = RallyHandoff::new(windows, 36).unwrap();
        assert!(handoff
            .push(0, &[1, 4, 8], |frame| *frame)
            .unwrap()
            .is_empty());
        let ready = handoff.push(9, &[10, 16, 17], |frame| *frame).unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(
            ready[0].window,
            RallyWindow {
                left: 1,
                right: 16,
                start: 3,
                end: 12
            }
        );
        assert_eq!(ready[0].observations, vec![1, 4, 8, 10, 16]);
        assert!(handoff
            .push(18, &[20, 21, 26], |frame| *frame)
            .unwrap()
            .is_empty());
        let ready = handoff.push(27, &[30, 35], |frame| *frame).unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].observations, vec![21, 26, 30, 35]);
        handoff.finish().unwrap();
    }

    #[test]
    fn empty_detection_interval_is_emitted_and_gaps_are_rejected() {
        let windows = rally_windows(&[(0, 8), (10, 17)], &[(2, 6), (11, 15)]).unwrap();
        let mut handoff = RallyHandoff::<u64>::new(windows, 18).unwrap();
        assert!(handoff.push(9, &[], |frame| *frame).is_err());
        let ready = handoff.push(0, &[], |frame| *frame).unwrap();
        assert_eq!(ready.len(), 1);
        assert!(ready[0].observations.is_empty());
        assert!(handoff.finish().is_err());
    }

    #[test]
    fn batch_crossing_two_rosters_does_not_mix_observations() {
        let windows = rally_windows(&[(0, 17)], &[(1, 2), (5, 7)]).unwrap();
        let mut handoff = RallyHandoff::new(windows, 18).unwrap();
        let ready = handoff.push(0, &[0, 3, 4, 8], |frame| *frame).unwrap();
        assert_eq!(ready[0].observations, vec![0, 3]);
        let ready = handoff.push(9, &[9, 17], |frame| *frame).unwrap();
        assert_eq!(ready[0].observations, vec![4, 8, 9, 17]);
        handoff.finish().unwrap();
    }
}
