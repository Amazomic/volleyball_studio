#[derive(Clone, Copy, Debug)]
pub struct LeadingBallSettings {
    pub search_radius_width: f64,
    pub uncertainty_per_frame_width: f64,
    pub max_search_radius_width: f64,
    pub max_prediction_frames: u64,
}

impl Default for LeadingBallSettings {
    fn default() -> Self {
        Self {
            search_radius_width: 0.06,
            uncertainty_per_frame_width: 0.008,
            max_search_radius_width: 0.20,
            max_prediction_frames: 27,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeadingBallObservation {
    pub candidate_count: usize,
    pub selected: Option<usize>,
    pub track_id: Option<u32>,
    pub acquired_frames: Option<[u64; 3]>,
}

#[derive(Clone, Copy)]
struct Anchor {
    frame: u64,
    point: [f64; 2],
}

#[derive(Clone, Copy)]
struct Acquisition {
    frames: [u64; 3],
    count: usize,
    last: Anchor,
    velocity: [f64; 2],
}

pub struct LeadingBallTracker {
    settings: LeadingBallSettings,
    size: [f64; 2],
    last: Option<Anchor>,
    velocity: [f64; 2],
    velocity_known: bool,
    track_id: u32,
    pending: Vec<Acquisition>,
}

fn valid(point: [f64; 2]) -> bool {
    point
        .iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn distance_to_path(point: [f64; 2], begin: [f64; 2], end: [f64; 2]) -> f64 {
    let delta = [end[0] - begin[0], end[1] - begin[1]];
    let squared_length = delta[0] * delta[0] + delta[1] * delta[1];
    let t = if squared_length > 0.0 {
        (((point[0] - begin[0]) * delta[0] + (point[1] - begin[1]) * delta[1]) / squared_length)
            .clamp(0.0, 1.0)
    } else {
        0.0
    };
    distance(point, [begin[0] + t * delta[0], begin[1] + t * delta[1]])
}

impl LeadingBallTracker {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            settings: LeadingBallSettings::default(),
            size: [f64::from(width), f64::from(height)],
            last: None,
            velocity: [0.0; 2],
            velocity_known: false,
            track_id: 0,
            pending: Vec::new(),
        }
    }

    fn pixels(&self, point: [f64; 2]) -> [f64; 2] {
        [point[0] * self.size[0], point[1] * self.size[1]]
    }

    fn predicted(&self, frame: u64) -> Option<[f64; 2]> {
        let last = self.last?;
        let dt = frame
            .checked_sub(last.frame)?
            .min(self.settings.max_prediction_frames) as f64;
        Some([
            last.point[0] + self.velocity[0] * dt,
            last.point[1] + self.velocity[1] * dt,
        ])
    }

    pub fn allows(&self, frame: u64, point: [f64; 2]) -> bool {
        let Some(last) = self.last else { return false };
        let Some(dt) = frame.checked_sub(last.frame) else {
            return false;
        };
        if dt == 0 || !valid(point) {
            return false;
        }
        let Some(predicted) = self.predicted(frame) else {
            return false;
        };
        if !self.velocity_known {
            return distance(last.point, self.pixels(point))
                <= self.size[0] * 0.12 * dt.min(3) as f64;
        }
        let radius = (self.settings.search_radius_width
            + self.settings.uncertainty_per_frame_width * (dt - 1) as f64)
            .min(self.settings.max_search_radius_width)
            * self.size[0];
        distance_to_path(self.pixels(point), last.point, predicted) <= radius
            && distance(last.point, self.pixels(point))
                <= self.size[0] * (0.12 * dt as f64 + self.settings.search_radius_width)
    }

    pub fn observe(&mut self, frame: u64, candidates: &[[f64; 2]]) -> LeadingBallObservation {
        let candidate_count = candidates.iter().filter(|p| valid(**p)).count();
        let mut result = LeadingBallObservation {
            candidate_count,
            selected: None,
            track_id: None,
            acquired_frames: None,
        };
        if self.last.is_some_and(|last| frame <= last.frame) {
            return result;
        }
        if self.last.is_none() {
            if candidate_count == 0 {
                self.pending.retain(|pending| {
                    frame > pending.last.frame && frame - pending.last.frame <= 3
                });
                return result;
            }
            let mut next = Vec::new();
            for (index, candidate) in candidates
                .iter()
                .enumerate()
                .filter(|(_, point)| valid(**point))
                .take(128)
            {
                let point = self.pixels(*candidate);
                let best = self
                    .pending
                    .iter()
                    .filter(|pending| frame > pending.last.frame && frame - pending.last.frame <= 3)
                    .filter_map(|pending| {
                        let dt = (frame - pending.last.frame) as f64;
                        let velocity = [
                            (point[0] - pending.last.point[0]) / dt,
                            (point[1] - pending.last.point[1]) / dt,
                        ];
                        let speed = velocity[0].hypot(velocity[1]);
                        if speed < self.size[0] * 0.0004 || speed > self.size[0] * 0.12 {
                            return None;
                        }
                        let prior_speed = pending.velocity[0].hypot(pending.velocity[1]);
                        if pending.count > 1
                            && ((velocity[0] * pending.velocity[0]
                                + velocity[1] * pending.velocity[1])
                                < 0.85 * speed * prior_speed
                                || speed < prior_speed * 0.25
                                || speed > prior_speed * 4.0)
                        {
                            return None;
                        }
                        Some((*pending, velocity, distance(point, pending.last.point)))
                    })
                    .min_by(|a, b| a.2.total_cmp(&b.2));
                let pending = if let Some((pending, velocity, _)) = best {
                    let mut updated = pending;
                    updated.frames[updated.count] = frame;
                    updated.count += 1;
                    updated.velocity = velocity;
                    updated.last = Anchor { frame, point };
                    updated
                } else {
                    Acquisition {
                        frames: [frame, 0, 0],
                        count: 1,
                        last: Anchor { frame, point },
                        velocity: [0.0; 2],
                    }
                };
                if pending.count == 3 {
                    self.last = Some(pending.last);
                    self.velocity = pending.velocity;
                    self.velocity_known = true;
                    self.track_id = self.track_id.max(1);
                    self.pending.clear();
                    result.selected = Some(index);
                    result.track_id = Some(self.track_id);
                    result.acquired_frames = Some(pending.frames);
                    return result;
                }
                next.push(pending);
            }
            self.pending = next;
            return result;
        }
        let Some(predicted) = self.predicted(frame) else {
            return result;
        };
        let selected = candidates
            .iter()
            .enumerate()
            .filter(|(_, p)| self.allows(frame, **p))
            .min_by(|(_, a), (_, b)| {
                distance(self.pixels(**a), predicted)
                    .total_cmp(&distance(self.pixels(**b), predicted))
            })
            .map(|(index, _)| index);
        if let Some(index) = selected {
            self.accept_anchor(frame, candidates[index]);
            result.selected = Some(index);
        }
        result.track_id = self.last.map(|_| self.track_id);
        result
    }

    pub fn begin_rally(&mut self) {
        self.last = None;
        self.velocity = [0.0; 2];
        self.velocity_known = false;
        self.track_id = self.track_id.saturating_add(1).max(1);
        self.pending.clear();
    }

    pub(super) fn accept_anchor(&mut self, frame: u64, point: [f64; 2]) {
        let point = self.pixels(point);
        if let Some(last) = self.last.filter(|last| frame > last.frame) {
            let dt = (frame - last.frame) as f64;
            self.velocity = [
                (point[0] - last.point[0]) / dt,
                (point[1] - last.point[1]) / dt,
            ];
            self.velocity_known = true;
        }
        self.last = Some(Anchor { frame, point });
        self.track_id = self.track_id.max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chooses_leading_candidate_instead_of_first_remote_candidate() {
        let mut tracker = LeadingBallTracker::new(1280, 720);
        tracker.observe(0, &[[0.3, 0.4]]);
        tracker.observe(1, &[[0.31, 0.4]]);
        let observation = tracker.observe(2, &[[0.9, 0.6], [0.32, 0.4]]);
        assert_eq!(
            observation,
            LeadingBallObservation {
                candidate_count: 2,
                selected: Some(1),
                track_id: Some(1),
                acquired_frames: Some([0, 1, 2]),
            }
        );
    }

    #[test]
    fn first_velocity_can_be_measured_after_an_initial_dropout() {
        let mut tracker = LeadingBallTracker::new(1280, 720);
        tracker.observe(0, &[[0.2, 0.4]]);
        tracker.observe(1, &[]);
        tracker.observe(2, &[]);
        assert_eq!(tracker.observe(3, &[[0.5, 0.4]]).selected, None);
        assert_eq!(tracker.observe(4, &[[0.6, 0.4]]).selected, Some(0));
    }

    #[test]
    fn long_loss_cannot_select_a_remote_ball_until_next_rally() {
        let mut tracker = LeadingBallTracker::new(1280, 720);
        tracker.observe(0, &[[0.2, 0.4]]);
        tracker.observe(1, &[[0.21, 0.4]]);
        tracker.observe(2, &[[0.22, 0.4]]);
        for frame in 92..130 {
            assert_eq!(tracker.observe(frame, &[[0.9, 0.6]]).selected, None);
        }
        tracker.begin_rally();
        assert_eq!(tracker.observe(130, &[[0.9, 0.6]]).selected, None);
        assert_eq!(tracker.observe(131, &[[0.89, 0.6]]).selected, None);
        assert_eq!(
            tracker.observe(132, &[[0.88, 0.6]]),
            LeadingBallObservation {
                candidate_count: 1,
                selected: Some(0),
                track_id: Some(2),
                acquired_frames: Some([130, 131, 132]),
            }
        );
    }

    #[test]
    fn stationary_remote_target_never_reacquires_a_lost_track() {
        let mut tracker = LeadingBallTracker::new(1280, 720);
        tracker.observe(0, &[[0.2, 0.4]]);
        for frame in 1..180 {
            assert_eq!(tracker.observe(frame, &[[0.92, 0.6]]).selected, None);
        }
    }

    #[test]
    fn invalid_coordinates_are_not_counted_or_accepted() {
        let mut tracker = LeadingBallTracker::new(1280, 720);
        assert_eq!(
            tracker.observe(0, &[[f64::NAN, 0.4], [1.1, 0.3]]),
            LeadingBallObservation {
                candidate_count: 0,
                selected: None,
                track_id: None,
                acquired_frames: None,
            }
        );
    }

    #[test]
    fn real_reversal_inside_last_ball_area_does_not_lose_the_leader() {
        let mut tracker = LeadingBallTracker::new(1280, 720);
        tracker.observe(0, &[[0.4, 0.5]]);
        tracker.observe(1, &[[0.4, 0.4]]);
        tracker.observe(2, &[[0.4, 0.3]]);
        assert_eq!(tracker.observe(3, &[[0.4, 0.4]]).selected, Some(0));
    }
}
