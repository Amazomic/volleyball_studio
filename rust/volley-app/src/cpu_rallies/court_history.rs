//! Bounded, append-only court-context evidence ring for accepted CPU rallies.
//! The authoritative Touch writer and source-frame reader remain unchanged.
use std::collections::VecDeque;
use volley_core::{
    rally_contact_context::{self, BallSample, ContactEvidence, RallyContext},
    rally_reconstruction::RallyProposal,
};

const HISTORY_FRAMES: u64 = 18_000; // 10 minutes at prepared 30 FPS.
const MAX_CONTEXT_CONTACTS: usize = 512;

#[derive(Default)]
pub(super) struct CourtHistory {
    balls: VecDeque<BallSample>,
    contacts: VecDeque<ContactEvidence>,
    // Only needed when a rally lasts so long that its beginning was discarded.
    truncated_before: u64,
}

impl CourtHistory {
    pub(super) fn push_ball(&mut self, sample: BallSample) {
        self.balls.push_back(sample);
    }

    pub(super) fn push_touch(&mut self, item: ContactEvidence) {
        self.contacts.push_back(item);
    }

    pub(super) fn trim(&mut self, frame: u64) {
        let before = frame.saturating_sub(HISTORY_FRAMES);
        while self
            .balls
            .front()
            .is_some_and(|sample| sample.frame < before)
        {
            self.balls.pop_front();
            self.truncated_before = before;
        }
        while self
            .contacts
            .front()
            .is_some_and(|item| item.frame < before)
        {
            self.contacts.pop_front();
            self.truncated_before = before;
        }
    }

    /// Compute possession from the exact accepted rally interval, not any
    /// player who happened to appear in the same long source video.
    pub(super) fn for_rally(&self, proposal: &RallyProposal) -> Option<RallyContext> {
        let start = proposal.detected_start_frame;
        let end = proposal.detected_end_frame;
        if start < self.truncated_before || end < start {
            return None;
        }
        let mut events = Vec::new();
        for contact in self
            .contacts
            .iter()
            .filter(|c| (start..=end).contains(&c.frame))
        {
            if contact.track_id.is_none() {
                continue;
            }
            // The same player can toss and then strike a serve. CPU Touch
            // already deduplicates adjacent motion peaks; never erase a later hit.
            if events.len() == MAX_CONTEXT_CONTACTS {
                // Never publish a misleading partial analysis.
                return None;
            }
            events.push(*contact);
        }
        if !events.iter().any(|c| c.side.is_some()) {
            return None;
        }
        let ball = self
            .balls
            .iter()
            .filter(|sample| (start..=end).contains(&sample.frame))
            .copied()
            .collect::<Vec<_>>();
        Some(rally_contact_context::analyze(&events, &ball))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volley_core::{action_prediction::ActionKind, court_geometry::CourtSide};

    #[test]
    fn preserves_same_server_toss_and_strike_for_rally_context() {
        let mut history = CourtHistory::default();
        for (frame, speed, residual) in [(100, 8.0, 7.0), (130, 16.0, 18.0)] {
            history.push_touch(ContactEvidence {
                frame,
                track_id: Some(7),
                side: Some(CourtSide::Near),
                in_serve_zone: true,
                kind: Some(ActionKind::Attack),
                ball_y_px: 250.0,
                speed_before_px_frame: 2.0,
                speed_after_px_frame: speed,
                velocity_residual_px_frame: residual,
            });
        }
        let proposal = RallyProposal {
            detected_start_frame: 90,
            detected_end_frame: 170,
            duration_sec: 2.7,
            ball_observations: 60,
            ball_coverage: 0.74,
            motion_supported_touch_proposals: 2,
            first_touch_frame: 100,
            last_touch_frame: 130,
            max_touch_probability: 0.9,
            provisional_track_ids: vec![7],
            unidentified_contacts: 0,
            classification: "provisional_rally",
            method: "test",
            authoritative_rally: false,
            athlete_identities_verified: false,
            serve_verified: false,
            suggested_context_before_frames: 45,
            suggested_context_after_frames: 45,
            end_reason: "test",
        };
        let context = history.for_rally(&proposal).expect("court evidence");
        assert_eq!(context.events.len(), 2);
        assert_eq!(context.serve_frame, Some(130));
        assert!(context.events[0].ignored_serve_toss);
    }
}
