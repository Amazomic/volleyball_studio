//! Four-point, court-plane projective geometry, ported from Python
//! court_transformer.py and player_contacts.py.
//! Input points are image-normalized; output is meters centered at the net.
//! The point order is the same as the Python homography: near end-line
//! sideline corner, opposite end-line on that sideline, then the remaining
//! two corners in clockwise/counter-clockwise perimeter order.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const COURT_LENGTH_M: f64 = 18.0;
pub const COURT_WIDTH_M: f64 = 9.0;
pub const ON_COURT_LENGTH_MARGIN_M: f64 = 6.0; // Servers may start well behind an end line.
pub const ON_COURT_WIDTH_MARGIN_M: f64 = 1.5;
pub const SERVE_ZONE_MARGIN_M: f64 = 0.35; // Only line-local tolerance inside the end line.
pub const BLOCK_NEAR_NET_M: f64 = 1.6;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CourtContactEvidence {
    pub x_m: f64,
    pub y_m: f64,
    /// Negative length-axis = near, positive = far, as in Python PlayerStore.
    pub side: CourtSide,
    pub near_net: bool,
    pub in_serve_zone: bool,
}
impl CourtContactEvidence {
    #[must_use]
    pub fn valid(self) -> bool {
        self.x_m.is_finite()
            && self.y_m.is_finite()
            && self.x_m.abs() <= COURT_LENGTH_M / 2.0 + ON_COURT_LENGTH_MARGIN_M + 1e-6
            && self.y_m.abs() <= COURT_WIDTH_M / 2.0 + ON_COURT_WIDTH_MARGIN_M + 1e-6
            // Side is median-stabilized across each provisional track.
            && self.near_net == (self.x_m.abs() <= BLOCK_NEAR_NET_M)
            // Historical CPU Touch snapshots used 1.2 m of inside-line
            // tolerance. Keep them readable; new estimates use 0.35 m.
            && (self.in_serve_zone
                == (self.x_m.abs() >= COURT_LENGTH_M / 2.0 - SERVE_ZONE_MARGIN_M)
                || (self.in_serve_zone
                    && self.x_m.abs() >= COURT_LENGTH_M / 2.0 - 1.2))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CourtSide {
    Near,
    Far,
}

#[derive(Clone, Debug)]
pub struct CourtTransform {
    matrix: [f64; 9],
    net: Option<[[f64; 2]; 2]>,
}

pub fn valid_net(net: Option<[[f64; 2]; 2]>) -> bool {
    net.is_none_or(|[a, b]| {
        a.iter()
            .chain(b.iter())
            .all(|v| v.is_finite() && (-0.5..=1.5).contains(v))
            && (a[0] - b[0]).hypot(a[1] - b[1]) > 0.01
    })
}

pub fn observed_net_contact(net: Option<[[f64; 2]; 2]>, [x, y]: [f64; 2]) -> bool {
    net.is_none_or(|[a, b]| {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let t = ((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy);
        (-0.1..=1.1).contains(&t)
            && ((x - a[0]) * dy - (y - a[1]) * dx).abs() / dx.hypot(dy) <= 0.12
    })
}

/// Perspective-correct division into the two 9 x 9 m court halves and
/// the *visible* net surface. No collision is implied by image overlap.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CourtZones {
    pub near_half: [[f64; 2]; 4],
    pub far_half: [[f64; 2]; 4],
    pub ground_divider: [[f64; 2]; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub net_surface: Option<[[f64; 2]; 4]>,
}

impl CourtTransform {
    /// Inverse homography: map a point on the physical floor to image space.
    /// The 2D camera projection is not an estimate of airborne ball height.
    pub fn floor_to_image(&self, [x, y]: [f64; 2]) -> Option<[f64; 2]> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let m = &self.matrix;
        let adj = [
            m[4] * m[8] - m[5] * m[7],
            m[2] * m[7] - m[1] * m[8],
            m[1] * m[5] - m[2] * m[4],
            m[5] * m[6] - m[3] * m[8],
            m[0] * m[8] - m[2] * m[6],
            m[2] * m[3] - m[0] * m[5],
            m[3] * m[7] - m[4] * m[6],
            m[1] * m[6] - m[0] * m[7],
            m[0] * m[4] - m[1] * m[3],
        ];
        let z = adj[6] * x + adj[7] * y + adj[8];
        if !z.is_finite() || z.abs() < 1e-10 {
            return None;
        }
        let point = [
            (adj[0] * x + adj[1] * y + adj[2]) / z,
            (adj[3] * x + adj[4] * y + adj[5]) / z,
        ];
        point.iter().all(|n| n.is_finite()).then_some(point)
    }

    pub(crate) fn flight_projection(&self, net_height: f64) -> Option<[[f64; 4]; 3]> {
        let [a, b, _, _] = self.zones()?.net_surface?;
        let m = &self.matrix;
        let mut h = [
            m[4] * m[8] - m[5] * m[7],
            m[2] * m[7] - m[1] * m[8],
            m[1] * m[5] - m[2] * m[4],
            m[5] * m[6] - m[3] * m[8],
            m[0] * m[8] - m[2] * m[6],
            m[2] * m[3] - m[0] * m[5],
            m[3] * m[7] - m[4] * m[6],
            m[1] * m[6] - m[0] * m[7],
            m[0] * m[4] - m[1] * m[3],
        ];
        let scale = h[8];
        if scale.abs() < 1e-10 || !net_height.is_finite() || net_height <= 0.0 {
            return None;
        }
        for item in &mut h {
            *item /= scale;
        }
        let ground = |y: f64| [h[1] * y + h[2], h[4] * y + h[5], h[7] * y + h[8]];
        let ga = ground(-COURT_WIDTH_M / 2.0);
        let gb = ground(COURT_WIDTH_M / 2.0);
        let rhs = |point: [f64; 2], g: [f64; 3]| {
            [
                (point[0] * g[2] - g[0]) / net_height,
                (point[1] * g[2] - g[1]) / net_height,
            ]
        };
        let ra = rhs(a, ga);
        let rb = rhs(b, gb);
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let denominator = dx * dx + dy * dy;
        if denominator < 1e-6 {
            return None;
        }
        let z2 = (dx * (rb[0] - ra[0]) + dy * (rb[1] - ra[1])) / denominator;
        let z0 = (ra[0] + rb[0] + (a[0] + b[0]) * z2) / 2.0;
        let z1 = (ra[1] + rb[1] + (a[1] + b[1]) * z2) / 2.0;
        let projection = [
            [h[0], h[1], z0, h[2]],
            [h[3], h[4], z1, h[5]],
            [h[6], h[7], z2, h[8]],
        ];
        if !projection.iter().flatten().all(|v| v.is_finite()) {
            return None;
        }
        for (point, g) in [(a, ga), (b, gb)] {
            let depth = g[2] + z2 * net_height;
            if depth <= 0.0
                || ((g[0] + z0 * net_height) / depth - point[0])
                    .hypot((g[1] + z1 * net_height) / depth - point[1])
                    > 0.02
            {
                return None;
            }
        }
        Some(projection)
    }

    pub fn zones(&self) -> Option<CourtZones> {
        let left = self.floor_to_image([0.0, -COURT_WIDTH_M / 2.0])?;
        let right = self.floor_to_image([0.0, COURT_WIDTH_M / 2.0])?;
        let c0 = self.floor_to_image([-COURT_LENGTH_M / 2.0, -COURT_WIDTH_M / 2.0])?;
        let c1 = self.floor_to_image([COURT_LENGTH_M / 2.0, -COURT_WIDTH_M / 2.0])?;
        let c2 = self.floor_to_image([COURT_LENGTH_M / 2.0, COURT_WIDTH_M / 2.0])?;
        let c3 = self.floor_to_image([-COURT_LENGTH_M / 2.0, COURT_WIDTH_M / 2.0])?;
        let net_surface = self.net.map(|[a, b]| {
            let (a, b) = if (a[0] - left[0]).hypot(a[1] - left[1])
                + (b[0] - right[0]).hypot(b[1] - right[1])
                <= (b[0] - left[0]).hypot(b[1] - left[1]) + (a[0] - right[0]).hypot(a[1] - right[1])
            {
                (a, b)
            } else {
                (b, a)
            };
            [a, b, right, left]
        });
        Some(CourtZones {
            near_half: [c0, left, right, c3],
            far_half: [left, c1, c2, right],
            ground_divider: [left, right],
            net_surface,
        })
    }

    /// A ball can be classified only after independent motion evidence of
    /// a *floor* bounce. A flying ball must not be projected onto the floor.
    pub fn tentative_landing(&self, floor_point: [f64; 2]) -> Option<&'static str> {
        let [x, y] = self.to_court(floor_point)?;
        const UNCERTAINTY_M: f64 = 0.15;
        let dx = x.abs() - COURT_LENGTH_M / 2.0;
        let dy = y.abs() - COURT_WIDTH_M / 2.0;
        Some(if dx > UNCERTAINTY_M || dy > UNCERTAINTY_M {
            "out"
        } else if dx >= -UNCERTAINTY_M || dy >= -UNCERTAINTY_M {
            "line_uncertain"
        } else if x < 0.0 {
            "near_half"
        } else {
            "far_half"
        })
    }
    pub fn with_net(mut self, net: Option<[[f64; 2]; 2]>) -> Result<Self, String> {
        if !valid_net(net) {
            return Err("Invalid observed net line".into());
        }
        self.net = net;
        Ok(self)
    }

    pub fn supports_net_contact(&self, point: [f64; 2]) -> bool {
        observed_net_contact(self.net, point)
    }

    /// Map four ordered normalized pixels to the 18 x 9 meter playing plane.
    /// The same four targets are used by Python CourtTransformer.
    ///
    /// Reject self-intersections, collinear corners, badly conditioned
    /// quadrilaterals and non-finite inputs rather than using a fake identity
    /// transform. Coordinates outside the image are allowed to +/-0.5.
    pub fn new(corners: [[f64; 2]; 4]) -> Result<Self, String> {
        if corners
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || !(-0.5..=1.5).contains(v))
        {
            return Err("Court points must be finite and within [-0.5, 1.5]".into());
        }
        let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
            (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        };
        let turns = std::array::from_fn::<_, 4, _>(|i| {
            cross(corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4])
        });
        if turns.iter().any(|t| t.abs() < 1e-5)
            || turns.iter().any(|t| t.signum() != turns[0].signum())
        {
            return Err("Court corners must form an ordered convex quadrilateral".into());
        }
        let area = (0..4)
            .map(|i| {
                let a = corners[i];
                let b = corners[(i + 1) % 4];
                a[0] * b[1] - b[0] * a[1]
            })
            .sum::<f64>()
            .abs()
            * 0.5;
        if area < 1e-3 {
            return Err("Court area is too small for reliable calibration".into());
        }
        let targets = [
            [-COURT_LENGTH_M / 2.0, -COURT_WIDTH_M / 2.0],
            [COURT_LENGTH_M / 2.0, -COURT_WIDTH_M / 2.0],
            [COURT_LENGTH_M / 2.0, COURT_WIDTH_M / 2.0],
            [-COURT_LENGTH_M / 2.0, COURT_WIDTH_M / 2.0],
        ];
        let mut rows = [[0.0; 9]; 8];
        for (i, (&[x, y], &[cx, cy])) in corners.iter().zip(targets.iter()).enumerate() {
            rows[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * cx, -y * cx, cx];
            rows[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -x * cy, -y * cy, cy];
        }
        // Partial-pivot Gaussian elimination of the 8-DOF homography.
        for pivot in 0..8 {
            let best = (pivot..8)
                .max_by(|&a, &b| rows[a][pivot].abs().total_cmp(&rows[b][pivot].abs()))
                .ok_or("Court homography has no pivot")?;
            if rows[best][pivot].abs() < 1e-10 {
                return Err("Court homography is singular".into());
            }
            rows.swap(pivot, best);
            let scale = rows[pivot][pivot];
            for value in &mut rows[pivot][pivot..] {
                *value /= scale;
            }
            // Copy the pivot row: simultaneous mutable/immutable borrows
            // of different rows are unnecessary for only eight equations.
            let normalized = rows[pivot];
            for (other, row) in rows.iter_mut().enumerate() {
                if other == pivot {
                    continue;
                }
                let factor = row[pivot];
                for (target, source) in row[pivot..].iter_mut().zip(normalized[pivot..].iter()) {
                    *target -= factor * source;
                }
            }
        }
        let matrix = [
            rows[0][8], rows[1][8], rows[2][8], rows[3][8], rows[4][8], rows[5][8], rows[6][8],
            rows[7][8], 1.0,
        ];
        let transform = Self { matrix, net: None };
        for (point, target) in corners.iter().zip(targets) {
            let got = transform
                .to_court(*point)
                .ok_or("Invalid court homography")?;
            if (got[0] - target[0]).abs() > 1e-5 || (got[1] - target[1]).abs() > 1e-5 {
                return Err("Court homography does not reproduce its calibration".into());
            }
        }
        Ok(transform)
    }

    /// Project a normalized image point onto the ground plane.
    #[must_use]
    pub fn to_court(&self, [x, y]: [f64; 2]) -> Option<[f64; 2]> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let h = &self.matrix;
        let w = h[6] * x + h[7] * y + h[8];
        if !w.is_finite() || w.abs() < 1e-10 {
            return None;
        }
        let point = [
            (h[0] * x + h[1] * y + h[2]) / w,
            (h[3] * x + h[4] * y + h[5]) / w,
        ];
        point.iter().all(|v| v.is_finite()).then_some(point)
    }

    /// The Python on-court corridor includes substitutes slightly beyond the
    /// lines, but excludes spectators and players far outside the game.
    #[must_use]
    pub fn evidence_for_feet(&self, feet_norm: [f64; 2]) -> Option<CourtContactEvidence> {
        let [x_m, y_m] = self.to_court(feet_norm)?;
        let evidence = CourtContactEvidence {
            x_m,
            y_m,
            side: if x_m < 0.0 {
                CourtSide::Near
            } else {
                CourtSide::Far
            },
            near_net: x_m.abs() <= BLOCK_NEAR_NET_M,
            in_serve_zone: x_m.abs() >= COURT_LENGTH_M / 2.0 - SERVE_ZONE_MARGIN_M,
        };
        evidence.valid().then_some(evidence)
    }

    /// Player foot anchor, not box center; jumps may shift the instantaneous
    /// estimate, so callers should use track context for side stability.
    #[must_use]
    pub fn player_evidence(&self, bbox: [f64; 4]) -> Option<CourtContactEvidence> {
        self.evidence_for_feet([(bbox[0] + bbox[2]) * 0.5, bbox[3]])
    }
}

#[derive(Default)]
pub struct TeamSideVotes {
    near: u64,
    far: u64,
    observed: u64,
    clear_near: u64,
    clear_far: u64,
}

impl TeamSideVotes {
    pub fn observe(&mut self, court: &CourtTransform, bbox: [f64; 4]) {
        if let Some(evidence) = court.player_evidence(bbox) {
            self.observed += 1;
            if evidence.x_m.abs() >= 0.25 {
                match evidence.side {
                    CourtSide::Near => self.near += 1,
                    CourtSide::Far => self.far += 1,
                }
            }
            if evidence.x_m.abs() >= 0.75 {
                match evidence.side {
                    CourtSide::Near => self.clear_near += 1,
                    CourtSide::Far => self.clear_far += 1,
                }
            }
        }
    }

    pub fn side(&self) -> Option<CourtSide> {
        let total = self.near + self.far;
        if total < 3 {
            return None;
        }
        if self.near * 3 >= total * 2 {
            Some(CourtSide::Near)
        } else if self.far * 3 >= total * 2 {
            Some(CourtSide::Far)
        } else {
            let clear_total = self.clear_near + self.clear_far;
            if clear_total < 3 || clear_total * 10 < self.observed {
                None
            } else if self.clear_near * 3 >= clear_total * 2 {
                Some(CourtSide::Near)
            } else if self.clear_far * 3 >= clear_total * 2 {
                Some(CourtSide::Far)
            } else {
                None
            }
        }
    }
}

pub fn resolve_team_sides(votes: BTreeMap<u64, TeamSideVotes>) -> BTreeMap<u64, CourtSide> {
    let mut sides = BTreeMap::new();
    let mut ambiguous = Vec::new();
    let mut fixed_near = 0usize;
    let mut fixed_far = 0usize;
    for (id, vote) in votes {
        match vote.side() {
            Some(side) => {
                sides.insert(id, side);
                match side {
                    CourtSide::Near => fixed_near += 1,
                    CourtSide::Far => fixed_far += 1,
                }
            }
            None if vote.observed >= 3 => {
                let preference = (vote.near as f64 - vote.far as f64) / vote.observed as f64;
                ambiguous.push((id, preference));
            }
            None => {}
        }
    }
    ambiguous.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut best_count = 0;
    let mut best_imbalance = fixed_near.abs_diff(fixed_far + ambiguous.len());
    let mut best_score = 0.0;
    let mut score = 0.0;
    for (index, (_, preference)) in ambiguous.iter().enumerate() {
        let count = index + 1;
        score += preference;
        let imbalance = (fixed_near + count).abs_diff(fixed_far + ambiguous.len() - count);
        if imbalance < best_imbalance || (imbalance == best_imbalance && score > best_score) {
            best_count = count;
            best_imbalance = imbalance;
            best_score = score;
        }
    }
    for (index, (id, _)) in ambiguous.into_iter().enumerate() {
        sides.insert(
            id,
            if index < best_count {
                CourtSide::Near
            } else {
                CourtSide::Far
            },
        );
    }
    sides
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUAD: [[f64; 2]; 4] = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]];
    fn vote(near: u64, far: u64, observed: u64) -> TeamSideVotes {
        TeamSideVotes {
            near,
            far,
            observed,
            ..Default::default()
        }
    }

    #[test]
    fn disputed_net_side_prefers_repeated_clear_geometry_over_roster_balance() {
        let court = CourtTransform::new(QUAD).unwrap();
        for sign in [1.0, -1.0] {
            let mut disputed = TeamSideVotes::default();
            for (x, count) in [(-0.65, 126), (0.50, 71), (0.85, 39), (0.0, 135)] {
                let [x, y] = court.floor_to_image([sign * x, 0.0]).unwrap();
                for _ in 0..count {
                    disputed.observe(&court, [x - 0.02, y - 0.1, x + 0.02, y]);
                }
            }
            let expected = if sign > 0.0 {
                CourtSide::Far
            } else {
                CourtSide::Near
            };
            assert_eq!(disputed.side(), Some(expected));
            let sides = resolve_team_sides(BTreeMap::from([
                (1, vote(10, 0, 10)),
                (2, vote(10, 0, 10)),
                (3, vote(10, 0, 10)),
                (4, vote(0, 10, 10)),
                (5, vote(0, 10, 10)),
                (6, vote(0, 10, 10)),
                (14, disputed),
            ]));
            assert_eq!(sides[&14], expected);
        }
    }

    #[test]
    fn rare_clear_frames_do_not_override_disputed_net_evidence() {
        let court = CourtTransform::new(QUAD).unwrap();
        let mut disputed = TeamSideVotes::default();
        for (x, count) in [(-0.5, 50), (0.5, 50), (0.85, 3)] {
            let [x, y] = court.floor_to_image([x, 0.0]).unwrap();
            for _ in 0..count {
                disputed.observe(&court, [x - 0.02, y - 0.1, x + 0.02, y]);
            }
        }
        assert_eq!(disputed.side(), None);
    }

    #[test]
    fn team_balance_only_changes_ambiguous_players() {
        let sides = resolve_team_sides(BTreeMap::from([
            (1, vote(5, 0, 5)),
            (2, vote(5, 0, 5)),
            (3, vote(5, 0, 5)),
            (4, vote(0, 5, 5)),
            (5, vote(0, 5, 5)),
            (6, vote(3, 2, 5)),
            (7, vote(1, 0, 1)),
        ]));
        assert_eq!(sides.len(), 6);
        assert_eq!(sides[&6], CourtSide::Far);
        assert_eq!(sides[&3], CourtSide::Near);
        assert!(!sides.contains_key(&7));

        let sides = resolve_team_sides(BTreeMap::from([
            (1, vote(5, 0, 5)),
            (2, vote(5, 0, 5)),
            (3, vote(5, 0, 5)),
            (4, vote(5, 0, 5)),
            (5, vote(0, 5, 5)),
            (6, vote(2, 2, 5)),
        ]));
        assert_eq!(sides.values().filter(|&&s| s == CourtSide::Near).count(), 4);
        assert_eq!(sides[&6], CourtSide::Far);
    }

    #[test]
    fn team_balance_resolves_multiple_disputed_tracks_by_geometry() {
        let sides = resolve_team_sides(BTreeMap::from([
            (1, vote(5, 0, 5)),
            (2, vote(5, 0, 5)),
            (3, vote(0, 5, 5)),
            (4, vote(0, 5, 5)),
            (5, vote(2, 3, 5)),
            (6, vote(3, 2, 5)),
        ]));
        assert_eq!(sides[&5], CourtSide::Far);
        assert_eq!(sides[&6], CourtSide::Near);

        let sides = resolve_team_sides(BTreeMap::from([
            (1, vote(5, 0, 5)),
            (2, vote(0, 5, 5)),
            (3, vote(3, 2, 5)),
        ]));
        assert_eq!(sides[&3], CourtSide::Near);

        let sides = resolve_team_sides(BTreeMap::from([(1, vote(5, 0, 5)), (2, vote(0, 0, 5))]));
        assert_eq!(sides[&2], CourtSide::Far);
    }

    #[test]
    fn team_side_uses_reliable_ground_observations() {
        let court = CourtTransform::new(QUAD).unwrap();
        let mut votes = TeamSideVotes::default();
        for x in [0.25, 0.28, 0.30, 0.75, 0.50] {
            votes.observe(&court, [x - 0.02, 0.3, x + 0.02, 0.6]);
        }
        assert_eq!(votes.side(), Some(CourtSide::Near));
        let mut uncertain = TeamSideVotes::default();
        for x in [0.25, 0.75, 0.50, 0.50] {
            uncertain.observe(&court, [x - 0.02, 0.3, x + 0.02, 0.6]);
        }
        assert_eq!(uncertain.side(), None);
        let mut far = TeamSideVotes::default();
        for x in [0.72, 0.78, 0.94] {
            far.observe(&court, [x - 0.02, 0.3, x + 0.02, 0.6]);
        }
        assert_eq!(far.side(), Some(CourtSide::Far));
    }

    #[test]
    fn zones_use_projective_court_midline_and_landing_margin() {
        // Trapezoid: arithmetic image midpoints would be wrong in perspective.
        let corners = [[0.15, 0.86], [0.37, 0.16], [0.70, 0.16], [0.94, 0.86]];
        let court = CourtTransform::new(corners)
            .unwrap()
            .with_net(Some([[0.37, 0.33], [0.71, 0.33]]))
            .unwrap();
        let zones = court.zones().unwrap();
        for point in zones.ground_divider {
            let metres = court.to_court(point).unwrap();
            assert!(metres[0].abs() < 1e-7);
        }
        assert!(zones.net_surface.is_some());
        let inside = court.floor_to_image([-3.0, 0.0]).unwrap();
        let outside = court.floor_to_image([12.0, 0.0]).unwrap();
        let line = court.floor_to_image([9.0, 0.0]).unwrap();
        assert_eq!(court.tentative_landing(inside), Some("near_half"));
        assert_eq!(court.tentative_landing(outside), Some("out"));
        assert_eq!(court.tentative_landing(line), Some("line_uncertain"));
        assert!(CourtTransform::new(corners)
            .unwrap()
            .zones()
            .unwrap()
            .net_surface
            .is_none());
    }

    #[test]
    fn python_court_transform_matches_corners_and_net() {
        let h = CourtTransform::new(QUAD).unwrap();
        let centre = h.to_court([0.5, 0.5]).unwrap();
        assert!(centre[0].abs() < 1e-7 && centre[1].abs() < 1e-7);
        for (pixel, expected) in
            QUAD.into_iter()
                .zip([[-9.0, -4.5], [9.0, -4.5], [9.0, 4.5], [-9.0, 4.5]])
        {
            let found = h.to_court(pixel).unwrap();
            assert!((found[0] - expected[0]).abs() < 1e-7);
            assert!((found[1] - expected[1]).abs() < 1e-7);
        }
    }

    #[test]
    fn perspective_calibration_remains_stable_off_image() {
        let h = CourtTransform::new([[-0.15, -0.1], [0.93, 0.14], [0.81, 0.88], [-0.33, 1.28]])
            .unwrap();
        assert!(h.to_court([0.5, 0.5]).is_some());
    }

    #[test]
    fn court_side_block_and_serve_zones_match_python_thresholds() {
        let h = CourtTransform::new(QUAD).unwrap();
        let center = h.evidence_for_feet([0.5, 0.5]).unwrap();
        assert_eq!(center.side, CourtSide::Far);
        assert!(center.near_net);
        assert!(!center.in_serve_zone);
        let server = h.evidence_for_feet([0.09, 0.5]).unwrap();
        assert_eq!(server.side, CourtSide::Near);
        assert!(server.in_serve_zone);
        assert!(!server.near_net);
        let mut legacy = h.evidence_for_feet([0.125, 0.5]).unwrap();
        assert!(!legacy.in_serve_zone);
        legacy.in_serve_zone = true;
        assert!(legacy.valid()); // Saved snapshots from the previous serve-zone threshold.
        assert!(h.evidence_for_feet([-0.08, 0.5]).unwrap().in_serve_zone); // 4m behind end line
        assert!(h.evidence_for_feet([0.98, 0.5]).is_some()); // 1.8 m outside end line
        assert!(h.evidence_for_feet([1.3, 0.5]).is_none());
    }

    #[test]
    fn invalid_geometry_never_silently_falls_back() {
        assert!(CourtTransform::new([[0.1, 0.1]; 4]).is_err());
        assert!(CourtTransform::new([[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]]).is_err());
        assert!(CourtTransform::new([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]).is_err());
        assert!(
            CourtTransform::new([[f64::NAN, 0.2], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],]).is_err()
        );
    }
}
