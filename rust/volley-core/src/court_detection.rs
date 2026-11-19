//! Lightweight CPU-only line evidence for a fixed volleyball camera.
//! All coordinates are normalized to the decoded image. Low-confidence
//! detections are returned as None, never as fabricated court coordinates.
use crate::court_geometry::{CourtTransform, COURT_LENGTH_M, COURT_WIDTH_M};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct CourtDetection {
    pub court: [[f64; 2]; 4],
    pub net: Option<[[f64; 2]; 2]>,
    pub confidence: f64,
    pub frames_considered: usize,
    /// At least one boundary was extrapolated rather than directly measured.
    pub inferred: bool,
}

#[derive(Clone, Copy)]
struct Line {
    theta: usize,
    rho: i32,
    votes: u32,
}
const ANGLES: usize = 180;

// A Hough line at 179 degrees has opposite signed rho to one at 0 degrees.
fn parallel_gap(a: Line, b: Line, max_angle: i32) -> Option<i32> {
    let angle = (a.theta as i32 - b.theta as i32).abs();
    if angle > max_angle && angle < 180 - max_angle {
        return None;
    }
    Some(if angle > 90 {
        (a.rho + b.rho).abs()
    } else {
        (a.rho - b.rho).abs()
    })
}

// Match court transform convention: near-left -> far-left -> far-right -> near-right.
fn canonical_court(c: [[f64; 2]; 4]) -> [[f64; 2]; 4] {
    let tl = (0..4)
        .min_by(|&a, &b| (c[a][0] + c[a][1]).total_cmp(&(c[b][0] + c[b][1])))
        .unwrap_or(0);
    let (tr, bl) = if c[(tl + 1) % 4][0] > c[(tl + 3) % 4][0] {
        ((tl + 1) % 4, (tl + 3) % 4)
    } else {
        ((tl + 3) % 4, (tl + 1) % 4)
    };
    [c[bl], c[tl], c[tr], c[(tl + 2) % 4]]
}

fn intersections(a: Line, b: Line, width: f64, height: f64) -> Option<[f64; 2]> {
    let (sa, ca) = (a.theta as f64).to_radians().sin_cos();
    let (sb, cb) = (b.theta as f64).to_radians().sin_cos();
    let det = ca * sb - cb * sa;
    if det.abs() < 0.12 {
        return None;
    }
    let ra = a.rho as f64 * 2.0;
    let rb = b.rho as f64 * 2.0;
    let x = (ra * sb - rb * sa) / det / width;
    let y = (rb * ca - ra * cb) / det / height;
    (x.is_finite() && y.is_finite() && (-0.45..=1.45).contains(&x) && (-0.45..=1.45).contains(&y))
        .then_some([x, y])
}

/// Pixel-space support of one Hough line (from the same chromatic edge mask).
/// Endpoints are evidence extents, not guaranteed physical line endpoints.
fn line_support(line: Line, pixels: &[(f64, f64)]) -> Option<([f64; 2], [f64; 2], usize)> {
    let (sin, cos) = (line.theta as f64).to_radians().sin_cos();
    let distance = line.rho as f64 * 2.0;
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    let mut count = 0;
    for &(x, y) in pixels {
        if (x * cos + y * sin - distance).abs() > 3.0 {
            continue;
        }
        min[0] = min[0].min(x);
        min[1] = min[1].min(y);
        max[0] = max[0].max(x);
        max[1] = max[1].max(y);
        count += 1;
    }
    (count >= 12).then_some((min, max, count))
}

/// Infer a provisional quadrilateral from a visible far endline and a segment
/// of *one* sideline. No homography is uniquely determined by two lines alone:
/// the missing sideline and near endline use an explicit bounded camera-pitch
/// prior and must be manually confirmed. Never treat inferred geometry as
/// authoritative calibration.
fn infer_partial_court(
    lines: &[Line],
    pixels: &[(f64, f64)],
    width: usize,
    height: usize,
) -> Option<(f64, [[f64; 2]; 4], [usize; 4])> {
    let (w, h) = (width as f64, height as f64);
    let mut best: Option<(f64, [[f64; 2]; 4], [usize; 4])> = None;
    // An endline is roughly horizontal in the standard elevated broadcast
    // view. A sideline normally projects diagonally or close to vertical.
    for (base_index, &base) in lines.iter().enumerate() {
        let (base_sin, base_cos) = (base.theta as f64).to_radians().sin_cos();
        if base_sin.abs() < 0.76 {
            continue;
        }
        let Some((base_min, base_max, base_count)) = line_support(base, pixels) else {
            continue;
        };
        let full_width = base_max[0] - base_min[0];
        if full_width < w * 0.20 || base_count < 20 {
            continue;
        }
        for (side_index, &side) in lines.iter().enumerate() {
            if side_index == base_index {
                continue;
            }
            let (side_sin, side_cos) = (side.theta as f64).to_radians().sin_cos();
            if side_cos.abs() < 0.48 {
                continue;
            }
            let angle = (base.theta as i32 - side.theta as i32).abs();
            if !(24..=156).contains(&angle) {
                continue;
            }
            let Some((side_min, side_max, side_count)) = line_support(side, pixels) else {
                continue;
            };
            let Some(far) = intersections(base, side, w, h) else {
                continue;
            };
            if !(-0.15..=1.15).contains(&far[0]) || !(0.08..=0.75).contains(&far[1]) {
                continue;
            }
            // The painted sideline must continue *towards the camera* from
            // the far corner, not just cross the endline briefly.
            let visible_side = side_max[1] / h - far[1];
            if visible_side < 0.10
                || (side_max[1] - side_min[1]) < 0.12 * h
                || side_count < 18
                || (visible_side < 0.17 && full_width < w * 0.32)
            {
                continue;
            }
            // Only a supported endpoint can anchor the opposite far corner.
            let far_x = far[0] * w;
            let other_x = if (base_max[0] - far_x).abs() >= (base_min[0] - far_x).abs() {
                base_max[0]
            } else {
                base_min[0]
            };
            let far_sep_px = (other_x - far_x).abs();
            if far_sep_px < w * 0.19 || far_sep_px > w * 1.05 {
                continue;
            }
            let other_far = [
                other_x / w,
                ((base.rho as f64 * 2.0 - other_x * base_cos) / base_sin) / h,
            ];
            if !(-0.15..=1.15).contains(&other_far[0]) || !(0.02..=0.82).contains(&other_far[1]) {
                continue;
            }
            // 18m x 9m is the physical prior. Perspective foreshortens
            // longitudinal depth: the ratio is scaled by a moderate camera-
            // pitch coefficient (0.38), with a bounded visible-frame extent.
            let projected_depth =
                (far_sep_px / h * (COURT_LENGTH_M / COURT_WIDTH_M) * 0.38).clamp(0.36, 0.82);
            let near_y = (far[1] + projected_depth)
                .max(side_max[1] / h + 0.025)
                .min(1.12);
            if near_y - far[1] < 0.26 {
                continue;
            }
            // Extend the *observed* sideline exactly, without inventing an
            // angle; infer only the opposite sideline's convergent width.
            let near_y_px = near_y * h;
            let side_near_x = ((side.rho as f64 * 2.0 - near_y_px * side_sin) / side_cos) / w;
            let near_scale = 1.0 + (near_y - far[1]) * 0.45;
            let other_near = [
                side_near_x + (other_far[0] - far[0]) * near_scale,
                near_y + (other_far[1] - far[1]) * near_scale,
            ];
            let proposed = canonical_court([far, other_far, other_near, [side_near_x, near_y]]);
            if CourtTransform::new(proposed).is_err() {
                continue;
            }
            let area = (0..4)
                .map(|i| {
                    let a = proposed[i];
                    let b = proposed[(i + 1) % 4];
                    a[0] * b[1] - b[0] * a[1]
                })
                .sum::<f64>()
                .abs()
                * 0.5;
            if !(0.065..=1.45).contains(&area) {
                continue;
            }
            let score = (base_count + side_count) as f64
                * (far_sep_px / w).sqrt()
                * ((side_max[1] - far[1] * h) / h).clamp(0.15, 0.9);
            if best
                .as_ref()
                .is_none_or(|(best_score, _, _)| score > *best_score)
            {
                best = Some((
                    score,
                    proposed,
                    [base_index, side_index, usize::MAX, usize::MAX],
                ));
            }
        }
    }
    best
}

/// Detect painted court lines; when incomplete, suggest a low-confidence
/// geometry-prior extrapolation rather than requiring four visible boundaries.
pub fn detect_frame(rgb: &[u8], width: usize, height: usize) -> Option<CourtDetection> {
    if width < 96 || height < 72 || rgb.len() != width.checked_mul(height)?.checked_mul(3)? {
        return None;
    }
    let mut pixels = Vec::new();
    // Thin the chromaticity-filtered line mask to avoid thick painted lines
    // generating multiple nearly identical Hough peaks.
    for y in (6..height - 6).step_by(2) {
        for x in (6..width - 6).step_by(2) {
            let i = (y * width + x) * 3;
            let (r, g, b) = (rgb[i], rgb[i + 1], rgb[i + 2]);
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            if min < 120 || max - min > 68 {
                continue;
            }
            let left = (y * width + x - 6) * 3;
            let right = (y * width + x + 6) * 3;
            let above = ((y - 6) * width + x) * 3;
            let below = ((y + 6) * width + x) * 3;
            let luma = |offset: usize| {
                (rgb[offset] as u16 + rgb[offset + 1] as u16 + rgb[offset + 2] as u16) / 3
            };
            let local = luma(left)
                .min(luma(right))
                .min(luma(above))
                .min(luma(below));
            if (min as u16) < local + 22 {
                continue;
            }
            pixels.push((x as f64, y as f64));
        }
    }
    if pixels.len() < 65 || pixels.len() > width * height / 6 {
        return None;
    }
    let diag = ((width * width + height * height) as f64).sqrt();
    let bins = diag.ceil() as usize + 3;
    let stride = bins * 2 + 1;
    let mut votes = vec![0u32; ANGLES * stride];
    let trig: Vec<(f64, f64)> = (0..ANGLES)
        .map(|angle| {
            let rad = (angle as f64).to_radians();
            (rad.cos(), rad.sin())
        })
        .collect();
    for &(x, y) in &pixels {
        for (angle, &(c, s)) in trig.iter().enumerate() {
            let r = ((x * c + y * s) * 0.5).round() as isize + bins as isize;
            if r >= 0 && (r as usize) < stride {
                votes[angle * stride + r as usize] += 1;
            }
        }
    }
    // Side lines may only be visible over a short distance behind players.
    // Keep weaker Hough peaks; geometry/segment support rejects noise later.
    let min_votes = ((width.min(height) as f64) * 0.075).round().max(16.0) as u32;
    let mut lines = Vec::new();
    for _ in 0..40 {
        let Some((idx, &count)) = votes.iter().enumerate().max_by_key(|(_, v)| *v) else {
            break;
        };
        if count < min_votes {
            break;
        }
        let angle = idx / stride;
        let rho = (idx % stride) as i32 - bins as i32;
        lines.push(Line {
            theta: angle,
            rho,
            votes: count,
        });
        // Suppress adjacent peaks generated by painted line thickness.
        for da in -4i32..=4 {
            let a = ((angle as i32 + da).rem_euclid(ANGLES as i32)) as usize;
            for dr in -5i32..=5 {
                let r = rho + dr + bins as i32;
                if r >= 0 && (r as usize) < stride {
                    votes[a * stride + r as usize] = 0;
                }
            }
        }
    }
    if lines.len() < 2 {
        return None;
    }
    let mut best: Option<(f64, [[f64; 2]; 4], [usize; 4])> = None;
    // Select two distinct line families. Require each pair to be spatially
    // separated, each corner to be finite, and a plausible quadrilateral.
    for a in 0..lines.len() {
        for b in a + 1..lines.len() {
            let first = [lines[a], lines[b]];
            let Some(gap) = parallel_gap(first[0], first[1], 32) else {
                continue;
            };
            if gap < 6 {
                continue;
            }
            for c in 0..lines.len() {
                if c == a || c == b {
                    continue;
                }
                for d in c + 1..lines.len() {
                    if d == a || d == b {
                        continue;
                    }
                    let second = [lines[c], lines[d]];
                    let Some(gap) = parallel_gap(second[0], second[1], 32) else {
                        continue;
                    };
                    if gap < 6 {
                        continue;
                    }
                    let cross = (first[0].theta as i32 - second[0].theta as i32).abs();
                    if !(22..=158).contains(&cross) {
                        continue;
                    }
                    let Some(p0) = intersections(first[0], second[0], width as f64, height as f64)
                    else {
                        continue;
                    };
                    let Some(p1) = intersections(first[0], second[1], width as f64, height as f64)
                    else {
                        continue;
                    };
                    let Some(p2) = intersections(first[1], second[1], width as f64, height as f64)
                    else {
                        continue;
                    };
                    let Some(p3) = intersections(first[1], second[0], width as f64, height as f64)
                    else {
                        continue;
                    };
                    let corners = [p0, p1, p2, p3];
                    if CourtTransform::new(corners).is_err() {
                        continue;
                    }
                    let area = (0..4)
                        .map(|k| {
                            let p = corners[k];
                            let q = corners[(k + 1) % 4];
                            p[0] * q[1] - q[0] * p[1]
                        })
                        .sum::<f64>()
                        .abs()
                        * 0.5;
                    if !(0.055..=1.25).contains(&area) {
                        continue;
                    }
                    // Reject implausibly narrow quadrilaterals.
                    let lengths: Vec<f64> = (0..4)
                        .map(|k| {
                            let p = corners[k];
                            let q = corners[(k + 1) % 4];
                            (p[0] - q[0]).hypot(p[1] - q[1])
                        })
                        .collect();
                    if lengths.iter().any(|l| *l < 0.11) {
                        continue;
                    }
                    let offscreen = corners
                        .iter()
                        .flatten()
                        .map(|v| (-*v).max(*v - 1.0).max(0.0))
                        .sum::<f64>();
                    let support = first
                        .iter()
                        .chain(second.iter())
                        .map(|l| l.votes as f64)
                        .sum::<f64>();
                    let score = support * area.sqrt() / (1.0 + offscreen * 3.0);
                    if best.as_ref().is_none_or(|(s, _, _)| score > *s) {
                        best = Some((score, corners, [a, b, c, d]));
                    }
                }
            }
        }
    }
    let inferred = best.is_none();
    let (score, court, selected) = match best {
        Some(found) => found,
        None => infer_partial_court(&lines, &pixels, width, height)?,
    };
    let court = canonical_court(court);
    CourtTransform::new(court).ok()?;
    // An inferred shape must never be presented with full confidence.
    let confidence = if inferred {
        (0.12 + score / pixels.len() as f64 * 0.14).min(0.28)
    } else {
        (score / (pixels.len() as f64 * 0.85)).clamp(0.0, 1.0)
    };
    if confidence < 0.10 {
        return None;
    }
    let net = lines
        .iter()
        .enumerate()
        .filter(|(index, _)| !selected.contains(index))
        .map(|(_, line)| line)
        .filter(|l| (l.theta as i32 - 90).abs() <= 18)
        .filter_map(|l| {
            let t = (l.theta as f64).to_radians();
            let y0 = (l.rho as f64 * 2.0) / (t.sin() * height as f64);
            let y1 = (l.rho as f64 * 2.0 - (width as f64) * t.cos()) / (t.sin() * height as f64);
            let avg = (y0 + y1) * 0.5;
            (avg > 0.10
                && avg < 0.72
                && (y0 - y1).abs() < 0.3
                && (-0.5..=1.5).contains(&y0)
                && (-0.5..=1.5).contains(&y1))
            .then_some((l.votes, [[0.0, y0], [1.0, y1]]))
        })
        .max_by_key(|(votes, _)| *votes)
        .map(|(_, line)| line);
    Some(CourtDetection {
        court,
        net,
        confidence,
        frames_considered: 1,
        inferred,
    })
}

/// Match all cyclic corner rotations and both windings before consensus.
pub fn combine_frames(frames: &[CourtDetection]) -> Option<CourtDetection> {
    combine_frames_with_support(frames, 3)
}

/// Two corroborating frames are usable as a low-confidence *manual* suggestion.
pub fn combine_frames_advisory(frames: &[CourtDetection]) -> Option<CourtDetection> {
    let mut result = combine_frames_with_support(frames, 2)?;
    result.confidence *= 0.65;
    Some(result)
}

fn align_to(reference: [[f64; 2]; 4], candidate: [[f64; 2]; 4]) -> Option<[[f64; 2]; 4]> {
    let mut best: Option<(f64, [[f64; 2]; 4])> = None;
    for reverse in [false, true] {
        for shift in 0..4 {
            let aligned: [[f64; 2]; 4] = std::array::from_fn(|i| {
                candidate[if reverse {
                    (shift + 4 - i) % 4
                } else {
                    (shift + i) % 4
                }]
            });
            let distances: [f64; 4] = std::array::from_fn(|i| {
                (reference[i][0] - aligned[i][0]).hypot(reference[i][1] - aligned[i][1])
            });
            if distances.iter().all(|&d| d < 0.09) {
                let error = distances.iter().sum::<f64>();
                if best.as_ref().is_none_or(|(score, _)| error < *score) {
                    best = Some((error, aligned));
                }
            }
        }
    }
    best.map(|(_, corners)| corners)
}

fn combine_frames_with_support(
    frames: &[CourtDetection],
    min_support: usize,
) -> Option<CourtDetection> {
    if frames.len() < min_support {
        return None;
    }
    let mut best_group: Vec<([[f64; 2]; 4], &CourtDetection)> = Vec::new();
    for reference in frames {
        let group: Vec<_> = frames
            .iter()
            .filter_map(|candidate| {
                align_to(reference.court, candidate.court).map(|aligned| (aligned, candidate))
            })
            .collect();
        if group.len() > best_group.len() {
            best_group = group;
        }
    }
    if best_group.len() < min_support || best_group.len() * 2 < frames.len() {
        return None;
    }
    let mut court = [[0.; 2]; 4];
    for i in 0..4 {
        for j in 0..2 {
            let mut v: Vec<_> = best_group.iter().map(|(c, _)| c[i][j]).collect();
            v.sort_by(f64::total_cmp);
            court[i][j] = v[v.len() / 2];
        }
    }
    let court = canonical_court(court);
    CourtTransform::new(court).ok()?;
    let net_candidates: Vec<_> = best_group.iter().filter_map(|(_, d)| d.net).collect();
    let net = if net_candidates.len() * 2 >= best_group.len() && net_candidates.len() >= 2 {
        let mut supported = Vec::new();
        for reference in &net_candidates {
            let group: Vec<_> = net_candidates
                .iter()
                .filter(|candidate| {
                    reference
                        .iter()
                        .zip(candidate.iter())
                        .all(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]) < 0.07)
                })
                .copied()
                .collect();
            if group.len() > supported.len() {
                supported = group;
            }
        }
        if supported.len() >= 2 && supported.len() * 2 >= best_group.len() {
            supported.sort_by(|a, b| a[0][1].total_cmp(&b[0][1]));
            Some(supported[supported.len() / 2])
        } else {
            None
        }
    } else {
        None
    };
    let confidence = best_group.iter().map(|(_, f)| f.confidence).sum::<f64>()
        / best_group.len() as f64
        * (best_group.len() as f64 / frames.len() as f64);
    Some(CourtDetection {
        court,
        net,
        confidence,
        frames_considered: frames.len(),
        inferred: best_group.iter().any(|(_, d)| d.inferred),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synthetic_painted_court() {
        const W: usize = 320;
        const H: usize = 180;
        let mut rgb = vec![35u8; W * H * 3];
        let mut line = |x0: i32, y0: i32, x1: i32, y1: i32| {
            for i in 0..=1000 {
                let x = x0 + (x1 - x0) * i / 1000;
                let y = y0 + (y1 - y0) * i / 1000;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let xx = x + dx;
                        let yy = y + dy;
                        if xx >= 0 && yy >= 0 && (xx as usize) < W && (yy as usize) < H {
                            let p = (yy as usize * W + xx as usize) * 3;
                            rgb[p..p + 3].fill(245);
                        }
                    }
                }
            }
        };
        line(35, 35, 275, 35);
        line(275, 35, 275, 155);
        line(275, 155, 35, 155);
        line(35, 155, 35, 35);
        line(15, 86, 305, 86); // net top cable, distinct from outer court
        let detection = detect_frame(&rgb, W, H).expect("painted court");
        assert!(
            detection.net.is_some(),
            "separate net cable should be found"
        );
    }
    #[test]
    fn dim_thick_court_lines_survive_higher_resolution() {
        const W: usize = 640;
        const H: usize = 360;
        let mut rgb = vec![40u8; W * H * 3];
        let mut paint = |x0: i32, y0: i32, x1: i32, y1: i32| {
            for i in 0..=1000 {
                let x = x0 + (x1 - x0) * i / 1000;
                let y = y0 + (y1 - y0) * i / 1000;
                for dy in -4..=4 {
                    for dx in -4..=4 {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx >= 0 && yy >= 0 && (xx as usize) < W && (yy as usize) < H {
                            rgb[(yy as usize * W + xx as usize) * 3
                                ..(yy as usize * W + xx as usize) * 3 + 3]
                                .fill(190);
                        }
                    }
                }
            }
        };
        paint(80, 75, 550, 80);
        paint(550, 80, 540, 305);
        paint(540, 305, 75, 305);
        paint(75, 305, 80, 75);
        assert!(detect_frame(&rgb, W, H).is_some());
    }

    #[test]
    fn no_lines_never_invents_a_court() {
        assert!(detect_frame(&vec![0; 320 * 180 * 3], 320, 180).is_none());
    }
    #[test]
    fn far_endline_and_one_sideline_offer_provisional_court() {
        const W: usize = 640;
        const H: usize = 360;
        let mut rgb = vec![38u8; W * H * 3];
        let mut paint = |x0: i32, y0: i32, x1: i32, y1: i32| {
            for step in 0..=1200 {
                let x = x0 + (x1 - x0) * step / 1200;
                let y = y0 + (y1 - y0) * step / 1200;
                for dy in -3..=3 {
                    for dx in -3..=3 {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx >= 0 && yy >= 0 && xx < W as i32 && yy < H as i32 {
                            let offset = (yy as usize * W + xx as usize) * 3;
                            rgb[offset..offset + 3].fill(222);
                        }
                    }
                }
            }
        };
        // No near endline, and no right sideline at all.
        paint(120, 95, 540, 95);
        paint(120, 95, 70, 310);
        let found = detect_frame(&rgb, W, H).expect("one endline plus one sideline");
        assert!(found.inferred, "missing boundaries are extrapolations");
        assert!(found.confidence <= 0.28);
        assert!(CourtTransform::new(found.court).is_ok());
        assert!((found.court[1][1] - 95.0 / H as f64).abs() < 0.09);
        assert!((found.court[2][1] - 95.0 / H as f64).abs() < 0.09);
        assert!(found.court[0][1] > found.court[1][1] + 0.25);
    }

    #[test]
    fn fragmented_far_line_and_short_sideline_remain_usable() {
        const W: usize = 640;
        const H: usize = 360;
        let mut rgb = vec![38u8; W * H * 3];
        let mut paint = |x0: i32, y0: i32, x1: i32, y1: i32| {
            for step in 0..=1000 {
                let x = x0 + (x1 - x0) * step / 1000;
                let y = y0 + (y1 - y0) * step / 1000;
                for dy in -3..=3 {
                    for dx in -3..=3 {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx >= 0 && yy >= 0 && xx < W as i32 && yy < H as i32 {
                            let offset = (yy as usize * W + xx as usize) * 3;
                            rgb[offset..offset + 3].fill(222);
                        }
                    }
                }
            }
        };
        // Players hide the middle section of the baseline, and most of the sideline.
        paint(120, 95, 235, 95);
        paint(320, 95, 540, 95);
        paint(120, 95, 105, 155);
        let found = detect_frame(&rgb, W, H).expect("fragmented line evidence");
        assert!(found.inferred, "missing boundaries must be marked inferred");
    }

    #[test]
    fn single_endline_cannot_define_a_court() {
        const W: usize = 640;
        const H: usize = 360;
        let mut rgb = vec![38u8; W * H * 3];
        for y in 90..96 {
            for x in 60..580 {
                rgb[(y * W + x) * 3..(y * W + x) * 3 + 3].fill(230);
            }
        }
        assert!(detect_frame(&rgb, W, H).is_none());
    }

    #[test]
    fn wrapped_hough_normals_have_signed_rho() {
        let a = Line {
            theta: 0,
            rho: 12,
            votes: 50,
        };
        let same = Line {
            theta: 179,
            rho: -12,
            votes: 50,
        };
        let distinct = Line {
            theta: 179,
            rho: -30,
            votes: 50,
        };
        assert_eq!(parallel_gap(a, same, 10), Some(0));
        assert_eq!(parallel_gap(a, distinct, 10), Some(18));
    }

    #[test]
    fn temporal_consensus_aligns_rotations_and_reverse_winding() {
        let c = [[0.10, 0.78], [0.24, 0.20], [0.79, 0.19], [0.92, 0.82]];
        let sample = |court| CourtDetection {
            court,
            net: Some([[0.0, 0.45], [1.0, 0.46]]),
            confidence: 0.8,
            frames_considered: 1,
            inferred: false,
        };
        let rotated = [c[1], c[2], c[3], c[0]];
        let reversed = [c[0], c[3], c[2], c[1]];
        let result = combine_frames(&[sample(c), sample(rotated), sample(reversed)]).unwrap();
        assert!(result.confidence > 0.7);
        assert_eq!(result.court, c);
        assert!(result.net.is_some());
    }

    #[test]
    fn advisory_requires_two_matching_frames() {
        let c = [[0.1, 0.8], [0.1, 0.2], [0.9, 0.2], [0.9, 0.8]];
        let sample = CourtDetection {
            court: c,
            net: None,
            confidence: 0.75,
            frames_considered: 1,
            inferred: false,
        };
        assert!(combine_frames(&[sample.clone(), sample.clone()]).is_none());
        let advisory = combine_frames_advisory(&[sample.clone(), sample]);
        assert!(advisory.is_some());
        assert!(advisory.unwrap().confidence < 0.75);
    }

    #[test]
    fn consensus_rejects_unsupported_suggestions() {
        let sample = CourtDetection {
            court: [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            net: None,
            confidence: 0.8,
            frames_considered: 1,
            inferred: false,
        };
        assert!(combine_frames(&[sample.clone(), sample.clone()]).is_none());
        assert!(combine_frames(&[sample.clone(), sample.clone(), sample]).is_some());
    }
}
