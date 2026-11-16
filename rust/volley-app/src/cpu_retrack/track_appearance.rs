//! Strict photo-based consent gate for joining provisional Track IDs.
//! Uses multiple measured body crops and the existing native person ReID
//! model, not roster photos, player names or future manual assignments.
use super::{track_merge::Row, BoundedReader};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    process::Command,
};
use volley_inference::reid_openvino::{self, Encoder};

const SAMPLES: usize = 5;
const MIN_SAMPLES: usize = 3;
const MIN_SEPARATION_FRAMES: u64 = 8;
const MIN_CROSS_SIMILARITY: f64 = 0.84;
const MIN_MEAN_SIMILARITY: f64 = 0.88;

pub(super) struct Source<'a> {
    pub video: &'a Path,
    pub repository: &'a Path,
}

/// Cosine similarity is an uncalibrated ReID score, NOT a probability.
/// All three or more independent crops from each provisional track must
/// have a strong counterpart in the other track, without contradictory
/// identity evidence inside either track.
fn same_appearance(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    if a.len() < MIN_SAMPLES || b.len() < MIN_SAMPLES {
        return false;
    }
    let valid = |vectors: &[Vec<f32>]| {
        vectors.iter().all(|v| {
            v.len() == reid_openvino::DIMENSIONS
                && v.iter().all(|n| n.is_finite())
                && (v.iter().map(|n| (*n as f64).powi(2)).sum::<f64>() - 1.0).abs() < 0.01
        })
    };
    if !valid(a) || !valid(b) {
        return false;
    }
    let similarity = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum::<f64>()
    };
    let consistent = |source: &[Vec<f32>], other: &[Vec<f32>]| {
        let nearest = source
            .iter()
            .map(|x| {
                other
                    .iter()
                    .map(|y| similarity(x, y))
                    .fold(-1.0_f64, f64::max)
            })
            .collect::<Vec<_>>();
        nearest.iter().all(|&score| score >= MIN_CROSS_SIMILARITY)
            && nearest.iter().sum::<f64>() / nearest.len() as f64 >= MIN_MEAN_SIMILARITY
    };
    // A long ID that switched owners must not authorize a merge based on
    // only its best (possibly wrong) crop.
    let internally_consistent = |vectors: &[Vec<f32>]| {
        vectors.iter().enumerate().all(|(i, x)| {
            vectors
                .iter()
                .enumerate()
                .all(|(j, y)| i == j || similarity(x, y) >= 0.80)
        })
    };
    internally_consistent(a) && internally_consistent(b) && consistent(a, b) && consistent(b, a)
}

fn representative_frames(samples: &[(u64, [f64; 4])]) -> Vec<(u64, [f64; 4])> {
    if samples.is_empty() {
        return Vec::new();
    }
    // Quantiles rather than the single best (possibly foreign) ReID crop.
    let mut chosen = Vec::<(u64, [f64; 4])>::new();
    for i in 0..SAMPLES {
        let index = i * (samples.len() - 1) / (SAMPLES - 1);
        let row = samples[index];
        if chosen
            .iter()
            .all(|previous| row.0.abs_diff(previous.0) >= MIN_SEPARATION_FRAMES)
        {
            chosen.push(row);
        }
    }
    chosen
}

fn crop_embedding(
    source: &Source<'_>,
    frame: u64,
    bbox: [f64; 4],
    dimensions: (u32, u32),
    encoder: &mut Encoder,
) -> Option<Vec<f32>> {
    let (width, height) = dimensions;
    if !bbox.iter().all(|x| x.is_finite())
        || bbox[0] < 0.0
        || bbox[1] < 0.0
        || bbox[2] > 1.0
        || bbox[3] > 1.0
    {
        return None;
    }
    let x = (bbox[0] * width as f64).floor() as u32;
    let y = (bbox[1] * height as f64).floor() as u32;
    let w = ((bbox[2] * width as f64).ceil() as u32)
        .min(width)
        .saturating_sub(x);
    let h = ((bbox[3] * height as f64).ceil() as u32)
        .min(height)
        .saturating_sub(y);
    if w < 12 || h < 28 || w > width || h > height {
        return None;
    }
    let vf = format!(
        "crop={w}:{h}:{x}:{y},scale={}:{}:flags=bilinear,format=bgr24",
        reid_openvino::WIDTH,
        reid_openvino::HEIGHT
    );
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-ss"])
        .arg(format!("{:.6}", frame as f64 / 30.0))
        .arg("-i")
        .arg(source.video)
        .args([
            "-vf",
            &vf,
            "-frames:v",
            "1",
            "-an",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .output()
        .ok()?;
    if !output.status.success() || output.stdout.len() != reid_openvino::PIXELS * 3 {
        return None;
    }
    let pixels = reid_openvino::bgr24_to_nchw(&output.stdout).ok()?;
    encoder.infer(&pixels).ok()
}

/// Fail closed if the model, sample frames or photo evidence are unavailable.
/// No image decoding is performed when there are no geometric merge candidates.
pub(super) fn approved_aliases(
    replay: &Path,
    expected_sha: &str,
    frames: u64,
    proposed: &BTreeMap<u64, u64>,
    source: Option<&Source<'_>>,
) -> BTreeMap<u64, u64> {
    let Some(source) = source else {
        return BTreeMap::new();
    };
    if proposed.is_empty() || proposed.len() > 64 {
        return BTreeMap::new();
    }
    let Ok(library) = std::env::var("VOLLEY_OPENVINO_C_LIB") else {
        return BTreeMap::new();
    };
    let Ok(mut encoder) = Encoder::load(source.repository, Path::new(&library)) else {
        return BTreeMap::new();
    };
    let Ok(video) = volley_media::probe(source.video) else {
        return BTreeMap::new();
    };
    let ids = proposed
        .iter()
        .flat_map(|(a, b)| [*a, *b])
        .collect::<BTreeSet<_>>();
    let mut crops = BTreeMap::<u64, Vec<(u64, [f64; 4])>>::new();
    let Ok(mut reader) = BoundedReader::open(replay) else {
        return BTreeMap::new();
    };
    // Large matches can contain millions of frames. Retain a bounded,
    // deterministic sample rather than storing every bbox for every ID.
    let stride = (frames / 1024).max(1);
    for frame in 0..frames {
        let Some(bytes) = reader.line().ok().flatten() else {
            return BTreeMap::new();
        };
        let Ok(row) = serde_json::from_slice::<Row>(&bytes) else {
            return BTreeMap::new();
        };
        if row.frame_index != frame {
            return BTreeMap::new();
        }
        for player in row.tracked_players {
            if ids.contains(&player.track_id) && player.detection.score >= 0.35 {
                let samples = crops.entry(player.track_id).or_default();
                if samples.len() < 1100 && (samples.len() < 12 || frame % stride == 0) {
                    samples.push((frame, player.detection.bbox_xyxy_norm));
                }
            }
        }
    }
    if reader.line().ok().flatten().is_some() || reader.finish() != expected_sha {
        return BTreeMap::new();
    }
    let mut vectors = BTreeMap::<u64, Vec<Vec<f32>>>::new();
    for id in &ids {
        let rows = crops.get(id).map(Vec::as_slice).unwrap_or_default();
        let features = representative_frames(rows)
            .into_iter()
            .filter_map(|(frame, bbox)| {
                crop_embedding(
                    source,
                    frame,
                    bbox,
                    (video.width, video.height),
                    &mut encoder,
                )
            })
            .collect::<Vec<_>>();
        vectors.insert(*id, features);
    }
    proposed
        .iter()
        .filter_map(|(&drop, &keep)| {
            same_appearance(&vectors[&drop], &vectors[&keep]).then_some((drop, keep))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn vector(i: usize) -> Vec<f32> {
        let mut v = vec![0.0; reid_openvino::DIMENSIONS];
        v[i] = 1.0;
        v
    }
    #[test]
    fn several_same_person_crops_are_required() {
        let a = vec![vector(0), vector(0), vector(0)];
        assert!(same_appearance(&a, &a));
        assert!(!same_appearance(&a[..2], &a));
        let foreign = vec![vector(0), vector(1), vector(1)];
        assert!(!same_appearance(&a, &foreign));
    }
    #[test]
    fn old_track_photo_cannot_authorize_merge_with_different_new_player() {
        let ireks = vec![vector(0), vector(0), vector(0), vector(0), vector(0)];
        let mixed = vec![vector(0), vector(0), vector(1), vector(1), vector(1)];
        assert!(!same_appearance(&ireks, &mixed));
        assert!(!same_appearance(&mixed, &ireks));
    }
    #[test]
    fn separate_observation_frames_not_adjacent_video_duplicates() {
        let f = (0..=40)
            .map(|x| (x, [0.1, 0.1, 0.2, 0.4]))
            .collect::<Vec<_>>();
        assert_eq!(representative_frames(&f).len(), 5);
        assert!(representative_frames(&f[..4]).len() < MIN_SAMPLES);
    }
}
