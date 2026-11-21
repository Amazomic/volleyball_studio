//! Strict, provider-independent decoding of VBallNet V4c ONNX
//! output ABI into **unlinked, unconfirmed** per-frame detection proposals.
//! No player Track ID or confirmed Touch is assigned by this layer.
use crate::Output;
use serde::Serialize;

const FRAMES: usize = 9;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RawDetection {
    pub frame_index: u64,
    pub class_name: &'static str,
    pub score: f32,
    pub bbox_xyxy_norm: [f32; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub touch_probability: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_point_norm: Option<[f32; 2]>,
}

fn required<'a>(outputs: &'a [Output], name: &str) -> Result<&'a Output, String> {
    outputs
        .iter()
        .find(|o| o.name == name)
        .ok_or_else(|| format!("ONNX output '{name}' is missing"))
}
fn dims(output: &Output, expected: &[i64]) -> Result<(), String> {
    if output.dimensions != expected {
        return Err(format!(
            "Invalid {} shape: {:?}, expected {expected:?}",
            output.name, output.dimensions
        ));
    }
    let len = expected
        .iter()
        .try_fold(1_usize, |a, &x| a.checked_mul(usize::try_from(x).ok()?))
        .ok_or_else(|| format!("Invalid {} element count", output.name))?;
    if output.data.len() != len {
        return Err(format!(
            "Incomplete output '{}': {} != {len}",
            output.name,
            output.data.len()
        ));
    }
    Ok(())
}
fn rect(cx: f32, cy: f32, w: f32, h: f32) -> Option<[f32; 4]> {
    if ![cx, cy, w, h].iter().all(|n| n.is_finite()) || w <= 0.0 || h <= 0.0 {
        return None;
    }
    let b = [
        (cx - w / 2.0).clamp(0.0, 1.0),
        (cy - h / 2.0).clamp(0.0, 1.0),
        (cx + w / 2.0).clamp(0.0, 1.0),
        (cy + h / 2.0).clamp(0.0, 1.0),
    ];
    if b[2] <= b[0] || b[3] <= b[1] {
        None
    } else {
        Some(b)
    }
}
/// VballNet 18 planes: first nine heatmaps + nine radius maps. Store
/// candidate coordinates and the model radius estimate for every frame.
pub fn decode_vball(
    outputs: &[Output],
    first_frame: u64,
    confidence_floor: f32,
) -> Result<Vec<RawDetection>, String> {
    if !confidence_floor.is_finite() || !(0.0..=1.0).contains(&confidence_floor) {
        return Err("Invalid Vball confidence floor".into());
    }
    let tensor = required(outputs, "heatmaps")?;
    dims(tensor, &[1, 18, 288, 512])?;
    let each = 288 * 512;
    let mut result = Vec::with_capacity(FRAMES);
    for frame in 0..FRAMES {
        let map = &tensor.data[frame * each..(frame + 1) * each];
        let Some((index, &score)) = map
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_finite())
            .max_by(|a, b| a.1.total_cmp(b.1))
        else {
            continue;
        };
        if score < confidence_floor || !(0.0..=1.0).contains(&score) {
            continue;
        }
        let radius = tensor.data[(frame + FRAMES) * each + index];
        if !radius.is_finite() || radius <= 0.0 {
            continue;
        }
        let x = (index % 512) as f32 / 512.0;
        let y = (index / 512) as f32 / 288.0;
        if let Some(bbox) = rect(x, y, radius * 2.0, radius * 2.0 * 512.0 / 288.0) {
            result.push(RawDetection {
                frame_index: first_frame + frame as u64,
                class_name: "ball",
                score,
                bbox_xyxy_norm: bbox,
                query_index: None,
                touch_probability: None,
                head_point_norm: None,
            });
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(name: &str, dimensions: &[i64], data: Vec<f32>) -> Output {
        Output {
            name: name.into(),
            dimensions: dimensions.into(),
            data,
        }
    }
    #[test]
    fn ball_heatmap_keeps_coordinate_and_normalized_extent() {
        let each = 288 * 512;
        let mut maps = vec![0.0; 18 * each];
        maps[50 * 512 + 100] = 0.9;
        maps[9 * each + 50 * 512 + 100] = 0.02;
        let records =
            decode_vball(&[tensor("heatmaps", &[1, 18, 288, 512], maps)], 0, 0.05).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].class_name, "ball");
        assert_eq!(records[0].frame_index, 0);
        assert!(records[0].bbox_xyxy_norm[2] > records[0].bbox_xyxy_norm[0]);
    }
}
