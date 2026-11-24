//! Optional, separately trained Volley3D-style 17-frame temporal segmentation.
//! This engine is CPU-only and never turns an untrained network into evidence.
//! Contract: input trajectory [1,4,17] (X,Y,dX,dY), output probabilities [1,1,17].
use ort::{execution_providers::CPUExecutionProvider, session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use std::path::Path;

pub const MODEL_FILENAME: &str = "volley3d_touch_tcn.onnx";
pub const WINDOW: usize = 17;
pub const STRIDE: usize = 3;
pub const PEAK_THRESHOLD: f32 = 0.35;

pub struct TrajectoryTcn {
    session: Session,
    sha256: String,
}

impl TrajectoryTcn {
    pub fn load(model: &Path) -> Result<Self, String> {
        let hash = checkpoint_sha256(model)?;
        #[cfg(any(target_os = "linux", windows))]
        let _native = super::runtime_library(super::Device::Cpu)?;
        ort::init()
            .commit()
            .map_err(|e| format!("Initialize ONNX Runtime TCN: {e}"))?;
        let session = Session::builder()
            .map_err(|e| format!("TCN session builder: {e}"))?
            .with_intra_threads(2)
            .map_err(|e| format!("TCN CPU threads: {e}"))?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| format!("TCN CPU execution provider: {e}"))?
            .commit_from_file(model)
            .map_err(|e| format!("Load trained TCN ONNX: {e}"))?;
        if session.inputs.len() != 1
            || session.inputs[0].name != "trajectory"
            || session.outputs.len() != 1
            || session.outputs[0].name != "probabilities"
        {
            return Err("TCN ONNX requires trajectory [1,4,17] -> probabilities [1,1,17]".into());
        }
        Ok(Self {
            session,
            sha256: hash,
        })
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Aggregate overlapping 17-frame windows, including short/end fragments.
    /// No probability is invented for an unobserved frame; the calling geometry
    /// layer requires real VBallNet observations and physical/actor evidence.
    pub fn score(&mut self, features: &[[f32; 4]]) -> Result<Vec<f32>, String> {
        if features.is_empty()
            || features.len() > 1200
            || features.iter().flatten().any(|f| !f.is_finite())
        {
            return Err("TCN input requires 1..=1200 finite trajectory frames".into());
        }
        let n = features.len();
        let last_start = n.saturating_sub(WINDOW);
        let mut starts: Vec<usize> = (0..=last_start).step_by(STRIDE).collect();
        if *starts.last().unwrap_or(&0) != last_start {
            starts.push(last_start);
        }
        let mut scores = vec![0f32; n];
        let mut counts = vec![0u16; n];
        for start in starts {
            let mut input = vec![0f32; 4 * WINDOW];
            for (dt, feature) in features[start..(start + WINDOW).min(n)].iter().enumerate() {
                for channel in 0..4 {
                    input[channel * WINDOW + dt] = feature[channel];
                }
            }
            let tensor = Tensor::from_array((vec![1_i64, 4, WINDOW as i64], input))
                .map_err(|e| format!("TCN tensor: {e}"))?;
            let out = self
                .session
                .run(ort::inputs!["trajectory" => tensor])
                .map_err(|e| format!("TCN ONNX inference: {e}"))?;
            let tensor = out
                .get("probabilities")
                .ok_or("TCN probabilities output missing")?;
            let (shape, data) = tensor
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("TCN output shape/type: {e}"))?;
            if shape.len() != 3
                || shape[0] != 1
                || shape[1] != 1
                || shape[2] != WINDOW as i64
                || data.len() != WINDOW
            {
                return Err("TCN output must be [1,1,17] f32".into());
            }
            for (dt, &probability) in data.iter().enumerate().take((n - start).min(WINDOW)) {
                if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                    return Err("TCN must output finite sigmoid probabilities".into());
                }
                scores[start + dt] += probability;
                counts[start + dt] += 1;
            }
        }
        for (score, count) in scores.iter_mut().zip(counts) {
            if count > 0 {
                *score /= f32::from(count);
            }
        }
        Ok(scores)
    }
}

pub fn checkpoint_sha256(model: &Path) -> Result<String, String> {
    let metadata = std::fs::symlink_metadata(model).map_err(|_| {
            format!(
                "Нет обученных весов TCN: {}. Нужен обученный ONNX checkpoint; произвольные веса не заменяют Volley3D.",
                model.display()
            )
        })?;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() == 0
        || metadata.len() > 200 * 1024 * 1024
    {
        return Err("Некорректный файл обученной TCN (или ссылка на файл)".into());
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(std::fs::read(model).map_err(|e| format!("Read TCN: {e}"))?)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_window_stride_contract() {
        assert_eq!(
            (0..=20usize.saturating_sub(WINDOW))
                .step_by(STRIDE)
                .collect::<Vec<_>>(),
            vec![0, 3]
        );
        assert_eq!(WINDOW, 17);
        assert_eq!(PEAK_THRESHOLD, 0.35);
    }
    #[test]
    fn untrained_checkpoint_is_rejected() {
        let p = std::env::temp_dir().join("missing-volley3d-tcn-model.onnx");
        assert!(TrajectoryTcn::load(&p)
            .err()
            .unwrap()
            .contains("обученных весов"));
    }
}
