use crate::Output;
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::Tensor,
};

pub struct CudaInput {
    tensor: Tensor<f32>,
    _allocator: Allocator,
}

impl std::ops::Deref for CudaInput {
    type Target = Tensor<f32>;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

const MAX_INPUT_ELEMENTS: usize = 100_000_000;
const MAX_OUTPUT_ELEMENTS: usize = 25_000_000;

fn element_count(shape: &[i64]) -> Result<usize, String> {
    if shape.is_empty() || shape.len() > 6 {
        return Err("CUDA model input rank is invalid".into());
    }
    shape
        .iter()
        .try_fold(1_usize, |acc, &dim| {
            let dim = usize::try_from(dim).ok().filter(|dim| *dim > 0)?;
            let product = acc.checked_mul(dim)?;
            (product <= MAX_INPUT_ELEMENTS).then_some(product)
        })
        .ok_or_else(|| "CUDA model input dimensions exceed the validated limit".into())
}

pub fn allocate(session: &Session, shape: &[i64], device_id: i32) -> Result<CudaInput, String> {
    element_count(shape)?;
    if !(0..=15).contains(&device_id) {
        return Err("Invalid CUDA input device ID".into());
    }
    let info = MemoryInfo::new(
        AllocationDevice::CUDA,
        device_id,
        AllocatorType::Device,
        MemoryType::Default,
    )
    .map_err(|error| format!("CUDA input memory description: {error}"))?;
    let allocator =
        Allocator::new(session, info).map_err(|error| format!("CUDA input allocator: {error}"))?;
    let tensor = Tensor::<f32>::new(&allocator, shape.to_vec())
        .map_err(|error| format!("Allocate CUDA model tensor: {error}"))?;
    Ok(CudaInput {
        tensor,
        _allocator: allocator,
    })
}

pub fn validate(input: &Tensor<f32>, expected_shape: &[i64], device_id: i32) -> Result<(), String> {
    element_count(expected_shape)?;
    let actual_shape = input
        .dtype()
        .tensor_shape()
        .ok_or("CUDA model input must be a tensor")?;
    if actual_shape.as_ref() != expected_shape {
        return Err(format!(
            "CUDA input shape mismatch: expected {expected_shape:?}, got {actual_shape:?}"
        ));
    }
    if input.memory_info().allocation_device() != AllocationDevice::CUDA
        || input.memory_info().device_id() != device_id
    {
        return Err("CUDA input must reside in the requested device VRAM".into());
    }
    if input.data_ptr().is_null() {
        return Err("CUDA input has no device allocation".into());
    }
    Ok(())
}

pub fn run(
    session: &mut Session,
    input: &Tensor<f32>,
    expected_shape: &[i64],
    input_name: &str,
    output_names: &[&str],
    device_id: i32,
) -> Result<Vec<Output>, String> {
    validate(input, expected_shape, device_id)?;
    if output_names.is_empty()
        || output_names.len() > 32
        || !session.inputs.iter().any(|item| item.name == input_name)
        || output_names
            .iter()
            .any(|name| !session.outputs.iter().any(|item| &item.name == name))
    {
        return Err("CUDA model tensor names do not match the ONNX session".into());
    }
    let mut binding = session
        .create_binding()
        .map_err(|error| format!("Create CUDA I/O binding: {error}"))?;
    binding
        .bind_input(input_name, input)
        .map_err(|error| format!("Bind CUDA model input: {error}"))?;
    let cpu_output = MemoryInfo::default();
    for &name in output_names {
        binding
            .bind_output_to_device(name, &cpu_output)
            .map_err(|error| format!("Bind model output {name}: {error}"))?;
    }
    let outputs = session
        .run_binding(&binding)
        .map_err(|error| format!("Run CUDA-bound model: {error}"))?;
    output_names
        .iter()
        .map(|&name| {
            let tensor = outputs
                .get(name)
                .ok_or_else(|| format!("CUDA model output {name} missing"))?;
            let (shape, values) = tensor
                .try_extract_tensor::<f32>()
                .map_err(|error| format!("Read CUDA model output {name}: {error}"))?;
            if values.len() > MAX_OUTPUT_ELEMENTS {
                return Err(format!("CUDA model output {name} exceeds the value limit"));
            }
            Ok(Output {
                name: name.to_owned(),
                dimensions: shape.iter().copied().collect(),
                data: values.to_vec(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_shapes_are_bounded_and_positive() {
        assert_eq!(element_count(&[1, 9, 288, 512]).unwrap(), 1_327_104);
        assert_eq!(element_count(&[8, 3, 256, 192]).unwrap(), 1_179_648);
        assert!(element_count(&[]).is_err());
        assert!(element_count(&[1, 3, 0, 192]).is_err());
        assert!(element_count(&[1, -1, 192]).is_err());
        assert!(element_count(&[1, i64::MAX]).is_err());
        assert!(element_count(&[1, 1, 1, 1, 1, 1, 1]).is_err());
    }
}
