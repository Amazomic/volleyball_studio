use mini_cuda::{
    driver::{try_init_driver, DriverApi},
    nvrtc::{nvrtcProgram, nvrtcResult, try_init_nvrtc},
    CUevent, CUfunction, CUmodule, CUresult, CUstream,
};
use ort::value::Tensor;
use std::{
    ffi::{c_void, CString},
    ptr,
    sync::{Mutex, OnceLock},
};

const MAX_NVRTC_LOG_BYTES: usize = 1024 * 1024;

fn nvrtc_log_capacity(bytes: usize) -> Result<usize, String> {
    if !(1..=MAX_NVRTC_LOG_BYTES).contains(&bytes) {
        return Err(format!(
            "NVRTC compilation log size is invalid: {bytes} bytes"
        ));
    }
    Ok(bytes)
}

#[derive(Clone, Copy, Debug)]
pub struct Nv12View {
    pub ptr: u64,
    pub pitch: usize,
    pub width: usize,
    pub height: usize,
}

impl Nv12View {
    pub fn validate(&self) -> Result<(), String> {
        if self.ptr == 0
            || self.pitch < self.width
            || self.pitch > 65536
            || !(2..=16384).contains(&self.width)
            || !(2..=16384).contains(&self.height)
            || !self.width.is_multiple_of(2)
            || !self.height.is_multiple_of(2)
            || self
                .pitch
                .checked_mul(self.height)
                .and_then(|bytes| bytes.checked_mul(3))
                .is_none_or(|bytes| bytes > 1024 * 1024 * 1024)
        {
            return Err("CUDA NV12 frame geometry is invalid".into());
        }
        Ok(())
    }
}

fn cuda(result: CUresult, operation: &str) -> Result<(), String> {
    if result == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation}: {result:?}"))
    }
}

pub fn bind_cuda_thread() -> Result<(), String> {
    mini_cuda::bind_primary_ctx_for_device(0)
        .map_err(|error| format!("Bind CUDA pose worker context: {error}"))
}

fn ensure() -> Result<&'static DriverApi, String> {
    mini_cuda::bind_primary_ctx_for_device(0)
        .map_err(|error| format!("Bind CUDA preprocessing context: {error}"))?;
    try_init_driver().ok_or_else(|| "CUDA driver is unavailable".into())
}

fn compile_ptx() -> Result<Vec<u8>, String> {
    static COMPILED_PTX: OnceLock<Mutex<Option<Vec<u8>>>> = OnceLock::new();
    let mut cached = COMPILED_PTX
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "NVRTC PTX cache was poisoned")?;
    if let Some(ptx) = cached.as_ref() {
        return Ok(ptx.clone());
    }
    let api = try_init_nvrtc().ok_or("CUDA NVRTC compiler is unavailable")?;
    let source = CString::new(include_str!("gpu_preprocess.cu"))
        .map_err(|error| format!("CUDA preprocessing source: {error}"))?;
    let name = CString::new("volley_zero_copy.cu").unwrap();
    let option = CString::new("--gpu-architecture=compute_80").unwrap();
    let options = [option.as_ptr()];
    let mut prog: nvrtcProgram = ptr::null_mut();
    let result = unsafe {
        (api.nvrtcCreateProgram)(
            &mut prog,
            source.as_ptr(),
            name.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        )
    };
    if result != nvrtcResult::NVRTC_SUCCESS {
        return Err(format!(
            "CUDA preprocessing NVRTC initialization: {result:?}"
        ));
    }
    let outcome = (|| {
        let result =
            unsafe { (api.nvrtcCompileProgram)(prog, options.len() as i32, options.as_ptr()) };
        if result != nvrtcResult::NVRTC_SUCCESS {
            let mut bytes = 0usize;
            let size_status = unsafe { (api.nvrtcGetProgramLogSize)(prog, &mut bytes) };
            let diagnostics = if size_status == nvrtcResult::NVRTC_SUCCESS {
                match nvrtc_log_capacity(bytes) {
                    Ok(size) => {
                        let mut log = vec![0u8; size];
                        let read_status =
                            unsafe { (api.nvrtcGetProgramLog)(prog, log.as_mut_ptr().cast()) };
                        if read_status == nvrtcResult::NVRTC_SUCCESS {
                            String::from_utf8_lossy(&log).into_owned()
                        } else {
                            format!("NVRTC compilation log read failed: {read_status:?}")
                        }
                    }
                    Err(error) => error,
                }
            } else {
                format!("NVRTC compilation log size unavailable: {size_status:?}")
            };
            return Err(format!(
                "CUDA preprocessing compilation: {result:?}: {}",
                diagnostics
            ));
        }
        let mut size = 0usize;
        let result = unsafe { (api.nvrtcGetPTXSize)(prog, &mut size) };
        if result != nvrtcResult::NVRTC_SUCCESS || !(1..=8 * 1024 * 1024).contains(&size) {
            return Err("CUDA preprocessing PTX exceeds validated bounds".into());
        }
        let mut bytes = vec![0u8; size];
        let result = unsafe { (api.nvrtcGetPTX)(prog, bytes.as_mut_ptr().cast()) };
        if result != nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("Read CUDA preprocessing PTX: {result:?}"));
        }
        Ok(bytes)
    })();
    let _ = unsafe { (api.nvrtcDestroyProgram)(&mut prog) };
    if let Ok(ptx) = &outcome {
        *cached = Some(ptx.clone());
    }
    outcome
}

pub struct GpuPreprocessor {
    module: CUmodule,
    ball: CUfunction,
    rgb: CUfunction,
    person: CUfunction,
    pose: CUfunction,
    canonical: CUfunction,
    stream: CUstream,
    ready: CUevent,
    rgb_scratch: GpuBuffer,
    pose_scratch: GpuBuffer,
}

impl GpuPreprocessor {
    pub fn new() -> Result<Self, String> {
        let driver = ensure()?;
        let ptx = compile_ptx()?;
        let mut module = ptr::null_mut();
        cuda(
            unsafe { (driver.cuModuleLoadData)(&mut module, ptx.as_ptr().cast()) },
            "Load CUDA preprocessing module",
        )?;
        let load = |name: &str| -> Result<CUfunction, String> {
            let mut function = ptr::null_mut();
            let symbol = CString::new(name).unwrap();
            cuda(
                unsafe { (driver.cuModuleGetFunction)(&mut function, module, symbol.as_ptr()) },
                "Load CUDA preprocessing kernel",
            )?;
            Ok(function)
        };
        let result = (|| {
            let mut stream = ptr::null_mut();
            cuda(
                unsafe { (driver.cuStreamCreate)(&mut stream, 1) },
                "Create CUDA preprocessing stream",
            )?;
            let mut ready = ptr::null_mut();
            if let Err(error) = cuda(
                unsafe { (driver.cuEventCreate)(&mut ready, 2) },
                "Create CUDA preprocessing event",
            ) {
                let _ = unsafe { (driver.cuStreamDestroy_v2)(stream) };
                return Err(error);
            }
            let assembled = (|| {
                Ok(Self {
                    module,
                    ball: load("ball_gray")?,
                    rgb: load("rgb_640")?,
                    person: load("rfdetr_image")?,
                    pose: load("pose_crop")?,
                    canonical: load("canonical_nv12")?,
                    stream,
                    ready,
                    rgb_scratch: GpuBuffer::allocate(640 * 360 * 3)?,
                    pose_scratch: GpuBuffer::allocate(8 * 4 * std::mem::size_of::<f32>())?,
                })
            })();
            if assembled.is_err() {
                let _ = unsafe { (driver.cuEventDestroy_v2)(ready) };
                let _ = unsafe { (driver.cuStreamDestroy_v2)(stream) };
            }
            assembled
        })();
        if result.is_err() {
            let _ = unsafe { (driver.cuModuleUnload)(module) };
        }
        result
    }

    fn launch(
        &self,
        function: CUfunction,
        width: u32,
        height: u32,
        batches: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), String> {
        let driver = ensure()?;
        cuda(
            unsafe {
                (driver.cuLaunchKernel)(
                    function,
                    width.div_ceil(16),
                    height.div_ceil(16),
                    batches,
                    16,
                    16,
                    1,
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "Launch CUDA preprocessing",
        )
    }

    pub fn finish(&self) -> Result<(), String> {
        let driver = ensure()?;
        cuda(
            unsafe { (driver.cuEventRecord)(self.ready, self.stream) },
            "Record CUDA preprocessing completion",
        )?;
        cuda(
            unsafe { (driver.cuEventSynchronize)(self.ready) },
            "Wait for CUDA preprocessing before consuming or unmapping NVDEC pixels",
        )
    }

    pub fn canonical_frame(
        &self,
        frame: Nv12View,
        bit_depth: u8,
        chroma: u8,
        rotation: i32,
        full_range: bool,
    ) -> Result<(GpuBuffer, usize, usize), String> {
        if !matches!(bit_depth, 8 | 10 | 12)
            || !matches!(chroma, 1..=3)
            || !matches!(rotation, 0 | 90 | 180 | 270)
            || frame.ptr == 0
            || frame.width < 2
            || frame.height < 2
            || (frame.width | frame.height) & 1 != 0
            || frame.pitch < frame.width * if bit_depth > 8 { 2 } else { 1 }
        {
            return Err("CUDA source surface or orientation is invalid".into());
        }
        let (ow, oh) = if rotation == 90 || rotation == 270 {
            (frame.height, frame.width)
        } else {
            (frame.width, frame.height)
        };
        let len = ow
            .checked_mul(oh)
            .and_then(|v| v.checked_mul(3))
            .map(|v| v / 2)
            .ok_or("CUDA canonical surface too large")?;
        let output = GpuBuffer::allocate(len)?;
        let mut src = frame.ptr;
        let mut pitch = frame.pitch as i32;
        let mut sw = frame.width as i32;
        let mut sh = frame.height as i32;
        let mut dst = output.ptr;
        let mut ow_arg = ow as i32;
        let mut oh_arg = oh as i32;
        let mut orientation = rotation;
        let mut depth = i32::from(bit_depth);
        let mut chroma = i32::from(chroma);
        let mut range = i32::from(full_range);
        let mut args: [*mut c_void; 11] = [
            (&mut src as *mut u64).cast(),
            (&mut pitch as *mut i32).cast(),
            (&mut sw as *mut i32).cast(),
            (&mut sh as *mut i32).cast(),
            (&mut dst as *mut u64).cast(),
            (&mut ow_arg as *mut i32).cast(),
            (&mut oh_arg as *mut i32).cast(),
            (&mut orientation as *mut i32).cast(),
            (&mut depth as *mut i32).cast(),
            (&mut chroma as *mut i32).cast(),
            (&mut range as *mut i32).cast(),
        ];
        self.launch(self.canonical, ow as u32, oh as u32, 1, &mut args)?;
        self.finish()?;
        Ok((output, ow, oh))
    }

    pub fn ball_input(
        &self,
        frame: Nv12View,
        tensor: &Tensor<f32>,
        slot: usize,
    ) -> Result<(), String> {
        frame.validate()?;
        if slot >= 9
            || tensor.dtype().tensor_shape().map(|v| v.as_ref()) != Some(&[1, 9, 288, 512][..])
        {
            return Err("VBallNet CUDA input shape or frame slot is invalid".into());
        }
        let mut src = frame.ptr;
        let mut pitch = frame.pitch as i32;
        let mut width = frame.width as i32;
        let mut height = frame.height as i32;
        let mut dst = tensor.data_ptr() as u64;
        let mut slot = slot as i32;
        let mut args: [*mut c_void; 6] = [
            (&mut src as *mut u64).cast(),
            (&mut pitch as *mut i32).cast(),
            (&mut width as *mut i32).cast(),
            (&mut height as *mut i32).cast(),
            (&mut dst as *mut u64).cast(),
            (&mut slot as *mut i32).cast(),
        ];
        self.launch(self.ball, 512, 288, 1, &mut args)
    }

    pub fn person_input(
        &self,
        frame: Nv12View,
        tensor: &Tensor<f32>,
        matrix: u8,
    ) -> Result<(), String> {
        frame.validate()?;
        if tensor.dtype().tensor_shape().map(|v| v.as_ref()) != Some(&[1, 3, 576, 576][..]) {
            return Err("RF-DETR CUDA input shape is invalid".into());
        }
        let rgb = &self.rgb_scratch;
        let mut src = frame.ptr;
        let mut pitch = frame.pitch as i32;
        let mut width = frame.width as i32;
        let mut height = frame.height as i32;
        let mut rgb_ptr = rgb.ptr;
        let mut matrix = i32::from(matrix);
        let mut args: [*mut c_void; 6] = [
            (&mut src as *mut u64).cast(),
            (&mut pitch as *mut i32).cast(),
            (&mut width as *mut i32).cast(),
            (&mut height as *mut i32).cast(),
            (&mut rgb_ptr as *mut u64).cast(),
            (&mut matrix as *mut i32).cast(),
        ];
        self.launch(self.rgb, 640, 360, 1, &mut args)?;
        let mut dst = tensor.data_ptr() as u64;
        let mut args: [*mut c_void; 2] = [
            (&mut rgb_ptr as *mut u64).cast(),
            (&mut dst as *mut u64).cast(),
        ];
        self.launch(self.person, 576, 576, 1, &mut args)
    }

    pub fn pose_input(
        &self,
        frame: Nv12View,
        tensor: &Tensor<f32>,
        boxes: &[[f64; 4]],
        matrix: u8,
    ) -> Result<(), String> {
        frame.validate()?;
        if boxes.is_empty()
            || boxes.len() > 8
            || tensor.dtype().tensor_shape().map(|v| v.as_ref())
                != Some(&[boxes.len() as i64, 3, 256, 192][..])
        {
            return Err("RTMPose CUDA tensor or player count is invalid".into());
        }
        let rectangles = crate::pose::cuda_crop_geometry(boxes, frame.width, frame.height)?;
        let values: Vec<f32> = rectangles.into_iter().flatten().collect();
        let coords = &self.pose_scratch;
        let driver = ensure()?;
        cuda(
            unsafe {
                (driver.cuMemcpyHtoD_v2)(
                    coords.ptr,
                    values.as_ptr().cast(),
                    values.len() * std::mem::size_of::<f32>(),
                )
            },
            "Upload GPU-only RTMPose geometry",
        )?;
        let mut src = frame.ptr;
        let mut pitch = frame.pitch as i32;
        let mut width = frame.width as i32;
        let mut height = frame.height as i32;
        let mut rects = coords.ptr;
        let mut output = tensor.data_ptr() as u64;
        let mut matrix = i32::from(matrix);
        let mut args: [*mut c_void; 7] = [
            (&mut src as *mut u64).cast(),
            (&mut pitch as *mut i32).cast(),
            (&mut width as *mut i32).cast(),
            (&mut height as *mut i32).cast(),
            (&mut rects as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut matrix as *mut i32).cast(),
        ];
        self.launch(self.pose, 192, 256, boxes.len() as u32, &mut args)?;
        self.finish()
    }
}

impl Drop for GpuPreprocessor {
    fn drop(&mut self) {
        if let Ok(driver) = ensure() {
            let _ = unsafe { (driver.cuEventDestroy_v2)(self.ready) };
            let _ = unsafe { (driver.cuStreamDestroy_v2)(self.stream) };
            let _ = unsafe { (driver.cuModuleUnload)(self.module) };
        }
    }
}

#[derive(Debug)]
pub struct GpuBuffer {
    pub ptr: u64,
    len: usize,
}

impl GpuBuffer {
    pub fn allocate(len: usize) -> Result<Self, String> {
        if len == 0 || len > 512 * 1024 * 1024 {
            return Err("CUDA GPU buffer size is outside validated bounds".into());
        }
        let driver = ensure()?;
        let mut ptr = 0_u64;
        cuda(
            unsafe { (driver.cuMemAlloc_v2)(&mut ptr, len) },
            "Allocate CUDA buffer",
        )?;
        Ok(Self { ptr, len })
    }

    pub fn copy_frame(frame: Nv12View) -> Result<Self, String> {
        frame.validate()?;
        let len = frame.pitch * frame.height * 3 / 2;
        let buffer = Self::allocate(len)?;
        let driver = ensure()?;
        cuda(
            unsafe { (driver.cuMemcpy)(buffer.ptr, frame.ptr, len) },
            "Retain NVDEC frame in VRAM",
        )?;
        Ok(buffer)
    }

    pub fn view(&self, pitch: usize, width: usize, height: usize) -> Result<Nv12View, String> {
        let frame = Nv12View {
            ptr: self.ptr,
            pitch,
            width,
            height,
        };
        frame.validate()?;
        if self.len < pitch * height * 3 / 2 {
            return Err("Saved NVDEC frame is incomplete".into());
        }
        Ok(frame)
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if let Ok(driver) = ensure() {
            let _ = unsafe { (driver.cuMemFree_v2)(self.ptr) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nvrtc_log_capacity_never_truncates_native_write_buffer() {
        assert!(nvrtc_log_capacity(0).is_err());
        assert_eq!(nvrtc_log_capacity(16 * 1024 + 1), Ok(16 * 1024 + 1));
        assert_eq!(
            nvrtc_log_capacity(MAX_NVRTC_LOG_BYTES),
            Ok(MAX_NVRTC_LOG_BYTES)
        );
        assert!(nvrtc_log_capacity(MAX_NVRTC_LOG_BYTES + 1).is_err());
    }
    #[test]
    fn cuda_p016_rotation_preserves_expected_luma_in_device_memory() {
        if mini_cuda::driver::try_init_driver().is_none()
            || mini_cuda::nvrtc::try_init_nvrtc().is_none()
        {
            return;
        }
        let gpu = GpuPreprocessor::new().unwrap();
        let driver = ensure().unwrap();
        for rotation in [0, 90, 180, 270] {
            let mut pixels = [0u8; 48];
            for y in 0..4 {
                for x in 0..4 {
                    let value = ((32 + y * 4 + x) as u16) << 8;
                    let offset = y * 8 + x * 2;
                    pixels[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
                }
            }
            for pair in pixels[32..].as_chunks_mut::<2>().0 {
                pair.copy_from_slice(&(128u16 << 8).to_le_bytes());
            }
            let input = GpuBuffer::allocate(pixels.len()).unwrap();
            cuda(
                unsafe {
                    (driver.cuMemcpyHtoD_v2)(input.ptr, pixels.as_ptr().cast(), pixels.len())
                },
                "P016 sample upload",
            )
            .unwrap();
            let (out, width, height) = gpu
                .canonical_frame(
                    Nv12View {
                        ptr: input.ptr,
                        pitch: 8,
                        width: 4,
                        height: 4,
                    },
                    10,
                    1,
                    rotation,
                    false,
                )
                .unwrap();
            assert_eq!((width, height), (4, 4));
            let mut data = vec![0u8; 24];
            cuda(
                unsafe { (driver.cuMemcpyDtoH_v2)(data.as_mut_ptr().cast(), out.ptr, data.len()) },
                "P016 canonical download",
            )
            .unwrap();
            for y in 0..4 {
                for x in 0..4 {
                    let (sx, sy) = match rotation {
                        90 => (3 - y, x),
                        180 => (3 - x, 3 - y),
                        270 => (y, 3 - x),
                        _ => (x, y),
                    };
                    assert_eq!(data[y * 4 + x], (32 + sy * 4 + sx) as u8);
                }
            }
            assert!(data[16..].iter().all(|value| *value == 128));
        }
    }

    #[test]
    fn cuda_nv16_p216_and_yuv444_convert_to_nv12_without_host_pixels() {
        if mini_cuda::driver::try_init_driver().is_none()
            || mini_cuda::nvrtc::try_init_nvrtc().is_none()
        {
            return;
        }
        let gpu = GpuPreprocessor::new().unwrap();
        let driver = ensure().unwrap();
        for chroma in [2, 3] {
            let mut pixels = if chroma == 2 {
                vec![0u8; 64]
            } else {
                vec![0u8; 48]
            };
            let pitch = if chroma == 2 { 8 } else { 4 };
            let depth = if chroma == 2 { 10 } else { 8 };
            if chroma == 2 {
                for row in 0..4 {
                    for x in 0..4 {
                        pixels[row * 8 + x * 2..row * 8 + x * 2 + 2].copy_from_slice(
                            &((32u16 + row as u16 * 4 + x as u16) << 8).to_le_bytes(),
                        );
                        let chroma_value = if x % 2 == 0 { 64 + row } else { 180 + row };
                        pixels[32 + row * 8 + x * 2..32 + row * 8 + x * 2 + 2]
                            .copy_from_slice(&((chroma_value as u16) << 8).to_le_bytes());
                    }
                }
            } else {
                pixels[..16].fill(64);
                pixels[16..32].fill(80);
                pixels[32..].fill(190);
            }
            let source = GpuBuffer::allocate(pixels.len()).unwrap();
            cuda(
                unsafe {
                    (driver.cuMemcpyHtoD_v2)(source.ptr, pixels.as_ptr().cast(), pixels.len())
                },
                "YUV444 or P216 upload",
            )
            .unwrap();
            let (out, w, h) = gpu
                .canonical_frame(
                    Nv12View {
                        ptr: source.ptr,
                        pitch,
                        width: 4,
                        height: 4,
                    },
                    depth,
                    chroma,
                    0,
                    false,
                )
                .unwrap();
            assert_eq!((w, h), (4, 4));
            let mut result = [0u8; 24];
            cuda(
                unsafe {
                    (driver.cuMemcpyDtoH_v2)(result.as_mut_ptr().cast(), out.ptr, result.len())
                },
                "Canonical multi-chroma download",
            )
            .unwrap();
            if chroma == 2 {
                assert_eq!(&result[16..20], &[64, 180, 64, 180]);
                assert_eq!(&result[20..24], &[66, 182, 66, 182]);
            } else {
                assert_eq!(&result[16..], &[80, 190, 80, 190, 80, 190, 80, 190]);
            }
        }
    }

    #[test]
    fn cuda_nv12_preprocessing_runs_on_device() {
        if mini_cuda::driver::try_init_driver().is_none()
            || mini_cuda::nvrtc::try_init_nvrtc().is_none()
        {
            return;
        }
        let converter = GpuPreprocessor::new().unwrap();
        let yuv = GpuBuffer::allocate(512 * 288 * 3 / 2).unwrap();
        let input = vec![128u8; 512 * 288 * 3 / 2];
        let driver = ensure().unwrap();
        cuda(
            unsafe { (driver.cuMemcpyHtoD_v2)(yuv.ptr, input.as_ptr().cast(), input.len()) },
            "Test NV12 upload",
        )
        .unwrap();
        let output = GpuBuffer::allocate(9 * 512 * 288 * 4).unwrap();
        let mut src = yuv.ptr;
        let mut pitch = 512i32;
        let mut width = 512i32;
        let mut height = 288i32;
        let mut dst = output.ptr;
        let mut slot = 0i32;
        let mut args: [*mut c_void; 6] = [
            (&mut src as *mut u64).cast(),
            (&mut pitch as *mut i32).cast(),
            (&mut width as *mut i32).cast(),
            (&mut height as *mut i32).cast(),
            (&mut dst as *mut u64).cast(),
            (&mut slot as *mut i32).cast(),
        ];
        converter
            .launch(converter.ball, 512, 288, 1, &mut args)
            .unwrap();
        converter.finish().unwrap();
        let mut sample = [0f32; 8];
        cuda(
            unsafe {
                (driver.cuMemcpyDtoH_v2)(
                    sample.as_mut_ptr().cast(),
                    output.ptr,
                    std::mem::size_of_val(&sample),
                )
            },
            "Test NV12 output download",
        )
        .unwrap();
        for &value in &sample {
            assert!((value - ((128f32 - 16f32) / 219f32)).abs() < 0.002);
        }
    }

    #[test]
    fn gpu_nv12_shape_validation_rejects_invalid_buffers() {
        assert!(Nv12View {
            ptr: 0,
            pitch: 512,
            width: 512,
            height: 288
        }
        .validate()
        .is_err());
        assert!(Nv12View {
            ptr: 1,
            pitch: 64,
            width: 512,
            height: 288
        }
        .validate()
        .is_err());
        assert!(Nv12View {
            ptr: 1,
            pitch: 512,
            width: 512,
            height: 288
        }
        .validate()
        .is_ok());
    }
}

#[cfg(test)]
mod ffmpeg_parity_tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn gpu_rgb_and_ball_resizes_match_ffmpeg_reference() {
        if mini_cuda::driver::try_init_driver().is_none()
            || mini_cuda::nvrtc::try_init_nvrtc().is_none()
            || Command::new("ffmpeg").arg("-version").output().is_err()
        {
            return;
        }
        for (src_width, src_height) in [(640usize, 360usize), (1920usize, 1080usize)] {
            let mut nv12 = vec![128u8; src_width * src_height * 3 / 2];
            for y in 0..src_height {
                for x in 0..src_width {
                    nv12[y * src_width + x] = (32 + ((x / 5 + y / 3) % 160)) as u8;
                }
            }
            for (idx, val) in nv12[src_width * src_height..].iter_mut().enumerate() {
                *val = if idx % 2 == 0 { 90 } else { 180 };
            }
            let path = std::env::temp_dir().join(format!(
                "volley-gpu-ffmpeg-parity-{}-{}.nv12",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::write(&path, &nv12).unwrap();
            let reference = |filter: &str, format: &str| {
                let output = Command::new("ffmpeg")
                    .args([
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "nv12",
                        "-video_size",
                        &format!("{src_width}x{src_height}"),
                        "-framerate",
                        "30",
                        "-i",
                    ])
                    .arg(&path)
                    .args([
                        "-frames:v",
                        "1",
                        "-vf",
                        filter,
                        "-pix_fmt",
                        format,
                        "-f",
                        "rawvideo",
                        "pipe:1",
                    ])
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                output.stdout
            };
            let rgb_ref = reference("scale=640:360:flags=bilinear,format=rgb24", "rgb24");
            let gray_ref = reference("scale=512:288:flags=bilinear,format=gray", "gray");
            let native_ref = reference("format=rgb24", "rgb24");
            let _ = std::fs::remove_file(&path);
            assert_eq!(rgb_ref.len(), 640 * 360 * 3);
            assert_eq!(gray_ref.len(), 512 * 288);

            let driver = ensure().unwrap();
            let src = GpuBuffer::allocate(nv12.len()).unwrap();
            cuda(
                unsafe { (driver.cuMemcpyHtoD_v2)(src.ptr, nv12.as_ptr().cast(), nv12.len()) },
                "parity nv12 upload",
            )
            .unwrap();
            let processing = GpuPreprocessor::new().unwrap();
            let mut src_ptr = src.ptr;
            let mut pitch = src_width as i32;
            let mut width = src_width as i32;
            let mut height = src_height as i32;
            let mut bt709 = 0i32;
            let rgb_gpu = GpuBuffer::allocate(rgb_ref.len()).unwrap();
            let mut rgb_ptr = rgb_gpu.ptr;
            let mut params: [*mut c_void; 6] = [
                (&mut src_ptr as *mut u64).cast(),
                (&mut pitch as *mut i32).cast(),
                (&mut width as *mut i32).cast(),
                (&mut height as *mut i32).cast(),
                (&mut rgb_ptr as *mut u64).cast(),
                (&mut bt709 as *mut i32).cast(),
            ];
            processing
                .launch(processing.rgb, 640, 360, 1, &mut params)
                .unwrap();
            processing.finish().unwrap();
            let mut rgb_actual = vec![0u8; rgb_ref.len()];
            cuda(
                unsafe {
                    (driver.cuMemcpyDtoH_v2)(
                        rgb_actual.as_mut_ptr().cast(),
                        rgb_gpu.ptr,
                        rgb_actual.len(),
                    )
                },
                "parity rgb download",
            )
            .unwrap();
            let errors: Vec<usize> = rgb_ref
                .iter()
                .zip(&rgb_actual)
                .map(|(a, b)| (*a as i16 - *b as i16).unsigned_abs() as usize)
                .collect();
            let max_rgb = errors.iter().copied().max().unwrap_or(0);
            let larger = errors.iter().filter(|&&e| e > 2).count();
            eprintln!(
                "NV12-RGB FFmpeg/GPU: maximum byte error={max_rgb}, >2 byte errors={larger}/{}",
                errors.len()
            );
            assert!(max_rgb <= 6, "GPU RGB conversion differs from FFmpeg");

            let ball = GpuBuffer::allocate(9 * 512 * 288 * 4).unwrap();
            let mut ball_ptr = ball.ptr;
            let mut slot = 0i32;
            let mut params: [*mut c_void; 6] = [
                (&mut src_ptr as *mut u64).cast(),
                (&mut pitch as *mut i32).cast(),
                (&mut width as *mut i32).cast(),
                (&mut height as *mut i32).cast(),
                (&mut ball_ptr as *mut u64).cast(),
                (&mut slot as *mut i32).cast(),
            ];
            processing
                .launch(processing.ball, 512, 288, 1, &mut params)
                .unwrap();
            processing.finish().unwrap();
            let mut actual_gray = vec![0f32; 512 * 288];
            cuda(
                unsafe {
                    (driver.cuMemcpyDtoH_v2)(
                        actual_gray.as_mut_ptr().cast(),
                        ball.ptr,
                        actual_gray.len() * 4,
                    )
                },
                "parity ball download",
            )
            .unwrap();
            let error: Vec<f32> = actual_gray
                .iter()
                .zip(gray_ref)
                .map(|(a, b)| (a - (b as f32 / 255.0)).abs())
                .collect();
            let max_gray = error.iter().copied().fold(0f32, f32::max);
            let count = error.iter().filter(|&&e| e > 0.025).count();
            eprintln!(
                "NV12-gray FFmpeg/GPU: maximum float error={max_gray}, >0.025 errors={count}/{}",
                error.len()
            );
            assert!(max_gray < 0.08, "GPU ball resize diverges from FFmpeg");
            assert_eq!(native_ref.len(), src_width * src_height * 3);
            let boxes = [[0.2f64, 0.1, 0.42, 0.75]];
            let rect = crate::pose::cuda_crop_geometry(&boxes, src_width, src_height).unwrap();
            let geometry = GpuBuffer::allocate(16).unwrap();
            cuda(
                unsafe { (driver.cuMemcpyHtoD_v2)(geometry.ptr, rect.as_ptr().cast(), 16) },
                "parity pose geometry",
            )
            .unwrap();
            let pose_data = GpuBuffer::allocate(3 * 256 * 192 * 4).unwrap();
            let mut coords = geometry.ptr;
            let mut pose_ptr = pose_data.ptr;
            let mut params: [*mut c_void; 7] = [
                (&mut src_ptr as *mut u64).cast(),
                (&mut pitch as *mut i32).cast(),
                (&mut width as *mut i32).cast(),
                (&mut height as *mut i32).cast(),
                (&mut coords as *mut u64).cast(),
                (&mut pose_ptr as *mut u64).cast(),
                (&mut bt709 as *mut i32).cast(),
            ];
            processing
                .launch(processing.pose, 192, 256, 1, &mut params)
                .unwrap();
            processing.finish().unwrap();
            let mut pose_actual = vec![0f32; 3 * 256 * 192];
            cuda(
                unsafe {
                    (driver.cuMemcpyDtoH_v2)(
                        pose_actual.as_mut_ptr().cast(),
                        pose_data.ptr,
                        pose_actual.len() * 4,
                    )
                },
                "parity pose crop",
            )
            .unwrap();
            let mut pose_error = 0f32;
            let pixel = |x: isize, y: isize, channel: usize| -> f32 {
                if x < 0 || y < 0 || x >= src_width as isize || y >= src_height as isize {
                    0.0
                } else {
                    native_ref[(y as usize * src_width + x as usize) * 3 + channel] as f32
                }
            };
            let means = [123.675, 116.28, 103.53];
            let stds = [58.395, 57.12, 57.375];
            let [cx, cy, cw, ch] = rect[0];
            for channel in 0..3 {
                for (x, y) in [(0, 0), (1, 1), (27, 36), (91, 131), (111, 244), (191, 255)] {
                    let px = cx + ((x as f32 + 0.5) / 192.0 - 0.5) * cw - 0.5;
                    let py = cy + ((y as f32 + 0.5) / 256.0 - 0.5) * ch - 0.5;
                    let x0 = px.floor() as isize;
                    let y0 = py.floor() as isize;
                    let wx = px - x0 as f32;
                    let wy = py - y0 as f32;
                    let a = pixel(x0, y0, channel) * (1.0 - wx) + pixel(x0 + 1, y0, channel) * wx;
                    let b = pixel(x0, y0 + 1, channel) * (1.0 - wx)
                        + pixel(x0 + 1, y0 + 1, channel) * wx;
                    let expected = ((1.0 - wy) * a + wy * b - means[channel]) / stds[channel];
                    let actual = pose_actual[(channel * 256 + y) * 192 + x];
                    pose_error = pose_error.max((actual - expected).abs());
                }
            }
            eprintln!(
                "NV12-RTMPose FFmpeg/GPU: max normalized crop error={pose_error}, source={}x{}",
                src_width, src_height,
            );
            assert!(
                pose_error < 0.08,
                "GPU RTMPose crop differs from original FFmpeg RGB"
            );
        }
    }
}
