use mini_cuda::{driver::try_init_driver, CUresult};
use mini_nvcodec::{
    cuvid_sys::{
        cudaVideoChromaFormat, cudaVideoCodec, cudaVideoCreate_PreferCUVID,
        cudaVideoDeinterlaceMode, cudaVideoSurfaceFormat, CUvideodecoder, CUvideoparser,
        CUVIDDECODECAPS, CUVIDDECODECREATEINFO, CUVIDEOFORMAT, CUVIDPARSERDISPINFO,
        CUVIDPARSERPARAMS, CUVIDPICPARAMS, CUVIDPROCPARAMS, CUVIDRECT, CUVIDSOURCEDATAPACKET,
        CUVID_PKT_ENDOFSTREAM,
    },
    loader::{try_init_cuvid, CuvidApi},
};
use std::{
    ffi::c_void,
    io::Read,
    marker::PhantomData,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    sync::atomic::{AtomicBool, Ordering},
};

const MAX_BYTES_PER_PACKET: usize = 128 * 1024;
const MAX_SURFACES: u8 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NvdecCodec {
    H264,
    Hevc,
}

impl NvdecCodec {
    fn native(self) -> cudaVideoCodec {
        match self {
            Self::H264 => cudaVideoCodec::H264,
            Self::Hevc => cudaVideoCodec::HEVC,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NvdecReceipt {
    pub frames: u64,
    pub width: usize,
    pub height: usize,
}

pub struct GpuNv12Frame<'a> {
    pub frame_index: u64,
    pub device_ptr: u64,
    pub pitch: usize,
    pub width: usize,
    pub height: usize,
    _mapped: PhantomData<&'a mut ()>,
}

struct DecoderState<F> {
    api: &'static CuvidApi,
    decoder: CUvideodecoder,
    codec: NvdecCodec,
    frames: u64,
    width: usize,
    height: usize,
    progressive: bool,
    surfaces: u8,
    callback: F,
    failure: Option<String>,
}

impl<F> DecoderState<F>
where
    F: for<'frame> FnMut(GpuNv12Frame<'frame>) -> Result<(), String>,
{
    fn sequence(&mut self, format: &CUVIDEOFORMAT) -> Result<i32, String> {
        if !self.decoder.is_null() {
            return Err(
                "NVDEC source changed sequence geometry after decoder initialization".into(),
            );
        }
        if format.codec != self.codec.native()
            || format.chroma_format != cudaVideoChromaFormat::YUV420
            || format.bit_depth_luma_minus8 != 0
            || format.progressive_sequence == 0
        {
            return Err("NVDEC source must use progressive 8-bit H264/HEVC YUV420".into());
        }
        let (coded_width, coded_height) =
            (format.coded_width as usize, format.coded_height as usize);
        let area = format.display_area;
        let width = (area.right - area.left) as usize;
        let height = (area.bottom - area.top) as usize;
        if !(144..=8192).contains(&width)
            || !(144..=8192).contains(&height)
            || width > coded_width
            || height > coded_height
            || !width.is_multiple_of(2)
            || !height.is_multiple_of(2)
            || (width as u64) * (height as u64) > 8192 * 4320
        {
            return Err("NVDEC sequence dimensions or crop are not supported".into());
        }
        let surfaces = format.min_num_decode_surfaces.max(8);
        if surfaces > MAX_SURFACES {
            return Err("NVDEC requested more than 32 decode surfaces".into());
        }
        let mut caps: CUVIDDECODECAPS = unsafe { std::mem::zeroed() };
        caps.eCodecType = self.codec.native();
        caps.eChromaFormat = cudaVideoChromaFormat::YUV420;
        let status = unsafe { (self.api.cuvidGetDecoderCaps)(&mut caps) };
        if status != CUresult::CUDA_SUCCESS || caps.bIsSupported != 1 {
            return Err(format!("NVDEC decoder capability probe failed: {status:?}"));
        }
        let mut info: CUVIDDECODECREATEINFO = unsafe { std::mem::zeroed() };
        info.ulWidth = coded_width as _;
        info.ulHeight = coded_height as _;
        info.ulNumDecodeSurfaces = surfaces as _;
        info.CodecType = self.codec.native();
        info.ChromaFormat = cudaVideoChromaFormat::YUV420;
        info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
        info.OutputFormat = cudaVideoSurfaceFormat::NV12;
        info.DeinterlaceMode = cudaVideoDeinterlaceMode::WEAVE;
        info.ulTargetWidth = width as _;
        info.ulTargetHeight = height as _;
        info.ulNumOutputSurfaces = 2;
        info.display_area = CUVIDRECT {
            left: i16::try_from(area.left).map_err(|_| "Invalid NVDEC crop left")?,
            top: i16::try_from(area.top).map_err(|_| "Invalid NVDEC crop top")?,
            right: i16::try_from(area.right).map_err(|_| "Invalid NVDEC crop right")?,
            bottom: i16::try_from(area.bottom).map_err(|_| "Invalid NVDEC crop bottom")?,
        };
        let status = unsafe { (self.api.cuvidCreateDecoder)(&mut self.decoder, &mut info) };
        if status != CUresult::CUDA_SUCCESS {
            return Err(format!("NVDEC create decoder failed: {status:?}"));
        }
        self.width = width;
        self.height = height;
        self.progressive = true;
        self.surfaces = surfaces;
        Ok(surfaces as i32)
    }

    fn decode(&mut self, picture: *mut CUVIDPICPARAMS) -> Result<(), String> {
        if self.decoder.is_null() {
            return Err("NVDEC picture arrived before sequence initialization".into());
        }
        let status = unsafe { (self.api.cuvidDecodePicture)(self.decoder, picture) };
        if status != CUresult::CUDA_SUCCESS {
            return Err(format!("NVDEC picture decode failed: {status:?}"));
        }
        Ok(())
    }

    fn display(&mut self, display: &CUVIDPARSERDISPINFO) -> Result<(), String> {
        if self.decoder.is_null() {
            return Err("NVDEC display arrived without an initialized decoder".into());
        }
        let mut ptr = 0_u64;
        let mut pitch = 0_u32;
        let mut params: CUVIDPROCPARAMS = unsafe { std::mem::zeroed() };
        params.progressive_frame = i32::from(self.progressive);
        params.top_field_first = display.top_field_first;
        let status = unsafe {
            (self.api.cuvidMapVideoFrame64)(
                self.decoder,
                display.picture_index,
                &mut ptr,
                &mut pitch,
                &mut params,
            )
        };
        if status != CUresult::CUDA_SUCCESS {
            return Err(format!("Map NVDEC GPU surface: {status:?}"));
        }
        let outcome = (self.callback)(GpuNv12Frame {
            frame_index: self.frames,
            device_ptr: ptr,
            pitch: pitch as usize,
            width: self.width,
            height: self.height,
            _mapped: PhantomData,
        });
        let completion = try_init_driver()
            .ok_or_else(|| "CUDA driver disappeared during NVDEC output".to_owned())
            .and_then(|driver| {
                let status = unsafe { (driver.cuCtxSynchronize)() };
                (status == CUresult::CUDA_SUCCESS)
                    .then_some(())
                    .ok_or_else(|| format!("Synchronize NVDEC consumer before unmap: {status:?}"))
            });
        let unmap = unsafe { (self.api.cuvidUnmapVideoFrame64)(self.decoder, ptr) };
        if unmap != CUresult::CUDA_SUCCESS {
            return Err(format!("Unmap NVDEC GPU surface failed: {unmap:?}"));
        }
        outcome?;
        completion?;
        self.frames += 1;
        Ok(())
    }
}

unsafe extern "C" fn sequence_callback<F>(state: *mut c_void, format: *mut CUVIDEOFORMAT) -> i32
where
    F: for<'frame> FnMut(GpuNv12Frame<'frame>) -> Result<(), String>,
{
    let state = unsafe { &mut *(state as *mut DecoderState<F>) };
    if state.failure.is_some() || format.is_null() {
        return 0;
    }
    match catch_unwind(AssertUnwindSafe(|| state.sequence(unsafe { &*format }))) {
        Ok(Ok(surfaces)) => surfaces,
        Ok(Err(error)) => {
            state.failure = Some(error);
            0
        }
        Err(_) => {
            state.failure = Some("NVDEC sequence callback panicked".into());
            0
        }
    }
}

unsafe extern "C" fn decode_callback<F>(state: *mut c_void, picture: *mut CUVIDPICPARAMS) -> i32
where
    F: for<'frame> FnMut(GpuNv12Frame<'frame>) -> Result<(), String>,
{
    let state = unsafe { &mut *(state as *mut DecoderState<F>) };
    if state.failure.is_some() || picture.is_null() {
        return 0;
    }
    match catch_unwind(AssertUnwindSafe(|| state.decode(picture))) {
        Ok(Ok(())) => 1,
        Ok(Err(error)) => {
            state.failure = Some(error);
            0
        }
        Err(_) => {
            state.failure = Some("NVDEC picture callback panicked".into());
            0
        }
    }
}

unsafe extern "C" fn display_callback<F>(
    state: *mut c_void,
    display: *mut CUVIDPARSERDISPINFO,
) -> i32
where
    F: for<'frame> FnMut(GpuNv12Frame<'frame>) -> Result<(), String>,
{
    let state = unsafe { &mut *(state as *mut DecoderState<F>) };
    if state.failure.is_some() || display.is_null() {
        return 0;
    }
    match catch_unwind(AssertUnwindSafe(|| state.display(unsafe { &*display }))) {
        Ok(Ok(())) => 1,
        Ok(Err(error)) => {
            state.failure = Some(error);
            0
        }
        Err(_) => {
            state.failure = Some("NVDEC display callback panicked".into());
            0
        }
    }
}

struct DecodeRun<F> {
    state: Box<DecoderState<F>>,
    parser: CUvideoparser,
}

impl<F> Drop for DecodeRun<F> {
    fn drop(&mut self) {
        if !self.parser.is_null() {
            unsafe { (self.state.api.cuvidDestroyVideoParser)(self.parser) };
        }
        if !self.state.decoder.is_null() {
            unsafe { (self.state.api.cuvidDestroyDecoder)(self.state.decoder) };
        }
    }
}

pub fn decode_annex_b<R: Read, F>(
    mut source: R,
    codec: NvdecCodec,
    cancelled: &AtomicBool,
    on_frame: F,
) -> Result<NvdecReceipt, String>
where
    F: for<'frame> FnMut(GpuNv12Frame<'frame>) -> Result<(), String>,
{
    mini_cuda::bind_primary_ctx_for_device(0)
        .map_err(|error| format!("NVDEC CUDA primary context: {error}"))?;
    let api = try_init_cuvid().ok_or("NVIDIA NVDEC driver library is unavailable")?;
    let mut state = Box::new(DecoderState {
        api,
        decoder: ptr::null_mut(),
        codec,
        frames: 0,
        width: 0,
        height: 0,
        progressive: false,
        surfaces: 0,
        callback: on_frame,
        failure: None,
    });
    let mut params: CUVIDPARSERPARAMS = unsafe { std::mem::zeroed() };
    params.CodecType = codec.native();
    params.ulMaxNumDecodeSurfaces = 1;
    params.ulMaxDisplayDelay = 0;
    params.pUserData = (&mut *state as *mut DecoderState<_>).cast();
    params.pfnSequenceCallback = Some(sequence_callback::<F>);
    params.pfnDecodePicture = Some(decode_callback::<F>);
    params.pfnDisplayPicture = Some(display_callback::<F>);
    let mut parser = ptr::null_mut();
    let status = unsafe { (api.cuvidCreateVideoParser)(&mut parser, &mut params) };
    if status != CUresult::CUDA_SUCCESS {
        return Err(format!("Create NVDEC video parser: {status:?}"));
    }
    let mut run = DecodeRun { state, parser };
    let mut bytes = [0_u8; MAX_BYTES_PER_PACKET];
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let count = source
            .read(&mut bytes)
            .map_err(|error| format!("Read compressed source: {error}"))?;
        if count == 0 {
            break;
        }
        let mut packet = CUVIDSOURCEDATAPACKET {
            flags: 0,
            payload_size: count as _,
            payload: bytes.as_ptr(),
            timestamp: 0,
        };
        let status = unsafe { (api.cuvidParseVideoData)(run.parser, &mut packet) };
        if let Some(error) = run.state.failure.take() {
            return Err(error);
        }
        if status != CUresult::CUDA_SUCCESS {
            return Err(format!("NVDEC parse video data: {status:?}"));
        }
    }
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let mut eos = CUVIDSOURCEDATAPACKET {
        flags: CUVID_PKT_ENDOFSTREAM,
        payload_size: 0,
        payload: ptr::null(),
        timestamp: 0,
    };
    let status = unsafe { (api.cuvidParseVideoData)(run.parser, &mut eos) };
    if let Some(error) = run.state.failure.take() {
        return Err(error);
    }
    if status != CUresult::CUDA_SUCCESS || run.state.frames == 0 {
        return Err(format!(
            "NVDEC did not complete a decoded frame: {status:?}"
        ));
    }
    Ok(NvdecReceipt {
        frames: run.state.frames,
        width: run.state.width,
        height: run.state.height,
    })
}

struct CompressedStream<'a> {
    receiver: std::sync::mpsc::Receiver<std::io::Result<Vec<u8>>>,
    cancelled: &'a AtomicBool,
    pending: Vec<u8>,
    offset: usize,
}

impl Read for CompressedStream<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let started = std::time::Instant::now();
        loop {
            if self.cancelled.load(Ordering::Acquire) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "cancelled",
                ));
            }
            if self.offset < self.pending.len() {
                let size = (self.pending.len() - self.offset).min(buffer.len());
                buffer[..size].copy_from_slice(&self.pending[self.offset..self.offset + size]);
                self.offset += size;
                return Ok(size);
            }
            match self
                .receiver
                .recv_timeout(std::time::Duration::from_millis(100))
            {
                Ok(Ok(bytes)) if bytes.is_empty() => return Ok(0),
                Ok(Ok(bytes)) => {
                    self.pending = bytes;
                    self.offset = 0;
                }
                Ok(Err(error)) => return Err(error),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Ok(0),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                    if started.elapsed() > std::time::Duration::from_secs(120) =>
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "Compressed FFmpeg demuxer stalled",
                    ));
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
}

pub fn decode_video_file<F>(
    source: &std::path::Path,
    codec: NvdecCodec,
    cancelled: &AtomicBool,
    on_frame: F,
) -> Result<NvdecReceipt, String>
where
    F: for<'frame> FnMut(GpuNv12Frame<'frame>) -> Result<(), String>,
{
    if cancelled.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    if !source.is_file() {
        return Err("NVDEC input source must exist".into());
    }
    let (filter, format) = match codec {
        NvdecCodec::H264 => ("h264_mp4toannexb", "h264"),
        NvdecCodec::Hevc => ("hevc_mp4toannexb", "hevc"),
    };
    let mut command = std::process::Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(source)
        .args([
            "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "copy", "-bsf:v", filter, "-f", format,
            "pipe:1",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    let mut process = command
        .spawn()
        .map_err(|error| format!("Start NVDEC compressed FFmpeg demuxer: {error}"))?;
    let stdout = process
        .stdout
        .take()
        .ok_or("NVDEC compressed stdout unavailable")?;
    let (sender, receiver) = std::sync::mpsc::sync_channel(2);
    let worker = std::thread::spawn(move || {
        let mut stream = stdout;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let chunk = match stream.read(&mut buffer) {
                Ok(n) => Ok(buffer[..n].to_vec()),
                Err(error) => Err(error),
            };
            let done = match &chunk {
                Ok(bytes) => bytes.is_empty(),
                Err(_) => true,
            };
            if sender.send(chunk).is_err() || done {
                break;
            }
        }
    });
    let decoder = CompressedStream {
        receiver,
        cancelled,
        pending: Vec::new(),
        offset: 0,
    };
    let result = decode_annex_b(decoder, codec, cancelled, on_frame);
    if result.is_err() || cancelled.load(Ordering::Acquire) {
        let _ = process.kill();
    }
    let status = process
        .wait()
        .map_err(|error| format!("Wait NVDEC compressed demuxer: {error}"))?;
    let _ = worker.join();
    let receipt = result?;
    if !status.success() {
        return Err(format!("NVDEC compressed demuxer exited with {status}"));
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        io::Cursor,
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn ffmpeg_compressed_demuxer_keeps_original_pixels_in_nvdec_vram() {
        if try_init_cuvid().is_none() || try_init_driver().is_none() {
            return;
        }
        let cancelled = AtomicBool::new(false);
        let source = std::env::temp_dir().join(format!(
            "volley-zero-copy-{}.mp4",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let result = (|| {
            let status = Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=320x180:rate=30:duration=0.6",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                ])
                .arg(&source)
                .status()
                .map_err(|error| error.to_string())?;
            if !status.success() {
                return Err("Create NVDEC compressed source failed".into());
            }
            let mut frames = 0;
            let summary = decode_video_file(&source, NvdecCodec::H264, &cancelled, |view| {
                assert_eq!(view.frame_index, frames);
                assert_eq!((view.width, view.height), (320, 180));
                assert!(view.device_ptr != 0);
                frames += 1;
                Ok(())
            })?;
            assert_eq!(frames, 18);
            assert_eq!(summary.frames, 18);
            Ok::<(), String>(())
        })();
        let _ = fs::remove_file(source);
        result.unwrap();
    }

    #[test]
    fn nvdec_yields_original_nv12_vram_surfaces_without_host_rgb() {
        let cancelled = AtomicBool::new(false);
        if try_init_cuvid().is_none() || try_init_driver().is_none() {
            return;
        }
        let temp = std::env::temp_dir().join(format!(
            "volley-nvdec-{}-{}.h264",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let result = (|| {
            let status = Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=320x180:rate=30:duration=1.0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-f",
                    "h264",
                    "-y",
                ])
                .arg(&temp)
                .status()
                .map_err(|error| error.to_string())?;
            if !status.success() {
                return Err("Unable to create NVDEC H264 synthetic fixture".into());
            }
            let bytes = fs::read(&temp).map_err(|error| error.to_string())?;
            let mut observed = 0u64;
            let receipt =
                decode_annex_b(Cursor::new(bytes), NvdecCodec::H264, &cancelled, |frame| {
                    assert_eq!(frame.frame_index, observed);
                    assert_eq!((frame.width, frame.height), (320, 180));
                    assert!(frame.device_ptr != 0);
                    assert!(frame.pitch >= frame.width);
                    observed += 1;
                    Ok(())
                })?;
            assert_eq!(receipt.frames, 30);
            assert_eq!(observed, 30);
            Ok::<(), String>(())
        })();
        let _ = fs::remove_file(&temp);
        result.unwrap();
    }
}
