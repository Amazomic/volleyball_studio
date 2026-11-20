use mini_cuda::{driver::try_init_driver, CUresult};
use mini_nvcodec::{
    cuvid_sys::{
        cudaVideoCodec, cudaVideoCreate_PreferCUVID, cudaVideoDeinterlaceMode,
        cudaVideoSurfaceFormat, CUvideodecoder, CUvideoparser, CUVIDDECODECAPS,
        CUVIDDECODECREATEINFO, CUVIDEOFORMAT, CUVIDPARSERDISPINFO, CUVIDPARSERPARAMS,
        CUVIDPICPARAMS, CUVIDPROCPARAMS, CUVIDRECT, CUVIDSOURCEDATAPACKET, CUVID_PKT_ENDOFSTREAM,
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
    Vp8,
    Vp9,
    Av1,
    Mpeg1,
    Mpeg2,
    Mpeg4,
    Vc1,
    Mjpeg,
}

impl NvdecCodec {
    fn native(self) -> cudaVideoCodec {
        match self {
            Self::H264 => cudaVideoCodec::H264,
            Self::Hevc => cudaVideoCodec::HEVC,
            Self::Vp8 => cudaVideoCodec::VP8,
            Self::Vp9 => cudaVideoCodec::VP9,
            Self::Av1 => cudaVideoCodec::AV1,
            Self::Mpeg1 => cudaVideoCodec::MPEG1,
            Self::Mpeg2 => cudaVideoCodec::MPEG2,
            Self::Mpeg4 => cudaVideoCodec::MPEG4,
            Self::Vc1 => cudaVideoCodec::VC1,
            Self::Mjpeg => cudaVideoCodec::JPEG,
        }
    }
}

pub fn codec_supported(codec: NvdecCodec, bit_depth: u8) -> bool {
    codec_supported_format(codec, bit_depth, 1)
}

pub fn codec_supported_format(codec: NvdecCodec, bit_depth: u8, chroma: u8) -> bool {
    if !matches!(bit_depth, 8 | 10 | 12)
        || !matches!(chroma, 1..=3)
        || mini_cuda::bind_primary_ctx_for_device(0).is_err()
    {
        return false;
    }
    let Some(api) = try_init_cuvid() else {
        return false;
    };
    let mut caps: CUVIDDECODECAPS = unsafe { std::mem::zeroed() };
    caps.eCodecType = codec.native();
    caps.eChromaFormat.0 = i32::from(chroma);
    caps.nBitDepthMinus8 = u32::from(bit_depth - 8);
    let status = unsafe { (api.cuvidGetDecoderCaps)(&mut caps) };
    let output = match (chroma, bit_depth == 8) {
        (1, true) => cudaVideoSurfaceFormat::NV12,
        (1, false) => cudaVideoSurfaceFormat::P016,
        (2, true) => cudaVideoSurfaceFormat::NV16,
        (2, false) => cudaVideoSurfaceFormat::P216,
        (3, true) => cudaVideoSurfaceFormat::YUV444,
        (3, false) => cudaVideoSurfaceFormat::YUV444_16BIT,
        _ => return false,
    };
    status == CUresult::CUDA_SUCCESS
        && caps.bIsSupported == 1
        && (caps.nOutputFormatMask & (1 << output.0 as u32)) != 0
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
    pub bit_depth: u8,
    pub chroma: u8,
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
    bit_depth: u8,
    chroma: u8,
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
            || !matches!(format.chroma_format.0, 1..=3)
            || !matches!(format.bit_depth_luma_minus8, 0 | 2 | 4)
        {
            return Err("NVDEC source chroma or bit depth is unsupported".into());
        }
        let chroma = format.chroma_format.0 as u8;
        let depth = format.bit_depth_luma_minus8 + 8;
        let output_format = match (chroma, depth == 8) {
            (1, true) => cudaVideoSurfaceFormat::NV12,
            (1, false) => cudaVideoSurfaceFormat::P016,
            (2, true) => cudaVideoSurfaceFormat::NV16,
            (2, false) => cudaVideoSurfaceFormat::P216,
            (3, true) => cudaVideoSurfaceFormat::YUV444,
            (3, false) => cudaVideoSurfaceFormat::YUV444_16BIT,
            _ => return Err("NVDEC surface format not supported".into()),
        };
        let (coded_width, coded_height) =
            (format.coded_width as usize, format.coded_height as usize);
        let area = format.display_area;
        let width = (area.right - area.left) as usize;
        let height = (area.bottom - area.top) as usize;
        if width == 0
            || height == 0
            || width > coded_width
            || height > coded_height
            || !width.is_multiple_of(2)
            || !height.is_multiple_of(2)
            || width > i16::MAX as usize
            || height > i16::MAX as usize
        {
            return Err("NVDEC sequence dimensions or crop are not supported".into());
        }
        let surfaces = format.min_num_decode_surfaces.max(8);
        if surfaces > MAX_SURFACES {
            return Err("NVDEC requested more than 32 decode surfaces".into());
        }
        let mut caps: CUVIDDECODECAPS = unsafe { std::mem::zeroed() };
        caps.eCodecType = self.codec.native();
        caps.eChromaFormat = format.chroma_format;
        caps.nBitDepthMinus8 = format.bit_depth_luma_minus8 as _;
        let status = unsafe { (self.api.cuvidGetDecoderCaps)(&mut caps) };
        if status != CUresult::CUDA_SUCCESS || caps.bIsSupported != 1 {
            return Err(format!("NVDEC decoder capability probe failed: {status:?}"));
        }
        if (caps.nOutputFormatMask & (1 << output_format.0 as u32)) == 0 {
            return Err("NVDEC cannot output the stream bit depth on this GPU".into());
        }
        if coded_width < caps.nMinWidth as usize
            || coded_height < caps.nMinHeight as usize
            || coded_width > caps.nMaxWidth as usize
            || coded_height > caps.nMaxHeight as usize
        {
            return Err("NVDEC codec cannot decode source dimensions on this GPU".into());
        }
        let mut info: CUVIDDECODECREATEINFO = unsafe { std::mem::zeroed() };
        info.ulWidth = coded_width as _;
        info.ulHeight = coded_height as _;
        info.ulNumDecodeSurfaces = surfaces as _;
        info.CodecType = self.codec.native();
        info.ChromaFormat = format.chroma_format;
        info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
        info.bitDepthMinus8 = format.bit_depth_luma_minus8 as _;
        info.OutputFormat = output_format;
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
        self.progressive = format.progressive_sequence != 0;
        self.bit_depth = depth;
        self.chroma = chroma;
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
            bit_depth: self.bit_depth,
            chroma: self.chroma,
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
        bit_depth: 0,
        chroma: 0,
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
    let ivf = matches!(codec, NvdecCodec::Vp8 | NvdecCodec::Vp9 | NvdecCodec::Av1);
    if ivf {
        let mut header = [0u8; 32];
        source
            .read_exact(&mut header)
            .map_err(|error| format!("Read compressed IVF header: {error}"))?;
        if &header[..4] != b"DKIF" {
            return Err("NVDEC IVF source does not have a valid signature".into());
        }
        let expected = match codec {
            NvdecCodec::Av1 => b"AV01",
            NvdecCodec::Vp9 => b"VP90",
            _ => b"VP80",
        };
        if &header[8..12] != expected {
            return Err("NVDEC IVF codec signature mismatch".into());
        }
    }
    let mut bytes = Vec::<u8>::new();
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        if ivf {
            let mut header = [0u8; 12];
            let size = source
                .read(&mut header[..1])
                .map_err(|error| format!("Read IVF packet: {error}"))?;
            if size == 0 {
                break;
            }
            source
                .read_exact(&mut header[1..])
                .map_err(|error| format!("Read IVF packet header: {error}"))?;
            let length = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
            if !(1..=64 * 1024 * 1024).contains(&length) {
                return Err("NVDEC IVF packet size outside accepted bounds".into());
            }
            bytes.resize(length, 0);
            source
                .read_exact(&mut bytes)
                .map_err(|error| format!("Read IVF packet payload: {error}"))?;
        } else {
            bytes.resize(MAX_BYTES_PER_PACKET, 0);
            let count = source
                .read(&mut bytes)
                .map_err(|error| format!("Read compressed source: {error}"))?;
            if count == 0 {
                break;
            }
            bytes.truncate(count);
        }
        let mut packet = CUVIDSOURCEDATAPACKET {
            flags: 0,
            payload_size: bytes.len() as _,
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

pub fn display_timestamps(source: &std::path::Path) -> Result<Vec<f64>, String> {
    use std::io::Read;
    let mut child = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
        ])
        .arg(source)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|error| format!("Read NVDEC display timestamps: {error}"))?;
    let mut pipe = child
        .stdout
        .take()
        .ok_or("Missing packet timestamp stream")?;
    let reader = std::thread::spawn(move || {
        let mut values = Vec::new();
        let mut buffer = [0u8; 8192];
        loop {
            let n = pipe.read(&mut buffer).map_err(|error| error.to_string())?;
            if n == 0 {
                break;
            }
            if values.len() > 96 * 1024 * 1024 {
                return Err("NVDEC timestamp metadata exceeds safe bounds".into());
            }
            values.extend_from_slice(&buffer[..n]);
        }
        Ok::<_, String>(values)
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Err("NVDEC timestamp probe timed out".into());
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(25)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Err(format!("NVDEC timestamp probe: {error}"));
            }
        }
    };
    let bytes = reader
        .join()
        .map_err(|_| "NVDEC timestamp worker panicked")??;
    if !status.success() {
        return Err("NVDEC timestamp probe failed".into());
    }
    let mut times = Vec::new();
    for line in bytes.split(|&byte| byte == b'\n') {
        let trimmed = std::str::from_utf8(line)
            .map_err(|_| "Invalid timestamp encoding")?
            .trim();
        if trimmed.is_empty() {
            continue;
        }
        let time = trimmed
            .parse::<f64>()
            .map_err(|_| "NVDEC packet has no presentation timestamp")?;
        if !time.is_finite() || time < -3600.0 || time > 48.0 * 3600.0 + 3600.0 {
            return Err("NVDEC presentation time outside media bounds".into());
        }
        times.push(time);
    }
    if times.is_empty() {
        return Err("NVDEC source has no display timestamps".into());
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let first = times[0];
    for timestamp in &mut times {
        *timestamp -= first;
    }
    Ok(times)
}

pub fn closest_display_frame(times: &[f64], time_sec: f64) -> Option<u64> {
    if times.is_empty() || !time_sec.is_finite() || time_sec < 0.0 {
        return None;
    }
    let next = times.partition_point(|value| *value < time_sec);
    let frame = if next == 0 {
        0
    } else if next == times.len() || time_sec - times[next - 1] <= times[next] - time_sec {
        next - 1
    } else {
        next
    };
    Some(frame as u64)
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
        NvdecCodec::H264 => (Some("h264_mp4toannexb"), "h264"),
        NvdecCodec::Hevc => (Some("hevc_mp4toannexb"), "hevc"),
        NvdecCodec::Vp8 | NvdecCodec::Vp9 | NvdecCodec::Av1 => (None, "ivf"),
        NvdecCodec::Mpeg1 => (None, "mpeg1video"),
        NvdecCodec::Mpeg2 => (None, "mpeg2video"),
        NvdecCodec::Mpeg4 => (Some("dump_extra"), "m4v"),
        NvdecCodec::Vc1 => (None, "vc1"),
        NvdecCodec::Mjpeg => (None, "mjpeg"),
    };
    let mut command = std::process::Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(source)
        .args(["-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "copy"]);
    if let Some(filter) = filter {
        command.args(["-bsf:v", filter]);
    }
    command
        .args(["-f", format, "pipe:1"])
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
    fn hevc_10bit_nvdec_delivers_p016_gpu_surfaces() {
        if !codec_supported(NvdecCodec::Hevc, 10) {
            return;
        }
        let source = std::env::temp_dir().join(format!(
            "volley-p016-{}-{}.mkv",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let output = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x180:rate=25:duration=0.8",
                "-c:v",
                "libx265",
                "-preset",
                "ultrafast",
                "-x265-params",
                "pools=1:frame-threads=1:log-level=error",
                "-pix_fmt",
                "yuv420p10le",
                "-y",
            ])
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let canceled = AtomicBool::new(false);
        let pts = display_timestamps(&source).unwrap();
        let mut count = 0;
        let result = decode_video_file(&source, NvdecCodec::Hevc, &canceled, |surface| {
            assert_eq!(surface.frame_index, count);
            assert_eq!(surface.bit_depth, 10);
            assert!(surface.pitch >= surface.width * 2);
            count += 1;
            Ok(())
        });
        let _ = fs::remove_file(source);
        let summary = result.unwrap();
        assert_eq!(summary.frames, 20);
        assert_eq!(pts.len(), 20);
    }

    #[test]
    fn nvdec_legacy_supported_codecs_use_vram_surfaces() {
        let cancelled = AtomicBool::new(false);
        for (codec, encoder) in [
            (NvdecCodec::Vp8, "libvpx"),
            (NvdecCodec::Mpeg1, "mpeg1video"),
            (NvdecCodec::Mpeg2, "mpeg2video"),
            (NvdecCodec::Mpeg4, "mpeg4"),
            (NvdecCodec::Mjpeg, "mjpeg"),
        ] {
            if !codec_supported(codec, 8) {
                continue;
            }
            let source = std::env::temp_dir().join(format!(
                "volley-extra-codec-{}-{}-{}.mkv",
                std::process::id(),
                encoder,
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
            ));
            let output = Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=320x180:rate=25:duration=0.4",
                    "-c:v",
                    encoder,
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                ])
                .arg(&source)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let mut seen = 0u64;
            let result = decode_video_file(&source, codec, &cancelled, |frame| {
                assert_eq!(frame.frame_index, seen);
                assert_eq!(frame.chroma, 1);
                assert_eq!(frame.bit_depth, 8);
                seen += 1;
                Ok(())
            });
            let _ = fs::remove_file(source);
            assert_eq!(
                result
                    .unwrap_or_else(|error| panic!("{codec:?}: {error}"))
                    .frames,
                10,
                "{codec:?}"
            );
        }
    }

    #[test]
    fn nvdec_handles_yuv422_and_yuv444_on_compatible_gpu() {
        let canceled = AtomicBool::new(false);
        for (codec, depth, chroma, encoder, pix_fmt) in [
            (NvdecCodec::Hevc, 10, 2, "libx265", "yuv422p10le"),
            (NvdecCodec::H264, 8, 3, "libx264", "yuv444p"),
        ] {
            if !codec_supported_format(codec, depth, chroma) {
                eprintln!(
                    "{codec:?} {depth}bit chroma {chroma} not supported by current NVDEC GPU"
                );
                continue;
            }
            let source = std::env::temp_dir().join(format!(
                "volley-chroma-{}-{}-{}.mkv",
                std::process::id(),
                pix_fmt,
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
            ));
            let output = Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=320x180:rate=25:duration=0.4",
                    "-c:v",
                    encoder,
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    pix_fmt,
                    "-y",
                ])
                .arg(&source)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let mut frames = 0;
            let result = decode_video_file(&source, codec, &canceled, |frame| {
                assert_eq!(frame.chroma, chroma);
                assert_eq!(frame.bit_depth, depth);
                assert_eq!(frame.frame_index, frames);
                frames += 1;
                Ok(())
            });
            let _ = fs::remove_file(source);
            assert_eq!(result.unwrap().frames, 10);
        }
    }

    #[test]
    fn ivf_vp9_and_av1_are_decoded_by_hardware_when_supported() {
        let canceled = AtomicBool::new(false);
        for (codec, encoder) in [
            (NvdecCodec::Vp9, "libvpx-vp9"),
            (NvdecCodec::Av1, "libaom-av1"),
        ] {
            if !codec_supported(codec, 8) {
                continue;
            }
            let source = std::env::temp_dir().join(format!(
                "volley-ivf-{}-{}-{}.mkv",
                std::process::id(),
                encoder,
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
            ));
            let output = Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=320x180:rate=24:duration=0.5",
                    "-c:v",
                    encoder,
                    "-cpu-used",
                    "8",
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                ])
                .arg(&source)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}: {}",
                encoder,
                String::from_utf8_lossy(&output.stderr)
            );
            let pts = display_timestamps(&source).unwrap();
            let mut seen = 0u64;
            let result = decode_video_file(&source, codec, &canceled, |frame| {
                assert_eq!(frame.frame_index, seen);
                assert_eq!(frame.bit_depth, 8);
                seen += 1;
                Ok(())
            });
            let _ = fs::remove_file(source);
            let summary = result.unwrap();
            assert_eq!(seen, 12);
            assert_eq!(pts.len() as u64, summary.frames);
        }
    }

    #[test]
    fn exact_presentation_frame_sampling_handles_24_and_60_fps() {
        let frames24 = (0..24).map(|index| index as f64 / 24.0).collect::<Vec<_>>();
        let frames60 = (0..60).map(|index| index as f64 / 60.0).collect::<Vec<_>>();
        let samples24 = (0..30)
            .map(|index| closest_display_frame(&frames24, index as f64 / 30.0).unwrap())
            .collect::<Vec<_>>();
        let samples60 = (0..30)
            .map(|index| closest_display_frame(&frames60, index as f64 / 30.0).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(samples24[0], 0);
        assert_eq!(samples24[29], 23);
        assert!(samples24.windows(2).any(|pair| pair[0] == pair[1]));
        assert_eq!(samples60[0], 0);
        assert_eq!(samples60[1], 2);
        assert_eq!(samples60[29], 58);
        let vfr = [0.0, 0.015, 0.071, 0.102];
        assert_eq!(closest_display_frame(&vfr, 0.033), Some(1));
        assert_eq!(closest_display_frame(&vfr, 0.066), Some(2));
    }

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
