use std::{
    env,
    fs::File,
    io::{BufWriter, Read, Write},
    path::Path,
    time::Instant,
};
use volley_inference::{person::PersonEngine, Device};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let repository = args.next().ok_or("missing repository")?;
    let input = args.next().ok_or("missing RGB24 video")?;
    let stride: u64 = args.next().ok_or("missing stride")?.parse()?;
    let destination = args.next().ok_or("missing output")?;
    if !matches!(stride, 1 | 2 | 5 | 10) {
        return Err("benchmark stride must be 1, 2, 5, or 10".into());
    }

    let mut reader = File::open(input)?;
    let mut writer = BufWriter::new(File::create(destination)?);
    let mut engine =
        PersonEngine::load_for_gpu_evidence(Path::new(&repository), Device::Cuda, 8, 0)?;
    let mut frame = vec![0u8; 640 * 360 * 3];
    let (mut frames, mut samples, mut detections) = (0u64, 0u64, 0u64);
    let (mut infer_secs, start) = (0f64, Instant::now());

    loop {
        let mut read = 0;
        while read < frame.len() {
            let n = reader.read(&mut frame[read..])?;
            if n == 0 {
                break;
            }
            read += n;
        }
        if read == 0 {
            break;
        }
        if read != frame.len() {
            return Err("incomplete RGB frame".into());
        }
        if frames % stride == 0 {
            let started = Instant::now();
            let observed = engine.detect(&frame)?;
            infer_secs += started.elapsed().as_secs_f64();
            let boxes: Vec<_> = observed.iter().map(|p| p.bbox).collect();
            let scores: Vec<_> = observed.iter().map(|p| p.score).collect();
            writeln!(
                writer,
                "{}",
                serde_json::json!({"frame":frames,"boxes":boxes,"scores":scores})
            )?;
            detections += observed.len() as u64;
            samples += 1;
        }
        frames += 1;
    }
    writer.flush()?;
    let total_secs = start.elapsed().as_secs_f64();
    println!(
        "stride={} frames={} samples={} detected_boxes={} inference_seconds={:.3} total_seconds={:.3} effective_source_fps={:.2}",
        stride, frames, samples, detections, infer_secs, total_secs, frames as f64 / total_secs
    );
    Ok(())
}
