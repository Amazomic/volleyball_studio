use std::{
    env,
    fs::File,
    io::{BufWriter, Read, Write},
    time::Instant,
};
use volley_inference::{person::PersonEngine, Device};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let repo = args.next().ok_or("missing repository path")?;
    let input = args.next().ok_or("missing RGB video path")?;
    let mut source = File::open(input)?;
    let mut output = args
        .next()
        .map(File::create)
        .transpose()?
        .map(BufWriter::new);
    let mut engine = PersonEngine::load(repo.as_ref(), Device::Cuda, 8, 0)?;
    let frame_size = 640 * 360 * 3;
    let mut frame = vec![0; frame_size * 9];
    let mut frames = 0u64;
    let mut observations = 0usize;
    let start = Instant::now();
    loop {
        let mut size = 0;
        while size < frame.len() {
            let n = source.read(&mut frame[size..])?;
            if n == 0 {
                break;
            }
            size += n;
        }
        if size == 0 {
            break;
        }
        if size % frame_size != 0 {
            return Err("incomplete RGB frame".into());
        }
        engine.detect_frames(&frame[..size], |_, poses| {
            observations += poses.len();
            if let Some(writer) = output.as_mut() {
                writeln!(writer, "{} {:?}", frames, poses).map_err(|e| e.to_string())?;
            }
            frames += 1;
            Ok(())
        })?;
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "frames={frames} observations={observations} elapsed_seconds={elapsed:.3} fps={:.3}",
        frames as f64 / elapsed
    );
    Ok(())
}
