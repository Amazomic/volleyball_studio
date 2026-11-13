use ort::{execution_providers::CUDAExecutionProvider, session::Session, value::Tensor};
use std::{
    env,
    fs::File,
    io::{Read, Seek, SeekFrom},
    time::Instant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model = args.next().ok_or("missing model")?;
    let mut source = File::open(args.next().ok_or("missing RGB frames")?)?;
    ort::init().commit()?;
    let mut session = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
        .commit_from_file(model)?;
    let mut binding = session.create_binding()?;
    binding.bind_output(
        "simcc_x",
        Tensor::<f32>::from_array((vec![8_i64, 17, 384], vec![0_f32; 8 * 17 * 384]))?,
    )?;
    binding.bind_output(
        "simcc_y",
        Tensor::<f32>::from_array((vec![8_i64, 17, 512], vec![0_f32; 8 * 17 * 512]))?,
    )?;
    let mut rgb = vec![0_u8; 640 * 360 * 3];
    let mut normal = 0.0;
    let mut bound = 0.0;
    let mut max_diff = 0_f32;
    let mut frames = 0_usize;
    let mut passes = 0;
    loop {
        match source.read_exact(&mut rgb) {
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                passes += 1;
                if passes == 6 {
                    break;
                }
                source.seek(SeekFrom::Start(0))?;
                continue;
            }
            Err(e) => return Err(e.into()),
        }
        let mut data = vec![0_f32; 8 * 3 * 256 * 192];
        for (i, v) in data.iter_mut().enumerate() {
            *v = rgb[(i * 37) % rgb.len()] as f32 / 255.0;
        }
        let tensor = Tensor::from_array((vec![8_i64, 3, 256, 192], data.clone()))?;
        let start = Instant::now();
        let outputs = session.run(ort::inputs!["input" => tensor])?;
        let x = outputs["simcc_x"].try_extract_tensor::<f32>()?.1.to_vec();
        let y = outputs["simcc_y"].try_extract_tensor::<f32>()?.1.to_vec();
        drop(outputs);
        if frames >= 5 {
            normal += start.elapsed().as_secs_f64();
        }
        let tensor = Tensor::from_array((vec![8_i64, 3, 256, 192], data))?;
        let start = Instant::now();
        binding.bind_input("input", &tensor)?;
        let outputs = session.run_binding(&binding)?;
        let bx = outputs["simcc_x"].try_extract_tensor::<f32>()?.1.to_vec();
        let by = outputs["simcc_y"].try_extract_tensor::<f32>()?.1.to_vec();
        if frames >= 5 {
            bound += start.elapsed().as_secs_f64();
        }
        for (&a, &b) in x.iter().chain(&y).zip(bx.iter().chain(&by)) {
            max_diff = max_diff.max((a - b).abs());
        }
        frames += 1;
    }
    println!("frames={frames} measured_frames={} batch=8 ordinary_seconds={normal:.3} binding_seconds={bound:.3} max_abs_diff={max_diff:.8}", frames.saturating_sub(5));
    Ok(())
}
