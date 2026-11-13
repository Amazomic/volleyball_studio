use ort::{execution_providers::CUDAExecutionProvider, session::Session, value::Tensor};
use std::{
    env,
    fs::File,
    io::{Read, Seek, SeekFrom},
    time::Instant,
};
use volley_inference::person::prepare_views;

fn load(path: &str) -> Result<Session, Box<dyn std::error::Error>> {
    Ok(Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
        .commit_from_file(path)?)
}

fn run(
    session: &mut Session,
    input: Vec<f32>,
    batch: usize,
) -> Result<Vec<Vec<[f32; 5]>>, Box<dyn std::error::Error>> {
    let tensor = Tensor::from_array((vec![batch as i64, 3, 320, 320], input))?;
    let output = session.run(ort::inputs!["input" => tensor])?;
    let (shape, dets) = output["dets"].try_extract_tensor::<f32>()?;
    let (labels_shape, labels) = output["labels"].try_extract_tensor::<i64>()?;
    if shape.len() != 3
        || shape[0] != batch as i64
        || shape[2] != 5
        || labels_shape.as_ref() != [batch as i64, shape[1]]
    {
        return Err("unexpected detector output".into());
    }
    let n = shape[1] as usize;
    let mut result = Vec::with_capacity(batch);
    for i in 0..batch {
        let mut rows = Vec::new();
        for (row, label) in dets[i * n * 5..(i + 1) * n * 5]
            .chunks_exact(5)
            .zip(&labels[i * n..(i + 1) * n])
        {
            if *label == 0 && row[4] >= 0.12 && row.iter().all(|v| v.is_finite()) {
                rows.push([row[0], row[1], row[2], row[3], row[4]]);
            }
        }
        result.push(rows);
    }
    Ok(result)
}

fn compare(
    a: &[Vec<[f32; 5]>],
    b: &[Vec<[f32; 5]>],
    count_changes: &mut usize,
    max_delta: &mut f32,
) {
    for (left, right) in a.iter().zip(b) {
        if left.len() != right.len() {
            *count_changes += 1;
        }
        for (x, y) in left.iter().zip(right) {
            for (u, v) in x.iter().zip(y) {
                *max_delta = max_delta.max((u - v).abs());
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let original = args.next().ok_or("missing original model")?;
    let patched = args.next().ok_or("missing patched model")?;
    let mut source = File::open(args.next().ok_or("missing frames")?)?;
    ort::init().commit()?;
    let mut a = load(&original)?;
    let mut b = load(&patched)?;
    let mut frame = vec![0_u8; 640 * 360 * 3];
    let (mut frames, mut single_changes, mut batch_changes) = (0_usize, 0_usize, 0_usize);
    let (mut single_delta, mut batch_delta, mut original_time, mut batch_time) =
        (0_f32, 0_f32, 0_f64, 0_f64);
    let mut passes = 0;
    loop {
        match source.read_exact(&mut frame) {
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                passes += 1;
                if passes == 2 {
                    break;
                }
                source.seek(SeekFrom::Start(0))?;
                continue;
            }
            Err(e) => return Err(e.into()),
        }
        let views = prepare_views(&frame)?;
        let start = Instant::now();
        let mut old = Vec::new();
        for view in &views {
            old.extend(run(&mut a, view.clone(), 1)?);
        }
        original_time += start.elapsed().as_secs_f64();
        let mut patched_single = Vec::new();
        for view in &views {
            patched_single.extend(run(&mut b, view.clone(), 1)?);
        }
        let start = Instant::now();
        let packed = views.into_iter().flatten().collect();
        let patched_batch = run(&mut b, packed, 4)?;
        batch_time += start.elapsed().as_secs_f64();
        if batch_changes == 0 {
            for (index, (left, right)) in old.iter().zip(&patched_batch).enumerate() {
                if left.len() != right.len() {
                    println!(
                        "first_count_difference frame={frames} view={index} baseline={} batch={}",
                        left.len(),
                        right.len()
                    );
                    break;
                }
            }
        }
        compare(
            &old,
            &patched_single,
            &mut single_changes,
            &mut single_delta,
        );
        compare(&old, &patched_batch, &mut batch_changes, &mut batch_delta);
        frames += 1;
    }
    println!("frames={frames} original_s={original_time:.3} batch_s={batch_time:.3} single_count_changes={single_changes} single_max_delta={single_delta:.6} batch_count_changes={batch_changes} batch_max_delta={batch_delta:.6}");
    Ok(())
}
