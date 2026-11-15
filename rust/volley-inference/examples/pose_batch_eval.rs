use std::{
    env,
    fs::File,
    io::Read,
    time::{Duration, Instant},
};
use volley_inference::{
    person::{skeleton_box, PersonEngine},
    pose::PoseEngine,
    Device,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let repo = args.next().ok_or("missing repository")?;
    let mut file = File::open(args.next().ok_or("missing frames")?)?;
    let mut engine = PoseEngine::load(repo.as_ref(), Device::Cuda, 8, 0)?;
    let mut detector = PersonEngine::load(repo.as_ref(), Device::Cuda, 8, 0)?;
    let mut frame = vec![0; 640 * 360 * 3];
    let (mut count, mut joints, mut changed, mut crossed) = (0_u64, 0_u64, 0_u64, 0_u64);
    let (mut max_xy, mut max_score) = (0_f32, 0_f32);
    let (mut single_time, mut batch_time) = (Duration::ZERO, Duration::ZERO);
    let (mut matched_people, mut acceptance_changes, mut changed_confident) = (0_u64, 0_u64, 0_u64);
    let (mut largest_frame, mut largest_joint, mut largest_a, mut largest_b) =
        (0_u64, 0_usize, 0_f32, 0_f32);
    loop {
        match file.read_exact(&mut frame) {
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let boxes: Vec<_> = detector
            .detect(&frame)?
            .into_iter()
            .filter(|person| {
                (person.bbox[2] - person.bbox[0]) * 640.0 >= 8.0
                    && (person.bbox[3] - person.bbox[1]) * 360.0 >= 20.0
            })
            .take(8)
            .map(|person| person.bbox)
            .collect();
        if boxes.is_empty() {
            count += 1;
            continue;
        }
        let start = Instant::now();
        let individual: Vec<_> = boxes
            .iter()
            .map(|&bbox| engine.infer(&frame, bbox))
            .collect::<Result<_, _>>()?;
        single_time += start.elapsed();
        let start = Instant::now();
        let batched = engine.infer_batch(&frame, &boxes)?;
        batch_time += start.elapsed();
        for (a, b) in individual.iter().zip(&batched) {
            matched_people += 1;
            acceptance_changes += u64::from(skeleton_box(a).is_some() != skeleton_box(b).is_some());
        }
        for (index, (a, b)) in individual
            .iter()
            .flatten()
            .zip(batched.iter().flatten())
            .enumerate()
        {
            let delta = (a.x - b.x).abs().max((a.y - b.y).abs());
            if delta > max_xy {
                max_xy = delta;
                largest_frame = count;
                largest_joint = index % 17;
                largest_a = a.score;
                largest_b = b.score;
            }
            max_score = max_score.max((a.score - b.score).abs());
            changed += u64::from(delta > 0.000001);
            changed_confident += u64::from(delta > 0.000001 && a.score >= 0.36 && b.score >= 0.36);
            crossed += u64::from(
                [0.28, 0.32, 0.36, 0.49]
                    .iter()
                    .any(|t| (a.score >= *t) != (b.score >= *t)),
            );
            joints += 1;
        }
        count += 1;
    }
    println!("frames={count} people={matched_people} joints={joints} changed_positions={changed} changed_confident={changed_confident} threshold_crossings={crossed} acceptance_changes={acceptance_changes} max_xy={max_xy:.7} max_score={max_score:.7} largest_frame={largest_frame} largest_joint={largest_joint} largest_scores={largest_a:.4}/{largest_b:.4} single_seconds={:.3} batch_seconds={:.3}",single_time.as_secs_f64(),batch_time.as_secs_f64());
    Ok(())
}
