use ort::{
    execution_providers::CUDAExecutionProvider,
    session::Session,
    value::Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ort::init().commit()?;
    let detector = std::env::args().nth(1).unwrap_or_else(|| "models/RTMDet-Nano-Person.onnx".into());
    for (path, height, width) in [
        (detector.as_str(), 320, 320),
        ("models/RTMPose-M-Body7.onnx", 256, 192),
    ] {
        let mut session = Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default()
                .build()
                .error_on_failure()])?
            .commit_from_file(path)?;
        let tensor = Tensor::from_array((
            vec![2_i64, 3, height, width],
            vec![0_f32; 2 * 3 * height as usize * width as usize],
        ))?;
        println!("{path}");
        match session.run(ort::inputs!["input" => tensor]) {
            Ok(output) => for (name, value) in output.iter() {
                println!("{name}: {:?}", value.dtype());
            },
            Err(error) => println!("batch failed: {error}"),
        };
    }
    Ok(())
}
