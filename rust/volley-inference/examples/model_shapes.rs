use ort::session::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ort::init().commit()?;
    for path in std::env::args().skip(1) {
        let session = Session::builder()?.commit_from_file(&path)?;
        println!("{path}");
        for input in &session.inputs {
            println!("input {} {:?}", input.name, input.input_type);
        }
        for output in &session.outputs {
            println!("output {} {:?}", output.name, output.output_type);
        }
    }
    Ok(())
}
