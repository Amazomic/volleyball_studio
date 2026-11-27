# Demonstration media

Captured on 2026-11-27 from the running Windows-native Volleyball Studio at `http://127.0.0.1:8875/studio`, build `c789d8190730d31fb795daffc0b1c29c095eabf9`.

- `studio-preview.png`: live Preview tab at 00:12.8 in a completed analysis, all seven overlay toggles enabled; video ready, canvas had non-transparent overlay pixels.
- `studio-analysis.png`: live Analysis tab for the same project, displaying the actual two-phase pipeline and saved stage results.
- `studio-demo.mp4`: 120 browser captures of the actual playing video and overlay at 8 fps, source time 00:06.0–00:20.9, encoded with FFmpeg; 15.000 seconds, 960 × 604, no audio. This is a recording of the player area, not new inference or a rendered mockup.

Screenshots are unmodified browser captures. No original match video, project database or personal photo library was copied into the repository. The captures document the interface and are not a benchmark of detection accuracy.

The old `examples/output.gif` is retained solely because two opt-in RF-DETR CUDA parity tests in `rust/volley-inference/src/person.rs` use it as an input fixture. It is not a current Studio demonstration.
