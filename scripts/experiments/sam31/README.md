# Experimental SAM 3.1 vs RTMDet/RTMPose tracking

State: isolated experimental branch, NOT wired into the Rust Studio runtime. Production GPU inference, UI and caches are unchanged.

Meta SAM 3.1 source: https://github.com/facebookresearch/sam3
Weights: https://huggingface.co/facebook/sam3.1 (requires approved access, model file sam3.1_multiplex.pt).

## Same-input benchmark

* Input: exact same prepared 640x360 @ 30000/1001 MP4 (180 frames), generated once from original examples/beach_st_lenina_20250622_g1_005.mp4. Both read /tmp/volley-sam31/sample_640x360.mp4.
* Baseline: RTMDet-Nano + RTMPose PersonEngine -> Rust ProvisionalTracker (default settings).
* Experimental: SAM 3.1 multiplex prompt "person" -> masks -> box + native track IDs. The SAM variant replaces detection/tracking **for this standalone benchmark only**; it does not yet supply RTMPose keypoints or Touch evidence.
* Both are timed excluding model initialization. Decode, detection, association/propagation included.
* GPU: run sequentially on same hardware. SAM settings: max_objects=16, no compilation/FA3, CPU state offload for 12 GB VRAM.
* Compare: FPS, total provisional track IDs, confirmed IDs (12+ detected frames), empty frames, observed boxes, missing frames inside same track ID, longest ID gap.
* Not ground-truth precision/recall/MOTA/HOTA/identity-switch counts. Manual identity annotation is required to know true ID changes and false positives.

### Runtime commands in WSL

Set repo path once:

    repo=/mnt/c/Users/amazomic/AppData/Local/SentinelX/workspace/volley-sam31-experiment
    mkdir -p /tmp/volley-sam31
    ffmpeg -hide_banner -y -i /mnt/c/Users/amazomic/Desktop/github/fast-volleyball-tracking-inference/examples/beach_st_lenina_20250622_g1_005.mp4 -vf scale=640:360,fps=30000/1001 -an -c:v libx264 -crf 18 /tmp/volley-sam31/sample_640x360.mp4
    export CARGO_TARGET_DIR=/tmp/volley-sam31-build

Run compiled native baseline:

    source_repo=/mnt/c/Users/amazomic/Desktop/github/fast-volleyball-tracking-inference
    source "$source_repo/scripts/setup_rust_ort_cuda.sh" check
    ~/.cargo/bin/cargo run --manifest-path "$repo/rust/Cargo.toml" -p volley-app --release --offline --example bench_person_tracks -- "$source_repo" /tmp/volley-sam31/sample_640x360.mp4 /tmp/volley-sam31/base.jsonl

Install SAM 3.1 in a separate Python 3.12 virtual environment AFTER Hugging Face approval:

    uv venv --python 3.12 ~/.venvs/volley-sam31
    uv pip install --python ~/.venvs/volley-sam31/bin/python torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
    git clone --depth 1 https://github.com/facebookresearch/sam3.git /tmp/volley-sam31-sam3
    uv pip install --python ~/.venvs/volley-sam31/bin/python -e /tmp/volley-sam31-sam3
    # Use official HF CLI to log in after approval. Never put credentials in scripts.
    ~/.venvs/volley-sam31/bin/python "$repo/scripts/experiments/sam31/run_sam31.py" --video /tmp/volley-sam31/sample_640x360.mp4 --output /tmp/volley-sam31/sam31.jsonl

Evaluate both same-frame outputs:

    python3 "$repo/scripts/experiments/sam31/compare_tracking.py" --baseline /tmp/volley-sam31/base.jsonl --sam31 /tmp/volley-sam31/sam31.jsonl --expected-persons 4

Integration to Studio would need a separately reviewed SAM -> PoseEngine -> immutable GPU evidence bridge; do not swap production without annotated benchmark.


### Detector-only substitution (keep RTMPose and Rust tracker)

The additional experimental variant uses SAM 3.1 person proposals INSTEAD OF RTMDet-Nano but preserves RTMPose keypoints and the existing Rust CPU tracker. SAM native IDs are deliberately ignored by this variant so any difference is due to the detector. It runs outside the production Studio pipeline.

    # Source same ONNX Runtime CUDA environment used in normal Studio.
    source "$source_repo/scripts/setup_rust_ort_cuda.sh" check
    ~/.cargo/bin/cargo run --manifest-path "$repo/rust/Cargo.toml" -p volley-app --release --offline --example bench_detector_swap -- "$source_repo" /tmp/volley-sam31/sample_640x360.mp4 /tmp/volley-sam31/sam31.jsonl /tmp/volley-sam31/sam_pose.jsonl

    python3 "$repo/scripts/experiments/sam31/compare_tracking.py" --baseline /tmp/volley-sam31/base.jsonl --sam31 /tmp/volley-sam31/sam_pose.jsonl --expected-persons 4

For real model inference, obtain approval and sign in to Hugging Face on the host:

    hf auth login

There must be an approved token in the WSL environment where SAM is run; HTTP 401 means weights are not available. Do not share token values. Results from a mock bbox adapter cannot be used as SAM performance figures.

Results to date: scripts/experiments/sam31/RESULTS.md


## RF-DETR Small and Medium (NO gated weights)

Both use Roboflow's Apache-2.0 weights, automatically downloaded on first run. Unlike SAM 3.1 they provide person detections, not stable IDs. Run these proposals through native RTMPose and Rust ProvisionalTracker to measure continuity and unique IDs.

All paths below are in WSL:

    repo=/mnt/c/Users/amazomic/AppData/Local/SentinelX/workspace/volley-sam31-experiment
    source_repo=/mnt/c/Users/amazomic/Desktop/github/fast-volleyball-tracking-inference
    video=/tmp/volley-sam31/sample_640x360.mp4

The experiment's isolated Python environment is /tmp/volley-rfdetr-venv, created with:

    ~/.local/bin/uv venv --python 3.12 /tmp/volley-rfdetr-venv
    ~/.local/bin/uv pip install --python /tmp/volley-rfdetr-venv/bin/python --torch-backend auto rfdetr opencv-python-headless

Run sequentially, not together (GPU interference):

    /tmp/volley-rfdetr-venv/bin/python "$repo/scripts/experiments/sam31/run_rfdetr.py" --model small --video "$video" --output /tmp/volley-sam31/rfdetr_small.jsonl
    /tmp/volley-rfdetr-venv/bin/python "$repo/scripts/experiments/sam31/run_rfdetr.py" --model medium --video "$video" --output /tmp/volley-sam31/rfdetr_medium.jsonl

Run the SAME pose+tracker as baseline:

    source "$source_repo/scripts/setup_rust_ort_cuda.sh" check
    export CARGO_TARGET_DIR=/tmp/volley-sam31-build
    ~/.cargo/bin/cargo run --manifest-path "$repo/rust/Cargo.toml" -p volley-app --release --offline --example bench_detector_swap -- "$source_repo" "$video" /tmp/volley-sam31/rfdetr_small.jsonl /tmp/volley-sam31/rfdetr_small_pose.jsonl
    ~/.cargo/bin/cargo run --manifest-path "$repo/rust/Cargo.toml" -p volley-app --release --offline --example bench_detector_swap -- "$source_repo" "$video" /tmp/volley-sam31/rfdetr_medium.jsonl /tmp/volley-sam31/rfdetr_medium_pose.jsonl

Compare all outputs:

    python3 "$repo/scripts/experiments/sam31/compare_variants.py" \
      --baseline /tmp/volley-sam31/base.jsonl \
      --candidate RF-DETR-Small=/tmp/volley-sam31/rfdetr_small_pose.jsonl \
      --candidate RF-DETR-Medium=/tmp/volley-sam31/rfdetr_medium_pose.jsonl \
      --output /tmp/volley-sam31/all_models.json

Interpretation: raw RF-DETR id is only a per-frame proposal index, not an athlete ID. The final file's Rust tracker IDs are counted. RF-DETR keeps detector boxes even when RTMPose skeleton is incomplete; the current baseline drops low-quality skeleton boxes. This is a combined detector/pose-gating comparison, not strictly one-variable. Accurate ID-switch rate/recall needs manual annotations.
