# Optional Volley3D-style TCN for CPU Touch postprocessing

Reference: https://arxiv.org/abs/2610.38358, supplementary B.5 (2026).
This project reproduces the reported architecture/input protocol, but DOES
NOT include the paper authors' model weights or claim their measured F1.

Runtime sequence:
VBallNet saved coordinates -> normalized X,Y,dX,dY -> ONNX 17-frame TCN
(stride 3, overlap averaged) -> peak >=0.35 AND configured Touch threshold
-> physically observed velocity break and reachable same-frame tracked actor
-> optional RTMPose confirmation -> provisional touch candidates.

The user checks the optional TCN experiment inside Studio > Postprocessing >
collapsed settings; no default algorithm selector is reintroduced.
Legacy and PoseMotion are untouched, and the GPU analysis is never rerun.
Missing weights cause a clear error, not silent heuristic fallback.
Weight SHA-256 is recorded in the touch manifest and checked on CPU reuse.

Expected model file: models/volley3d_touch_tcn.onnx
ONNX input name/shape: trajectory float32 [1,4,17].
ONNX output name/shape: probabilities float32 [1,1,17] after sigmoid.
Architecture: Conv1D(4->128,k5), BN/ReLU/Dropout(0.5), residual TCN
blocks 128 dilation 2 and 256 dilation 4 (two convs per block, k5),
Conv1D(256->1,k1), sigmoid.

No official trained checkpoint is currently bundled. You can train your
own with the standalone OFFLINE utility:
scripts/training/train_volley3d_tcn.py
Training dependencies: torch, numpy, onnx. This script is not a Studio
runtime backend, and is never executed by the application.

Training JSONL (one full rally per line, at least 17 frames):
~~~json
{"match_id":"m1","width":1280,"height":720,"ball_xy":[[300,200],[301,199],null],"contacts":[1]}
~~~
The short row above illustrates the schema only and is not a valid training
sequence. Extend ball_xy to 17+ frames. Values are original pixel positions;
null marks missing VBallNet detection; contacts are local 0-based frame indices.
Use at least two distinct match_id groups to avoid leaking frames across
training/validation. Only manually reviewed contacts qualify as labels.

~~~bash
python scripts/training/train_volley3d_tcn.py --dataset data/tcn_reviewed_contacts.jsonl --out models/volley3d_touch_tcn.onnx
~~~

The trainer uses Gaussian label smoothing (sigma=1, radius=3),
Soft-Focal alpha=0.25/gamma=2, Adam, cosine LR, 60 epochs, batch 64,
and reported spatial/noise/cutout augmentations. Some architecture/training
details may differ from unpublished reference code.
Evaluate separately on unseen beach volleyball matches, recording
precision, recall and F1 within +/-1 and +/-3 frames before enabling by default.

## Export reviewed Studio Touch labels

Use scripts/training/export_reviewed_tcn.py to read confirmed human-review
records and matching immutable CPU replay data. The script validates checksums,
original video provenance, and uniqueness of labels; it merges overlapping
rallies to avoid duplicate contacts. It writes to a new file outside the
Studio data root and never alters GPU or CPU evidence.

From the repository root (WSL):

    python3 scripts/training/export_reviewed_tcn.py --data-root "$HOME/.cache/volley-rust-sandbox" --out results/training/volley3d_reviewed.jsonl --report results/training/volley3d_reviewed.report.json

For training at least two different match/source SHA values are required,
because one match cannot supply an independent validation match. Prefer
five or more independently reviewed games and a separate holdout test set.

Caveats: reviewed touches can be corrected auto-generated suggestions,
not blind annotations. The exporter uses VBallNet score threshold filtering,
but does not fully reproduce the runtime ball-track/court mask.
