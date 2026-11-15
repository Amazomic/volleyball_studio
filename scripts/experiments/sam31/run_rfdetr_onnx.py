#!/usr/bin/env python3
"""Experimental RF-DETR Small/Medium ONNX person detector (GPU, no Torch).

Same input video, normalized detection schema, and Rust pose/tracker as baseline.
The id field here is frame-local proposal index, NOT an athlete/track identity.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_person(boxes, logits, threshold, person_index=1):
    """Decode RF-DETR normalized cxcywh and pre-sigmoid sparse COCO logits."""
    import numpy as np

    if boxes.ndim == 3:
        boxes = boxes[0]
    if logits.ndim == 3:
        logits = logits[0]
    if boxes.ndim != 2 or logits.ndim != 2:
        raise ValueError(f"unexpected ONNX ranks: {boxes.shape}, {logits.shape}")
    if boxes.shape[1] != 4 or logits.shape[0] != boxes.shape[0]:
        raise ValueError(f"unexpected ONNX detections: {boxes.shape}, {logits.shape}")
    if not (0 <= person_index < logits.shape[1]):
        raise ValueError(f"person class {person_index} not in {logits.shape}")
    # COCO category ids in the pretrained 91-slot model: person = 1.
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits[:, person_index], -50, 50)))
    result = []
    for cxywh, confidence in zip(boxes, scores):
        score = float(confidence)
        cx, cy, w, h = map(float, cxywh)
        if score < threshold or not all(math.isfinite(x) for x in (cx, cy, w, h, score)):
            continue
        coords = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        coords = [max(0.0, min(1.0, x)) for x in coords]
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        result.append({
            "id": len(result), "bbox": [round(x, 7) for x in coords],
            "score": round(score, 6),
        })
    return result


def preprocess(rgb, side):
    import cv2
    import numpy as np

    resized = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_LINEAR)
    image = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = ((image - mean) / std).transpose(2, 0, 1)
    return np.ascontiguousarray(image[None], dtype=np.float32)


def run(args):
    import cv2
    import onnxruntime as ort

    model = Path(args.model_path).expanduser().resolve(strict=True)
    video = Path(args.video).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"CUDAExecutionProvider not installed; available: {ort.get_available_providers()}")
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 8
    init_start = time.perf_counter()
    session = ort.InferenceSession(
        str(model), sess_options=opt,
        providers=["CUDAExecutionProvider"],
    )
    session.disable_fallback()
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"model GPU provider failed to initialize: {session.get_providers()}")
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"RF-DETR expected one ONNX input, got {inputs}")
    shape = inputs[0].shape
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3 or shape[2] != shape[3]:
        raise ValueError(f"unexpected RF-DETR ONNX input shape: {shape}")
    side = int(shape[2])
    if side not in (512, 576):
        raise ValueError(f"unexpected Small/Medium input dimension: {side}")
    input_name = inputs[0].name
    output_names = [item.name for item in session.get_outputs()]
    if {"pred_boxes", "pred_logits"}.issubset(output_names):
        box_name, logit_name = "pred_boxes", "pred_logits"
    elif {"dets", "labels"}.issubset(output_names):
        box_name, logit_name = "dets", "labels"
    else:
        raise ValueError(f"unexpected RF-DETR ONNX output names: {output_names}")
    load_seconds = time.perf_counter() - init_start

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode video {video}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (640, 360):
        raise ValueError(f"comparison video must be 640x360, got {width}x{height}")
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame = total_boxes = 0
    started = time.perf_counter()
    try:
        with output.open("w", encoding="utf-8") as stream:
            while True:
                valid, bgr = cap.read()
                if not valid:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                output_raw = session.run([box_name, logit_name], {input_name: preprocess(rgb, side)})
                tracks = decode_person(output_raw[0], output_raw[1], args.threshold, args.person_index)
                stream.write(json.dumps({"frame": frame, "tracks": tracks}, separators=(",", ":")) + "\n")
                total_boxes += len(tracks)
                frame += 1
    finally:
        cap.release()
    elapsed = time.perf_counter() - started
    if frame == 0 or (expected and frame != expected):
        raise RuntimeError(f"incomplete video: {frame} vs {expected} frames")
    report = {
        "model": f"rfdetr_{args.variant}_onnx",
        "source_model": str(model), "model_sha256": sha256(model),
        "frames": frame, "video": str(video), "video_sha256": sha256(video),
        "threshold": args.threshold, "person_class_index": args.person_index,
        "onnx_input_shape": shape, "onnx_output_names": output_names,
        "runtime_providers": session.get_providers(),
        "total_person_boxes": total_boxes,
        "mean_boxes_per_frame": round(total_boxes / frame, 3),
        "inference_seconds": round(elapsed, 4),
        "fps": round(frame / elapsed, 3),
        "load_seconds_excluded": round(load_seconds, 3),
        "note": "ONNX detector output only; id field is frame-local proposal index",
    }
    output.with_suffix(".stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["small", "medium"], required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.28)
    ap.add_argument("--person-index", type=int, default=1)
    try:
        run(ap.parse_args())
    except Exception as exc:
        print(f"ONNX RF-DETR experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
