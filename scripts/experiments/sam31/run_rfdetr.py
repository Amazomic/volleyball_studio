#!/usr/bin/env python3
"""Experimental RF-DETR Small/Medium person proposals for the SAME Volley benchmark.

Output schema matches the SAM 3.1 bbox adapter. The temporary per-frame ids
are proposal indices, NOT tracked persons; pass into the existing Rust tracker
before calculating track IDs and continuity.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def person_boxes(detections, width, height):
    names = detections.data.get("class_name")
    if names is None:
        raise ValueError("RF-DETR predictions lack class_name; cannot filter person safely")
    result = []
    for box, confidence, class_name in zip(detections.xyxy, detections.confidence, names):
        if str(class_name).lower() != "person":
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        box_norm = [
            max(0.0, min(1.0, x1 / width)),
            max(0.0, min(1.0, y1 / height)),
            max(0.0, min(1.0, x2 / width)),
            max(0.0, min(1.0, y2 / height)),
        ]
        if box_norm[0] >= box_norm[2] or box_norm[1] >= box_norm[3]:
            continue
        result.append({
            "id": len(result),  # ephemeral query index, NOT athlete/track ID
            "bbox": [round(v, 7) for v in box_norm],
            "score": round(float(confidence), 6),
        })
    return result


def run(args):
    import cv2
    import torch
    from rfdetr import RFDETRSmall, RFDETRMedium

    if not torch.cuda.is_available():
        raise RuntimeError("RF-DETR benchmark requires CUDA to compare RTX GPU performance")
    cls = {"small": RFDETRSmall, "medium": RFDETRMedium}[args.model]
    video = Path(args.video).expanduser().resolve(strict=True)
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    loading = time.perf_counter()
    model = cls()
    loading_seconds = time.perf_counter() - loading
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode video: {video}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (640, 360):
        raise ValueError(f"expected common 640x360 benchmark MP4, got {width}x{height}")
    if cv2.CAP_PROP_FRAME_COUNT is not None:
        expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        expected_frames = 0
    index = 0
    count = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with out.open("w", encoding="utf-8") as stream:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = model.predict(rgb, threshold=args.threshold)
                tracks = person_boxes(detections, width, height)
                count += len(tracks)
                stream.write(json.dumps({"frame": index, "tracks": tracks},
                                        separators=(",", ":")) + "\n")
                index += 1
        torch.cuda.synchronize()
    finally:
        cap.release()
    elapsed = time.perf_counter() - started
    if index == 0 or (expected_frames and index != expected_frames):
        raise RuntimeError(f"decoded {index} frames, expected {expected_frames}")
    summary = {
        "model": f"rfdetr_{args.model}", "frames": index,
        "video": str(video), "video_sha256": sha256(video),
        "threshold": args.threshold, "total_person_boxes": count,
        "mean_boxes_per_frame": round(count / index, 3),
        "inference_seconds": round(elapsed, 4),
        "fps": round(index / elapsed, 3),
        "load_seconds_excluded": round(loading_seconds, 3),
        "peak_vram_mb": round(torch.cuda.max_memory_reserved() / 1048576, 1),
        "note": "Detection proposals only; id field is a frame-local index. Use Rust tracker for IDs.",
    }
    out.with_suffix(".stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["small", "medium"], required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.28)
    try:
        run(parser.parse_args())
    except Exception as exc:
        print(f"RF-DETR benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
