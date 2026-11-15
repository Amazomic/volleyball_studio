#!/usr/bin/env python3
"""Isolated SAM 3.1 tracking benchmark (not used by the Studio runtime).

SAM 3.1 replaces RTMDet-Nano and provides its own track ID + person mask.
This does NOT provide RTMPose keypoints; no pose-dependent classification.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boxes_from_masks(outputs):
    """Return [0,1] xyxy boxes and stable provisional IDs from SAM 3.1 masks."""
    import numpy as np

    ids = outputs.get("out_obj_ids", [])
    masks = outputs.get("out_binary_masks", [])
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().numpy()
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    ids, masks = np.asarray(ids).reshape(-1), np.asarray(masks)
    if not len(ids):
        return []
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or len(ids) != masks.shape[0]:
        raise ValueError(f"unexpected masks: {masks.shape}, IDs: {len(ids)}")
    result = []
    for obj_id, mask in zip(ids, masks):
        ys, xs = np.nonzero(mask > 0.5)
        if len(xs):
            h, w = mask.shape
            result.append({
                "id": int(obj_id),
                "bbox": [round(float(xs.min()) / w, 7),
                         round(float(ys.min()) / h, 7),
                         round(float(xs.max() + 1) / w, 7),
                         round(float(ys.max() + 1) / h, 7)],
            })
    return result


def run(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    video = Path(args.video).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    load_at = time.perf_counter()
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=args.checkpoint,
        max_num_objects=args.max_objects,
        multiplex_count=args.max_objects,
        use_fa3=False,
        compile=False,
        async_loading_frames=False,
    )
    load_seconds = time.perf_counter() - load_at
    started = time.perf_counter()
    session_id = None
    observed = {}
    try:
        session = predictor.handle_request({
            "type": "start_session", "resource_path": str(video),
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
        })
        session_id = session["session_id"]
        first = predictor.handle_request({
            "type": "add_prompt", "session_id": session_id,
            "frame_index": 0, "text": args.prompt,
        })
        observed[0] = boxes_from_masks(first["outputs"])
        for frame in predictor.handle_stream_request({
            "type": "propagate_in_video", "session_id": session_id,
            "propagation_direction": "forward",
        }):
            observed[int(frame["frame_index"])] = boxes_from_masks(frame["outputs"])
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
    finally:
        if session_id is not None:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
    if not observed:
        raise RuntimeError("model returned no frames")
    count = max(observed) + 1
    with output.open("w", encoding="utf-8") as stream:
        for index in range(count):
            stream.write(json.dumps({
                "frame": index, "tracks": observed.get(index, []),
            }, separators=(",", ":")) + "\n")
    metrics = {
        "model": "sam3.1_multiplex", "video": str(video),
        "video_sha256": sha256(video), "frames": count,
        "inference_seconds": round(inference_seconds, 4),
        "fps": round(count / inference_seconds, 3),
        "load_seconds_excluded": round(load_seconds, 3),
        "peak_vram_mb": round(torch.cuda.max_memory_reserved() / 1048576, 1),
        "prompt": args.prompt, "max_objects": args.max_objects,
    }
    output.with_suffix(".stats.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", help="Path to sam3.1_multiplex.pt; otherwise HF download")
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--max-objects", type=int, default=16, choices=range(1, 17))
    try:
        run(parser.parse_args())
    except Exception as exc:
        print(f"SAM 3.1 unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
