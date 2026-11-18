#!/usr/bin/env python3
"""All variants side-by-side on same video; identity metrics are diagnostics."""
import argparse
import json
from pathlib import Path
from compare_tracking import read_track_file, summarize


def compare(baseline_path, candidates, expected_persons=4, min_observations=12):
    items = {"RTMDet-Nano + RTMPose + Rust": Path(baseline_path)}
    for item in candidates:
        if "=" not in item:
            raise ValueError("candidate must be LABEL=PATH")
        label, path = item.split("=", 1)
        if not label or not path or label in items:
            raise ValueError("empty or duplicate candidate name")
        items[label] = Path(path)
    results = {}
    reference_hash, reference_frames = None, None
    for name, path in items.items():
        frames = read_track_file(path)
        meta_path = path.with_suffix(".stats.json")
        if not meta_path.is_file():
            raise ValueError(f"Missing benchmark stats file: {meta_path}")
        stats = json.loads(meta_path.read_text(encoding="utf-8"))
        if stats.get("frames") != len(frames):
            raise ValueError(f"{path}: JSONL/stats frame count mismatch")
        video_hash = stats.get("video_sha256")
        if not video_hash:
            raise ValueError(f"{path}: missing video_sha256")
        if reference_hash is None:
            reference_hash, reference_frames = video_hash, len(frames)
        if video_hash != reference_hash or len(frames) != reference_frames:
            raise ValueError(f"{path}: inconsistent video/frame count")
        score = summarize(frames, min_observations, expected_persons)
        score.update({
            "fps": stats.get("fps"),
            "peak_vram_mb": stats.get("peak_vram_mb"),
            "source_model": stats.get("model"),
        })
        results[name] = score
    return {
        "video_sha256": reference_hash,
        "frames": reference_frames,
        "warning": "Provisional IDs only, no manual athlete ground truth; a smaller ID count is not proof of better identity tracking.",
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", action="append", required=True,
                    help='e.g. "RF-DETR Small=/tmp/volley-rfdetr/small_pose.jsonl"')
    ap.add_argument("--expected-persons", type=int, default=4)
    ap.add_argument("--min-observations", type=int, default=12)
    ap.add_argument("--output")
    args = ap.parse_args()
    output = compare(args.baseline, args.candidate,
                     args.expected_persons, args.min_observations)
    text = json.dumps(output, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
