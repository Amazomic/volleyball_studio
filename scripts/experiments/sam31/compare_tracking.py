#!/usr/bin/env python3
"""Comparable continuity/ID/FPS summary from exactly the same video frames.

No ground-truth = no valid ID-switch rate, precision, recall or MOTA/HOTA.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_track_file(path):
    result = []
    with open(path, encoding="utf-8") as f:
        for line_number, text in enumerate(f, 1):
            if not text.strip():
                continue
            item = json.loads(text)
            frame = item["frame"]
            if frame != len(result):
                raise ValueError(f"{path}:{line_number}: expected frame {len(result)}, got {frame}")
            ids = set()
            for detection in item["tracks"]:
                track_id = str(detection["id"])
                box = detection["bbox"]
                if track_id in ids or len(box) != 4 or not all(0 <= x <= 1 for x in box):
                    raise ValueError(f"{path}:{line_number}: duplicate track ID or invalid bbox")
                if box[0] >= box[2] or box[1] >= box[3]:
                    raise ValueError(f"{path}:{line_number}: invalid xyxy")
                ids.add(track_id)
            result.append(item)
    if not result:
        raise ValueError(f"empty track file: {path}")
    return result


def summarize(frames, min_observations=12, expected_persons=4):
    lifetime = defaultdict(list)
    complete = 0
    empty = 0
    longest_empty = 0
    empty_streak = 0
    count_detections = 0
    for item in frames:
        count = len(item["tracks"])
        count_detections += count
        if count >= expected_persons:
            complete += 1
        if count == 0:
            empty += 1
            empty_streak += 1
            longest_empty = max(longest_empty, empty_streak)
        else:
            empty_streak = 0
        for row in item["tracks"]:
            lifetime[str(row["id"])].append(item["frame"])
    gaps = []
    gap_events = 0
    confirmed = 0
    for track, indexes in lifetime.items():
        if len(indexes) < min_observations:
            continue
        confirmed += 1
        for before, after in zip(indexes, indexes[1:]):
            if after - before > 1:
                gap_events += 1
                gaps.append(after - before - 1)
    return {
        "frames": len(frames),
        "all_provisional_ids": len(lifetime),
        "ids_at_least_min_observations": confirmed,
        "min_observations": min_observations,
        "total_observed_boxes": count_detections,
        "mean_boxes_per_frame": round(count_detections / len(frames), 3),
        "frames_with_at_least_expected_people": complete,
        "expected_people": expected_persons,
        "all_people_absent_frames": empty,
        "longest_no_person_streak": longest_empty,
        "track_gap_events": gap_events,
        "track_missing_frames_inside_lifespan": sum(gaps),
        "longest_track_gap_frames": max(gaps, default=0),
    }


def run(base, sam, expected_persons=4, min_observations=12):
    before = read_track_file(base)
    after = read_track_file(sam)
    if len(before) != len(after):
        raise ValueError(f"different frame ranges: {len(before)} vs {len(after)}")
    scores = {
        "baseline": summarize(before, min_observations, expected_persons),
        "sam31": summarize(after, min_observations, expected_persons),
    }
    info = {}
    for name, path in (("baseline", base), ("sam31", sam)):
        meta = Path(path).with_suffix(".stats.json")
        if meta.is_file():
            stats = json.loads(meta.read_text(encoding="utf-8"))
            if stats.get("frames") != len(before):
                raise ValueError(f"{meta}: frame mismatch")
            info[name] = stats
            scores[name]["fps"] = stats.get("fps")
            scores[name]["peak_vram_mb"] = stats.get("peak_vram_mb")
    if len(info) == 2 and info["baseline"].get("video_sha256") != info["sam31"].get("video_sha256"):
        raise ValueError("different input clips; comparison is invalid")
    return {
        "warning": "Unsupervised diagnostic proxies; NOT actual recall, person identity switches, HOTA or MOTA.",
        "results": scores,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="RTMDet+RTMPose+Rust output JSONL")
    ap.add_argument("--sam31", required=True, help="SAM3.1 output JSONL")
    ap.add_argument("--expected-persons", type=int, default=4)
    ap.add_argument("--min-observations", type=int, default=12)
    ap.add_argument("--output")
    args = ap.parse_args()
    result = run(args.baseline, args.sam31, args.expected_persons, args.min_observations)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
