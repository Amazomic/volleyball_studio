#!/usr/bin/env python3
"""Read-only export of *reviewed* Volleyball Studio Touch labels to TCN JSONL.

Uses immutable, SHA-verified CPU replays and manual touch annotations. There is
no model training, video decoding, GPU inference or project data mutation.
Overlapping reviewed rally intervals are merged so contacts are not duplicated.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ANNOTATION_SCHEMA = "volley-touch-annotations/1"
MAX_FRAMES = 48 * 3600 * 30


def sha256_file(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as src:
        for data in iter(lambda: src.read(1024 * 1024), b""):
            hash_.update(data)
    return hash_.hexdigest()


def provenance_hash(provenance: dict) -> str:
    # Mirrors serde_json::to_vec(&Value), whose object keys are sorted.
    return hashlib.sha256(json.dumps(
        provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merged_reviewed_intervals(marks: list, total_frames: int):
    intervals = sorted((int(m["start"]), int(m["end"])) for m in marks)
    if not intervals:
        raise ValueError("Reviewed video lacks rally intervals")
    if any(start < 0 or end < start or end >= total_frames for start, end in intervals):
        raise ValueError("Out-of-bounds manual rally interval")
    merged = []
    overlap_count = 0
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 1:
            if start <= merged[-1][1]:
                overlap_count += 1
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged, overlap_count


def matching_touch_run(project: Path, annotation: dict) -> tuple:
    root = project / "native_cpu_touch_runs"
    candidates = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.is_symlink():
            continue
        manifest_file = directory / "manifest.json"
        if not manifest_file.is_file():
            continue
        record = read_json(manifest_file)
        if record.get("sha256") != annotation["baseline_touch_sha"]:
            continue
        if record.get("video") != annotation["video"] or record.get("project_id") != annotation["project_id"]:
            continue
        if record.get("frames") != annotation["total_frames"] or record.get("fps") != 30:
            continue
        output = directory / "touch_candidates.jsonl"
        if not output.is_file() or sha256_file(output) != record["sha256"]:
            raise ValueError("Baseline Touch output failed SHA verification")
        candidates.append((record, directory))
    if not candidates:
        raise ValueError("Reviewed baseline Touch run or verified file not found")
    # Duplicate identical runs are harmless; pick deterministically.
    return candidates[0]


def load_ball_sequence(replay_root: Path, replay_manifest: dict, total_frames: int,
                       width: int, height: int, minimum_score: float):
    if replay_manifest.get("schema") != "volley-native-cpu-tracking/1" \
            or replay_manifest.get("file") != "tracked_frames.jsonl" \
            or replay_manifest.get("frames") != total_frames \
            or replay_manifest.get("fps") != 30:
        raise ValueError("Incompatible CPU tracking replay")
    source = replay_root / "tracked_frames.jsonl"
    if not source.is_file() or sha256_file(source) != replay_manifest.get("sha256"):
        raise ValueError("CPU tracking input failed SHA verification")
    result = []
    observed = 0
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if len(result) >= total_frames:
                raise ValueError("Tracking replay longer than manifest")
            frame = json.loads(line)
            if frame.get("frame_index") != len(result) \
                    or frame.get("confirmed_athlete_identities") is not False \
                    or frame.get("track_ids_provisional") is not True:
                raise ValueError("Untrusted tracking frame or index gap")
            ball = frame.get("selected_ball")
            pos = None
            if ball and ball.get("source_model") == "vballnet":
                detection = ball["detection"]
                score = float(detection["score"])
                b = detection["bbox_xyxy_norm"]
                if not (math.isfinite(score) and 0 <= score <= 1 and len(b) == 4
                        and all(math.isfinite(v) and 0 <= v <= 1 for v in b)
                        and b[2] > b[0] and b[3] > b[1]):
                    raise ValueError("Invalid VBallNet ball observation")
                if score >= minimum_score:
                    pos = [round((b[0] + b[2]) * 0.5 * width, 7),
                           round((b[1] + b[3]) * 0.5 * height, 7)]
                    observed += 1
            result.append(pos)
    if len(result) != total_frames:
        raise ValueError("Tracking replay shorter than manifest")
    return result, observed


def export_dataset(data_root: Path, output: Path, report_path: Path):
    root = data_root.resolve(strict=True)
    dest = output.resolve(strict=False)
    meta_dest = report_path.resolve(strict=False)
    if dest == meta_dest or dest.is_relative_to(root) or meta_dest.is_relative_to(root):
        raise ValueError("Output must be outside the immutable Studio data root")
    if dest.exists() or meta_dest.exists():
        raise FileExistsError("Output already exists; choose new paths, never overwrite prior exports")

    annotation_file = root / "training" / "touch_annotations.json"
    library = read_json(annotation_file)
    if library.get("schema") != ANNOTATION_SCHEMA:
        raise ValueError("Unsupported human annotation schema")
    reviewed = [x for x in library.get("videos", []) if x.get("reviewed") is True]
    if not reviewed:
        raise ValueError("There are no human-reviewed videos. No pseudo-labels will be generated.")
    records = []
    summary = []
    for annotation in reviewed:
        total = annotation.get("total_frames")
        if not isinstance(total, int) or not (17 <= total <= MAX_FRAMES):
            raise ValueError("Invalid reviewed video frame count")
        intervals, overlaps = merged_reviewed_intervals(annotation["rallies"], total)
        gold = [int(e["frame"]) for e in annotation["touches"]]
        if len(set(gold)) != len(gold) or not gold \
                or any(t < 0 or t >= total for t in gold):
            raise ValueError("Unusable or duplicate human touch labels")
        if any(not any(start <= t <= end for start, end in intervals) for t in gold):
            raise ValueError("Manual touch outside the reviewed rally intervals")

        project = root / "projects" / annotation["project_id"]
        touch, _ = matching_touch_run(project, annotation)
        replay_id = touch["source_replay_id"]
        replay_dir = project / "native_cpu_replays" / replay_id
        replay = read_json(replay_dir / "manifest.json")
        original = touch["original_source"]
        if replay.get("sha256") != touch.get("source_replay_sha256") \
                or replay.get("original_source") != original \
                or replay.get("video") != annotation["video"] \
                or replay.get("project_id") != annotation["project_id"] \
                or provenance_hash(original) != annotation["source_sha256"]:
            raise ValueError("Human labels/source provenance do not match immutable replay")
        width, height = int(touch["width"]), int(touch["height"])
        if not (64 <= width <= 16384 and 64 <= height <= 16384):
            raise ValueError("Invalid original frame dimensions")
        min_score = float(touch["settings"]["min_ball_score"])
        if not 0.01 <= min_score <= 1:
            raise ValueError("Invalid ball threshold")
        balls, observed = load_ball_sequence(replay_dir, replay, total, width, height, min_score)
        exported_touches = 0
        for start, end in intervals:
            if end - start + 1 < 17:
                continue
            local_contacts = sorted(t - start for t in gold if start <= t <= end)
            exported_touches += len(local_contacts)
            records.append({
                "match_id": original["sha256"],
                "width": width, "height": height,
                "ball_xy": balls[start:end + 1],
                "contacts": local_contacts,
                "provenance": {
                    "project_id": annotation["project_id"],
                    "video": annotation["video"],
                    "source_sha256": annotation["source_sha256"],
                    "replay_sha256": replay["sha256"],
                    "touch_sha256": touch["sha256"],
                    "start_frame": start, "end_frame_inclusive": end,
                    "analysis_fps": 30,
                    "min_ball_score": min_score,
                },
            })
        if exported_touches != len(gold):
            raise ValueError("Some reviewed touch labels would be silently dropped")
        summary.append({
            "video": annotation["video"], "original_rally_intervals": len(annotation["rallies"]),
            "merged_intervals": len(intervals), "overlap_pairs_merged": overlaps,
            "reviewed_touches": len(gold), "selected_vballnet_frames": observed,
            "total_frames": total, "baseline_quality": annotation.get("quality"),
        })
    if not records:
        raise ValueError("No 17-frame reviewed rally intervals to export")
    match_ids = {rec["match_id"] for rec in records}
    payload = "".join(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
                      for rec in records)
    report = {
        "schema": "volley-tcn-reviewed-export/1",
        "reviewed_videos": len(summary),
        "segments": len(records),
        "segments_frames": sum(len(rec["ball_xy"]) for rec in records),
        "contacts": sum(len(rec["contacts"]) for rec in records),
        "distinct_source_hashes": len(match_ids),
        "can_holdout_by_source": len(match_ids) >= 2,
        "recommended_at_least_distinct_matches": 5,
        "training_warning": "Human review is not necessarily independent blind annotation",
        "preprocessing_warning": "Only VBallNet min score filtering applied; production ball-track mask, teleport and court filters are NOT reproduced",
        "annotation_sha256": sha256_file(annotation_file),
        "dataset_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "videos": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as file:
        file.write(payload)
    with report_path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    out = args.out
    report = args.report or out.with_suffix(".report.json")
    result = export_dataset(args.data_root, out, report)
    print(json.dumps({
        "output": str(out), "report": str(report),
        "reviewed_videos": result["reviewed_videos"],
        "segments": result["segments"],
        "contacts": result["contacts"],
        "distinct_source_hashes": result["distinct_source_hashes"],
        "can_holdout_by_source": result["can_holdout_by_source"],
        "dataset_sha256": result["dataset_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
