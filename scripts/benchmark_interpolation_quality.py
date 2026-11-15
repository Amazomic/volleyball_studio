import json
import math
import statistics
import sys
from pathlib import Path

MAX_GAP = 120
WIDTH = 640
HEIGHT = 360


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (aa + bb - intersection) if aa + bb > intersection else 0.0


def center(a):
    return ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)


def pairing_score(left, at, right):
    gap = at - left["frame"]
    if not 0 < gap <= MAX_GAP:
        return None
    a, b = left["bbox"], right
    ha, hb = a[3] - a[1], b[3] - b[1]
    wa, wb = a[2] - a[0], b[2] - b[0]
    ratio = max(ha / hb, hb / ha, wa / wb, wb / wa)
    if ratio > 1.9:
        return None
    max_move = min(0.035 + gap * 0.005, 0.32)
    c1, c2 = center(a), center(b)
    displacement = math.dist(c1, c2)
    if displacement > max_move:
        return None
    overlap = iou(a, b)
    if gap > 30 and overlap < 0.18 and displacement > 0.055:
        return None
    location = max(0.0, min(1.0, 1.0 - displacement / max_move))
    shape = max(0.0, min(1.0, 1.0 - (ratio - 1.0) / 0.9))
    return 0.47 * location + 0.20 * overlap + 0.13 * shape


def paired(anchors, observed, at):
    scores = [[pairing_score(left, at, right) for right in observed] for left in anchors]
    pairs = []
    for ai, row in enumerate(scores):
        rank = sorted(((v, j) for j, v in enumerate(row) if v is not None), reverse=True)
        if not rank:
            continue
        score, bi = rank[0]
        if score < 0.47 or len(rank) > 1 and score - rank[1][0] < 0.10:
            continue
        reverse = [r[bi] for r in scores if r[bi] is not None]
        if any(other > score + 1e-8 for other in reverse):
            continue
        if sum(other >= score - 0.10 for other in reverse) > 1:
            continue
        pairs.append((ai, bi))
    return sorted(pairs)


def load_frames(path):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            distinct = []
            for box in row["boxes"]:
                if all(iou(box, previous) < 0.60 for previous in distinct):
                    distinct.append(box)
            result[row["frame"]] = distinct
    return result


def interpolate_rows(samples, length):
    frames = [[] for _ in range(length)]
    anchors = []
    matched_pairs = 0
    for at, observed in sorted(samples.items()):
        frames[at] = [list(box) for box in observed]
        anchors = [item for item in anchors if at - item["frame"] <= MAX_GAP]
        pairs = paired(anchors, observed, at)
        occupied = set()
        for ai, bi in pairs:
            previous = anchors[ai]
            bbox = observed[bi]
            span = at - previous["frame"]
            for index in range(previous["frame"] + 1, at):
                fraction = (index - previous["frame"]) / span
                candidate = [v + (w - v) * fraction for v, w in zip(previous["bbox"], bbox)]
                if any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in candidate):
                    continue
                if candidate[2] - candidate[0] < 8 / WIDTH or candidate[3] - candidate[1] < 20 / HEIGHT:
                    continue
                if len(frames[index]) >= 128 or any(iou(candidate, other) >= 0.38 for other in frames[index]):
                    continue
                frames[index].append(candidate)
            anchors[ai] = {"frame": at, "bbox": bbox}
            occupied.add(bi)
            matched_pairs += 1
        for bi, bbox in enumerate(observed):
            if bi not in occupied:
                anchors.append({"frame": at, "bbox": bbox})
    return frames, matched_pairs


def compare(reference, predicted, ignored_frames):
    reference_total = predicted_total = matched = complete = 0
    errors = []
    matched_ious = []
    frames_with_gt = 0
    misses_at_nonanchors = 0
    for frame, actual in sorted(reference.items()):
        if frame in ignored_frames:
            continue
        frames_with_gt += bool(actual)
        inferred = predicted[frame]
        reference_total += len(actual)
        predicted_total += len(inferred)
        overlaps = sorted(((iou(a, b), ai, bi) for ai, a in enumerate(actual) for bi, b in enumerate(inferred)), reverse=True)
        used_a, used_b = set(), set()
        for overlap, ai, bi in overlaps:
            if overlap < 0.5:
                break
            if ai in used_a or bi in used_b:
                continue
            used_a.add(ai)
            used_b.add(bi)
            matched += 1
            matched_ious.append(overlap)
            a, b = center(actual[ai]), center(inferred[bi])
            errors.append(math.hypot((a[0] - b[0]) * WIDTH, (a[1] - b[1]) * HEIGHT))
        complete += len(used_a) == len(actual)
        misses_at_nonanchors += len(actual) - len(used_a)
    return {
        "frames_checked": len(reference) - len(ignored_frames),
        "frames_with_dense_players": frames_with_gt,
        "dense_boxes": reference_total,
        "interpolated_boxes": predicted_total,
        "matches_iou_0_5": matched,
        "missed_dense_boxes": misses_at_nonanchors,
        "unmatched_predicted_boxes": predicted_total - matched,
        "dense_box_coverage_percent": round(100 * matched / reference_total, 2) if reference_total else None,
        "prediction_match_percent": round(100 * matched / predicted_total, 2) if predicted_total else None,
        "frames_all_dense_boxes_matched_percent": round(100 * complete / (len(reference) - len(ignored_frames)), 2),
        "median_center_error_pixels_matched": round(statistics.median(errors), 2) if errors else None,
        "p95_center_error_pixels_matched": round(sorted(errors)[int(0.95 * (len(errors) - 1))], 2) if errors else None,
        "mean_iou_matched": round(statistics.mean(matched_ious), 3) if matched_ious else None,
    }


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    dense = load_frames(folder / "stride_1.jsonl")
    length = max(dense) + 1
    results = {"source_frames": length, "fps": 30, "dense_observation_count": sum(map(len, dense.values())), "profiles": {}}
    for stride in (1, 2, 5, 10):
        sampled = load_frames(folder / f"stride_{stride}.jsonl")
        expected = set(range(0, length, stride))
        if set(sampled) != expected:
            raise RuntimeError(f"Unexpected sampling frames for {stride}")
        for frame, boxes in sampled.items():
            if len(boxes) != len(dense[frame]) or any(
                max(abs(a-b) for a,b in zip(left,right)) > 1e-5 for left,right in zip(boxes,dense[frame])
            ):
                raise RuntimeError(f"Detector output diverged for stride {stride} at {frame}")
        reconstructed, pair_count = interpolate_rows(sampled, length)
        ignored = set(sampled) if stride != 1 else set()
        quality = compare(dense, reconstructed, ignored) if stride != 1 else {"dense_box_coverage_percent": 100.0, "prediction_match_percent": 100.0}
        results["profiles"][str(stride)] = {
            "detector_frames": len(sampled),
            "interpolation_pairs": pair_count,
            "quality_on_intermediate_frames": quality,
        }
    output = folder / "quality.json"
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
