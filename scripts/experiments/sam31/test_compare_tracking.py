import json
import tempfile
import unittest
from pathlib import Path
from compare_tracking import read_track_file, summarize, run


def frames_for(*tracks):
    return [{"frame": n, "tracks": [
        {"id": identity, "bbox": [0.1, 0.1, 0.3, 0.5]}
        for identity, seen in tracks if n in seen
    ]} for n in range(6)]


class MetricsTest(unittest.TestCase):
    def test_missing_gap_is_counted_for_confirmed_id(self):
        rows = frames_for(("a", [0, 1, 4, 5]), ("b", [2, 3]))
        data = summarize(rows, min_observations=4, expected_persons=1)
        self.assertEqual(data["all_provisional_ids"], 2)
        self.assertEqual(data["ids_at_least_min_observations"], 1)
        self.assertEqual(data["track_gap_events"], 1)
        self.assertEqual(data["track_missing_frames_inside_lifespan"], 2)

    def test_rejects_mismatched_length(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "base.jsonl"
            right = Path(directory) / "sam.jsonl"
            left.write_text("\n".join(map(json.dumps, frames_for(("a", [0, 1])))) + "\n")
            right.write_text("\n".join(map(json.dumps, frames_for(("b", [0, 1]))[:5])) + "\n")
            with self.assertRaisesRegex(ValueError, "different frame ranges"):
                run(left, right)

    def test_validates_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            sample = frames_for(("a", [0]), ("a", [0]))
            path.write_text("\n".join(map(json.dumps, sample)) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_track_file(path)


if __name__ == "__main__":
    unittest.main()
