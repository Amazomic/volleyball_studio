import json
import tempfile
import unittest
from pathlib import Path
from run_rfdetr import person_boxes
from compare_variants import compare


class Detections:
    xyxy = [[64, 72, 128, 300], [0, 0, 120, 50]]
    confidence = [0.83, 0.9]
    data = {"class_name": ["person", "ball"]}


class RFDetrExperimentTest(unittest.TestCase):
    def test_person_filter_and_normalization(self):
        output = person_boxes(Detections(), 640, 360)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["bbox"], [0.1, 0.2, 0.2, 0.8333333])
        self.assertEqual(output[0]["id"], 0)
        self.assertEqual(output[0]["score"], 0.83)

    def test_comparison_rejects_different_video_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "base.jsonl", Path(directory) / "medium.jsonl"
            for file, sha in ((a, "a"), (b, "b")):
                file.write_text(json.dumps({"frame": 0, "tracks": []}) + "\n")
                file.with_suffix(".stats.json").write_text(
                    json.dumps({"frames": 1, "video_sha256": sha, "fps": 1.0}))
            with self.assertRaisesRegex(ValueError, "inconsistent video"):
                compare(a, ["Medium=" + str(b)])


if __name__ == "__main__":
    unittest.main()
