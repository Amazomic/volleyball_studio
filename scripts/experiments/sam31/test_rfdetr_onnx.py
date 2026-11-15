import unittest
import numpy as np
from run_rfdetr_onnx import decode_person


class RFDETRONNXTest(unittest.TestCase):
    def test_sparse_coco_person_index_one(self):
        # Raw ONNX: normalized cxcywh boxes plus pre-sigmoid logits.
        boxes = np.array([[[0.5, 0.5, 0.2, 0.6], [0.1, 0.1, 0.1, 0.1]]], dtype=np.float32)
        logits = np.full((1, 2, 91), -10, dtype=np.float32)
        logits[0, 0, 1] = 5.0
        logits[0, 1, 18] = 6.0  # dog, not a human
        found = decode_person(boxes, logits, 0.28)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["bbox"], [0.4, 0.2, 0.6, 0.8])

    def test_non_finite_boxes_rejected(self):
        boxes = np.array([[[np.nan, 0.5, 0.1, 0.1]]], dtype=np.float32)
        logits = np.zeros((1, 1, 91), dtype=np.float32)
        logits[0, 0, 1] = 10.0
        self.assertEqual(decode_person(boxes, logits, 0.28), [])


if __name__ == "__main__":
    unittest.main()
