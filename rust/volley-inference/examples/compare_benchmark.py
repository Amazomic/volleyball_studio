import re
import sys

pattern = re.compile(r"bbox: \[([^]]+)\], score: ([^,]+)")
with open(sys.argv[1], encoding="utf-8") as f:
    first = f.readlines()
with open(sys.argv[2], encoding="utf-8") as f:
    second = f.readlines()
if len(first) != len(second):
    raise ValueError("Frame counts differ")
counts = pairs = over = 0
max_box = max_score = 0.0
for left, right in zip(first, second):
    left_items = pattern.findall(left)
    right_items = pattern.findall(right)
    counts += len(left_items) != len(right_items)
    for (left_box, left_score), (right_box, right_score) in zip(left_items, right_items):
        a = list(map(float, left_box.split(",")))
        b = list(map(float, right_box.split(",")))
        delta = max(abs(x - y) for x, y in zip(a, b))
        max_box = max(max_box, delta)
        max_score = max(max_score, abs(float(left_score) - float(right_score)))
        over += delta > 0.005
        pairs += 1
print(f"frames={len(first)} count_changed={counts} pairs={pairs} max_box_delta={max_box:.8f} boxes_over_0.005={over} max_score_delta={max_score:.8f}")
