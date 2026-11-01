
from pathlib import Path
from collections import defaultdict
import csv

ROOT = Path(__file__).resolve().parent.parent

GROUP_FILE = ROOT / "dataset/annotations/sequence_groups.csv"
LABEL_DIR = ROOT / "dataset/annotations/primary_yolo_clipped"

groups = defaultdict(lambda: {
    "images": 0,
    "pothole": 0,
    "crack": 0,
    "manhole": 0,
    "objects": 0
})

with GROUP_FILE.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    group_id = row["group_id"]
    groups[group_id]["images"] += 1

    label_file = LABEL_DIR / row["annotation"]

    for line in label_file.read_text().splitlines():
        if not line.strip():
            continue

        class_id = int(line.split()[0])

        if class_id == 0:
            groups[group_id]["pothole"] += 1
        elif class_id == 1:
            groups[group_id]["crack"] += 1
        elif class_id == 2:
            groups[group_id]["manhole"] += 1

        groups[group_id]["objects"] += 1

print("\n===== GROUP CLASS DISTRIBUTION =====")

for group_id, data in sorted(
    groups.items(),
    key=lambda item: item[1]["images"],
    reverse=True
):
    print(
        f"{group_id} | Images: {data['images']} | "
        f"Pothole: {data['pothole']} | "
        f"Crack: {data['crack']} | "
        f"Manhole: {data['manhole']} | "
        f"Objects: {data['objects']}"
    )

print("\nTotal groups:", len(groups))
