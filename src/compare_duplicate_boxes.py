
from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "dataset/raw/primary/data/images"
LABEL_DIR = ROOT / "dataset/annotations/primary_yolo_clipped"

def read_boxes(path):
    boxes = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        cls, x, y, w, h = line.split()
        x, y, w, h = map(float, (x, y, w, h))

        boxes.append({
            "class": int(cls),
            "box": (
                x - w / 2, y - h / 2,
                x + w / 2, y + h / 2
            )
        })

    return boxes


def calculate_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


groups = defaultdict(list)

for image in IMAGE_DIR.glob("*.jpg"):
    digest = hashlib.md5(image.read_bytes()).hexdigest()
    groups[digest].append(image)

duplicates = [files for files in groups.values() if len(files) > 1]

for number, files in enumerate(duplicates, 1):
    first = read_boxes(LABEL_DIR / f"{files[0].stem}.txt")
    second = read_boxes(LABEL_DIR / f"{files[1].stem}.txt")

    candidates = []

    for i, a in enumerate(first):
        for j, b in enumerate(second):
            if a["class"] == b["class"]:
                iou = calculate_iou(a["box"], b["box"])
                if iou >= 0.5:
                    candidates.append((iou, i, j))

    candidates.sort(reverse=True)

    matched_first = set()
    matched_second = set()

    for iou, i, j in candidates:
        if i not in matched_first and j not in matched_second:
            matched_first.add(i)
            matched_second.add(j)

    print(f"\nDUPLICATE GROUP {number}")
    print("First image:", files[0].name)
    print("Second image:", files[1].name)
    print("First annotation count:", len(first))
    print("Second annotation count:", len(second))
    print("Matched boxes (same class, IoU >= 0.5):",
          len(matched_first))
    print("Unmatched first boxes:", len(first) - len(matched_first))
    print("Unmatched second boxes:", len(second) - len(matched_second))

print("\nComparison completed.")
