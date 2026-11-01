
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

LABEL_DIR = (
    PROJECT_DIR / "dataset" / "annotations" / "primary_yolo_clipped"
)

total_objects = 0
invalid_objects = 0
empty_files = 0

for file in LABEL_DIR.glob("*.txt"):
    lines = file.read_text().splitlines()

    if not lines:
        empty_files += 1

    for line in lines:
        if not line.strip():
            continue

        values = line.split()

        try:
            if len(values) != 5:
                raise ValueError

            class_id = int(values[0])
            x, y, w, h = map(float, values[1:])

            if class_id not in (0, 1, 2):
                raise ValueError

            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError

            if not (0 < w <= 1 and 0 < h <= 1):
                raise ValueError

            if x - w / 2 < -1e-8 or x + w / 2 > 1 + 1e-8:
                raise ValueError

            if y - h / 2 < -1e-8 or y + h / 2 > 1 + 1e-8:
                raise ValueError

            total_objects += 1

        except ValueError:
            invalid_objects += 1

print("\n===== PREPARED DATASET VALIDATION =====")
print("Total valid objects:", total_objects)
print("Invalid objects:", invalid_objects)
print("Empty annotation files:", empty_files)
print("Annotation files:", len(list(LABEL_DIR.glob("*.txt"))))
