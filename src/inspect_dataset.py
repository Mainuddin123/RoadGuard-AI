
from pathlib import Path

# Project dataset location
PROJECT_DIR = Path(__file__).resolve().parent.parent
LABEL_DIR = PROJECT_DIR / "dataset" / "raw" / "primary" / "data" / "labels-YOLO"

total_objects = 0
empty_files = 0
invalid_annotations = []

for label_file in LABEL_DIR.glob("*.txt"):
    lines = label_file.read_text().splitlines()

    if not lines:
        empty_files += 1

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        total_objects += 1
        values = line.split()

        try:
            if len(values) != 5:
                raise ValueError("Expected exactly 5 values")

            class_id = int(values[0])
            x, y, width, height = map(float, values[1:])

            if class_id not in (0, 1, 2):
                raise ValueError("Unknown class ID")

            if not all(map(lambda v: 0 <= v <= 1, (x, y))):
                raise ValueError("Center coordinates outside 0–1")

            if not (0 < width <= 1 and 0 < height <= 1):
                raise ValueError("Invalid box width or height")

            if x - width / 2 < 0 or x + width / 2 > 1:
                raise ValueError("Bounding box exceeds image width")

            if y - height / 2 < 0 or y + height / 2 > 1:
                raise ValueError("Bounding box exceeds image height")

        except (ValueError, OverflowError) as error:
            invalid_annotations.append(
                f"{label_file.name}, line {line_number}: {error}"
            )

print("\n===== ROADGUARD AI DATASET VALIDATION =====")
print(f"Total annotation objects: {total_objects}")
print(f"Empty label files: {empty_files}")
print(f"Invalid annotations: {len(invalid_annotations)}")

if invalid_annotations:
    print("\nFirst 10 errors:")
    for error in invalid_annotations[:10]:
        print(error)
else:
    print("\nAll non-empty annotations passed validation.")
