
from pathlib import Path
from collections import Counter
from PIL import Image
import csv

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset/raw/primary/data/images"
MANIFEST = ROOT / "dataset/annotations/final_split_manifest.csv"

with MANIFEST.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

dimensions = Counter()
formats = Counter()
errors = []

for row in rows:
    image_path = IMAGE_DIR / row["image"]

    try:
        with Image.open(image_path) as image:
            dimensions[image.size] += 1
            formats[image.format] += 1

    except Exception as error:
        errors.append((row["image"], str(error)))

print("\n===== IMAGE DIMENSION INSPECTION =====")
print("Total images:", len(rows))
print("\nImage dimensions:")

for size, count in dimensions.most_common():
    print(f"{size[0]} x {size[1]}: {count} images")

print("\nImage formats:", dict(formats))
print("Unreadable images:", len(errors))

for filename, error in errors[:10]:
    print(filename, error)
