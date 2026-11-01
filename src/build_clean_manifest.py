
from pathlib import Path
from collections import defaultdict
import hashlib
import csv

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset/raw/primary/data/images"
LABEL_DIR = ROOT / "dataset/annotations/primary_yolo_clipped"
OUTPUT_DIR = ROOT / "dataset/annotations"

hash_groups = defaultdict(list)

for image_path in IMAGE_DIR.glob("*.jpg"):
    digest = hashlib.md5(image_path.read_bytes()).hexdigest()
    hash_groups[digest].append(image_path)

duplicate_files = set()

for files in hash_groups.values():
    if len(files) > 1:
        for file in files:
            duplicate_files.add(file.name)

manifest = []

for image_path in sorted(IMAGE_DIR.glob("*.jpg")):

    if image_path.name in duplicate_files:
        continue

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        print("Missing annotation:", image_path.name)
        continue

    class_counts = {0: 0, 1: 0, 2: 0}

    for line in label_path.read_text().splitlines():
        if line.strip():
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

    manifest.append({
        "image": image_path.name,
        "annotation": label_path.name,
        "pothole": class_counts[0],
        "crack": class_counts[1],
        "manhole": class_counts[2],
        "total_objects": sum(class_counts.values())
    })

output_file = OUTPUT_DIR / "clean_manifest.csv"

with output_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
    writer.writeheader()
    writer.writerows(manifest)

print("\n===== CLEAN DATASET MANIFEST =====")
print("Original images:", len(list(IMAGE_DIR.glob("*.jpg"))))
print("Excluded duplicate files:", len(duplicate_files))
print("Retained unique images:", len(manifest))
print("Manifest:", output_file)
print("\nClass object totals:")

for cls in ["pothole", "crack", "manhole"]:
    print(f"{cls}: {sum(row[cls] for row in manifest)}")
