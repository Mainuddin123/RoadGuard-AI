
from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset/raw/primary/data/images"
LABEL_DIR = ROOT / "dataset/annotations/primary_yolo_clipped"

groups = defaultdict(list)

for image_path in IMAGE_DIR.glob("*.jpg"):
    digest = hashlib.md5(image_path.read_bytes()).hexdigest()
    groups[digest].append(image_path)

duplicates = [files for files in groups.values() if len(files) > 1]

print("\n===== DUPLICATE ANNOTATION COMPARISON =====")

for group_number, files in enumerate(duplicates, start=1):
    print(f"\nDuplicate group {group_number}:")

    reference_label = LABEL_DIR / f"{files[0].stem}.txt"

    if not reference_label.exists():
        print("Reference annotation missing!")
        continue

    reference_content = reference_label.read_text().strip()

    for image_file in files:
        label_file = LABEL_DIR / f"{image_file.stem}.txt"

        if not label_file.exists():
            print(f"{image_file.name}: Annotation missing")
            continue

        content = label_file.read_text().strip()

        if content == reference_content:
            print(f"{image_file.name}: SAME annotations")
        else:
            print(f"{image_file.name}: DIFFERENT annotations")

print("\nComparison completed.")
