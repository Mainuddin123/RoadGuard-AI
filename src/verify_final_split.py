
from pathlib import Path
from collections import Counter, defaultdict
import csv

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "dataset/annotations/clean_manifest.csv"
SPLIT = ROOT / "dataset/annotations/final_split_manifest.csv"

with SOURCE.open(encoding="utf-8") as f:
    source = list(csv.DictReader(f))

with SPLIT.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

source_images = {r["image"] for r in source}
split_images = [r["image"] for r in rows]

group_splits = defaultdict(set)

for row in rows:
    group_splits[row["group_id"]].add(row["split"])

print("\n===== FINAL SPLIT VERIFICATION =====")
print("Source images:", len(source_images))
print("Split images:", len(split_images))
print("Unique split images:", len(set(split_images)))
print("Missing images:", len(source_images - set(split_images)))
print("Unexpected images:", len(set(split_images) - source_images))
print("Duplicate images:", len(split_images) - len(set(split_images)))

leaks = {
    group: splits
    for group, splits in group_splits.items()
    if len(splits) > 1
}

print("Groups appearing in multiple splits:", len(leaks))

for split in ["train", "validation", "test"]:
    selected = [r for r in rows if r["split"] == split]

    print(
        f"{split.upper()}: "
        f"{len(selected)} images, "
        f"{len({r['group_id'] for r in selected})} groups"
    )

print("\nVerification completed.")
