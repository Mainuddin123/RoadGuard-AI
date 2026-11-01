
from pathlib import Path
from collections import Counter, defaultdict
import csv

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "dataset/annotations/clean_manifest.csv"
GROUPS = ROOT / "dataset/annotations/sequence_groups.csv"

with SOURCE.open(encoding="utf-8") as f:
    source_rows = list(csv.DictReader(f))

with GROUPS.open(encoding="utf-8") as f:
    group_rows = list(csv.DictReader(f))

source_images = [row["image"] for row in source_rows]
group_images = [row["image"] for row in group_rows]

source_set = set(source_images)
group_set = set(group_images)

group_sizes = Counter(row["group_id"] for row in group_rows)

methods = defaultdict(Counter)

for row in group_rows:
    methods[row["group_method"]][row["group_id"]] += 1

print("\n===== SEQUENCE GROUP VALIDATION =====")
print("Source images:", len(source_images))
print("Grouped images:", len(group_images))
print("Unique grouped images:", len(group_set))
print("Duplicate assignments:", len(group_images) - len(group_set))
print("Missing images:", len(source_set - group_set))
print("Unexpected images:", len(group_set - source_set))
print("Total groups:", len(group_sizes))

print("\nLargest 10 groups:")

for group_id, size in group_sizes.most_common(10):
    method = next(
        row["group_method"]
        for row in group_rows
        if row["group_id"] == group_id
    )
    print(f"{group_id}: {size} images | {method}")

print("\nGroup-size distribution:")

for size in sorted(set(group_sizes.values())):
    count = sum(1 for value in group_sizes.values() if value == size)
    print(f"{size} images: {count} groups")

print("\nValidation completed.")
