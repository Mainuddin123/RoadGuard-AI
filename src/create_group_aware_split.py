
from pathlib import Path
from collections import defaultdict, Counter
import csv
import random

ROOT = Path(__file__).resolve().parent.parent

GROUP_FILE = ROOT / "dataset/annotations/sequence_groups.csv"
LABEL_DIR = ROOT / "dataset/annotations/primary_yolo_clipped"
OUTPUT_FILE = ROOT / "dataset/annotations/final_split_manifest.csv"

RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15
}

SPLITS = list(RATIOS)
random.seed(42)

# Read image and group information.
with GROUP_FILE.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)

for row in rows:
    groups[row["group_id"]].append(row)

# Calculate image and object counts for every group.
stats = {}

for group_id, images in groups.items():
    counts = [len(images), 0, 0, 0]

    for image in images:
        label_file = LABEL_DIR / image["annotation"]

        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue

            class_id = int(line.split()[0])
            counts[class_id + 1] += 1

    stats[group_id] = counts

total = [
    sum(values[i] for values in stats.values())
    for i in range(4)
]

targets = {
    split: [total[i] * RATIOS[split] for i in range(4)]
    for split in SPLITS
}

def calculate_score(assignment):
    actual = {
        split: [0, 0, 0, 0]
        for split in SPLITS
    }

    for group_id, split in assignment.items():
        for i in range(4):
            actual[split][i] += stats[group_id][i]

    score = 0.0

    for split in SPLITS:
        for i in range(4):
            target = max(targets[split][i], 1)
            difference = (
                actual[split][i] - targets[split][i]
            ) / target

            weight = 2.0 if i == 0 else 1.0
            score += weight * difference ** 2

        # Penalize a split missing any object class.
        for class_index in range(1, 4):
            if actual[split][class_index] == 0:
                score += 1000

    return score


# Generate candidate group assignments.
best_assignment = None
best_score = float("inf")

group_ids = list(groups)

for _ in range(30000):
    assignment = {}

    for group_id in group_ids:
        assignment[group_id] = random.choices(
            SPLITS,
            weights=[0.70, 0.15, 0.15],
            k=1
        )[0]

    score = calculate_score(assignment)

    if score < best_score:
        best_score = score
        best_assignment = assignment.copy()


# Improve the best assignment by moving or swapping groups.
changed = True

while changed:
    changed = False
    current_score = calculate_score(best_assignment)
    improved_assignment = best_assignment.copy()
    improved_score = current_score

    # Try moving one complete group.
    for group_id in group_ids:
        old_split = best_assignment[group_id]

        for new_split in SPLITS:
            if new_split == old_split:
                continue

            candidate = best_assignment.copy()
            candidate[group_id] = new_split

            score = calculate_score(candidate)

            if score < improved_score:
                improved_score = score
                improved_assignment = candidate

    # Try swapping two complete groups.
    for i, first in enumerate(group_ids):
        for second in group_ids[i + 1:]:
            if best_assignment[first] == best_assignment[second]:
                continue

            candidate = best_assignment.copy()
            candidate[first], candidate[second] = (
                candidate[second], candidate[first]
            )

            score = calculate_score(candidate)

            if score < improved_score:
                improved_score = score
                improved_assignment = candidate

    if improved_score < current_score:
        best_assignment = improved_assignment
        changed = True


# Save image-level split manifest.
output_rows = []

for group_id, images in groups.items():
    split = best_assignment[group_id]

    for image in images:
        output_rows.append({
            "image": image["image"],
            "annotation": image["annotation"],
            "group_id": group_id,
            "split": split
        })

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image", "annotation", "group_id", "split"]
    )
    writer.writeheader()
    writer.writerows(output_rows)


# Print split statistics.
print("\n===== GROUP-AWARE DATASET SPLIT =====")

for split in SPLITS:
    selected = [
        row for row in output_rows
        if row["split"] == split
    ]

    image_names = [row["image"] for row in selected]

    group_count = len({
        row["group_id"] for row in selected
    })

    class_counts = Counter()

    for row in selected:
        label_file = LABEL_DIR / row["annotation"]

        for line in label_file.read_text().splitlines():
            if line.strip():
                class_counts[int(line.split()[0])] += 1

    print(f"\n{split.upper()}")
    print("Images:", len(image_names))
    print("Groups:", group_count)
    print("Percentage:", round(
        len(image_names) / len(rows) * 100, 2
    ))
    print("Potholes:", class_counts[0])
    print("Cracks:", class_counts[1])
    print("Manholes:", class_counts[2])

print("\nTotal images:", len(output_rows))
print("Output:", OUTPUT_FILE)
print("Optimization score:", round(best_score, 6))
