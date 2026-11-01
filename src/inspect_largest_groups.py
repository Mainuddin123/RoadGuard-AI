
from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import csv
import re

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset/raw/primary/data/images"
GROUP_FILE = ROOT / "dataset/annotations/sequence_groups.csv"

with GROUP_FILE.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

counts = Counter(row["group_id"] for row in rows)

largest_groups = [
    group for group, count in counts.most_common(3)
]

fig, axes = plt.subplots(3, 8, figsize=(20, 11))

def timestamp(filename):
    match = re.search(
        r"(\d{4})-(\d{2})-(\d{2})-(\d{2})h"
        r"(\d{2})m(\d{2})s(\d+)",
        filename
    )

    if match:
        return tuple(map(int, match.groups()))

    match = re.search(r"vlcsnap-(\d+)", filename)

    if match:
        return (int(match.group(1)),)

    return (filename,)


for row_index, group_id in enumerate(largest_groups):

    group_images = [
        row["image"] for row in rows
        if row["group_id"] == group_id
    ]

    group_images.sort(key=timestamp)

    step = max(1, len(group_images) // 8)
    samples = group_images[::step][:8]

    for col_index, filename in enumerate(samples):
        ax = axes[row_index, col_index]

        image_path = IMAGE_DIR / filename
        image = Image.open(image_path).convert("RGB")

        ax.imshow(image)
        ax.set_title(filename[:22], fontsize=7)
        ax.axis("off")

    axes[row_index, 0].set_ylabel(
        f"{group_id}\n{len(group_images)} images",
        fontsize=9
    )

plt.suptitle("Largest Sequence Groups — Visual Review")
plt.tight_layout()

output = ROOT / "results" / "largest_groups_review.png"
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.show()

print("Saved:", output)
