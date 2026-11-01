
from pathlib import Path
from collections import defaultdict
import hashlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset/raw/primary/data/images"
LABEL_DIR = ROOT / "dataset/annotations/primary_yolo_clipped"

CLASS_NAMES = {0: "Pothole", 1: "Crack", 2: "Manhole"}

groups = defaultdict(list)

for image_path in IMAGE_DIR.glob("*.jpg"):
    digest = hashlib.md5(image_path.read_bytes()).hexdigest()
    groups[digest].append(image_path)

duplicates = [
    files for files in groups.values() if len(files) > 1
]

fig, axes = plt.subplots(
    len(duplicates), 2,
    figsize=(16, 5 * len(duplicates))
)

for row, files in enumerate(duplicates):

    image_path = files[0]
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    for col, file in enumerate(files):
        ax = axes[row, col]

        ax.imshow(image)
        label_path = LABEL_DIR / f"{file.stem}.txt"

        if not label_path.exists():
            ax.set_title(f"{file.name} - LABEL MISSING")
            ax.axis("off")
            continue

        lines = label_path.read_text().splitlines()
        object_count = 0

        for line in lines:
            if not line.strip():
                continue

            class_id, x, y, w, h = line.split()
            class_id = int(class_id)
            x, y, w, h = map(float, (x, y, w, h))

            left = (x - w / 2) * width
            top = (y - h / 2) * height

            rectangle = patches.Rectangle(
                (left, top),
                w * width,
                h * height,
                linewidth=2,
                edgecolor="yellow" if col == 0 else "magenta",
                facecolor="none"
            )

            ax.add_patch(rectangle)
            object_count += 1

        ax.set_title(
            f"{file.name}\nObjects: {object_count} | "
            f"{'YELLOW' if col == 0 else 'MAGENTA'} boxes"
        )
        ax.axis("off")

plt.tight_layout()

output = ROOT / "results" / "duplicate_annotation_comparison.png"
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.show()

print("Comparison saved:", output)
