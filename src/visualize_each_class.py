
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset/raw/primary/data/images"
LABEL_DIR = ROOT / "dataset/annotations/primary_yolo_clipped"

CLASSES = {0: "Pothole", 1: "Crack", 2: "Manhole"}
COLORS = {0: "red", 1: "lime", 2: "cyan"}

random.seed(42)

candidates = {0: [], 1: [], 2: []}

for label_file in LABEL_DIR.glob("*.txt"):
    class_ids = set()

    for line in label_file.read_text().splitlines():
        if line.strip():
            class_ids.add(int(line.split()[0]))

    for class_id in class_ids:
        candidates[class_id].append(label_file)

fig, axes = plt.subplots(3, 3, figsize=(16, 13))

for row, class_id in enumerate(CLASSES):
    selected = random.sample(candidates[class_id], 3)

    for col, label_file in enumerate(selected):
        ax = axes[row, col]
        image_path = IMAGE_DIR / f"{label_file.stem}.jpg"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        ax.imshow(image)

        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue

            cid, x, y, w, h = line.split()
            cid = int(cid)

            if cid != class_id:
                continue

            x, y, w, h = map(float, (x, y, w, h))

            rect = patches.Rectangle(
                ((x - w / 2) * width, (y - h / 2) * height),
                w * width,
                h * height,
                linewidth=2,
                edgecolor=COLORS[cid],
                facecolor="none"
            )

            ax.add_patch(rect)

        ax.set_title(f"{CLASSES[class_id]}: {label_file.stem}")
        ax.axis("off")

plt.tight_layout()
plt.savefig(ROOT / "results" / "class_samples_random.png", dpi=150)
plt.show()

print("Random class visualization completed.")
