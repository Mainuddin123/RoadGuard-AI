
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = (
    PROJECT_DIR / "dataset" / "raw" / "primary" / "data" / "images"
)

LABEL_DIR = (
    PROJECT_DIR / "dataset" / "annotations" / "primary_yolo_clipped"
)

CLASS_NAMES = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}

COLORS = {
    0: "red",
    1: "lime",
    2: "cyan"
}

# Select four reproducible sample images.
random.seed(42)
label_files = sorted(LABEL_DIR.glob("*.txt"))
samples = random.sample(label_files, 4)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, label_file in zip(axes, samples):

    image_path = IMAGE_DIR / f"{label_file.stem}.jpg"

    if not image_path.exists():
        ax.set_title(f"Image not found: {label_file.stem}")
        ax.axis("off")
        continue

    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    ax.imshow(image)

    for line in label_file.read_text().splitlines():

        if not line.strip():
            continue

        class_id, x, y, w, h = line.split()
        class_id = int(class_id)
        x, y, w, h = map(float, (x, y, w, h))

        left = (x - w / 2) * image_width
        top = (y - h / 2) * image_height

        box_width = w * image_width
        box_height = h * image_height

        rectangle = patches.Rectangle(
            (left, top),
            box_width,
            box_height,
            linewidth=2,
            edgecolor=COLORS[class_id],
            facecolor="none"
        )

        ax.add_patch(rectangle)

        ax.text(
            left,
            max(0, top - 5),
            CLASS_NAMES[class_id],
            color="white",
            fontsize=9,
            bbox=dict(facecolor=COLORS[class_id], alpha=0.8)
        )

    ax.set_title(label_file.stem)
    ax.axis("off")

plt.tight_layout()

output_path = PROJECT_DIR / "results" / "dataset_samples.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Visualization saved at: {output_path}")
