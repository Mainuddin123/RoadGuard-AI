
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_DIR = Path(__file__).resolve().parent.parent
LABEL_DIR = PROJECT_DIR / "dataset" / "raw" / "primary" / "data" / "labels-YOLO"
IMAGE_DIR = PROJECT_DIR / "dataset" / "raw" / "primary" / "data" / "images"

class_names = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}

issues = []

for label_file in LABEL_DIR.glob("*.txt"):
    for line_no, line in enumerate(label_file.read_text().splitlines(), 1):
        if not line.strip():
            continue

        values = line.split()
        if len(values) != 5:
            continue

        class_id = int(values[0])
        x, y, w, h = map(float, values[1:])

        left = x - w / 2
        right = x + w / 2
        top = y - h / 2
        bottom = y + h / 2

        overflow = max(0, -left, right - 1, -top, bottom - 1)

        if overflow > 0.005:
            issues.append((overflow, label_file, line_no,
                           class_id, x, y, w, h))

issues.sort(key=lambda item: item[0], reverse=True)
selected = issues[:7]

fig, axes = plt.subplots(4, 2, figsize=(15, 18))
axes = axes.flatten()

for ax, item in zip(axes, selected):
    overflow, label_file, line_no, class_id, x, y, w, h = item

    image_path = IMAGE_DIR / f"{label_file.stem}.jpg"

    if not image_path.exists():
        ax.set_title(f"Image not found: {label_file.stem}")
        ax.axis("off")
        continue

    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    left = (x - w / 2) * image_width
    top = (y - h / 2) * image_height
    right = (x + w / 2) * image_width
    bottom = (y + h / 2) * image_height

    # Clip only the displayed rectangle, not the original annotation.
    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)

    ax.imshow(image)
    rectangle = patches.Rectangle(
        (left, top),
        right - left,
        bottom - top,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax.add_patch(rectangle)

    ax.set_title(
        f"{label_file.name}\n"
        f"{class_names[class_id]} | Overflow: {overflow:.5f}"
    )
    ax.axis("off")

for ax in axes[len(selected):]:
    ax.axis("off")

plt.tight_layout()

output_path = PROJECT_DIR / "results" / "boundary_review.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Review image saved to: {output_path}")
