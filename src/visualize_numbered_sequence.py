
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import re
import math

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "dataset/raw/primary/data/images"

frames = []

for image in IMAGE_DIR.glob("*.jpg"):
    match = re.fullmatch(r"vlcsnap-(\d+)", image.stem, re.IGNORECASE)

    if match:
        frames.append((int(match.group(1)), image))

frames.sort(key=lambda item: item[0])

# Select 24 frames evenly across the sequence.
step = max(1, math.ceil(len(frames) / 24))
selected = frames[::step][:24]

fig, axes = plt.subplots(6, 4, figsize=(16, 17))
axes = axes.flatten()

for ax, (number, image_path) in zip(axes, selected):
    image = Image.open(image_path).convert("RGB")

    ax.imshow(image)
    ax.set_title(f"Frame {number:05d}")
    ax.axis("off")

for ax in axes[len(selected):]:
    ax.axis("off")

plt.suptitle("Numbered Road Dataset — Sequence Inspection")
plt.tight_layout()

output = ROOT / "results" / "numbered_sequence_review.png"
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.show()

print("Frames inspected:", len(selected))
print("Saved:", output)
