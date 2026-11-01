
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from road_dataset import RoadDamageDataset
from road_transforms import augment_image

# Load one training sample.
dataset = RoadDamageDataset("validation")
image, target = dataset[0]

# Reopen the original image without augmentation.
from PIL import Image
from road_dataset import IMAGE_DIR

row = dataset.rows[0]
original = Image.open(IMAGE_DIR / row["image"]).convert("RGB")

original_boxes = target["boxes"].clone()

# Apply augmentation twice for comparison.
augmented_1, boxes_1 = augment_image(
    original.copy(), original_boxes.clone()
)

augmented_2, boxes_2 = augment_image(
    original.copy(), original_boxes.clone()
)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

samples = [
    (original, original_boxes, "Original"),
    (augmented_1, boxes_1, "Augmentation 1"),
    (augmented_2, boxes_2, "Augmentation 2")
]

for ax, (img, boxes, title) in zip(axes, samples):

    ax.imshow(img)

    for box in boxes:
        xmin, ymin, xmax, ymax = box.tolist()

        rectangle = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rectangle)

    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    "results/augmentation_verification.png",
    dpi=150
)
plt.show()

print("Original boxes:", original_boxes.tolist())
print("Augmentation 1 boxes:", boxes_1.tolist())
print("Augmentation 2 boxes:", boxes_2.tolist())
print("Augmentation test completed.")
