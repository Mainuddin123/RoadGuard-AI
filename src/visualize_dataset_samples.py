
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from road_dataset import RoadDamageDataset, CLASS_NAMES

dataset = RoadDamageDataset("train")

selected = {}
required_classes = {1, 2, 3}

# Find one image containing each class.
for index in range(len(dataset)):
    image, target = dataset[index]

    for class_id in target["labels"].tolist():
        if class_id in required_classes and class_id not in selected:
            selected[class_id] = (image, target)

    if len(selected) == 3:
        break

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, class_id in zip(axes, [1, 2, 3]):

    if class_id not in selected:
        ax.set_title(f"{CLASS_NAMES[class_id]} not found")
        ax.axis("off")
        continue

    image, target = selected[class_id]

    ax.imshow(image.permute(1, 2, 0).numpy())

    for box, label in zip(
        target["boxes"],
        target["labels"]
    ):
        if label.item() != class_id:
            continue

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

        ax.text(
            xmin,
            max(0, ymin - 5),
            CLASS_NAMES[class_id],
            color="white",
            backgroundcolor="red"
        )

    ax.set_title(CLASS_NAMES[class_id].upper())
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    "results/dataset_box_verification.png",
    dpi=150
)
plt.show()

print("Visual annotation verification completed.")
