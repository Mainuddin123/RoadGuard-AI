
from pathlib import Path
import csv
from road_transforms import augment_image
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor


ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset/raw/primary/data/images"
LABEL_DIR = ROOT / "dataset/annotations/primary_yolo_clipped"
MANIFEST = ROOT / "dataset/annotations/final_split_manifest.csv"

CLASS_NAMES = {
    1: "pothole",
    2: "crack",
    3: "manhole"
}


class RoadDamageDataset(Dataset):

    def __init__(self, split):
        if split not in ["train", "validation", "test"]:
            raise ValueError("Invalid dataset split")

        with MANIFEST.open(encoding="utf-8") as file:
            all_rows = list(csv.DictReader(file))

        self.rows = [
            row for row in all_rows
            if row["split"] == split
        ]

        self.split = split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):

        row = self.rows[index]

        image_path = IMAGE_DIR / row["image"]
        label_path = LABEL_DIR / row["annotation"]

        # Load image as RGB.
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size


        boxes = []
        labels = []

        # Read YOLO annotations.
        with label_path.open(encoding="utf-8") as file:

            for line in file:

                values = line.strip().split()

                if not values:
                    continue

                class_id = int(values[0])

                x_center, y_center, width, height = map(
                    float, values[1:]
                )

                # Convert normalized YOLO coordinates
                # into absolute pixel coordinates.
                x_center *= image_width
                y_center *= image_height
                width *= image_width
                height *= image_height

                xmin = max(0, x_center - width / 2)
                ymin = max(0, y_center - height / 2)
                xmax = min(image_width, x_center + width / 2)
                ymax = min(image_height, y_center + height / 2)

                if xmax <= xmin or ymax <= ymin:
                    continue

                # Torchvision reserves label 0 for background.
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id + 1)

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)
        # Augment training images only.
        if self.split == "train":
            image, boxes = augment_image(image, boxes)
        # Convert image to tensor after augmentation.
        image_tensor = to_tensor(image)
        
        area = (
            (boxes[:, 2] - boxes[:, 0]) *
            (boxes[:, 3] - boxes[:, 1])
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": torch.zeros(
                (len(labels),), dtype=torch.int64
            )
        }

        return image_tensor, target


def detection_collate(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":

    train_dataset = RoadDamageDataset("train")
    validation_dataset = RoadDamageDataset("validation")
    test_dataset = RoadDamageDataset("test")

    print("\n===== ROADGUARD DATASET LOADER =====")
    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Testing images:", len(test_dataset))

    # Test one sample.
    image, target = train_dataset[0]

    print("\n===== SAMPLE INSPECTION =====")
    print("Image shape:", image.shape)
    print("Bounding boxes:", target["boxes"].shape)
    print("Labels:", target["labels"].tolist())
    print("Class names:", [
        CLASS_NAMES[label]
        for label in target["labels"].tolist()
    ])

    print("\nFirst 5 bounding boxes:")
    print(target["boxes"][:5])

    # Test DataLoader.
    loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=detection_collate
    )

    images, targets = next(iter(loader))

    print("\n===== DATALOADER TEST =====")
    print("Batch images:", len(images))
    print("First image shape:", images[0].shape)
    print("Second image shape:", images[1].shape)
    print("DataLoader working successfully!")
