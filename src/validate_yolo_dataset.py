
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset" / "yolo_dataset"

SPLITS = ["train", "val", "test"]

CLASS_NAMES = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_split(split):

    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    print(f"\n========== {split.upper()} ==========")

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Missing image or label directory for {split}"
        )

    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    }

    labels = {
        path.stem: path
        for path in label_dir.glob("*.txt")
    }

    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))

    invalid_lines = []
    invalid_classes = []
    invalid_boxes = []

    class_counts = Counter()
    total_objects = 0

    for stem, label_path in labels.items():

        with label_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(file, start=1):

                values = line.strip().split()

                if not values:
                    continue

                if len(values) != 5:
                    invalid_lines.append(
                        f"{label_path.name}:{line_number}"
                    )
                    continue

                try:
                    class_id = int(values[0])
                    x_center, y_center, width, height = map(
                        float, values[1:]
                    )
                except ValueError:
                    invalid_lines.append(
                        f"{label_path.name}:{line_number}"
                    )
                    continue

                if class_id not in CLASS_NAMES:
                    invalid_classes.append(
                        f"{label_path.name}:{line_number}"
                    )
                    continue

                coordinates = [
                    x_center, y_center, width, height
                ]

                valid_coordinates = all(
                    0.0 <= value <= 1.0
                    for value in coordinates
                )

                if (
                    not valid_coordinates
                    or width <= 0
                    or height <= 0
                    or x_center - width / 2 < -1e-6
                    or y_center - height / 2 < -1e-6
                    or x_center + width / 2 > 1.000001
                    or y_center + height / 2 > 1.000001
                ):
                    invalid_boxes.append(
                        f"{label_path.name}:{line_number}"
                    )
                    continue

                class_counts[class_id] += 1
                total_objects += 1

    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Missing labels:", len(missing_labels))
    print("Missing images:", len(missing_images))
    print("Invalid annotation lines:", len(invalid_lines))
    print("Invalid class IDs:", len(invalid_classes))
    print("Invalid bounding boxes:", len(invalid_boxes))
    print("Total valid objects:", total_objects)

    print("\nClass distribution:")

    for class_id, class_name in CLASS_NAMES.items():
        print(
            f"{class_name}: {class_counts[class_id]}"
        )

    return {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "invalid_lines": invalid_lines,
        "invalid_classes": invalid_classes,
        "invalid_boxes": invalid_boxes,
        "objects": total_objects
    }


if __name__ == "__main__":

    print("ROADGUARD AI - YOLO DATASET VALIDATION")

    results = {}

    for split in SPLITS:
        results[split] = validate_split(split)

    print("\n========== FINAL VALIDATION ==========")

    total_images = sum(
        result["images"]
        for result in results.values()
    )

    total_objects = sum(
        result["objects"]
        for result in results.values()
    )

    errors = sum(
        len(result[key])
        for result in results.values()
        for key in [
            "missing_labels",
            "missing_images",
            "invalid_lines",
            "invalid_classes",
            "invalid_boxes"
        ]
    )

    print("Total images:", total_images)
    print("Total valid objects:", total_objects)
    print("Total validation errors:", errors)

    if total_images != 1999:
        print("WARNING: Expected 1999 images.")

    if errors == 0 and total_images == 1999:
        print("\nYOLO DATASET VALIDATION PASSED")
    else:
        print("\nYOLO DATASET VALIDATION FAILED")
        print("Fix the reported issues before training.")
