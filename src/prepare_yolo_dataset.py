
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IMAGE_SOURCE = ROOT / "dataset/raw/primary/data/images"

LABEL_SOURCE = (
    ROOT / "dataset/annotations/primary_yolo_clipped"
)

MANIFEST = ROOT / "dataset/annotations/final_split_manifest.csv"

OUTPUT = ROOT / "dataset/yolo_dataset"

SPLIT_MAPPING = {
    "train": "train",
    "validation": "val",
    "test": "test"
}

CLASS_MAPPING = {
    0: 0,  # Pothole
    1: 1,  # Crack
    2: 2   # Manhole
}


def prepare_dataset():

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    if not IMAGE_SOURCE.exists():
        raise FileNotFoundError(IMAGE_SOURCE)

    if not LABEL_SOURCE.exists():
        raise FileNotFoundError(LABEL_SOURCE)

    # Create output directories
    for split in SPLIT_MAPPING.values():

        (OUTPUT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )

    copied_images = 0
    copied_labels = 0

    with MANIFEST.open(
        "r",
        encoding="utf-8"
    ) as file:

        rows = list(csv.DictReader(file))

    print("ROADGUARD AI - YOLO DATASET PREPARATION")
    print("Manifest entries:", len(rows))

    for row in rows:

        source_image = IMAGE_SOURCE / row["image"]

        source_label = LABEL_SOURCE / row["annotation"]

        split = row["split"]

        if split not in SPLIT_MAPPING:
            raise ValueError(
                f"Unknown split: {split}"
            )

        if not source_image.exists():
            raise FileNotFoundError(source_image)

        if not source_label.exists():
            raise FileNotFoundError(source_label)

        destination_split = SPLIT_MAPPING[split]

        destination_image = (
            OUTPUT / "images" / destination_split
            / source_image.name
        )

        destination_label = (
            OUTPUT / "labels" / destination_split
            / f"{source_image.stem}.txt"
        )

        # Copy image without modifying original
        shutil.copy2(
            source_image,
            destination_image
        )

        # Convert class IDs: 1,2,3 -> 0,1,2
        converted_lines = []

        with source_label.open(
            "r",
            encoding="utf-8"
        ) as label_file:

            for line in label_file:

                values = line.strip().split()

                if not values:
                    continue

                old_class = int(values[0])

                if old_class not in CLASS_MAPPING:
                    raise ValueError(
                        f"Unknown class ID: {old_class}"
                    )

                new_class = CLASS_MAPPING[old_class]

                converted_lines.append(
                    " ".join(
                        [str(new_class)] + values[1:]
                    )
                )

        destination_label.write_text(
            "\n".join(converted_lines)
            + ("\n" if converted_lines else ""),
            encoding="utf-8"
        )

        copied_images += 1
        copied_labels += 1

    print("\nDATASET PREPARATION COMPLETED")

    print("Images copied:", copied_images)
    print("Labels copied:", copied_labels)

    for split in SPLIT_MAPPING.values():

        image_count = len(
            list((OUTPUT / "images" / split).glob("*"))
        )

        label_count = len(
            list((OUTPUT / "labels" / split).glob("*.txt"))
        )

        print(
            f"{split}: {image_count} images | "
            f"{label_count} labels"
        )

    print("\nOutput:", OUTPUT)


if __name__ == "__main__":
    prepare_dataset()
