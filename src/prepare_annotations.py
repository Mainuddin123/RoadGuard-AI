
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = (
    PROJECT_DIR / "dataset" / "raw" / "primary" / "data" / "labels-YOLO"
)

OUTPUT_DIR = (
    PROJECT_DIR / "dataset" / "annotations" / "primary_yolo_clipped"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total_objects = 0
clipped_objects = 0
discarded_objects = 0

for label_file in SOURCE_DIR.glob("*.txt"):

    corrected_lines = []

    for line_number, line in enumerate(
        label_file.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f"Invalid annotation: {label_file.name}, line {line_number}"
            )

        class_id = int(values[0])
        x, y, width, height = map(float, values[1:])

        if class_id not in (0, 1, 2):
            raise ValueError(
                f"Unknown class ID: {label_file.name}, line {line_number}"
            )

        if not all(0 <= v <= 1 for v in (x, y, width, height)):
            raise ValueError(
                f"Invalid normalized values: {label_file.name}, line {line_number}"
            )

        total_objects += 1

        # Convert YOLO center coordinates to corner coordinates.
        left = x - width / 2
        right = x + width / 2
        top = y - height / 2
        bottom = y + height / 2

        # Clip the box to image boundaries.
        clipped_left = max(0.0, left)
        clipped_right = min(1.0, right)
        clipped_top = max(0.0, top)
        clipped_bottom = min(1.0, bottom)

        # Discard only boxes with no remaining area.
        if (
            clipped_right <= clipped_left
            or clipped_bottom <= clipped_top
        ):
            discarded_objects += 1
            continue

        if (
            clipped_left != left
            or clipped_right != right
            or clipped_top != top
            or clipped_bottom != bottom
        ):
            clipped_objects += 1

        # Convert clipped corners back to YOLO format.
        new_x = (clipped_left + clipped_right) / 2
        new_y = (clipped_top + clipped_bottom) / 2
        new_width = clipped_right - clipped_left
        new_height = clipped_bottom - clipped_top

        corrected_lines.append(
            f"{class_id} {new_x:.10f} {new_y:.10f} "
            f"{new_width:.10f} {new_height:.10f}"
        )

    output_file = OUTPUT_DIR / label_file.name
    output_file.write_text(
        "\n".join(corrected_lines) +
        ("\n" if corrected_lines else "")
    )

print("\n===== ANNOTATION PREPARATION =====")
print(f"Original objects: {total_objects}")
print(f"Clipped objects: {clipped_objects}")
print(f"Discarded objects: {discarded_objects}")
print(f"Final objects: {total_objects - discarded_objects}")
print(f"Output folder: {OUTPUT_DIR}")
print("\nOriginal annotations remain unchanged.")
