
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
LABEL_DIR = PROJECT_DIR / "dataset" / "raw" / "primary" / "data" / "labels-YOLO"

issues = []

for label_file in LABEL_DIR.glob("*.txt"):
    for line_number, line in enumerate(
        label_file.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            continue

        class_id = int(values[0])
        x, y, width, height = map(float, values[1:])

        right = x + width / 2
        left = x - width / 2
        bottom = y + height / 2
        top = y - height / 2

        if left < 0 or right > 1 or top < 0 or bottom > 1:
            issues.append({
                "file": label_file.name,
                "line": line_number,
                "class": class_id,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom
            })

print(f"Total boundary issues: {len(issues)}")
print("\nFirst 10 boundary issues:")

for item in issues[:10]:
    print(item)
