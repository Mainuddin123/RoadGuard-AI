
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
LABEL_DIR = (
    PROJECT_DIR / "dataset" / "raw" / "primary" / "data" / "labels-YOLO"
)

errors = []

if not LABEL_DIR.exists():
    raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

for file in LABEL_DIR.glob("*.txt"):
    for line_no, line in enumerate(
        file.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            continue

        try:
            class_id = int(values[0])
            x, y, w, h = map(float, values[1:])
        except ValueError:
            continue

        left = x - w / 2
        right = x + w / 2
        top = y - h / 2
        bottom = y + h / 2

        overflow = max(
            0, -left, right - 1, -top, bottom - 1
        )

        if overflow > 0:
            errors.append(
                (overflow, file.name, line_no, class_id,
                 left, right, top, bottom)
            )

errors.sort(reverse=True)

print("BOUNDARY ERROR ANALYSIS")
print("=" * 50)

print("Total boundary issues:", len(errors))

if errors:
    print(
        "Maximum boundary overflow:",
        f"{max(e[0] for e in errors):.8f}"
    )
else:
    print("Maximum boundary overflow: 0")
    print("No boundary overflow errors found.")

print("\nOverflow categories:")
print("At most 0.00001:", sum(e[0] <= 0.00001 for e in errors))
print("At most 0.001:", sum(e[0] <= 0.001 for e in errors))
print("At most 0.005:", sum(e[0] <= 0.005 for e in errors))
print("Above 0.005:", sum(e[0] > 0.005 for e in errors))

print("\n10 largest boundary errors:")

if errors:
    for e in errors[:10]:
        print(
            f"{e[1]} | line {e[2]} | class {e[3]} "
            f"| overflow={e[0]:.8f}"
        )
else:
    print("No errors to display.")

print("\nAnalysis completed.")
