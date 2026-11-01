
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "dataset/raw/primary/data/images"

numbered_frames = []
other_files = []

for image in IMAGE_DIR.glob("*.jpg"):
    match = re.fullmatch(r"vlcsnap-(\d+)", image.stem, re.IGNORECASE)

    if match:
        numbered_frames.append((int(match.group(1)), image.name))
    else:
        other_files.append(image.name)

numbered_frames.sort()

numbers = [item[0] for item in numbered_frames]

missing = []

if numbers:
    missing = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))

print("\n===== NUMBERED FRAME INSPECTION =====")
print("Numbered frame images:", len(numbered_frames))
print("Other filename images:", len(other_files))

if numbers:
    print("First frame number:", numbers[0])
    print("Last frame number:", numbers[-1])
    print("Missing frame numbers:", len(missing))
    print("First 20 missing numbers:", missing[:20])

print("\nFirst 15 numbered filenames:")
for number, filename in numbered_frames[:15]:
    print(filename)

print("\nLast 15 numbered filenames:")
for number, filename in numbered_frames[-15:]:
    print(filename)

print("\nFirst 15 other filenames:")
for filename in sorted(other_files)[:15]:
    print(filename)
