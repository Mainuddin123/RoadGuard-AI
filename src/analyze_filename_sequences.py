
from pathlib import Path
from datetime import datetime
import csv
import re

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "dataset/annotations/clean_manifest.csv"
OUTPUT = ROOT / "dataset/annotations/sequence_candidates.csv"

def extract_timestamp(filename):
    # Format: vlcsnap-2025-02-18-17h03m41s574.jpg
    match = re.search(
        r"(\d{4})-(\d{2})-(\d{2})-(\d{2})h(\d{2})m(\d{2})s(\d+)",
        filename
    )

    if match:
        year, month, day, hour, minute, second, millis = match.groups()
        return datetime(
            int(year), int(month), int(day),
            int(hour), int(minute), int(second),
            int(millis.ljust(3, "0")[:3]) * 1000
        )

    # Format: 20250216_164325.jpg
    match = re.search(r"(\d{8})_(\d{6})", filename)

    if match:
        return datetime.strptime(
            match.group(1) + match.group(2),
            "%Y%m%d%H%M%S"
        )

    return None


with MANIFEST.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

dated_images = []
unknown = []

for row in rows:
    timestamp = extract_timestamp(row["image"])

    if timestamp is None:
        unknown.append(row["image"])
    else:
        dated_images.append((timestamp, row["image"]))

dated_images.sort()

candidates = []

for i in range(1, len(dated_images)):
    previous_time, previous_name = dated_images[i - 1]
    current_time, current_name = dated_images[i]

    gap = (current_time - previous_time).total_seconds()

    if 0 <= gap <= 10:
        candidates.append({
            "previous_image": previous_name,
            "current_image": current_name,
            "gap_seconds": gap
        })

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "previous_image",
            "current_image",
            "gap_seconds"
        ]
    )
    writer.writeheader()
    writer.writerows(candidates)

print("\n===== FILENAME SEQUENCE ANALYSIS =====")
print("Images in manifest:", len(rows))
print("Images with readable timestamps:", len(dated_images))
print("Images with unknown timestamps:", len(unknown))
print("Adjacent pairs within 10 seconds:", len(candidates))
print("Candidate pairs saved:", OUTPUT)
print("\nFirst 10 candidate pairs:")

for item in candidates[:10]:
    print(item)
