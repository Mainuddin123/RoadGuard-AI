
from pathlib import Path
from datetime import datetime
from collections import Counter
import csv
import re

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "dataset/annotations/clean_manifest.csv"

def extract_timestamp(filename):
    patterns = [
        (
            r"(\d{4})-(\d{2})-(\d{2})-(\d{2})h"
            r"(\d{2})m(\d{2})s(\d+)"
        ),
        r"(\d{8})_(\d{6})"
    ]

    match = re.search(patterns[0], filename)

    if match:
        parts = match.groups()
        year, month, day, hour, minute, second, millis = parts

        return datetime(
            int(year), int(month), int(day),
            int(hour), int(minute), int(second),
            int(millis.ljust(3, "0")[:3]) * 1000
        )

    match = re.search(patterns[1], filename)

    if match:
        return datetime.strptime(
            match.group(1) + match.group(2),
            "%Y%m%d%H%M%S"
        )

    return None


with MANIFEST.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

dated = []
unknown = []

for row in rows:
    timestamp = extract_timestamp(row["image"])

    if timestamp:
        dated.append((timestamp, row["image"]))
    else:
        unknown.append(row["image"])

dated.sort()

print("\n===== RECORDING GROUP ANALYSIS =====")
print("Timestamped images:", len(dated))
print("Unknown filenames:", len(unknown))

print("\nPotential sequence groups using different gap limits:")

for threshold in [10, 30, 60, 120, 300]:
    groups = []

    for item in dated:
        timestamp, filename = item

        if not groups:
            groups.append([item])
            continue

        previous_time = groups[-1][-1][0]
        gap = (timestamp - previous_time).total_seconds()

        if gap <= threshold:
            groups[-1].append(item)
        else:
            groups.append([item])

    sizes = [len(group) for group in groups]

    print(
        f"Gap limit {threshold:>3} seconds: "
        f"{len(groups)} groups | "
        f"{sum(size == 1 for size in sizes)} singletons | "
        f"largest group: {max(sizes)} images"
    )

print("\nFirst 25 unrecognized filenames:")

for filename in unknown[:25]:
    print(filename)

print("\nFilename prefixes:")
prefixes = Counter(
    "vlcsnap" if name.lower().startswith("vlcsnap")
    else "date_timestamp" if re.search(r"\d{8}_\d{6}", name)
    else name.split("-")[0].split("_")[0]
    for name in [row["image"] for row in rows]
)

for prefix, count in prefixes.most_common():
    print(prefix, ":", count)
